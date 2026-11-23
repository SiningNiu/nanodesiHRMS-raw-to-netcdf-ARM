from __future__ import annotations

"""Reusable CoreMS processing for independent Thermo RAW files.

The workflow is intentionally file-based. It does not identify, combine, or merge
technical replicates, and it does not automatically combine original/A2 files.

Main steps
----------
1. Extract a tracer-ion EIC from one RAW file.
2. Select the N scans with the highest tracer-ion intensity.
3. Average those scans and perform CoreMS formula assignment.
4. Optionally process background scans from the same RAW file and remove formulas
   found in the paired background. Retained sample intensities are never subtracted.
5. Optionally normalize retained sample intensities with multiple internal standards.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
import json
import hashlib
import math
import re
import traceback
import warnings

import numpy as np
import pandas as pd


FORMULA_COLUMN_DEFAULT = "Molecular Formula"
INTENSITY_COLUMN_DEFAULT = "Peak Height"


class ProcessingError(RuntimeError):
    """Raised when a RAW file cannot be processed safely."""


@dataclass
class ScanSelectionConfig:
    tracer_mz: float = 215.0234
    top_n_scans: int = 47
    eic_tolerance_ppm: float = 10.0
    ms_filter: str = "MS !d"
    ranking_smooth_window: int = 1
    minimum_tic_quantile: Optional[float] = None


@dataclass
class BackgroundConfig:
    enabled: bool = False
    method: str = "edge"  # edge | explicit_ranges | lowest_tracer_all
    n_scans: int = 10

    # For method="edge". These are 1-based positions in the acquired scan table.
    front_start_position: int = 2
    front_end_position: int = 15
    back_start_from_end: int = 12
    back_end_from_end: int = 2

    # For method="explicit_ranges". Values are inclusive Thermo scan-number ranges.
    explicit_scan_ranges: list[tuple[int, int]] = field(default_factory=list)

    selection_metric: str = "lowest_tracer_intensity"  # lowest_tracer_intensity | lowest_tic
    removal_mode: str = "presence_only"  # presence_only | ratio_guarded_raw_intensity
    sample_to_background_ratio_cutoff: float = 5.0


@dataclass
class InternalStandardConfig:
    enabled: bool = False
    formulas: list[str] = field(
        default_factory=lambda: [
            "C6 H5 O6 N1 S1",
            "C7 H8 O4 S1",
            "C10 H8 O6 S1",
        ]
    )
    mz_targets: dict[str, float] = field(default_factory=dict)
    mz_tolerance_da: float = 0.003
    minimum_detected_standards: int = 2
    remove_internal_standard_features: bool = True
    normalized_intensity_column: str = "Peak Height_IS_normalized"

    # QC thresholds for each standard's ratio to its reference intensity.
    low_ratio_review: float = 0.30
    high_ratio_review: float = 3.00
    low_ratio_exclude: float = 0.10
    high_ratio_exclude: float = 10.00


@dataclass
class CoreMSConfig:
    reference_mass_list_path: str = r"C:\path\to\Hawkes_neg 1.ref"
    formula_database_url: str = "sqlite:///C:/path/to/corems_mf.db"
    ppm_tolerance: float = 5.0
    formula_ppm_min: float = -1.5
    formula_ppm_max: float = 1.5
    first_hit: bool = False

    # Preserved from the uploaded validated production notebooks.
    noise_threshold_method: str = "log"
    noise_threshold_log_nsigma: float = 100.0
    legacy_resolving_power: bool = False
    picking_point_extrapolate: int = 3
    peak_min_prominence_percent: float = 0.001
    noise_min_mz: float = 100.0
    noise_max_mz: float = 850.0
    min_picking_mz: float = 100.0
    max_picking_mz: float = 850.0
    calib_pol_order: int = 2
    calib_sn_threshold: float = 15.0
    calibration_ref_match_method: str = "merged"
    calibration_ref_match_tolerance: float = 0.0015

    rp_filter_enabled: bool = True
    rp_mz_ref: float = 200.0
    rp_fraction: float = 0.5
    rp_mz_window: float = 0.5
    recalibration_enabled: bool = True

    used_atoms: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "C": (1, 90),
            "H": (1, 200),
            "O": (0, 20),
            "N": (0, 2),
            "S": (0, 1),
            "B": (0, 0),
        }
    )
    max_oc_filter: float = 5.0
    max_hc_filter: float = 3.0
    min_dbe: float = 0.0
    max_dbe: float = 30.0
    is_protonated: bool = True
    is_radical: bool = False


@dataclass
class ProcessingConfig:
    scan_selection: ScanSelectionConfig = field(default_factory=ScanSelectionConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    internal_standard: InternalStandardConfig = field(default_factory=InternalStandardConfig)
    corems: CoreMSConfig = field(default_factory=CoreMSConfig)

    formula_column: str = FORMULA_COLUMN_DEFAULT
    intensity_column: str = INTENSITY_COLUMN_DEFAULT
    save_diagnostic_plot: bool = True
    save_averaged_profile: bool = True
    export_all_assignments: bool = True
    export_ambiguous_assignments: bool = False


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def safe_output_key(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_.-]+", "__", str(value)).strip("._")
    return key or "raw_file"


def normalize_formula(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value).strip())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def config_signature(config: ProcessingConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_config(config: ProcessingConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2, default=_json_default), encoding="utf-8")


def _close_reader(reader: Any) -> None:
    if reader is None:
        return
    for method_name in ("close_file", "close"):
        method = getattr(reader, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
            return


def _require_path(path_like: str | Path, label: str) -> Path:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


# -----------------------------------------------------------------------------
# Lazy CoreMS imports
# -----------------------------------------------------------------------------

def _import_corems() -> dict[str, Any]:
    try:
        from corems.encapsulation.factory.parameters import MSParameters
        from corems.mass_spectrum.calc.Calibration import MzDomainCalibration
        from corems.mass_spectra.input.rawFileReader import (
            ImportMassSpectraThermoMSFileReader,
            ThermoBaseClass,
        )
        from corems.molecular_id.search.molecularFormulaSearch import SearchMolecularFormulas
    except Exception as exc:
        raise ImportError(
            "CoreMS/Thermo RAW APIs could not be imported. Run this code in the same "
            "CoreMS conda environment used for the original notebooks."
        ) from exc

    return {
        "MSParameters": MSParameters,
        "MzDomainCalibration": MzDomainCalibration,
        "ImportMassSpectraThermoMSFileReader": ImportMassSpectraThermoMSFileReader,
        "ThermoBaseClass": ThermoBaseClass,
        "SearchMolecularFormulas": SearchMolecularFormulas,
    }


# -----------------------------------------------------------------------------
# Tracer EIC extraction and scan selection
# -----------------------------------------------------------------------------

def _closest_eic(eic_dict: Mapping[Any, Any], requested_mz: float, tolerance_ppm: float) -> Any:
    if requested_mz in eic_dict:
        return eic_dict[requested_mz]
    keys = list(eic_dict.keys())
    if not keys:
        raise KeyError(f"No EIC was returned for m/z {requested_mz:.8f}")
    closest = min(keys, key=lambda x: abs(float(x) - requested_mz))
    ppm_error = abs(float(closest) - requested_mz) / requested_mz * 1e6
    if ppm_error > tolerance_ppm * 2:
        raise KeyError(
            f"No EIC matching m/z {requested_mz:.8f}; nearest key was {closest} "
            f"({ppm_error:.2f} ppm)."
        )
    return eic_dict[closest]


def extract_tracer_scan_table(raw_path: str | Path, config: ScanSelectionConfig) -> pd.DataFrame:
    raw_path = _require_path(raw_path, "RAW file")
    corems = _import_corems()
    reader = None
    try:
        reader = corems["ThermoBaseClass"](str(raw_path))
        reader.chromatogram_settings.scans = (-1, -1)
        reader.chromatogram_settings.eic_tolerance_ppm = float(config.eic_tolerance_ppm)

        tic_data, _ = reader.get_tic(
            ms_type=config.ms_filter,
            peak_detection=False,
            smooth=False,
            plot=False,
            trace_type="TIC",
        )
        if tic_data is None or len(tic_data.scans) == 0:
            raise ProcessingError(f"No scans were returned from {raw_path.name}")

        eic_dict, _ = reader.get_eics(
            target_mzs=[float(config.tracer_mz)],
            tic_data=tic_data,
            ms_type=config.ms_filter,
            peak_detection=False,
            smooth=False,
            plot=False,
        )
        eic = _closest_eic(eic_dict, float(config.tracer_mz), float(config.eic_tolerance_ppm))

        scan_df = pd.DataFrame(
            {
                "scan": np.asarray(tic_data.scans, dtype=int),
                "time_min": np.asarray(tic_data.time, dtype=float),
                "tic": np.asarray(tic_data.tic, dtype=float),
            }
        )
        eic_df = pd.DataFrame(
            {
                "scan": np.asarray(eic.scans, dtype=int),
                "tracer_intensity": np.asarray(eic.eic, dtype=float),
            }
        )
        scan_df = scan_df.merge(eic_df, on="scan", how="left")
        scan_df["tracer_intensity"] = scan_df["tracer_intensity"].fillna(0.0)
        scan_df["raw_file"] = str(raw_path)
        return scan_df.sort_values("scan").reset_index(drop=True)
    finally:
        _close_reader(reader)


def select_top_n_tracer_scans(scan_df: pd.DataFrame, config: ScanSelectionConfig) -> pd.DataFrame:
    required = {"scan", "tracer_intensity", "tic"}
    missing = required - set(scan_df.columns)
    if missing:
        raise KeyError(f"scan_df is missing required columns: {sorted(missing)}")
    if config.top_n_scans <= 0:
        raise ValueError("top_n_scans must be positive")

    candidates = scan_df.copy()
    if config.minimum_tic_quantile is not None:
        q = float(config.minimum_tic_quantile)
        if not 0 <= q < 1:
            raise ValueError("minimum_tic_quantile must be within [0, 1)")
        tic_cutoff = float(candidates["tic"].quantile(q))
        candidates = candidates.loc[candidates["tic"] >= tic_cutoff].copy()

    window = max(int(config.ranking_smooth_window), 1)
    if window > 1:
        candidates["tracer_ranking_intensity"] = (
            candidates["tracer_intensity"].rolling(window, center=True, min_periods=1).median()
        )
    else:
        candidates["tracer_ranking_intensity"] = candidates["tracer_intensity"]

    if len(candidates) < config.top_n_scans:
        raise ProcessingError(
            f"Only {len(candidates)} eligible scans are available; "
            f"{config.top_n_scans} were requested."
        )

    selected = (
        candidates.sort_values(
            ["tracer_ranking_intensity", "tracer_intensity", "scan"],
            ascending=[False, False, True],
        )
        .head(int(config.top_n_scans))
        .copy()
    )
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    selected["selection_type"] = "sample_top_tracer"
    return selected.sort_values("scan").reset_index(drop=True)


def _positions_to_rows(scan_df: pd.DataFrame, start_position: int, end_position: int) -> pd.DataFrame:
    start = max(int(start_position), 1)
    end = min(int(end_position), len(scan_df))
    if end < start:
        return scan_df.iloc[0:0].copy()
    return scan_df.iloc[start - 1 : end].copy()


def select_background_scans(
    scan_df: pd.DataFrame,
    sample_scans: Sequence[int],
    config: BackgroundConfig,
) -> pd.DataFrame:
    if not config.enabled:
        return scan_df.iloc[0:0].copy()
    if config.n_scans <= 0:
        raise ValueError("background n_scans must be positive")

    method = config.method.strip().lower()
    sample_scan_set = {int(x) for x in sample_scans}

    if method == "edge":
        front = _positions_to_rows(
            scan_df,
            config.front_start_position,
            config.front_end_position,
        )
        n = len(scan_df)
        back_start_position = n - int(config.back_start_from_end) + 1
        back_end_position = n - int(config.back_end_from_end) + 1
        back = _positions_to_rows(scan_df, back_start_position, back_end_position)
        candidates = pd.concat([front, back], ignore_index=True).drop_duplicates("scan")
    elif method == "explicit_ranges":
        if not config.explicit_scan_ranges:
            raise ValueError("explicit_scan_ranges must be provided for method='explicit_ranges'")
        mask = np.zeros(len(scan_df), dtype=bool)
        scans = scan_df["scan"].to_numpy(dtype=int)
        for start_scan, end_scan in config.explicit_scan_ranges:
            lo, hi = sorted((int(start_scan), int(end_scan)))
            mask |= (scans >= lo) & (scans <= hi)
        candidates = scan_df.loc[mask].copy()
    elif method == "lowest_tracer_all":
        candidates = scan_df.copy()
    else:
        raise ValueError("background method must be edge, explicit_ranges, or lowest_tracer_all")

    candidates = candidates.loc[~candidates["scan"].astype(int).isin(sample_scan_set)].copy()
    if len(candidates) < config.n_scans:
        raise ProcessingError(
            f"Only {len(candidates)} non-sample background candidates are available; "
            f"{config.n_scans} were requested."
        )

    metric = config.selection_metric.strip().lower()
    if metric == "lowest_tracer_intensity":
        sort_cols = ["tracer_intensity", "tic", "scan"]
    elif metric == "lowest_tic":
        sort_cols = ["tic", "tracer_intensity", "scan"]
    else:
        raise ValueError("background selection_metric must be lowest_tracer_intensity or lowest_tic")

    selected = candidates.sort_values(sort_cols, ascending=True).head(int(config.n_scans)).copy()
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    selected["selection_type"] = "paired_background"
    return selected.sort_values("scan").reset_index(drop=True)


def save_scan_selection_plot(
    scan_df: pd.DataFrame,
    sample_selected: pd.DataFrame,
    background_selected: Optional[pd.DataFrame],
    output_path: str | Path,
    tracer_mz: float,
) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(scan_df["scan"], scan_df["tracer_intensity"], label=f"Tracer m/z {tracer_mz:.4f}")
    ax.scatter(
        sample_selected["scan"],
        sample_selected["tracer_intensity"],
        marker="o",
        label="Selected sample scans",
        zorder=3,
    )
    if background_selected is not None and not background_selected.empty:
        ax.scatter(
            background_selected["scan"],
            background_selected["tracer_intensity"],
            marker="x",
            label="Selected background scans",
            zorder=3,
        )
    ax.set_xlabel("Thermo scan number")
    ax.set_ylabel("Tracer EIC intensity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CoreMS formula assignment
# -----------------------------------------------------------------------------

def _apply_ms_parameters(corems: Mapping[str, Any], cfg: CoreMSConfig) -> None:
    MSParameters = corems["MSParameters"]
    ms = MSParameters.mass_spectrum
    pk = MSParameters.ms_peak

    ms.noise_threshold_method = cfg.noise_threshold_method
    ms.noise_threshold_log_nsigma = cfg.noise_threshold_log_nsigma
    pk.legacy_resolving_power = cfg.legacy_resolving_power
    ms.picking_point_extrapolate = cfg.picking_point_extrapolate
    pk.peak_min_prominence_percent = cfg.peak_min_prominence_percent
    ms.noise_min_mz = cfg.noise_min_mz
    ms.noise_max_mz = cfg.noise_max_mz
    ms.min_picking_mz = cfg.min_picking_mz
    ms.max_picking_mz = cfg.max_picking_mz
    ms.calib_pol_order = cfg.calib_pol_order
    ms.calib_sn_threshold = cfg.calib_sn_threshold
    ms.calibration_ref_match_method = cfg.calibration_ref_match_method
    ms.calibration_ref_match_tolerance = cfg.calibration_ref_match_tolerance


def _orbitrap_rp_filter(
    mass_spectrum: Any,
    mz_ref: float,
    fraction: float,
    mz_window: float,
) -> dict[str, Any]:
    peaks = list(mass_spectrum.mspeaks)
    if not peaks:
        return {"kept": 0, "removed": 0, "rp_ref": None}

    local_values = [
        peak.resolving_power
        for peak in peaks
        if (mz_ref - mz_window) < float(peak.mz_exp) < (mz_ref + mz_window)
    ]
    all_values = [peak.resolving_power for peak in peaks]
    rp_ref = float(np.median(local_values if local_values else all_values))

    def minimum_rp(mz: float) -> float:
        return fraction * rp_ref * math.sqrt(mz_ref / mz)

    remove_indices = [
        i for i, peak in enumerate(peaks)
        if float(peak.resolving_power) < minimum_rp(float(peak.mz_exp))
    ]
    mass_spectrum.filter_by_index(remove_indices)
    return {
        "kept": int(len(mass_spectrum.mspeaks)),
        "removed": int(len(remove_indices)),
        "rp_ref": rp_ref,
    }


def _calibration_metrics(ms_obj: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "n_ref_matched": np.nan,
        "residual_median_abs_ppm": np.nan,
        "residual_rmse_ppm": np.nan,
        "residual_mean_ppm": np.nan,
        "residual_std_ppm": np.nan,
        "is_calibrated": np.nan,
    }
    attribute_map = {
        "n_ref_matched": "calibration_points",
        "residual_rmse_ppm": "calibration_RMS",
        "residual_mean_ppm": "calibration_raw_error_median",
        "residual_std_ppm": "calibration_raw_error_stdev",
        "is_calibrated": "is_calibrated",
    }
    for output_name, attribute_name in attribute_map.items():
        try:
            value = getattr(ms_obj, attribute_name)
            if output_name == "n_ref_matched":
                value = int(value)
            elif output_name == "is_calibrated":
                value = bool(value)
            else:
                value = float(value)
            values[output_name] = value
        except Exception:
            pass
    if np.isfinite(values.get("residual_mean_ppm", np.nan)):
        values["residual_median_abs_ppm"] = abs(float(values["residual_mean_ppm"]))
    return values


def _configure_molecular_search(ms_obj: Any, cfg: CoreMSConfig) -> None:
    settings = ms_obj.molecular_search_settings
    settings.url_database = cfg.formula_database_url
    settings.db_jobs = 1
    settings.min_ppm_error = cfg.formula_ppm_min
    settings.max_ppm_error = cfg.formula_ppm_max
    settings.max_oc_filter = cfg.max_oc_filter
    settings.max_hc_filter = cfg.max_hc_filter
    settings.min_dbe = cfg.min_dbe
    settings.max_dbe = cfg.max_dbe
    for element, bounds in cfg.used_atoms.items():
        settings.usedAtoms[element] = tuple(bounds)
    settings.isProtonated = bool(cfg.is_protonated)
    settings.isRadical = bool(cfg.is_radical)


def average_profile_spectrum(
    raw_path: str | Path,
    scan_numbers: Sequence[int],
    cfg: CoreMSConfig,
) -> Any:
    raw_path = _require_path(raw_path, "RAW file")
    if not scan_numbers:
        raise ValueError("scan_numbers is empty")

    corems = _import_corems()
    _apply_ms_parameters(corems, cfg)
    reader = None
    try:
        reader = corems["ImportMassSpectraThermoMSFileReader"](str(raw_path))
        scans = [int(scan) for scan in scan_numbers]
        return reader.get_average_mass_spectrum_by_scanlist(scans, True, float(cfg.ppm_tolerance))
    finally:
        _close_reader(reader)


def run_formula_assignment(
    raw_path: str | Path,
    scan_numbers: Sequence[int],
    cfg: CoreMSConfig,
) -> tuple[Any, dict[str, pd.DataFrame], dict[str, Any]]:
    ref_path = Path(cfg.reference_mass_list_path)
    if cfg.recalibration_enabled:
        _require_path(ref_path, "CoreMS calibration reference mass list")

    corems = _import_corems()
    spectrum = average_profile_spectrum(raw_path, scan_numbers, cfg)
    meta: dict[str, Any] = {
        "n_scans_averaged": int(len(scan_numbers)),
        "selected_scans": [int(x) for x in scan_numbers],
        "n_peaks_initial": int(len(spectrum)),
    }

    if cfg.rp_filter_enabled:
        meta["rp_filter"] = _orbitrap_rp_filter(
            spectrum,
            mz_ref=float(cfg.rp_mz_ref),
            fraction=float(cfg.rp_fraction),
            mz_window=float(cfg.rp_mz_window),
        )
    meta["n_peaks_after_rp"] = int(len(spectrum))

    try:
        spectrum.run_noise_threshold_calc()
        meta["noise_threshold"] = float(spectrum.get_noise_threshold()[1][0])
    except Exception as exc:
        meta["noise_threshold_error"] = repr(exc)

    if cfg.recalibration_enabled:
        calibration = corems["MzDomainCalibration"](spectrum, str(ref_path))
        calibration.run()
        meta["recalibration"] = {
            "reference_mass_list": str(ref_path),
            **_calibration_metrics(spectrum),
        }

    _configure_molecular_search(spectrum, cfg)
    corems["SearchMolecularFormulas"](
        spectrum,
        first_hit=bool(cfg.first_hit),
    ).run_worker_mass_spectrum()
    try:
        spectrum.percentile_assigned()
    except Exception:
        pass

    classified = classify_assignments(spectrum.to_dataframe())
    return spectrum, classified, meta


def classify_assignments(
    ms_df: pd.DataFrame,
    formula_column: str = FORMULA_COLUMN_DEFAULT,
) -> dict[str, pd.DataFrame]:
    df = ms_df.copy()
    if formula_column not in df.columns:
        raise ValueError(
            f"CoreMS output does not contain {formula_column!r}. "
            f"Available columns: {df.columns.tolist()}"
        )
    if "m/z" not in df.columns:
        raise ValueError("CoreMS output does not contain 'm/z'")

    df["assigned"] = df[formula_column].notna()
    formula_counts = (
        df.loc[df["assigned"]]
        .groupby("m/z")[formula_column]
        .nunique()
        .rename("n_formulas")
    )
    if "n_formulas" in df.columns:
        df = df.drop(columns="n_formulas")
    df = df.merge(formula_counts, on="m/z", how="left")
    df["n_formulas"] = df["n_formulas"].fillna(0).astype(int)
    df["assign_type"] = np.select(
        [
            (~df["assigned"]) | (df["n_formulas"] == 0),
            df["assigned"] & (df["n_formulas"] == 1),
            df["assigned"] & (df["n_formulas"] > 1),
        ],
        ["unassigned", "unambiguous", "ambiguous"],
        default="unknown",
    )
    return {
        "all": df,
        "unambiguous": df.loc[df["assign_type"] == "unambiguous"].copy(),
        "ambiguous": df.loc[df["assign_type"] == "ambiguous"].copy(),
        "unassigned": df.loc[df["assign_type"] == "unassigned"].copy(),
    }


def save_averaged_profile(spectrum: Any, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "mz": np.asarray(spectrum.mz_exp_profile, dtype=float),
            "abundance": np.asarray(spectrum.abundance_profile, dtype=float),
        }
    ).to_csv(output_path, index=False)


# -----------------------------------------------------------------------------
# Paired background formula removal
# -----------------------------------------------------------------------------

def remove_paired_background_formulas(
    sample_df: pd.DataFrame,
    background_df: pd.DataFrame,
    config: BackgroundConfig,
    formula_column: str = FORMULA_COLUMN_DEFAULT,
    intensity_column: str = INTENSITY_COLUMN_DEFAULT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for table_name, table in (("sample", sample_df), ("background", background_df)):
        for column in (formula_column, intensity_column):
            if column not in table.columns:
                raise KeyError(f"{table_name} table is missing {column!r}")

    sample = sample_df.copy()
    background = background_df.copy()
    sample["_formula_key"] = sample[formula_column].map(normalize_formula)
    background["_formula_key"] = background[formula_column].map(normalize_formula)
    sample[intensity_column] = pd.to_numeric(sample[intensity_column], errors="coerce").fillna(0.0)
    background[intensity_column] = pd.to_numeric(background[intensity_column], errors="coerce").fillna(0.0)

    background_by_formula = (
        background.loc[background["_formula_key"].ne("")]
        .groupby("_formula_key", as_index=False)[intensity_column]
        .max()
        .rename(columns={intensity_column: "paired_background_peak_height"})
    )
    audit = sample.merge(background_by_formula, on="_formula_key", how="left")
    audit["matched_in_paired_background"] = audit["paired_background_peak_height"].notna()
    audit["paired_background_peak_height"] = audit["paired_background_peak_height"].fillna(0.0)
    audit["sample_to_background_peak_height_ratio"] = np.where(
        audit["paired_background_peak_height"] > 0,
        audit[intensity_column] / audit["paired_background_peak_height"],
        np.inf,
    )
    audit["background_to_sample_peak_height_fraction"] = np.where(
        audit[intensity_column] > 0,
        audit["paired_background_peak_height"] / audit[intensity_column],
        np.nan,
    )

    mode = config.removal_mode.strip().lower()
    if mode == "presence_only":
        remove_mask = audit["matched_in_paired_background"]
    elif mode == "ratio_guarded_raw_intensity":
        cutoff = float(config.sample_to_background_ratio_cutoff)
        if cutoff <= 0:
            raise ValueError("sample_to_background_ratio_cutoff must be positive")
        remove_mask = (
            audit["matched_in_paired_background"]
            & (audit["sample_to_background_peak_height_ratio"] <= cutoff)
        )
    else:
        raise ValueError("removal_mode must be presence_only or ratio_guarded_raw_intensity")

    audit["remove_as_paired_background"] = remove_mask.astype(bool)
    audit["background_removal_rule"] = mode

    original_columns = list(sample_df.columns)
    clean = audit.loc[~audit["remove_as_paired_background"], original_columns].copy()

    sample_total = float(sample[intensity_column].sum())
    removed_total = float(audit.loc[remove_mask, intensity_column].sum())
    summary = {
        "background_removal_mode": mode,
        "n_sample_formulas_before_removal": int(len(sample)),
        "n_background_formulas": int(background["_formula_key"].nunique()),
        "n_sample_formulas_matched_in_background": int(audit["matched_in_paired_background"].sum()),
        "n_sample_formulas_removed": int(remove_mask.sum()),
        "n_clean_formulas_retained": int(len(clean)),
        "fraction_sample_formulas_removed": float(remove_mask.mean()) if len(audit) else np.nan,
        "sample_total_peak_height": sample_total,
        "removed_sample_peak_height": removed_total,
        "fraction_sample_peak_height_removed": removed_total / sample_total if sample_total > 0 else np.nan,
        "intensity_subtraction_performed": False,
    }
    return clean, audit.drop(columns=["_formula_key"], errors="ignore"), summary


# -----------------------------------------------------------------------------
# Internal-standard extraction and normalization
# -----------------------------------------------------------------------------

def extract_internal_standard_intensities(
    formula_df: pd.DataFrame,
    config: InternalStandardConfig,
    formula_column: str = FORMULA_COLUMN_DEFAULT,
    intensity_column: str = INTENSITY_COLUMN_DEFAULT,
) -> pd.DataFrame:
    for column in (formula_column, intensity_column):
        if column not in formula_df.columns:
            raise KeyError(f"formula table is missing {column!r}")

    df = formula_df.copy()
    df["_formula_key"] = df[formula_column].map(normalize_formula)
    df["_intensity"] = pd.to_numeric(df[intensity_column], errors="coerce")
    if "m/z" in df.columns:
        df["_mz"] = pd.to_numeric(df["m/z"], errors="coerce")
    else:
        df["_mz"] = np.nan

    rows: list[dict[str, Any]] = []
    for index, target_formula in enumerate(config.formulas, start=1):
        is_id = f"IS{index}"
        key = normalize_formula(target_formula)
        hits = df.loc[df["_formula_key"].eq(key)].sort_values("_intensity", ascending=False)
        hit = hits.iloc[0] if len(hits) else None
        rows.append(
            {
                "internal_standard_id": is_id,
                "internal_standard_target_formula": target_formula,
                "internal_standard_target_mz": np.nan,
                "internal_standard_formula": hit[formula_column] if hit is not None else np.nan,
                "internal_standard_mz": float(hit["_mz"]) if hit is not None and pd.notna(hit["_mz"]) else np.nan,
                "internal_standard_intensity": float(hit["_intensity"]) if hit is not None and pd.notna(hit["_intensity"]) else np.nan,
            }
        )

    for is_id, target_mz in config.mz_targets.items():
        hits = df.loc[(df["_mz"] - float(target_mz)).abs() <= float(config.mz_tolerance_da)].copy()
        hits = hits.sort_values(["_intensity", "_mz"], ascending=[False, True])
        hit = hits.iloc[0] if len(hits) else None
        rows.append(
            {
                "internal_standard_id": str(is_id),
                "internal_standard_target_formula": np.nan,
                "internal_standard_target_mz": float(target_mz),
                "internal_standard_formula": hit[formula_column] if hit is not None else np.nan,
                "internal_standard_mz": float(hit["_mz"]) if hit is not None and pd.notna(hit["_mz"]) else np.nan,
                "internal_standard_intensity": float(hit["_intensity"]) if hit is not None and pd.notna(hit["_intensity"]) else np.nan,
            }
        )

    if not rows:
        raise ValueError("No internal-standard formulas or m/z targets were configured")
    return pd.DataFrame(rows).drop_duplicates("internal_standard_id", keep="first")


def compute_batch_internal_standard_references(
    internal_standard_tables: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    frames = [table.copy() for table in internal_standard_tables if table is not None and not table.empty]
    if not frames:
        raise ProcessingError("No internal-standard intensity tables were available")
    long_df = pd.concat(frames, ignore_index=True)
    long_df["internal_standard_intensity"] = pd.to_numeric(
        long_df["internal_standard_intensity"], errors="coerce"
    )
    reference = (
        long_df.groupby("internal_standard_id", as_index=False)
        .agg(
            internal_standard_reference_intensity=("internal_standard_intensity", "median"),
            n_files_detected=("internal_standard_intensity", lambda x: int(x.notna().sum())),
        )
    )
    return reference


def _reference_mapping(reference_intensities: Mapping[str, float] | pd.DataFrame) -> dict[str, float]:
    if isinstance(reference_intensities, pd.DataFrame):
        required = {"internal_standard_id", "internal_standard_reference_intensity"}
        missing = required - set(reference_intensities.columns)
        if missing:
            raise KeyError(f"reference table is missing columns: {sorted(missing)}")
        mapping = dict(
            zip(
                reference_intensities["internal_standard_id"].astype(str),
                pd.to_numeric(
                    reference_intensities["internal_standard_reference_intensity"],
                    errors="coerce",
                ),
            )
        )
    else:
        mapping = {str(k): float(v) for k, v in reference_intensities.items()}
    return {k: float(v) for k, v in mapping.items() if pd.notna(v) and float(v) > 0}


def apply_internal_standard_normalization(
    clean_formula_df: pd.DataFrame,
    internal_standard_table: pd.DataFrame,
    reference_intensities: Mapping[str, float] | pd.DataFrame,
    config: InternalStandardConfig,
    formula_column: str = FORMULA_COLUMN_DEFAULT,
    intensity_column: str = INTENSITY_COLUMN_DEFAULT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if intensity_column not in clean_formula_df.columns:
        raise KeyError(f"clean formula table is missing {intensity_column!r}")

    refs = _reference_mapping(reference_intensities)
    is_qc = internal_standard_table.copy()
    is_qc["internal_standard_reference_intensity"] = is_qc["internal_standard_id"].map(refs)
    is_qc["internal_standard_ratio_to_reference"] = (
        pd.to_numeric(is_qc["internal_standard_intensity"], errors="coerce")
        / pd.to_numeric(is_qc["internal_standard_reference_intensity"], errors="coerce")
    )

    ratio = is_qc["internal_standard_ratio_to_reference"]
    is_qc["internal_standard_qc_status"] = np.select(
        [
            is_qc["internal_standard_intensity"].isna(),
            pd.to_numeric(is_qc["internal_standard_intensity"], errors="coerce").le(0),
            ratio.lt(config.low_ratio_exclude),
            ratio.gt(config.high_ratio_exclude),
            ratio.lt(config.low_ratio_review),
            ratio.gt(config.high_ratio_review),
        ],
        [
            "exclude_missing_IS",
            "exclude_zero_IS",
            "exclude_extremely_low_IS",
            "exclude_extremely_high_IS",
            "review_low_IS",
            "review_high_IS",
        ],
        default="pass",
    )

    valid_ratios = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    valid_ratios = valid_ratios.loc[valid_ratios > 0]
    n_detected = int(valid_ratios.size)
    if n_detected < int(config.minimum_detected_standards):
        composite_ratio = np.nan
        overall_status = "review_insufficient_IS"
    else:
        composite_ratio = float(valid_ratios.median())
        overall_status = "pass" if (is_qc["internal_standard_qc_status"] == "pass").all() else "review_some_IS_nonpass"

    output = clean_formula_df.copy()
    raw_intensity = pd.to_numeric(output[intensity_column], errors="coerce")
    output["composite_IS_ratio"] = composite_ratio
    output["n_internal_standards_detected"] = n_detected
    output["internal_standard_qc_status"] = overall_status
    output[config.normalized_intensity_column] = (
        raw_intensity / composite_ratio if np.isfinite(composite_ratio) and composite_ratio > 0 else np.nan
    )

    if config.remove_internal_standard_features and formula_column in output.columns:
        target_formula_keys = {normalize_formula(x) for x in config.formulas}
        output = output.loc[
            ~output[formula_column].map(normalize_formula).isin(target_formula_keys)
        ].copy()

    ratio_spread = (
        float(valid_ratios.max() / valid_ratios.min())
        if len(valid_ratios) and float(valid_ratios.min()) > 0
        else np.nan
    )
    summary = {
        "internal_standard_normalization_enabled": True,
        "internal_standard_normalization_method": "median_ratio_composite",
        "n_internal_standards_expected": int(len(is_qc)),
        "n_internal_standards_detected": n_detected,
        "composite_IS_ratio": composite_ratio,
        "IS_ratio_spread": ratio_spread,
        "internal_standard_qc_status": overall_status,
    }
    return output, is_qc, summary


# -----------------------------------------------------------------------------
# Complete single-file workflow
# -----------------------------------------------------------------------------

def process_raw_file(
    raw_path: str | Path,
    output_root: str | Path,
    config: ProcessingConfig,
    *,
    sample_key: Optional[str] = None,
    internal_standard_reference_intensities: Optional[Mapping[str, float] | pd.DataFrame] = None,
    defer_internal_standard_normalization: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    raw_path = _require_path(raw_path, "RAW file")
    output_root = Path(output_root)
    sample_key = sample_key or raw_path.stem
    file_output_dir = output_root / safe_output_key(sample_key)
    file_output_dir.mkdir(parents=True, exist_ok=True)

    status_path = file_output_dir / "processing_status.json"
    current_config_signature = config_signature(config)
    if status_path.exists() and not overwrite:
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "success"
                and existing.get("config_signature") == current_config_signature
            ):
                existing["loaded_from_existing_output"] = True
                return existing
        except Exception:
            pass

    status: dict[str, Any] = {
        "sample_key": sample_key,
        "raw_file": str(raw_path),
        "output_directory": str(file_output_dir),
        "status": "running",
        "loaded_from_existing_output": False,
        "background_enabled": bool(config.background.enabled),
        "internal_standard_enabled": bool(config.internal_standard.enabled),
        "config_signature": current_config_signature,
    }
    save_config(config, file_output_dir / "processing_config.json")

    try:
        scan_df = extract_tracer_scan_table(raw_path, config.scan_selection)
        sample_selected = select_top_n_tracer_scans(scan_df, config.scan_selection)
        sample_scans = sample_selected["scan"].astype(int).tolist()

        scan_table_path = file_output_dir / "scan_table_tracer_eic.csv"
        sample_scan_path = file_output_dir / "selected_sample_scans.csv"
        scan_df.to_csv(scan_table_path, index=False)
        sample_selected.to_csv(sample_scan_path, index=False)

        background_selected = pd.DataFrame()
        if config.background.enabled:
            background_selected = select_background_scans(
                scan_df,
                sample_scans,
                config.background,
            )
            background_selected.to_csv(file_output_dir / "selected_background_scans.csv", index=False)

        if config.save_diagnostic_plot:
            save_scan_selection_plot(
                scan_df,
                sample_selected,
                background_selected,
                file_output_dir / "scan_selection_diagnostic.png",
                config.scan_selection.tracer_mz,
            )

        sample_spectrum, sample_classified, sample_meta = run_formula_assignment(
            raw_path,
            sample_scans,
            config.corems,
        )
        if config.save_averaged_profile:
            save_averaged_profile(sample_spectrum, file_output_dir / "sample_averaged_profile.csv")

        sample_all_path = file_output_dir / "sample_formula_assignments_all.csv"
        sample_unambiguous_path = file_output_dir / "sample_unambiguous_pre_background.csv"
        if config.export_all_assignments:
            sample_classified["all"].to_csv(sample_all_path, index=False)
        sample_classified["unambiguous"].to_csv(sample_unambiguous_path, index=False)
        if config.export_ambiguous_assignments:
            sample_classified["ambiguous"].to_csv(
                file_output_dir / "sample_ambiguous.csv", index=False
            )

        # Internal standards are always extracted from the pre-background sample output.
        is_table = pd.DataFrame()
        if config.internal_standard.enabled:
            is_table = extract_internal_standard_intensities(
                sample_classified["unambiguous"],
                config.internal_standard,
                config.formula_column,
                config.intensity_column,
            )
            is_table["sample_key"] = sample_key
            is_table["raw_file"] = str(raw_path)
            is_table.to_csv(file_output_dir / "internal_standard_intensities.csv", index=False)

        clean_raw = sample_classified["unambiguous"].copy()
        background_meta: dict[str, Any] = {}
        removal_summary: dict[str, Any] = {
            "background_removal_enabled": False,
            "intensity_subtraction_performed": False,
        }

        if config.background.enabled:
            background_scans = background_selected["scan"].astype(int).tolist()
            background_spectrum, background_classified, background_meta = run_formula_assignment(
                raw_path,
                background_scans,
                config.corems,
            )
            if config.save_averaged_profile:
                save_averaged_profile(
                    background_spectrum,
                    file_output_dir / "background_averaged_profile.csv",
                )
            background_classified["unambiguous"].to_csv(
                file_output_dir / "background_unambiguous.csv", index=False
            )
            if config.export_all_assignments:
                background_classified["all"].to_csv(
                    file_output_dir / "background_formula_assignments_all.csv", index=False
                )

            clean_raw, removal_audit, removal_summary = remove_paired_background_formulas(
                sample_classified["unambiguous"],
                background_classified["unambiguous"],
                config.background,
                config.formula_column,
                config.intensity_column,
            )
            removal_summary["background_removal_enabled"] = True
            removal_audit.to_csv(file_output_dir / "background_removal_audit.csv", index=False)

        clean_raw_path = file_output_dir / "clean_unambiguous_raw_intensity.csv"
        clean_raw.to_csv(clean_raw_path, index=False)

        final_df = clean_raw.copy()
        is_summary: dict[str, Any] = {
            "internal_standard_normalization_enabled": False,
            "internal_standard_normalization_status": "disabled",
        }
        final_path = file_output_dir / "clean_unambiguous_final.csv"

        if config.internal_standard.enabled:
            if internal_standard_reference_intensities is not None:
                final_df, is_qc, is_summary = apply_internal_standard_normalization(
                    clean_raw,
                    is_table,
                    internal_standard_reference_intensities,
                    config.internal_standard,
                    config.formula_column,
                    config.intensity_column,
                )
                is_qc.to_csv(file_output_dir / "internal_standard_qc.csv", index=False)
                is_summary["internal_standard_normalization_status"] = "completed"
            elif defer_internal_standard_normalization:
                is_summary = {
                    "internal_standard_normalization_enabled": True,
                    "internal_standard_normalization_status": "pending_batch_reference",
                }
            else:
                is_summary = {
                    "internal_standard_normalization_enabled": True,
                    "internal_standard_normalization_status": "pending_reference_intensities",
                    "message": (
                        "Supply reference intensities for a standalone file, or run the batch "
                        "workflow so campaign medians can be calculated automatically."
                    ),
                }
        final_df.to_csv(final_path, index=False)

        status.update(
            {
                "status": "success",
                "n_total_scans": int(len(scan_df)),
                "n_selected_sample_scans": int(len(sample_selected)),
                "selected_sample_scans": sample_scans,
                "n_selected_background_scans": int(len(background_selected)),
                "n_sample_unambiguous_pre_background": int(len(sample_classified["unambiguous"])),
                "n_final_clean_formulas": int(len(final_df)),
                "sample_corems_meta": sample_meta,
                "background_corems_meta": background_meta,
                "background_removal_summary": removal_summary,
                "internal_standard_summary": is_summary,
                "scan_table_csv": str(scan_table_path),
                "selected_sample_scans_csv": str(sample_scan_path),
                "sample_unambiguous_pre_background_csv": str(sample_unambiguous_path),
                "clean_raw_intensity_csv": str(clean_raw_path),
                "final_clean_csv": str(final_path),
            }
        )
        status_path.write_text(
            json.dumps(status, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return status

    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        status_path.write_text(
            json.dumps(status, indent=2, default=_json_default),
            encoding="utf-8",
        )
        raise


def finalize_file_internal_standard_normalization(
    file_output_dir: str | Path,
    config: ProcessingConfig,
    reference_intensities: Mapping[str, float] | pd.DataFrame,
) -> dict[str, Any]:
    file_output_dir = Path(file_output_dir)
    clean_path = file_output_dir / "clean_unambiguous_raw_intensity.csv"
    is_path = file_output_dir / "internal_standard_intensities.csv"
    status_path = file_output_dir / "processing_status.json"
    for path, label in (
        (clean_path, "clean raw-intensity table"),
        (is_path, "internal-standard table"),
        (status_path, "processing status"),
    ):
        _require_path(path, label)

    clean_df = pd.read_csv(clean_path)
    is_table = pd.read_csv(is_path)
    final_df, is_qc, is_summary = apply_internal_standard_normalization(
        clean_df,
        is_table,
        reference_intensities,
        config.internal_standard,
        config.formula_column,
        config.intensity_column,
    )
    final_path = file_output_dir / "clean_unambiguous_final.csv"
    final_df.to_csv(final_path, index=False)
    is_qc.to_csv(file_output_dir / "internal_standard_qc.csv", index=False)

    status = json.loads(status_path.read_text(encoding="utf-8"))
    is_summary["internal_standard_normalization_status"] = "completed"
    status["internal_standard_summary"] = is_summary
    status["n_final_clean_formulas"] = int(len(final_df))
    status["final_clean_csv"] = str(final_path)
    status_path.write_text(json.dumps(status, indent=2, default=_json_default), encoding="utf-8")
    return status


def discover_raw_files(raw_root: str | Path, recursive: bool = True) -> list[Path]:
    raw_root = _require_path(raw_root, "RAW folder")
    iterator = raw_root.rglob("*") if recursive else raw_root.glob("*")
    paths = [path for path in iterator if path.is_file() and path.suffix.lower() == ".raw"]
    return sorted(paths, key=lambda p: p.relative_to(raw_root).as_posix().lower())
