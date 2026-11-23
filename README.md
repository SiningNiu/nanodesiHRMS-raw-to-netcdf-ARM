# Nano-DESI HRMS: Thermo RAW to NetCDF

A CoreMS-based workflow for processing Nano-DESI Thermo RAW files, running batches from an Excel manifest, and sharing results as readable NetCDF files.

## Workflow

| File | Purpose |
| --- | --- |
| `01_process_single_raw_file.ipynb` | Configure shared processing settings and process one RAW file. |
| `02_batch_process.ipynb` | Load the settings and functions from 01 and process an Excel manifest. |
| `corems_topn_processing.py` | Local processing functions required by 01 and 02. |

Keep 01, 02, and `corems_topn_processing.py` together in the repository root. **You do not need to process a RAW file in 01 before running 02.** Save your settings in 01; 02 reads them directly.

## 1. Set up the environment

The supplied environment snapshot records **64-bit Windows, Python 3.11.13, and CoreMS 3.2.0**.

Install Miniconda, download the repository, and open Anaconda Prompt in the folder containing the environment files. Run:

```bash
conda create -n coreMS --file conda-win-64-explicit.txt
conda activate coreMS
python -m pip install --no-deps -r requirements-pip.txt
python verify_environment.py
```

If `coreMS` already exists, choose a new name in both the create and activate commands. In VS Code, select the new environment as the notebook kernel.

Keep `--no-deps`: this snapshot preserves the recorded package versions, including existing differences from CoreMS's declared dependency ranges. The verifier checks package versions, not scientific results. Installation on a fresh computer has not yet been validated; test a known RAW sample before processing a full batch. This snapshot is not a macOS/Linux installation recipe.

The commands above use `conda-win-64-explicit.txt`; `environment.yml` is an alternative Conda specification and is not needed for this route. Keep `expected_versions.json` beside `verify_environment.py`.

## 2. Prepare local files and paths

Obtain these separately:

- Sample RAW files and, for external background mode, background RAW files.
- The calibration reference mass list.
- An existing, prepared molecular-formula database.
- An Excel manifest for batch processing.

Open the repository folder in VS Code. The publication notebooks default to `Path.cwd()`, meaning the **kernel's current working folder**, not automatic detection of the notebook's location. If this points elsewhere, explicitly set `CODE_DIR` in 01 or `WORK_DIR` in 02 to the repository folder.

### Shared settings: 01, Section 2 — User settings

Both single-file and batch processing use these settings:

| Setting | What to configure |
| --- | --- |
| `REFERENCE_MASS_LIST_PATH` | Actual path to the calibration reference file. The default expects `references/Hawkes_neg 1.ref` under the repository. |
| `FORMULA_DATABASE_URL` | Location of the existing formula database. The default expects `references/corems_mf.db` under the repository. |
| Scientific settings | Review background ratio cutoff, internal-standard formulas, scan selection, calibration, and formula-search settings for your samples. |

Save 01 after editing. There is no need to repeat these settings in 02.

## 3. Run one RAW file

In **01, Section 1 — Imports and paths**, set `RAW_FILE` to your sample RAW. `OUTPUT_ROOT` defaults to the repository's `outputs` folder.

In **Section 2 — User settings**, set:

- `NEW_FILE_NAME`: your output stem, in `prefix.YYYYMMDD.HHMMSS` format, without a level code or `.nc` extension. Replace the illustrative default.
- `BACKGROUND_MODE`: `same_file`, `external_file`, or `none`.
- `BACKGROUND_RAW_PATH`: the separate background RAW for `external_file`; otherwise leave it as `None`.
- `FIELD_METADATA`: any applicable field sample information.

Run the notebook in order. Review diagnostics before applying the settings to a full batch.

## 4. Run a batch

In **02, Section 1 — the first configuration cell**, set:

| Setting | Purpose |
| --- | --- |
| `WORK_DIR` | Repository folder; defaults to the kernel working folder. |
| `JOBS_XLSX` | Your Excel manifest. The default `inputs/batch_manifest.xlsx` is a placeholder, not a bundled workbook. |
| `JOBS_SHEET` | Excel sheet name; default `Sheet1`. |
| `DEFAULT_OUTPUT_ROOT` | Output folder when Excel has no `Output Location` column; default `outputs`. |
| `RUN_MODE` | Use `"process"` to process RAW data. |
| `ONLY_EXCEL_ROWS` | Use `{2}` to test the first data row; use `None` for the full manifest. |

### Excel manifest

Use these processing columns:

| Column | Contents |
| --- | --- |
| `Raw File Name` | Sample RAW filename. |
| `Raw File Location` | Folder containing the sample RAW. |
| `New File Name` | Unique output stem: `prefix.YYYYMMDD.HHMMSS`. |
| `Background Mode` | `same_file`, `external_file`, or `none`. |
| `Background File` | External background filename or full path; leave blank for other modes. Keep the column even when its cells are blank. |
| `Background File Location` | Folder for an external background filename; unnecessary if `Background File` contains a full path. |
| `Output Location` | Optional column overriding `DEFAULT_OUTPUT_ROOT`. If included, fill it for every job. |

Also include field metadata columns below. Their values may be blank when unavailable:

`Location`, `Site`, `System`, `Min. Altitude (m)`, `Max. Altitude (m)`, `Start Day`, `Start Time (UTC)`, `End Day`, `End Time (UTC)`, `Field Note`.

Use dates in `YYYY-MM-DD` format and UTC times in `HH:MM` or `HH:MM:SS` format. Altitudes are numeric meters. Relative paths in Excel are resolved from the **Excel file's folder**. Do not wrap cell paths in quotation marks. Output stems must be unique within an output root.

Save the workbook, run 02 in order, and check the preflight table. Resolve validation errors before starting processing.

### Resume and metadata updates

Successful jobs with unchanged scientific inputs are normally skipped in `process` mode. Preserve the workbook identity, report folder, and resume ledger when continuing an existing batch. Reports and the ledger default to `named_output_reports` under `WORK_DIR`.

`RUN_MODE = "metadata_only"` updates metadata for successful runs recorded in the existing ledger; it does not process RAW data or recompute retained peaks. Changing a filtering cutoff requires scientific reprocessing or an appropriate audit-based rebuild, not metadata-only mode.

Publication copies change default paths and the source fingerprint. Do not substitute them into an existing batch and assume the previous resume state will match.

## Processing and outputs

| Product | Contents |
| --- | --- |
| **a1** | Averaged selected sample profile and picked peaks. |
| **b1** | Resolving-power filtering, mass recalibration, and formula assignment. Includes unambiguous, ambiguous, and unassigned peaks, with a separate formula-candidate table. |
| **c1** | Retained unambiguous peaks after background removal, optional internal-standard removal, oxygen filtering, and DBE−O cleaning. Includes an embedded van Krevelen PNG. |

`same_file` selects sample regions using tracer signals and uses scans outside accepted regions for background. `external_file` uses a separate background RAW and assumes the sample RAW contains sample throughout. `none` skips c1. c1 is also skipped if no eligible background scans remain.

The current publication settings use a sample/background ratio cutoff of **10.0** and enable internal-standard removal. These are configurable choices. Background treatment removes whole peaks. Internal-standard removal matches configured elemental formulas, so review those formulas for each application.

For an input stem `example.20260101.000000`, the public filenames are:

```
example.a1.20260101.000000.nc
example.b1.20260101.000000.nc
example.c1.20260101.000000.nc
```
