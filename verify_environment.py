"""Verify snapshot versions; does not install packages or process RAW data."""
from pathlib import Path
import importlib.metadata as metadata
import json
import platform
import sys

root = Path(__file__).resolve().parent
expected = json.loads((root / "expected_versions.json").read_text(encoding="utf-8"))
issues = []
if sys.platform != "win32" or platform.architecture()[0] != "64bit":
    issues.append("This snapshot targets 64-bit Windows")
if platform.python_version() != expected["python"]:
    issues.append(f"Python: expected {expected['python']}, found {platform.python_version()}")
for package in expected["conda"]:
    name, version, build = package["name"], package["version"], package["build"]
    record = Path(sys.prefix) / "conda-meta" / f"{name}-{version}-{build}.json"
    if not record.is_file():
        issues.append(f"Missing exact Conda record: {name}={version}={build}")
for name, version in expected["python_distributions"].items():
    try:
        found = metadata.version(name)
    except metadata.PackageNotFoundError:
        found = "not installed"
    if found != version:
        issues.append(f"{name}: expected {version}, found {found}")
if issues:
    print("Snapshot differences:")
    print("\n".join("- " + issue for issue in issues))
    raise SystemExit(1)
print("PASS: recorded Conda builds and Python distribution versions match this snapshot.")
print("This is not a RAW-processing test. Validate a known sample on the new computer.")
