"""Repository-relative data paths. Raw VitalDB data are downloaded at run time and never committed.

Cohort keys: `sdev` (S-dev) and `sob` (S-ob, SPEC §2.2). Raw S-ob data exist only after the
frozen download at the frozen scoring (SPEC §9 step 7); `btc.io` refuses to read them otherwise.
The functions below read the module constants at call time, so tests can point them elsewhere.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_SDEV = RAW / "sdev"                    # one .npz per S-dev case (scripts/download.py --cohort sdev)
RAW_SOB = RAW / "sob"                      # one .npz per S-ob case (frozen scoring only)
MANIFEST_SDEV = RAW / "sdev_manifest.csv"  # per-case download and signal-integrity checks (SPEC §2.3)
MANIFEST_SOB = RAW / "sob_manifest.csv"
META = RAW / "meta"                        # cached VitalDB case table and track list
CASE_TABLE = META / "cases.csv"            # VitalDB case table (sex, age)
COHORTS = DATA / "cohorts.csv"             # case-table fields for all spinal cases (S-dev and S-ob)
DERIVED = DATA / "derived"                 # all derived tables (gitignored)

COHORTS_KEYS = ("sdev", "sob")
COHORT_LABEL = {"sdev": "S-dev", "sob": "S-ob"}   # labels in `data/cohorts.csv`


def _key(cohort: str) -> str:
    if cohort not in COHORTS_KEYS:
        raise ValueError(f"unknown cohort {cohort!r}; expected one of {COHORTS_KEYS}")
    return cohort


def raw_dir(cohort: str) -> Path:
    return {"sdev": RAW_SDEV, "sob": RAW_SOB}[_key(cohort)]


def manifest(cohort: str) -> Path:
    return {"sdev": MANIFEST_SDEV, "sob": MANIFEST_SOB}[_key(cohort)]


def derived(cohort: str, name: str) -> Path:
    """`data/derived/<cohort>_<name>` (btc/SCHEMA.md)."""
    return DERIVED / f"{_key(cohort)}_{name}"
