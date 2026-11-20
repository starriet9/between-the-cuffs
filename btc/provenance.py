"""Provenance of derived files (SPEC §10) and the data credit (SPEC §2.1).

Every builder records the git state of the code that wrote its outputs, in the parquet
metadata (key `btc`) and in its json log. A builder refuses to run from a dirty or unknown
git state unless `--allow-dirty` is given; the flag is recorded too.

Dirty = any modified, staged or untracked file that git does not ignore. Untracked files
count, because code that is not committed is not identified by the commit hash.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import paths

SPEC_VERSION = "SPEC.md v1.5 / PREREG.md v1.5"

# SPEC §2.1: every public output carries the credit, the licence link and this statement of changes.
CREDIT = ("Derived from VitalDB (CC BY 4.0). Changes: cuff readings extracted by change detection; "
          "signals resampled, filtered and segmented; features computed; plots and summary statistics "
          "derived. No attempt was made to identify any person.")
SOURCE = ("Source: VitalDB (Lee HC, et al. Sci Data 2022;9:279, doi:10.1038/s41597-022-01411-5). "
          "Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.")

_MAX_STATUS_LINES = 50


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def git_state() -> dict:
    """{"commit", "describe", "state", "dirty", "status"}; `state` is clean, dirty or unknown."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=normal")
    if not commit or status is None:
        return {"commit": commit or None, "describe": None, "state": "unknown", "dirty": None, "status": []}
    lines = status.splitlines()
    return {"commit": commit, "describe": _git("describe", "--tags", "--always", "--dirty"),
            "state": "dirty" if lines else "clean", "dirty": bool(lines),
            "status": lines[:_MAX_STATUS_LINES] + (["..."] if len(lines) > _MAX_STATUS_LINES else [])}


class DirtyStateError(RuntimeError):
    """The git state is dirty or unknown and `--allow-dirty` was not given."""


def require_clean(allow_dirty: bool, state: dict | None = None) -> dict:
    """The git state to record; raises DirtyStateError if it is not clean and not allowed."""
    st = dict(git_state() if state is None else state)
    st["allow_dirty"] = bool(allow_dirty)
    if st["state"] != "clean" and not allow_dirty:
        raise DirtyStateError(f"git state is {st['state']} (commit {st['commit']}); commit the code or "
                              "pass --allow-dirty (recorded in the outputs)")
    return st


def add_allow_dirty(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--allow-dirty", action="store_true",
                        help="run from a dirty or unknown git state (recorded in every output)")


def write_parquet(df: pd.DataFrame, path: Path, meta: dict) -> Path:
    """Write `df` with `meta` (json) in the parquet schema metadata under key `btc`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    md = dict(table.schema.metadata or {})
    md[b"btc"] = json.dumps(meta, default=str).encode()
    pq.write_table(table.replace_schema_metadata(md), path)
    return path


def read_meta(path: Path) -> dict:
    """The `btc` metadata of a parquet file written by `write_parquet` ({} if none)."""
    md = pq.read_schema(path).metadata or {}
    return json.loads(md[b"btc"]) if b"btc" in md else {}


def write_json(path: Path, obj: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    return path


def credit_block() -> str:
    """The SPEC §2.1 credit, appended to every generated markdown report."""
    return f"\n---\n\n{CREDIT}\n{SOURCE}\n"


def with_credit(md: str) -> str:
    return md.rstrip("\n") + "\n" + credit_block()
