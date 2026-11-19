"""Result files. Every file records the commit it came from (SPEC §10): JSON files carry a `meta`
key, CSV and Markdown files a first comment line, and each output folder a MANIFEST.json with
SHA-256 hashes. Result files hold no timestamps, so a rerun on the same inputs and commit gives
identical bytes; timestamps go only to logs."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths, provenance
from ..provenance import with_credit


def git_meta() -> dict:
    """The git state of the code (SPEC §10), as `btc.provenance.git_state`: `state` is clean, dirty or
    unknown, and any modified, staged or untracked file that git does not ignore makes it dirty
    (`dirty` True). Plus the tags at HEAD."""
    st = provenance.git_state()
    try:
        tags = subprocess.run(["git", "tag", "--points-at", "HEAD"], cwd=paths.ROOT, capture_output=True,
                              text=True, check=True).stdout.split()
    except Exception:
        tags = []
    return {"commit": st["commit"], "describe": st["describe"], "state": st["state"], "dirty": st["dirty"],
            "status": st["status"], "tags_at_head": tags}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _clean(o.as_dict() if hasattr(o, "as_dict") else dataclasses.asdict(o))
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(o, pd.DataFrame):
        return _clean(o.to_dict(orient="records"))
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


class Writer:
    def __init__(self, out_dir: Path, meta: dict):
        self.out = Path(out_dir)
        self.meta = meta
        self.files = {}

    def _path(self, rel: str) -> Path:
        p = self.out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _done(self, p: Path):
        self.files[str(p.relative_to(self.out))] = hashlib.sha256(p.read_bytes()).hexdigest()

    def json(self, rel: str, obj) -> Path:
        p = self._path(rel)
        p.write_text(json.dumps({"meta": _clean(self.meta), **_clean(obj)}, indent=1, ensure_ascii=False) + "\n")
        self._done(p)
        return p

    def csv(self, rel: str, df: pd.DataFrame) -> Path:
        p = self._path(rel)
        head = "# " + json.dumps(_clean({k: self.meta.get(k) for k in ("commit", "describe", "data", "stage")}),
                                 ensure_ascii=False) + "  (read with skiprows=1)\n"
        p.write_text(head + df.to_csv(index=False))
        self._done(p)
        return p

    def text(self, rel: str, s: str) -> Path:
        """A Markdown file; on real data it carries the SPEC §2.1 data credit at the end."""
        p = self._path(rel)
        synthetic = str(self.meta.get("data", "")).startswith("synthetic")
        p.write_text(f"<!-- commit {self.meta.get('commit')} ({self.meta.get('describe')}); "
                     f"data: {self.meta.get('data')} -->\n\n" + (s if synthetic else with_credit(s)))
        self._done(p)
        return p

    def manifest(self, rel: str = "MANIFEST.json") -> Path:
        p = self._path(rel)
        p.write_text(json.dumps({"meta": _clean(self.meta), "files": dict(sorted(self.files.items()))},
                                indent=1) + "\n")
        return p
