#!/usr/bin/env python3
"""Per-patient tables for the public record, from the withheld per-forecast result files.

The per-forecast files `predictions_primary.csv` and `yardstick_predictions.csv` hold VitalDB cuff
values (`y_map`, `last_map`) with case IDs and times, so they are not committed (`.gitignore`,
`results/README.md`); their SHA-256 are in the MANIFEST.json of the run that wrote them. This script
reads them (read-only) after a run and writes, next to them, per-patient tables that are committed
instead. From these tables alone one can recompute every reported patient-level RMSE, Δ and relative
reduction with its paired subject bootstrap CI, G and g, the fast-cycling contrast and the
withheld-cuff yardstick N, D and F:

  per_patient_primary_wide.csv  one row per case with primary forecasts: caseid, subjectid,
                                fast_cycling (and nmb_recorded where the run records it),
                                n_forecasts, n_forecasts_waveform (waveform features available),
                                entered (>= 3 forecasts, SPEC §6.1) and rmse_<model> for every model
                                of the run (NaN below 3 forecasts). The RMSE equal those of
                                `per_patient_primary.csv` (checked).
  per_patient_yardstick.csv     one row per case with yardstick triples (SPEC §3): caseid, subjectid,
                                fast_cycling, n_triples, entered (>= 3 triples) and rmse_<model> for
                                L, LH, T, LH-P1, T-P1, U and any other yardstick-frame model the run
                                scored (the frozen scoring adds LH-ia, T-ia and the 2-reading-ahead
                                M2, which is scored on the same k = 2 rows).
  public_tables.json            SHA-256 of the tables written and of the files they were read from.

Folders handled, under `--results` (each only if its per-forecast files exist):
`sdev/confirmatory/`, `sob/frozen_scoring/`, `sob/frozen_scoring/sensitivity_including_nmb/`.

Every source file must be listed, with the same SHA-256, in a MANIFEST.json above it, so only
recorded outputs are exported. The tables are checked against the run's own `per_patient_primary.csv`
and summary.json before they are written. No model is fitted and no result file of the run is
changed. The output is deterministic (no timestamps). With `--out DIR` the tables go to DIR (same
relative layout) instead of next to their sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc.models.constants import MIN_FORECASTS_RMSE, MIN_TRIPLES  # noqa: E402
from btc.models.metrics import per_patient  # noqa: E402
from btc.models.outputs import git_meta  # noqa: E402
from btc.paths import ROOT  # noqa: E402

STAGES = ("sdev/confirmatory", "sob/frozen_scoring", "sob/frozen_scoring/sensitivity_including_nmb")
PRIMARY, YARDSTICK = "predictions_primary.csv", "yardstick_predictions.csv"
OUT_PRIMARY, OUT_YARDSTICK, OUT_MANIFEST = "per_patient_primary_wide.csv", "per_patient_yardstick.csv", \
    "public_tables.json"
CASE_ATTRS = ("subjectid", "fast_cycling", "nmb_recorded")       # constant within a case
WF_FLAG = "wf_available@real"
STAGE_NAME = "public per-patient table (scripts/export_public_tables.py)"
TOL = 1e-9                                                       # mmHg, for the checks against the run


class ExportError(RuntimeError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_result_csv(p: Path) -> tuple[dict, pd.DataFrame]:
    """(header meta, table) of a result CSV written by `btc.models.outputs.Writer`."""
    with p.open() as f:
        first = f.readline()
    if not first.startswith("# "):
        raise ExportError(f"{p.name}: no provenance line")
    head = json.loads(first[2:].split("  (read with skiprows=1)")[0])
    return head, pd.read_csv(p, skiprows=1)


def recorded_sha(path: Path, results: Path) -> tuple[str, str] | None:
    """(manifest path relative to `results`, recorded SHA-256) of `path`, from the nearest MANIFEST.json
    in a folder above it (up to `results`) that lists it, or None."""
    path, results = path.resolve(), results.resolve()
    d = path.parent
    while True:
        m = d / "MANIFEST.json"
        if m.exists():
            files = json.loads(m.read_text()).get("files", {})
            for base in [m.parent, *m.parent.parents]:
                try:
                    k = path.relative_to(base).as_posix()
                except ValueError:
                    continue
                if k in files:
                    return m.relative_to(results).as_posix(), files[k]
                if base == results:
                    break
        if d == results or results not in d.parents:
            return None
        d = d.parent


def _case_attrs(d: pd.DataFrame, name: str) -> pd.DataFrame:
    cols = [c for c in CASE_ATTRS if c in d.columns]
    g = d.groupby("caseid", sort=True)[cols]
    if (g.nunique(dropna=False) > 1).any().any():
        raise ExportError(f"{name}: a case attribute ({', '.join(cols)}) varies within a case")
    a = g.first()
    if "fast_cycling" in a:
        a["fast_cycling"] = a["fast_cycling"].astype(bool)
    if "nmb_recorded" in a:
        a["nmb_recorded"] = a["nmb_recorded"].astype(bool)
    return a


def _models(d: pd.DataFrame) -> list:
    return [c[len("pred_"):] for c in d.columns if c.startswith("pred_")]


def _rmse_wide(d: pd.DataFrame, models: list) -> tuple[pd.DataFrame, pd.Series]:
    caseid, y = d["caseid"].to_numpy(int), d["y_map"].to_numpy(float)
    pp = {m: per_patient(caseid, y, d[f"pred_{m}"].to_numpy(float)) for m in models}
    n = next(iter(pp.values()))["n"]
    return pd.DataFrame({f"rmse_{m}": pp[m]["rmse"] for m in models}), n


def primary_table(d: pd.DataFrame) -> pd.DataFrame:
    models = _models(d)
    rm, n = _rmse_wide(d, models)
    a = _case_attrs(d, PRIMARY)
    out = a.copy()
    out["n_forecasts"] = n.astype(int)
    if WF_FLAG in d.columns:
        out["n_forecasts_waveform"] = d.groupby("caseid", sort=True)[WF_FLAG].apply(
            lambda s: int(s.astype(bool).sum())).astype(int)
    out["entered"] = out["n_forecasts"] >= MIN_FORECASTS_RMSE
    return pd.concat([out, rm], axis=1).reset_index()


def yardstick_table(d: pd.DataFrame) -> pd.DataFrame:
    models = _models(d)
    rm, n = _rmse_wide(d, models)
    out = _case_attrs(d, YARDSTICK)[[c for c in ("subjectid", "fast_cycling") if c in d.columns]].copy()
    out["n_triples"] = n.astype(int)
    out["entered"] = out["n_triples"] >= MIN_TRIPLES
    return pd.concat([out, rm], axis=1).reset_index()


# ----------------------------------------------------------------------------
# checks against the run's own outputs (before anything is written)
# ----------------------------------------------------------------------------

def _close(a, b) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= TOL)))


def check_primary(wide: pd.DataFrame, stage: Path, summary: dict | None) -> dict:
    out = {}
    pl = stage / "per_patient_primary.csv"
    if pl.exists():
        _, long = read_result_csv(pl)
        piv = long.pivot(index="caseid", columns="model", values="rmse")
        w = wide.set_index("caseid")
        models = [c[len("rmse_"):] for c in w.columns if c.startswith("rmse_")]
        if sorted(piv.index) != sorted(w.index) or set(piv.columns) != set(models):
            raise ExportError(f"{stage.name}: cases or models differ from per_patient_primary.csv")
        for m in models:
            if not _close(w[f"rmse_{m}"], piv.loc[w.index, m]):
                raise ExportError(f"{stage.name}: RMSE of {m} differs from per_patient_primary.csv")
        out["rmse_equal_to_per_patient_primary"] = True
    if summary is not None:
        n_key = "N" if "N" in summary else "n"
        e = wide.loc[wide["entered"]]
        if int(len(e)) != int(summary[n_key]) or int(e["n_forecasts"].sum()) != int(summary["n_forecasts"]):
            raise ExportError(f"{stage.name}: entering patients or forecasts differ from summary.json")
        c = summary["contrasts"]["M4 vs M2"]
        if not _close(e["rmse_M4"].mean() - e["rmse_M2"].mean(), c["delta"]):
            raise ExportError(f"{stage.name}: Δ (M4 vs M2) differs from summary.json")
        out["entering_patients_forecasts_and_delta_equal_to_summary"] = True
    return out


def check_yardstick(wide: pd.DataFrame, stage: Path, summary: dict | None) -> dict:
    if summary is None:
        return {}
    y = summary.get("yardstick_all", summary.get("yardstick"))
    if not y or not y.get("n_patients"):
        return {}
    e = wide.loc[wide["entered"]]
    if int(len(e)) != int(y["n_patients"]) or int(e["n_triples"].sum()) != int(y["n_triples"]):
        raise ExportError(f"{stage.name}: yardstick patients or triples differ from summary.json")
    N = e["rmse_L"].mean() - e["rmse_T"].mean()
    D = e["rmse_L"].mean() - e["rmse_U"].mean()
    if not (_close(N, y["N"]) and _close(D, y["D"])):
        raise ExportError(f"{stage.name}: yardstick N or D differs from summary.json")
    return {"yardstick_patients_triples_N_D_equal_to_summary": True}


# ----------------------------------------------------------------------------

def _write_csv(p: Path, df: pd.DataFrame, head: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# " + json.dumps(head, ensure_ascii=False) + "  (read with skiprows=1)\n" + df.to_csv(index=False))


def export_stage(results: Path, rel: str, out_root: Path, g: dict) -> dict | None:
    stage = results / rel
    srcs = [f for f in (PRIMARY, YARDSTICK) if (stage / f).exists()]
    if not srcs:
        return None
    summary = json.loads((stage / "summary.json").read_text()) if (stage / "summary.json").exists() else None
    sources, tables, checks = {}, {}, {}
    for f in srcs:
        p = stage / f
        sha = _sha(p)
        rec = recorded_sha(p, results)
        if rec is None:
            raise ExportError(f"{rel}/{f} is not listed in any MANIFEST.json above it")
        if rec[1] != sha:
            raise ExportError(f"{rel}/{f}: SHA-256 differs from {rec[0]}")
        head, d = read_result_csv(p)
        sources[f] = {"sha256": sha, "manifest": rec[0], "commit": head.get("commit"),
                      "describe": head.get("describe"), "data": head.get("data"), "stage": head.get("stage")}
        if f == PRIMARY:
            t = primary_table(d)
            checks.update(check_primary(t, stage, summary))
            tables[OUT_PRIMARY] = (t, f, head)
        else:
            t = yardstick_table(d)
            checks.update(check_yardstick(t, stage, summary))
            tables[OUT_YARDSTICK] = (t, f, head)
    dest = out_root / rel
    written = {}
    for name, (t, f, head) in tables.items():
        h = {"commit": g.get("commit"), "describe": g.get("describe"), "data": head.get("data"),
             "stage": STAGE_NAME, "source": f, "source_sha256": sources[f]["sha256"],
             "source_commit": head.get("commit")}
        _write_csv(dest / name, t, h)
        written[name] = _sha(dest / name)
    man = {"meta": {"commit": g.get("commit"), "describe": g.get("describe"), "stage": STAGE_NAME},
           "folder": rel, "files": written, "sources": sources, "checks": checks,
           "note": "The sources hold VitalDB cuff values and are not distributed; the tables written here "
                   "are per-patient summaries (results/README.md)."}
    (dest / OUT_MANIFEST).write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results", help="result root (default results/)")
    ap.add_argument("--out", type=Path, default=None, help="write the tables here instead of next to the sources")
    a = ap.parse_args(argv)
    results = a.results
    out_root = a.out or results
    g = git_meta()
    done = 0
    try:
        for rel in STAGES:
            man = export_stage(results, rel, out_root, g)
            if man is None:
                continue
            done += 1
            print(f"{rel}: wrote {', '.join(man['files'])} (checks: {', '.join(man['checks']) or 'none'})")
    except ExportError as ex:
        print(f"refused: {ex}", file=sys.stderr)
        return 2
    if not done:
        print(f"refused: no per-forecast result file under {results}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
