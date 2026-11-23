#!/usr/bin/env python3
"""Download one cohort of VitalDB spinal-anesthesia cases (S-dev, or S-ob at the frozen scoring).

This script only downloads and checks data integrity. It does not extract cuff
readings, compute features, fit models or summarise blood-pressure values.

Cohort access (SPEC §9): `--cohort sdev` (default) requests only S-dev signals.
`--cohort sob` requests only S-ob signals and runs only at the frozen scoring: it
stops before any request unless the environment variable BTC_FROZEN_SCORING=1 is
set. A run never requests a signal of the other cohort (hard guard, `guard`).

Data source
-----------
VitalDB (Lee HC, et al. Sci Data 2022, doi:10.1038/s41597-022-01411-5),
public API at https://api.vitaldb.net, released under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/) and the VitalDB Data Use
Agreement. Nothing downloaded here is redistributed: everything is written to
the git-ignored `data/` folder. Changes made by this script: the selected
tracks are copied out of each case file into one NumPy archive per case;
waveforms are converted from stored integers to physical units (float32) and
cut to 0-3660 s; nothing is resampled or filtered.

Cohorts
-------
spinal = case-table `ane_type` equal to "spinal" (case-insensitive, stripped).
S-ob   = spinal cases whose `opname` contains "cesarean", "c-sec" or
         "caesarean" (case-insensitive). S-ob is the held-out test series:
         its signals are requested only with `--cohort sob` and
         BTC_FROZEN_SCORING=1 (hard guard, see `guard`). Otherwise S-ob rows
         appear in `cohorts.csv` as case-table metadata only.
S-dev  = all other spinal cases. Expected: 273 spinal, 21 S-ob, 252 S-dev
         from 251 subjects; the script stops before any signal request if the
         counts differ.

Method (which source, and why)
------------------------------
Primary source: the packed case file of the versioned open dataset,
`https://api.vitaldb.net/<DATASET_VERSION>/<caseid>.vital`. This is the file
that `vitaldb.load_case` (package 1.7.x) reads; the package documents it as the
timeline-correct source in which all tracks share one clock, and notes that
the older per-track CSV endpoint (`https://api.vitaldb.net/<tid>`) can be
shifted by a per-case constant. The file is downloaded whole, checked
(content length and gzip CRC), parsed locally with `vitaldb.VitalFile`, and
deleted after extraction. The vitaldb package itself never touches the network
here.

Cross-checks (integrity only):
* The original, unpacked case file (`https://api.vitaldb.net/<caseid>.vital`)
  is also downloaded. Its waveform records are the blocks as recorded, each
  with its own timestamp, so the sampling rate is estimated from the data
  (median of block samples / time to the next block, over gaps < 2 s) instead
  of being taken from the header. Its blocks are also re-assembled with the
  packing rule and compared sample-for-sample with the packed waveform, and
  its numeric records are compared with the packed ones. `<key>_overlap_frac`
  is the share of original samples that the packed file loses because blocks
  placed at the header rate overlap (a sign that the header rate or the block
  timestamps are wrong); cases above 1% are flagged, never corrected.
* The per-track CSV of `Solar8000/HR` is fetched and its record times are
  compared with the case file's, to measure the per-case CSV time shift
  (`csv_hr_shift_s` = case-file time minus CSV time). Only times and record
  identity are compared.

Time zero
---------
t = 0 is the start of the recording, i.e. the `dtstart` field in the header of
the case file (VitalDB anonymises it to 2100-01-01 00:00 UTC). This is the
`casestart` = 0 of the case table. Every time below is `record dt - dtstart`
in seconds.
* Numeric tracks (NIBP_SBP/DBP/MBP, HR, PHEN_RATE): every native record of the
  whole case, as (time_s, value), in file order, no resampling.
* Waveforms (PLETH, ECG_II): the packed file stores each waveform as one
  dense run of samples at the header sampling rate, starting at the first
  block's timestamp, with gaps filled by the format's NaN code. Sample i is at
  `<key>_t0 + i / <key>_srate`. Only samples with 0 <= t < 3660 s are kept.
  The first sample is generally not at t = 0 (recording begins about 1 s
  before the waveform); the array is not shifted onto a grid at t = 0.
  Gap samples are NaN. If a header rate is not 500 Hz, the track is kept at
  its own rate and flagged; it is never resampled.

Outputs (paths relative to the repository root; <cohort> = sdev or sob)
----------------------------------------------------------------------
data/cohorts.csv                      caseid, subjectid, cohort, opname and case-table fields
data/raw/<cohort>/<caseid>.npz        one file per case (arrays + `meta_json`)
data/raw/<cohort>_manifest.csv        one row per case: presence, counts, times, checks
data/raw/<cohort>_checksums.csv       SHA-256 of every stored array (SPEC §10; no data)
data/raw/<cohort>_download_log.json   run summary: versions, counts, timing, failures
data/raw/requests.csv                 every HTTP request made (case id, cohort, URL, status)
data/raw/meta/                        cached case table and track list

Re-running skips cases whose archive is complete (same SCRIPT_VERSION) and
retries the rest. `--checksums-only` rewrites the checksum list from the local
archives and makes no request.

Usage: python scripts/download.py [--cohort sdev|sob] [--workers 4] [--caseids 15,39] [--limit N]
"""

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import os
import platform
import random
import shutil
import struct
import sys
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import vitaldb

SCRIPT_VERSION = "1.1"
API = "https://api.vitaldb.net"
DATASET_VERSION = "1.0.1"  # versioned open-dataset path used by vitaldb 1.7.x

EXPECTED = {"spinal": 273, "s_ob": 21, "s_dev": 252, "s_dev_subjects": 251}
OB_PATTERNS = ("cesarean", "c-sec", "caesarean")

# short key -> VitalDB track name
NUMERIC = {
    "sbp": "Solar8000/NIBP_SBP",
    "dbp": "Solar8000/NIBP_DBP",
    "mbp": "Solar8000/NIBP_MBP",
    "hr": "Solar8000/HR",
    "phen": "Orchestra/PHEN_RATE",
}
WAVES = {
    "pleth": "SNUADC/PLETH",
    "ecg": "SNUADC/ECG_II",
}
TRACKS = {**NUMERIC, **WAVES}
CSV_SHIFT_TRACK = "Solar8000/HR"

WAVE_KEEP_S = (0.0, 3660.0)
EXPECTED_SRATE = 500.0
SRATE_TOL = 0.01  # relative tolerance for the data-based rate estimate (registered: 1%)
OVERLAP_TOL = 0.01  # flag if more than 1% of original samples are overwritten

# NaN code used to fill gaps in packed waveforms, by vital-file sample format
FMT_NAN = {1: np.nan, 2: np.nan, 3: -128, 4: 255, 5: -32768, 6: 65535,
           7: -2147483648, 8: 4294967295}
FMT_DTYPE = {1: "<f4", 2: "<f8", 3: "<i1", 4: "<u1", 5: "<i2", 6: "<u2",
             7: "<i4", 8: "<u4"}

REFUSED_STATUS = {401, 403, 406, 451}
TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}
MAX_WORKERS = 4

FROZEN_SCORING_ENV = "BTC_FROZEN_SCORING"   # SPEC §9: S-ob only at the frozen scoring
COHORT_LABEL = {"sdev": "S-dev", "sob": "S-ob"}

# Filled in main(); used by the hard guard.
S_DEV: frozenset = frozenset()
S_OB: frozenset = frozenset()
COHORT = ""
ROOT = Path(".")


# ----------------------------------------------------------------------------
# Guard: only the selected cohort's signals may be requested
# ----------------------------------------------------------------------------

def guard(caseid):
    """Abort the whole process if `caseid` is not a case of the selected cohort,
    or if it is an S-ob case and BTC_FROZEN_SCORING=1 is not set.

    Called before every request that names a case. `os._exit` stops all
    threads at once, so no further request can follow.
    """
    cid = int(caseid)
    allowed = {"sdev": S_DEV, "sob": S_OB}.get(COHORT, frozenset())
    other = S_OB if COHORT == "sdev" else S_DEV
    sob_ok = COHORT != "sob" or os.environ.get(FROZEN_SCORING_ENV) == "1"
    if not S_DEV or not S_OB or cid not in allowed or cid in other or not sob_ok:
        sys.stderr.write(f"ABORT: case {cid} may not be requested in this run "
                         f"(cohort {COHORT or 'unset'}); no request was made for it.\n")
        sys.stderr.flush()
        os._exit(3)
    return cid


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------

class Refused(Exception):
    pass


class Http:
    """Small HTTP client with the default User-Agent. A refused request
    (REFUSED_STATUS) is not retried: it raises Refused, naming the HTTP status
    and the URL, and the run stops. Transient errors are retried with
    exponential backoff."""

    def __init__(self, log_path, timeout=120, retries=4):
        self.timeout = timeout
        self.retries = retries
        self.lock = threading.Lock()
        self.local = threading.local()
        self.log_path = log_path
        new = not log_path.exists()
        self.log_f = open(log_path, "a", newline="")
        self.log_w = csv.writer(self.log_f)
        if new:
            self.log_w.writerow(["utc", "caseid", "cohort", "kind", "url",
                                 "status", "bytes", "user_agent"])

    def _session(self):
        s = getattr(self.local, "s", None)
        if s is None:
            s = requests.Session()
            self.local.s = s
        return s

    def _log(self, caseid, kind, url, status, nbytes):
        cohort = "" if caseid is None else ("S-dev" if caseid in S_DEV else "S-ob")
        with self.lock:
            # the user_agent column stays, so that the log keeps one layout; it is always "default"
            self.log_w.writerow([dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                 "" if caseid is None else caseid, cohort, kind, url,
                                 status, nbytes, "default"])
            self.log_f.flush()

    def _once(self, url, dest, caseid, kind):
        with self._session().get(url, stream=True, timeout=self.timeout) as r:
            if r.status_code != 200:
                self._log(caseid, kind, url, r.status_code, 0)
                return r.status_code, None
            # Content-Length (if given) refers to the encoded body: count raw bytes.
            expected = r.headers.get("Content-Length")
            n = 0
            buf = None if dest else io.BytesIO()
            out = open(dest, "wb") if dest else buf
            try:
                for chunk in r.raw.stream(1 << 20, decode_content=False):
                    n += len(chunk)
                    out.write(chunk)
            finally:
                if dest:
                    out.close()
            self._log(caseid, kind, url, 200, n)
            if expected is not None and int(expected) != n:
                raise IOError(f"short read {n} of {expected} bytes")
            enc = r.headers.get("Content-Encoding", "")
            return 200, (n if dest else (buf.getvalue(), enc))

    def get(self, url, caseid=None, kind="", dest=None):
        """Fetch `url`. With `dest`, stream raw bytes to that file and return
        its size; otherwise return (body bytes, content-encoding)."""
        if caseid is not None:
            guard(caseid)
        attempt = 0
        while True:
            try:
                status, res = self._once(url, dest, caseid, kind)
            except (requests.ConnectionError, requests.Timeout,
                    requests.exceptions.ChunkedEncodingError, IOError) as e:
                status, res = None, e
            if status == 200:
                return res
            if status in REFUSED_STATUS:
                raise Refused(f"HTTP {status} for {url}")
            if status is not None and status not in TRANSIENT_STATUS:
                raise IOError(f"HTTP {status}: {url}")
            attempt += 1
            if attempt > self.retries:
                raise IOError(f"giving up after {self.retries} retries "
                              f"({status if status else type(res).__name__}): {url}")
            time.sleep(2 ** attempt + random.random())


def decode_body(body, enc):
    if enc == "gzip" or body[:2] == b"\x1f\x8b":
        return gzip.decompress(body)
    if enc == "deflate":
        return zlib.decompress(body)
    return body


# ----------------------------------------------------------------------------
# Case table, track list, cohorts
# ----------------------------------------------------------------------------

def load_meta(http, meta_dir, refresh):
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in ("cases", "trks"):
        p = meta_dir / f"{name}.csv"
        if refresh or not p.exists():
            body, enc = http.get(f"{API}/{name}", kind=f"table:{name}")
            tmp = p.with_suffix(".csv.part")
            tmp.write_bytes(decode_body(body, enc))
            os.replace(tmp, p)
        out[name] = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
    return out["cases"], out["trks"]


def define_cohorts(cases):
    ane = cases["ane_type"].astype(str).str.strip().str.lower()
    sp = cases[ane == "spinal"].copy()
    op = sp["opname"].astype(str).str.lower()
    is_ob = np.zeros(len(sp), dtype=bool)
    for pat in OB_PATTERNS:
        is_ob |= op.str.contains(pat, regex=False).to_numpy()
    sp["cohort"] = np.where(is_ob, "S-ob", "S-dev")
    sp = sp.sort_values("caseid")
    counts = {
        "spinal": len(sp),
        "s_ob": int(is_ob.sum()),
        "s_dev": int((~is_ob).sum()),
        "s_dev_subjects": int(sp.loc[sp.cohort == "S-dev", "subjectid"].nunique()),
        "s_ob_subjects": int(sp.loc[sp.cohort == "S-ob", "subjectid"].nunique()),
        "subjects_in_both": len(set(sp.loc[sp.cohort == "S-dev", "subjectid"])
                                & set(sp.loc[sp.cohort == "S-ob", "subjectid"])),
    }
    return sp, counts


def write_cohorts(sp, path):
    cols = ["caseid", "subjectid", "cohort", "opname", "casestart", "anestart",
            "opstart", "intraop_phe", "intraop_eph", "emop"]
    sp[cols].to_csv(path, index=False)


# ----------------------------------------------------------------------------
# Case-file parsing
# ----------------------------------------------------------------------------

def check_gzip(path):
    """Decompress the whole file (verifies the gzip CRC); return its size."""
    n = 0
    with gzip.open(path, "rb") as g:
        while True:
            b = g.read(1 << 22)
            if not b:
                break
            n += len(b)
    return n


def read_header(path):
    """Read dgmt, dtstart, dtend and the packed flag from a vital-file header."""
    with gzip.open(path, "rb") as g:
        if g.read(4) != b"VITA":
            raise ValueError("not a vital file")
        g.read(4)  # format version
        hlen = struct.unpack("<H", g.read(2))[0]
        h = g.read(hlen)
    if hlen < 26:
        raise ValueError("header has no dtstart/dtend")
    return {
        "dgmt": struct.unpack_from("<h", h, 0)[0],
        "dtstart": struct.unpack_from("<d", h, 10)[0],
        "dtend": struct.unpack_from("<d", h, 18)[0],
        "packed": bool(h[26] == 1) if hlen >= 27 else False,
    }


def parse_vital(path):
    """Parse only the wanted tracks with vitaldb.VitalFile (local file)."""
    vf = vitaldb.VitalFile()
    vf.dtstart = vf.dtend = 0
    ok = vf.load_vital(str(path), list(TRACKS.values()))
    if ok is False:
        raise ValueError("vitaldb could not parse the case file")
    # keep exact name matches only (VitalFile also matches by suffix)
    return {k: vf.trks.get(name) for k, name in TRACKS.items()}


def assemble_wave(recs, srate, fmt):
    """Merge waveform blocks into one dense run with the packing rule used for
    the packed open-dataset files: sort by time, place each block at
    round((dt - dt0) * srate), later blocks overwrite, gaps get the NaN code.
    Returns (dt0, raw array, stats). stats: `conflicts` = blocks sharing a
    timestamp with an earlier block but holding different samples;
    `unique_samples` = samples in distinct blocks (exact repeats removed);
    `overwritten` = of those, samples overwritten by a later distinct block.
    Overlap means the block timestamps advance more slowly than the header
    rate implies, i.e. the header rate does not fit the data."""
    recs = sorted(recs, key=lambda r: r["dt"])
    dtype = np.dtype(FMT_DTYPE[fmt])
    dt0 = recs[0]["dt"]
    dt_end = max(r["dt"] + len(r["val"]) / srate for r in recs)
    n = int(round((dt_end - dt0) * srate))
    out = np.full(n, FMT_NAN[fmt], dtype=dtype)
    filled = np.zeros(n, dtype=bool)
    seen = {}
    conflicts = unique = overwritten = 0
    for r in recs:
        v = np.asarray(r["val"], dtype=dtype)
        prev = seen.get(r["dt"])
        repeat = prev is not None and np.array_equal(prev, v)
        if prev is not None and not repeat:
            conflicts += 1
        seen[r["dt"]] = v
        i = int(round((r["dt"] - dt0) * srate))
        s, e = max(0, i), min(n, i + len(v))
        if e > s:
            if not repeat:
                unique += e - s
                overwritten += int(filled[s:e].sum())
            out[s:e] = v[s - i:e - i]
            filled[s:e] = True
    return dt0, out, {"conflicts": conflicts, "unique_samples": unique,
                      "overwritten": overwritten}


def estimate_srate(recs):
    """Sampling rate from original block timestamps: median over consecutive
    distinct blocks (gap < 2 s) of block samples / time to the next block."""
    if len(recs) < 2:
        return float("nan")
    recs = sorted(recs, key=lambda r: r["dt"])
    t = np.array([r["dt"] for r in recs])
    n = np.array([len(r["val"]) for r in recs], dtype=float)
    keep = np.r_[True, np.diff(t) > 0]  # drop repeated blocks
    t, n = t[keep], n[keep]
    d = np.diff(t)
    m = (d > 0) & (d < 2.0)
    if not m.any():
        return float("nan")
    return float(np.median(n[:-1][m] / d[m]))


def wave_to_physical(raw, trk):
    nanmask = np.isnan(raw) if trk.fmt in (1, 2) else (raw == FMT_NAN[trk.fmt])
    v = raw.astype(np.float64)
    if trk.fmt > 2:  # integer formats carry gain/offset (as in vitaldb)
        v = v * trk.gain + trk.offset
    v[nanmask] = np.nan
    return v.astype(np.float32)


def numeric_records(trk, t0):
    t = np.array([r["dt"] for r in trk.recs], dtype=np.float64) - t0
    v = np.array([r["val"] for r in trk.recs], dtype=np.float64)
    return t, v


def csv_shift(body_enc, t_file, v_file):
    """Median (case-file time - CSV time) for the HR track, if the records
    match one-to-one; else NaN. Returns (shift, max deviation, note)."""
    body = decode_body(*body_enc)
    arr = pd.read_csv(io.BytesIO(body), na_values="-nan(ind)").to_numpy(dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return float("nan"), float("nan"), "csv unreadable"
    tc, vc = arr[:, 0], arr[:, 1]
    if len(tc) != len(t_file):
        return float("nan"), float("nan"), f"record count differs ({len(tc)} vs {len(t_file)})"
    if not np.allclose(vc, v_file, rtol=1e-4, atol=1e-3, equal_nan=True):
        return float("nan"), float("nan"), "record values differ"
    d = t_file - tc
    s = float(np.median(d))
    return s, float(np.max(np.abs(d - s))), ""


# ----------------------------------------------------------------------------
# One case
# ----------------------------------------------------------------------------

def process_case(caseid, http, trk_index, case_row, out_dir, tmp_root, csv_check):
    cid = guard(caseid)
    meta = {"caseid": cid, "script_version": SCRIPT_VERSION, "flags": []}
    flags = meta["flags"]
    tmp = tmp_root / str(cid)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        packed_p = tmp / "packed.vital"
        legacy_p = tmp / "legacy.vital"
        # primary source: any failure here fails the case
        http.get(f"{API}/{DATASET_VERSION}/{cid}.vital", cid, "vital:packed", dest=packed_p)
        meta["vital_packed_bytes"] = packed_p.stat().st_size
        check_gzip(packed_p)
        hdr = read_header(packed_p)
        if not hdr["packed"]:
            flags.append("primary_file_not_packed")
        t0 = hdr["dtstart"]
        if not (t0 > 0):
            raise ValueError("header dtstart missing")
        meta["dur_hdr_s"] = hdr["dtend"] - t0
        meta["caseend"] = case_row.get("caseend")
        P = parse_vital(packed_p)

        # original file: cross-check only; a failure is flagged, not fatal
        try:
            http.get(f"{API}/{cid}.vital", cid, "vital:original", dest=legacy_p)
            meta["vital_original_bytes"] = legacy_p.stat().st_size
            check_gzip(legacy_p)
            if abs(read_header(legacy_p)["dtstart"] - t0) > 1e-6:
                flags.append("original_dtstart_differs")
            L = parse_vital(legacy_p)
        except Refused:
            raise
        except Exception as e:
            flags.append("original_unavailable")
            meta["original_error"] = clean(f"{type(e).__name__}: {e}")
            L = {k: None for k in TRACKS}
        arrays = {}

        # numeric tracks: all native records, whole case
        num_eq = True
        for k, name in NUMERIC.items():
            meta[f"{k}_idx"] = name in trk_index
            trk = P[k]
            present = trk is not None and len(trk.recs) > 0
            meta[f"{k}_present"] = present
            if present != meta[f"{k}_idx"]:
                flags.append(f"{k}_index_vs_file")
            if not present:
                meta[f"{k}_n"] = 0
                continue
            if trk.type != 2:
                flags.append(f"{k}_not_numeric_type")
            t, v = numeric_records(trk, t0)
            arrays[f"{k}_t"] = t
            arrays[f"{k}_v"] = v
            meta[f"{k}_n"] = int(len(t))
            meta[f"{k}_t_first"] = float(t.min())
            meta[f"{k}_t_last"] = float(t.max())
            meta[f"{k}_fmt"] = int(trk.fmt)
            meta[f"{k}_unit"] = trk.unit
            if np.any(np.diff(t) < 0):
                flags.append(f"{k}_times_not_sorted")
            if np.any(t < 0):
                flags.append(f"{k}_records_before_t0")
            lt = L[k]
            if lt is None:
                num_eq = False
            else:
                tl, vl = numeric_records(lt, t0)
                o1, o2 = np.argsort(t, kind="stable"), np.argsort(tl, kind="stable")
                if not (len(tl) == len(t) and np.array_equal(t[o1], tl[o2])
                        and np.array_equal(v[o1], vl[o2], equal_nan=True)):
                    num_eq = False
        meta["num_original_eq"] = num_eq
        if not num_eq:
            flags.append("numeric_differs_from_original")

        # waveforms: native rate, float32, 0-3660 s
        for k, name in WAVES.items():
            meta[f"{k}_idx"] = name in trk_index
            trk = P[k]
            present = trk is not None and len(trk.recs) > 0
            meta[f"{k}_present"] = present
            if present != meta[f"{k}_idx"]:
                flags.append(f"{k}_index_vs_file")
            if not present:
                meta[f"{k}_n"] = 0
                continue
            sr = float(trk.srate)
            meta[f"{k}_srate_hdr"] = sr
            if trk.type != 1 or sr <= 0:
                raise ValueError(f"{name} is not a waveform track")
            if len(trk.recs) == 1:
                wdt0, raw = trk.recs[0]["dt"], np.asarray(trk.recs[0]["val"])
            else:
                flags.append(f"{k}_packed_multiblock")
                wdt0, raw, _ = assemble_wave(trk.recs, sr, trk.fmt)
            # cross-check against the original blocks
            lt = L[k]
            if lt is None or not lt.recs:
                meta[f"{k}_original_eq"] = False
                meta[f"{k}_srate_est"] = float("nan")
                meta[f"{k}_overlap_frac"] = float("nan")
            else:
                ldt0, lraw, st = assemble_wave(lt.recs, float(lt.srate), lt.fmt)
                meta[f"{k}_original_eq"] = bool(abs(ldt0 - wdt0) < 1e-6
                                                and np.array_equal(lraw, raw))
                meta[f"{k}_dup_conflicts"] = int(st["conflicts"])
                meta[f"{k}_overlap_frac"] = st["overwritten"] / max(1, st["unique_samples"])
                if meta[f"{k}_overlap_frac"] > OVERLAP_TOL:
                    flags.append(f"{k}_overlap_{meta[f'{k}_overlap_frac']:.3f}")
                meta[f"{k}_original_blocks"] = len(lt.recs)
                meta[f"{k}_srate_est"] = estimate_srate(lt.recs)
            if not meta[f"{k}_original_eq"]:
                flags.append(f"{k}_differs_from_original")
            est = meta[f"{k}_srate_est"]
            if sr != EXPECTED_SRATE:
                flags.append(f"{k}_srate_hdr_{sr:g}")
            if not (abs(est - EXPECTED_SRATE) <= SRATE_TOL * EXPECTED_SRATE):
                flags.append(f"{k}_srate_est_{est:.2f}")
            tf = wdt0 - t0
            meta[f"{k}_track_t_first"] = float(tf)
            meta[f"{k}_track_t_last"] = float(tf + (len(raw) - 1) / sr)
            meta[f"{k}_track_n"] = int(len(raw))
            i0 = max(0, math.ceil((WAVE_KEEP_S[0] - tf) * sr - 1e-6))
            i1 = min(len(raw), math.ceil((WAVE_KEEP_S[1] - tf) * sr - 1e-6))
            i1 = max(i1, i0)
            x = wave_to_physical(raw[i0:i1], trk)
            arrays[k] = x
            arrays[f"{k}_t0"] = np.float64(tf + i0 / sr)
            arrays[f"{k}_srate"] = np.float64(sr)
            meta[f"{k}_n"] = int(len(x))
            meta[f"{k}_t_first"] = float(tf + i0 / sr) if len(x) else float("nan")
            meta[f"{k}_t_last"] = float(tf + (i1 - 1) / sr) if len(x) else float("nan")
            meta[f"{k}_nan_frac"] = float(np.isnan(x).mean()) if len(x) else float("nan")
            meta[f"{k}_gain"] = float(trk.gain)
            meta[f"{k}_offset"] = float(trk.offset)
            meta[f"{k}_fmt"] = int(trk.fmt)
            meta[f"{k}_unit"] = trk.unit

        # CSV time-shift cross-check on HR (times and record identity only)
        meta["csv_hr_shift_s"] = float("nan")
        if csv_check and "hr_t" in arrays:
            tid = trk_index.get(CSV_SHIFT_TRACK)
            if tid:
                try:
                    be = http.get(f"{API}/{tid}", cid, "csv:Solar8000/HR")
                    s, dev, note = csv_shift(be, arrays["hr_t"], arrays["hr_v"])
                    meta["csv_hr_shift_s"] = s
                    meta["csv_hr_shift_maxdev_s"] = dev
                    if note:
                        flags.append("csv_hr_" + note.split(" (")[0].replace(" ", "_"))
                except Refused:
                    raise
                except Exception as e:  # a failed cross-check does not fail the case
                    flags.append("csv_hr_fetch_failed")
                    meta["csv_hr_error"] = clean(str(e))

        meta["complete"] = True
        meta["status"] = "ok"
        meta["downloaded_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        arrays["meta_json"] = np.array(json.dumps(meta, default=_json_default))
        final = out_dir / f"{cid}.npz"
        part = out_dir / f"{cid}.npz.part"
        with open(part, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(part, final)
        meta["file_bytes"] = final.stat().st_size
        return meta
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def clean(msg):
    """Remove local paths from messages that end up in outputs."""
    for p in (str(ROOT.resolve()), str(Path.home())):
        msg = msg.replace(p, ".")
    return msg


def load_done(path):
    """Return the stored meta if `path` is a complete archive of this version."""
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
        if meta.get("complete") and meta.get("script_version") == SCRIPT_VERSION:
            meta["file_bytes"] = path.stat().st_size
            return meta
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------

def manifest_columns():
    cols = ["caseid", "status", "flags", "error", "file_bytes",
            "vital_packed_bytes", "vital_original_bytes", "dur_hdr_s", "caseend",
            "csv_hr_shift_s", "csv_hr_shift_maxdev_s", "num_original_eq"]
    for k in NUMERIC:
        cols += [f"{k}_idx", f"{k}_present", f"{k}_n", f"{k}_t_first", f"{k}_t_last"]
    for k in WAVES:
        cols += [f"{k}_idx", f"{k}_present", f"{k}_n", f"{k}_t_first", f"{k}_t_last",
                 f"{k}_nan_frac", f"{k}_srate_hdr", f"{k}_srate_est",
                 f"{k}_overlap_frac", f"{k}_original_eq", f"{k}_original_blocks",
                 f"{k}_dup_conflicts",
                 f"{k}_track_n", f"{k}_track_t_first", f"{k}_track_t_last"]
    return cols


def write_manifest(rows, path):
    cols = manifest_columns()
    out = []
    for r in sorted(rows, key=lambda r: r["caseid"]):
        r = dict(r)
        r["flags"] = ";".join(r.get("flags", []))
        out.append({c: r.get(c, "") for c in cols})
    part = path.with_suffix(".csv.part")
    pd.DataFrame(out, columns=cols).to_csv(part, index=False)
    os.replace(part, path)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def write_checksums(out_dir, caseids, path):
    """SHA-256 of the bytes of every array stored in each case archive (SPEC §10). Only
    checksums are written; the list can be published to detect later changes in the data."""
    rows = []
    for c in sorted(caseids):
        f = out_dir / f"{c}.npz"
        if not f.exists():
            continue
        with np.load(f, allow_pickle=False) as z:
            for k in sorted(z.files):
                a = np.ascontiguousarray(z[k])
                rows.append({"caseid": c, "array": k, "dtype": str(a.dtype), "n": int(a.size),
                             "sha256": hashlib.sha256(a.tobytes()).hexdigest()})
    part = path.with_suffix(".csv.part")
    pd.DataFrame(rows, columns=["caseid", "array", "dtype", "n", "sha256"]).to_csv(part, index=False)
    os.replace(part, path)
    return len(rows)


def main(argv=None):
    global S_DEV, S_OB, COHORT, ROOT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", choices=("sdev", "sob"), default="sdev")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]),
                    help="repository root (default: parent of scripts/)")
    ap.add_argument("--workers", type=int, default=4, help="parallel cases (max 4)")
    ap.add_argument("--caseids", default="", help="comma-separated case ids of the cohort (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--refresh-meta", action="store_true", help="re-download case table and track list")
    ap.add_argument("--no-csv-check", action="store_true", help="skip the HR CSV time-shift check")
    ap.add_argument("--checksums-only", action="store_true",
                    help="rewrite <cohort>_checksums.csv from the local archives; no request")
    args = ap.parse_args(argv)
    # SPEC §9: checked before anything else, so no S-ob request or read can happen without it
    if args.cohort == "sob" and os.environ.get(FROZEN_SCORING_ENV) != "1":
        sys.exit(f"STOP: S-ob signals are downloaded only at the frozen scoring (SPEC §9 step 7); "
                 f"set {FROZEN_SCORING_ENV}=1 for that run. Nothing requested.")
    COHORT = args.cohort
    label = COHORT_LABEL[COHORT]

    ROOT = Path(args.root)
    data = ROOT / "data"
    raw = data / "raw"
    out_dir = raw / COHORT
    tmp_root = raw / f"tmp_{COHORT}"
    if args.checksums_only:
        ids = [int(f.stem) for f in out_dir.glob("*.npz")]
        n = write_checksums(out_dir, ids, raw / f"{COHORT}_checksums.csv")
        print(f"{n} checksums of {len(ids)} {label} archives written")
        return
    for d in (out_dir, tmp_root):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    http = Http(raw / "requests.csv", timeout=args.timeout, retries=args.retries)
    cases, trks = load_meta(http, raw / "meta", args.refresh_meta)
    sp, counts = define_cohorts(cases)
    print("cohort counts:", counts)
    for k, v in EXPECTED.items():
        if counts[k] != v:
            sys.exit(f"STOP: {k} = {counts[k]}, expected {v}. Nothing downloaded.")
    if counts["subjects_in_both"]:
        print(f"note: {counts['subjects_in_both']} subject(s) in both cohorts")
    S_DEV = frozenset(int(c) for c in sp.loc[sp.cohort == "S-dev", "caseid"])
    S_OB = frozenset(int(c) for c in sp.loc[sp.cohort == "S-ob", "caseid"])
    assert len(S_DEV) == EXPECTED["s_dev"] and len(S_OB) == EXPECTED["s_ob"]
    assert not (S_DEV & S_OB)
    write_cohorts(sp, data / "cohorts.csv")
    selected = S_DEV if COHORT == "sdev" else S_OB

    todo = sorted(selected)
    if args.caseids:
        req = [int(x) for x in args.caseids.split(",") if x.strip()]
        for c in req:
            guard(c)  # aborts on any id outside the selected cohort
        todo = req
    if args.limit:
        todo = todo[:args.limit]
    for c in todo:
        guard(c)

    # track index restricted to the selected cohort: {caseid: {tname: tid}}
    trks = trks[trks["caseid"].isin(selected)]
    index = {}
    for cid, tname, tid in trks[["caseid", "tname", "tid"]].itertuples(index=False):
        index.setdefault(int(cid), {})[tname] = tid
    rows_by_case = sp.set_index("caseid").to_dict("index")

    results = {}
    pending = []
    for c in todo:
        m = load_done(out_dir / f"{c}.npz")
        if m is not None:
            results[c] = m
        else:
            pending.append(c)
    print(f"{len(todo)} {label} cases selected: {len(results)} already complete, {len(pending)} to download")

    workers = max(1, min(args.workers, MAX_WORKERS))
    refused = None
    done_n = 0
    try:
        with cf.ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(process_case, c, http, index.get(c, {}), rows_by_case[c],
                              out_dir, tmp_root, not args.no_csv_check): c for c in pending}
            for f in cf.as_completed(futs):
                c = futs[f]
                done_n += 1
                try:
                    results[c] = f.result()
                    msg = "ok" + (f" [{';'.join(results[c]['flags'])}]" if results[c]["flags"] else "")
                except Refused as e:
                    refused = clean(str(e))
                    results[c] = {"caseid": c, "status": "failed", "flags": [], "error": refused}
                    msg = "REFUSED"
                    for g in futs:
                        g.cancel()
                except Exception as e:
                    results[c] = {"caseid": c, "status": "failed", "flags": [],
                                  "error": clean(f"{type(e).__name__}: {e}")}
                    msg = "failed: " + results[c]["error"]
                print(f"[{done_n}/{len(pending)}] case {c}: {msg}", flush=True)
    finally:
        # the manifest always covers every case of the cohort, whatever was selected
        for c in sorted(selected):
            if c not in results:
                m = load_done(out_dir / f"{c}.npz")
                results[c] = m if m is not None else {"caseid": c, "status": "not attempted",
                                                      "flags": []}
        write_manifest(results.values(), raw / f"{COHORT}_manifest.csv")
        shutil.rmtree(tmp_root, ignore_errors=True)

    ok = [r for r in results.values() if r["status"] == "ok"]
    n_sums = write_checksums(out_dir, [r["caseid"] for r in ok], raw / f"{COHORT}_checksums.csv")
    failed = {r["caseid"]: r.get("error", r["status"]) for r in results.values() if r["status"] != "ok"}
    both = sum(1 for r in ok if r.get("pleth_present") and r.get("ecg_present"))
    log = {
        "script_version": SCRIPT_VERSION,
        "cohort": label,
        "started_utc": started,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "run_seconds": round(time.time() - t_start, 1),
        "python": platform.python_version(),
        "packages": {"vitaldb": _version("vitaldb"), "numpy": np.__version__,
                     "pandas": pd.__version__, "requests": requests.__version__},
        "source": {"primary": f"{API}/{DATASET_VERSION}/<caseid>.vital",
                   "cross_check_original": f"{API}/<caseid>.vital",
                   "cross_check_csv": f"{API}/<tid> ({CSV_SHIFT_TRACK})"},
        "time_zero": "vital-file header dtstart (= case-table casestart 0)",
        "cohort_counts": counts,
        "selected_this_run": len(todo),
        "cohort_cases": len(selected),
        "complete": len(ok),
        "failed": failed,
        "both_waveforms_present": both,
        "refused": refused,
        "checksums": n_sums,
        "total_npz_bytes": int(sum(r.get("file_bytes", 0) or 0 for r in ok)),
    }
    (raw / f"{COHORT}_download_log.json").write_text(json.dumps(log, indent=2, default=_json_default))
    print(json.dumps({k: log[k] for k in ("complete", "failed", "both_waveforms_present",
                                          "total_npz_bytes", "run_seconds")}, default=_json_default))
    if refused:
        sys.exit(f"STOPPED: request refused ({refused}); refused requests are not retried.")


def _version(pkg):
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
