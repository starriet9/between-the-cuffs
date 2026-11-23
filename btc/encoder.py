"""Frozen pretrained PPG encoder (PaPaGei-S): window embeddings and per-forecast encoder features.

Exploratory check of SPEC.md §6.8, fixed by ENCODER_PLAN.md (§1 encoder, §2 input, §4 missing data).
Only the feature extractor differs from M4: the windows, their quality mask, the intervals and the
fallback are those of `btc.waveform`.

Encoder (ENCODER_PLAN.md §1)
----------------------------
PaPaGei-S (Pillai et al., ICLR 2025; arXiv:2410.20542), weights `papagei_s.pt` of Zenodo record
13983110, loaded into the authors' `ResNet1DMoE` (vendored, `btc/third_party/papagei_resnet.py`).
Size and MD5 are checked before loading; the state dict is read on CPU with `weights_only=True`
and loaded strictly; the model runs in `eval()` mode under `torch.inference_mode()`, in float32,
with one thread per process (`configure_torch`). The embedding is element 0 of the forward output,
the 512-d projected embedding. Each window is passed through the model on its own (`BATCH` = 1).
Nothing here downloads anything (`scripts/download_encoder.py` does, once); torch is imported only
by the functions that need it.

Per-window processing (ENCODER_PLAN.md §2)
-----------------------------------------
Each usable 10-s window (SPEC §5, `btc.waveform`) is processed from its own 5,000 samples at 500 Hz
only, `pleth.x[5000 w : 5000 (w + 1)]` as in `btc.waveform.case_windows`:
1. Chebyshev type II band-pass, order 4, 20 dB stop-band attenuation, 0.5-12 Hz, as second-order
   sections, run forward and backward (`scipy.signal.sosfiltfilt`); then a 50-ms moving average
   (25 samples) run forward and backward (`scipy.signal.filtfilt`). SciPy's default padding of both
   functions is written out below.
2. z-score: (x - mean) / (SD + 1e-7), population SD, at 500 Hz.
3. `scipy.signal.resample_poly(x, 1, 4)` to 125 Hz: 1,250 samples, float32.
A usable window has all 5,000 samples finite (`btc.beats.window_features` gives a window with a
missing sample no beats); a non-finite sample here is a bug and raises.

Per-forecast features (ENCODER_PLAN.md §2, §4) [the time rules are those of `btc.waveform`]
-------------------------------------------------------------------------------------------
Inputs are times only. With a = max(t_prev, t_issue - 120 s), b = min(t_issue, t_end):
- current = mean of the embeddings of the usable windows between a and b (t_start >= a and
  t_end_window <= b); `enc_available` = at least 2 such windows (otherwise M4-enc falls back to M3);
- previous = the same over max(t_prev2, t_prev - 120 s) to min(t_prev, t_end), missing unless at
  least 2 usable windows;
- reference = mean over the first 6 usable windows of the record, missing unless 6 exist and the 6th
  ends at or before b;
- `enc_lvl_<k>` = current - reference, `enc_chg_<k>` = current - previous (k = 000..511); both NaN
  when `enc_available` is False. The absolute embedding is not used.
Means are taken in float64 [judgment]. `enc_n_windows`, `enc_n_usable` and `enc_available` must equal
`wf_n_windows`, `wf_n_usable` and `wf_available` of the waveform table (`check_same_windows`).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import cheby2, filtfilt, resample_poly, sosfiltfilt

from . import io, paths, pqi, provenance
from . import waveform as W
from .constants import LOOKBACK_S, MIN_USABLE_WINDOWS

# --------------------------------------------------------------------------- encoder (§1)
ZENODO_RECORD = 13983110
ZENODO_DOI = "10.5281/zenodo.13983110"
WEIGHTS_NAME = "papagei_s.pt"
WEIGHTS_SIZE = 23_339_216
WEIGHTS_MD5 = "a4cdb32392e2a7b25999128af92813b5"
WEIGHTS = paths.DATA / "external" / "papagei" / WEIGHTS_NAME      # git-ignored, not redistributed
UPSTREAM_COMMIT = "0c537dad4d2850e15b724260de820dd68d77f0b0"        # Nokia-Bell-Labs/papagei-foundation-model
MODEL_KW = dict(in_channels=1, base_filters=32, kernel_size=3, stride=2, groups=1, n_block=18,
                n_classes=512, n_experts=3)
EMB_DIM = 512
OUTPUT_INDEX = 0                 # forward output element 0: the projected embedding (§1)
N_OUTPUTS = 4                    # ResNet1DMoE.forward returns (out_class, out_moe1, out_moe2, out)
# Windows per forward pass [judgment]: 1, in ascending (caseid, w) order. In batches of several windows
# the oneDNN kernels give results that differ by about 1e-7 with the batch size, which would make a
# window's embedding depend on how many usable windows share its batch (and so on later samples).
# One window per pass makes every embedding a function of that window's 5,000 samples only, bit for bit.
BATCH = 1

# --------------------------------------------------------------------------- input (§2)
FS_IN = pqi.FS_RAW               # 500 Hz
WIN_N_IN = pqi.WIN_N_RAW         # 5,000 samples
RESAMPLE_UP, RESAMPLE_DOWN = 1, 4
FS_OUT = FS_IN * RESAMPLE_UP / RESAMPLE_DOWN    # 125 Hz
WIN_N_OUT = WIN_N_IN * RESAMPLE_UP // RESAMPLE_DOWN   # 1,250 samples
BAND = (0.5, 12.0)
CHEBY_ORDER = 4
CHEBY_RS_DB = 20.0
SMOOTH_MS = 50.0
SMOOTH_N = int(round(FS_IN * SMOOTH_MS / 1000.0))   # 25 samples
Z_EPS = 1e-7
SOS = cheby2(CHEBY_ORDER, CHEBY_RS_DB, BAND, btype="bandpass", fs=FS_IN, output="sos")
MA = np.ones(SMOOTH_N) / SMOOTH_N
FILTER_NOTE = ("Filter parameters as in pyPPG's preproc.py (Chebyshev II, order 4, 20 dB, 0.5-12 Hz, filtfilt; then a "
               "50-ms moving average with filtfilt at fs >= 75 Hz), as ENCODER_PLAN.md §2 states them. Here the band-pass "
               "runs as second-order sections (sosfiltfilt) on each 10-s window, with SciPy's default padding, so the "
               "edge padding may differ from pyPPG's filtfilt on (b, a) coefficients [unverified].")

# --------------------------------------------------------------------------- features (§2)
N_REF = W.N_REF                  # first 6 usable windows (SPEC §4.3)
VARIANTS = ("real", "p1")        # ENCODER_PLAN.md §6: M4-enc and its placebo M4-enc-P1
ENC_LVL = tuple(f"enc_lvl_{k:03d}" for k in range(EMB_DIM))
ENC_CHG = tuple(f"enc_chg_{k:03d}" for k in range(EMB_DIM))
ENC_COLS = ENC_LVL + ENC_CHG      # 1,024 columns
KEY_COLS = ["caseid", "window", "issue_n", "variant", "signal_caseid", "t_issue", "t_prev", "t_end",
            "enc_n_windows", "enc_n_usable", "enc_available"]


# --------------------------------------------------------------------------- weights and model
def file_digests(path: Path) -> dict:
    data = Path(path).read_bytes()
    return {"size": len(data), "md5": hashlib.md5(data).hexdigest(), "sha256": hashlib.sha256(data).hexdigest()}


def check_weights(path: Path = WEIGHTS) -> dict:
    """Size and published MD5 of the weights file (ENCODER_PLAN.md §1); raises ValueError otherwise."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{p.name} not found; run scripts/download_encoder.py first")
    d = file_digests(p)
    if d["size"] != WEIGHTS_SIZE or d["md5"] != WEIGHTS_MD5:
        raise ValueError(f"{p.name}: size {d['size']} / md5 {d['md5']}, expected {WEIGHTS_SIZE} / {WEIGHTS_MD5}")
    return d


def configure_torch() -> dict:
    """One thread per process, deterministic algorithms (ENCODER_PLAN.md §1). Returns the settings."""
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:          # already set in this process (allowed once only)
        pass
    torch.use_deterministic_algorithms(True)
    return runtime_info()


def runtime_info() -> dict:
    import scipy
    import torch
    return {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
            "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
            "mkldnn_available": bool(torch.backends.mkldnn.is_available())}


def load_model(path: Path = WEIGHTS):
    """(model, info): the checked weights loaded strictly into `ResNet1DMoE(**MODEL_KW)`, in eval mode."""
    import torch
    from .third_party.papagei_resnet import ResNet1DMoE
    info = check_weights(path)
    sd = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise ValueError(f"{Path(path).name} is not a state dict")
    prefixed = sum(k.startswith("module.") for k in sd)
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    dtypes = sorted({str(v.dtype) for v in sd.values()})
    model = ResNet1DMoE(**MODEL_KW)
    res = model.load_state_dict(sd, strict=True)
    if res.missing_keys or res.unexpected_keys:          # strict=True raises first; kept as a record
        raise ValueError(f"missing {res.missing_keys[:3]}, unexpected {res.unexpected_keys[:3]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("model parameters are not float32")
    info.update(n_tensors=len(sd), n_keys_with_module_prefix=int(prefixed), dtypes=dtypes,
                n_parameters=int(sum(p.numel() for p in model.parameters())), strict=True)
    return model, info


def embed(model, X: np.ndarray, batch: int = BATCH) -> np.ndarray:
    """Embeddings (N, 512) float32 of processed windows X (N, 1250), in the given order and batches."""
    import torch
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != WIN_N_OUT:
        raise ValueError(f"input must be (N, {WIN_N_OUT}), got {X.shape}")
    if model.training:
        raise RuntimeError("the model must be in eval mode")
    out = np.empty((X.shape[0], EMB_DIM), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[i:i + batch, None, :])
            res = model(xb)
            if len(res) != N_OUTPUTS or tuple(res[0].shape) != (xb.shape[0], EMB_DIM) \
                    or tuple(res[3].shape) != (xb.shape[0], EMB_DIM):
                raise RuntimeError(f"unexpected forward output: {[tuple(r.shape) for r in res]}")
            out[i:i + batch] = res[OUTPUT_INDEX].numpy()
    if not np.all(np.isfinite(out)):
        raise RuntimeError("non-finite embedding")
    return out


# --------------------------------------------------------------------------- per-window processing
def bandpass(x: np.ndarray) -> np.ndarray:
    return sosfiltfilt(SOS, x, axis=-1, padtype="odd", padlen=None)        # SciPy defaults, explicit


def smooth(x: np.ndarray) -> np.ndarray:
    return filtfilt(MA, [1.0], x, axis=-1, padtype="odd", padlen=None, method="pad")   # SciPy defaults


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + Z_EPS)          # population SD (ddof 0)


def resample(x: np.ndarray) -> np.ndarray:
    return resample_poly(x, RESAMPLE_UP, RESAMPLE_DOWN, axis=-1, window=("kaiser", 5.0),
                         padtype="constant", cval=None)                    # SciPy defaults, explicit


def preprocess_window(x_raw: np.ndarray) -> np.ndarray:
    """One window's 5,000 raw 500 Hz samples -> the encoder input (1,250,) float32 (§2, in this order)."""
    x = np.asarray(x_raw, dtype=np.float64)
    if x.shape != (WIN_N_IN,):
        raise ValueError(f"a window has {WIN_N_IN} samples, got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("a usable window has no missing sample")
    r = resample(zscore(smooth(bandpass(x))))
    if r.shape != (WIN_N_OUT,) or not np.all(np.isfinite(r)):
        raise RuntimeError("unexpected processed window")
    return r.astype(np.float32)


def processed_dom_freq(r: np.ndarray) -> float:
    """Dominant frequency of a processed window, with the spectrum of the quality index (SPEC §5: Hann
    periodogram, 0.1 Hz bins, largest bin in 0.5-8 Hz). For the signal-level sanity check [judgment]."""
    return float(pqi.assess_window(np.asarray(r, dtype=np.float64))["dom_freq"])


def case_inputs(caseid: int, pleth, win_case: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(w, X): the usable windows of one case in ascending w, processed (n, 1250) float32.

    The window grid of `pleth` must be that of the window table (same count and start times)."""
    win_case = win_case.sort_values("w")
    nw = pqi.n_windows(pleth.x.size)
    starts, _ = pqi.window_times(pleth.t0, nw, pleth.srate)
    if len(win_case) != nw or not np.array_equal(win_case["w"].to_numpy(), np.arange(nw)) \
            or not np.allclose(win_case["t_start"].to_numpy(float), starts, rtol=0, atol=1e-9):
        raise ValueError(f"case {caseid}: pleth window grid differs from the window table")
    ws = win_case.loc[win_case["usable"].astype(bool), "w"].to_numpy(np.int64)
    X = np.empty((len(ws), WIN_N_OUT), dtype=np.float32)
    for i, w in enumerate(ws):
        X[i] = preprocess_window(pleth.x[w * WIN_N_IN:(w + 1) * WIN_N_IN])
    return ws, X


def embed_case(caseid: int, pleth, win_case: pd.DataFrame, embed_fn) -> dict:
    """Embeddings of one case's usable windows. `embed_fn`: (n, 1250) float32 -> (n, d) array."""
    ws, X = case_inputs(caseid, pleth, win_case)
    E = embed_fn(X) if len(ws) else np.zeros((0, EMB_DIM), dtype=np.float32)
    dom = win_case.set_index("w").loc[ws, "dom_freq"].to_numpy(float) if len(ws) else np.zeros(0)
    pdom = np.array([processed_dom_freq(x) for x in X])
    return {"caseid": int(caseid), "w": ws, "E": np.asarray(E), "dom_freq": dom, "processed_dom_freq": pdom}


# --------------------------------------------------------------------------- per forecast
class EncWindows:
    """One case's windows (all of them, for the counts) and the embeddings of its usable windows."""

    def __init__(self, win_case: pd.DataFrame | None, E_usable: np.ndarray | None = None, dim: int = EMB_DIM):
        df = win_case if win_case is not None else pd.DataFrame(columns=["w", "t_start", "t_end", "usable"])
        df = df.sort_values("w")
        self.dim = dim
        self.ts = df["t_start"].to_numpy(np.float64)
        self.te = df["t_end"].to_numpy(np.float64)
        self.usable = df["usable"].to_numpy(bool)
        nu = int(self.usable.sum())
        E_usable = np.zeros((0, dim)) if E_usable is None else np.asarray(E_usable)
        if E_usable.shape != (nu, dim):
            raise ValueError(f"{nu} usable windows but embeddings of shape {E_usable.shape}")
        self.E = np.full((len(df), dim), np.nan)
        self.E[self.usable] = E_usable.astype(np.float64)

    def aggregate(self, a: float, b: float):
        """(mean embedding or NaN, n windows, n usable) over the windows between a and b."""
        sel = (self.ts >= a) & (self.te <= b)
        use = sel & self.usable
        n_use = int(use.sum())
        mean = self.E[use].mean(axis=0) if n_use >= MIN_USABLE_WINDOWS else np.full(self.dim, np.nan)
        return mean, int(sel.sum()), n_use

    def reference(self, t_cut: float) -> np.ndarray:
        idx = np.flatnonzero(self.usable)[:N_REF]
        if idx.size < N_REF or self.te[idx[-1]] > t_cut:
            return np.full(self.dim, np.nan)
        return self.E[idx].mean(axis=0)


def forecast_embedding_features(ew: EncWindows, t_issue: float, t_prev: float, t_prev2: float,
                                t_end: float) -> dict:
    """Encoder features of one forecast (see module docstring). Inputs are times only."""
    b = min(t_issue, t_end)
    a = max(t_prev, t_issue - LOOKBACK_S)
    cur, n_win, n_use = ew.aggregate(a, b)
    prv, _, _ = ew.aggregate(max(t_prev2, t_prev - LOOKBACK_S), min(t_prev, t_end))
    ref = ew.reference(b)
    available = n_use >= MIN_USABLE_WINDOWS
    nan = np.full(ew.dim, np.nan)
    return {"enc_n_windows": n_win, "enc_n_usable": n_use, "enc_available": bool(available),
            "lvl": cur - ref if available else nan, "chg": cur - prv if available else nan}


def case_enc_windows(win: pd.DataFrame, index: pd.DataFrame, E: np.ndarray) -> dict:
    """caseid -> EncWindows, from the window table, the embedding index (caseid, w, row) and the matrix.
    The index must hold exactly the usable windows of the window table."""
    u = win.loc[win["usable"].astype(bool), ["caseid", "w"]].sort_values(["caseid", "w"]).reset_index(drop=True)
    ix = index.sort_values(["caseid", "w"]).reset_index(drop=True)
    if not (u["caseid"].astype(int).equals(ix["caseid"].astype(int)) and u["w"].astype(int).equals(ix["w"].astype(int))):
        raise ValueError("the embedding index does not hold exactly the usable windows of the window table")
    rows = ix["row"].to_numpy(np.int64)
    out = {}
    for c, g in win.groupby("caseid", sort=True):
        r = rows[(ix["caseid"] == c).to_numpy()]
        out[int(c)] = EncWindows(g, E[r], E.shape[1])
    return out


def build_features(issues: pd.DataFrame, t_of: dict, cases: pd.DataFrame, enc_windows: dict,
                   variants=VARIANTS, window: str = "primary", dim: int = EMB_DIM) -> pd.DataFrame:
    """Per-forecast encoder table, one row per (caseid, window, issue_n, variant) of the issue points of
    `window` (mirrors `scripts/build_features_wf.py::build`, variants real and p1 only)."""
    empty = EncWindows(None, None, dim)
    donor = dict(zip(cases["caseid"].astype(int), cases["donor_caseid"]))
    iss = issues.loc[issues["window"] == window]
    keys, lvl, chg = [], [], []
    for r in iss.itertuples(index=False):
        c, n = int(r.caseid), int(r.issue_n)
        tp = t_of.get((c, n - 1))
        if tp is None or abs(tp - r.t_prev) > 1e-6:
            raise ValueError(f"case {c} issue {n}: t_prev does not match reading {n - 1}")
        t_pp = t_of[(c, n - 2)]
        for v in variants:
            src = c
            if v == "p1":
                d = donor.get(c, np.nan)
                src = int(d) if pd.notna(d) else None
            ti, tpv, tpp, te, mask = W.variant_times(v, float(r.t_issue), float(r.t_prev), t_pp)
            if mask != "usable":
                raise ValueError(f"variant {v} is not part of the encoder check")
            ew = enc_windows.get(src, empty) if src is not None else empty
            f = forecast_embedding_features(ew, ti, tpv, tpp, te)
            keys.append({"caseid": c, "window": r.window, "issue_n": n, "variant": v,
                         "signal_caseid": float(src) if src is not None else np.nan,
                         "t_issue": float(r.t_issue), "t_prev": float(r.t_prev), "t_end": te,
                         "enc_n_windows": f["enc_n_windows"], "enc_n_usable": f["enc_n_usable"],
                         "enc_available": f["enc_available"]})
            lvl.append(f["lvl"])
            chg.append(f["chg"])
    k = pd.DataFrame(keys, columns=KEY_COLS)
    names = (ENC_LVL + ENC_CHG) if dim == EMB_DIM else \
        tuple(f"enc_lvl_{j:03d}" for j in range(dim)) + tuple(f"enc_chg_{j:03d}" for j in range(dim))
    M = np.hstack([np.vstack(lvl), np.vstack(chg)]) if keys else np.zeros((0, 2 * dim))
    out = pd.concat([k, pd.DataFrame(M, columns=list(names))], axis=1)
    out["enc_available"] = out["enc_available"].astype(bool)
    return out


def check_same_windows(enc: pd.DataFrame, wf: pd.DataFrame) -> dict:
    """The hard check of "only the extractor changes": on every (caseid, window, issue_n, variant) row of
    the encoder table, enc_n_windows / enc_n_usable / enc_available equal wf_n_windows / wf_n_usable /
    wf_available of the waveform table. Raises otherwise; returns the counts compared."""
    key = ["caseid", "window", "issue_n", "variant"]
    m = enc[key + ["enc_n_windows", "enc_n_usable", "enc_available"]].merge(
        wf[key + ["wf_n_windows", "wf_n_usable", "wf_available"]], on=key, how="left", validate="1:1",
        indicator=True)
    if (m["_merge"] != "both").any():
        raise ValueError(f"{int((m['_merge'] != 'both').sum())} encoder rows have no waveform row")
    bad = (m["enc_n_windows"].astype(int) != m["wf_n_windows"].astype(int)) | \
          (m["enc_n_usable"].astype(int) != m["wf_n_usable"].astype(int)) | \
          (m["enc_available"].astype(bool) != m["wf_available"].astype(bool))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows: encoder windows differ from M4's, e.g. "
                         f"{m.loc[bad, key].head(3).to_dict('records')}")
    return {"rows_compared": int(len(m)), "rows_available": int(m["enc_available"].sum())}


# --------------------------------------------------------------------------- run guards (§11)
PLAN_FILE = "ENCODER_PLAN.md"


def plan_record() -> dict:
    """SHA-256 of ENCODER_PLAN.md, whether the file equals the one committed at HEAD, and the last commit
    that changed it (ENCODER_PLAN.md §11: its commit precedes every encoder output)."""
    p = paths.ROOT / PLAN_FILE
    sha = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    head = provenance._git("rev-parse", "--verify", "--quiet", f"HEAD:{PLAN_FILE}")
    here = provenance._git("hash-object", "--", PLAN_FILE) if p.is_file() else None
    last = provenance._git("log", "-1", "--format=%H", "--", PLAN_FILE)
    return {"file": PLAN_FILE, "sha256": sha, "committed_at_head": bool(head and here and head == here),
            "last_commit": last or None}


def start_problems(g: dict, plan: dict) -> list:
    """Why a script that builds or uses encoder outputs on real data must not start: a work tree that is
    not clean (`btc.provenance`: untracked files count), ENCODER_PLAN.md not committed at HEAD, or the
    S-ob switch set (no S-ob data in the S-dev steps)."""
    probs = []
    if g.get("state") != "clean":
        probs.append(f"the work tree is not clean (git state {g.get('state')}; untracked files count)")
    if not plan.get("committed_at_head"):
        probs.append(f"{PLAN_FILE} is not committed at HEAD, or differs from the committed file")
    if os.environ.get(io.FROZEN_SCORING_ENV) is not None:
        probs.append(f"{io.FROZEN_SCORING_ENV} is set; the S-dev encoder steps never read S-ob data")
    return probs
