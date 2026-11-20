"""Registered constants. Every value cites SPEC.md / PREREG.md v1.5.

Values shared with the feature code (seed, window end, h-bin edges, cycling cut, P2 shift,
inflation-artefact exclusion, usable-window minimum) come from `btc.constants`, the one source.
"""
from __future__ import annotations

import numpy as np

from ..constants import (FAST_CYCLING_S, H_BIN_EDGES_S, IA_EXCLUDE_S, MIN_USABLE_WINDOWS,  # noqa: F401
                         P2_SHIFT_S, PRIMARY_END_S, SEED)

# Seeds (SPEC §4.1, §4.5, §6.2, §10; PREREG §4, §6): SEED = 20261118 (btc.constants)
FOLD_SEEDS_SENSITIVITY = tuple(range(20261119, 20261129))   # SPEC §6.7, §10 (ten seeds)

# Folds (SPEC §4.5)
N_SPLITS = 5

# Ridge penalty grid: 10^-3 ... 10^3, 13 values (SPEC §4.1; PREREG §4)
LAMBDA_GRID = np.logspace(-3.0, 3.0, 13)
DROPPED = None          # the "dropped" option of an added block (SPEC §4.1, §4.5)
TIE_RTOL = 1e-12        # [judgment] criterion values equal within this relative tolerance are ties

# Horizon bins in seconds (SPEC §3, §4.5): H_BIN_EDGES_S (btc.constants); lower edge in, upper edge
# out, last bin includes 400 s
N_BINS = 4
PERBIN_MIN_TARGETS = 300    # SPEC §4.1: a bin with < 300 training targets in all of S-dev uses the pooled model

# Cycling class (SPEC §4.5, §6.7): FAST_CYCLING_S = 90 s (btc.constants)

# Metrics (SPEC §6.1; PREREG §6, §7)
MIN_FORECASTS_RMSE = 3      # RMSE, MAE, mean signed error
MIN_FORECASTS_R2 = 5        # R^2 and FIT (and SST > 0)
MIN_TRIPLES = 3             # SPEC §3 yardstick: a patient enters with >= 3 scored triples

# Bootstrap (SPEC §6.2; PREREG §6)
N_BOOT = 2000
CI_PERCENTILES = (2.5, 97.5)

# Decision rule (SPEC §6.3; PREREG §6)
RULE_MIN_RELATIVE_REDUCTION = 0.05

# Coverage rule (SPEC §6.5; PREREG §6)
COVERAGE_MIN = 0.50

# Controls (SPEC §6.6)
P2_MIN_ISSUE_S = 720.0      # P2 is scored on forecasts issued after 12 min
QUALITY_SHARE_CUT = 0.80    # quality strata (SPEC §6.6 item 5; the 80% cut is a registered [judgment])
# MIN_USABLE_WINDOWS = 2 (SPEC §4.3, §6.6), P2_SHIFT_S = 600 s, IA_EXCLUDE_S = 30 s (btc.constants)

# Secondary analyses (SPEC §6.7)
LARGE_CHANGE_MMHG = 10.0
CONFORMAL_LEVEL = 0.90
HYPO_REL_BASELINE = 0.80
HYPO_MAP_MMHG = 65.0
HYPO_SBP_MMHG = 100.0
MINUTES_PER_ALARM_RATE = 30.0

# Gradient-boosting check (SPEC §6.8; PREREG §7)
GB_LEARNING_RATES = (0.03, 0.1)
GB_MAX_LEAF_NODES = (3, 7, 15)
GB_MAX_ITER = (30, 100, 300)

# Windows (SPEC §2.5): PRIMARY_END_S = 1800 s (btc.constants)

# Feature names (btc/SCHEMA.md §3-§4; SPEC §4.1-§4.3). The waveform features equal
# `btc.beats.FEATURES`; the dicrotic-notch quantities are not model inputs (SPEC §4.3).
HR_COLS = ("hr_level", "hr_change", "hr_slope")
WF_FEATURES = ("pat", "rise", "width", "bav")
WF_COLS = tuple(f"wf_{f}_{form}" for f in WF_FEATURES for form in ("lvl", "chg"))

# Feature-group ablation (SPEC §6.6 item 4)
ABLATION_GROUPS = {
    "hr": HR_COLS,
    "pat": ("wf_pat_lvl", "wf_pat_chg"),
    "shape": tuple(f"wf_{f}_{form}" for f in ("rise", "width") for form in ("lvl", "chg")),
    "bav": ("wf_bav_lvl", "wf_bav_chg"),
    "avail": (),            # availability indicators only (built in data.py)
}

# Signal-feature variants (btc/SCHEMA.md §4)
VARIANTS_HR = ("real", "p1", "p2", "ia")
VARIANTS_WF = ("real", "p1", "p2", "ia", "strict")

# Plain question (SPEC §6.3; PREREG §6)
QUESTION = ("After spinal anesthesia, do heart rate and the pulse-oximeter waveform improve the "
            "forecast of the next cuff reading (MAP) beyond the cuff readings alone?")
