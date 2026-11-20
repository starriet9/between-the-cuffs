"""Registered constants shared by the feature code (`btc/`) and the model code (`btc/models/`).

One source, so that the two sides cannot disagree. Each value cites SPEC.md / PREREG.md v1.5.
"""
from __future__ import annotations

SEED = 20261118                 # SPEC §10: folds, bootstrap, placebo donors, gradient boosting

# Windows and targets (SPEC §2.5, §3)
PRIMARY_END_S = 1800.0          # primary window 0-30 min, both ends included
SECONDARY_CAP_S = 3600.0        # secondary window: 0 to incision, capped at 60 min
MAX_H_S = 400.0                 # targets within 400 s of the issue time (inclusive)
H_BIN_EDGES_S = (15.0, 90.0, 150.0, 210.0, 400.0)   # §3, §4.5: lower edge in, upper out, last bin includes 400 s
FAST_CYCLING_S = 90.0           # SPEC §6.7: fast cycling = primary-window median interval < 90 s

# Signal-feature variants (SPEC §3, §6.6)
P2_SHIFT_S = 600.0              # P2: reading and issue times shifted back by 600 s
IA_EXCLUDE_S = 30.0             # inflation-artefact variant: the 30 s before the issue time excluded

# Waveform features (SPEC §4.3)
LOOKBACK_S = 120.0              # windows between the previous reading (at most 120 s back) and t_r
MIN_USABLE_WINDOWS = 2          # at least 2 usable windows, else M4 falls back to M3
