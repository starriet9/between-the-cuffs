"""Model fitting and evaluation (SPEC.md §4.1, §4.5, §6; PREREG.md §4, §6, §7).

Modules
-------
constants  registered constants, each with its SPEC/PREREG section
folds      outer and inner StratifiedGroupKFold folds (§4.5)
data       loading the derived tables (btc/SCHEMA.md) and building analysis frames
design     design matrices: imputation, indicators, interactions, scaling (§4.5)
ridge      block-penalised ridge, penalty selection and the model ladder (§4.1, §4.5)
metrics    per-patient metrics (§6.1)
bootstrap  paired subject-level bootstrap (§6.2, §3 yardstick)
headline   decision rule, placebo validity, headline rows and sentences (§6.3)
analyses   the registered model chains and their cross-validated runs
secondary  secondary and control analyses (§6.6, §6.7)
gbm        gradient-boosting check (§6.8)
pipeline   confirmatory S-dev run, freeze, frozen S-ob scoring (§9)
synthetic  synthetic tables in the btc/SCHEMA.md format, for tests only
"""

# The ridge systems are tiny (tens of columns) and the boosting data small: multithreaded BLAS and
# OpenMP only add contention, and the parallel work is spread over processes (joblib). Limit the
# libraries loaded so far (BLAS) to one thread; the boosting code limits OpenMP where it runs.
from threadpoolctl import threadpool_limits as _threadpool_limits

_threadpool_limits(1)
