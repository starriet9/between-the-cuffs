"""Featured replays: the selection rules fixed in SPEC §8.2 / PREREG §8.

1. Median case: among cases with >= 1 new-onset reading below 80% of the record-start baseline,
   the case with the median patient-level M4 RMSE; with an even count, the higher-RMSE case of the
   two middle cases. This case is the README GIF and the summary figure's left panel.
2. Worst case: the case with the highest patient-level M4 RMSE.
3. Pump case: the pump case with the most PHEN_RATE changes in the primary window (no forecast used).

Rules 1 and 2 consider only cases with >= 8 scored primary forecasts. Ties go to the lowest case ID.
If one case meets two rules, the later rule takes its next case.
"""
from __future__ import annotations

MIN_FORECASTS_FEATURED = 8
RULE_TAGS = {1: "median", 2: "highest error", 3: "pump"}


def select_featured(cases: list) -> dict:
    """{1: caseid | None, 2: ..., 3: ...} from the scored CaseSummary entries."""
    scored = [c for c in cases if c.scored]
    taken = set()
    out = {}
    elig = [c for c in scored if c.n_forecasts >= MIN_FORECASTS_FEATURED]
    # rule 1: ascending M4 RMSE; index len//2 is the middle (odd) or the higher middle case (even);
    # a tie at that RMSE goes to the lowest case ID
    r1 = sorted((c for c in elig if c.new_onset_80 >= 1), key=lambda c: (c.rmse["M4"], c.caseid))
    if r1:
        v = r1[len(r1) // 2].rmse["M4"]
        out[1] = min(c.caseid for c in r1 if c.rmse["M4"] == v)
        taken.add(out[1])
    else:
        out[1] = None
    # rule 2: descending M4 RMSE, ties to the lowest case ID; skip a case already taken
    r2 = [c for c in sorted(elig, key=lambda c: (-c.rmse["M4"], c.caseid)) if c.caseid not in taken]
    out[2] = r2[0].caseid if r2 else None
    if out[2] is not None:
        taken.add(out[2])
    # rule 3: pump cases by the number of rate changes, ties to the lowest case ID
    r3 = [c for c in sorted((c for c in scored if c.pump_track), key=lambda c: (-c.pump_changes, c.caseid))
          if c.caseid not in taken]
    out[3] = r3[0].caseid if r3 else None
    return out


def rule_of(featured: dict) -> dict:
    """{caseid: rule number} for the featured cases."""
    return {cid: r for r, cid in featured.items() if cid is not None}
