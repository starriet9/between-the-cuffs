#!/usr/bin/env python3
"""Build the cuff-side and heart-rate tables of one cohort (btc/SCHEMA.md §1-§4).

Reads only the cohort's case archives (`data/raw/<cohort>/`), the case table
(`data/cohorts.csv`, `data/raw/meta/cases.csv`) and the cohort's download manifest.
Fits no model and computes no forecast error. Writes to `data/derived/`:

  <cohort>_cases.parquet          one row per case (§2.2, §2.3, §2.4, §2.6, §6.6), with `analysable`,
                                  `nmb_recorded` and `analysed` (§2.2)
  <cohort>_readings.parquet       one row per merged cuff reading (§2.4)
  <cohort>_instances.parquet      issue points x targets with cuff features (§3, §4.1), all analysable cases
  <cohort>_features_hr.parquet    HR features, variants real / p1 / p2 / ia (§4.2), all analysable cases
  <cohort>_cohort_flow.json/.md   cohort-flow counts under the final cuff rule (§2.2)

S-ob (`--cohort sob`) runs only at the frozen scoring, with BTC_FROZEN_SCORING=1 (SPEC §9).
The git state is recorded in every output; a dirty or unknown state needs --allow-dirty.

Usage: python scripts/build_instances.py [--cohort sdev|sob] [--allow-dirty]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc import hr, instances as ins, io, paths, provenance  # noqa: E402

SCRIPT = "scripts/build_instances.py"
# Sedation totals counted in the cohort flow only (SPEC §2.2: sedation is not an exclusion)
SEDATION_COLUMNS = ("intraop_mdz", "intraop_ppf")


def check_sdev_expectations(sdev: pd.DataFrame, cases: pd.DataFrame | None = None) -> None:
    """S-dev counts and §2.3 outcomes stated in SPEC §2.2-§2.3 (rules applied generally, checked here)."""
    assert len(sdev) == 252 and sdev["subjectid"].nunique() == 251, "S-dev counts differ from SPEC §2.2"
    assert int(sdev["cerclage"].sum()) == 6, "SPEC §2.2 expects six cerclage cases"
    if cases is None:
        return
    st = cases.set_index("caseid")
    assert st.loc[2328, "pleth_status"] == io.ABSENT_INTEGRITY and st.loc[2328, "ecg_status"] == io.ABSENT_INTEGRITY
    assert st.loc[2030, "ecg_status"] == io.ABSENT_IN_FILE and st.loc[2030, "pleth_status"] == io.OK
    assert not st.loc[2328, "analysable"] and not st.loc[2030, "analysable"]


def build(cohort: str, state: dict) -> dict:
    label = paths.COHORT_LABEL[cohort]
    cohorts = io.load_cohorts()
    rows = cohorts.loc[cohorts["cohort"] == label].set_index("caseid")
    sob_subjects = set(cohorts.loc[cohorts["cohort"] == "S-ob", "subjectid"])   # case table only
    if cohort == "sdev":
        check_sdev_expectations(rows)
    demo = pd.read_csv(paths.CASE_TABLE, usecols=["caseid", "sex", "age", *ins.NMB_COLUMNS, *SEDATION_COLUMNS]
                       ).set_index("caseid")

    case_rows, readings_all, hr_tracks = [], [], {}
    for cid in io.caseids(cohort):
        case = io.load_case(cid, cohort)
        readings, med = ins.case_readings(case)
        # §2.2: a subject with cases in both cohorts loses its S-dev case; S-ob cases are kept
        in_sob = cohort == "sdev" and int(rows.loc[cid, "subjectid"]) in sob_subjects
        row = ins.case_row(case, rows.loc[cid], readings, med, subject_in_sob=in_sob)
        row["sex"] = demo.loc[cid, "sex"]
        row["age"], row["age_censored"] = ins.parse_age(demo.loc[cid, "age"])
        row.update(ins.nmb_columns(demo.loc[cid], row["analysable"]))   # §2.2: analysed = analysable, no NMB
        if cohort == "sob":
            row.update(ins.vaso_columns(case, rows.loc[cid]))      # §2.6 strata (descriptive)
        case_rows.append(row)
        readings.insert(0, "caseid", cid)
        readings_all.append(readings)
        hr_tracks[cid] = (case.hr.t, case.hr.v)
    cases = pd.DataFrame(case_rows)
    readings = pd.concat(readings_all, ignore_index=True)
    if cohort == "sdev":
        check_sdev_expectations(rows, cases)

    cases["donor_caseid"] = cases["caseid"].map(ins.assign_donors(cases))

    # instances (§3), all analysable cases (with or without a recorded neuromuscular blocker, §2.2)
    inst = []
    for r in cases.loc[cases["analysable"]].itertuples(index=False):
        rd = readings.loc[readings["caseid"] == r.caseid]
        inst.append(ins.case_instances(r.caseid, r.subjectid, r.fast_cycling, rd, r.opstart))
    instances = pd.concat(inst, ignore_index=True)

    # HR features (§4.2) for every issue point in the instances, all variants
    issues = instances.drop_duplicates(["caseid", "window", "issue_n"])[
        ["caseid", "window", "issue_n", "t_issue", "t_prev"]]
    donors = cases.set_index("caseid")["donor_caseid"]
    feats = []
    for cid, iss in issues.groupby("caseid", sort=True):
        d = donors.loc[cid]
        dtrk = hr_tracks[int(d)] if np.isfinite(d) else None
        feats.append(hr.case_hr_features(iss, hr_tracks[cid], dtrk, int(d) if np.isfinite(d) else None))
    hr_feat = pd.concat(feats, ignore_index=True)
    return {"cases": cases, "readings": readings, "instances": instances, "features_hr": hr_feat,
            "hr_tracks": hr_tracks, "demo": demo}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", choices=paths.COHORTS_KEYS, default="sdev")
    provenance.add_allow_dirty(ap)
    a = ap.parse_args(argv)
    io.check_cohort_access(a.cohort)
    state = provenance.require_clean(a.allow_dirty)
    paths.DERIVED.mkdir(parents=True, exist_ok=True)
    t = build(a.cohort, state)
    label = paths.COHORT_LABEL[a.cohort]
    meta = {"git": state, "spec": provenance.SPEC_VERSION, "cohort": label, "script": SCRIPT}
    for name in ("cases", "readings", "instances", "features_hr"):
        provenance.write_parquet(t[name], paths.derived(a.cohort, f"{name}.parquet"), meta)

    flow = cohort_flow(t["cases"], t["readings"], t["instances"], t["features_hr"], label, t["hr_tracks"],
                       demo=t.get("demo"))
    flow["git"] = state
    flow["script"] = SCRIPT
    provenance.write_json(paths.derived(a.cohort, "cohort_flow.json"), flow)
    md = flow_markdown(flow)
    paths.derived(a.cohort, "cohort_flow.md").write_text(provenance.with_credit(md))
    print(md)


# ---------------------------------------------------------------------------
# Cohort flow (§2.2: "Both cohorts are reported with a cohort-flow diagram")
# ---------------------------------------------------------------------------

def hr_zero_counts(hr_tracks: dict, caseids) -> dict:
    """Monitor HR samples equal to 0 in the primary window (treated as missing, `btc.hr.clean_hr`)."""
    n = nz = k = 0
    for c in caseids:
        t, v = hr_tracks[c]
        m = (t >= 0) & (t <= ins.PRIMARY_END_S) & np.isfinite(v)
        z = int((v[m] == 0).sum())
        n, nz, k = n + int(m.sum()), nz + z, k + (z > 0)
    return {"samples": n, "zero": nz, "share_zero": nz / n if n else float("nan"), "cases_with_zero": k}


def cohort_flow(cases, readings, instances, hr_feat, label: str = "S-dev", hr_tracks=None, demo=None) -> dict:
    a = cases.loc[cases["analysable"]]
    ana_ids = set(a["caseid"])
    an = a.loc[a["analysed"]]                  # §2.2: analysable and no neuromuscular blocker recorded
    nmb = a.loc[a["nmb_recorded"]]
    k1_an = instances.loc[(instances["window"] == "primary") & (instances["k"] == 1)
                          & instances["caseid"].isin(set(an["caseid"]))]
    reasons_first = {}
    for r in cases.loc[~cases["analysable"], "not_analysable_reason"]:
        k = r.split(";")[0]
        reasons_first[k] = reasons_first.get(k, 0) + 1
    reasons_any = {}
    for r in cases.loc[~cases["analysable"], "not_analysable_reason"]:
        for k in r.split(";"):
            reasons_any[k] = reasons_any.get(k, 0) + 1
    not_ana = cases.loc[~cases["analysable"], ["caseid", "not_analysable_reason", "n_valid_primary",
                                                  "pleth_status", "ecg_status"]]

    def reading_counts(rd):
        return {"readings": int(len(rd)), "valid": int(rd["valid"].sum()),
                "dropped": int((~rd["valid"]).sum()),
                "dropped_by_first_rule": {k: int((rd["drop_reason"] == k).sum())
                                          for k in ("incomplete", "map_range", "order")},
                "failing_each_rule": {"incomplete": int(rd["fail_incomplete"].sum()),
                                      "map_range": int(rd["fail_map_range"].sum()),
                                      "order": int(rd["fail_order"].sum())},
                "gaps_flagged": int(rd["gap_flag"].sum()),
                "change_spread_le_0p1_s": int((rd["spread_s"] <= 0.1).sum()),
                "change_spread_max_s": float(rd["spread_s"].max()) if len(rd) else float("nan")}

    in_p = (readings["t"] >= 0) & (readings["t"] <= ins.PRIMARY_END_S)
    rd_ana = readings["caseid"].isin(ana_ids)

    def target_counts(df):
        out = {"targets": int(len(df)), "cases": int(df["caseid"].nunique()),
               "subjects": int(df["subjectid"].nunique()), "by_h_bin": {}}
        labels = {1: "15-90 s", 2: "90-150 s", 3: "150-210 s", 4: "210-400 s"}
        for b, label in labels.items():
            d = df.loc[df["h_bin"] == b]
            out["by_h_bin"][label] = {"targets": int(len(d)), "cases": int(d["caseid"].nunique()),
                                    "subjects": int(d["subjectid"].nunique()),
                                    "ge_300_targets": bool(len(d) >= 300)}
        return out

    flow = {}
    for window in ins.WINDOWS:
        w = instances.loc[instances["window"] == window]
        k1 = w.loc[w["k"] == 1]
        per_case_k1 = k1.groupby("caseid").size()
        k2 = w.loc[w["k"] == 2]
        per_case_k2 = k2.groupby("caseid").size()
        fast_ids = set(a.loc[a["fast_cycling"], "caseid"])
        flow[window] = {
            "issue_points_with_target": int(len(k1)),
            "cases_with_issue_points": int(k1["caseid"].nunique()),
            "targets_by_k": {int(k): int((w["k"] == k).sum()) for k in (1, 2, 3)},
            "primary_targets_next_reading": target_counts(k1),
            "cases_ge3_next_reading_targets": int((per_case_k1 >= 3).sum()),
            "cases_ge5_next_reading_targets": int((per_case_k1 >= 5).sum()),
            "fast_cycling_cases_ge3_next_reading_targets": int(
                (per_case_k1[per_case_k1.index.isin(fast_ids)] >= 3).sum()),
            "yardstick_triples": int(len(k2)),
            "cases_ge3_triples": int((per_case_k2 >= 3).sum()),
            "fast_cycling_cases_ge3_triples": int(
                (per_case_k2[per_case_k2.index.isin(fast_ids)] >= 3).sum()),
        }
    hp = hr_feat.loc[hr_feat["window"] == "primary"]
    hr_missing = {v: {"issue_points": int((hp["variant"] == v).sum()),
                      "hr_group_missing": int(hp.loc[hp["variant"] == v, "hr_level"].isna().sum()),
                      "hr_change_missing": int(hp.loc[hp["variant"] == v, "hr_change"].isna().sum())}
                  for v in hr.VARIANTS}
    flow_out = {
        "cohort": label,
        "rule": "final cuff rule (SPEC §2.4); signal integrity (SPEC §2.3); analysable (SPEC §2.2); "
                "analysed = analysable without a recorded neuromuscular blocker (SPEC §2.2)",
        "cases": int(len(cases)), "subjects": int(cases["subjectid"].nunique()),
        "cerclage_cases": int(cases["cerclage"].sum()),
        "signal_integrity": {
            "pleth": cases["pleth_status"].value_counts().to_dict(),
            "ecg": cases["ecg_status"].value_counts().to_dict(),
            "flagged_cases": cases.loc[cases["waveform_flag"] != "", ["caseid", "waveform_flag"]]
                                  .to_dict(orient="records"),
        },
        "not_analysable": int((~cases["analysable"]).sum()),
        "not_analysable_by_first_reason": reasons_first,
        "not_analysable_any_reason": reasons_any,
        "not_analysable_cases": not_ana.to_dict(orient="records"),
        "analysable": int(len(a)), "analysable_subjects": int(a["subjectid"].nunique()),
        "analysable_cerclage": int(a["cerclage"].sum()),
        "analysable_sex": a["sex"].value_counts().to_dict(),
        # §2.2: case-table rocuronium or vecuronium total > 0; excluded from fitting and scoring
        "nmb_recorded_all_cases": int(cases["nmb_recorded"].sum()),
        "nmb_recorded_analysable": int(len(nmb)),
        "nmb_recorded_analysable_cases": nmb["caseid"].astype(int).tolist(),
        "analysed": int(len(an)), "analysed_subjects": int(an["subjectid"].nunique()),
        "analysed_primary_next_reading_targets": {"targets": int(len(k1_an)),
                                                  "cases": int(k1_an["caseid"].nunique())},
        # cases with >= 3 primary next-reading targets: those that can enter the primary outcome (§6.1)
        "analysed_cases_with_ge3_primary_targets": int((k1_an.groupby("caseid").size() >= 3).sum()),
        "fast_cycling_analysed": int(an["fast_cycling"].sum()),
        "fast_cycling_nmb_recorded": int(nmb["fast_cycling"].sum()),
        # P1 donors are drawn from all analysable cases (§6.6); analysed recipients whose donor has a blocker
        "analysed_with_nmb_donor": int(an["donor_caseid"].isin(set(nmb["caseid"])).sum())
        if "donor_caseid" in an else None,
        # sedation is not an exclusion (§2.2); case-table totals > 0 among the analysed cases
        "sedation_analysed": ({c: int((pd.to_numeric(demo.reindex(an["caseid"])[c], errors="coerce") > 0).sum())
                               for c in SEDATION_COLUMNS} if demo is not None else None),
        "fast_cycling_analysable": int(a["fast_cycling"].sum()),
        "fast_cycling_all_cases": int(cases["fast_cycling"].sum()),
        "median_of_case_median_interval_primary_s": {
            "all_cases": float(cases["median_interval_primary_s"].median()),
            "analysable": float(a["median_interval_primary_s"].median())},
        "late_waveform_start_analysable": a.loc[a["pleth_t_first"] > ins.PRIMARY_END_S, "caseid"].tolist(),
        "readings_all_cases_whole_record": reading_counts(readings),
        "readings_analysable_primary_window": reading_counts(readings.loc[rd_ana & in_p]),
        "readings_analysable_whole_record": reading_counts(readings.loc[rd_ana]),
        "issue_points_primary_by_definition": int(a["n_issue_primary"].sum()),
        "windows": flow,
        "p1_donors": {"recipients": int(len(a)),
                      "with_donor": int(a["caseid"].map(cases.set_index("caseid")["donor_caseid"]).notna().sum())},
        "hr_missing_primary": hr_missing,
    }
    if hr_tracks is not None:
        flow_out["hr_zero_primary_window"] = {"all_cases": hr_zero_counts(hr_tracks, cases["caseid"]),
                                              "analysable": hr_zero_counts(hr_tracks, a["caseid"])}
    if "vaso_stratum" in cases.columns:       # §2.6, descriptive; µg and mg are never added
        flow_out["vaso_strata_analysable"] = {
            "phenylephrine_or_ephedrine": a["vaso_stratum"].value_counts().to_dict(),
            "phenylephrine_alone": a["vaso_stratum_phe"].value_counts().to_dict()}
    return flow_out


def flow_markdown(f: dict) -> str:
    p = f["windows"]["primary"]
    s = f["windows"]["secondary"]
    rp = f["readings_analysable_primary_window"]
    ra = f["readings_all_cases_whole_record"]
    co = f["cohort"]
    L = [f"# {co} cohort flow (final cuff rule)", "",
         "Counts only; no model fitted, no forecast error computed (SPEC §9).", "",
         f"- {co} cases: {f['cases']} ({f['subjects']} subjects; {f['cerclage_cases']} cerclage, kept and flagged)",
         f"- Not analysable: {f['not_analysable']}"]
    for c in f["not_analysable_cases"]:
        L.append(f"  - case {c['caseid']}: {c['not_analysable_reason']} "
                 f"(valid readings in primary window: {c['n_valid_primary']}; "
                 f"pleth {c['pleth_status']}, ECG {c['ecg_status']})")
    L += [f"  - by first reason: {f['not_analysable_by_first_reason']}",
          f"  - any reason: {f['not_analysable_any_reason']}",
          f"- **Analysable: {f['analysable']}** ({f['analysable_subjects']} subjects; "
          f"{f['analysable_cerclage']} cerclage; sex {f['analysable_sex']})",
          f"  - fast cycling (primary-window median interval < 90 s): {f['fast_cycling_analysable']} "
          f"(all {co}: {f['fast_cycling_all_cases']})",
          f"  - median of case median intervals (primary window): "
          f"{f['median_of_case_median_interval_primary_s']['analysable']:.0f} s analysable, "
          f"{f['median_of_case_median_interval_primary_s']['all_cases']:.0f} s all {co}",
          f"  - waveforms start after the primary window: {f['late_waveform_start_analysable']}",
          f"- Neuromuscular blocker recorded (case-table rocuronium or vecuronium > 0; SPEC §2.2), analysable "
          f"cases: {f['nmb_recorded_analysable']} {f['nmb_recorded_analysable_cases']}; excluded from fitting "
          f"and scoring (all {co}: {f['nmb_recorded_all_cases']})",
          f"- **Analysed: {f['analysed']}** ({f['analysed_subjects']} subjects; "
          f"{f['analysed_primary_next_reading_targets']['targets']} primary next-reading targets in "
          f"{f['analysed_primary_next_reading_targets']['cases']} cases). The forecast, P1-donor and HR counts "
          f"below are over all {f['analysable']} analysable cases.",
          f"  - with >= 3 primary next-reading targets: {f['analysed_cases_with_ge3_primary_targets']}; "
          f"fast-cycling: {f['fast_cycling_analysed']} (and {f['fast_cycling_nmb_recorded']} of the excluded cases); "
          f"analysed recipients whose P1 donor has a blocker recorded: {f['analysed_with_nmb_donor']}",
          f"  - sedation recorded among the analysed (case-table total > 0; not an exclusion): "
          f"{f['sedation_analysed']}",
          "", "## Readings",
          f"- All {co}, whole record: {ra['readings']} readings, {ra['valid']} valid, {ra['dropped']} dropped "
          f"{ra['dropped_by_first_rule']}; gaps flagged {ra['gaps_flagged']}",
          f"- Analysable, primary window: {rp['readings']} readings, {rp['valid']} valid, {rp['dropped']} dropped "
          f"{rp['dropped_by_first_rule']}; gaps flagged {rp['gaps_flagged']}",
          f"  - failing each rule (overlapping): {rp['failing_each_rule']}",
          f"- Readings whose merged changes span <= 0.1 s: {ra['change_spread_le_0p1_s']} of {ra['readings']} "
          f"(largest span {ra['change_spread_max_s']:.1f} s)",
          "", "## Forecasts",
          f"- Issue points in the primary window (§3 definition): {f['issue_points_primary_by_definition']}; "
          f"with a next-reading target: {p['issue_points_with_target']}",
          f"- Primary targets (next reading, primary window): {p['primary_targets_next_reading']['targets']} "
          f"in {p['primary_targets_next_reading']['cases']} cases",
          f"- Targets by k (primary): {p['targets_by_k']}; secondary window: {s['targets_by_k']}",
          f"- Cases with >= 3 primary targets: {p['cases_ge3_next_reading_targets']} "
          f"(fast cycling: {p['fast_cycling_cases_ge3_next_reading_targets']}); "
          f">= 5: {p['cases_ge5_next_reading_targets']}",
          f"- Yardstick triples (primary): {p['yardstick_triples']}; cases with >= 3 triples: "
          f"{p['cases_ge3_triples']} (fast cycling: {p['fast_cycling_cases_ge3_triples']})",
          "", "| h-bin | primary targets | cases | subjects | >= 300 |", "|---|---|---|---|---|"]
    for label, d in p["primary_targets_next_reading"]["by_h_bin"].items():
        L.append(f"| {label} | {d['targets']} | {d['cases']} | {d['subjects']} | {'yes' if d['ge_300_targets'] else 'no'} |")
    L += ["", f"- Secondary window: {s['primary_targets_next_reading']['targets']} next-reading targets in "
          f"{s['primary_targets_next_reading']['cases']} cases",
          f"- P1 donors: {f['p1_donors']['with_donor']} of {f['p1_donors']['recipients']} recipients",
          f"- HR group missing (primary issue points): " +
          ", ".join(f"{v} {d['hr_group_missing']}/{d['issue_points']}" for v, d in f["hr_missing_primary"].items()),
          ""]
    if "hr_zero_primary_window" in f:
        z, za = f["hr_zero_primary_window"]["all_cases"], f["hr_zero_primary_window"]["analysable"]
        L.insert(-1, f"- Monitor HR = 0 in the primary window (treated as missing): {z['zero']} of {z['samples']} "
                     f"samples ({100 * z['share_zero']:.1f}%) in {z['cases_with_zero']} cases; analysable cases: "
                     f"{za['zero']} of {za['samples']} ({100 * za['share_zero']:.1f}%) in {za['cases_with_zero']} cases")
    if "vaso_strata_analysable" in f:
        v = f["vaso_strata_analysable"]
        L.insert(-1, f"- Vasopressor strata, analysable (SPEC §2.6): phenylephrine or ephedrine "
                     f"{v['phenylephrine_or_ephedrine']}; phenylephrine alone {v['phenylephrine_alone']}")
    return "\n".join(L)


if __name__ == "__main__":
    main()
