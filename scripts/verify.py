#!/usr/bin/env python3
"""Check the public record of this repository against itself (SPEC.md §8.3).

    python scripts/verify.py           # report as text
    python scripts/verify.py --json    # the same report as JSON

Offline, without data downloads, in seconds. It needs git, a clone with its tags, and the packages
of pyproject.toml. It fits no model and writes no file. Checks:

  1  Registration and freeze order  SHA-256 of PREREG.md and SPEC.md at the commits of REPORT.md's
                                    registration table; the tags and their commits; their ancestry
  2  Manifests                      SHA-256 of every committed file listed in results/**/MANIFEST.json
                                    and public_tables.json; listed files that are not committed
  3  Frozen models, one scoring     the models-frozen tag message, FROZEN.json, STATE.json, the log
  4  Recomputation                  the headline numbers and sentences from the committed per-patient
                                    tables, with the project's bootstrap (btc/models/bootstrap.py) and
                                    sentence code (btc/models/headline.py), against the stored files
  5  README and REPORT              the registered lines, the table, the hypotension line, deviations
  6  Data checksums                 results/data_checksums.csv against results/sdev/data_checksums.csv
  7  Public-data rule               no per-forecast cuff column in a committed CSV file; no raw data

Each check prints PASS, FAIL or SKIPPED, with one line per item and the files or commands it used.
NOTE lines say what an item cannot show. Exit code 0 if no check fails (SKIPPED is not a failure).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from btc.models import headline as H  # noqa: E402
from btc.models.bootstrap import Contrast, SubjectBootstrap, contrast, fraction_F, mean_ci  # noqa: E402
from btc.models.constants import MIN_FORECASTS_RMSE, MIN_TRIPLES, N_BOOT, SEED  # noqa: E402
from btc.models.pipeline import MAIN_CONTRASTS, README_MODELS  # noqa: E402
from btc.models.secondary import hypotension_line  # noqa: E402
from btc.site.text import fmt_ci_mm  # noqa: E402

ROOT = CODE                 # the repository checked (--root); the code imported above is always this one
TOL = 1e-9                  # recomputed vs stored numbers: allows last-digit differences between numerical
                            # libraries (results/README.md), nothing more
KST = timezone(timedelta(hours=9))
TAGS = ("prereg-v1.5", "feature-freeze", "models-frozen")
SDEV, SOB = "results/sdev/confirmatory/", "results/sob/frozen_scoring/"
NMB, LOG = SOB + "sensitivity_including_nmb/", "results/sob/frozen_scoring_log.jsonl"
CODE_PATHS = ("btc", "scripts", "tests", "pyproject.toml", "uv.lock")
# REPORT.md, "Frozen models", "Unchanged code": files that are the same at feature-freeze and models-frozen
CORE = [f"btc/{f}.py" for f in ("beats", "waveform", "hr", "cuff", "pqi", "rpeaks", "instances", "io",
                                 "constants", "paths", "provenance", "__init__")] + \
       [f"scripts/{f}.py" for f in ("build_instances", "build_windows", "build_features_wf", "download",
                                     "select_rpeak_detector", "pleth_coverage")] + \
       [f"btc/models/{f}.py" for f in ("ridge", "design", "data", "folds", "analyses", "bootstrap",
                                        "metrics", "headline", "constants", "gbm")]
# results/README.md, first paragraph: the committed result files that are in no manifest
UNLISTED = {"results/README.md", "results/data_checksums.csv", "results/sdev/data_checksums.csv",
            SOB + "STATE.json", LOG}
CUFF_COLUMN = re.compile(r"^(y|last)_(map|sbp|dbp)$")      # per-forecast cuff values (results/README.md)


# ---------------------------------------------------------------------------- helpers

class Check:
    def __init__(self, num: int, title: str):
        self.num, self.title, self.items = num, title, []

    def add(self, status: str, text: str, *evidence: str):
        self.items.append({"status": status, "text": text, "evidence": list(evidence)})

    def test(self, ok, text: str, *evidence: str):
        self.add("PASS" if ok else "FAIL", text, *evidence)

    @property
    def status(self) -> str:
        s = {i["status"] for i in self.items}
        return "FAIL" if "FAIL" in s else "SKIPPED" if "SKIPPED" in s else "PASS"


def run_git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)


def git(*args) -> str:
    r = run_git(*args)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.decode().strip()}")
    return r.stdout.decode()


def commit_of(rev: str) -> str:
    return git("rev-parse", "--verify", f"{rev}^{{commit}}").strip()


def first_commit_adding(path: str) -> str:
    return git("log", "--diff-filter=A", "--format=%H", "--", path).split()[-1]


def short(rev: str) -> str:
    return rev[:7] if re.fullmatch(r"[0-9a-f]{40}", rev) else rev


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def load(rel: str) -> dict:
    return json.loads(read(rel))


def table(rel: str) -> pd.DataFrame:
    """A result CSV; its first line names the commit that wrote it (results/README.md)."""
    return pd.read_csv(ROOT / rel, skiprows=1)


def listed(xs) -> str:
    return f" ({', '.join(map(str, xs))})" if xs else ""


def section(text: str, heading: str) -> str:
    """Body of the Markdown section `heading` (e.g. '## Deviations'), up to the next heading of its level."""
    level = heading.split()[0]
    m = re.search(rf"^{re.escape(heading)}\n(.*?)(?=^{level} |\Z)", text, re.M | re.S)
    if not m:
        raise ValueError(f"section {heading!r} not found")
    return m.group(1)


def num(v) -> float:
    """A stored number: the result files write NaN as null and infinities as "inf" / "-inf"."""
    return float("nan") if v is None else float(v)


def diff(stored, new, path: str = "") -> list:
    """Differences between a stored value and the recomputed one (numbers within TOL)."""
    if isinstance(new, dict):
        return [d for k, v in new.items()
                for d in (diff(stored[k], v, f"{path}{k}.") if k in stored else [f"{path}{k}: not stored"])]
    if new is None or isinstance(new, (bool, str)) or type(new).__name__ == "bool_":
        return [] if stored == new else [f"{path[:-1]}: stored {stored!r}, recomputed {new!r}"]
    a, b = num(stored), float(new)
    same = a == b or abs(a - b) <= TOL or (a != a and b != b)
    return [] if same else [f"{path[:-1]}: stored {a!r}, recomputed {b!r}"]


def report(c: Check, bad: list, text: str, *evidence: str):
    c.test(not bad, text + (f": {len(bad)} difference(s), e.g. {'; '.join(bad[:3])}" if bad else ""), *evidence)


def entering(df: pd.DataFrame, count: str, minimum: int, fast: bool = False) -> tuple:
    """The patients that enter (`minimum` or more forecasts or triples) in ascending case-ID order, with
    their subject bootstrap, as `Scored` builds them in btc/models/analyses.py."""
    e = df.loc[df[count] >= minimum].sort_values("caseid")
    if fast:
        e = e.loc[e["fast_cycling"].astype(bool)]
    return e, SubjectBootstrap(e["caseid"].to_numpy(), e["subjectid"].to_numpy(), N_BOOT, SEED)


def contrasts(e: pd.DataFrame, bs: SubjectBootstrap, pairs) -> dict:
    return {f"{a} vs {b}": contrast(bs, e[f"rmse_{a}"], e[f"rmse_{b}"], a, b) for a, b in pairs}


def as_dicts(con: dict) -> dict:
    return {k: x.as_dict() for k, x in con.items()}


def yardstick(df: pd.DataFrame, fast: bool = False) -> dict:
    """N, D, F, the model means and the placebo guard, as `yardstick_scores` in btc/models/analyses.py."""
    e, bs = entering(df, "n_triples", MIN_TRIPLES, fast)
    out = {"n_patients": len(e), "n_triples": int(e["n_triples"].sum()),
           **fraction_F(bs, e["rmse_L"], e["rmse_T"], e["rmse_U"])}
    out.update({col: float(e[col].mean()) for col in e.columns if col.startswith("rmse_")})
    g = contrast(bs, e["rmse_T"], e["rmse_T-P1"], "T", "T-P1")
    out.update(T_minus_TP1=g.as_dict(), placebo_guard_passed=bool(g.delta_hi < 0))
    if "rmse_T-ia" in e:
        out["ia"] = fraction_F(bs, e["rmse_L"], e["rmse_T-ia"], e["rmse_U"])
    return out


def stored_contrast(d: dict) -> Contrast:
    return Contrast(**{f.name: d[f.name] if f.type in ("str", "int") else num(d[f.name])
                       for f in dataclasses.fields(Contrast)})


# ---------------------------------------------------------------------------- 1. registration

def check_registration(c: Check):
    reg = section(read("REPORT.md"), "## Registration")
    versions = re.findall(r"^\| (\d\.\d) \| `([0-9a-f]{7,40})` \| [\d-]+ \| `([0-9a-f]{64})` \| `([0-9a-f]{64})` \|$",
                          reg, re.M)
    c.test(len(versions) >= 2, f"REPORT.md registration table: {len(versions)} versions", "REPORT.md#registration")
    for ver, com, *hashes in versions:
        for name, h in zip(("PREREG.md", "SPEC.md"), hashes):
            c.test(sha(run_git("show", f"{com}:{name}").stdout) == h,
                   f"version {ver}: {name} at commit {com} has the SHA-256 of REPORT.md ({h[:12]}…)",
                   f"git show {com}:{name}", "REPORT.md#registration")
    reg_commit = commit_of(versions[-1][1])                     # the last version is the registration
    for name, h in zip(("PREREG.md", "SPEC.md"), versions[-1][2:]):
        c.test(sha((ROOT / name).read_bytes()) == h, f"{name} in the working tree is the registered version", name)
    later = git("log", "--format=%h", f"{reg_commit}..HEAD", "--", "PREREG.md", "SPEC.md").split()
    c.test(not later, f"no commit after the registration changes PREREG.md or SPEC.md{listed(later)}",
           f"git log {short(reg_commit)}..HEAD -- PREREG.md SPEC.md")
    # commit date in KST, optionally followed by " KST (… UTC)"
    rows = re.findall(r"^\| `([\w.-]+)` \| `([0-9a-f]{7,40})` \| (\d{4}-\d\d-\d\d \d\d:\d\d)(?: KST \([^)]*\))? \|",
                      reg, re.M)
    named = {t: (com, date) for t, com, date in rows}
    for t in TAGS:
        com, date = named.get(t, ("(none)", "(none)"))
        tc = commit_of(t)
        c.test(tc.startswith(com), f"tag {t} points to {short(tc)}; REPORT.md names {com}",
               f"git rev-parse {t}^{{commit}}")
        when = datetime.fromtimestamp(int(git("log", "-1", "--format=%ct", tc)), KST).strftime("%Y-%m-%d %H:%M")
        c.test(when == date, f"tag {t}: commit date {when} KST, as in REPORT.md (a date set by the author's clock)",
               f"git log -1 --format=%cd {short(tc)}")
    c.test(commit_of("prereg-v1.5") == reg_commit, f"tag prereg-v1.5 is the registration commit {short(reg_commit)}",
           "REPORT.md#registration")
    for t in ("prereg-v1.5", "feature-freeze"):
        files = git("ls-tree", "-r", "--name-only", t, "--", "results").split()
        c.test(not files, f"no file under results/ at tag {t} ({len(files)} found)", f"git ls-tree -r {t} -- results")
    first_sob = first_commit_adding(SOB + "summary.json")
    chain = [("registration commit", reg_commit), ("feature-freeze", commit_of("feature-freeze")),
             ("models-frozen", commit_of("models-frozen")), (f"the first commit adding {SOB}summary.json", first_sob)]
    for (na, a), (nb, b) in zip(chain, chain[1:]):
        c.test(a != b and run_git("merge-base", "--is-ancestor", a, b).returncode == 0,
               f"{na} ({short(a)}) is an ancestor of {nb} ({short(b)})",
               f"git merge-base --is-ancestor {short(a)} {short(b)}")
    changed = git("diff", "--name-only", "feature-freeze", "models-frozen", "--", *CORE).split()
    c.test(not changed, f"the {len(CORE)} feature-code and model-core files that REPORT.md lists are the same at "
           f"feature-freeze and models-frozen{listed(changed)}", "git diff feature-freeze models-frozen -- <files>")
    refit = load("results/frozen/FROZEN.json")["meta"]["commit"]
    for a, b in ((refit, "models-frozen"), ("models-frozen", first_sob),
                 ("models-frozen", first_commit_adding(SOB + "per_patient_primary_wide.csv"))):
        changed = git("diff", "--name-only", a, b, "--", *CODE_PATHS).split()
        c.test(not changed, f"no change in {', '.join(CODE_PATHS)} between {short(a)} and {short(b)}{listed(changed)}",
               f"git diff {short(a)} {short(b)} -- {' '.join(CODE_PATHS)}")
    repo = load("btc/site/site.json")["repo_url"]
    c.add("NOTE", "Commit and tag dates are set by the author's clock and can be rewritten; git ancestry shows order, "
          "not time. Times set by GitHub's server are the published time of each release and the pushes on the "
          "repository's activity page; this script does not read them (offline).",
          f"{repo}/releases", f"{repo}/activity")


# ---------------------------------------------------------------------------- 2. manifests

def withheld() -> dict:
    """The files results/README.md lists as 'listed in a manifest but not committed': path -> reason."""
    body = section(read("results/README.md"), "## Files listed in a manifest but not committed")
    out, reason = {}, ""
    for line in body.splitlines():
        reason = line[4:].strip() if line.startswith("### ") else reason
        for p in re.findall(r"`((?:sdev|sob|frozen)/[\w./-]+\.csv)`", line):
            out.setdefault("results/" + p, reason)
    return out


def ignore_reason(p: str) -> str | None:
    """First sentence of the comment above the .gitignore rule that ignores `p`; None if `p` is not ignored."""
    r = run_git("check-ignore", "-v", p)
    if r.returncode:
        return None
    src, line = r.stdout.decode().split("\t")[0].split(":")[:2]
    above, com = (ROOT / src).read_text(encoding="utf-8").splitlines()[:int(line) - 1], []
    while above and above[-1].strip() and not above[-1].startswith("#"):
        above.pop()                                     # other rules of the same block
    while above and above[-1].startswith("#"):
        com.insert(0, above.pop().lstrip("# ").strip())
    return " ".join(com).split(". ")[0] + f" ({src}, line {line})"


def manifest_base(m: str, keys, tracked: set) -> str:
    """The folder a manifest's keys are relative to: its own folder or one above it, up to results/."""
    d = Path(m).parent
    cands = [d] + [p for p in d.parents if p.is_relative_to("results")]
    return max(cands, key=lambda b: sum(f"{b.as_posix()}/{k}" in tracked for k in keys)).as_posix()


def check_manifests(c: Check):
    tracked = set(git("ls-files", "-z", "results").split("\0")) - {""}
    reasons, covered = withheld(), set()
    for m in sorted(p for p in tracked if p.endswith("/MANIFEST.json")):
        files = load(m)["files"]
        base, ok = manifest_base(m, files, tracked), []
        for k, h in sorted(files.items()):
            p = f"{base}/{k}"
            covered.add(p)
            if p in tracked:
                ok.append(sha((ROOT / p).read_bytes()) == h)
                if not ok[-1]:
                    c.test(False, f"{p}: SHA-256 differs from the one listed in {m} ({h[:12]}…)", p, m)
                continue
            why, ign = reasons.get(p), ignore_reason(p)          # not committed: is the reason written down?
            if not (why or ign):
                c.test(False, f"{p}: listed in {m} but not committed, not git-ignored, and results/README.md does not "
                       "say why", m)
                continue
            why = why or f"{ign}; not in results/README.md, 'Files listed in a manifest but not committed'"
            local = "" if not (ROOT / p).exists() else "; a local copy is present, its SHA-256 " + (
                "matches" if sha((ROOT / p).read_bytes()) == h else "differs")
            c.add("NOTE", f"{p}: listed, not distributed{' (git-ignored)' if ign else ''}: {why}{local}", m,
                  "results/README.md", *([".gitignore"] if ign else []))
        c.test(all(ok), f"{m}: {sum(ok)} of {len(ok)} committed files have the listed SHA-256", m)
    for pt in sorted(p for p in tracked if p.endswith("/public_tables.json")):
        d, folder = load(pt), Path(pt).parent.as_posix()
        bad = [n for n, h in d["files"].items() if sha((ROOT / folder / n).read_bytes()) != h]
        covered |= {f"{folder}/{n}" for n in d["files"]}
        for n, s in d["sources"].items():           # each source is the file that its run's manifest lists
            m = "results/" + s["manifest"]
            files = load(m)["files"]
            key = Path(folder, n).relative_to(manifest_base(m, files, tracked)).as_posix()
            bad += [] if files.get(key) == s["sha256"] else [f"source {n} differs from {m}"]
        c.test(not bad, f"{pt}: the {len(d['files'])} per-patient tables have the listed SHA-256, and their sources "
               f"are the files listed in the run's manifest{listed(bad)}", pt)
    lists = {p for p in tracked if p.endswith(("/MANIFEST.json", "/public_tables.json"))}
    extra = sorted(tracked - covered - UNLISTED - lists)
    c.test(not extra, f"every committed file under results/ is in a manifest or public_tables.json, or is one of the "
           f"exceptions that results/README.md names{listed(extra)}", "results/README.md")


# ---------------------------------------------------------------------------- 3. frozen models, one scoring

def check_frozen(c: Check):
    msg = git("tag", "-l", "--format=%(contents)", "models-frozen")
    in_tag = dict(re.findall(r"^(results/\S+)\s+SHA-256 ([0-9a-f]{64})$", msg, re.M))
    c.test(set(in_tag) == {"results/frozen/models.pkl", "results/frozen/FROZEN.json"},
           f"the models-frozen tag message gives the SHA-256 of{listed(sorted(in_tag))}",
           "git tag -l --format='%(contents)' models-frozen")
    for p, h in in_tag.items():
        c.test(sha((ROOT / p).read_bytes()) == h, f"{p}: SHA-256 = tag message ({h[:12]}…)", p)
        c.test(sha(run_git("show", f"models-frozen:{p}").stdout) == h,
               f"{p} at tag models-frozen: SHA-256 = tag message",
               f"git show models-frozen:{p}")
    fz, pkl = load("results/frozen/FROZEN.json"), sha((ROOT / "results/frozen/models.pkl").read_bytes())
    c.test(fz["models_sha256"] == pkl, "FROZEN.json models_sha256 = SHA-256 of models.pkl",
           "results/frozen/FROZEN.json")
    text = section(read("REPORT.md"), "## Frozen models (`SPEC.md` §9 step 6)")
    c.test(pkl in text and in_tag.get("results/frozen/FROZEN.json", "(none)") in text,
           "REPORT.md 'Frozen models' gives the same two SHA-256", "REPORT.md")
    c.test(fz["meta"]["inputs_sha256"] == load("results/sdev/MANIFEST.json")["meta"]["inputs_sha256"],
           "FROZEN.json inputs_sha256 = those of the confirmatory run", "results/sdev/MANIFEST.json")
    tag, st = commit_of("models-frozen"), load(SOB + "STATE.json")
    c.test(st.get("status") == "completed" and st.get("commit") == tag and st.get("rerun_after_crash") is False,
           f"STATE.json: status {st.get('status')}, commit {short(str(st.get('commit')))}, rerun_after_crash "
           f"{st.get('rerun_after_crash')}", SOB + "STATE.json")
    ev = [json.loads(x) for x in read(LOG).splitlines() if x.strip()]
    kinds = [e["event"] for e in ev]
    c.test(kinds.count("started") == 1 and kinds.count("completed") == 1
           and not {"refused", "crashed", "rerun_after_crash"} & set(kinds)
           and all(e["commit"] == tag for e in ev if e["event"] in ("started", "completed")),
           f"scoring log: {len(ev)} events{listed(kinds)}; one 'started' and one 'completed', both at commit "
           f"{short(tag)} (tag models-frozen); no 'refused', 'crashed' or 'rerun_after_crash' event", LOG)
    for p in (SOB + "summary.json", SOB + "MANIFEST.json"):
        m = load(p)["meta"]
        c.test(m["commit"] == tag and m["state"] == "clean" and "models-frozen" in m["tags_at_head"]
               and m["frozen_models_sha256"] == pkl, f"{p} meta: commit {short(m['commit'])}, state {m['state']}, "
               f"tags at HEAD {m['tags_at_head']}, frozen_models_sha256 = models.pkl", p)
    c.add("NOTE", "That the cesarean cases were scored once rests on STATE.json and the log, plain files written by "
          "scripts/run_frozen_scoring.py; git shows when they were committed, not how often the script ran.",
          SOB + "STATE.json", LOG)


# ---------------------------------------------------------------------------- 4. recomputation

def check_recompute(c: Check):
    used = ("btc/constants.py", "btc/models/constants.py", "btc/models/bootstrap.py", "btc/models/headline.py")
    changed = [p for p in used if (CODE / p).read_bytes() != run_git("show", f"feature-freeze:{p}").stdout]
    c.test(not changed, f"the bootstrap, sentence and constant code used here is that of tag feature-freeze, at which "
           f"the S-dev result was computed{listed(changed)}", *(f"git show feature-freeze:{p}" for p in used))
    # S-dev: N, the ten contrasts, the decision row with G and g, and the plain answer line and Sentence 1
    S, w = load(SDEV + "summary.json"), table(SDEV + "per_patient_primary_wide.csv")
    ev = (SDEV + "per_patient_primary_wide.csv", SDEV + "summary.json")
    c.test((w["entered"] == (w["n_forecasts"] >= MIN_FORECASTS_RMSE)).all(),
           "S-dev table: 'entered' = at least 3 forecasts (SPEC §6.1)", ev[0])
    e, bs = entering(w, "n_forecasts", MIN_FORECASTS_RMSE)
    con, cov = contrasts(e, bs, MAIN_CONTRASTS), w["n_forecasts_waveform"].sum() / w["n_forecasts"].sum()
    report(c, diff(S, {"N": len(e), "n_forecasts": int(e["n_forecasts"].sum()), "waveform_coverage": cov,
                       "contrasts": as_dicts(con)}),
           f"S-dev: N = {len(e)}, waveform coverage {H.pct(cov)}%, {len(con)} contrasts (Δ, relative reduction, 95% "
           "CIs; 2,000 subject resamples, seed 20261118) = summary.json", *ev)
    hi = H.HeadlineInputs(N=len(e), m4_m2=con["M4 vs M2"], m3_m2=con["M3 vs M2"], m4_m3=con["M4 vs M3"],
                          m4p1_m2=con["M4-P1 vs M2"], m4_m4p1=con["M4 vs M4-P1"], m3p1_m2=con["M3-P1 vs M2"],
                          m3_m3p1=con["M3 vs M3-P1"])
    sel = H.select_row(hi)
    report(c, diff(S["decision"], sel), f"S-dev decision: row {sel['row']}, G = {100 * sel['G']:.3f}%, "
           f"g = {sel['g']:.4f} mmHg = summary.json 'decision'", *ev)
    # Sentence 3: the fast-cycling contrast and the yardstick F with its placebo guard
    ef, bsf = entering(w, "n_forecasts", MIN_FORECASTS_RMSE, fast=True)
    xf = contrast(bsf, ef["rmse_M4"], ef["rmse_M2"], "M4", "M2")
    report(c, diff(S["fast_cycling"], {"K": len(ef), "m4_vs_m2": xf.as_dict()}),
           f"S-dev fast-cycling cases: K = {len(ef)}, M4 vs M2 = summary.json", *ev)
    yw = table(SDEV + "per_patient_yardstick.csv")
    ya, yf = yardstick(yw), yardstick(yw, fast=True)
    for name, y in (("yardstick_all", ya), ("yardstick_fast_cycling", yf)):
        report(c, diff(S[name], y), f"S-dev {name}: {y['n_patients']} patients, D {y['D']:.3f} mmHg, F "
               f"{H.pct(y['F'])}%, placebo guard passed {y['placebo_guard_passed']} = summary.json",
               SDEV + "per_patient_yardstick.csv", ev[1])
    head = read(SDEV + "headline.md")
    s3 = H.sentence3(len(ef), xf, yf if yf["n_patients"] else None, yf.get("placebo_guard_passed"))
    for lab, key, s in (("Question", "question", H.question()), ("Answer line", "answer_line", H.answer_line(hi, sel)),
                        ("Sentence 1", "sentence_1", H.sentence1(hi, sel)), ("Sentence 3", "sentence_3", s3)):
        c.test(S[key] == s and f"**{lab}.** {s}\n" in head, f"{lab}, rendered by btc/models/headline.py from the "
               "recomputed numbers = summary.json and headline.md", SDEV + "headline.md", ev[1])
    # S-ob: Sentence 2, the cesarean line, the yardstick and the sensitivity analysis
    O, wo = load(SOB + "summary.json"), table(SOB + "per_patient_primary_wide.csv")
    eo, bso = entering(wo, "n_forecasts", MIN_FORECASTS_RMSE)
    co, covo = contrasts(eo, bso, MAIN_CONTRASTS), wo["n_forecasts_waveform"].sum() / wo["n_forecasts"].sum()
    report(c, diff(O, {"n": len(eo), "n_forecasts": int(eo["n_forecasts"].sum()), "waveform_coverage": covo,
                       "coverage_rule_applies": H.coverage_rule_applies(covo), "contrasts": as_dicts(co)}),
           f"S-ob: n = {len(eo)}, waveform coverage {H.pct(covo)}%, {len(co)} contrasts = summary.json",
           SOB + "per_patient_primary_wide.csv", SOB + "summary.json")
    yo, after = yardstick(table(SOB + "per_patient_yardstick.csv")), "results/sob/after_frozen_scoring/summary.json"
    report(c, diff(O["yardstick"], yo), f"S-ob yardstick: {yo['n_patients']} patients, F {H.pct(yo['F'])}%, with the "
           "inflation-artefact variant = summary.json", SOB + "per_patient_yardstick.csv", SOB + "summary.json")
    report(c, diff(load(after)["yardstick_with_ia"], yo), "S-ob yardstick = 'yardstick_with_ia' of the analyses "
           "after the frozen scoring", after)
    oh = read(SOB + "headline.md")
    for lab, key, s in (("Sentence 2", "sentence_2", H.sentence2(len(eo), co["M4 vs M2"], covo)),
                        ("Cesarean line", "cesarean_line", H.cesarean_line(len(eo), co["M4 vs M2"], covo))):
        c.test(O[key] == s and f"**{lab}.** {s}\n" in oh, f"{lab}, rendered by btc/models/headline.py from the "
               "recomputed numbers = summary.json and headline.md", SOB + "headline.md", SOB + "summary.json")
    wn = table(NMB + "per_patient_primary_wide.csv")
    en, bsn = entering(wn, "n_forecasts", MIN_FORECASTS_RMSE)
    report(c, diff(load(NMB + "summary.json"), {"n": len(en), "n_forecasts": int(en["n_forecasts"].sum()),
                                                "contrasts": as_dicts(contrasts(en, bsn, MAIN_CONTRASTS))}),
           f"S-ob sensitivity analysis with the blocker case: n = {len(en)}, contrasts = summary.json",
           NMB + "per_patient_primary_wide.csv", NMB + "summary.json")
    # the README table: mean RMSE with its 95% CI for every model of both cohorts
    for cohort, (ee, b, d) in {"S-dev": (e, bs, SDEV), "S-ob": (eo, bso, SOB)}.items():
        ms, bad = table(d + "model_summary.csv").set_index("model"), []
        for col in (x for x in ee.columns if x.startswith("rmse_")):
            mu, lo, up = mean_ci(b, ee[col])
            bad += diff(ms.loc[col[5:]].to_dict(), {"rmse_mean": mu, "rmse_mean_ci_lo": lo, "rmse_mean_ci_hi": up},
                        col[5:] + ".")
        report(c, bad, f"{cohort}: mean RMSE with its 95% CI, every model (the README table) = model_summary.csv",
               d + "per_patient_primary_wide.csv", d + "model_summary.csv")
    # the exploratory encoder check (ENCODER_PLAN.md), where its results exist
    for cohort, d in (("S-dev", "results/sdev/exploratory_encoder/"),
                      ("S-ob", "results/sob/after_frozen_scoring/exploratory_encoder/")):
        if not (ROOT / d / "summary.json").exists():
            continue
        X, wx = load(d + "summary.json"), table(d + "per_patient_primary_wide.csv")
        ex, bx = entering(wx, "n_forecasts", MIN_FORECASTS_RMSE)
        new = {"N" if "N" in X else "n": len(ex), "n_forecasts": int(ex["n_forecasts"].sum()),
               "contrasts": as_dicts(contrasts(ex, bx, [tuple(k.split(" vs ")) for k in X["contrasts"]])),
               "rmse_patient_mean": {x[5:]: dict(zip(("mean", "ci_lo", "ci_hi"), mean_ci(bx, ex[x])))
                                     for x in ex.columns if x.startswith("rmse_")}}
        if isinstance(X.get("fast_cycling"), dict):
            fx, bf = entering(wx, "n_forecasts", MIN_FORECASTS_RMSE, fast=True)
            new["fast_cycling"] = {"K": len(fx), "M4-enc vs M2": contrast(bf, fx["rmse_M4-enc"], fx["rmse_M2"],
                                                                         "M4-enc", "M2").as_dict()}
        report(c, diff(X, new), f"{cohort}, exploratory encoder check: {len(ex)} patients, {len(X['contrasts'])} "
               "contrasts and the mean RMSE with 95% CIs = summary.json", d + "per_patient_primary_wide.csv",
               d + "summary.json")


# ---------------------------------------------------------------------------- 5. README and REPORT

def check_documents(c: Check):
    S, O, sec = load(SDEV + "summary.json"), load(SOB + "summary.json"), load("results/sdev/secondary/summary.json")
    rep, readme = read("REPORT.md"), read("README.md")
    lines = {"Answer line": S["answer_line"], "Cesarean line": O["cesarean_line"], "Sentence 1": S["sentence_1"],
             "Sentence 2": O["sentence_2"], "Sentence 3": S["sentence_3"]}
    for lab, s in lines.items():
        c.test(f"**{lab}.** {s}" in rep, f"REPORT.md quotes the {lab} of the headline file verbatim", "REPORT.md")
    hyp = {"S-dev": table("results/sdev/secondary/hypotension.csv"),
           "S-ob": table("results/sob/after_frozen_scoring/hypotension.csv")}
    c.test(hypotension_line(hyp) in rep, "REPORT.md hypotension line = the line rendered by btc/models/secondary.py "
           "from the two hypotension.csv files", "results/sdev/secondary/hypotension.csv",
           "results/sob/after_frozen_scoring/hypotension.csv")
    dev = section(rep, "## Deviations").split("The following are not deviations")[0].strip()
    n_dev = 0 if dev.startswith("None") else sum(x.startswith("- ") for x in dev.splitlines())
    site = load("btc/site/site.json")
    c.test(int(site["deviations"]) == n_dev, f"REPORT.md 'Deviations' lists {n_dev}; btc/site/site.json, from which "
           f"the README's registration line is built, gives {site['deviations']}", "REPORT.md#deviations",
           "btc/site/site.json")
    c.test(commit_of(site["registration_commit"]) == commit_of("prereg-v1.5") and site["frozen_tag"] == "models-frozen"
           and first_commit_adding(SOB + "summary.json").startswith(site["scored_ref"]),
           "btc/site/site.json: registration commit, freeze tag and scoring commit (README links) match git",
           "btc/site/site.json")
    gb = {k: stored_contrast(v) for k, v in sec["gradient_boosting"]["contrasts"].items()}
    full = gb["GB-full vs GB-cuff"]
    meets = full.meets_rule and H.placebo_valid(gb["GB-full-P1 vs GB-cuff"], gb["GB-full vs GB-full-P1"])
    gb_line = H.gb_line(meets, int(S["headline_row"]), full)
    stored = sec["gradient_boosting"]
    c.test(meets == stored["meets_rule_with_placebo"] and gb_line == stored["readme_line"],
           f"gradient-boosting check: rule met with placebo validity = {meets}, so its fixed README line is "
           f"{'not ' if gb_line is None else ''}triggered (SPEC §6.8; from the stored GB contrasts, as no per-patient "
           "GB table is published)", "results/sdev/secondary/summary.json")
    if "not scored yet" in readme:
        c.add("SKIPPED", "README.md is the interim version written before the cesarean scoring ('not scored yet'); the "
              "final one is generated by scripts/build_site.py. Not checked: the registered lines, the table, the "
              "hypotension line, the deviation note and the gradient-boosting line of README.md", "README.md")
        return
    plain = re.sub(r"\\([\\`*_\[\]<>])", r"\1", readme.replace("**", ""))   # without emphasis and escapes
    for lab, s in {"Question": S["question"], **lines}.items():
        c.test(s in plain, f"README.md has the {lab} of the headline file verbatim", "README.md")
    ms = [table(d + "model_summary.csv").set_index("model") for d in (SDEV, SOB)]
    rows = plain.splitlines()
    for m in README_MODELS:
        row = next((x for x in rows if re.match(rf"^\| {re.escape(m)}( ·| \|)", x)), "")
        want = [fmt_ci_mm(tuple(t.loc[m, ["rmse_mean", "rmse_mean_ci_lo", "rmse_mean_ci_hi"]])) for t in ms]
        c.test(all(x in row for x in want), f"README.md table, {m}: {' | '.join(want)}", "README.md")
    cov = next((x for x in rows if x.startswith("| Waveform coverage")), "")
    c.test(all(f"{H.pct(x)}%" in cov for x in (S["waveform_coverage"], O["waveform_coverage"]))
           and f"N = {S['N']}" in plain and f"n = {O['n']}" in plain, "README.md table: N, n and waveform coverage",
           "README.md")
    c.test(hypotension_line({"Non-cesarean": hyp["S-dev"], "Cesarean": hyp["S-ob"]}) in plain,
           "README.md hypotension line = the line rendered from the two hypotension.csv files", "README.md")
    m = re.search(r"(\d+) (?:minor )?deviations? from the registration", plain)
    c.test((int(m.group(1)) if m else 0) == n_dev, f"README.md registration line: "
           f"{m.group(0) if m else 'no deviation note'}; REPORT.md lists {n_dev}", "README.md", "REPORT.md#deviations")
    c.test(gb_line in plain if gb_line else "gave a different answer" not in plain,
           f"README.md {'has' if gb_line else 'has no'} gradient-boosting line, as the check requires", "README.md")


# ---------------------------------------------------------------------------- 6. data checksums

def check_checksums(c: Check):
    a = pd.read_csv(ROOT / "results/data_checksums.csv", dtype=str)
    b = pd.read_csv(ROOT / "results/sdev/data_checksums.csv", dtype=str)
    c.test(list(a.columns) == ["cohort", "caseid", "array", "dtype", "n", "sha256"]
           and list(b.columns) == ["caseid", "array", "dtype", "n", "sha256"],
           "columns cohort, caseid, array, dtype, n, sha256 (the S-dev list without cohort)",
           "results/data_checksums.csv", "results/sdev/data_checksums.csv")
    c.test(a["sha256"].str.fullmatch(r"[0-9a-f]{64}").all() and set(a["cohort"]) == {"S-dev", "S-ob"},
           "every row is S-dev or S-ob and has a 64-digit hexadecimal SHA-256", "results/data_checksums.csv")
    sd = a.loc[a["cohort"] == "S-dev", list(b.columns)].reset_index(drop=True)
    c.test(sd.equals(b), f"the {len(sd)} S-dev rows equal the {len(b)} rows of results/sdev/data_checksums.csv, "
           "in order",
           "results/data_checksums.csv", "results/sdev/data_checksums.csv")
    n = {k: (int((a["cohort"] == k).sum()), a.loc[a["cohort"] == k, "caseid"].nunique()) for k in ("S-dev", "S-ob")}
    txt = section(read("REPORT.md"), "## Data and result files").replace(",", "")
    said = (re.search(r"every stored array of the (\d+) S-dev cases", txt), re.search(r"its (\d+) S-dev rows", txt),
            re.search(r"(\d+) rows cover the (\d+) cesarean cases", txt))
    claim = [int(said[0][1]), int(said[1][1]), int(said[2][1]), int(said[2][2])] if all(said) else None
    c.test(claim == [n["S-dev"][1], n["S-dev"][0], n["S-ob"][0], n["S-ob"][1]],
           f"S-dev {n['S-dev'][0]} rows of {n['S-dev'][1]} cases, S-ob {n['S-ob'][0]} rows of {n['S-ob'][1]} cases, "
           "as REPORT.md 'Data and result files' states", "results/data_checksums.csv", "REPORT.md")


# ---------------------------------------------------------------------------- 7. public-data rule

def check_public_data(c: Check):
    tracked = [p for p in git("ls-files", "-z").split("\0") if p]
    csvs, bad = [p for p in tracked if p.endswith(".csv") and p.startswith(("results/", "docs/"))], []
    for p in csvs:
        with open(ROOT / p, encoding="utf-8") as f:
            first = f.readline()
            header = f.readline() if first.startswith("# ") else first
        bad += [f"{p}: {col}" for col in header.strip().split(",") if CUFF_COLUMN.match(col)]
    c.test(not bad, f"{len(csvs)} committed CSV files under results/ and docs/: no per-forecast cuff column (y_map, "
           f"last_map, y_sbp, last_sbp, ...){listed(bad)}", "git ls-files results docs")
    raw = [p for p in tracked if p.startswith("data/") or p.endswith((".vital", ".parquet", ".npz", ".npy"))]
    c.test(not raw, f"no committed file under data/, and no .vital, .parquet, .npz or .npy file{listed(raw)}",
           "git ls-files")
    c.add("NOTE", "Only CSV headers and file types are scanned; images, HTML and JSON files are not inspected.")


CHECKS = (("Registration and freeze order", check_registration), ("Manifests", check_manifests),
          ("Frozen models and the one cesarean scoring", check_frozen),
          ("Recomputation from the per-patient tables", check_recompute), ("README and REPORT", check_documents),
          ("Published data checksums", check_checksums), ("Public-data rule", check_public_data))


def main(argv=None) -> int:
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--root", type=Path, default=CODE, help="repository to check (default: the one of this script)")
    a = ap.parse_args(argv)
    ROOT = a.root.resolve()
    checks = []
    for i, (title, fn) in enumerate(CHECKS, 1):
        c = Check(i, title)
        try:
            fn(c)
        except Exception as ex:                             # a missing file or git object fails its check
            c.add("FAIL", f"could not complete: {type(ex).__name__}: {ex}")
        checks.append(c)
    ok = all(c.status != "FAIL" for c in checks)
    head = run_git("describe", "--tags", "--always", "--dirty").stdout.decode().strip()
    if a.json:
        print(json.dumps({"head": head, "status": "PASS" if ok else "FAIL",
                          "checks": [{"num": c.num, "title": c.title, "status": c.status, "items": c.items}
                                     for c in checks]}, indent=1, ensure_ascii=False))
        return 0 if ok else 1
    print(f"Repository at {head}")
    for c in checks:
        print(f"\n{c.status:7} {c.num}. {c.title}")
        for it in c.items:
            ev = f"\n          [{', '.join(it['evidence'])}]" if it["evidence"] else ""
            print(f"  {it['status']:7} {it['text']}{ev}")
    print(f"\n{'PASS' if ok else 'FAIL'}: " + ", ".join(f"{c.num} {c.status}" for c in checks))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
