"""scripts/reproduce.py: refs, guards, shared data folder, comparison and clean-up.

No network and no real data. The orchestration is run on a throwaway git repository whose scripts
stand in for the analysis scripts: they take the same command lines, apply the same guards (tag at
HEAD, a clean work tree, `results/frozen/FROZEN.json` committed at HEAD, `BTC_FROZEN_SCORING=1` for
S-ob and only there), and write outputs that record the git state, so a run at another commit or in a
dirty tree gives other bytes. The repository's history mirrors the real one: the "original" runs are
made in the main work tree at the same points (S-dev run at tag feature-freeze, public tables exported
at a later commit, freeze at the next commit, S-ob at tag models-frozen) and their outputs committed.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("script_reproduce", ROOT / "scripts" / "reproduce.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load()

GIT_ID = {"GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
          "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.invalid"}

FAKEPIPE = r'''
"""Stand-in for the analysis scripts (tests of scripts/reproduce.py only)."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
CASES = {"sdev": [1, 2, 3], "sob": [7, 8]}
TABLES = ("cases", "readings", "instances", "features_hr", "features_wf")
ENV = "BTC_FROZEN_SCORING"


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def state():
    return {"commit": git("rev-parse", "HEAD"), "describe": git("describe", "--tags", "--always", "--dirty"),
            "status": [x for x in git("status", "--porcelain", "--untracked-files=normal").splitlines() if x],
            "tags_at_head": git("tag", "--points-at", "HEAD").split()}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, text):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def cohort_access(cohort):
    if cohort == "sob" and os.environ.get(ENV) != "1":
        sys.exit("refused: S-ob signals only at the frozen scoring")
    if cohort == "sdev" and ENV in os.environ:
        sys.exit(f"refused: {ENV} is set outside the S-ob steps")


def tree_ok(st, allow_untracked_results):
    for x in st["status"]:
        if not (allow_untracked_results and x.startswith("?? results/")):
            sys.exit(f"refused: work tree not clean: {x}")


class Writer:
    def __init__(self, out, meta):
        self.out, self.meta, self.files = Path(out), meta, {}

    def write(self, rel, text):
        self.files[rel] = sha(write(self.out / rel, text))

    def manifest(self, rel):
        write(self.out / rel, json.dumps({"meta": self.meta, "files": dict(sorted(self.files.items()))}) + "\n")


def log(event, **kw):
    p = RESULTS / "sob" / "frozen_scoring_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({"time_utc": str(time.time_ns()), "event": event, **kw}) + "\n")


def request(what):
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    with (DATA / "raw" / "requests.csv").open("a") as f:
        f.write(what + "\n")


def download(cohort):
    cohort_access(cohort)
    raw = DATA / "raw"
    if not (raw / "meta" / "cases.csv").exists():
        write(raw / "meta" / "cases.csv", "caseid,cohort\n1,S-dev\n2,S-dev\n3,S-dev\n7,S-ob\n8,S-ob\n")
        request("table:cases")
    rows = []
    for c in CASES[cohort]:
        f = raw / cohort / f"{c}.npz"
        if not f.exists():
            write(f, f"signal of case {c}\n")
            request(f"case:{c}")
        rows += [f"{c},meta_json,<U64,1,{hashlib.sha256(str(time.time_ns()).encode()).hexdigest()}",
                 f"{c},x,float64,1,{sha(f)}"]
    write(raw / f"{cohort}_checksums.csv", "caseid,array,dtype,n,sha256\n" + "\n".join(rows) + "\n")


def build(cohort, names):
    cohort_access(cohort)
    st = state()
    tree_ok(st, False)
    raw = [sha(DATA / "raw" / cohort / f"{c}.npz") for c in CASES[cohort]]
    for n in names:
        write(DATA / "derived" / f"{cohort}_{n}.parquet", json.dumps({"git": st, "raw": raw, "table": n}))


def inputs(cohort):
    return {f"{cohort}_{n}.parquet": sha(DATA / "derived" / f"{cohort}_{n}.parquet") for n in TABLES}


def opt(args, name):
    return args[args.index(name) + 1] if name in args else None


def confirmatory(args):
    cohort_access("sdev")
    st, tag = state(), opt(args, "--feature-tag")
    if tag not in st["tags_at_head"] or any(not x.startswith("?? ") for x in st["status"]):
        sys.exit("refused: HEAD must carry the feature tag with no modified tracked file")
    out = RESULTS / "sdev"
    if (out / "confirmatory" / "summary.json").exists():
        sys.exit("refused: a confirmatory result exists")
    meta = {"commit": st["commit"], "describe": st["describe"], "feature_tag": tag, "inputs_sha256": inputs("sdev")}
    w = Writer(out, meta)
    w.write("confirmatory/summary.json", json.dumps({"meta": meta, "n": 3}))
    w.write("confirmatory/predictions_primary.csv", "# " + json.dumps(meta) + "\ncaseid,y_map\n1,80\n")
    if "--with-secondary" in args:
        w.write("secondary/summary.json", json.dumps({"meta": meta, "secondary": True}))
    w.manifest("MANIFEST.json")


def frozen_committed():
    p = "results/frozen/FROZEN.json"
    r = subprocess.run(["git", "rev-parse", "--verify", "-q", f"HEAD:{p}"], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0 and (ROOT / p).exists() and r.stdout.strip() == git("hash-object", p)


def frozen_scoring(args):
    cmd, st = args[0], state()
    if cmd == "freeze":
        cohort_access("sdev")
        tree_ok(st, True)
        if (RESULTS / "frozen").exists():
            sys.exit("refused: frozen/ exists")
        conf = json.loads((RESULTS / "sdev" / "confirmatory" / "summary.json").read_text())
        if conf["meta"]["inputs_sha256"] != inputs("sdev"):
            sys.exit("refused: the S-dev tables differ from the inputs of the confirmatory run")
        w = Writer(RESULTS, {"commit": st["commit"], "describe": st["describe"], "status": st["status"]})
        w.write("frozen/FROZEN.json", json.dumps({"models": "fake", "n": conf["n"]}))
        w.write("frozen/models.pkl", "model bytes " + st["commit"])
        w.manifest("frozen/MANIFEST.json")
        return
    tag = opt(args, "--frozen-tag")
    if tag not in st["tags_at_head"] or not frozen_committed():
        sys.exit("refused: HEAD must carry the frozen tag and frozen/FROZEN.json must be committed")
    tree_ok(st, True)
    cohort_access("sob")
    meta = {"commit": st["commit"], "describe": st["describe"], "status": st["status"], "frozen_tag": tag}
    if cmd == "score":
        state_p = RESULTS / "sob" / "frozen_scoring" / "STATE.json"
        if state_p.exists():
            sys.exit("refused: the frozen scoring has already run")
        log("started", commit=st["commit"])
        w = Writer(RESULTS / "sob", {**meta, "inputs_sha256": inputs("sob")})
        w.write("frozen_scoring/summary.json", json.dumps({"meta": meta, "n": 2}))
        w.write("frozen_scoring/predictions_primary.csv", "# " + json.dumps(meta) + "\ncaseid,y_map\n7,90\n")
        w.manifest("frozen_scoring/MANIFEST.json")
        write(state_p, json.dumps({"status": "completed", "commit": st["commit"]}) + "\n")
        log("completed", commit=st["commit"])
    elif cmd == "after":
        w = Writer(RESULTS / "sob", meta)
        w.write("after_frozen_scoring/summary.json", json.dumps({"meta": meta}))
        w.manifest("after_frozen_scoring/MANIFEST.json")
        log("after_frozen_scoring_written", commit=st["commit"])
    elif cmd == "series":
        p = write(DATA / "derived" / "site" / "cases.json", json.dumps({"meta": meta, "cases": CASES["sob"]}))
        log("site_cases_written", commit=st["commit"], cases_json_sha256=sha(p))


def export(args):
    st, res = state(), Path(opt(args, "--results"))
    out = Path(opt(args, "--out") or res)
    for rel in ("sdev/confirmatory", "sob/frozen_scoring"):
        src = res / rel / "predictions_primary.csv"
        if src.exists():
            head = {"commit": st["commit"], "describe": st["describe"], "source_sha256": sha(src)}
            t = write(out / rel / "per_patient_primary_wide.csv", "# " + json.dumps(head) + "\ncaseid,rmse\n")
            write(out / rel / "public_tables.json",
                  json.dumps({"meta": {"commit": st["commit"]}, "files": {t.name: sha(t)}}) + "\n")


def main(script, args):
    cohort = opt(args, "--cohort")
    {"download": lambda: download(cohort),
     "build_instances": lambda: build(cohort, ("cases", "readings", "instances", "features_hr")),
     "build_windows": lambda: build(cohort, ("windows",)),
     "build_features_wf": lambda: build(cohort, ("features_wf",)),
     "pleth_coverage": lambda: build(cohort, ("pleth_coverage",)),
     "run_confirmatory": lambda: confirmatory(args),
     "run_frozen_scoring": lambda: frozen_scoring(args),
     "export_public_tables": lambda: export(args)}[script]()
    return 0
'''

SCRIPT = ("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
          "import fakepipe\nsys.exit(fakepipe.main({name!r}, sys.argv[1:]))\n")
SCRIPTS = ("download", "build_instances", "build_windows", "build_features_wf", "pleth_coverage", "run_confirmatory",
           "run_frozen_scoring", "export_public_tables")


def git(repo, *args, env=None):
    e = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")} | GIT_ID | (env or {})
    return subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args], cwd=repo,
                          env=e, capture_output=True, text=True, check=True).stdout.strip()


def commit(repo, msg, *paths):
    git(repo, "add", "-A", *paths)
    git(repo, "commit", "-q", "--allow-empty", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def py(repo, name, *args, sob=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k != "BTC_FROZEN_SCORING"}
    if sob:
        env["BTC_FROZEN_SCORING"] = "1"
    subprocess.run([sys.executable, f"scripts/{name}.py", *args], cwd=repo, env=env, check=True)


def tree_digest(folder: Path) -> dict:
    return {str(p.relative_to(folder)): (hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir")
            for p in sorted(folder.rglob("*"))}


@pytest.fixture(scope="module")
def fake_repo(tmp_path_factory):
    """A repository with the shape of the real one; its main work tree holds the original data folder."""
    repo = tmp_path_factory.mktemp("fake") / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "fakepipe.py").write_text(FAKEPIPE)
    (repo / "scripts").mkdir()
    for n in SCRIPTS:
        (repo / "scripts" / f"{n}.py").write_text(SCRIPT.format(name=n))
    (repo / ".gitignore").write_text("data/\nbuild/\n__pycache__/\nresults/**/predictions_primary.csv\n")
    (repo / "pyproject.toml").write_text('[project]\nname = "fake"\nversion = "0"\nrequires-python = ">=3.10"\n'
                                         'dependencies = []\n')
    ff = commit(repo, "feature code")
    git(repo, "tag", "feature-freeze")
    for n in ("download", "build_instances", "build_windows", "build_features_wf", "pleth_coverage"):
        py(repo, n, "--cohort", "sdev")
    py(repo, "run_confirmatory", "--feature-tag", "feature-freeze", "--with-secondary")
    (repo / "NOTES").write_text("export code\n")
    export_sdev = commit(repo, "export code", "NOTES")
    py(repo, "export_public_tables", "--results", "results")
    sums = (repo / "data" / "raw" / "sdev_checksums.csv").read_text().splitlines()
    (repo / "results" / "data_checksums.csv").write_text(
        "cohort," + sums[0] + "\n" + "".join(f"S-dev,{x}\n" for x in sums[1:]))
    (repo / "results" / "sdev" / "data_checksums.csv").write_text("\n".join(sums) + "\n")
    commit(repo, "S-dev results")
    py(repo, "run_frozen_scoring", "freeze")
    freeze = git(repo, "rev-parse", "HEAD")
    mf = commit(repo, "frozen models")
    git(repo, "tag", "models-frozen")
    py(repo, "download", "--cohort", "sob", sob=True)
    for n in ("build_instances", "build_windows", "build_features_wf"):
        py(repo, n, "--cohort", "sob", sob=True)
    for c in ("score", "after", "series"):
        py(repo, "run_frozen_scoring", c, "--frozen-tag", "models-frozen", sob=True)
    (repo / "NOTES").write_text("export code, S-ob\n")
    export_sob = commit(repo, "export S-ob", "NOTES")
    tmp = repo / "build" / "export"
    py(repo, "export_public_tables", "--results", "results", "--out", str(tmp))
    for f in (tmp / "sob" / "frozen_scoring").iterdir():
        (repo / "results" / "sob" / "frozen_scoring" / f.name).write_bytes(f.read_bytes())
    sob_sums = (repo / "data" / "raw" / "sob_checksums.csv").read_text().splitlines()[1:]
    with (repo / "results" / "data_checksums.csv").open("a") as f:
        f.write("".join(f"S-ob,{x}\n" for x in sob_sums))
    head = commit(repo, "S-ob results")
    assert git(repo, "status", "--porcelain") == ""
    return {"repo": repo, "feature_freeze": ff, "export_sdev": export_sdev, "freeze": freeze, "models_frozen": mf,
            "export_sob": export_sob, "head": head}


def run(repo, *args):
    return R.main(list(args), repo=repo)


def report(workdir):
    return json.loads((workdir / "report.json").read_text())


def worktrees(repo):
    return [x for x in git(repo, "worktree", "list", "--porcelain").splitlines() if x.startswith("worktree ")]


def only_main(repo):
    return worktrees(repo) == [f"worktree {Path(repo).resolve()}"]


# ---------------------------------------------------------------------------------------- end to end
def test_full_run_is_identical_and_leaves_the_repository_unchanged(fake_repo, monkeypatch, capsys):
    repo = fake_repo["repo"]
    monkeypatch.setenv("BTC_FROZEN_SCORING", "1")      # must not reach the S-dev and freeze steps
    before = git(repo, "status", "--porcelain", "--ignored")
    assert run(repo) == 0                               # default workdir: build/reproduce, ignored by git
    out = capsys.readouterr().out
    assert "IDENTICAL" in out and "NOT IDENTICAL" not in out
    work = repo / "build" / "reproduce"
    rep = report(work)
    assert rep["summary"]["identical"] == rep["summary"]["compared"] > 0
    assert rep["main"]["commit"] == fake_repo["head"]
    assert {s: r["commit"] for s, r in rep["refs"].items()} == {
        "sdev": fake_repo["feature_freeze"], "freeze": fake_repo["freeze"], "sob": fake_repo["models_frozen"]}
    items = {r["item"]: r for r in rep["rows"]}
    for p in ("results/sdev/confirmatory/predictions_primary.csv", "results/sob/frozen_scoring/predictions_primary.csv"):
        assert items[p]["identical"] == "yes" and items[p]["note"] == "not committed"
    for p in ("results/sdev/MANIFEST.json", "results/frozen/MANIFEST.json", "results/frozen/FROZEN.json",
              "results/sob/frozen_scoring/MANIFEST.json", "results/sob/after_frozen_scoring/MANIFEST.json",
              "results/sob/frozen_scoring/STATE.json", "results/sob/frozen_scoring_log.jsonl",
              "data/derived/site/cases.json", "data/derived/sdev_cases.parquet", "data/derived/sob_features_wf.parquet",
              "results/sdev/confirmatory/public_tables.json", "results/sob/frozen_scoring/per_patient_primary_wide.csv"):
        assert items[p]["identical"] == "yes", p
    data = [r for r in rep["rows"] if r["kind"] == "data"]
    assert [r["identical"] for r in data] == ["yes", "yes"] and "meta_json" in data[0]["item"]
    # each step ran at its stage's commit; S-ob steps, and only they, with BTC_FROZEN_SCORING=1
    by = {(s["stage"], " ".join(s["argv"][1:3])): s for s in rep["steps"]}
    assert by[("sdev", "scripts/run_confirmatory.py --feature-tag")]["commit"] == fake_repo["feature_freeze"]
    assert by[("freeze", "scripts/run_frozen_scoring.py freeze")]["commit"] == fake_repo["freeze"]
    assert by[("sob", "scripts/run_frozen_scoring.py score")]["commit"] == fake_repo["models_frozen"]
    assert all((s["env"] == {"BTC_FROZEN_SCORING": "1"}) == (s["stage"] == "sob") for s in rep["steps"])
    exports = sorted(s["commit"] for s in rep["steps"] if s["stage"] == "export")
    assert exports == sorted([fake_repo["export_sdev"], fake_repo["export_sob"]])
    assert set(rep["elapsed_seconds"]) >= {"sdev", "freeze", "sob", "export", "total"}
    assert rep["published_record"] and all(r["identical"] == "yes" for r in rep["published_record"])
    assert "results/data_checksums.csv" in rep["not_compared"]
    # nothing committed or tracked changed, no worktree left, nothing written outside build/
    assert git(repo, "rev-parse", "HEAD") == fake_repo["head"] and git(repo, "status", "--porcelain") == ""
    assert only_main(repo)
    assert list((work / "worktrees").iterdir()) == []
    after = git(repo, "status", "--porcelain", "--ignored")
    assert set(after.splitlines()) - set(before.splitlines()) <= {"!! build/"}
    assert "# written by scripts/reproduce.py" in (work / "git" / "exclude").read_text()
    md = (work / "report.md").read_text()
    assert fake_repo["head"] in md and str(Path.home()) not in md and str(work) not in md


def test_reused_data_folder_is_only_read_and_skips_downloads(fake_repo, tmp_path):
    repo = fake_repo["repo"]
    src = repo / "data"                                  # the original data folder
    before = tree_digest(src)
    assert run(repo, "--stages", "sdev", "--data", str(src), "--workdir", str(tmp_path / "w")) == 0
    assert tree_digest(src) == before
    assert not (tmp_path / "w" / "data" / "raw" / "requests.csv").exists()   # nothing was "downloaded"
    links = list((tmp_path / "w" / "data" / "raw" / "sdev").iterdir())
    assert links and all(p.is_symlink() and p.resolve().parent == (src / "raw" / "sdev").resolve() for p in links)
    assert not (tmp_path / "w" / "data" / "raw" / "sob").exists()            # S-ob archives not linked for sdev


def test_freeze_alone_builds_the_sdev_tables_first(fake_repo, tmp_path):
    work = tmp_path / "w"
    assert run(fake_repo["repo"], "--stages", "freeze", "--workdir", str(work)) == 0
    rep = report(work)
    stages = [(s["stage"], s["argv"][1], s["commit"]) for s in rep["steps"]]
    assert ("sdev-data", "scripts/download.py", fake_repo["feature_freeze"]) in stages
    assert ("freeze", "scripts/run_frozen_scoring.py", fake_repo["freeze"]) in stages
    assert "sdev-data" in rep["elapsed_seconds"]
    assert not any(s == "export" for s, _, _ in stages)                      # no public table belongs to freeze
    # a second run finds the tables and does not rebuild them
    assert run(fake_repo["repo"], "--stages", "freeze", "--workdir", str(work)) == 0
    assert [s["stage"] for s in report(work)["steps"]] == ["freeze"]


def test_a_changed_published_hash_is_reported(fake_repo, tmp_path, capsys):
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(fake_repo["repo"]), str(clone))
    m = clone / "results" / "sdev" / "MANIFEST.json"
    d = json.loads(m.read_text())
    d["files"]["confirmatory/predictions_primary.csv"] = "0" * 64
    m.write_text(json.dumps(d) + "\n")
    commit(clone, "tampered")
    work = tmp_path / "w"
    assert run(clone, "--stages", "sdev", "--workdir", str(work)) == 1
    assert "NOT IDENTICAL" in capsys.readouterr().out
    rows = {r["item"]: r["identical"] for r in report(work)["rows"]}
    assert rows["results/sdev/confirmatory/predictions_primary.csv"] == "no"
    assert rows["results/sdev/MANIFEST.json"] == "no"
    assert rows["results/sdev/confirmatory/summary.json"] == "yes"


def test_keep_then_next_run_removes_the_kept_worktrees(fake_repo, tmp_path):
    repo, work = fake_repo["repo"], tmp_path / "w"
    assert run(repo, "--stages", "sdev", "--workdir", str(work), "--keep") == 0
    kept = sorted(p.name for p in (work / "worktrees").iterdir())
    assert kept == ["export-" + fake_repo["export_sdev"][:7], "feature-freeze"]    # the public tables are exported too
    assert (work / "worktrees" / "feature-freeze" / "data").is_symlink()
    assert len(worktrees(repo)) == 1 + len(kept)
    assert run(repo, "--stages", "freeze", "--workdir", str(work)) == 0
    assert only_main(repo)


def test_refusals(fake_repo, tmp_path, capsys):
    repo = fake_repo["repo"]
    assert run(repo, "--workdir", str(repo / "scripts" / "out")) == 2          # inside the repo, not ignored
    assert "not ignored" in capsys.readouterr().err
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(repo), str(clone))
    git(clone, "tag", "-d", "models-frozen")
    assert run(clone, "--stages", "sob", "--workdir", str(tmp_path / "w")) == 2
    assert "tag models-frozen does not point at" in capsys.readouterr().err
    shallow = tmp_path / "shallow"
    git(tmp_path, "clone", "-q", "--depth", "1", f"file://{repo}", str(shallow))
    assert run(shallow, "--workdir", str(tmp_path / "w2")) == 2
    assert "shallow" in capsys.readouterr().err
    assert only_main(repo)


# ---------------------------------------------------------------------------------------- units
def test_manifest_entry_paths():
    assert R.manifest_entry_path("results/sdev/MANIFEST.json", "confirmatory/a.csv") == "results/sdev/confirmatory/a.csv"
    assert R.manifest_entry_path("results/frozen/MANIFEST.json", "frozen/FROZEN.json") == "results/frozen/FROZEN.json"
    assert (R.manifest_entry_path("results/sob/frozen_scoring/MANIFEST.json", "frozen_scoring/sensitivity/x.csv")
            == "results/sob/frozen_scoring/sensitivity/x.csv")
    assert (R.manifest_entry_path("results/sob/after_frozen_scoring/MANIFEST.json", "after_frozen_scoring/s.json")
            == "results/sob/after_frozen_scoring/s.json")
    assert R.stage_of("results/sob/frozen_scoring_log.jsonl") == "sob" and R.stage_of("results/README.md") is None


def test_checksum_comparison_excludes_meta_json_and_filters_the_cohort():
    pub = ("cohort,caseid,array,dtype,n,sha256\nS-dev,1,meta_json,<U9,1,aaa\nS-dev,1,x,float64,3,bbb\n"
           "S-dev,2,x,float64,3,ccc\nS-ob,7,x,float64,3,ddd\n")
    got = "caseid,array,dtype,n,sha256\n1,meta_json,<U9,1,zzz\n1,x,float64,3,bbb\n2,x,float64,3,ccc\n"
    c = R.compare_checksums(R.checksum_rows(pub, "S-dev"), R.checksum_rows(got, None))
    assert c["ok"] and c["identical"] == 2 and c["n_cases"] == 2
    c = R.compare_checksums(R.checksum_rows(pub, "S-ob"), R.checksum_rows(got, None))
    assert not c["ok"] and c["missing"] == [(7, "x")] and c["extra"] == [(1, "x"), (2, "x")]
    bad = got.replace("ccc", "eee")
    c = R.compare_checksums(R.checksum_rows(pub, "S-dev"), R.checksum_rows(bad, None))
    assert c["differ"] == [(2, "x")] and not c["ok"]
    assert not R.compare_checksums({}, {})["ok"]                       # an empty list proves nothing


def test_published_checksum_list_fallbacks():
    class Pub:
        def __init__(self, files):
            self.files = files

        def read(self, p):
            return self.files.get(p)
    both = {R.CHECKSUMS: b"cohort,caseid,array,dtype,n,sha256\n", R.CHECKSUMS_SDEV: b"caseid,array,dtype,n,sha256\n"}
    assert R.published_checksums(Pub(both), "sob")[1] == R.CHECKSUMS
    assert R.published_checksums(Pub({R.CHECKSUMS_SDEV: both[R.CHECKSUMS_SDEV]}), "sdev")[1] == R.CHECKSUMS_SDEV
    assert R.published_checksums(Pub({R.CHECKSUMS_SDEV: both[R.CHECKSUMS_SDEV]}), "sob") == (None, None)


def test_normalized_log_ignores_only_the_times():
    a = '{"time_utc": "2026-01-01T00:00:00", "event": "started", "commit": "c"}\n{"time_utc": "x", "event": "completed"}\n'
    b = '{"event": "started", "time_utc": "2027-05-05T00:00:00", "commit": "c"}\n{"event": "completed", "time_utc": "y"}\n'
    assert R.normalized_log(a) == R.normalized_log(b.encode())
    assert R.normalized_log(a) != R.normalized_log(a.replace('"c"', '"d"'))


def test_version_problems():
    pp = (b'[project]\nrequires-python = "==3.13.*"\ndependencies = ["numpy==2.5.5", "Typing_Extensions==4.16.0"]\n'
          b'[tool.uv]\nconstraint-dependencies = ["pytest==9.2.0", "numpy==2.5.5"]\n')
    ok = {"numpy": "2.5.5", "typing-extensions": "4.16.0", "pytest": "9.2.0"}
    assert R.version_problems(pp, ok, "3.13.15") == []
    assert R.version_problems(pp, {**ok, "numpy": "2.5.6"}, "3.13.15") == ["numpy 2.5.6 installed, 2.5.5 pinned"]
    assert R.version_problems(pp, {"numpy": "2.5.5", "typing-extensions": "4.16.0"}, "3.13.0") == []  # dev tools optional
    assert R.version_problems(pp, {"numpy": "2.5.5"}, "3.12.9") == [
        "Python 3.12.9 (requires-python ==3.13.*)", "typing-extensions not installed (pinned 4.16.0)"]
    assert R.version_problems(None, {}, "3.13.0") == []


def test_git_env_appends_settings_and_drops_redirections(tmp_path):
    env = R.git_env({"GIT_DIR": "/elsewhere", "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "a.b",
                     "GIT_CONFIG_VALUE_0": "c", "PATH": os.environ["PATH"]}, tmp_path / "exclude")
    assert "GIT_DIR" not in env and env["GIT_CONFIG_COUNT"] == "3" and env["GIT_CONFIG_KEY_0"] == "a.b"
    assert (env["GIT_CONFIG_KEY_1"], env["GIT_CONFIG_VALUE_1"]) == ("core.excludesFile", str(tmp_path / "exclude"))
    assert (env["GIT_CONFIG_KEY_2"], env["GIT_CONFIG_VALUE_2"]) == ("core.abbrev", "7")


def test_the_exclude_hides_only_the_linked_data_folder(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / ".gitignore").write_text("data/\n")                   # matches a folder, not a symbolic link
    commit(repo, "x")
    (tmp_path / "shared").mkdir()
    (repo / "data").symlink_to(tmp_path / "shared", target_is_directory=True)
    plain = R.Git(repo)
    assert plain.status(repo) == ["?? data"]
    (tmp_path / "exclude").write_text("/data\n")
    g = R.Git(repo, R.git_env(dict(os.environ), tmp_path / "exclude"))
    assert g.status(repo) == []
    (repo / "other").write_text("y")
    assert g.status(repo) == ["?? other"]                          # anything else still counts


def test_parse_stages():
    assert R.parse_stages("all") == ["sdev", "freeze", "sob"]
    assert R.parse_stages("sob,sdev") == ["sdev", "sob"]             # always in the registered order
    with pytest.raises(Exception):
        R.parse_stages("sdev,ob")
