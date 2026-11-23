#!/usr/bin/env python3
"""Reproduce the registered analysis from the public VitalDB API and compare every result file with
the SHA-256 published in this repository (SPEC.md §8.3, §10).

From a clone of the repository at main, with uv (https://docs.astral.sh/uv/) and git:

    uv run --locked --group encoder scripts/reproduce.py

`uv run --locked --group encoder` creates the environment pinned in `uv.lock` (Python 3.13 and the exact
package versions of the original runs, with the CPU build of torch for the exploratory encoder check) and
runs this script in it. The run needs network access to https://api.vitaldb.net (about 1.8 GB are
downloaded and 0.4 GB of case archives are kept) and to https://zenodo.org (the 23-MB encoder weights),
and a POSIX system (Linux or macOS: it links folders with symbolic links).

What it runs
------------
Each stage runs the original scripts at the commit that wrote the published results, with the
arguments of the original run, in a temporary git worktree of this repository (`git worktree add
--detach`). The commits are read from the committed result files (MANIFEST.json meta, and the meta of
build_check.json), and each tag is checked to point at the recorded commit.

  sdev         tag feature-freeze, the commit in results/sdev/MANIFEST.json (SPEC §9 steps 2-5, §6.6-§6.8)
                 scripts/download.py --cohort sdev
                 scripts/build_instances.py --cohort sdev
                 scripts/build_windows.py --cohort sdev
                 scripts/build_features_wf.py --cohort sdev
                 scripts/pleth_coverage.py --cohort sdev        (descriptive; no published hash)
                 scripts/run_confirmatory.py --feature-tag feature-freeze --with-secondary
  freeze       the commit in results/frozen/MANIFEST.json (SPEC §9 step 6)
                 scripts/run_frozen_scoring.py freeze
  sob          tag models-frozen, the commit in results/sob/frozen_scoring/MANIFEST.json, every step with
               BTC_FROZEN_SCORING=1 (SPEC §9 steps 7-8)
                 scripts/download.py --cohort sob
                 scripts/build_instances.py --cohort sob
                 scripts/build_windows.py --cohort sob
                 scripts/build_features_wf.py --cohort sob
                 scripts/run_frozen_scoring.py score --frozen-tag models-frozen
                 scripts/run_frozen_scoring.py after --frozen-tag models-frozen
                 scripts/run_frozen_scoring.py series --frozen-tag models-frozen
  encoder      the exploratory encoder check on S-dev (ENCODER_PLAN.md §1-§6, SPEC §6.8), at the commits
               recorded in results/sdev/exploratory_encoder/MANIFEST.json (embeddings_built_at_commit,
               encoder_features_built_at_commit, the run's commit) and in the meta of build_check.json
                 scripts/download_encoder.py                    (the weights, from Zenodo)
                 scripts/build_embeddings.py --jobs <workers recorded in the MANIFEST meta>
                 scripts/build_features_enc.py
                 scripts/run_encoder_check.py build-check
                 scripts/run_encoder_check.py sdev --n-jobs 2
  encoder-sob  the S-ob step of the encoder check (ENCODER_PLAN.md §7), at the commit in
               results/sob/after_frozen_scoring/exploratory_encoder/MANIFEST.json, with BTC_FROZEN_SCORING=1
                 scripts/run_encoder_sob.py --jobs <workers recorded in the MANIFEST meta> --n-jobs 2

The encoder steps keep the worker counts of the original runs, because their result files record them.
`freeze`, `sob` and `encoder` read the S-dev derived tables: when the data folder does not hold tables equal
to the published ones (`inputs_sha256`), the S-dev part of `sdev` up to the tables runs first, at tag
feature-freeze. `encoder` reads the reproduced confirmatory forecasts (stage sdev) and `encoder-sob` the
reproduced frozen-scoring forecasts and S-ob tables (stage sob) and the S-dev encoder tables and weights
(stage encoder); a stage whose inputs are not already in `--workdir`, with the published SHA-256, is added
to the run. These per-forecast files are not committed (they hold VitalDB cuff values); the reproduced copy
is placed, git-ignored, in the worktree of the stage that reads it, where the original script checks its
SHA-256 against the MANIFEST. Each stage reads the committed results of the earlier stages at its own
commit (for example `freeze` reads the committed confirmatory summary), as the original runs did; each
stage's own outputs are compared.

After the stages, scripts/export_public_tables.py runs on the reproduced files at the commit recorded in
each committed public_tables.json (`--results <reproduced files> --out <folder>`). The encoder stages write
their public tables themselves (with that script's `export_stage`), as the original runs did.

Data folder and worktrees
-------------------------
All stages share one data folder, `<workdir>/data`, linked into each worktree as `data`. git runs with
two settings for this script and every script it starts (environment variables GIT_CONFIG_COUNT,
GIT_CONFIG_KEY_n, GIT_CONFIG_VALUE_n; nothing is written to any git configuration):
  core.excludesFile = <workdir>/git/exclude, which lists `/data`, so that the original guards (tag at
                      HEAD, no modified or untracked file) see a clean tree with the linked folder;
  core.abbrev = 7, the length of the abbreviated commits in the recorded `git describe` strings
                      (git's own default length grows with the number of objects in a clone).
Child processes also run with PYTHONUTF8=1 (the result files are written in the default encoding,
UTF-8 in the original runs), without PYTHONPATH, and with BTC_FROZEN_SCORING=1 only in `sob` and
`encoder-sob`. The download script keeps its own limits (at most 4 parallel cases, retries with backoff).

`--data DIR` reuses a data folder of an earlier run or download: the case archives `DIR/raw/<cohort>/`
and the encoder weights `DIR/external/papagei/papagei_s.pt` are linked, and the cached case table and
track list `DIR/raw/meta/` are copied into `<workdir>/data`, so the download scripts skip what is already
complete (the weights are checked against their size and MD5 first). DIR is only read, never written: the
derived tables and logs are rebuilt in `<workdir>/data`; encoder tables left there by an earlier run of
the same `--workdir` are removed before they are rebuilt (the original scripts build them once).

Nothing in the repository is changed. The worktrees, the data folder, the logs and the reproduced
files are under `--workdir` (default `build/reproduce`, which git ignores); the worktrees are removed
at the end (`git worktree remove`) unless `--keep`. A next run with the same `--workdir` removes worktrees
kept by an earlier one and reuses its data folder.

Comparison
----------
Against the files committed at HEAD (the clone's main), never against the working tree:
  - every file listed in a committed results/**/MANIFEST.json, with its SHA-256, including the files that
    are not committed (per-forecast files, which hold VitalDB cuff values), and the MANIFEST.json itself;
  - the derived tables, against `inputs_sha256` in the MANIFEST meta of the run that read them; the S-dev
    encoder embeddings and features and the weights, against the SHA-256 in the encoder MANIFEST meta;
  - the downloaded arrays, against the published checksum list (results/data_checksums.csv; for S-dev
    also results/sdev/data_checksums.csv), except the `meta_json` arrays, which hold the download time;
  - the committed files of a run that no MANIFEST lists: STATE.json byte for byte, the frozen-scoring
    log without its `time_utc` fields, the per-patient public tables and public_tables.json;
  - the site case files, against the SHA-256 of cases.json in the frozen-scoring log;
  - and, before the run, the committed copies against their MANIFEST entries.
The table printed at the end shows the first 12 hex digits; `<workdir>/report.md` and `report.json`
hold the full values, the commands, the refs, the elapsed times, the platform, the BLAS and the version
of every installed package. Exit status: 0 when every compared item is identical, 1 when any differs or
is missing, 2 when the run could not start (for example a package version differs from the pins of a
commit used, `--ignore-versions` runs anyway; or torch is missing for the encoder stages).

Result files hold no timestamps and every seed is fixed, so the same inputs, code and locked
environment give the same bytes on the same platform. On another CPU or numerical library the last
digits of floating-point results can differ, and the hashes with them (results/README.md); the
per-patient tables then allow the numbers to be compared. The encoder results also record the torch
build and settings of the embedding workers (`embedding_runtime` in their MANIFEST meta).

The cesarean cases (S-ob) were scored once, in the run whose files are published (SPEC §9 step 7), and
the S-ob step of the encoder check ran once (ENCODER_PLAN.md §7). The `sob` and `encoder-sob` stages
repeat those runs in worktrees, outside the repository's results/, to check the published files; they are
checks, not new results. The descriptive S-ob pleth coverage (scripts/pleth_coverage.py --cohort sob) was
run after the scoring at a later commit, publishes no hash, and is not run here.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
STAGES = ("sdev", "freeze", "sob", "encoder", "encoder-sob")
ENCODER_STAGES = ("encoder", "encoder-sob")
SOB_STAGES = ("sob", "encoder-sob")                 # run with BTC_FROZEN_SCORING=1
SELF_EXPORTING = ("encoder", "encoder-sob")         # write their public tables themselves
STAGE_DIR = {"sdev": "results/sdev", "freeze": "results/frozen", "sob": "results/sob",
             "encoder": "results/sdev/exploratory_encoder",
             "encoder-sob": "results/sob/after_frozen_scoring/exploratory_encoder"}
COHORT_LABEL = {"sdev": "S-dev", "sob": "S-ob"}

MANIFEST_SDEV = "results/sdev/MANIFEST.json"
MANIFEST_FROZEN = "results/frozen/MANIFEST.json"
MANIFEST_SCORE = "results/sob/frozen_scoring/MANIFEST.json"
MANIFEST_AFTER = "results/sob/after_frozen_scoring/MANIFEST.json"
MANIFEST_ENC = "results/sdev/exploratory_encoder/MANIFEST.json"
MANIFEST_ENC_SOB = "results/sob/after_frozen_scoring/exploratory_encoder/MANIFEST.json"
BUILD_CHECK = "results/sdev/exploratory_encoder/build_check.json"
STATE = "results/sob/frozen_scoring/STATE.json"
SCORING_LOG = "results/sob/frozen_scoring_log.jsonl"
SDEV_PREDICTIONS = "results/sdev/confirmatory/predictions_primary.csv"     # read by `encoder`
SOB_PREDICTIONS = "results/sob/frozen_scoring/predictions_primary.csv"     # read by `encoder-sob`
CHECKSUMS = "results/data_checksums.csv"            # both cohorts, column `cohort`
CHECKSUMS_SDEV = "results/sdev/data_checksums.csv"  # S-dev only, no `cohort` column
PUBLIC_TABLES = "public_tables.json"
NOT_COMPARED_ARRAY = "meta_json"                    # download metadata, includes the download time
SITE_CASES = "data/derived/site/cases.json"         # written by `run_frozen_scoring.py series`
WEIGHTS = "external/papagei/papagei_s.pt"           # in the data folder (scripts/download_encoder.py)
ENC_EMBEDDINGS = "sdev_enc_window_embeddings.npy"
ENC_FEATURES = "sdev_enc_features.parquet"
ENC_TABLES = {c: tuple(f"{c}_enc_{n}" for n in ("window_embeddings.npy", "window_index.parquet",
                                               "window_embeddings_log.json", "features.parquet",
                                               "features_log.json")) for c in ("sdev", "sob")}
ENC_N_JOBS = "2"                                    # --n-jobs of the original encoder runs (their default)
NOTES = {  # committed files under results/ that no rule here reproduces, and why
    "results/README.md": "documentation",
    CHECKSUMS: "the published checksum list; the downloaded arrays are compared with it",
    CHECKSUMS_SDEV: "the published S-dev checksum list; the downloaded arrays are compared with it",
}
STAGE_ORDER = {"sdev": 0, "sdev-data": 0, "freeze": 1, "sob": 2, "encoder": 3, "encoder-sob": 4}
GIT_SCRUB = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
             "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE", "GIT_PREFIX")
TORCH_MISSING = ("the encoder stages need torch, from the dependency group `encoder` of pyproject.toml: run "
                 "`uv run --locked --group encoder scripts/reproduce.py`, or leave them out with "
                 "--stages sdev,freeze,sob")
SHORT = 12


class Refused(RuntimeError):
    """The run cannot start or continue (exit status 2)."""


class StepFailed(RuntimeError):
    """A script of a stage exited with an error; the stage stops, the others run."""


# ----------------------------------------------------------------------------------------------
# git
# ----------------------------------------------------------------------------------------------

def git_env(base: dict, exclude_file: Path) -> dict:
    """`base` without the variables that would point git elsewhere, plus the two settings of this
    script, appended to any GIT_CONFIG_COUNT entries already set."""
    env = {k: v for k, v in base.items() if k not in GIT_SCRUB}
    n = int(env.get("GIT_CONFIG_COUNT") or 0)
    for key, value in (("core.excludesFile", str(exclude_file)), ("core.abbrev", "7")):
        env[f"GIT_CONFIG_KEY_{n}"], env[f"GIT_CONFIG_VALUE_{n}"] = key, value
        n += 1
    env["GIT_CONFIG_COUNT"] = str(n)
    return env


class Git:
    def __init__(self, repo: Path, env: dict | None = None):
        self.repo = repo
        self.env = env if env is not None else {k: v for k, v in os.environ.items() if k not in GIT_SCRUB}

    def run(self, *args, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
        r = subprocess.run(["git", *args], cwd=cwd or self.repo, env=self.env, capture_output=True)
        if check and r.returncode != 0:
            raise Refused(f"git {' '.join(map(str, args))} failed: {r.stderr.decode(errors='replace').strip()}")
        return r

    def out(self, *args, cwd: Path | None = None) -> str:
        return self.run(*args, cwd=cwd).stdout.decode().strip()

    def blob(self, commit: str, path: str) -> bytes | None:
        """The committed bytes of `path` at `commit`, or None."""
        r = self.run("cat-file", "blob", f"{commit}:{path}", check=False)
        return r.stdout if r.returncode == 0 else None

    def commit_of(self, ref: str) -> str | None:
        r = self.run("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
        return r.stdout.decode().strip() if r.returncode == 0 else None

    def files(self, commit: str, prefix: str) -> list[str]:
        out = self.run("ls-tree", "-r", "-z", "--name-only", commit, "--", prefix).stdout.decode()
        return [p for p in out.split("\0") if p]

    def status(self, cwd: Path) -> list[str]:
        """`git status --porcelain` lines, as the original guards read them (untracked files count)."""
        out = self.run("status", "--porcelain", "--untracked-files=normal", cwd=cwd).stdout.decode()
        return [x for x in out.splitlines() if x]


# ----------------------------------------------------------------------------------------------
# The published record (files committed at HEAD)
# ----------------------------------------------------------------------------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Published:
    def __init__(self, git: Git, commit: str):
        self.git, self.commit = git, commit
        self.tree = set(git.files(commit, "results"))
        self._cache: dict[str, bytes | None] = {}

    def read(self, path: str) -> bytes | None:
        if path not in self._cache:
            self._cache[path] = self.git.blob(self.commit, path) if path in self.tree else None
        return self._cache[path]

    def json(self, path: str) -> dict | None:
        b = self.read(path)
        return None if b is None else json.loads(b)

    def meta(self, path: str) -> dict:
        return (self.json(path) or {}).get("meta") or {}

    def manifests(self) -> list[str]:
        return sorted(p for p in self.tree if PurePosixPath(p).name == "MANIFEST.json")

    def entry_hash(self, manifest: str, path: str) -> str | None:
        """The SHA-256 that `manifest` lists for the published path `path`."""
        for k, h in ((self.json(manifest) or {}).get("files") or {}).items():
            if manifest_entry_path(manifest, k) == path:
                return h
        return None


def manifest_entry_path(manifest: str, key: str) -> str:
    """Published path of a MANIFEST entry. The keys are relative to the folder the run wrote to: the
    MANIFEST's own folder (results/sdev: `confirmatory/...`), or its parent when the key starts with the
    folder's name (results/frozen: `frozen/FROZEN.json`; results/sob/frozen_scoring: `frozen_scoring/...`)."""
    d = PurePosixPath(manifest).parent
    return str((d.parent if PurePosixPath(key).parts[0] == d.name else d) / key)


def under(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder + "/")


def stage_of(path: str) -> str | None:
    """The stage whose output folder holds `path` (the deepest folder: the encoder folders lie inside
    results/sdev and results/sob)."""
    hits = [s for s, d in STAGE_DIR.items() if under(path, d)]
    return max(hits, key=lambda s: len(STAGE_DIR[s])) if hits else None


def nested_dirs(stage: str) -> list[str]:
    """Output folders of other stages inside this stage's folder (kept apart when files are copied)."""
    d = STAGE_DIR[stage]
    return [x for s, x in STAGE_DIR.items() if s != stage and x != d and under(x, d)]


@dataclasses.dataclass
class Ref:
    stage: str
    commit: str
    tag: str | None
    source: str                     # where the commit is recorded
    extra: dict = dataclasses.field(default_factory=dict)    # other commits of the stage, by step

    def label(self) -> str:
        return f"tag {self.tag} ({self.commit[:SHORT]})" if self.tag else f"commit {self.commit[:SHORT]}"

    def commits(self) -> dict:
        return {"run": self.commit, **self.extra}


def stage_refs(git: Git, pub: Published, stages: list[str]) -> dict[str, Ref]:
    """The commit of each stage, from the meta of its published result; a tag must point at it."""
    spec = {"sdev": (MANIFEST_SDEV, "feature_tag"), "freeze": (MANIFEST_FROZEN, None),
            "sob": (MANIFEST_SCORE, "frozen_tag"), "encoder": (MANIFEST_ENC, None),
            "encoder-sob": (MANIFEST_ENC_SOB, None)}

    def known(commit, where):
        if git.commit_of(commit) != commit:
            raise Refused(f"commit {commit} ({where}) is not in this clone; a shallow clone needs "
                          "`git fetch --unshallow --tags`")
    refs = {}
    for s in stages:
        path, tag_key = spec[s]
        if pub.json(path) is None:
            raise Refused(f"{path} is not committed at HEAD: there is no published {s} run to reproduce")
        meta = pub.meta(path)
        commit, tag = meta.get("commit"), meta.get(tag_key) if tag_key else None
        if not commit:
            raise Refused(f"{path} records no commit")
        known(commit, path)
        if tag_key:
            if not tag:
                raise Refused(f"{path} records no {tag_key}")
            if git.commit_of(f"refs/tags/{tag}") != commit:
                raise Refused(f"tag {tag} does not point at {commit}, the commit recorded in {path} "
                              "(fetch the tags with `git fetch --tags`)")
        extra = {}
        if s == "encoder":
            extra = {"embeddings": meta.get("embeddings_built_at_commit"),
                     "features": meta.get("encoder_features_built_at_commit"),
                     "build-check": pub.meta(BUILD_CHECK).get("commit")}
            for step, c in extra.items():
                if not c:
                    raise Refused(f"no commit recorded for the encoder {step} step ({path}, {BUILD_CHECK})")
                known(c, f"encoder {step}")
        refs[s] = Ref(s, commit, tag, path, extra)
    if "sob" in refs:
        after = pub.meta(MANIFEST_AFTER).get("commit")
        if after and after != refs["sob"].commit:
            raise Refused(f"{MANIFEST_AFTER} records commit {after}, {MANIFEST_SCORE} {refs['sob'].commit}")
    return refs


# ----------------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------------

def pins(pyproject: bytes) -> tuple[str | None, dict[str, str], set[str]]:
    """(requires-python, {package: exact version}, direct dependencies) of a pyproject.toml: the exact
    pins (`name==version`) of `dependencies` and of `[tool.uv] constraint-dependencies`."""
    d = tomllib.loads(pyproject.decode())
    proj = d.get("project", {})

    def exact(specs):
        out = {}
        for spec in specs:
            spec = spec.split(";")[0]                      # no environment markers here
            if "==" in spec:
                name, v = spec.split("==", 1)
                out[norm(name.split("[")[0])] = v.strip()
        return out
    direct = exact(proj.get("dependencies", []))
    allp = {**exact(d.get("tool", {}).get("uv", {}).get("constraint-dependencies", [])), **direct}
    return proj.get("requires-python"), allp, set(direct)


def norm(name: str) -> str:
    """Package name as PEP 503 compares it."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def installed() -> dict[str, str]:
    import importlib.metadata as md
    out = {}
    for d in md.distributions():
        n = d.metadata["Name"]
        if n:
            out[norm(n)] = d.version
    return out


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def python_ok(spec: str | None, version: str) -> bool:
    if not spec:
        return True
    try:
        from packaging.specifiers import SpecifierSet
        return version in SpecifierSet(spec)
    except ImportError:                       # a plain "==3.13.*" without packaging
        s = spec.strip()
        return not s.startswith("==") or version.startswith(s[2:].rstrip("*").rstrip("."))


def version_problems(pyproject: bytes | None, have: dict[str, str], py: str) -> list[str]:
    if pyproject is None:
        return []
    req, allp, direct = pins(pyproject)
    probs = [] if python_ok(req, py) else [f"Python {py} (requires-python {req})"]
    for name, v in sorted(allp.items()):
        got = have.get(name)
        if got is None and name in direct:
            probs.append(f"{name} not installed (pinned {v})")
        # PEP 440: a pin without a local label (`==2.15.0`) is met by a local build (`2.15.0+cpu`)
        elif got is not None and got != v and got.split("+", 1)[0] != v:
            probs.append(f"{name} {got} installed, {v} pinned")
    return probs


def recorded_workers(meta: dict, source: str) -> str:
    """The embedding worker count of an original encoder run (`embedding_runtime.workers` in its meta)."""
    w = (meta.get("embedding_runtime") or {}).get("workers")
    if not isinstance(w, int) or w < 1:
        raise StepFailed(f"{source} records no embedding worker count (embedding_runtime.workers)")
    return str(w)


def platform_record() -> dict:
    rec = {"python": platform.python_version(), "implementation": platform.python_implementation(),
           "platform": platform.platform(), "machine": platform.machine(),
           "processor": platform.processor() or None, "cpu_count": os.cpu_count(),
           "byteorder": sys.byteorder}
    try:
        import numpy  # noqa: F401  (loads the BLAS so that threadpoolctl can see it)
        import scipy.linalg  # noqa: F401
        from threadpoolctl import threadpool_info
        rec["threadpools"] = [{k: v for k, v in d.items() if k != "filepath"} for d in threadpool_info()]
    except Exception as e:                    # recorded, not fatal
        rec["threadpools"] = f"unavailable ({type(e).__name__})"
    try:
        rec["git"] = subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip()
    except OSError:
        rec["git"] = None
    return rec


# ----------------------------------------------------------------------------------------------
# Data checks
# ----------------------------------------------------------------------------------------------

def checksum_rows(text: str, cohort_label: str | None) -> dict[tuple[int, str], tuple[str, str, str]]:
    """{(caseid, array): (dtype, n, sha256)} of a checksum list, for one cohort when the list has a
    `cohort` column, without the `meta_json` arrays."""
    out = {}
    for r in csv.DictReader(io.StringIO(text)):
        if "cohort" in r and cohort_label is not None and r["cohort"] != cohort_label:
            continue
        if r["array"] == NOT_COMPARED_ARRAY:
            continue
        out[(int(r["caseid"]), r["array"])] = (r["dtype"], str(int(r["n"])), r["sha256"])
    return out


def compare_checksums(expected: dict, got: dict) -> dict:
    same = [k for k in expected if got.get(k) == expected[k]]
    differ = sorted(k for k in expected if k in got and got[k] != expected[k])
    missing = sorted(k for k in expected if k not in got)
    extra = sorted(k for k in got if k not in expected)
    return {"n_expected": len(expected), "n_cases": len({k[0] for k in expected}), "identical": len(same),
            "differ": differ, "missing": missing, "extra": extra,
            "ok": not (differ or missing or extra) and len(expected) > 0}


def published_checksums(pub: Published, cohort: str) -> tuple[str | None, str | None]:
    """(list text, its path) for a cohort: results/data_checksums.csv, else (S-dev only) the S-dev list."""
    for path in ((CHECKSUMS, CHECKSUMS_SDEV) if cohort == "sdev" else (CHECKSUMS,)):
        b = pub.read(path)
        if b is not None:
            return b.decode(), path
    return None, None


def normalized_log(text: str | bytes | None) -> bytes | None:
    """The frozen-scoring log without its `time_utc` fields, one JSON object per line."""
    if text is None:
        return None
    if isinstance(text, bytes):
        text = text.decode()
    lines = []
    for line in text.splitlines():
        if line.strip():
            d = json.loads(line)
            d.pop("time_utc", None)
            lines.append(json.dumps(d, sort_keys=True))
    return ("\n".join(lines) + "\n").encode()


# ----------------------------------------------------------------------------------------------
# Rows of the comparison
# ----------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Row:
    stage: str
    item: str
    expected: str | None            # SHA-256, or a count for the data rows
    got: str | None                 # None: not produced
    note: str = ""
    ok: bool | None = None          # None: not checked; set from expected == got unless given
    kind: str = "file"              # file or data

    def __post_init__(self):
        if self.ok is None and self.expected is not None:
            self.ok = self.got is not None and self.got == self.expected

    def word(self) -> str:
        return "n/a" if self.ok is None else ("yes" if self.ok else ("missing" if self.got is None else "no"))


# ----------------------------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------------------------

class Run:
    def __init__(self, a, repo: Path):
        self.a, self.repo = a, repo
        self.work = Path(a.workdir if a.workdir is not None else repo / "build" / "reproduce").resolve()
        self.data = self.work / "data"
        self.out = self.work / "out"               # reproduced result files, at their published paths
        self.wt_root = self.work / "worktrees"
        self.logs = self.work / "logs"
        self.exclude = self.work / "git" / "exclude"
        self.env = git_env(dict(os.environ), self.exclude)
        self.git = Git(repo, self.env)
        self.py = sys.executable
        self.stages: list[str] = []
        self.added: dict[str, str] = {}            # stage added to the run -> why
        self.worktrees: dict[str, Path] = {}
        self.steps: list[dict] = []
        self.times: dict[str, float] = {}
        self.failed: dict[str, str] = {}
        self.rows: list[Row] = []
        self.export_dirs: dict[str, Path] = {}
        self.not_compared: dict[str, str] = {}
        self.version_problems: dict[str, list[str]] = {}
        self.data_details: dict[str, dict] = {}
        self.sdev_state: str | None = None        # None, "built" or "failed": the S-dev data build of this run
        self.n_step = 0
        self._lock = None

    # -- set-up -----------------------------------------------------------------------------
    def prepare(self, stages: list[str]):
        g = self.git
        top = Path(g.out("rev-parse", "--show-toplevel")).resolve()
        if top != self.repo.resolve():
            raise Refused(f"{self.repo} is not the top level of its git work tree ({top})")
        if g.out("rev-parse", "--is-shallow-repository") == "true":
            raise Refused("this is a shallow clone; run `git fetch --unshallow --tags` first")
        self.head = g.out("rev-parse", "HEAD")
        self.head_describe = g.out("describe", "--tags", "--always", "--dirty")
        self.head_branch = g.run("symbolic-ref", "--short", "-q", "HEAD", check=False).stdout.decode().strip() or None
        self.pub = Published(g, self.head)
        self._check_workdir()
        self.stages, self.added = self.close_stages(stages)
        if any(s in ENCODER_STAGES for s in self.stages) and not torch_available():
            raise Refused(TORCH_MISSING)
        self.refs = stage_refs(g, self.pub, self.stages)
        self.work.mkdir(parents=True, exist_ok=True)
        self._take_lock()
        self.exclude.parent.mkdir(parents=True, exist_ok=True)
        self.exclude.write_text("# written by scripts/reproduce.py: the shared data folder, linked into each "
                                "worktree\n/data\n")
        for d in (self.data / "raw" / "meta", self.data / "derived", self.out, self.wt_root, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        self._remove_old_worktrees()

    def _check_workdir(self):
        try:
            rel = self.work.relative_to(self.repo.resolve())
        except ValueError:
            return                                   # outside the repository
        if not rel.parts:
            raise Refused("--workdir must not be the repository root")
        r = subprocess.run(["git", "check-ignore", "-q", "--no-index", str(rel) + "/"], cwd=self.repo,
                           env=self.env, capture_output=True)
        if r.returncode != 0:
            raise Refused(f"--workdir {rel} is inside the repository and not ignored by git; use build/... "
                          "or a folder outside the repository")

    def _take_lock(self):
        try:
            import fcntl
        except ImportError:
            return
        self._lock = open(self.work / ".lock", "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Refused("another run uses this --workdir")

    def _remove_old_worktrees(self):
        self.git.run("worktree", "prune")
        for p in self._registered_worktrees():
            if self.wt_root in p.parents:
                print(f"removing the worktree of an earlier run: {p.relative_to(self.work)}")
                self._remove_worktree(p)
        left = [p.name for p in self.wt_root.iterdir()]
        if left:
            raise Refused(f"{self.wt_root} holds entries that are not worktrees of this repository: {left}; "
                          "remove them or use another --workdir")

    def _registered_worktrees(self) -> list[Path]:
        txt = self.git.out("worktree", "list", "--porcelain")
        return [Path(line[len("worktree "):]).resolve() for line in txt.splitlines() if line.startswith("worktree ")]

    def _remove_worktree(self, p: Path):
        link = p / "data"
        if link.is_symlink():
            link.unlink()
        self.git.run("worktree", "remove", "--force", str(p), check=False)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
        self.git.run("worktree", "prune", check=False)

    def check_versions(self, commits: dict[str, str]):
        have, py = installed(), platform.python_version()
        probs = {}
        for label, c in commits.items():
            p = version_problems(self.git.blob(c, "pyproject.toml"), have, py)
            if p:
                probs[label] = p
        self.version_problems = probs
        if probs:
            lines = [f"  {label}: " + "; ".join(p) for label, p in probs.items()]
            msg = ("the environment differs from the pins of a commit used (pyproject.toml):\n" + "\n".join(lines) +
                   "\nCreate the locked environment (`uv run --locked --group encoder scripts/reproduce.py`), or "
                   "pass --ignore-versions to run anyway (the hashes may then differ).")
            if not self.a.ignore_versions:
                raise Refused(msg)
            print("warning: " + msg, file=sys.stderr)

    # -- inputs a stage reads from another stage ----------------------------------------------
    def _tables_ok(self, manifest: str, cohort: str) -> bool:
        exp = self.pub.meta(manifest).get("inputs_sha256") or {}
        return (bool(exp) and (self.data / "derived" / f"{cohort}_windows.parquet").is_file()
                and all(sha256_file(self.data / "derived" / n) == h for n, h in exp.items()))

    def _reproduced_ok(self, path: str, manifest: str) -> bool:
        h = self.pub.entry_hash(manifest, path)
        return h is not None and sha256_file(self.out / path) == h

    def _encoder_tables_ok(self) -> bool:
        """The S-dev encoder tables of an earlier run in this workdir, and the weights (here or in --data),
        with the SHA-256 of the encoder MANIFEST meta."""
        m, d, src = self.pub.meta(MANIFEST_ENC), self.data / "derived", self._data_src()
        want = {d / ENC_EMBEDDINGS: m.get("embeddings_sha256"), d / ENC_FEATURES: m.get("encoder_features_sha256")}
        weights = (m.get("weights") or {}).get("sha256")
        if None in want.values() or weights is None or not all((d / n).is_file() for n in ENC_TABLES["sdev"]):
            return False
        if any(sha256_file(f) != h for f, h in want.items()):
            return False
        return weights in {sha256_file(self.data / WEIGHTS), sha256_file(src / WEIGHTS) if src else None}

    def needs(self, stage: str) -> list[tuple[str, bool, str]]:
        """(stage that writes it, already in the workdir with its published SHA-256, what) for every input
        that `stage` reads from another stage's outputs (the S-dev derived tables aside: see
        `ensure_sdev_tables`)."""
        if stage == "encoder":
            return [("sdev", self._reproduced_ok(SDEV_PREDICTIONS, MANIFEST_SDEV), SDEV_PREDICTIONS)]
        if stage == "encoder-sob":
            sob_ok = self._reproduced_ok(SOB_PREDICTIONS, MANIFEST_SCORE) and self._tables_ok(MANIFEST_SCORE, "sob")
            return [("sob", sob_ok,
                     f"{SOB_PREDICTIONS} and the S-ob derived tables"),
                    ("encoder", self._encoder_tables_ok(), "the S-dev encoder tables and the weights")]
        return []

    def close_stages(self, stages: list[str]) -> tuple[list[str], dict[str, str]]:
        """`stages` with every stage that writes an input they need and that is not in the workdir yet."""
        have, added = set(stages), {}
        changed = True
        while changed:
            changed = False
            for s in [x for x in STAGES if x in have]:
                for dep, ok, what in self.needs(s):
                    if dep not in have and not ok:
                        have.add(dep)
                        added[dep] = f"stage {s} reads {what}, which this workdir does not hold"
                        changed = True
        return [x for x in STAGES if x in have], added

    # -- worktrees and steps ------------------------------------------------------------------
    def worktree(self, label: str, commit: str) -> Path:
        if label in self.worktrees:
            return self.worktrees[label]
        p = self.wt_root / label
        self.git.run("worktree", "add", "--quiet", "--detach", str(p), commit)
        self.worktrees[label] = p
        if (p / "data").exists() or (p / "data").is_symlink():
            raise Refused(f"the commit {commit[:SHORT]} has a path `data`; it cannot hold the linked data folder")
        (p / "data").symlink_to(self.data, target_is_directory=True)
        st = self.git.status(p)
        if st:
            raise Refused(f"the worktree at {commit[:SHORT]} is not clean: {st[:5]}")
        if self.git.out("rev-parse", "HEAD", cwd=p) != commit:
            raise Refused(f"the worktree {label} is not at {commit}")
        return p

    def close(self):
        for label, p in self.worktrees.items():
            if self.a.keep:
                print(f"kept worktree: {p}")
            else:
                self._remove_worktree(p)
        if self._lock is not None:
            self._lock.close()

    def child_env(self, sob: bool) -> dict:
        env = dict(self.env)
        env.pop("PYTHONPATH", None)
        env.pop("BTC_FROZEN_SCORING", None)
        env["PYTHONUTF8"] = "1"
        if sob:
            env["BTC_FROZEN_SCORING"] = "1"
        return env

    def _shown(self, x: str) -> str:
        """An argument as recorded: `python` for the interpreter, `<workdir>/...` for paths in the workdir."""
        if x == self.py:
            return "python"
        if os.path.isabs(x):
            try:
                return "<workdir>/" + Path(x).resolve().relative_to(self.work).as_posix()
            except ValueError:
                pass
        return x

    def step(self, stage: str, commit: str, tag: str | None, wt: Path, script: str, *args: str, sob: bool = False):
        argv = [self.py, f"scripts/{script}", *[str(x) for x in args]]
        self.n_step += 1
        sub = f"-{args[0]}" if args and not args[0].startswith("-") else ""
        name = f"{self.n_step:02d}-{stage}-{script.removesuffix('.py')}{sub}"
        log = self.logs / f"{name}.log"
        shown = " ".join(self._shown(x) for x in argv)
        where = f"tag {tag} ({commit[:SHORT]})" if tag else f"commit {commit[:SHORT]}"
        print(f"\n--> [{stage} @ {where}] {'BTC_FROZEN_SCORING=1 ' if sob else ''}{shown}", flush=True)
        t0 = time.time()
        with log.open("w", encoding="utf-8") as f, subprocess.Popen(
                argv, cwd=wt, env=self.child_env(sob), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                encoding="utf-8", errors="replace") as p:
            for line in p.stdout:
                sys.stdout.write("    " + line)
                f.write(line)
            code = p.wait()
        sec = time.time() - t0
        self.steps.append({"stage": stage, "commit": commit, "tag": tag, "argv": [self._shown(x) for x in argv],
                           "env": {"BTC_FROZEN_SCORING": "1"} if sob else {}, "exit": code,
                           "seconds": round(sec, 1), "log": f"logs/{log.name}"})
        print(f"    (exit {code}, {sec:.1f} s)", flush=True)
        if code != 0:
            raise StepFailed(f"{shown} exited with {code} (log: {log.relative_to(self.work)})")

    def wipe(self, stage: str):
        """Remove the reproduced files of `stage` from an earlier run (not those of a nested stage)."""
        dst, keep = self.out / STAGE_DIR[stage], nested_dirs(stage)
        if not dst.exists():
            return
        for f in sorted(dst.rglob("*"), reverse=True):          # a folder's files come before the folder
            r = f.relative_to(self.out).as_posix()
            if any(under(r, k) for k in keep):
                continue
            if f.is_dir() and not f.is_symlink():
                if not any(f.iterdir()):
                    f.rmdir()
            else:
                f.unlink()

    def collect(self, wt: Path, stage: str):
        """Copy the files the stage wrote in its worktree (untracked there) to <workdir>/out/, at their paths."""
        rel, keep = STAGE_DIR[stage], nested_dirs(stage)
        src = wt / rel
        if not src.exists():
            return
        tracked = set(self.git.out("ls-files", "--", rel, cwd=wt).splitlines())
        for f in sorted(src.rglob("*")):
            r = f.relative_to(wt).as_posix()
            if f.is_file() and r not in tracked and not any(under(r, k) for k in keep):
                (self.out / r).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, self.out / r)

    def place(self, wt: Path, path: str, manifest: str):
        """Put the reproduced, git-ignored file `path` (read by the stage at `wt`) into the worktree. The
        original script checks it against the SHA-256 in `manifest`; a difference is reported here first."""
        src = self.out / path
        if not src.is_file():
            raise StepFailed(f"{path} was not reproduced in this workdir")
        if sha256_file(src) != self.pub.entry_hash(manifest, path):
            print(f"warning: the reproduced {path} differs from its SHA-256 in {manifest}")
        (wt / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, wt / path)
        if self.git.status(wt):
            raise Refused(f"{path} is not git-ignored at {self.git.out('rev-parse', 'HEAD', cwd=wt)[:SHORT]}")

    # -- data -----------------------------------------------------------------------------------
    def _link(self, s: Path, d: Path) -> bool:
        """Link `d` to the file `s` unless `d` is a file of this workdir (downloaded by an earlier run)."""
        if d.is_symlink():
            d.unlink()
        elif d.exists():
            return False
        d.parent.mkdir(parents=True, exist_ok=True)
        d.symlink_to(s.resolve())
        return True

    def _data_src(self) -> Path | None:
        if self.a.data is None:
            return None
        src = Path(self.a.data).resolve()
        return None if src == self.data.resolve() else src

    def link_cached(self, cohort: str):
        """--data DIR: copy DIR/raw/meta/{cases,trks}.csv if absent here, link DIR/raw/<cohort>/*.npz."""
        src = self._data_src()
        if src is None:
            return
        for name in ("cases.csv", "trks.csv"):
            s, d = src / "raw" / "meta" / name, self.data / "raw" / "meta" / name
            if s.is_file() and not d.exists():
                shutil.copy2(s, d)
        sdir, ddir = src / "raw" / cohort, self.data / "raw" / cohort
        ddir.mkdir(parents=True, exist_ok=True)
        n = sum(self._link(s, ddir / s.name) for s in (sorted(sdir.glob("*.npz")) if sdir.is_dir() else []))
        print(f"linked {n} {COHORT_LABEL[cohort]} case archives from --data (read only)")

    def link_weights(self):
        """--data DIR: link DIR/external/papagei/papagei_s.pt (scripts/download_encoder.py checks it)."""
        src = self._data_src()
        if src is not None and (src / WEIGHTS).is_file() and self._link(src / WEIGHTS, self.data / WEIGHTS):
            print(f"linked the encoder weights from --data (read only): {WEIGHTS}")

    def remove_encoder_tables(self, cohort: str):
        """The original scripts build the encoder tables once and refuse to overwrite them."""
        old = [n for n in ENC_TABLES[cohort] if (self.data / "derived" / n).exists()]
        for n in old:
            (self.data / "derived" / n).unlink()
        if old:
            print(f"removed the {COHORT_LABEL[cohort]} encoder tables of an earlier run of this workdir: {old}")

    def check_data(self, stage: str, cohort: str):
        label = COHORT_LABEL[cohort]
        text, path = published_checksums(self.pub, cohort)
        got_p = self.data / "raw" / f"{cohort}_checksums.csv"
        item = f"{label} arrays: data/raw/{cohort}_checksums.csv vs {path or CHECKSUMS} (without meta_json)"
        if text is None:
            print(f"note: no published {label} checksum list at HEAD ({CHECKSUMS}); the {label} data are not checked")
            self.rows.append(Row(stage, item, None, None, note="no published list at HEAD", kind="data"))
            return
        exp = checksum_rows(text, label)
        got = checksum_rows(got_p.read_text(), None) if got_p.is_file() else {}
        c = compare_checksums(exp, got)
        self.data_details[cohort] = {"list": path, **{k: (v if not isinstance(v, list) else [list(x) for x in v])
                                                       for k, v in c.items()}}
        got_s = (f"{c['identical']} identical, {len(c['differ'])} differ, {len(c['missing'])} missing, "
                 f"{len(c['extra'])} extra")
        self.rows.append(Row(stage, item, f"{c['n_expected']} arrays of {c['n_cases']} cases", got_s,
                             ok=c["ok"], kind="data"))
        print(f"data check, {label}: {got_s} (of {c['n_expected']} arrays published in {path})")
        for k in (c["differ"] + c["missing"] + c["extra"])[:10]:
            print(f"    case {k[0]} array {k[1]}: published {exp.get(k)}, downloaded {got.get(k)}")

    def check_files(self, stage: str, expected: dict[str, str | None], source: str):
        """Rows for files of the data folder against the SHA-256 recorded in `source`."""
        rows = [Row(stage, f"data/{rel}", h, sha256_file(self.data / rel), note=source) for rel, h in expected.items()]
        self.rows += rows
        bad = [r.item for r in rows if not r.ok]
        print(f"{len(rows) - len(bad)} of {len(rows)} files equal to the SHA-256 in {source}"
              + (f"; different: {bad}" if bad else ""))

    def check_tables(self, stage: str, manifest: str):
        exp = self.pub.meta(manifest).get("inputs_sha256") or {}
        self.check_files(stage, {f"derived/{n}": h for n, h in sorted(exp.items())}, f"inputs_sha256 in {manifest}")

    def sdev_tables_ok(self) -> bool:
        return self._tables_ok(MANIFEST_SDEV, "sdev")

    # -- stages -----------------------------------------------------------------------------------
    def sdev_data(self, stage: str, ref: Ref) -> Path:
        """Download the S-dev cases and build their derived tables, at the S-dev commit."""
        self.sdev_state = "failed"
        wt = self.worktree(ref.tag or f"sdev-{ref.commit[:7]}", ref.commit)
        self.link_cached("sdev")
        run = lambda *a: self.step(stage, ref.commit, ref.tag, wt, *a)
        run("download.py", "--cohort", "sdev")
        self.check_data(stage, "sdev")
        run("build_instances.py", "--cohort", "sdev")
        run("build_windows.py", "--cohort", "sdev", *self.jobs("--jobs"))
        run("build_features_wf.py", "--cohort", "sdev")
        run("pleth_coverage.py", "--cohort", "sdev")
        self.sdev_state = "built"
        self.check_tables(stage, MANIFEST_SDEV)
        return wt

    def jobs(self, flag: str) -> list[str]:
        return [] if self.a.jobs is None else [flag, str(self.a.jobs)]

    def ensure_sdev_tables(self, for_stage: str):
        """`freeze`, `sob` and `encoder` read the S-dev derived tables: build them unless the data folder
        holds the published ones."""
        if self.sdev_state == "built":
            return
        if self.sdev_state == "failed":
            raise StepFailed("the S-dev data build of this run failed; this stage needs its tables")
        if self.sdev_tables_ok():
            print(f"S-dev derived tables in <workdir>/data/derived equal the published ones (needed by {for_stage})")
            self.check_tables(for_stage, MANIFEST_SDEV)
            self.sdev_state = "built"
            return
        sd = stage_refs(self.git, self.pub, ["sdev"])["sdev"]
        print(f"\n== S-dev data build (needed by {for_stage}), at {sd.label()}")
        t0 = time.time()
        try:
            self.sdev_data("sdev-data", sd)
        finally:
            self.times["sdev-data"] = time.time() - t0
            print(f"== S-dev data build: {fmt_time(self.times['sdev-data'])}")

    def stage_sdev(self, ref: Ref):
        self.wipe("sdev")
        wt = self.sdev_data("sdev", ref)
        try:
            self.step("sdev", ref.commit, ref.tag, wt, "run_confirmatory.py", "--feature-tag", ref.tag,
                      "--with-secondary", *self.jobs("--n-jobs"))
        finally:
            self.collect(wt, "sdev")

    def stage_freeze(self, ref: Ref):
        self.wipe("freeze")
        wt = self.worktree(f"freeze-{ref.commit[:7]}", ref.commit)
        try:
            self.step("freeze", ref.commit, ref.tag, wt, "run_frozen_scoring.py", "freeze", *self.jobs("--n-jobs"))
        finally:
            self.collect(wt, "freeze")

    def stage_sob(self, ref: Ref):
        self.wipe("sob")
        wt = self.worktree(ref.tag or f"sob-{ref.commit[:7]}", ref.commit)
        site = self.data / "derived" / "site"           # written by `series`; an earlier run's copy must not count
        if site.exists():
            shutil.rmtree(site)
        run = lambda *a: self.step("sob", ref.commit, ref.tag, wt, *a, sob=True)
        try:
            self.link_cached("sob")
            run("download.py", "--cohort", "sob")
            self.check_data("sob", "sob")
            run("build_instances.py", "--cohort", "sob")
            run("build_windows.py", "--cohort", "sob", *self.jobs("--jobs"))
            run("build_features_wf.py", "--cohort", "sob")
            self.check_tables("sob", MANIFEST_SCORE)
            for cmd in ("score", "after", "series"):
                run("run_frozen_scoring.py", cmd, "--frozen-tag", ref.tag, *self.jobs("--n-jobs"))
        finally:
            self.collect(wt, "sob")

    def stage_encoder(self, ref: Ref):
        """ENCODER_PLAN.md §1-§6: weights, embeddings, features and build check at the commits recorded for
        them, then the S-dev run at the run's commit (which reads build_check.json as committed there)."""
        self.wipe("encoder")
        m = self.pub.meta(MANIFEST_ENC)
        c = ref.extra
        wts = {}

        def at(step_commit):
            if step_commit not in wts:
                wts[step_commit] = self.worktree(f"encoder-{step_commit[:7]}", step_commit)
            return wts[step_commit]
        run = lambda commit, *a: self.step("encoder", commit, None, at(commit), *a)
        try:
            self.remove_encoder_tables("sdev")
            self.link_weights()
            run(c["embeddings"], "download_encoder.py")
            self.check_files("encoder", {WEIGHTS: (m.get("weights") or {}).get("sha256")},
                             f"weights.sha256 in {MANIFEST_ENC}")
            run(c["embeddings"], "build_embeddings.py", "--jobs", recorded_workers(m, MANIFEST_ENC))
            run(c["features"], "build_features_enc.py")
            self.check_files("encoder", {f"derived/{ENC_EMBEDDINGS}": m.get("embeddings_sha256")},
                             f"embeddings_sha256 in {MANIFEST_ENC}")
            self.check_files("encoder", {f"derived/{ENC_FEATURES}": m.get("encoder_features_sha256")},
                             f"encoder_features_sha256 in {MANIFEST_ENC}")
            run(c["build-check"], "run_encoder_check.py", "build-check")
            self.place(at(ref.commit), SDEV_PREDICTIONS, MANIFEST_SDEV)
            run(ref.commit, "run_encoder_check.py", "sdev", "--n-jobs", ENC_N_JOBS)
        finally:
            for wt in wts.values():
                self.collect(wt, "encoder")

    def stage_encoder_sob(self, ref: Ref):
        """ENCODER_PLAN.md §7: S-ob embeddings, features, refit and scoring in one run of the original script."""
        self.wipe("encoder-sob")
        m = self.pub.meta(MANIFEST_ENC_SOB)
        wt = self.worktree(f"encoder-sob-{ref.commit[:7]}", ref.commit)
        try:
            self.remove_encoder_tables("sob")
            self.link_cached("sob")
            self.link_weights()
            self.place(wt, SOB_PREDICTIONS, MANIFEST_SCORE)
            self.step("encoder-sob", ref.commit, None, wt, "run_encoder_sob.py",
                      "--jobs", recorded_workers(m, MANIFEST_ENC_SOB), "--n-jobs", ENC_N_JOBS, sob=True)
        finally:
            self.collect(wt, "encoder-sob")

    def export(self, done: list[str]):
        """scripts/export_public_tables.py at the commit of each committed public_tables.json (the stages that
        write their tables themselves aside)."""
        groups: dict[str, list[str]] = {}
        self.export_dirs = {}
        for p in sorted(self.pub.tree):
            s = stage_of(p)
            if PurePosixPath(p).name != PUBLIC_TABLES or s not in done or s in SELF_EXPORTING:
                continue
            c = self.pub.meta(p).get("commit")
            if c:
                groups.setdefault(c, []).append(str(PurePosixPath(p).parent))
        for c, folders in groups.items():
            src = ", ".join(f + "/" + PUBLIC_TABLES for f in folders)
            if self.git.commit_of(c) != c:
                self.failed[f"export at {c[:SHORT]}"] = f"commit {c} ({src}) is not in this clone"
                continue
            base = self.work / "export" / c[:SHORT]
            if base.exists():
                shutil.rmtree(base)
            src_root, dst_root = base / "results", base / "tables"
            for s in sorted({stage_of(f) for f in folders}):
                src_dir, keep = self.out / STAGE_DIR[s], nested_dirs(s)
                for f in sorted(src_dir.rglob("*")) if src_dir.exists() else []:
                    r = f.relative_to(self.out).as_posix()
                    if f.is_file() and not any(under(r, k) for k in keep):
                        (base / r).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, base / r)
            if not src_root.exists():
                continue
            try:
                wt = self.worktree(f"export-{c[:7]}", c)
                self.step("export", c, None, wt, "export_public_tables.py", "--results", str(src_root), "--out",
                          str(dst_root))
            except (StepFailed, Refused) as e:
                self.failed[f"export at {c[:SHORT]}"] = str(e)
                print(f"export at {c[:SHORT]} FAILED: {e}", file=sys.stderr)
                continue
            for f in folders:
                self.export_dirs[f] = dst_root / PurePosixPath(f).relative_to("results")

    # -- comparison -----------------------------------------------------------------------------
    def compare(self, done: list[str]):
        pub, covered = self.pub, set()
        for m in pub.manifests():
            s = stage_of(m)
            if s is None:
                continue
            entries = [(manifest_entry_path(m, k), h) for k, h in sorted(pub.json(m).get("files", {}).items())]
            covered.update([m, *(p for p, _ in entries)])
            if s not in done:
                continue
            for p, h in entries:
                note = "" if p in pub.tree else "not committed"
                self.rows.append(Row(s, p, h, sha256_file(self.out / p), note=note))
            self.rows.append(Row(s, m, sha256_bytes(pub.read(m)), sha256_file(self.out / m), note="the MANIFEST itself"))
        covered.update([STATE, SCORING_LOG])
        if "sob" in done:
            b = pub.read(STATE)
            if b is not None:
                self.rows.append(Row("sob", STATE, sha256_bytes(b), sha256_file(self.out / STATE)))
            committed = pub.read(SCORING_LOG)
            if committed is not None:
                local = self.out / SCORING_LOG
                got = normalized_log(local.read_bytes()) if local.is_file() else None
                self.rows.append(Row("sob", SCORING_LOG, sha256_bytes(normalized_log(committed)),
                                     None if got is None else sha256_bytes(got), note="without time_utc"))
                ev = [json.loads(x) for x in committed.decode().splitlines() if x.strip()]
                site = [e for e in ev if e.get("event") == "site_cases_written" and
                        e.get("commit") == self.refs["sob"].commit]
                if site:
                    self.rows.append(Row("sob", SITE_CASES, site[-1].get("cases_json_sha256"),
                                         sha256_file(self.data / "derived" / "site" / "cases.json"),
                                         note=f"cases_json_sha256 in {SCORING_LOG}"))
        for p in sorted(pub.tree):
            if PurePosixPath(p).name != PUBLIC_TABLES:
                continue
            folder = str(PurePosixPath(p).parent)
            names = [PUBLIC_TABLES, *sorted((pub.json(p) or {}).get("files", {}))]
            for n in names:
                covered.add(f"{folder}/{n}")
            s = stage_of(p)
            if s not in done:
                continue
            d = self.out / folder if s in SELF_EXPORTING else self.export_dirs.get(folder)
            how = "written by the stage" if s in SELF_EXPORTING else "scripts/export_public_tables.py"
            for n in names:
                b = pub.read(f"{folder}/{n}")
                if b is not None:
                    self.rows.append(Row(s, f"{folder}/{n}", sha256_bytes(b),
                                         sha256_file(d / n) if d is not None else None, note=how))
        self.not_compared = {p: NOTES.get(p, "not reproduced by this script") for p in sorted(pub.tree - covered)}

    def record_check(self) -> list[Row]:
        """Committed copies against their MANIFEST entries (the published record itself)."""
        rows = []
        for m in self.pub.manifests():
            for key, h in sorted((self.pub.json(m) or {}).get("files", {}).items()):
                p = manifest_entry_path(m, key)
                b = self.pub.read(p)
                if b is not None:
                    rows.append(Row("record", p, h, sha256_bytes(b), note=f"committed copy vs {m}"))
        return rows


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------

def fmt_time(s: float) -> str:
    m, s = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def short(h: str | None) -> str:
    if h is None:
        return "-"
    return h[:SHORT] if len(h) == 64 and all(c in "0123456789abcdef" for c in h) else h


def print_table(rows: list[Row]):
    w = max([len(r.stage) for r in rows] + [5])
    print(f"  {'stage':<{w}}  identical  {'expected':<{SHORT}}  {'got':<{SHORT}}  file")
    for r in rows:
        if r.kind == "data":
            continue
        extra = f"  ({r.note})" if r.note else ""
        print(f"  {r.stage:<{w}}  {r.word():<9}  {short(r.expected):<{SHORT}}  {short(r.got):<{SHORT}}  {r.item}{extra}")


def ref_text(r: Ref) -> str:
    extra = {k: c for k, c in r.extra.items() if c != r.commit}
    by = {}
    for k, c in extra.items():
        by.setdefault(c, []).append(k)
    return r.label() + "".join(f"; {', '.join(ks)} at commit {c[:SHORT]}" for c, ks in by.items())


def write_report(run: Run, rows: list[Row], record: list[Row], summary: dict, plat: dict):
    rep = {
        "summary": summary,
        "main": {"commit": run.head, "describe": run.head_describe, "branch": run.head_branch},
        "refs": {s: dataclasses.asdict(r) for s, r in run.refs.items()},
        "stages_added": run.added,
        "stages_failed": run.failed,
        "elapsed_seconds": {k: round(v, 1) for k, v in run.times.items()},
        "steps": run.steps,
        "platform": plat,
        "packages": dict(sorted(installed().items())),
        "version_problems": run.version_problems,
        "options": {"stages": run.stages, "jobs": run.a.jobs, "data_reused": run.a.data is not None,
                    "ignore_versions": run.a.ignore_versions},
        "rows": [dataclasses.asdict(r) | {"identical": r.word()} for r in rows],
        "published_record": [dataclasses.asdict(r) | {"identical": r.word()} for r in record],
        "data_checks": run.data_details,
        "not_compared": run.not_compared,
    }
    (run.work / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    L = ["# Reproduction report", "",
         f"- main: `{run.head}` ({run.head_describe}{', branch ' + run.head_branch if run.head_branch else ''})",
         *[f"- {s}: {ref_text(r)}, recorded in `{r.source}`; full commits: "
           + ", ".join(f"{k} `{c}`" for k, c in r.commits().items()) for s, r in run.refs.items()],
         *[f"- added: {s} ({why})" for s, why in run.added.items()],
         f"- result: {summary['identical']} of {summary['compared']} reproduced items identical"
         + (f"; failed: {', '.join(sorted(run.failed))}" if run.failed else ""),
         f"- elapsed: " + ", ".join(f"{k} {fmt_time(v)}" for k, v in run.times.items()),
         f"- Python {plat['python']}, {plat['platform']}, {plat['cpu_count']} CPUs; BLAS: "
         + ", ".join(f"{d.get('internal_api')} {d.get('version')}" for d in plat.get("threadpools") or []
                     if isinstance(d, dict))
         + (f"; torch {installed().get('torch')}" if installed().get("torch") else ""),
         "", "| stage | identical | expected | got | file | note |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r.stage} | {r.word()} | {r.expected or '-'} | {r.got or '-'} | `{r.item}` | {r.note} |")
    bad = [r for r in record if r.ok is False]
    L += ["", f"Published record: {len(record) - len(bad)} of {len(record)} committed result files equal the SHA-256 "
          "in their MANIFEST.json."] + [f"- differs: `{r.item}` ({r.note})" for r in bad]
    if run.not_compared:
        L += ["", "Committed files under results/ that are not compared:"] + \
             [f"- `{p}`: {why}" for p, why in run.not_compared.items()]
    L += ["", "Commands (in a worktree at the commit given):"] + \
         [f"- {s['stage']} @ {s['tag'] or s['commit'][:SHORT]}: "
          f"`{('BTC_FROZEN_SCORING=1 ' if s['env'] else '') + ' '.join(s['argv'])}` (exit {s['exit']}, {s['seconds']} s)"
          for s in run.steps]
    (run.work / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------------------------

def parse_stages(s: str) -> list[str]:
    if s.strip() == "all":
        return list(STAGES)
    got = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in got if x not in STAGES]
    if bad or not got:
        raise argparse.ArgumentTypeError(f"unknown stage(s) {bad}; choose from {', '.join(STAGES)} or all")
    return [x for x in STAGES if x in got]


def main(argv=None, repo: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", type=parse_stages, default=list(STAGES),
                    help=f"comma-separated subset of {','.join(STAGES)}, run in that order (default: all); a stage "
                         "whose inputs are not in --workdir yet is added")
    ap.add_argument("--workdir", type=Path, default=None,
                    help="worktrees, data folder, logs, reproduced files and report (default: build/reproduce)")
    ap.add_argument("--data", type=Path, default=None,
                    help="an existing data folder (holding raw/<cohort>/*.npz, raw/meta/ and, for the encoder "
                         "stages, external/papagei/papagei_s.pt) whose files are reused, read only; what is "
                         "already complete is not downloaded")
    ap.add_argument("--keep", action="store_true", help="keep the worktrees (until the next run with this --workdir)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel jobs, passed as --n-jobs to the model scripts and --jobs to build_windows.py "
                         "(default: their own defaults, as in the original runs; the results do not depend on it); "
                         "not passed to the encoder steps, whose results record their worker counts")
    ap.add_argument("--ignore-versions", action="store_true",
                    help="run even if installed package versions differ from the pins of a commit used")
    a = ap.parse_args(argv)
    if a.data is not None and not (Path(a.data) / "raw").is_dir():
        ap.error(f"--data {a.data}: no raw/ folder")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass

    run = Run(a, (repo or REPO).resolve())
    t_all = time.time()
    try:
        run.prepare(a.stages)
        commits = {}
        for s, r in run.refs.items():
            for step, c in r.commits().items():
                commits.setdefault(c, f"{s} ({step})")
        for p in sorted(run.pub.tree):
            if PurePosixPath(p).name == PUBLIC_TABLES and stage_of(p) in run.stages:
                c = run.pub.meta(p).get("commit")
                if c:
                    commits.setdefault(c, f"export {c[:SHORT]}")
        if any(s != "sdev" for s in run.stages):
            sd = stage_refs(run.git, run.pub, ["sdev"])["sdev"]
            commits.setdefault(sd.commit, "S-dev data build")
        run.check_versions({label: c for c, label in commits.items()})
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        if run._lock is not None:
            run._lock.close()
        return 2

    plat = platform_record()
    print(f"main: {run.head} ({run.head_describe})")
    print("stages: " + ", ".join(f"{s} at {ref_text(r)}" for s, r in run.refs.items()))
    for s, why in run.added.items():
        print(f"added stage {s}: {why}")
    print(f"workdir: {run.work}")
    print(f"Python {plat['python']} on {plat['platform']}, {plat['cpu_count']} CPUs")
    record = run.record_check()
    done = []
    runner = {"sdev": run.stage_sdev, "freeze": run.stage_freeze, "sob": run.stage_sob,
              "encoder": run.stage_encoder, "encoder-sob": run.stage_encoder_sob}
    try:
        for s in run.stages:
            ref = run.refs[s]
            print(f"\n== Stage {s}, at {ref_text(ref)}")
            if s in SOB_STAGES:
                print("   (repeats a published run on the cesarean cases in a worktree to check its files; "
                      "not a new result)")
            t0 = time.time()
            try:
                for dep, _, what in run.needs(s):
                    if dep in run.failed:
                        raise StepFailed(f"stage {dep} failed; this stage reads {what}")
                if s != "sdev":
                    run.ensure_sdev_tables(s)        # timed on its own, as sdev-data
                    t0 = time.time()
                runner[s](ref)
            except (StepFailed, Refused) as e:
                run.failed[s] = str(e)
                print(f"stage {s} FAILED: {e}", file=sys.stderr)
            finally:
                run.times[s] = time.time() - t0
                done.append(s)
                print(f"== Stage {s}: {fmt_time(run.times[s])}")
        t0 = time.time()
        try:
            run.export(done)
        except (StepFailed, Refused) as e:
            run.failed["export"] = str(e)
            print(f"export FAILED: {e}", file=sys.stderr)
        run.times["export"] = time.time() - t0
    finally:
        run.close()
    run.compare(done)
    run.times["total"] = time.time() - t_all

    rows = sorted(run.rows, key=lambda r: STAGE_ORDER.get(r.stage, 9))
    checked = [r for r in rows if r.ok is not None]
    rec_bad = [r for r in record if r.ok is False]
    summary = {"compared": len(checked), "identical": sum(1 for r in checked if r.ok),
               "differ": sum(1 for r in checked if r.ok is False and r.got is not None),
               "missing": sum(1 for r in checked if r.ok is False and r.got is None),
               "not_checked": [r.item for r in rows if r.ok is None],
               "published_record": {"committed_copies": len(record), "equal_to_manifest": len(record) - len(rec_bad)},
               "stages_failed": sorted(run.failed)}
    print("\n== Data (published array checksums; meta_json arrays hold the download time and are not compared)")
    for r in rows:
        if r.kind == "data":
            print(f"  {r.stage:<6} {r.word():<4} {r.item}: published {r.expected or '-'}; downloaded {r.got or '-'}")
    print(f"\n== Files (SHA-256, first {SHORT} hex digits; full values in {run.work / 'report.md'})")
    print_table(rows)
    print(f"\nPublished record: {len(record) - len(rec_bad)} of {len(record)} committed result files equal the "
          "SHA-256 in their MANIFEST.json" + "".join(f"\n  differs: {r.item} ({r.note})" for r in rec_bad))
    if run.not_compared:
        print("Committed files under results/ not compared:" +
              "".join(f"\n  {p}: {why}" for p, why in run.not_compared.items()))
    if summary["not_checked"]:
        print("Not checked: " + "; ".join(summary["not_checked"]))
    for s, e in run.failed.items():
        print(f"FAILED {s}: {e}")
    print("Elapsed: " + ", ".join(f"{k} {fmt_time(v)}" for k, v in run.times.items()))
    write_report(run, rows, record, summary, plat)
    ok = (summary["identical"] == summary["compared"] > 0) and not run.failed and not rec_bad
    print(f"\n{'IDENTICAL' if ok else 'NOT IDENTICAL'}: {summary['identical']} of {summary['compared']} reproduced "
          f"items identical ({summary['differ']} differ, {summary['missing']} missing)"
          + (f"; failed: {', '.join(sorted(run.failed))}" if run.failed else "")
          + (f"; {len(rec_bad)} committed files differ from their MANIFEST" if rec_bad else "")
          + f". Report: {run.work / 'report.md'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
