"""scripts/verify.py on the committed record (offline, no data).

It passes on this repository and on a fresh clone, where the withheld per-forecast files are absent;
it fails when a committed per-patient table or a manifest hash is altered, and it checks README.md
once README.md is no longer the interim version.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "scripts" / "verify.py"
WIDE = "results/sdev/confirmatory/per_patient_primary_wide.csv"
MANIFEST = "results/sdev/MANIFEST.json"
README_WITHHELD = ("results/sdev/confirmatory/predictions_primary.csv",
                   "results/sdev/confirmatory/yardstick_predictions.csv",
                   "results/sob/frozen_scoring/predictions_primary.csv",
                   "results/sob/frozen_scoring/yardstick_predictions.csv",
                   "results/sob/frozen_scoring/sensitivity_including_nmb/predictions_primary.csv",
                   "results/sdev/secondary/coefficients_all_sdev.csv")

needs_git = pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(),
                               reason="needs git and the repository's history")


def run(*args) -> tuple:
    r = subprocess.run([sys.executable, str(VERIFY), "--json", *args], capture_output=True, text=True, timeout=300)
    return r.returncode, json.loads(r.stdout)


def statuses(rep: dict) -> dict:
    return {c["num"]: c["status"] for c in rep["checks"]}


@pytest.fixture(scope="module")
def clone(tmp_path_factory):
    """A fresh clone of the committed history (tags included), as a reviewer would have it."""
    d = tmp_path_factory.mktemp("verify") / "repo"
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(ROOT), str(d)], check=True)
    return d


class altered:
    """Change one file of the clone for the duration of a `with` block."""

    def __init__(self, path: Path, change):
        self.path, self.change = path, change

    def __enter__(self):
        self.orig = self.path.read_bytes()
        self.path.write_text(self.change(self.orig.decode("utf-8")), encoding="utf-8")

    def __exit__(self, *exc):
        self.path.write_bytes(self.orig)


@needs_git
def test_passes_on_this_repository():
    rc, rep = run()
    st = statuses(rep)
    assert rc == 0, [i["text"] for c in rep["checks"] for i in c["items"] if i["status"] == "FAIL"]
    assert set(st.values()) <= {"PASS", "SKIPPED"}
    assert {n for n, s in st.items() if s == "SKIPPED"} <= {5}          # only the README part can be skipped
    if st[5] == "SKIPPED":
        assert "not scored yet" in (ROOT / "README.md").read_text(encoding="utf-8")


@needs_git
def test_passes_on_a_fresh_clone_without_the_withheld_files(clone):
    rc, rep = run("--root", str(clone))
    assert rc == 0
    notes = [i["text"] for c in rep["checks"] if c["num"] == 2 for i in c["items"] if i["status"] == "NOTE"]
    withheld = {t.split(":")[0]: t for t in notes if "listed, not distributed" in t}
    assert set(README_WITHHELD) <= set(withheld)    # every file results/README.md lists as withheld
    per_forecast = [p for p in withheld if p.endswith(("/predictions_primary.csv", "/yardstick_predictions.csv"))]
    assert per_forecast and all("(git-ignored)" in withheld[p] for p in per_forecast)
    assert not any("local copy" in t for t in withheld.values())


def _bump_first_m4(text: str) -> str:
    lines = text.splitlines(keepends=True)
    head = lines[1].rstrip("\n").split(",")
    i, e = head.index("rmse_M4"), head.index("entered")
    for k in range(2, len(lines)):
        cells = lines[k].rstrip("\n").split(",")
        if cells[e] == "True":
            cells[i] = repr(float(cells[i]) + 0.5)
            lines[k] = ",".join(cells) + "\n"
            return "".join(lines)
    raise AssertionError("no entering patient")


@needs_git
def test_altered_per_patient_table_fails_the_recomputation(clone):
    with altered(clone / WIDE, _bump_first_m4):
        rc, rep = run("--root", str(clone))
    st = statuses(rep)
    assert rc == 1 and st[4] == "FAIL"
    assert st[2] == "FAIL"                          # its SHA-256 in public_tables.json no longer matches


@needs_git
def test_altered_manifest_hash_fails_the_manifest_check(clone):
    def change(text):
        d = json.loads(text)
        d["files"]["confirmatory/contrasts.csv"] = "0" * 64
        return json.dumps(d, indent=1) + "\n"

    with altered(clone / MANIFEST, change):
        rc, rep = run("--root", str(clone))
    st = statuses(rep)
    assert rc == 1 and st[2] == "FAIL"
    assert st[4] == "PASS"                          # the tables themselves are unchanged


@needs_git
def test_a_final_readme_is_checked(clone):
    """Once README.md is not the interim version, its registered lines must match the headline files."""
    s = json.loads((clone / "results/sdev/confirmatory/summary.json").read_text(encoding="utf-8"))
    text = f"# between-the-cuffs\n\n**{s['question']}**\n\n{s['answer_line'].replace('200', '201', 1)}\n"
    with altered(clone / "README.md", lambda _: text):
        rc, rep = run("--root", str(clone))
    st = statuses(rep)
    assert rc == 1 and st[5] == "FAIL"
    failed = [i["text"] for c in rep["checks"] if c["num"] == 5 for i in c["items"] if i["status"] == "FAIL"]
    assert any("Answer line" in t for t in failed)
