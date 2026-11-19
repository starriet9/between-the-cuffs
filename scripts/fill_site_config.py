#!/usr/bin/env python3
"""Fill the site settings that the repository already knows (btc/site/site.json; SPEC §8.1, §8.3).

Values and their sources (author_name and contact are never touched):
  registration_commit, registered_date   the last row of the registration table in REPORT.md; the
        commit must be an ancestor of HEAD and its PREREG.md and SPEC.md must have the SHA-256 listed
        there (SPEC §10); the date is that commit's date.
  frozen_tag, frozen_date   the frozen-model tag (SPEC §9 step 6): --frozen-tag, or the earliest tag
        that contains the commit adding <results>/frozen/FROZEN.json; the tag date (tagger date, or
        the commit date of a lightweight tag).
  scored_ref, scored_date   the frozen S-ob scoring outputs (SPEC §9 step 7): --scored-ref, or the
        earliest tag that contains the commit adding <results>/sob/frozen_scoring/summary.json, else
        that commit; its date.
  deviations   the number of entries (list items or table rows) in the "## Deviations" section of
        REPORT.md; 0 if the section says "None".
  reproduce_command, checksums_path   from --reproduce-command and --checksums-path, else kept.
        --collect-checksums writes the published checksum list (SPEC §10: checksums only, no data)
        from data/raw/<cohort>_checksums.csv to --checksums-path, with a `cohort` column.
  last_updated   --last-updated, default today.

A value that cannot be determined is left as it is and reported; the publish build refuses any
leftover {{...}} placeholder. --check prints the changes without writing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path("btc/site/site.json")
NEVER = ("author_name", "contact")


class FillError(RuntimeError):
    pass


def git(repo: Path, *args, check: bool = True) -> str | None:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if r.returncode != 0:
        if check:
            raise FillError(f"git {' '.join(args)}: {r.stderr.strip()}")
        return None
    return r.stdout.strip()


def git_bytes(repo: Path, spec: str) -> bytes:
    r = subprocess.run(["git", "show", spec], cwd=repo, capture_output=True)
    if r.returncode != 0:
        raise FillError(f"git show {spec}: {r.stderr.decode().strip()}")
    return r.stdout


def section(md: str, title: str) -> str | None:
    m = re.search(rf"(?ms)^## {re.escape(title)}\s*\n(.*?)(?=^## |^---\s*$|\Z)", md)
    return m.group(1).strip() if m else None


def registration_row(report: str) -> tuple:
    """(version, commit, prereg sha256, spec sha256) of the last row of the registration table."""
    s = section(report, "Registration")
    rows = [ln for ln in (s or "").splitlines() if re.match(r"^\|\s*\d+(\.\d+)*\s*\|", ln)]
    if not rows:
        raise FillError("REPORT.md has no registration table (## Registration)")
    cells = [c.strip().strip("`") for c in rows[-1].strip().strip("|").split("|")]
    if len(cells) < 5:
        raise FillError(f"unexpected registration row: {rows[-1]}")
    return cells[0], cells[1], cells[3], cells[4]


def count_deviations(report: str) -> int:
    """Entries of the '## Deviations' section: top-level list items or table rows; 0 for 'None ...'."""
    s = section(report, "Deviations")
    if s is None:
        raise FillError("REPORT.md has no '## Deviations' section")
    if re.match(r"(?i)^none\b", s):
        return 0
    items = [ln for ln in s.splitlines() if re.match(r"^(?:[-*] |\d+[.)] )", ln)]
    rows = [ln for ln in s.splitlines() if ln.startswith("|") and not re.match(r"^\|[-:| ]+\|$", ln.strip())]
    n = len(items) + max(len(rows) - 1, 0)          # a table's first row is its header
    if n == 0:
        raise FillError("cannot count the deviations in REPORT.md: list them as '- ' items or table rows")
    return n


def ref_date(repo: Path, ref: str) -> str:
    """Tagger date of an annotated tag, else the commit date (YYYY-MM-DD, the ref's own time zone)."""
    if git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{ref}", check=False):
        d = git(repo, "for-each-ref", f"refs/tags/{ref}", "--format=%(taggerdate:short)")
        if d:
            return d
    return git(repo, "log", "-1", "--format=%cd", "--date=short", f"{ref}^{{commit}}")


def first_commit_with(repo: Path, path: str) -> str | None:
    out = git(repo, "log", "--diff-filter=A", "--format=%H", "--reverse", "--", path, check=False)
    return out.splitlines()[0] if out else None


def earliest_tag_containing(repo: Path, commit: str) -> str | None:
    out = git(repo, "for-each-ref", "--sort=creatordate", "--contains", commit, "--format=%(refname:short)",
              "refs/tags", check=False)
    return out.splitlines()[0] if out else None


def resolve_registration(repo: Path, report: str) -> dict:
    ver, commit, sha_prereg, sha_spec = registration_row(report)
    full = git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if git(repo, "merge-base", "--is-ancestor", full, "HEAD", check=False) is None:
        raise FillError(f"registration commit {commit} (version {ver}) is not an ancestor of HEAD")
    for name, want in (("PREREG.md", sha_prereg), ("SPEC.md", sha_spec)):
        got = hashlib.sha256(git_bytes(repo, f"{full}:{name}")).hexdigest()
        if got != want:
            raise FillError(f"{name} at {commit} has SHA-256 {got}, REPORT.md lists {want}")
    return {"registration_commit": commit, "registered_date": git(repo, "log", "-1", "--format=%cd",
                                                                   "--date=short", full)}


def resolve_ref(repo: Path, given: str | None, marker: str, what: str, notes: list) -> tuple | None:
    """(ref, date) of a tag or commit whose tree holds `marker`, or None if none is found."""
    if given:
        if git(repo, "cat-file", "-e", f"{given}:{marker}", check=False) is None:
            raise FillError(f"{given} does not hold {marker} ({what})")
        return given, ref_date(repo, given)
    c = first_commit_with(repo, marker)
    if c is None:
        notes.append(f"{what}: no commit holds {marker} yet; left as it is")
        return None
    tag = earliest_tag_containing(repo, c)
    if tag is None:
        notes.append(f"{what}: no tag contains {c[:7]}; using the commit")
        return c[:7], ref_date(repo, c)
    return tag, ref_date(repo, tag)


def collect_checksums(repo: Path, dest: Path, cohorts=("sdev", "sob")) -> int:
    """Concatenate data/raw/<cohort>_checksums.csv (checksums only) into `dest` with a cohort column."""
    import pandas as pd
    parts = []
    for c in cohorts:
        p = repo / "data" / "raw" / f"{c}_checksums.csv"
        if not p.exists():
            raise FillError(f"missing {p.relative_to(repo)} (scripts/download.py --cohort {c} --checksums-only)")
        d = pd.read_csv(p)
        d.insert(0, "cohort", {"sdev": "S-dev", "sob": "S-ob"}[c])
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    return len(out)


def fill(repo: Path, config: dict, *, results: str = "results", frozen_tag: str | None = None,
         scored_ref: str | None = None, reproduce_command: str | None = None, checksums_path: str | None = None,
         last_updated: str | None = None) -> tuple[dict, list]:
    """The filled settings and notes. Never changes author_name or contact."""
    new, notes = dict(config), []
    report = (repo / "REPORT.md").read_text()
    new.update(resolve_registration(repo, report))
    fz = resolve_ref(repo, frozen_tag, f"{results}/frozen/FROZEN.json", "frozen models (SPEC §9 step 6)", notes)
    if fz:
        new["frozen_tag"], new["frozen_date"] = fz
    sc = resolve_ref(repo, scored_ref, f"{results}/sob/frozen_scoring/summary.json",
                     "frozen S-ob scoring (SPEC §9 step 7)", notes)
    if sc:
        new["scored_ref"], new["scored_date"] = sc
    new["deviations"] = count_deviations(report)
    if reproduce_command:
        new["reproduce_command"] = reproduce_command
    if checksums_path:
        new["checksums_path"] = checksums_path
    if new.get("checksums_path") and "{{" not in new["checksums_path"]:
        p = new["checksums_path"]
        if not (repo / p).is_file():
            notes.append(f"checksums_path {p} does not exist in the work tree")
        elif not git(repo, "ls-files", "--error-unmatch", p, check=False):
            notes.append(f"checksums_path {p} is not tracked by git yet")
    new["last_updated"] = last_updated or dt.date.today().isoformat()
    for k in NEVER:
        if k in config:
            new[k] = config[k]
    left = sorted(k for k, v in new.items()
                  if isinstance(v, str) and "{{" in v and k not in NEVER and not k.startswith("_"))
    if left:
        notes.append("still placeholders: " + ", ".join(left))
    return new, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=ROOT, help=argparse.SUPPRESS)        # tests only
    ap.add_argument("--config", type=Path, default=None, help=f"default {CONFIG}")
    ap.add_argument("--results", default="results", help="results folder, relative to the repository")
    ap.add_argument("--frozen-tag", default=None)
    ap.add_argument("--scored-ref", default=None)
    ap.add_argument("--reproduce-command", default=None)
    ap.add_argument("--checksums-path", default=None, help="published checksum list, relative to the repository")
    ap.add_argument("--collect-checksums", action="store_true",
                    help="write --checksums-path from data/raw/<cohort>_checksums.csv")
    ap.add_argument("--last-updated", default=None, help="YYYY-MM-DD (default today)")
    ap.add_argument("--check", action="store_true", help="print the changes, write nothing")
    a = ap.parse_args(argv)
    repo = a.repo.resolve()
    cfg_path = a.config or repo / CONFIG
    old = json.loads(cfg_path.read_text())
    try:
        if a.collect_checksums:
            if not a.checksums_path:
                ap.error("--collect-checksums needs --checksums-path")
            if not a.check:
                n = collect_checksums(repo, repo / a.checksums_path)
                print(f"{n} checksums written to {a.checksums_path}")
        new, notes = fill(repo, old, results=a.results, frozen_tag=a.frozen_tag, scored_ref=a.scored_ref,
                          reproduce_command=a.reproduce_command, checksums_path=a.checksums_path,
                          last_updated=a.last_updated)
    except FillError as ex:
        print(f"refused: {ex}", file=sys.stderr)
        return 2
    for k in new:
        if new[k] != old.get(k):
            print(f"{k}: {old.get(k)!r} -> {new[k]!r}")
    for n in notes:
        print(f"note: {n}")
    if not a.check:
        cfg_path.write_text(json.dumps(new, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
