"""Site settings that are not results: dates, public links, author and contact.

Read from a JSON file (default `btc/site/site.json`). Unset values stay as `{{...}}` placeholders;
the publish build refuses to run while any placeholder remains in its output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent / "site.json"


@dataclass
class SiteConfig:
    author_name: str = "{{AUTHOR_NAME}}"
    contact: str = "{{CONTACT}}"
    repo_url: str = "{{REPO_URL}}"
    page_url: str = "{{PAGE_URL}}"
    branch: str = "main"
    registered_date: str = "2026-11-21"              # SPEC/PREREG v1.5 commit
    registered_public_date: str = "2026-11-22"       # GitHub release prereg-v1.5 (server time)
    registration_commit: str = "59939b7"
    frozen_date: str = "{{FROZEN_DATE}}"             # SPEC §9 step 6
    frozen_tag: str = "{{FROZEN_TAG}}"
    scored_date: str = "{{SCORED_DATE}}"             # SPEC §9 step 7
    scored_ref: str = "{{SCORED_REF}}"               # commit or tag of the frozen-scoring outputs
    deviations: int | None = None                    # count listed in REPORT.md; None = not yet known
    reproduce_command: str = "{{REPRODUCE_COMMAND}}"
    checksums_path: str = "{{CHECKSUMS_PATH}}"       # published list of data checksums (SPEC §10)
    narrated_replay_url: str | None = None           # optional (SPEC §8.3)
    pdf_url: str | None = None                       # optional (SPEC §8.3)
    last_updated: str = "{{LAST_UPDATED}}"
    extra: dict = field(default_factory=dict)

    # public links ------------------------------------------------------------
    @property
    def contact_url(self) -> str:
        c = self.contact
        if c.startswith(("http://", "https://")):
            return c
        return f"mailto:{c}" if "@" in c else f"https://{c}"

    def blob(self, path: str) -> str:
        return f"{self.repo_url}/blob/{self.branch}/{path}"

    def tree(self, path: str) -> str:
        return f"{self.repo_url}/tree/{self.branch}/{path}"

    @property
    def registration_url(self) -> str:
        return f"{self.repo_url}/commit/{self.registration_commit}"

    @property
    def frozen_url(self) -> str:
        return f"{self.repo_url}/tree/{self.frozen_tag}"

    @property
    def scored_url(self) -> str:
        return f"{self.repo_url}/tree/{self.scored_ref}"


def load_config(path: Path | None = None, **overrides) -> SiteConfig:
    p = Path(path) if path else DEFAULT_CONFIG
    d = json.loads(p.read_text()) if p.exists() else {}
    d = {k: v for k, v in d.items() if not k.startswith("_")}
    names = {f.name for f in fields(SiteConfig)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown site settings: {sorted(unknown)}")
    d.update(overrides)
    return SiteConfig(**d)
