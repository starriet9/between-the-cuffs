#!/usr/bin/env python3
"""Download the PaPaGei-S weights once (ENCODER_PLAN.md §1).

The weights are not redistributed with this repository: the Zenodo record states no licence for
them (ENCODER_PLAN.md §8). This script fetches `papagei_s.pt` from Zenodo record 13983110
(doi:10.5281/zenodo.13983110) into the git-ignored folder `data/external/papagei/`, keeps the file
only if its size (23,339,216 bytes) and the published MD5 match, and writes
`data/external/papagei/download_log.json` with the URL, size, MD5, SHA-256 and the download time
(a log, so it holds a timestamp; result files do not).

This is the only step of the encoder check that uses the network. The encoder code never fetches
anything (`btc/encoder.py` holds no URL) and checks size and MD5 again before it loads the file.
The file is untrusted input: `btc.encoder.load_model` reads it with `torch.load(...,
weights_only=True)` only.

Requests use the default User-Agent; if refused (HTTP 403 or 429), one retry with a Chrome
User-Agent, then the script stops (project rule). No personal detail is sent.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from btc import encoder as E  # noqa: E402

URL = f"https://zenodo.org/api/records/{E.ZENODO_RECORD}/files/{E.WEIGHTS_NAME}/content"
CHROME_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
             "Chrome/131.0.0.0 Safari/537.36")
REFUSED = (403, 429)


def _fetch(url: str, dest: Path, ua: str | None, timeout: float) -> int:
    headers = {"User-Agent": ua} if ua else {}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        if r.status_code != 200:
            return r.status_code
        with dest.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return 200


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=E.WEIGHTS, help="default: data/external/papagei/papagei_s.pt")
    ap.add_argument("--timeout", type=float, default=300.0)
    a = ap.parse_args(argv)
    out = a.out
    if out.exists():
        info = E.check_weights(out)
        print(f"already present and checked: {out.name} ({info['size']} bytes, md5 {info['md5']})")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    status, ua_used = _fetch(URL, part, None, a.timeout), "default"
    if status in REFUSED:
        status, ua_used = _fetch(URL, part, CHROME_UA, a.timeout), "chrome"
    if status != 200:
        part.unlink(missing_ok=True)
        print(f"STOPPED: HTTP {status} ({ua_used} User-Agent) for {URL}", file=sys.stderr)
        return 2
    try:
        info = E.check_weights(part)
    except ValueError as ex:
        part.unlink(missing_ok=True)
        print(f"STOPPED: {ex}; the file was not kept", file=sys.stderr)
        return 3
    part.rename(out)
    log = {"url": URL, "record": E.ZENODO_RECORD, "doi": E.ZENODO_DOI, "file": E.WEIGHTS_NAME,
           "size": info["size"], "md5": info["md5"], "sha256": info["sha256"], "user_agent": ua_used,
           "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    (out.parent / "download_log.json").write_text(json.dumps(log, indent=1) + "\n")
    print(json.dumps(log, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
