"""The HTTP behaviour of the two download scripts, with fake responses (no network).

A refused request is not retried: it stops with the HTTP status and the URL. Transient errors are
still retried (scripts/download.py). Every request uses the default User-Agent.
"""
import csv
import importlib.util
import sys

import pytest

from btc import paths


def _script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}_http", paths.ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Raw:
    def __init__(self, body):
        self.body = body

    def stream(self, n, decode_content=False):
        yield self.body


class _Resp:
    def __init__(self, status, body=b""):
        self.status_code = status
        self.headers = {"Content-Length": str(len(body))} if status == 200 else {}
        self.raw = _Raw(body)

    def iter_content(self, n):
        yield self.raw.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Session:
    """Returns the given statuses in turn and records every call."""

    def __init__(self, statuses, body=b"ok"):
        self.statuses = list(statuses)
        self.body = body
        self.calls = []

    def get(self, url, **kw):
        self.calls.append((url, kw))
        s = self.statuses.pop(0)
        return _Resp(s, self.body if s == 200 else b"")


def _http(D, tmp_path, statuses, monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    h = D.Http(tmp_path / "requests.csv", timeout=5, retries=2)
    h.local.s = _Session(statuses)
    return h


@pytest.mark.parametrize("status", [401, 403, 406, 451])
def test_vitaldb_refusal_stops_without_retry(tmp_path, monkeypatch, status):
    D = _script("download")
    h = _http(D, tmp_path, [status], monkeypatch)
    url = f"{D.API}/cases"
    with pytest.raises(D.Refused, match=f"HTTP {status} for {url}"):
        h.get(url, kind="table:cases")
    assert len(h.local.s.calls) == 1                              # one request, no second try
    assert "headers" not in h.local.s.calls[0][1]                 # default User-Agent
    h.log_f.close()
    rows = list(csv.DictReader((tmp_path / "requests.csv").open()))
    assert [(r["status"], r["user_agent"]) for r in rows] == [(str(status), "default")]


def test_vitaldb_transient_errors_are_still_retried(tmp_path, monkeypatch):
    D = _script("download")
    h = _http(D, tmp_path, [503, 429, 200], monkeypatch)
    body, enc = h.get(f"{D.API}/trks", kind="table:trks")
    assert body == b"ok" and len(h.local.s.calls) == 3
    h.log_f.close()
    h = _http(D, tmp_path, [503, 503, 503], monkeypatch)
    with pytest.raises(IOError, match="giving up after 2 retries"):
        h.get(f"{D.API}/trks", kind="table:trks")
    h.log_f.close()
    assert not hasattr(D, "CHROME_UA")


@pytest.mark.parametrize("status,what", [(403, "request refused"), (429, "request refused"),
                                         (404, "request failed")])
def test_encoder_download_stops_on_the_first_refusal(tmp_path, monkeypatch, capsys, status, what):
    DE = _script("download_encoder")
    s = _Session([status])
    monkeypatch.setattr(DE.requests, "get", s.get)
    out = tmp_path / "papagei" / "papagei_s.pt"
    assert DE.main(["--out", str(out)]) == 2
    assert len(s.calls) == 1 and "headers" not in s.calls[0][1]
    err = capsys.readouterr().err
    assert f"STOPPED: {what}: HTTP {status} for {DE.URL}" in err
    assert not out.exists() and not list(out.parent.glob("*.part"))
    assert not hasattr(DE, "CHROME_UA")
