"""The vendored PaPaGei-S model class and the weights (ENCODER_PLAN.md §1). Needs torch (dependency group
`encoder`); the tests that need the weights file are skipped without it (scripts/download_encoder.py).
Synthetic signals only: no real pleth is passed through the model here.
"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from btc import encoder as E  # noqa: E402
from btc.third_party.papagei_resnet import ResNet1DMoE  # noqa: E402
from wf_synth import Beat, make_signals  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = E.WEIGHTS.parent / "upstream"
REMOVED = {"import numpy as np", "from collections import Counter", "from tqdm import tqdm",
           "from matplotlib import pyplot as plt", "from sklearn.metrics import classification_report ",
           "import torch.optim as optim", "from torch.utils.data import Dataset, DataLoader"}
weights = pytest.mark.skipif(not E.WEIGHTS.exists(), reason="weights not downloaded")


@pytest.fixture(scope="module")
def windows():
    p, *_ = make_signals(200, Beat(), noise=0.3, seed=4)
    return np.stack([E.preprocess_window(p[i * 5000:(i + 1) * 5000]) for i in range(8)])


@pytest.mark.skipif(not (UPSTREAM / "resnet.py").exists(), reason="no upstream copy")
def test_vendored_file_is_upstream_minus_unused_imports():
    up = (UPSTREAM / "resnet.py").read_text().split("\n")
    ours = (ROOT / "btc" / "third_party" / "papagei_resnet.py").read_text().split("\n")
    added = [l for l in ours if l not in up]
    assert all(l.startswith("#") for l in added)                    # only the provenance header is added
    assert [l for l in up if l not in REMOVED] == [l for l in ours if l not in added]
    assert len([l for l in up if l in REMOVED]) == 7
    assert E.file_digests(UPSTREAM / "resnet.py")["sha256"] in "\n".join(added)


@pytest.mark.skipif(not (UPSTREAM / "LICENSE").exists(), reason="no upstream copy")
def test_licence_verbatim():
    assert (ROOT / "btc" / "third_party" / "PAPAGEI_LICENSE").read_bytes() == (UPSTREAM / "LICENSE").read_bytes()


def test_architecture_and_outputs(windows):
    torch.manual_seed(0)
    m = ResNet1DMoE(**E.MODEL_KW).eval()
    assert sum(p.numel() for p in m.parameters()) == 5_785_612
    with torch.inference_mode():
        out = m(torch.from_numpy(windows[:3, None, :]))
    assert len(out) == 4 and [tuple(o.shape) for o in out] == [(3, 512), (3, 1), (3, 1), (3, 512)]
    Y = E.embed(m, windows)
    assert Y.shape == (8, 512) and Y.dtype == np.float32
    # eval mode (dropout and batch norm inactive): a window gives the same embedding alone, twice, or in a batch
    assert np.array_equal(E.embed(m, windows[2:3]), E.embed(m, windows[2:3]))
    assert np.array_equal(Y[2], E.embed(m, windows[2:3])[0])           # one window per pass (BATCH = 1)
    assert np.allclose(E.embed(m, windows, batch=8)[2], Y[2], atol=1e-5)
    m.train()
    with pytest.raises(RuntimeError):
        E.embed(m, windows)
    with pytest.raises(ValueError):
        E.embed(m.eval(), windows[:, :1000])


def test_weights_check_refuses_another_file(tmp_path):
    p = tmp_path / "papagei_s.pt"
    p.write_bytes(b"0" * 100)
    with pytest.raises(ValueError):
        E.check_weights(p)
    with pytest.raises(ValueError):
        E.load_model(p)


@weights
def test_weights_strict_load_and_determinism(windows):
    E.configure_torch()
    m, info = E.load_model()
    assert (info["size"], info["md5"]) == (E.WEIGHTS_SIZE, E.WEIGHTS_MD5) and info["strict"]
    assert info["n_tensors"] == 294 and info["n_parameters"] == 5_785_612
    assert not m.training and not any(p.requires_grad for p in m.parameters())
    Y = E.embed(m, windows)
    assert Y.shape == (8, 512) and np.all(np.isfinite(Y))
    assert np.array_equal(Y, E.embed(m, windows))
    assert np.array_equal(Y[5], E.embed(m, windows[5:6])[0])
    assert np.std(Y, axis=0).max() > 0                                  # windows differ


@weights
def test_leakage_with_the_model():
    """SPEC §4.4 with the real extractor: perturbing samples at or after the cutoff changes no feature."""
    from test_encoder_features import _signals, check_leakage
    E.configure_torch()
    m, _ = E.load_model()
    own = _signals(400, Beat(), 0.62, 1.0, 1)
    donor = _signals(400, Beat(T=0.75, N=0.18), 0.6, 0.4, 2)
    n_avail, n_lvl = check_leakage(own, donor, [(300.4, 211.9, 150.2), (45.3, 12.0, 3.0)],
                                   embed_fn=lambda X: E.embed(m, X), dim=E.EMB_DIM)
    assert n_avail >= 3 and n_lvl >= 2
