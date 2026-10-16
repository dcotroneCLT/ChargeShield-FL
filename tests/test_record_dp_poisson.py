"""DP per record con campionamento di Poisson e vmap (2026-10-16)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ml.autoencoder_trainer import AutoencoderTrainer  # noqa: E402
from ml.record_dp_accounting import epsilon_poisson  # noqa: E402


def _trainer(sigma=1.0, C=0.5, epochs=2, mu=0.0):
    cfg = {"input_dim": 6, "lr": 0.01, "epochs": epochs, "batch_size": 8, "seed": 3,
           "proximal_mu": mu, "norm": "group",
           "record_dp": {"enabled": True, "max_grad_norm": C, "noise_multiplier": sigma}}
    torch.manual_seed(0)
    return AutoencoderTrainer(config=cfg, node_id="n", cluster_id="c")


def test_vmap_uguale_al_ciclo():
    t = _trainer()
    x = torch.rand(17, 6) * 3
    a, la = t._record_dp_clipped_sum_loop(x)
    b, lb = t._record_dp_clipped_sum_vmap(x)
    for u, v in zip(a, b):
        assert torch.allclose(u, v, atol=1e-6)
    assert la == pytest.approx(lb, rel=1e-6)


def test_taglio_per_esempio():
    t = _trainer(C=0.01)
    x = torch.rand(5, 6) * 10
    somma, _ = t._record_dp_clipped_sum_vmap(x)
    norma = torch.sqrt(sum((s ** 2).sum() for s in somma))
    assert float(norma) <= 5 * 0.01 + 1e-6


def test_passi_e_q_come_accountant():
    t = _trainer(epochs=3)
    sess = [{f: 0.5 for f in t.CONTINUOUS_FEATURES} for _ in range(50)]
    t.train_local(sess, 1)
    assert t.record_dp_last == {"n": 50, "q": 8 / 50, "passi_per_epoca": 50 // 8}
    _, passi, q = epsilon_poisson(50, 8, 3, 1, 1.0, 1e-5)
    assert passi == t.record_dp_last["passi_per_epoca"] * 3 and q == t.record_dp_last["q"]


def test_campionamento_poisson_dimensioni_variabili():
    """I lotti hanno dimensione variabile e media B (Poisson), non fissa."""
    t = _trainer(epochs=40)
    viste = []
    orig = t._record_dp_clipped_sum_vmap

    def spia(b):
        viste.append(int(b.shape[0]))
        return orig(b)
    t._record_dp_clipped_sum_vmap = spia
    sess = [{f: 0.5 for f in t.CONTINUOUS_FEATURES} for _ in range(64)]
    t.train_local(sess, 1)
    assert len(set(viste)) > 3
    assert abs(sum(viste) / (40 * 8) - 8) < 1.0


def test_input_non_finito_rifiutato():
    t = _trainer()
    sess = [{f: 0.5 for f in t.CONTINUOUS_FEATURES} for _ in range(20)]
    sess[0]["total_energy_kwh"] = float("inf")
    assert t._sessions_to_tensor(sess).shape[0] == 19


def test_interruttori_stretti():
    from ml.subject_dp import flag, record_dp_on
    assert flag("false") is False and flag("True") is True and flag(None) is False
    assert record_dp_on({"record_dp": {"enabled": "no"}}) is False
    with pytest.raises(ValueError):
        flag("forse")
