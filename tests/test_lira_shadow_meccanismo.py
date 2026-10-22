"""Gli shadow di LiRA (e quindi i riferimenti di RMIA) si allenano con lo
stesso meccanismo del client reale quando la DP e' per record o per soggetto
(fix 2026-10-16).

Prima del fix gli shadow usavano sempre un ciclo Adam senza DP: con la DP per
record o per soggetto (che girano con --no-dp, quindi senza il ramo di
privatizzazione per client) la calibrazione IN/OUT confrontava un bersaglio con
DP con shadow senza DP. Questi test verificano che:
  - con DP per soggetto e per record ogni shadow passa da
    AutoencoderTrainer.train_local, con la stessa configurazione di DP del
    bersaglio, gli stessi pesi dei soggetti del client e un seed proprio;
  - senza queste DP il percorso storico resta quello di prima (nessuna chiamata
    al trainer per gli shadow), quindi le run gia' prodotte non cambiano;
  - a parita' di seed LiRA e RMIA danno numeri identici.
"""

from __future__ import annotations

import copy
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_experiments as run_exp  # noqa: E402
from ml.autoencoder_trainer import AutoencoderTrainer  # noqa: E402

N_SHADOW = 2
ROUNDS = 2


def _sessioni(n: int, seed: int, n_utenti: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    base = datetime(2020, 1, 1)
    out = []
    for i in range(n):
        start = base + timedelta(hours=rng.randint(0, 24 * 60), minutes=rng.randint(0, 59))
        end = start + timedelta(hours=rng.uniform(0.5, 8.0))
        out.append({
            "session_id": f"s-{seed}-{i}",
            # Utenti ripetuti (anche fra cluster diversi) e alcune sessioni
            # senza utente, che diventano soggetti singoli.
            "user_id": (f"u{rng.randrange(n_utenti)}" if i % 5 else None),
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "total_energy_kwh": rng.uniform(1.0, 30.0),
            "max_power_kw": rng.uniform(3.0, 22.0),
            "kwh_requested": rng.uniform(1.0, 30.0),
            "minutes_available": rng.uniform(30, 600),
        })
    return run_exp.enrich_sessions(out)


def _cfg(meccanismo: str) -> dict:
    cfg = {
        "experiment": {"name": "pytest_shadow", "fl_rounds": ROUNDS, "seed": 42,
                       "epsilon": 1.0, "delta": 1.0e-5, "max_grad_norm": 1.0},
        "ml": {"input_dim": 6, "lr": 0.01, "epochs": 2, "batch_size": 16,
               "proximal_mu": 0.01},
        "lira": {"n_shadow": N_SHADOW},
        "byzantine_attack": {"enabled": False},
    }
    if meccanismo == "subject":
        cfg["ml"]["norm"] = "group"
        cfg["ml"]["subject_dp"] = {
            "enabled": True, "subject_key": "user_id", "max_grad_norm": 1.0,
            "noise_multiplier": 2.0, "expected_subjects_per_step": 8,
            "max_records_per_subject": 16,
        }
    elif meccanismo == "record":
        cfg["ml"]["norm"] = "group"
        cfg["ml"]["record_dp"] = {"enabled": True, "max_grad_norm": 1.0,
                                  "noise_multiplier": 2.0}
    return cfg


@pytest.fixture(scope="module")
def train_sessions():
    return _sessioni(160, seed=1, n_utenti=25)


@pytest.fixture(scope="module")
def holdout_sessions():
    return _sessioni(40, seed=2, n_utenti=25)


def _esegui(meccanismo, train_sessions, holdout_sessions, monkeypatch):
    """Esegue FL + LiRA registrando ogni chiamata a train_local."""
    chiamate: list[dict[str, Any]] = []
    originale = AutoencoderTrainer.train_local

    def _spia(self, sessions, round_num):
        chiamate.append({
            "node_id": self.node_id, "cluster_id": self.cluster_id,
            "round": round_num, "seed": self._seed, "epochs": self.epochs,
            "subject_dp": dict(self.subject_dp) if self.subject_dp_enabled else None,
            "record_dp": (self.record_dp_max_grad_norm, self.record_dp_noise_multiplier)
            if self.record_dp_enabled else None,
            "pesi_soggetti": dict(getattr(self, "_subject_weights", {}) or {}),
            "ancora_fedprox": self._global_weights is not None,
            "n_sessioni": len(sessions),
        })
        return originale(self, sessions, round_num)

    monkeypatch.setattr(AutoencoderTrainer, "train_local", _spia)
    cfg = _cfg(meccanismo)
    torch.manual_seed(0)
    fl = run_exp.run_fl_rounds(cfg, train_sessions, no_dp=True)
    lira = run_exp.run_lira(cfg, train_sessions, holdout_sessions, fl,
                            n_shadow=N_SHADOW, no_dp=True)
    return chiamate, lira


def _divise(chiamate):
    shadow = [c for c in chiamate if c["node_id"].startswith("shadow-")]
    client = [c for c in chiamate if not c["node_id"].startswith("shadow-")]
    return shadow, client


@pytest.mark.parametrize("meccanismo", ["subject", "record"])
def test_shadow_con_lo_stesso_meccanismo_del_client(
        meccanismo, train_sessions, holdout_sessions, monkeypatch):
    chiamate, lira = _esegui(meccanismo, train_sessions, holdout_sessions, monkeypatch)
    shadow, client = _divise(chiamate)
    n_cluster = len({c["cluster_id"] for c in client})
    assert n_cluster > 1
    # Uno shadow per cluster, per indice e per round.
    assert len(shadow) == ROUNDS * n_cluster * N_SHADOW
    rif = {c["cluster_id"]: c for c in client}
    for s in shadow:
        r = rif[s["cluster_id"]]
        assert s["subject_dp"] == r["subject_dp"]
        assert s["record_dp"] == r["record_dp"]
        assert s["epochs"] == r["epochs"]
        # Seed proprio dello shadow: campionamento e rumore indipendenti.
        assert s["seed"] != r["seed"]
        # FedProx come il client reale: ancora assente al round 1, presente dopo.
        assert s["ancora_fedprox"] == (s["round"] > 1)
        if meccanismo == "subject":
            assert s["pesi_soggetti"] == r["pesi_soggetti"]
    # Seed diversi fra shadow dello stesso cluster e round.
    for cid in rif:
        semi = {s["seed"] for s in shadow if s["cluster_id"] == cid and s["round"] == 1}
        assert len(semi) == N_SHADOW
    if meccanismo == "subject":
        # Il caso di test ha davvero soggetti in piu' client (pesi < 1).
        assert any(r["pesi_soggetti"] for r in rif.values())
    for r in range(1, ROUNDS + 1):
        assert 0.0 <= lira[r]["lira_auc_roc"] <= 1.0


def test_senza_dp_per_record_o_soggetto_percorso_storico(
        train_sessions, holdout_sessions, monkeypatch):
    chiamate, lira = _esegui("nessuna", train_sessions, holdout_sessions, monkeypatch)
    shadow, client = _divise(chiamate)
    assert shadow == []
    assert client
    for r in range(1, ROUNDS + 1):
        assert 0.0 <= lira[r]["lira_auc_roc"] <= 1.0


def test_riproducibile_a_parita_di_seed(train_sessions, holdout_sessions, monkeypatch):
    _, a = _esegui("subject", train_sessions, holdout_sessions, monkeypatch)
    _, b = _esegui("subject", train_sessions, holdout_sessions, monkeypatch)
    chiavi = [k for k in a[ROUNDS] if k.startswith(("lira_auc", "rmia_"))]
    assert "lira_auc_roc" in chiavi
    for r in range(1, ROUNDS + 1):
        for k in chiavi:
            assert a[r].get(k) == b[r].get(k), (r, k)
