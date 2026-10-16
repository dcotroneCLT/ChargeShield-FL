"""LiRA e RMIA versione 2 (src/plugins/attacks/lira2.py, 2026-10-16).

Il test centrale e' lo "specchio": uno shadow allenato con gli stessi dati,
lo stesso seed e la stessa inizializzazione del client reale deve riprodurre,
round per round, il modello che il server osserva per quel client. Se lo
shadow non segue la procedura del client (FedProx, stato di Adam fra i
round, meccanismo DP, privatizzazione osservata, aggregazione), lo specchio
si rompe. Vale dove il meccanismo non estrae rumore: senza DP, DP per
soggetto e per record con sigma = 0, dp-fedavg (si osserva l'update grezzo).
"""

from __future__ import annotations

import copy
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_experiments as rx  # noqa: E402
from plugins.attacks import lira2  # noqa: E402

ROUNDS = 3


def _sessioni(n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    base = datetime(2020, 1, 1)
    out = []
    for i in range(n):
        start = base + timedelta(hours=rng.randint(0, 24 * 60), minutes=rng.randint(0, 59))
        end = start + timedelta(hours=rng.uniform(0.5, 8.0))
        out.append({
            "session_id": f"s-{seed}-{i}",
            "user_id": (f"u{rng.randrange(25)}" if i % 5 else None),
            "start_time": start.isoformat(), "end_time": end.isoformat(),
            "total_energy_kwh": rng.uniform(0.0, 1.0), "max_power_kw": rng.uniform(0.0, 1.0),
            "kwh_requested": rng.uniform(0.0, 1.0), "minutes_available": rng.uniform(0.0, 1.0),
        })
    sess = rx.enrich_sessions(out)
    for s in sess:   # feature in [0, 1] come dopo la normalizzazione
        s["hour_of_day"] = s["hour_of_day"] / 24.0
        s["duration_hours"] = s["duration_hours"] / 8.0
    return sess


def _cfg(meccanismo: str = "nessuna") -> dict:
    cfg = {
        "experiment": {"name": "t", "fl_rounds": ROUNDS, "seed": 42, "epsilon": 4.0,
                       "delta": 1e-5, "max_grad_norm": 1.0},
        "ml": {"input_dim": 6, "lr": 0.01, "epochs": 2, "batch_size": 16,
               "proximal_mu": 0.01, "common_init": True, "norm": "group"},
        "lira": {"n_shadow": 4},
        "byzantine_attack": {"enabled": False},
    }
    if meccanismo == "soggetto":
        cfg["ml"]["subject_dp"] = {"enabled": True, "subject_key": "user_id",
                                   "max_grad_norm": 0.5, "noise_multiplier": 0.0,
                                   "expected_subjects_per_step": 8,
                                   "max_records_per_subject": 16}
    elif meccanismo == "record":
        cfg["ml"]["record_dp"] = {"enabled": True, "max_grad_norm": 0.5,
                                  "noise_multiplier": 0.0}
    return cfg


@pytest.fixture(scope="module")
def dati():
    return _sessioni(160, 1), _sessioni(48, 2)


# ── funzioni pure ──────────────────────────────────────────────────────────────

def test_varianza_fissa_e_medie():
    s = np.array([[1.0, 5.0], [3.0, 7.0], [2.0, 9.0], [4.0, 11.0]])
    m = np.array([[True, True], [True, False], [False, True], [False, False]])
    mu_in, mu_out, n_in, n_out, s_in, s_out = lira2.stime_in_out(s, m)
    assert np.allclose(mu_in, [2.0, 7.0]) and np.allclose(mu_out, [3.0, 9.0])
    assert list(n_in) == [2, 2] and list(n_out) == [2, 2]
    # varianze per campione (ddof 1): IN 2 e 8, OUT 2 e 8 -> media 5
    assert s_in == pytest.approx(np.sqrt(5.0)) and s_out == pytest.approx(np.sqrt(5.0))


def test_punteggio_lira_segno_e_scarti():
    rng = np.random.default_rng(0)
    n, k = 200, 16
    in_mask = rng.random((k, n)) < 0.5
    base = rng.normal(0, 1, n)
    stat = base + np.where(in_mask, -1.0, 0.0) + rng.normal(0, 0.1, (k, n))
    membro = rng.random(n) < 0.5
    t = base + np.where(membro, -1.0, 0.0) + rng.normal(0, 0.1, n)
    score, diag = lira2.punteggio_lira(t, stat, in_mask)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(membro, score) > 0.95
    # senza shadow OUT il campione non si valuta
    in_mask[:, 0] = True
    score, diag = lira2.punteggio_lira(t, stat, in_mask)
    assert np.isnan(score[0]) and diag["n_senza_out"] == 1


def test_punteggio_lira_nullo_se_target_indipendente():
    rng = np.random.default_rng(1)
    n, k = 2000, 16
    in_mask = rng.random((k, n)) < 0.8
    base = rng.normal(0, 1, n)
    stat = base + np.where(in_mask, -0.5, 0.0) + rng.normal(0, 0.2, (k, n))
    membro = rng.random(n) < 0.5
    t = base - 0.4 + rng.normal(0, 0.2, n)            # non dipende da membro
    score, _ = lira2.punteggio_lira(t, stat, in_mask)
    ok = np.isfinite(score)
    from sklearn.metrics import roc_auc_score
    assert abs(roc_auc_score(membro[ok], score[ok]) - 0.5) < 0.04


def test_rmia_popolazione_e_monotonia():
    z = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([-1.0, 0.5, 2.5, 10.0, np.nan])
    s = lira2.punteggio_rmia(x, z, gamma=1.0)
    assert list(s[:4]) == [0.0, 0.25, 0.75, 1.0] and np.isnan(s[4])
    # gamma > 1 abbassa il punteggio
    assert lira2.punteggio_rmia(np.array([2.5]), z, gamma=np.e)[0] == 0.5


def test_log_lr_rmia():
    mt = np.array([0.1, 0.2])
    ms = np.array([[0.1, 0.4], [0.1, 0.4]])
    llr = lira2.log_lr_rmia(mt, ms, tau=0.1)
    assert llr[0] == pytest.approx(0.0) and llr[1] == pytest.approx(2.0)


def test_non_membri_iid_e_per_sito():
    hold = [{"site_id": "0002"}] * 3 + [{"site_id": "0001"}] * 2
    out = lira2.assegna_non_membri(hold, ["caltech", "jpl"], {"experiment": {"seed": 1}},
                                   {"caltech": 10, "jpl": 10})
    assert len(out["caltech"]) == 3 and len(out["jpl"]) == 2
    out = lira2.assegna_non_membri(hold * 20, ["caltech", "jpl"],
                                   {"experiment": {"seed": 1}, "partition": {"strategy": "iid"}},
                                   {"caltech": 30, "jpl": 10})
    assert len(out["caltech"]) == 75 and len(out["jpl"]) == 25
    # con iid i siti si mescolano
    assert {s["site_id"] for s in out["jpl"]} == {"0001", "0002"}


# ── specchio: shadow con i dati del client = client ───────────────────────────

def _specchio(cfg, dati, no_dp=True, dp_mode="dp-fedavg", superficie="client"):
    train, _ = dati
    torch.manual_seed(42)
    fl = rx.run_fl_rounds(copy.deepcopy(cfg), train, no_dp=no_dp, dp_mode=dp_mode)
    c = "urban"
    size = len(train) // 4
    membri = {cc: train[i * size:(i + 1) * size if i < 3 else len(train)]
              for i, cc in enumerate(["highway", "urban", "residential", "corporate"])}
    feats = list(rx._mia_feature_names(cfg))
    x = np.array([[s[f] for f in feats] for s in train], dtype=np.float32)
    sdp = rx._subject_dp_config(cfg["ml"])
    pesi = (rx._subject_cross_client_weights(membri, "user_id", feats)[c]
            if sdp["enabled"] else None)
    rounds = list(range(1, ROUNDS + 1))
    altri, ncli = {}, {}
    for r in rounds:
        upd = fl[r]["updates"]
        altri[r] = [None if u.cluster_id == c else u for u in upd]
        ncli[r] = [u for u in upd if u.cluster_id == c][0].n_samples
    spec = lira2.SpecShadow(
        cid=c, k=0, seed=42, rounds=rounds, in_sessions=membri[c], eval_x=x,
        ml_cfg=dict(cfg["ml"]), arch=rx._autoencoder_arch_kwargs(cfg), input_dim=6,
        gm_cfg={"epsilon": 4.0, "delta": 1e-5, "max_grad_norm": 1.0},
        no_dp=no_dp, dp_mode=dp_mode, dp_ablation="full",
        init_weights=fl[0]["raw_global_weights"],
        global_prev={r: fl[r - 1]["global_weights"] for r in rounds if r > 1},
        subject_weights=pesi, superficie=superficie, altri_update=altri,
        n_samples_client=ncli, n_partecipanti=4,
    )
    stato = torch.get_num_threads()
    out = lira2.traiettoria_shadow(spec)
    torch.set_num_threads(stato)
    arch = rx._autoencoder_arch_kwargs(cfg)
    for r in rounds:
        if superficie == "global":
            pesi_t = fl[r]["global_weights"]
        else:
            fonte = fl[r]["raw_updates"] if (not no_dp and dp_mode == "dp-fedavg") else fl[r]["updates"]
            pesi_t = [u for u in fonte if u.cluster_id == c][0].weights
        atteso = lira2.mse_per_campione(arch, 6, pesi_t, x)
        np.testing.assert_allclose(out[r], atteso, rtol=1e-5, atol=1e-8,
                                   err_msg=f"round {r}: lo shadow non riproduce il client")


@pytest.mark.parametrize("meccanismo", ["nessuna", "soggetto", "record"])
def test_specchio_superficie_client(meccanismo, dati):
    _specchio(_cfg(meccanismo), dati)


def test_specchio_dp_fedavg_update_grezzo(dati):
    _specchio(_cfg(), dati, no_dp=False, dp_mode="dp-fedavg")


def test_specchio_superficie_global(dati):
    cfg = _cfg()
    cfg["lira"]["observation_surface"] = "global"
    _specchio(cfg, dati, superficie="global")


def test_specchio_senza_common_init_round_successivi(dati):
    """Senza inizializzazione comune il round 1 differisce (init proprio dello
    shadow), ma dal round 2 lo shadow riparte dal globale come il client: il
    test verifica solo che la traiettoria giri e resti finita."""
    cfg = _cfg()
    cfg["ml"]["common_init"] = False
    train, _ = dati
    torch.manual_seed(42)
    fl = rx.run_fl_rounds(copy.deepcopy(cfg), train, no_dp=True)
    res = lira2.run_lira_v2(cfg, train, dati[1], fl, n_shadow=4, no_dp=True)
    for r in range(1, ROUNDS + 1):
        assert res[r]["lira_auc_roc"] is None or 0.0 <= res[r]["lira_auc_roc"] <= 1.0


# ── attacco completo ───────────────────────────────────────────────────────────

def test_attacco_completo_e_processi(dati, tmp_path):
    cfg = _cfg("soggetto")
    train, hold = dati
    torch.manual_seed(42)
    fl = rx.run_fl_rounds(copy.deepcopy(cfg), train, no_dp=True)
    comp1: dict[str, Any] = {}
    a = lira2.run_lira_v2(cfg, train, hold, fl, n_shadow=4, no_dp=True,
                          composed_output=comp1, per_sample_dump_path=str(tmp_path / "d.json"))
    comp2: dict[str, Any] = {}
    b = lira2.run_lira_v2(cfg, train, hold, fl, n_shadow=4, no_dp=True,
                          composed_output=comp2, workers=2)
    for r in range(1, ROUNDS + 1):
        for k in ("lira_auc_roc", "rmia_auc_roc", "lira_debug_raw_mse_auc_roc"):
            assert a[r][k] == b[r][k], (r, k)
        assert a[r]["lira_version"] == 2
        for c, d in a[r]["lira_per_client"].items():
            assert d["n_valutati"] > 0
    assert comp1 == comp2 and "composed_lira_auc_roc" in comp1 and "composed_rmia_loglr_auc_roc" in comp1
    import json
    dump = json.loads((tmp_path / "d.json").read_text())
    assert dump["n_records"] == comp1["composed_n_samples"]
