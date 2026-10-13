"""DP a livello di soggetto (2026-10-13): src/ml/subject_dp.py, il trainer e
il collegamento in scripts/run_experiments.py. Nessun test legge i dati reali."""
from __future__ import annotations

import copy
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_experiments as run_exp  # noqa: E402
from ml import subject_dp as sdp  # noqa: E402
from ml.autoencoder_trainer import AutoencoderTrainer  # noqa: E402


# ── definizione di soggetto ─────────────────────────────────────────────────────

class TestSoggetti:
    def test_chiave_canary_e_fallback(self):
        ss = [
            {"user_id": "u1"},
            {"user_id": None},
            {"user_id": ""},
            {"user_id": "u1", "_canary_group": "g3"},
            {"user_id": "u2"},
        ]
        ids = sdp.subject_ids(ss, "user_id", "c1")
        assert ids[0] == ids[4].replace("u2", "u1") == "user_id:u1"
        assert ids[1] == "solo:c1:1" and ids[2] == "solo:c1:2"
        assert ids[3] == "canary:g3"   # le copie di un canary sono un soggetto unico

    def test_pesi_tra_client_sommano_a_uno(self):
        cs = {
            "a": [{"user_id": "u1"}] * 3 + [{"user_id": "u2"}],
            "b": [{"user_id": "u1"}] + [{"user_id": "u3"}] * 2,
        }
        w = sdp.cross_client_weights(cs, "user_id")
        assert w["a"] == {"user_id:u1": 0.75}
        assert w["b"] == {"user_id:u1": 0.25}
        assert math.isclose(w["a"]["user_id:u1"] + w["b"]["user_id:u1"], 1.0)

    def test_campionamento(self):
        assert sdp.sampling_schedule(0, 32) == (0.0, 0)
        assert sdp.sampling_schedule(20, 32) == (1.0, 1)
        q, per_epoch = sdp.sampling_schedule(100, 32)
        assert q == pytest.approx(0.32) and per_epoch == 4

    def test_statistiche(self):
        cs = {
            "a": [{"user_id": "u1"}] * 3 + [{"user_id": None}],
            "b": [{"user_id": "u1"}, {"user_id": "u3"}],
        }
        st = sdp.subject_stats(cs, "user_id")
        assert st["per_client_subjects"] == {"a": 2, "b": 2}
        assert st["per_client_singletons"] == {"a": 1, "b": 0}
        assert st["n_multi_client_subjects"] == 1
        assert st["multi_client_weight_vectors"] == [
            {"weights": {"a": 0.75, "b": 0.25}, "n_subjects": 1}]


# ── accounting ──────────────────────────────────────────────────────────────────

class TestAccounting:
    def test_un_client_coincide_con_dp_accounting(self):
        from dp_accounting import dp_event, rdp
        eps, q, steps = sdp.epsilon_single_client(100, 10, 5, 3, 1.5, 1e-5)
        acc = rdp.RdpAccountant()
        acc.compose(dp_event.PoissonSampledDpEvent(0.1, dp_event.GaussianDpEvent(1.5)), 10 * 5 * 3)
        assert q == pytest.approx(0.1) and steps == 150
        assert eps == pytest.approx(acc.get_epsilon(target_delta=1e-5))

    def test_peso_ridotto_riduce_epsilon(self):
        e1, _, _ = sdp.epsilon_single_client(100, 10, 5, 3, 1.5, 1e-5, weight=1.0)
        e_half, _, _ = sdp.epsilon_single_client(100, 10, 5, 3, 1.5, 1e-5, weight=0.5)
        assert e_half < e1

    def test_soggetto_su_due_client_con_pesi_meta(self):
        pcs = {"a": 100, "b": 100}
        e_multi = sdp.epsilon_multi_client({"a": 0.5, "b": 0.5}, pcs, 10, 5, 3, 1.5, 1e-5)
        e_single, _, _ = sdp.epsilon_single_client(100, 10, 5, 3, 1.5, 1e-5)
        # due meccanismi con sensibilita' dimezzata: non peggio di uno a piena
        assert e_multi <= e_single + 1e-9
        # senza i pesi (w = 1 in entrambi) sarebbe la composizione di due
        e_nopesi = sdp.epsilon_multi_client({"a": 1.0, "b": 1.0}, pcs, 10, 5, 3, 1.5, 1e-5)
        assert e_nopesi > e_single

    def test_campi_json(self):
        cfg = {"experiment": {"fl_rounds": 3, "delta": 1e-5},
               "ml": {"epochs": 5, "subject_dp": {
                   "enabled": True, "noise_multiplier": 1.5, "expected_subjects_per_step": 10}}}
        stats = {"subject_key": "user_id", "per_client_subjects": {"a": 100, "b": 50},
                 "multi_client_weight_vectors": [{"weights": {"a": 0.5, "b": 0.5}, "n_subjects": 2}]}
        out = sdp.subject_dp_fields(cfg, stats)
        attesi = list(out["epsilon_subject_dp_per_client"].values()) + [
            out["epsilon_subject_dp_multi_client"]]
        assert out["epsilon_subject_dp"] == pytest.approx(max(attesi))
        assert "NON il campo 'epsilon'" in out["subject_dp_accounting_note"]

    def test_campi_json_disattiva_e_senza_rumore(self):
        assert sdp.subject_dp_fields({"ml": {}}, None)["subject_dp"] is None
        cfg = {"experiment": {"fl_rounds": 3, "delta": 1e-5},
               "ml": {"epochs": 5, "subject_dp": {"enabled": True, "noise_multiplier": 0.0}}}
        out = sdp.subject_dp_fields(cfg, {"per_client_subjects": {"a": 10}})
        assert out["epsilon_subject_dp"] is None
        assert "nessuna garanzia" in out["subject_dp_accounting_note"]


# ── trainer ─────────────────────────────────────────────────────────────────────

def _ml(**sd: Any) -> dict:
    base = {"enabled": True, "subject_key": "user_id", "max_grad_norm": 0.5,
            "noise_multiplier": 1.0, "expected_subjects_per_step": 4,
            "max_records_per_subject": 3}
    base.update(sd)
    return {"input_dim": 6, "lr": 0.01, "epochs": 2, "batch_size": 8,
            "proximal_mu": 0.0, "seed": 42, "subject_dp": base}


def _make_sessions(n: int, seed: int, n_users: int = 6) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    base = datetime(2020, 1, 1)
    out = []
    for i in range(n):
        start = base + timedelta(hours=rng.randint(0, 24 * 60), minutes=rng.randint(0, 59))
        end = start + timedelta(hours=rng.uniform(0.5, 8.0))
        out.append({
            "session_id": f"s-{seed}-{i}",
            "user_id": (f"u{rng.randrange(n_users)}" if i % 5 else None),
            "start_time": start.isoformat(), "end_time": end.isoformat(),
            "total_energy_kwh": rng.uniform(1.0, 30.0), "max_power_kw": rng.uniform(3.0, 22.0),
            "kwh_requested": rng.uniform(1.0, 30.0), "minutes_available": rng.uniform(30, 600),
        })
    return run_exp.enrich_sessions(out)


class TestTrainer:
    def test_norm_forzata_a_group(self):
        t = AutoencoderTrainer(config=_ml(), node_id="n", cluster_id="c")
        assert t.norm == "group"
        cfg = {"ml": _ml()}
        assert run_exp._autoencoder_arch_kwargs(cfg)["norm"] == "group"

    def test_record_e_subject_insieme_rifiutati(self):
        ml = _ml()
        ml["record_dp"] = {"enabled": True, "max_grad_norm": 1.0, "noise_multiplier": 1.0}
        with pytest.raises(ValueError, match="insieme"):
            AutoencoderTrainer(config=ml, node_id="n", cluster_id="c")

    def test_sensibilita_per_soggetto(self):
        """Aggiungere o togliere un soggetto cambia la somma tagliata (prima del
        rumore) di al piu' C * w, anche se il soggetto ha molte sessioni."""
        torch.manual_seed(0)
        t = AutoencoderTrainer(config=_ml(max_grad_norm=0.05), node_id="n", cluster_id="c")
        x = torch.rand(40, 6) * 10.0  # dati grandi: gradienti ben oltre C
        rows = [torch.arange(0, 10), torch.arange(10, 30), torch.arange(30, 40)]
        con, _ = t._subject_clipped_sum(x, rows, [1.0, 0.4, 1.0], [11, 12, 13])
        senza, _ = t._subject_clipped_sum(x, [rows[0], rows[2]], [1.0, 1.0], [11, 13])
        diff = math.sqrt(sum(float(((a - b) ** 2).sum()) for a, b in zip(con, senza)))
        assert diff <= 0.05 * 0.4 + 1e-6

    def test_cap_dei_record_per_soggetto(self):
        t = AutoencoderTrainer(config=_ml(max_records_per_subject=2), node_id="n", cluster_id="c")
        x = torch.rand(30, 6)
        _, losses = t._subject_clipped_sum(x, [torch.arange(0, 30)], [1.0], [0])
        assert len(losses) == 1  # un soggetto, un gradiente

    def test_train_local_deterministico_senza_rumore(self):
        ss = _make_sessions(60, seed=3)
        risultati = []
        for _ in range(2):
            torch.manual_seed(7)
            t = AutoencoderTrainer(config=_ml(noise_multiplier=0.0), node_id="n", cluster_id="c")
            upd = t.train_local(ss, round_num=1)
            risultati.append(upd.weights)
        for a, b in zip(*risultati):
            assert torch.equal(a, b)

    def test_train_local_cambia_i_pesi(self):
        ss = _make_sessions(60, seed=4)
        torch.manual_seed(7)
        t = AutoencoderTrainer(config=_ml(), node_id="n", cluster_id="c")
        prima = [w.clone() for w in t.get_weights()]
        upd = t.train_local(ss, round_num=1)
        assert upd.n_samples == len(ss)
        assert any(not torch.equal(a, b) for a, b in zip(prima, upd.weights))

    def test_senza_subject_dp_nulla_cambia(self):
        """Config senza subject_dp: stessi pesi di prima della modifica (il
        ramo DataLoader e' identico)."""
        ss = _make_sessions(64, seed=5)
        ml = {"input_dim": 6, "lr": 0.01, "epochs": 2, "batch_size": 8,
              "proximal_mu": 0.0, "seed": 42}
        out = []
        for cfg in (ml, {**ml, "subject_dp": {"enabled": False}}):
            torch.manual_seed(11)
            t = AutoencoderTrainer(config=cfg, node_id="n", cluster_id="c")
            out.append(t.train_local(ss, round_num=1).weights)
        for a, b in zip(*out):
            assert torch.equal(a, b)


# ── run_fl_rounds ───────────────────────────────────────────────────────────────

@pytest.fixture()
def cfg_rounds() -> dict:
    return {
        "experiment": {"name": "pytest_subject_dp", "fl_rounds": 2, "seed": 42,
                       "epsilon": 1.0, "delta": 1.0e-5, "max_grad_norm": 1.0},
        "ml": {k: v for k, v in _ml().items() if k != "seed"},
        "lira": {"n_shadow": 2},
        "byzantine_attack": {"enabled": False},
    }


class TestRunFLRounds:
    def test_richiede_no_dp(self, cfg_rounds):
        cs = {"a": _make_sessions(30, 1), "b": _make_sessions(30, 2)}
        with pytest.raises(ValueError, match="no-dp"):
            run_exp.run_fl_rounds(copy.deepcopy(cfg_rounds), [], no_dp=False, cluster_sessions=cs)

    def test_imposta_i_pesi_dei_soggetti_condivisi(self, cfg_rounds, monkeypatch):
        cs = {"a": _make_sessions(30, 1, n_users=3), "b": _make_sessions(30, 2, n_users=3)}
        visti: dict[str, dict] = {}
        orig = AutoencoderTrainer.set_subject_weights

        def spia(self, w):
            visti[self.cluster_id] = dict(w)
            return orig(self, w)

        monkeypatch.setattr(AutoencoderTrainer, "set_subject_weights", spia)
        res = run_exp.run_fl_rounds(copy.deepcopy(cfg_rounds), [], no_dp=True, cluster_sessions=cs)
        assert set(visti) == {"a", "b"}
        comuni = set(visti["a"]) & set(visti["b"])
        assert comuni  # u0..u2 compaiono in entrambi i client
        for s in comuni:
            assert visti["a"][s] + visti["b"][s] == pytest.approx(1.0)
        assert 2 in res

    def test_campi_nel_json(self, cfg_rounds, tmp_path):
        cs = {"a": _make_sessions(30, 1), "b": _make_sessions(30, 2)}
        stats = sdp.subject_stats(cs, "user_id")
        cfg = copy.deepcopy(cfg_rounds)
        cfg["output"] = {"experiments_dir": str(tmp_path)}
        path = run_exp.save_results(cfg, {}, {}, {}, sweep_dir=tmp_path,
                                    n_train_sessions=60,
                                    n_train_per_client={"a": 30, "b": 30},
                                    subject_stats=stats)
        import json
        d = json.loads(Path(path).read_text())
        c = d["config"]
        assert c["epsilon_subject_dp"] is not None and c["epsilon_subject_dp"] > 0
        assert c["subject_dp_stats"]["per_client_subjects"] == stats["per_client_subjects"]


# ── correzioni dopo la revisione indipendente (2026-10-13) ─────────────────────

class TestCorrezioniRevisione:
    def test_chiavi_canoniche_e_nan(self):
        assert sdp.subject_of({"user_id": 22}, "user_id", "x") == \
            sdp.subject_of({"user_id": "000022"}, "user_id", "y") == \
            sdp.subject_of({"user_id": 22.0}, "user_id", "z") == "user_id:22"
        assert sdp.subject_of({"user_id": float("nan")}, "user_id", "c:3") == "solo:c:3"
        assert sdp.subject_of({"user_id": " None "}, "user_id", "c:4") == "solo:c:4"

    def test_enabled_stringa(self):
        assert sdp.subject_dp_config({"subject_dp": {"enabled": "false"}})["enabled"] is False
        assert sdp.subject_dp_config({"subject_dp": {"enabled": "true"}})["enabled"] is True

    def test_righe_non_valide_escluse_dal_conteggio(self):
        """Il trainer scarta le righe con feature mancanti: l'accountant deve
        contare gli stessi soggetti (prima li contava tutti)."""
        feats = ["a", "b"]
        cs = {"c": [{"user_id": "u1", "a": 1, "b": 2},
                    {"user_id": "u2", "a": None, "b": 2},
                    {"user_id": "u3", "a": "x", "b": 2},
                    {"user_id": "u4", "a": 1, "b": 2}]}
        st = sdp.subject_stats(cs, "user_id", feats)
        assert st["per_client_subjects"] == {"c": 2}

    def test_trainer_e_accountant_stesso_n(self):
        ss = _make_sessions(50, seed=8)
        for s in ss[:10]:
            s["kwh_requested"] = None  # righe non valide per il trainer
        t = AutoencoderTrainer(config=_ml(), node_id="n", cluster_id="c")
        t.train_local(ss, round_num=1)
        st = sdp.subject_stats({"c": ss}, "user_id", list(t.CONTINUOUS_FEATURES))
        assert t.subject_dp_last["n_soggetti"] == st["per_client_subjects"]["c"]

    def test_denominatore_costante_con_q_uno(self, monkeypatch):
        """Con meno soggetti di quelli attesi per passo (q = 1) il gradiente
        si divide comunque per il numero atteso, non per N."""
        ss = _make_sessions(12, seed=9, n_users=3)
        t = AutoencoderTrainer(config=_ml(expected_subjects_per_step=50, noise_multiplier=0.0),
                               node_id="n", cluster_id="c")
        visti = []
        orig = t.optimizer.step

        def spia(*a, **k):
            visti.append([p.grad.clone() for p in t.model.parameters() if p.grad is not None])
            return orig(*a, **k)

        monkeypatch.setattr(t.optimizer, "step", spia)
        somma_attesa = {}

        orig_sum = t._subject_clipped_sum

        def spia_sum(*a, **k):
            s, l = orig_sum(*a, **k)
            somma_attesa["s"] = [x.clone() for x in s]
            return s, l

        monkeypatch.setattr(t, "_subject_clipped_sum", spia_sum)
        t.epochs = 1
        t.train_local(ss, round_num=1)
        assert t.subject_dp_last["q"] == 1.0
        for g, s in zip(visti[-1], somma_attesa["s"]):
            assert torch.allclose(g, s / 50.0, atol=1e-7)

    def test_campionamento_diverso_fra_client(self):
        ss = _make_sessions(80, seed=10, n_users=40)
        scelti = {}
        for cid in ("a", "b"):
            torch.manual_seed(1)
            t = AutoencoderTrainer(config=_ml(noise_multiplier=0.0), node_id=cid, cluster_id=cid)
            registro = []
            orig = t._subject_clipped_sum

            def spia(x, rows, w, seeds, _o=orig, _r=registro):
                _r.append(tuple(int(r[0]) for r in rows))
                return _o(x, rows, w, seeds)

            t._subject_clipped_sum = spia
            t.epochs = 1
            t.train_local(ss, round_num=1)
            scelti[cid] = registro
        assert scelti["a"] != scelti["b"]

    def test_client_sintetici_rifiutati(self, cfg_rounds):
        cs = {"a": _make_sessions(30, 1), "synthetic_1": _make_sessions(30, 2)}
        with pytest.raises(ValueError, match="sintetici"):
            run_exp.run_fl_rounds(copy.deepcopy(cfg_rounds), [], no_dp=True, cluster_sessions=cs)


class TestPrivacyDiGruppo:
    def test_k1_coincide_con_dp_accounting(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        import privacy_di_gruppo as pg
        from dp_accounting import dp_event, rdp
        q, s = 32 / 25160, 1.0
        ordini, mio = pg.rdp_gruppo_poisson(q, s, 1)
        acc = rdp.RdpAccountant(orders=ordini)
        acc.compose(dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(s)), 1)
        for a, b in zip(acc._rdp, mio):
            assert b == pytest.approx(a, rel=1e-3)

    def test_ordine_2_forma_chiusa(self):
        import numpy as np
        from scipy.stats import binom
        import privacy_di_gruppo as pg
        q, s, k = 32 / 1335, 2.0, 20
        w = binom.pmf(np.arange(k + 1), k, q)
        i = np.arange(k + 1)
        lw = np.log(w)[:, None] + np.log(w)[None, :] + np.outer(i, i) / s ** 2
        esatto = float(np.log(np.exp(lw - lw.max()).sum()) + lw.max())
        _, v = pg.rdp_gruppo_poisson(q, s, k, ordini=[2])
        assert v[0] == pytest.approx(esatto, rel=1e-6)


def test_architettura_coerente_con_enabled_stringa():
    """subject_dp.enabled = "false": trainer e ricostruzione degli attacchi
    devono usare la stessa norm (trovato nella review finale)."""
    ml = _ml(enabled="false")
    t = AutoencoderTrainer(config=ml, node_id="n", cluster_id="c")
    assert t.norm == run_exp._autoencoder_arch_kwargs({"ml": ml})["norm"] == "batch"
