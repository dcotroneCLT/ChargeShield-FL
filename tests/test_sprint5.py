# tests/test_sprint5.py
# ChargeShield-FL — Sprint 5: ML Plane Unit Tests
#
# Testa: AbstractMLModel, AutoencoderTrainer, GradientManager, FedAvgAggregator
# Requisiti: pytest, torch

from __future__ import annotations

import math
import pytest
import torch

from ml.base_ml import (
    AggregatedUpdate,
    GradientUpdate,
    MLPlaneEvent,
    MLPlaneListener,
)
from ml.autoencoder_trainer import AutoencoderTrainer
from ml.gradient_manager import GradientManager
from ml.fedavg_aggregator import FedAvgAggregator


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def trainer_config():
    return {
        "input_dim": 6,
        "lr": 0.001,
        "epochs": 2,
        "batch_size": 4,
        "proximal_mu": 0.0,
    }

@pytest.fixture
def trainer(trainer_config):
    return AutoencoderTrainer(
        config=trainer_config,
        node_id="node_A_01",
        cluster_id="A",
    )

@pytest.fixture
def dp_config():
    return {
        "epsilon": 1.0,
        "delta": 1.0e-5,
        "max_grad_norm": 1.0,
    }

@pytest.fixture
def sessions():
    """Sessioni EV sintetiche con le 6 feature continue di CONTINUOUS_FEATURES."""
    return [
        {
            "total_energy_kwh": 10.0,
            "max_power_kw": 3.7,
            "kwh_requested": 12.0,
            "minutes_available": 180.0,
            "hour_of_day": 9.0,
            "duration_hours": 3.0,
        }
        for _ in range(20)
    ]

@pytest.fixture
def gradient_update(trainer, sessions):
    return trainer.train_local(sessions, round_num=1)


# ── AutoencoderTrainer ─────────────────────────────────────────────────────────

class TestAutoencoderTrainer:

    def test_init_fedavg(self, trainer):
        assert trainer.proximal_mu == 0.0
        assert trainer._global_weights is None

    def test_init_fedprox(self, trainer_config):
        cfg = {**trainer_config, "proximal_mu": 0.01}
        t = AutoencoderTrainer(cfg, "node_B_01", "B")
        assert t.proximal_mu == 0.01

    def test_missing_config_raises(self):
        with pytest.raises(ValueError):
            AutoencoderTrainer({"lr": 0.001, "epochs": 2, "batch_size": 4},
                               "n", "A")

    def test_get_weights_returns_list(self, trainer):
        weights = trainer.get_weights()
        assert isinstance(weights, list)
        assert len(weights) > 0
        assert all(isinstance(w, torch.Tensor) for w in weights)

    def test_set_weights_roundtrip(self, trainer):
        original = trainer.get_weights()
        # Modifica pesi
        for w in original:
            w.fill_(0.0)
        trainer.set_weights(original)
        restored = trainer.get_weights()
        for w in restored:
            assert torch.all(w == 0.0)

    def test_set_weights_saves_global(self, trainer):
        weights = trainer.get_weights()
        trainer.set_weights(weights)
        assert trainer._global_weights is not None

    def test_train_local_returns_update(self, trainer, sessions):
        update = trainer.train_local(sessions, round_num=1)
        assert isinstance(update, GradientUpdate)
        assert update.node_id == "node_A_01"
        assert update.cluster_id == "A"
        assert update.round_num == 1
        assert update.n_samples == len(sessions)
        assert update.loss is not None
        assert update.loss >= 0.0

    def test_train_local_empty_sessions(self, trainer):
        update = trainer.train_local([], round_num=1)
        assert update.n_samples == 0
        assert update.loss is None

    def test_train_local_none_feature_skipped(self, trainer):
        # Usa una feature reale (total_energy_kwh) con valore None per testare
        # effettivamente la logica di skip in _sessions_to_tensor.
        # Le chiavi precedenti (voltage_v, ecc.) non appartengono a CONTINUOUS_FEATURES
        # e venivano saltate per chiave mancante, non per valore None — test non valido.
        sessions = [
            {
                "total_energy_kwh": None,  # feature reale ACN con valore None → skip
                "max_power_kw": 3.7,
                "kwh_requested": 12.0,
                "minutes_available": 180.0,
                "hour_of_day": 9.0,
                "duration_hours": 3.0,
            },
        ]
        update = trainer.train_local(sessions, round_num=1)
        assert update.n_samples == 0

    def test_fedprox_term_applied(self, trainer_config, sessions):
        cfg = {**trainer_config, "proximal_mu": 0.1}
        t = AutoencoderTrainer(cfg, "node_C_01", "C")
        # Prima imposta global weights, poi train
        t.set_weights(t.get_weights())
        update = t.train_local(sessions, round_num=1)
        assert update.loss is not None

    def test_ml_plane_event_emitted(self, trainer, sessions):
        events: list[MLPlaneEvent] = []

        class Spy(MLPlaneListener):
            def on_ml_event(self, event: MLPlaneEvent) -> None:
                events.append(event)

        trainer.subscribe(Spy())
        trainer.train_local(sessions, round_num=1)
        assert len(events) == 1
        assert events[0].event_type == "gradient_upload"
        assert events[0].purdue_level == 1

    def test_apply_global_model_emits_event(self, trainer, gradient_update):
        events: list[MLPlaneEvent] = []

        class Spy(MLPlaneListener):
            def on_ml_event(self, event: MLPlaneEvent) -> None:
                events.append(event)

        trainer.subscribe(Spy())
        aggregated = AggregatedUpdate(
            round_num=1,
            global_weights=gradient_update.weights,
            n_participants=4,
            mean_loss=0.01,
        )
        trainer.apply_global_model(aggregated)
        assert any(e.event_type == "weight_download" for e in events)


# ── GradientManager ────────────────────────────────────────────────────────────

class TestGradientManager:

    def test_sigma_computed(self, dp_config):
        # formula classica: solo con sigma_calibration "classica" (run vecchie)
        gm = GradientManager({**dp_config, "sigma_calibration": "classica"})
        expected = (
            dp_config["max_grad_norm"]
            * math.sqrt(2 * math.log(1.25 / dp_config["delta"]))
            / dp_config["epsilon"]
        )
        assert abs(gm.sigma - expected) < 1e-6

    def test_sigma_analitica_default(self, dp_config):
        # 2026-10-16: default analitica, sostituzione (sensibilita' 2C)
        from ml.client_dp_accounting import sigma_gaussiano_analitico
        gm = GradientManager(dp_config)
        expected = 2 * dp_config["max_grad_norm"] * sigma_gaussiano_analitico(
            dp_config["epsilon"], dp_config["delta"])
        assert abs(gm.sigma - expected) < 1e-9

    def test_missing_config_raises(self):
        with pytest.raises(ValueError):
            GradientManager({"epsilon": 1.0, "delta": 1e-5})

    def test_privatize_returns_update(self, dp_config, gradient_update):
        gm = GradientManager(dp_config)
        private = gm.privatize(gradient_update)
        assert isinstance(private, GradientUpdate)
        assert private.metadata.get("noise_perturbation_applied") is True

    def test_privatize_changes_weights(self, dp_config, gradient_update):
        torch.manual_seed(42)
        gm = GradientManager(dp_config)
        private = gm.privatize(gradient_update)
        # I pesi floating-point devono essere diversi dopo DP
        orig_fp  = [w for w in gradient_update.weights if w.is_floating_point()]
        priv_fp  = [w for w in private.weights        if w.is_floating_point()]
        original_flat = torch.cat([w.flatten() for w in orig_fp])
        private_flat  = torch.cat([w.flatten() for w in priv_fp])
        assert not torch.allclose(original_flat, private_flat)
        # Verifica che il rumore sia effettivamente non-zero
        noise = private_flat - original_flat
        assert noise.norm().item() > 0.0

    def test_privatize_preserves_metadata(self, dp_config, gradient_update):
        gm = GradientManager(dp_config)
        private = gm.privatize(gradient_update)
        assert private.node_id    == gradient_update.node_id
        assert private.cluster_id == gradient_update.cluster_id
        assert private.round_num  == gradient_update.round_num
        assert private.n_samples  == gradient_update.n_samples

    def test_ml_plane_event_emitted(self, dp_config, gradient_update):
        events: list[MLPlaneEvent] = []

        class Spy(MLPlaneListener):
            def on_ml_event(self, event: MLPlaneEvent) -> None:
                events.append(event)

        gm = GradientManager(dp_config)
        gm.subscribe(Spy())
        gm.privatize(gradient_update)
        assert len(events) == 1
        assert events[0].purdue_level == 2

    def test_clipping_reduces_norm(self, dp_config, gradient_update):
        gm = GradientManager(dp_config)
        clipped = gm._clip_weights(gradient_update.weights)
        # Solo tensori float: _clip_weights non tocca i buffer int64 di BatchNorm
        # (num_batches_tracked). La garanzia norm ≤ max_grad_norm vale solo per i
        # parametri floating-point — non per il vettore pesi completo.
        float_tensors = [w for w in clipped if w.is_floating_point()]
        flat = torch.cat([w.flatten() for w in float_tensors])
        norm = float(torch.norm(flat, p=2))
        assert norm <= dp_config["max_grad_norm"] + 1e-5

    def test_clip_weights_with_reference_bounds_delta_not_absolute_norm(
        self, dp_config, gradient_update
    ):
        """
        Regressione per il fix 2026-07-22 (review B1): _clip_weights() con
        `reference` fornito deve bound la norma del DELTA (weights - reference),
        NON la norma assoluta del vettore pesi risultante. Prima del fix,
        _clip_weights() clippava sempre l'assoluto — questo test lo dimostra
        costruendo un reference "lontano" (pesi con norma assoluta grande) e
        verificando che il risultato clippato resti VICINO al reference (norma
        assoluta grande, quindi ben oltre max_grad_norm), non vicino a zero.
        """
        gm = GradientManager(dp_config)
        float_weights = [w for w in gradient_update.weights if w.is_floating_point()]

        # Reference "lontano dall'origine": ogni tensore + 100 — la sua norma
        # assoluta è enorme (>> max_grad_norm=1.0), ma il delta rispetto ad
        # esso (i pesi originali - reference) è piccolo (i pesi veri sono
        # vicini a reference, non all'origine).
        reference = [w + 100.0 for w in gradient_update.weights]
        clipped = gm._clip_weights(gradient_update.weights, reference=reference)

        clipped_float = [c for c, w in zip(clipped, gradient_update.weights) if w.is_floating_point()]
        ref_float     = [r for r, w in zip(reference, gradient_update.weights) if w.is_floating_point()]

        # Il delta (clippato - reference) deve avere norma <= max_grad_norm.
        delta = torch.cat([(c - r).flatten() for c, r in zip(clipped_float, ref_float)])
        delta_norm = float(torch.norm(delta, p=2))
        assert delta_norm <= dp_config["max_grad_norm"] + 1e-3, (
            f"delta norm={delta_norm:.4f} supera max_grad_norm — il clipping "
            "con reference non sta bound-ando il delta come atteso"
        )

        # Il risultato deve restare VICINO al reference (norma assoluta grande),
        # non essere stato tirato verso zero come nel vecchio clipping assoluto.
        clipped_abs_norm = float(torch.cat([c.flatten() for c in clipped_float]).norm())
        assert clipped_abs_norm > 50.0, (
            f"norma assoluta del risultato clippato = {clipped_abs_norm:.2f} — "
            "atteso vicino al reference (norma grande, offset +100), non "
            "vicino a zero: suggerisce che il clipping stia ancora operando "
            "sul vettore assoluto invece che sul delta (regressione fix 2026-07-22)"
        )

    def test_clip_only_does_not_add_noise(self, dp_config, gradient_update):
        """clip_only() (central DP, 2026-07-22): stessa garanzia di norma di
        _clip_weights(), ma senza alcun rumore — i pesi restituiti devono essere
        deterministici (identici a _clip_weights() diretto), non un draw casuale."""
        gm = GradientManager(dp_config)
        clipped_direct = gm._clip_weights(gradient_update.weights)
        result = gm.clip_only(gradient_update)
        assert result.metadata.get("clipped_only_central_dp") is True
        assert "noise_perturbation_applied" not in result.metadata
        for a, b in zip(clipped_direct, result.weights):
            assert torch.equal(a, b), (
                "clip_only() ha prodotto pesi diversi da _clip_weights() diretto — "
                "sospetto che stia aggiungendo rumore o applicando un clip diverso"
            )

    def test_privatize_aggregate_sigma_scales_with_n_participants(self, dp_config):
        """
        Regressione per privatize_aggregate() (central DP, 2026-07-22): il rumore
        sull'aggregato deve avere sigma = self.sigma / n_participants — non
        self.sigma tal quale (che sarebbe equivalente a NON applicare la
        riduzione di sensibilità 1/n di McMahan et al. 2018) e non un fattore
        arbitrario diverso.

        Verifica empirica (non solo "il risultato differisce dall'input", che
        passerebbe anche con uno scaling sbagliato): genera molti draw di rumore
        con n_participants=1 e n_participants=4 su un tensore di zeri grande a
        sufficienza da stimare la deviazione standard empirica, e verifica che il
        rapporto delle std misurate sia vicino a 4 (entro tolleranza statistica).
        """
        gm = GradientManager(dp_config)
        big_zero = [torch.zeros(4000)]

        torch.manual_seed(123)
        noised_n1 = gm.privatize_aggregate(big_zero, n_participants=1)
        std_n1 = float(noised_n1[0].std())

        torch.manual_seed(123)
        noised_n4 = gm.privatize_aggregate(big_zero, n_participants=4)
        std_n4 = float(noised_n4[0].std())

        assert std_n1 > 0.0 and std_n4 > 0.0
        ratio = std_n1 / std_n4
        assert 3.5 < ratio < 4.5, (
            f"rapporto atteso delle std sigma_n1/sigma_n4 ≈ 4 (n_participants "
            f"scala sigma come 1/n), osservato {ratio:.2f} — la riduzione di "
            "sensitività 1/n di privatize_aggregate() potrebbe essere rotta"
        )


# ── FedAvgAggregator ───────────────────────────────────────────────────────────

class TestFedAvgAggregator:

    def _make_update(self, node_id: str, cluster_id: str,
                     weights: list, n_samples: int, loss: float) -> GradientUpdate:
        return GradientUpdate(
            node_id=node_id,
            cluster_id=cluster_id,
            round_num=1,
            weights=weights,
            gradients=None,
            loss=loss,
            n_samples=n_samples,
        )

    def test_aggregate_returns_result(self, trainer):
        agg = FedAvgAggregator({"min_participants": 2})
        w = trainer.get_weights()
        agg.collect(self._make_update("n1", "A", w, 100, 0.1))
        agg.collect(self._make_update("n2", "B", w, 100, 0.2))
        result = agg.aggregate(round_num=1)
        assert result is not None
        assert result.n_participants == 2
        assert result.round_num == 1

    def test_aggregate_weighted_average(self, trainer):
        agg = FedAvgAggregator({"min_participants": 2})
        w1 = [torch.ones(p.shape) * 0.0 for p in trainer.model.parameters()]
        w2 = [torch.ones(p.shape) * 1.0 for p in trainer.model.parameters()]
        agg.collect(self._make_update("n1", "A", w1, 50, 0.1))
        agg.collect(self._make_update("n2", "B", w2, 50, 0.2))
        result = agg.aggregate(round_num=1)
        # Media pesata 50/50 → 0.5
        for w in result.global_weights:
            assert torch.allclose(w, torch.ones_like(w) * 0.5, atol=1e-5)

    def test_aggregate_incompatible_weights_excluded_from_metadata(self, trainer):
        """Fix 2026-09-11 (deep review round 2): un update con struttura pesi
        incompatibile (len diverso) veniva scartato da global_weights ma restava
        conteggiato in total_samples/participant_ids/participant_n_samples —
        quest'ultimo alimenta max_weight_fraction in
        GradientManager.privatize_aggregate() (sensibilità del rumore DP central),
        quindi un partecipante fantasma ne distorceva silenziosamente il calibro."""
        agg = FedAvgAggregator({"min_participants": 2})
        w_ok = trainer.get_weights()
        w_bad = w_ok[:-1]  # un tensore in meno → struttura incompatibile
        agg.collect(self._make_update("n1", "A", w_ok, 100, 0.1))
        agg.collect(self._make_update("n2", "B", w_ok, 100, 0.1))
        agg.collect(self._make_update("n_bad", "C", w_bad, 900, 0.9))
        result = agg.aggregate(round_num=1)
        assert result is not None
        assert result.n_participants == 2
        assert "n_bad" not in result.metadata["participant_ids"]
        assert "n_bad" not in result.metadata["participant_n_samples"]
        assert result.metadata["total_samples"] == 200

    def test_aggregate_below_min_returns_none(self, trainer):
        agg = FedAvgAggregator({"min_participants": 3})
        w = trainer.get_weights()
        agg.collect(self._make_update("n1", "A", w, 100, 0.1))
        agg.collect(self._make_update("n2", "B", w, 100, 0.2))
        result = agg.aggregate(round_num=1)
        assert result is None

    def test_aggregate_clears_buffer(self, trainer):
        agg = FedAvgAggregator({"min_participants": 2})
        w = trainer.get_weights()
        agg.collect(self._make_update("n1", "A", w, 100, 0.1))
        agg.collect(self._make_update("n2", "B", w, 100, 0.2))
        agg.aggregate(round_num=1)
        # Secondo aggregate senza nuovi update → None
        result = agg.aggregate(round_num=2)
        assert result is None

    def test_mean_loss_weighted(self, trainer):
        """La media pesata FedAvg con n_samples asimmetrici deve differire dalla media semplice."""
        agg = FedAvgAggregator({"min_participants": 2})
        w = trainer.get_weights()
        # n1: 100 campioni, loss=0.0 — n2: 300 campioni, loss=1.0
        # Media pesata: (0.0*100 + 1.0*300) / 400 = 0.75
        # Media semplice: (0.0 + 1.0) / 2 = 0.50 — diversa, il test ora verifica davvero
        agg.collect(self._make_update("n1", "A", w, 100, 0.0))
        agg.collect(self._make_update("n2", "B", w, 300, 1.0))
        result = agg.aggregate(round_num=1)
        assert result.mean_loss is not None
        assert abs(result.mean_loss - 0.75) < 1e-5

    def test_mean_loss_equal_weights_average(self, trainer):
        """Con n_samples identici la media pesata coincide con la media semplice."""
        agg = FedAvgAggregator({"min_participants": 2})
        w = trainer.get_weights()
        agg.collect(self._make_update("n1", "A", w, 100, 0.1))
        agg.collect(self._make_update("n2", "B", w, 100, 0.3))
        result = agg.aggregate(round_num=1)
        assert result.mean_loss is not None
        assert abs(result.mean_loss - 0.2) < 1e-5

    def test_min_participants_vs_valid_filter(self, trainer):
        """
        Se il filtro valid riduce i partecipanti sotto min_participants,
        aggregate() restituisce None (non aggrega con partecipanti insufficienti).
        Con 3 update di cui 2 con n_samples=0: valid=[n3], len(valid)=1 < min_participants=2.
        """
        agg = FedAvgAggregator({"min_participants": 2})
        w = trainer.get_weights()
        agg.collect(self._make_update("n1", "A", w, 0,   0.1))  # invalido
        agg.collect(self._make_update("n2", "B", w, 0,   0.2))  # invalido
        agg.collect(self._make_update("n3", "C", w, 100, 0.3))  # valido
        result = agg.aggregate(round_num=1)
        # 1 solo nodo valido < min_participants=2 → aggregazione rifiutata
        assert result is None

    def test_min_participants_vs_valid_filter_sufficient(self, trainer):
        """
        Se il filtro valid lascia esattamente min_participants nodi, aggregate() ha successo.
        Con 3 update di cui 1 con n_samples=0: valid=[n1, n2], len(valid)=2 == min_participants=2.
        """
        agg = FedAvgAggregator({"min_participants": 2})
        w = trainer.get_weights()
        agg.collect(self._make_update("n1", "A", w, 50,  0.1))  # valido
        agg.collect(self._make_update("n2", "B", w, 50,  0.2))  # valido
        agg.collect(self._make_update("n3", "C", w, 0,   0.3))  # invalido
        result = agg.aggregate(round_num=1)
        assert result is not None
        assert result.n_participants == 2

    def test_set_weights_bn_buffer_dtype(self, trainer):
        """num_batches_tracked (buffer BN) deve mantenere dtype int64 dopo il round-trip."""
        weights = trainer.get_weights()
        trainer.set_weights(weights)
        restored = trainer.get_weights()
        state = trainer.model.state_dict()
        # Cerca num_batches_tracked — deve essere int64
        for key, tensor in state.items():
            if "num_batches_tracked" in key:
                assert tensor.dtype == torch.int64, (
                    f"{key} ha dtype {tensor.dtype} invece di torch.int64"
                )

    def test_ml_plane_event_emitted(self, trainer):
        events: list[MLPlaneEvent] = []

        class Spy(MLPlaneListener):
            def on_ml_event(self, event: MLPlaneEvent) -> None:
                events.append(event)

        agg = FedAvgAggregator({"min_participants": 2})
        agg.subscribe(Spy())
        w = trainer.get_weights()
        agg.collect(self._make_update("n1", "A", w, 100, 0.1))
        agg.collect(self._make_update("n2", "B", w, 100, 0.2))
        agg.aggregate(round_num=1)
        assert len(events) == 1
        assert events[0].event_type == "aggregation"
        assert events[0].purdue_level == 3


# ── Normalization Functions ────────────────────────────────────────────────────
# Testa compute_feature_stats() e normalize_sessions() di run_experiments.py.
# Queste due funzioni sono critiche: un bug produce valori fuori [0,1] e
# corrompe il MSE loss senza errori visibili.

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent / "scripts"))

from run_experiments import compute_feature_stats, normalize_sessions

_FEATURES = [
    "total_energy_kwh", "max_power_kw", "kwh_requested",
    "minutes_available", "hour_of_day", "duration_hours",
]


@pytest.fixture
def raw_sessions():
    return [
        {"total_energy_kwh": 0.0,  "max_power_kw": 1.0, "kwh_requested": 2.0,
         "minutes_available": 0.0, "hour_of_day": 0.0,  "duration_hours": 0.0},
        {"total_energy_kwh": 10.0, "max_power_kw": 5.0, "kwh_requested": 8.0,
         "minutes_available": 100.0, "hour_of_day": 12.0, "duration_hours": 2.0},
        {"total_energy_kwh": 20.0, "max_power_kw": 9.0, "kwh_requested": 14.0,
         "minutes_available": 200.0, "hour_of_day": 23.0, "duration_hours": 4.0},
    ]


class TestNormalization:
    # compute_feature_stats restituisce {feat: (fmin, fmax)} — tuple, non dict.

    def test_compute_feature_stats_keys(self, raw_sessions):
        stats = compute_feature_stats(raw_sessions, _FEATURES)
        for feat in _FEATURES:
            assert feat in stats, f"Feature '{feat}' mancante da stats"
            assert isinstance(stats[feat], tuple) and len(stats[feat]) == 2

    def test_compute_feature_stats_values(self, raw_sessions):
        stats = compute_feature_stats(raw_sessions, _FEATURES)
        fmin, fmax = stats["total_energy_kwh"]
        assert fmin == pytest.approx(0.0)
        assert fmax == pytest.approx(20.0)
        hmin, hmax = stats["hour_of_day"]
        assert hmin == pytest.approx(0.0)
        assert hmax == pytest.approx(23.0)

    def test_normalize_sessions_range(self, raw_sessions):
        """Tutti i valori normalizzati devono essere in [0, 1]."""
        stats = compute_feature_stats(raw_sessions, _FEATURES)
        normalized = normalize_sessions(raw_sessions, stats, _FEATURES)
        for s in normalized:
            for feat in _FEATURES:
                val = s[feat]
                assert 0.0 - 1e-6 <= val <= 1.0 + 1e-6, (
                    f"{feat}={val} fuori da [0,1]"
                )

    def test_normalize_sessions_min_is_zero(self, raw_sessions):
        """Il minimo (prima sessione) deve essere normalizzato a 0."""
        stats = compute_feature_stats(raw_sessions, _FEATURES)
        normalized = normalize_sessions(raw_sessions, stats, _FEATURES)
        assert normalized[0]["total_energy_kwh"] == pytest.approx(0.0)

    def test_normalize_sessions_max_is_one(self, raw_sessions):
        """Il massimo (ultima sessione) deve essere normalizzato a 1."""
        stats = compute_feature_stats(raw_sessions, _FEATURES)
        normalized = normalize_sessions(raw_sessions, stats, _FEATURES)
        assert normalized[-1]["total_energy_kwh"] == pytest.approx(1.0)

    def test_normalize_sessions_does_not_mutate_input(self, raw_sessions):
        """normalize_sessions non deve modificare le sessioni originali."""
        import copy
        original = copy.deepcopy(raw_sessions)
        stats = compute_feature_stats(raw_sessions, _FEATURES)
        normalize_sessions(raw_sessions, stats, _FEATURES)
        for orig, current in zip(original, raw_sessions):
            for feat in _FEATURES:
                assert orig[feat] == current[feat], (
                    f"Feature '{feat}' mutata: {orig[feat]} → {current[feat]}"
                )

    def test_normalize_constant_feature_no_nan(self):
        """Con min==max (feature costante), compute_feature_stats usa fmin+1.0
        come fmax per evitare divisione per zero. Il risultato deve essere 0.0
        e non NaN."""
        sessions = [
            {"total_energy_kwh": 5.0, "max_power_kw": 3.0, "kwh_requested": 8.0,
             "minutes_available": 60.0, "hour_of_day": 8.0, "duration_hours": 1.0},
            {"total_energy_kwh": 5.0, "max_power_kw": 7.0, "kwh_requested": 10.0,
             "minutes_available": 120.0, "hour_of_day": 9.0, "duration_hours": 2.0},
        ]
        stats = compute_feature_stats(sessions, _FEATURES)
        # compute_feature_stats gestisce il caso costante: fmax = fmin + 1.0
        fmin, fmax = stats["total_energy_kwh"]
        assert fmax == fmin + 1.0  # non divide per zero
        normalized = normalize_sessions(sessions, stats, _FEATURES)
        # Valore costante → (5.0 - 5.0) / 1.0 = 0.0, non NaN
        for s in normalized:
            val = s["total_energy_kwh"]
            assert val == val, "NaN rilevato in feature costante normalizzata"
            assert val == pytest.approx(0.0)
