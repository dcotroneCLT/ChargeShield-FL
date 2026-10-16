"""Normalizzazione con limiti pubblici (2026-10-16, Segnalazioni R voce 76)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("torch")
import run_experiments as rx  # noqa: E402

F = ["total_energy_kwh", "hour_of_day"]


def test_default_limiti_pubblici_non_dipende_dai_dati():
    a, clip, info = rx.statistiche_normalizzazione({"ml": {}}, [{"total_energy_kwh": 5, "hour_of_day": 3}], F)
    b, _, _ = rx.statistiche_normalizzazione({"ml": {}}, [{"total_energy_kwh": 90, "hour_of_day": 23}], F)
    assert a == b and clip and info["method"] == "limiti_pubblici"
    out = rx.normalize_sessions([{"total_energy_kwh": 250.0, "hour_of_day": 12}], a, F, clip=clip)
    assert out[0]["total_energy_kwh"] == 1.0 and out[0]["hour_of_day"] == 0.5


def test_storico_minmax():
    s, clip, info = rx.statistiche_normalizzazione(
        {"ml": {"normalization": {"method": "minmax_training"}}},
        [{"total_energy_kwh": 5, "hour_of_day": 3}, {"total_energy_kwh": 9, "hour_of_day": 7}], F)
    assert s["total_energy_kwh"] == (5.0, 9.0) and not clip


def test_limiti_mancanti_errore_e_override():
    with pytest.raises(ValueError):
        rx.statistiche_normalizzazione({"ml": {}}, [], ["feature_nuova"])
    s, _, _ = rx.statistiche_normalizzazione(
        {"ml": {"normalization": {"bounds": {"feature_nuova": [0, 2]}}}}, [], ["feature_nuova"])
    assert s["feature_nuova"] == (0.0, 2.0)


def test_scozia():
    s, _, info = rx.statistiche_normalizzazione(
        {"ml": {}, "dataset_adapter": "chargeplace_scotland"}, [], ["max_power_kw"])
    assert info["dataset"] == "chargeplace_scotland" and s["max_power_kw"] == (0.0, 350.0)
