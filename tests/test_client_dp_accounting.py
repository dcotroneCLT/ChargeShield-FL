"""Calibrazione analitica della DP per client (2026-10-16)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml.client_dp_accounting import (  # noqa: E402
    client_dp_fields, delta_gaussiano, epsilon_totale_rdp, sigma_classico,
    sigma_gaussiano_analitico,
)


@pytest.mark.parametrize("eps", [0.5, 1.0, 4.0, 16.0, 64.0, 256.0])
def test_sigma_analitico_raggiunge_delta(eps):
    s = sigma_gaussiano_analitico(eps, 1e-5)
    assert delta_gaussiano(s, eps) <= 1e-5 * (1 + 1e-6)
    assert delta_gaussiano(s * 0.999, eps) > 1e-5      # minimo


def test_classica_insufficiente_per_epsilon_grandi():
    # la formula classica da' meno rumore del necessario per eps grandi
    # (da circa 10 in su con delta = 1e-5; per eps piccoli ne da' di piu')
    for eps in (16.0, 64.0, 256.0):
        assert sigma_classico(eps, 1e-5) < sigma_gaussiano_analitico(eps, 1e-5)
        assert delta_gaussiano(sigma_classico(eps, 1e-5), eps) > 1e-5


def test_lineare_nella_sensibilita():
    assert sigma_gaussiano_analitico(2.0, 1e-5, 3.0) == pytest.approx(
        3.0 * sigma_gaussiano_analitico(2.0, 1e-5, 1.0), rel=1e-9)


def test_rdp_un_round_vicino_a_epsilon():
    pytest.importorskip("dp_accounting")
    s = sigma_gaussiano_analitico(4.0, 1e-5)
    e1 = epsilon_totale_rdp(s, 1, 1e-5)
    assert 4.0 <= e1 < 6.0          # RDP e' un limite superiore, poco largo
    assert epsilon_totale_rdp(s, 10, 1e-5) > e1


def test_campi_json():
    pytest.importorskip("dp_accounting")
    cfg = {"experiment": {"epsilon": 64.0, "delta": 1e-5, "max_grad_norm": 1.0, "fl_rounds": 10}}
    f = client_dp_fields(cfg, "local", False, {"a": 3, "b": 1})["client_dp_accounting"]
    assert f["adiacenza"] == "sostituzione" and f["sensibilita"] == 2.0
    g = client_dp_fields(cfg, "central", False, {"a": 3, "b": 1})["client_dp_accounting"]
    assert g["adiacenza"] == "sostituzione" and g["sensibilita"] == pytest.approx(1.5)
    assert client_dp_fields(cfg, "local", True, None)["client_dp_accounting"] is None
    cfg["experiment"]["dp_ablation"] = "clip-only"
    assert client_dp_fields(cfg, "local", False, None)["client_dp_accounting"]["epsilon_totale_rdp"] is None
