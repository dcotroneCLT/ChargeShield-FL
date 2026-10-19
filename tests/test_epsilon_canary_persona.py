"""scripts/epsilon_canary_persona.py: epsilon della persona canary dai log (2026-10-18)."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

pytest.importorskip("dp_accounting")

QUI = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("eps_cp", QUI.parent / "scripts" / "epsilon_canary_persona.py")
eps_cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eps_cp)

from ml import subject_dp  # noqa: E402


def _scrivi(tmp, sigma, epoche=10, rounds=2, n_jpl=100, n_altro=50):
    esiti = tmp / "esiti"
    esiti.mkdir()
    nome = f"canary_persona_indotto_x_c30_d0.5_r{rounds}_e{epoche}_A_s42.json"
    (esiti / nome).write_text(json.dumps({"sito": "jpl", "epochs": epoche, "rounds": rounds}))
    righe = []
    for r in range(1, rounds + 1):
        for nodo, n in (("caltech-01", n_altro), ("jpl-01", n_jpl)):
            q, pe = subject_dp.sampling_schedule(n, 32)
            righe.append(f"2026-10-18 [INFO] x: [SUBJECT-DP] [{nodo}] round {r}: {n} soggetti, "
                         f"q={q:.4g}, {pe} passi per epoca, sigma={sigma}, C=1.0, pesi < 1: 0")
    righe.append(f"braccio A, ... -> /Users/x/esiti/canary_persona/{nome}")
    log = tmp / "v2b_cps_x_A_s42.log"
    log.write_text("\n".join(righe) + "\n")
    return log, esiti


def test_epsilon_uguale_all_accountant(tmp_path):
    log, esiti = _scrivi(tmp_path, 2.0)
    r = eps_cp.analizza(log, esiti, 32, 1e-5)
    atteso = subject_dp.epsilon_single_client(100, 32, 10, 2, 2.0, 1e-5)
    assert r["sito"] == "jpl" and r["passi"] == atteso[2]
    assert r["epsilon"] == pytest.approx(atteso[0], rel=1e-3)


def test_solo_taglio_e_senza_dp(tmp_path):
    log, esiti = _scrivi(tmp_path, 0.0)
    assert eps_cp.analizza(log, esiti, 32, 1e-5)["epsilon"] == "inf"
    log.write_text(log.read_text().replace("[SUBJECT-DP]", "[ALTRO]"))
    assert eps_cp.analizza(log, esiti, 32, 1e-5)["epsilon"] == "senza DP per soggetto"


def test_round_mancante_e_passi_incoerenti(tmp_path):
    log, esiti = _scrivi(tmp_path, 2.0)
    testo = log.read_text()
    log.write_text("\n".join(x for x in testo.splitlines() if "[jpl-01] round 2" not in x))
    with pytest.raises(SystemExit):
        eps_cp.analizza(log, esiti, 32, 1e-5)
    log.write_text(testo)
    with pytest.raises(SystemExit):
        eps_cp.analizza(log, esiti, 16, 1e-5)


def test_epsilon_decresce_con_sigma(tmp_path):
    valori = []
    for s in (2.0, 5.0):
        d = tmp_path / str(s)
        d.mkdir()
        log, esiti = _scrivi(d, s)
        valori.append(eps_cp.analizza(log, esiti, 32, 1e-5)["epsilon"])
    assert valori[0] > valori[1] > 0 and all(math.isfinite(v) for v in valori)


def test_json_mancante_saltato(tmp_path):
    log, esiti = _scrivi(tmp_path, 2.0)
    for f in esiti.iterdir():
        f.unlink()
    assert eps_cp.analizza(log, esiti, 32, 1e-5) is None
