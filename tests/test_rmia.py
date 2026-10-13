"""RMIA dentro run_lira() (2026-10-13): _rmia_scores e _rmia_round_fields.
Nessun dato reale."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_experiments as run_exp  # noqa: E402


def _righe(n: int, effetto: float, seed: int = 0, client: int = 0) -> list[dict]:
    """Membri con loss del bersaglio piu' bassa dei riferimenti di `effetto`."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        membro = i % 2 == 0
        base = rng.uniform(0.5, 2.0)          # difficolta' del campione
        rif = list(base * rng.uniform(0.9, 1.1, size=2))
        t = base * (1.0 - effetto if membro else 1.0) * rng.uniform(0.95, 1.05)
        out.append({"target": float(t), "refs": [(0, float(rif[0]), True), (1, float(rif[1]), False)],
                    "member": membro,
                    "client": client, "sid": i + 1000 * client,
                    "canary": ("g" if i < 20 else None)})
    return out


def test_rileva_un_effetto_e_non_lo_inventa():
    forte = run_exp._rmia_round_fields(_righe(400, 0.3), {}, {}, {})
    nullo = run_exp._rmia_round_fields(_righe(400, 0.0, seed=1), {}, {}, {})
    assert forte["rmia_auc_roc"] > 0.9
    assert abs(nullo["rmia_auc_roc"] - 0.5) < 0.1
    assert forte["rmia_n_samples"] == 400


def test_calibrazione_per_campione():
    """La difficolta' del campione (base) varia di 4 volte: senza i riferimenti
    la loss grezza separerebbe poco; RMIA la normalizza."""
    righe = _righe(400, 0.15, seed=2)
    from sklearn.metrics import roc_auc_score
    grezza = roc_auc_score([r["member"] for r in righe], [-r["target"] for r in righe])
    rmia = run_exp._rmia_round_fields(righe, {}, {}, {})["rmia_auc_roc"]
    assert rmia > grezza + 0.2


def test_popolazione_esclude_x_e_score_in_0_1():
    sc = run_exp._rmia_scores(_righe(50, 0.2, seed=3))
    for kind in ("online", "offline"):
        assert sc[kind]
        assert all(0.0 <= s <= 1.0 for (_sid, _m, _c, s, _l) in sc[kind])


def test_client_separati_e_composizione():
    righe = _righe(100, 0.2, seed=4, client=0) + _righe(100, 0.2, seed=5, client=1)
    cum, mem, can = {}, {}, {}
    run_exp._rmia_round_fields(righe, cum, mem, can)
    run_exp._rmia_round_fields(righe, cum, mem, can)
    assert len(cum) == 200
    assert all(mem[s] == ((s % 1000) % 2 == 0) for s in cum)


def test_canary_auc():
    f = run_exp._rmia_round_fields(_righe(200, 0.3, seed=6), {}, {}, {})
    assert f["canary_rmia_auc_roc"] is not None and f["canary_rmia_auc_roc"] > 0.8


def test_senza_riferimenti_nessun_score():
    righe = [{"target": 1.0, "refs": [], "member": True, "client": 0, "sid": 1, "canary": None}]
    assert run_exp._rmia_scores(righe) == {"online": [], "offline": []}


def test_scala_degli_shadow_diversa_dal_bersaglio():
    """Shadow addestrati meno (loss 6 volte piu' alte): con la normalizzazione
    per modello lo score non si inverte."""
    rng = np.random.default_rng(9)
    righe = []
    for i in range(600):
        membro = i % 2 == 0
        base = rng.uniform(0.5, 2.0)
        t = base * (0.4 if membro else 1.0) * rng.uniform(0.9, 1.1)
        ref_in = 6 * base * (0.7 if membro else 1.0) * rng.uniform(0.9, 1.1)
        ref_out = 6 * base * rng.uniform(0.9, 1.1)
        righe.append({"target": t, "refs": [(0, ref_in, True), (1, ref_out, False)],
                      "member": membro, "client": 0, "sid": i, "canary": None})
    f = run_exp._rmia_round_fields(righe, {}, {}, {})
    assert f["rmia_auc_roc"] > 0.8 and f["rmia_offline_auc_roc"] > 0.8
