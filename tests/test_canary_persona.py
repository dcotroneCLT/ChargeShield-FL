"""Test del controllo positivo per persona (scripts/canary_persona.py), 2026-10-13."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

cp = pytest.importorskip("canary_persona")


def _sessioni(n_utenti=60, per_utente=40, sito="0001"):
    ss = []
    for u in range(n_utenti):
        for j in range(per_utente):
            ss.append({"user_id": f"{u:06d}", "site_id": sito, "start_time": f"2019-01-{j % 28 + 1:02d}",
                       "x": float(u + j)})
    for j in range(200):  # sfondo senza utente e altro sito
        ss.append({"user_id": None, "site_id": "0002", "start_time": "2019-02-01", "x": float(j)})
    return ss


def test_pool_dipende_solo_dal_seed_e_gruppi_disgiunti():
    ss = _sessioni()
    p1 = cp.costruisci_pool(ss, "jpl", 10, 30, 20, 10, 42, "casuale")
    p2 = cp.costruisci_pool(ss, "jpl", 10, 30, 20, 10, 42, "casuale")
    assert p1["g1"] == p2["g1"] and p1["g2"] == p2["g2"] and p1["parti"] == p2["parti"]
    assert not set(p1["g1"]) & set(p1["g2"])
    assert len(p1["g1"]) == len(p1["g2"]) == 10
    for u, parti in p1["parti"].items():
        assert len(parti["training"]) == 20 and len(parti["fresche"]) == 10
        assert not set(parti["training"]) & set(parti["fresche"])
    p3 = cp.costruisci_pool(ss, "jpl", 10, 30, 20, 10, 43, "casuale")
    assert p3["g1"] != p1["g1"]


def test_sfondo_esclude_tutte_le_sessioni_del_pool():
    ss = _sessioni()
    p = cp.costruisci_pool(ss, "jpl", 10, 30, 20, 10, 42, "casuale")
    pool = set(p["g1"]) | set(p["g2"])
    assert all(str(ss[i]["user_id"]) not in pool for i in p["sfondo"])
    assert len(p["sfondo"]) == len(ss) - 20 * 40


def test_training_del_braccio_membri_ripetuti_non_membri_assenti():
    ss = _sessioni()
    p = cp.costruisci_pool(ss, "jpl", 10, 30, 20, 10, 42, "casuale")
    for braccio, membri, fuori in (("A", p["g1"], p["g2"]), ("B", p["g2"], p["g1"])):
        tr = cp.training_del_braccio(ss, p, braccio, 5)
        utenti = [s["user_id"] for s in tr]
        for u in membri:
            assert utenti.count(u) == 20 * 5  # solo la parte di training, ripetuta
        for u in fuori:
            assert u not in utenti
        assert len(tr) == len(p["sfondo"]) + 10 * 20 * 5
        # le fresche non sono mai nel training
        fresche = {id(ss[i]) for u in membri for i in p["parti"][u]["fresche"]}
        assert not fresche & {id(s) for s in tr}


def test_cronologica_prende_le_prime_in_ordine_di_tempo():
    ss = _sessioni()
    p = cp.costruisci_pool(ss, "jpl", 5, 30, 20, 10, 1, "cronologica")
    for u, parti in p["parti"].items():
        t_tr = max(ss[i]["start_time"] for i in parti["training"])
        t_fr = min(ss[i]["start_time"] for i in parti["fresche"])
        assert t_tr <= t_fr


def test_pochi_utenti_errore_chiaro():
    with pytest.raises(SystemExit):
        cp.costruisci_pool(_sessioni(n_utenti=10), "jpl", 10, 30, 20, 10, 42, "casuale")


def _braccio(path, braccio, g1, g2, perdite):
    lab = {u: (1 if (u in g1) == (braccio == "A") else 0) for u in g1 + g2}
    from sklearn.metrics import roc_auc_score
    auc = {sup: {p: roc_auc_score([lab[u] for u in g1 + g2], [-perdite[p][u] for u in g1 + g2])
                 for p in ("training", "fresche")} for sup in cp.SUPERFICI}
    path.write_text(json.dumps({
        "seed": 1, "braccio": braccio, "copie": 30, "config": "x.yaml", "g1": g1, "g2": g2, "sito": "jpl",
        "divisione": "casuale", "rounds": 1, "epochs": 1, "sessioni_training": 20, "sessioni_fresche": 10,
        "perdite_per_round": {"1": {sup: perdite for sup in cp.SUPERFICI}}, "auc_per_round": {"1": auc}}))


def test_combina_vede_la_memorizzazione_e_toglie_la_difficolta():
    rng = np.random.default_rng(0)
    g1 = [f"a{i}" for i in range(20)]
    g2 = [f"b{i}" for i in range(20)]
    diff = {u: float(rng.lognormal(0, 1)) for u in g1 + g2}  # difficolta' propria, grande
    eff = 0.9  # chi e' dentro e' ricostruito il 10% meglio
    pa = {p: {u: diff[u] * (eff if u in g1 else 1) for u in g1 + g2} for p in ("training", "fresche")}
    pb = {p: {u: diff[u] * (eff if u in g2 else 1) for u in g1 + g2} for p in ("training", "fresche")}
    d = Path(__file__).parent / "_tmp_cp"
    d.mkdir(exist_ok=True)
    try:
        _braccio(d / "A.json", "A", g1, g2, pa)
        _braccio(d / "B.json", "B", g1, g2, pb)
        r = cp.combina(d / "A.json", d / "B.json", d / "AB.json")["per_round"]["1"]
        assert r["client_training_appaiata"]["auc"] == 1.0          # l'appaiata la vede tutta
        assert r["client_training_appaiata"]["p"] < 0.01
        assert r["client_training_somma_AB"] < 1.6                   # la grezza e' coperta dalla difficolta'
        # nullo: nessun effetto -> appaiata a 0.5 (differenze tutte nulle)
        pn = {p: {u: diff[u] for u in g1 + g2} for p in ("training", "fresche")}
        _braccio(d / "A.json", "A", g1, g2, pn)
        _braccio(d / "B.json", "B", g1, g2, pn)
        r0 = cp.combina(d / "A.json", d / "B.json", d / "AB.json")["per_round"]["1"]
        assert r0["globale_training_appaiata"]["auc"] == pytest.approx(0.5)
        assert r0["globale_training_somma_AB"] == pytest.approx(1.0)
    finally:
        for f in d.glob("*.json"):
            f.unlink()
        d.rmdir()


def test_spostamento_uguale_nei_bracci_e_solo_sul_pool():
    ss = _sessioni()
    for s in ss:
        s["a"], s["b"] = 0.5, 0.2
    p = cp.costruisci_pool(ss, "jpl", 10, 30, 20, 10, 42, "casuale")
    out = cp.applica_spostamento(ss, p, ["a", "b"], 0.6, 42)
    out2 = cp.applica_spostamento(ss, p, ["a", "b"], 0.6, 42)
    assert out == out2  # dipende solo dal seed: uguale nei due bracci
    for i in p["sfondo"]:
        assert out[i] is ss[i]
    centri = set()
    for u, parti in p["parti"].items():
        vs = {(round(out[i]["a"], 9), round(out[i]["b"], 9)) for i in parti["training"] + parti["fresche"]}
        assert len(vs) == 1  # sessioni uguali in partenza -> stesso punto per la persona
        (va, vb), = vs
        assert 0 <= va <= 1 and 0 <= vb <= 1  # resta dove il decoder puo' arrivare
        centri.add((va, vb))
    assert len(centri) == len(p["parti"])  # ogni persona ha il suo centro
    assert ss[p["parti"][p["g1"][0]]["training"][0]]["a"] == 0.5  # originale non toccato
    assert cp.applica_spostamento(ss, p, ["a"], 0.0, 42) is ss
    with pytest.raises(ValueError):
        cp.applica_spostamento(ss, p, ["a"], 1.5, 42)


def test_combina_rifiuta_bracci_incoerenti(tmp_path):
    """2026-10-16 (Segnalazioni R voce 83): A copiato come B, round o
    spostamento diversi devono dare errore, non un AUC 0,5 falso."""
    g1 = [f"a{i}" for i in range(5)]
    g2 = [f"b{i}" for i in range(5)]
    pn = {p: {u: 1.0 for u in g1 + g2} for p in ("training", "fresche")}
    _braccio(tmp_path / "A.json", "A", g1, g2, pn)
    _braccio(tmp_path / "B.json", "A", g1, g2, pn)          # A copiato come B
    with pytest.raises(cp.BracciIncoerenti):
        cp.combina(tmp_path / "A.json", tmp_path / "B.json", tmp_path / "AB.json")
    _braccio(tmp_path / "B.json", "B", g1, g2, pn)
    b = json.loads((tmp_path / "B.json").read_text())
    b["epochs"] = 999
    (tmp_path / "B.json").write_text(json.dumps(b))
    with pytest.raises(cp.BracciIncoerenti):
        cp.combina(tmp_path / "A.json", tmp_path / "B.json", tmp_path / "AB.json")
    _braccio(tmp_path / "B.json", "B", g1, g2, pn)
    with pytest.raises(cp.BracciIncoerenti):
        cp.combina(tmp_path / "A.json", tmp_path / "B.json", tmp_path / "AB.json", spostamento=0.5)
