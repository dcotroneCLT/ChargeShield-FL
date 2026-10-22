#!/usr/bin/env python3
"""Misura dei profili di abitudine per la DP per soggetto (2026-10-16).

Per ogni dimensione media dei profili (sessioni di TRAINING per profilo): quanti profili tocca una persona vera
(misurato sulle persone con userID del training, ignorando lo userID), quanti
soggetti ha ogni client e l'epsilon per soggetto che ne risulta. Una persona i
cui record cadono in k profili e' protetta per privacy di gruppo su k soggetti.

Uso: python scripts/misura_profili.py --config config/experiment_acn_subjectdp.yaml \\
        --seed 42 --dimensioni 5,10,20,50 --out risultati/misura_profili_s42.json
Replica split, riserva e normalizzazione di run_experiments.main().
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_experiments as rx  # noqa: E402
from ml import profili_abitudini as pa  # noqa: E402
from ml.subject_dp import epsilon_single_client, subject_dp_config  # noqa: E402


def prepara(cfg: dict, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    sessions = rx.enrich_sessions(rx.load_sessions(cfg))
    random.shuffle(sessions)
    split = max(1, int(len(sessions) * 0.8))
    train, hold = sessions[:split], sessions[split:]
    frac = float(cfg["experiment"].get("riserva_profili", 0.3))
    idx = list(range(len(hold)))
    random.Random(seed + 4242).shuffle(idx)
    ris_i = set(idx[:int(round(frac * len(hold)))])
    riserva = [s for i, s in enumerate(hold) if i in ris_i]
    feats = rx._mia_feature_names(cfg)
    stats, clip, _ = rx.statistiche_normalizzazione(cfg, train, feats)
    return (rx.normalize_sessions(train, stats, feats, clip=clip),
            rx.normalize_sessions(riserva, stats, feats, clip=clip))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_acn_subjectdp.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dimensioni", default="5,10,20,50")
    ap.add_argument("--sigma", default="2,5")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    sdp = subject_dp_config(cfg["ml"])
    E, R = int(cfg["ml"]["epochs"]), int(cfg["experiment"]["fl_rounds"])
    delta = float(cfg["experiment"]["delta"])
    train, riserva = prepara(cfg, a.seed)
    siti = {s: rx._SITE_ID_TO_NAME.get(s, s) for s in {x.get("site_id") for x in train}}
    persone = defaultdict(list)
    for s in train:
        if s.get("user_id") not in (None, ""):
            persone[str(s["user_id"])].append(s)
    risultati = {"seed": a.seed, "config": a.config, "n_train": len(train), "n_riserva": len(riserva),
                 "n_persone_con_userid": len(persone), "per_dimensione": {}}
    for dim in [float(x) for x in a.dimensioni.split(",")]:
        n_sito: dict[str, int] = {}
        for s in train:
            n_sito[str(s.get("site_id") or "")] = n_sito.get(str(s.get("site_id") or ""), 0) + 1
        mod = pa.addestra(riserva, dimensione_media=dim, seed=a.seed, n_training_per_sito=n_sito)
        k = np.array([len({mod.profilo(s) for s in ss}) for ss in persone.values()])
        n_sess = np.array([len(ss) for ss in persone.values()])
        # soggetti per client: userID + profili delle sessioni senza userID
        sogg = defaultdict(set)
        for s in train:
            c = siti.get(s.get("site_id"), s.get("site_id"))
            sogg[c].add(f"u:{s['user_id']}" if s.get("user_id") not in (None, "") else f"p:{mod.profilo(s)}")
        eps = {}
        for sg in [float(x) for x in a.sigma.split(",")]:
            eps[str(sg)] = {c: round(epsilon_single_client(len(v), int(sdp["expected_subjects_per_step"]),
                                                           E, R, sg, delta)[0], 3) for c, v in sogg.items()}
        risultati["per_dimensione"][str(dim)] = {
            "profili_per_sito": mod.descrizione()["profili_per_sito"],
            "profili_toccati_per_persona": {
                "mediana": float(np.median(k)), "p90": float(np.quantile(k, 0.9)), "max": int(k.max()),
                "quota_al_piu_5": round(float((k <= 5).mean()), 4),
                "mediana_sessioni_per_persona": float(np.median(n_sess)),
                "rapporto_mediano_profili_su_sessioni": round(float(np.median(k / n_sess)), 4)},
            "soggetti_per_client": {c: len(v) for c, v in sogg.items()},
            "epsilon_per_client": eps,
        }
        print(dim, json.dumps(risultati["per_dimensione"][str(dim)], ensure_ascii=False))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(risultati, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
