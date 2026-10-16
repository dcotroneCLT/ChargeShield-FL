#!/usr/bin/env python3
"""Controllo di confondente: i canary si separano dai non-membri ANCHE
prima di qualunque addestramento? Se si', il segnale e' difficolta'
intrinseca di ricostruzione, non appartenenza."""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_experiments import (  # noqa: E402
    load_config,
    load_sessions,
    enrich_sessions,
    entity_aware_split,
    inject_canaries,
    compute_feature_stats,
    statistiche_normalizzazione,
    normalize_sessions,
    _mia_feature_names,
)
from core.autoencoder import Autoencoder  # noqa: E402


def auc_roc(pos, neg):
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), dtype=float)
    r[order] = np.arange(1, len(s) + 1)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, r)
    r = (sums / counts)[inv]
    n_p, n_n = int(y.sum()), int((1 - y).sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    return (r[y == 1].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n)


def losses(model, sessions, features):
    X = np.array([[float(s[f]) for f in features] for s in sessions],
                 dtype=np.float32)
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(X)
        out = model(t)
        if isinstance(out, tuple):
            out = out[0]
        return ((out - t) ** 2).mean(dim=1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-init", type=int, default=20)
    # --seed (2026-09-16, Sprint 10zz+113): PRIMA questo script leggeva il
    # seed SOLO dalla config, quindi un loop "for s in 42 123 456 ...; do
    # check_canary_init_confound.py --config X; run_experiments.py --seed $s"
    # produceva la STESSA baseline (quella di config.experiment.seed) per
    # tutti i seed, mentre i run veri usavano seed diversi — le baseline non
    # descrivevano i canary dei run con cui venivano confrontate. Ora il
    # flag esiste e ha la precedenza sulla config, come --seed in
    # run_experiments.py.
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(Path(args.config), {})
    seed = cfg.get("experiment", {}).get("seed", cfg.get("seed", 42))
    if args.seed is not None:
        seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sessions = enrich_sessions(load_sessions(cfg))

    sc = cfg.get("split", {})
    if sc.get("strategy", "random") == "entity_aware":
        train, holdout = entity_aware_split(
            sessions,
            entity_key=sc.get("entity_key", "node_id"),
            holdout_fraction=sc.get("holdout_fraction", 0.2),
            seed=seed,
        )
    else:
        # random.shuffle PRIMA del taglio (2026-09-16, Sprint 10zz+113):
        # deve replicare ESATTAMENTE lo split del run vero
        # (run_experiments.py:6877 fa random.shuffle(sessions) e poi taglia
        # all'80%). Senza lo shuffle questo script faceva uno split
        # POSIZIONALE, quindi site_train_sessions era un insieme diverso e
        # rng.sample() dentro inject_canaries() estraeva TEMPLATE DIVERSI da
        # quelli del run: la baseline misurava la difficolta' intrinseca di
        # canary che non erano quelli sotto test.
        random.shuffle(sessions)
        cut = max(1, int(len(sessions) * 0.8))
        train, holdout = sessions[:cut], sessions[cut:]

    train, holdout = inject_canaries(train, holdout, cfg, seed)

    features = _mia_feature_names(cfg)
    stats, clip, _ = statistiche_normalizzazione(cfg, train, features)   # 2026-10-16
    train = normalize_sessions(train, stats, features, clip=clip)
    holdout = normalize_sessions(holdout, stats, features, clip=clip)

    members = [s for s in train if s.get("_canary_role") == "member"]
    nonmembers = [s for s in holdout if s.get("_canary_role") == "nonmember"]

    if not members or not nonmembers:
        print("ERRORE: nessun canary taggato trovato. Verificare che")
        print("cfg['canary']['enabled'] sia true nella config indicata.")
        sys.exit(1)

    def uniq(xs):
        return len({tuple(s[f] for f in features) for s in xs})

    print(f"seed              : {seed}")
    print(f"feature           : {features}")
    print(f"canary membri     : {len(members)} ({uniq(members)} distinti)")
    print(f"canary non-membri : {len(nonmembers)} ({uniq(nonmembers)} distinti)")
    print()

    aucs = []
    for i in range(args.n_init):
        torch.manual_seed(seed + 10_000 + i)
        model = Autoencoder(input_dim=len(features))
        lm = losses(model, members, features)
        ln = losses(model, nonmembers, features)
        aucs.append(auc_roc(-lm, -ln))
        if i == 0:
            print(f"loss media membri     : {lm.mean():.6f}")
            print(f"loss media non-membri : {ln.mean():.6f}")
            print(f"rapporto              : {ln.mean() / lm.mean():.2f}x")
            print()

    a = np.array(aucs)
    print(f"AUC a init casuale ({args.n_init} inizializzazioni): "
          f"media {a.mean():.4f}  std {a.std():.4f}  "
          f"min {a.min():.4f}  max {a.max():.4f}")
    print()
    if a.mean() > 0.60 or a.mean() < 0.40:
        print(">>> CONFONDENTE CONFERMATO: un modello mai addestrato separa")
        print(">>> gia' i due gruppi. Il controllo positivo misura difficolta'")
        print(">>> di ricostruzione, non appartenenza.")
    else:
        print(">>> Nessun confondente a inizializzazione casuale: il segnale")
        print(">>> post-training e' attribuibile all'addestramento.")


if __name__ == "__main__":
    main()
