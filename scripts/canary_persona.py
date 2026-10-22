"""
Controllo positivo a livello di PERSONA ("canary persona"), ACN-Data (2026-10-13).

Perche'. L'esperimento per persona (split per utente, 5 seed) e la MIA per
persona danno AUC al caso. Un nullo dice qualcosa solo se lo strumento vedrebbe
la persona quando il modello la memorizza. Per le sessioni questo lo prova il
canary; qui si fa lo stesso per le persone.

Disegno (stesso schema del canary per sessione: pool unico, assegnazione casuale,
due bracci a ruoli scambiati).
  1. Pool: utenti reali del sito scelto (default JPL) con almeno MIN sessioni in
     quel sito. Se ne estraggono 2K a caso col seed e si dividono in due gruppi
     G1 e G2 di K utenti. Il pool e i gruppi dipendono solo dal seed.
  2. Sessioni di ogni utente del pool, nel sito, mescolate col seed (o in ordine
     di tempo con --divisione cronologica): le prime T sono la parte "di
     training", le successive F sono "fresche". Le fresche non entrano mai nel
     training: servono a chiedere se il modello riconosce la PERSONA anche su
     sessioni che non ha visto (appartenenza del soggetto, Suri et al. 2022),
     non solo le sessioni viste.
  3. Braccio A: i membri sono G1. Il training contiene lo sfondo (tutte le
     sessioni che non sono di utenti del pool) piu' la parte di training di G1,
     ripetuta COPIE volte (COPIE = 1: nessuna amplificazione). G2 e' fuori del
     tutto. Braccio B: uguale con G1 e G2 scambiati. Stesso seed di
     inizializzazione e stessa normalizzazione (calcolata su sfondo + parti di
     training di tutto il pool) nei due bracci: cambia solo chi e' dentro.
  4. Attaccante: errore di ricostruzione del modello globale, per round.
     Punteggio della persona = media sulle sue sessioni, separatamente sulla
     parte di training e sulle fresche.
     Due superfici: il modello globale e l'update del client del sito (dove il
     canary per sessione e' letto); la media FedAvg con i client che non hanno
     visto la persona puo' cancellare dal globale cio' che il client ha imparato.
     - In un braccio: AUC grezza (membri contro non membri del pool).
     - Sui due bracci: la somma delle due AUC grezze (caso = 1.0) e l'AUC
       appaiata: per ogni utente s(u) = L_B(u) - L_A(u), positiva se l'utente
       e' ricostruito meglio quando e' dentro; etichetta = appartiene a G1. La
       differenza toglie la difficolta' propria della persona (ogni utente fa
       da riferimento a se stesso). IC bootstrap sugli utenti e p di
       permutazione dei gruppi.

Persona "costruita per essere memorizzata" (--spostamento L, fra 0 e 1): ogni
sessione del pool di un utente (training e fresche, in entrambi i bracci) viene
avvicinata, nello spazio normalizzato, a un centro proprio dell'utente scelto a
caso nel cubo [0,1]^d: x' = (1 - L) x + L c_u. Ogni persona diventa un gruppo
riconoscibile fuori dalla nuvola dei dati (combinazioni di feature che i dati veri
non hanno), come il canary per sessione e' un record ripetuto: e' il controllo
positivo piu' forte. I valori restano in [0,1], dove il decoder (sigmoide) puo'
arrivare: uno spostamento che esce da [0,1] darebbe un errore minimo uguale per
membri e non membri. Lo spostamento e' uguale nei due bracci, quindi l'appaiata
resta valida.

Regimi previsti (vedi docs/ESPERIMENTI.md): naturale (config/experiment.yaml,
10 x 50) con COPIE 1, 10, 30 per la curva di sensibilita'; indotto (config del
canary multisito, 3 x 1000, 7 feature) con COPIE 30 come controllo positivo.

Uso:
  python scripts/canary_persona.py --config config/experiment.yaml --seed 42 \
      --copie 30 --braccio A --out risultati/canary_persona
  (poi lo stesso con --braccio B: alla fine del secondo braccio si scrive anche
  il file appaiato *_AB_*.json; oppure --combina per rifarlo dai due JSON.)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_experiments as rx  # noqa: E402
import mia_persona as mp  # noqa: E402

SITI = {"0001": "jpl", "0002": "caltech", "0019": "office1"}
# Superfici osservate: il modello globale (rilasciato) e l'update del client del sito
# delle persone (superficie "client", quella su cui il canary per sessione e' letto).
SUPERFICI = ("globale", "client")


def sito_di(s: dict) -> str:
    raw = str(s.get("site_id", ""))
    return SITI.get(raw, raw)


def costruisci_pool(sessions: list[dict], sito: str, k: int, min_sess: int, t: int, f: int,
                    seed: int, divisione: str) -> dict[str, Any]:
    """Pool, gruppi e parti (training / fresche) per utente. Dipende solo dal seed."""
    per_utente: dict[str, list[int]] = {}
    for i, s in enumerate(sessions):
        u = s.get("user_id")
        if u in (None, "") or sito_di(s) != sito:
            continue
        per_utente.setdefault(str(u), []).append(i)
    idonei = sorted(u for u, ix in per_utente.items() if len(ix) >= max(min_sess, t + f))
    if len(idonei) < 2 * k:
        raise SystemExit(f"nel sito {sito} ci sono {len(idonei)} utenti con almeno "
                         f"{max(min_sess, t + f)} sessioni: ne servono {2 * k}")
    rng = random.Random(seed + 424242)
    scelti = rng.sample(idonei, 2 * k)
    rng.shuffle(scelti)
    g1, g2 = sorted(scelti[:k]), sorted(scelti[k:])
    parti = {}
    for u in scelti:
        ix = list(per_utente[u])
        if divisione == "cronologica":
            ix.sort(key=lambda i: str(sessions[i].get("start_time")))
        else:
            rng.shuffle(ix)
        parti[u] = {"training": ix[:t], "fresche": ix[t:t + f]}
    # tutte le sessioni degli utenti del pool (in qualunque sito) restano fuori dallo sfondo
    pool_set = set(scelti)
    sfondo = [i for i, s in enumerate(sessions) if str(s.get("user_id")) not in pool_set]
    return {"g1": g1, "g2": g2, "parti": parti, "sfondo": sfondo, "idonei": len(idonei)}


def applica_spostamento(sessions_n: list[dict], pool: dict, feats: list[str], lam: float,
                        seed: int) -> list[dict]:
    """Copia delle sessioni con ogni utente del pool avvicinato al suo centro c_u."""
    if lam <= 0:
        return sessions_n
    if lam > 1:
        raise ValueError("--spostamento va fra 0 e 1")
    rng = np.random.default_rng(seed + 8675309)
    out = list(sessions_n)
    for u in sorted(pool["parti"]):
        c = rng.uniform(0.05, 0.95, size=len(feats))
        for i in pool["parti"][u]["training"] + pool["parti"][u]["fresche"]:
            s = dict(out[i])
            for f, cf in zip(feats, c):
                x = min(max(float(s[f]), 0.0), 1.0)
                s[f] = (1 - lam) * x + lam * float(cf)
            out[i] = s
    return out


def training_del_braccio(sessions_n: list[dict], pool: dict, braccio: str, copie: int) -> list[dict]:
    membri = pool["g1"] if braccio == "A" else pool["g2"]
    train = [sessions_n[i] for i in pool["sfondo"]]
    for u in membri:
        for i in pool["parti"][u]["training"]:
            for _ in range(copie):
                train.append(dict(sessions_n[i]))
    return train


def auc(lab, sc) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(lab, sc))


def auc_ci_perm(lab, sc, n_boot=1000, n_perm=1000, seed=0) -> dict:
    lab, sc = np.asarray(lab), np.asarray(sc, dtype=float)
    a = auc(lab, sc)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(lab), len(lab))
        if len(set(lab[i])) == 2:
            bs.append(auc(lab[i], sc[i]))
    # p unilaterale (i membri ricostruiti meglio), Mann-Whitney: stesso test
    # di mia_persona, senza il minimo 1/(n+1) di una permutazione finita
    p = mp._p_auc(lab, sc)
    return {"auc": a, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p": p, "test": "Mann-Whitney unilaterale"}


class BracciIncoerenti(ValueError):
    """I due file non sono i bracci A e B dello stesso esperimento."""


def combina(fa: Path, fb: Path, out: Path, spostamento: float | None = None) -> dict:
    """AUC appaiata e somma A+B, per round, dalle perdite per utente dei due bracci.

    2026-10-16 (Segnalazioni R voce 83): prima di combinare si controlla che i
    file siano davvero il braccio A e il braccio B dello stesso esperimento
    (stessi parametri, stesso config, stessi round), altrimenti errore."""
    A, B = json.loads(fa.read_text()), json.loads(fb.read_text())
    if A.get("braccio") != "A" or B.get("braccio") != "B":
        raise BracciIncoerenti(f"bracci {A.get('braccio')!r} e {B.get('braccio')!r} invece di 'A' e 'B'")
    for chiave in ("seed", "copie", "config", "config_sha", "g1", "g2", "sito", "divisione",
                   "spostamento", "rounds", "epochs", "sessioni_training", "sessioni_fresche",
                   "feature", "utenti_idonei"):
        if A.get(chiave) != B.get(chiave):
            raise BracciIncoerenti(f"i due bracci differiscono in {chiave}")
    if spostamento is not None and float(A.get("spostamento", 0.0)) != float(spostamento):
        raise BracciIncoerenti(f"spostamento nei file {A.get('spostamento')} invece di {spostamento}")
    if set(A["perdite_per_round"]) != set(B["perdite_per_round"]):
        raise BracciIncoerenti("i due bracci hanno round diversi")
    utenti = A["g1"] + A["g2"]
    lab = [1] * len(A["g1"]) + [0] * len(A["g2"])
    ris: dict[str, Any] = {k: A.get(k) for k in ("seed", "copie", "spostamento", "config", "sito", "divisione",
                                             "rounds", "epochs", "sessioni_training", "sessioni_fresche")}
    ris["per_round"] = {}
    for r in A["perdite_per_round"]:
        riga = {}
        for sup in SUPERFICI:
            for parte in ("training", "fresche"):
                la = np.array([A["perdite_per_round"][r][sup][parte][u] for u in utenti])
                lb = np.array([B["perdite_per_round"][r][sup][parte][u] for u in utenti])
                aa, ab = A["auc_per_round"][r][sup][parte], B["auc_per_round"][r][sup][parte]
                riga[f"{sup}_{parte}_appaiata"] = auc_ci_perm(lab, lb - la)
                riga[f"{sup}_{parte}_somma_AB"] = aa + ab
                riga[f"{sup}_{parte}_auc_A"] = aa
                riga[f"{sup}_{parte}_auc_B"] = ab
        ris["per_round"][r] = riga
    # confronti multipli (2026-10-16, Segnalazioni R voce 84): metrica primaria
    # dichiarata prima dei numeri (ultimo round, superficie client, sessioni di
    # training); p corretti con Holm su tutte le appaiate della combinazione
    ultimo = max(ris["per_round"], key=int)
    ris["metrica_primaria"] = f"per_round[{ultimo}].client_training_appaiata"
    prim = f"{ultimo}.client_training_appaiata"
    pv = {f"{r}.{k}": v["p"] for r, riga in ris["per_round"].items()
          for k, v in riga.items() if k.endswith("_appaiata")}
    holm = mp._holm({k: v for k, v in pv.items() if k != prim})   # solo le secondarie
    holm[prim] = pv[prim]
    for r, riga in ris["per_round"].items():
        for k, v in riga.items():
            if k.endswith("_appaiata"):
                v["p_holm"] = holm[f"{r}.{k}"]
    out.write_text(json.dumps(ris, indent=2))
    return ris


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config/experiment.yaml"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--braccio", choices=("A", "B"), default="A")
    ap.add_argument("--copie", type=int, default=30, help="ripetizioni della parte di training dei membri")
    ap.add_argument("--sito", default="jpl")
    ap.add_argument("--utenti", type=int, default=20, help="utenti per gruppo (K)")
    ap.add_argument("--min-sessioni", type=int, default=30)
    ap.add_argument("--sessioni-training", type=int, default=20, help="T: sessioni per persona nel training")
    ap.add_argument("--sessioni-fresche", type=int, default=10, help="F: sessioni per persona mai viste")
    ap.add_argument("--divisione", choices=("casuale", "cronologica"), default="casuale")
    ap.add_argument("--spostamento", type=float, default=0.0,
                    help="persona costruita: quota L verso il centro proprio dell'utente, 0-1 (0 = utenti reali)")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "risultati/canary_persona"))
    ap.add_argument("--etichetta", default=None, help="nome del regime nei file (default: nome del config)")
    ap.add_argument("--combina", action="store_true", help="solo il file appaiato dai due bracci gia' fatti")
    ap.add_argument("--combina-se-pronto", action="store_true",
                    help="a fine run combina con l'altro braccio se esiste (default: no)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = a.etichetta or Path(a.config).stem
    sp = f"_d{a.spostamento:g}" if a.spostamento > 0 else ""
    cfg = rx.load_config(Path(a.config), {"rounds": a.rounds, "epochs": a.epochs})
    # round ed epoche nel nome: una run con parametri diversi non sovrascrive
    rr, ee = cfg["experiment"]["fl_rounds"], cfg["ml"]["epochs"]
    nome = lambda b: out / f"canary_persona_{tag}_c{a.copie}{sp}_r{rr}_e{ee}_{b}_s{a.seed}.json"  # noqa: E731
    if a.combina:
        r = combina(nome("A"), nome("B"), nome("AB"), spostamento=a.spostamento)
        print(json.dumps(r["per_round"][max(r["per_round"], key=int)], indent=2))
        return

    t0 = time.time()
    cfg["experiment"]["seed"] = a.seed
    cfg.setdefault("canary", {})["enabled"] = False  # il canary per sessione non c'entra
    mp._semina(a.seed)
    sessions = rx.enrich_sessions(rx.load_sessions(cfg))
    pool = costruisci_pool(sessions, a.sito, a.utenti, a.min_sessioni, a.sessioni_training,
                           a.sessioni_fresche, a.seed, a.divisione)
    feats = rx._mia_feature_names(cfg)
    # normalizzazione con limiti pubblici, uguale nei due bracci (2026-10-16)
    stats, clip, info_norm = rx.statistiche_normalizzazione(cfg, [], feats)
    sessions_n = rx.normalize_sessions(sessions, stats, feats, clip=clip)
    sessions_n = applica_spostamento(sessions_n, pool, feats, a.spostamento, a.seed)
    # DP per soggetto con profili: riserva di sessioni di sfondo senza userID,
    # fuori dal training di entrambi i bracci, per stimare i profili
    profili = None
    sdp = rx._subject_dp_config(cfg["ml"])
    if sdp["enabled"] and sdp["senza_chiave"] == "profili":
        from ml.profili_abitudini import addestra, assegna
        senza = [i for i in pool["sfondo"] if sessions[i].get("user_id") in (None, "")]
        random.Random(a.seed + 4242).shuffle(senza)
        ris = set(senza[:int(round(0.3 * len(senza)))])
        pool["sfondo"] = [i for i in pool["sfondo"] if i not in ris]
        pp = sdp.get("profili") or {}
        n_sito: dict[str, int] = {}
        for i in pool["sfondo"]:          # uguale nei due bracci
            k_s = str(sessions_n[i].get("site_id") or "")
            n_sito[k_s] = n_sito.get(k_s, 0) + 1
        mod = addestra([sessions_n[i] for i in sorted(ris)],
                       dimensione_media=float(pp.get("dimensione_media", 10.0)),
                       peso_gruppo=float(pp.get("peso_gruppo", 1.0)), seed=a.seed,
                       fonte="riserva dello sfondo", n_training_per_sito=n_sito)
        assegna(mod, sessions_n)
        profili = mod.descrizione()
    train = training_del_braccio(sessions_n, pool, a.braccio, a.copie)

    mp._semina(a.seed)
    cs_map = rx.group_indices_by_site(train)
    cs = {cid: [train[i] for i in idx] for cid, idx in cs_map.items()}
    t1 = time.time()
    res = rx.run_fl_rounds(cfg, train, no_dp=True, dp_mode="dp-fedavg", cluster_sessions=cs)
    t_train = time.time() - t1

    membri = set(pool["g1"] if a.braccio == "A" else pool["g2"])
    utenti = pool["g1"] + pool["g2"]
    lab = [1 if u in membri else 0 for u in utenti]
    perdite, aucs = {}, {}
    idx_parte = {parte: [i for u in utenti for i in pool["parti"][u][parte]] for parte in ("training", "fresche")}
    for r in sorted(k for k in res if isinstance(k, int) and k > 0):
        upd = [x for x in (res[r].get("updates") or [])
               if str(getattr(x, "cluster_id", "")) == a.sito or str(getattr(x, "node_id", "")).startswith(a.sito)]
        if len(upd) != 1:
            raise RuntimeError(f"round {r}: update del client {a.sito} non trovato ({len(upd)})")
        pesi = {"globale": res[r]["global_weights"], "client": upd[0].weights}
        perdite[r], aucs[r] = {}, {}
        for sup in SUPERFICI:
            perdite[r][sup], aucs[r][sup] = {}, {}
            for parte in ("training", "fresche"):
                loss = mp._perdite(cfg, pesi[sup], [sessions_n[i] for i in idx_parte[parte]], feats)
                pos, per_u = 0, {}
                for u in utenti:
                    n = len(pool["parti"][u][parte])
                    per_u[u] = float(loss[pos:pos + n].mean())
                    pos += n
                perdite[r][sup][parte] = per_u
                aucs[r][sup][parte] = auc(lab, [-per_u[u] for u in utenti])
    sfondo_per_sito = Counter(sito_di(sessions[i]) for i in pool["sfondo"])
    esito = {
        "seed": a.seed, "braccio": a.braccio, "copie": a.copie, "config": Path(a.config).name,
        "config_sha": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16],
        "commit": mp._commit(), "normalizzazione": info_norm, "profili": profili,
        "sito": a.sito, "divisione": a.divisione, "spostamento": a.spostamento, "rounds": cfg["experiment"]["fl_rounds"],
        "epochs": cfg["ml"]["epochs"], "feature": feats,
        "sessioni_training": a.sessioni_training, "sessioni_fresche": a.sessioni_fresche,
        "utenti_idonei": pool["idonei"], "g1": pool["g1"], "g2": pool["g2"],
        "sessioni_nel_training": len(train), "sfondo_per_sito": dict(sfondo_per_sito),
        "client": {c: len(v) for c, v in cs.items()},
        "loss_globale_per_round": {r: res[r].get("mean_loss") for r in perdite},
        "auc_per_round": aucs, "perdite_per_round": perdite,
        "secondi_training": t_train, "secondi_totali": time.time() - t0,
    }
    f = nome(a.braccio)
    f.write_text(json.dumps(esito, indent=2))
    ultimo = max(aucs)
    print(f"braccio {a.braccio}, copie {a.copie}, round {ultimo}: AUC grezza "
          + ", ".join(f"{sup} {parte} {aucs[ultimo][sup][parte]:.3f}" for sup in SUPERFICI
                      for parte in ("training", "fresche")) + f" -> {f}")
    altro = nome("B" if a.braccio == "A" else "A")
    if a.combina_se_pronto and altro.exists():
        r = combina(nome("A"), nome("B"), nome("AB"), spostamento=a.spostamento)
        print("appaiato (ultimo round):", json.dumps(r["per_round"][max(r["per_round"], key=int)], indent=2))


if __name__ == "__main__":
    main()
