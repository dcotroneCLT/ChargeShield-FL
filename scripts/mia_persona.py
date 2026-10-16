"""
MIA a livello di PERSONA su ACN-Data, con e senza re-identificazione
(2026-10-13).

Domanda: un attaccante puo' dire se una persona ha contribuito al training,
anche quando i dati non portano un identificativo utente?

Disegno.
  1. Utenti con almeno MIN sessioni (con userID), divisi a caso con il seed
     in tre gruppi: membri, non membri, persone di riferimento. Il training
     contiene TUTTE le sessioni dei membri, piu' lo sfondo (sessioni senza
     userID e utenti con poche sessioni, tolta una riserva per i profili di
     abitudine della DP per soggetto). Non membri e persone di riferimento
     restano fuori dal training.
  2. Modello bersaglio: federazione ACN a 3 siti, stesso protocollo del regime
     naturale, con il meccanismo scelto (--meccanismo).
  3. Modello di riferimento (lato attaccante): stesso meccanismo del
     bersaglio, addestrato sulle persone di riferimento piu' lo sfondo, cioe'
     su persone simili a quelle valutate ma disgiunte (2026-10-16,
     Segnalazioni R voce 82: prima era senza DP e sul solo sfondo, poco simile
     alle persone valutate). Calibra la difficolta' di ogni sessione (attacco
     "offline" con un riferimento, come RMIA). Cache indicizzata dal
     contenuto di config, meccanismo, seed, MIN, normalizzazione e dati.
  4. Per ogni persona valutata l'attaccante conosce SEME sessioni (le prime in
     ordine di tempo). Quattro modi di decidere se e' membro, dal piu' debole al
     piu' forte:
       - sessione singola: AUC per sessione (riferimento, livello record);
       - seme: media sulle sole sessioni note;
       - re-identificazione: alle sessioni note aggiunge le RITROVATE sessioni
         piu' simili trovate in tutto il pool SENZA identificativi (profilo a
         nucleo: colonnina, giorno, ora, energia, durata, kWh richiesti, minuti
         disponibili), poi media;
       - oracolo: media su tutte le sessioni vere della persona (limite
         superiore: re-identificazione perfetta).
     Ogni punteggio in due versioni: grezzo (- loss del bersaglio) e calibrato
     (loss del riferimento - loss del bersaglio).
  5. AUC sulle persone con intervallo bootstrap al 95% (1000 ricampionamenti
     delle persone, anche per l'AUC per sessione), p di permutazione
     unilaterale e p corretti con Holm sulle metriche della run; metrica
     primaria dichiarata: oracolo calibrato. Precisione della
     re-identificazione (quota delle sessioni ritrovate che sono davvero
     della persona). Normalizzazione con limiti pubblici; epsilon della cella
     nel JSON (campo "contabilita").

Uso:
    python scripts/mia_persona.py --seed 42 --meccanismo nodp [--rounds 10 --epochs 50]
Meccanismi: nodp, client_eps64 (DP per client, rumore lato client, eps 64,
C = 1), record_s2 / record_s5 (DP per record), subject_s2 / subject_s5 (DP per
soggetto = utente).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_experiments as rx  # noqa: E402
from core.autoencoder import Autoencoder  # noqa: E402

MECCANISMI: dict[str, dict[str, Any]] = {
    "nodp": {"no_dp": True},
    "client_eps64": {"no_dp": False, "dp_mode": "local", "epsilon": 64.0},
    "record_s2": {"no_dp": True, "ml": {"norm": "group", "record_dp": {
        "enabled": True, "max_grad_norm": 1.0, "noise_multiplier": 2.0}}},
    "record_s5": {"no_dp": True, "ml": {"norm": "group", "record_dp": {
        "enabled": True, "max_grad_norm": 1.0, "noise_multiplier": 5.0}}},
    "subject_s2": {"no_dp": True, "ml": {"norm": "group", "subject_dp": {
        "enabled": True, "subject_key": "user_id", "max_grad_norm": 1.0,
        "noise_multiplier": 2.0, "expected_subjects_per_step": 32,
        "max_records_per_subject": 64}}},
    "subject_s5": {"no_dp": True, "ml": {"norm": "group", "subject_dp": {
        "enabled": True, "subject_key": "user_id", "max_grad_norm": 1.0,
        "noise_multiplier": 5.0, "expected_subjects_per_step": 32,
        "max_records_per_subject": 64}}},
}


def _semina(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dividi_persone(sessions, min_sess: int, seed: int):
    """Persone con almeno min_sess sessioni (con userID) divise a caso in tre
    gruppi: membri, non membri e persone di riferimento (2026-10-16: il
    riferimento si allena su persone simili a quelle valutate ma disgiunte da
    esse, Segnalazioni R voce 82)."""
    per_utente: dict[str, list[dict]] = {}
    for s in sessions:
        u = s.get("user_id")
        if u not in (None, ""):
            per_utente.setdefault(str(u), []).append(s)
    valutabili = sorted(u for u, ss in per_utente.items() if len(ss) >= min_sess)
    rng = random.Random(seed + 7177)
    rng.shuffle(valutabili)
    terzo = len(valutabili) // 3
    membri = set(valutabili[:terzo])
    non_membri = set(valutabili[terzo:2 * terzo])
    riferimento = set(valutabili[2 * terzo:])
    return membri, non_membri, riferimento, per_utente


def _allena(cfg, train, no_dp, dp_mode):
    cm = rx.group_indices_by_site(train)
    cs = {cid: [train[i] for i in idx] for cid, idx in cm.items()}
    res = rx.run_fl_rounds(cfg, train, no_dp=no_dp, dp_mode=dp_mode, cluster_sessions=cs)
    ultimo = max(k for k in res if isinstance(k, int) and k > 0)
    return res[ultimo]["global_weights"], cs


def _perdite(cfg, pesi, sessioni, feats) -> np.ndarray:
    """Errore di ricostruzione per sessione del modello con i pesi dati."""
    m = Autoencoder(input_dim=len(feats), **rx._autoencoder_arch_kwargs(cfg))
    sd = m.state_dict()
    nuovo = {k: (w if isinstance(w, torch.Tensor) else torch.tensor(w)).to(sd[k].dtype)
             for k, w in zip(sd.keys(), pesi)}
    m.load_state_dict(nuovo, strict=True)
    m.eval()
    x = torch.tensor([[float(s[f]) for f in feats] for s in sessioni], dtype=torch.float32)
    with torch.no_grad():
        r = m(x)
    return ((r - x) ** 2).mean(dim=1).numpy()


def _feature_profilo(sessioni) -> tuple[np.ndarray, np.ndarray]:
    """Feature grezze per la re-identificazione: continue standardizzate e
    categoriche (colonnina, giorno)."""
    cont, cat = [], []
    staz: dict[str, int] = {}
    for s in sessioni:
        try:
            st = datetime.fromisoformat(str(s["start_time"]))
            wd = st.weekday()
        except Exception:  # noqa: BLE001
            wd = 0
        h = float(s.get("hour_of_day") or 0.0)
        cont.append([math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24),
                     float(s.get("total_energy_kwh") or 0.0), float(s.get("duration_hours") or 0.0),
                     float(s.get("kwh_requested") or 0.0), float(s.get("minutes_available") or 0.0)])
        cat.append([staz.setdefault(str(s.get("node_id")), len(staz)), wd])
    c = np.asarray(cont, dtype=np.float32)
    c = (c - c.mean(0)) / (c.std(0) + 1e-9)
    return c, np.asarray(cat, dtype=np.int64)


def _ritrova(c, k, idx_seme, n, b=0.15, eps=0.3, n_staz=None):
    """Indici delle n sessioni del pool piu' simili alle sessioni seme
    (somma dei nuclei, i semi esclusi)."""
    lk = np.zeros((len(idx_seme), len(c)), np.float32)
    lk -= 0.5 * ((c[idx_seme][:, None, :] - c[None, :, :]) ** 2).sum(-1) / (b * b)
    for j, nv in enumerate(n_staz):
        uguali = k[idx_seme, j][:, None] == k[None, :, j]
        lk += np.where(uguali, math.log(1 - eps), math.log(eps / max(nv - 1, 1)))
    m = lk.max(0)
    score = m + np.log(np.exp(lk - m).sum(0))
    score[idx_seme] = -np.inf
    return np.argpartition(-score, n)[:n]


def _auc_ci(lab, sc, n_boot=1000, seed=0, gruppi=None):
    """AUC con intervallo bootstrap al 95%. Con `gruppi` (persona di ogni
    elemento) si ricampionano le persone, non gli elementi: le sessioni della
    stessa persona sono correlate (2026-10-16, Segnalazioni R voce 82)."""
    from sklearn.metrics import roc_auc_score
    lab, sc = np.asarray(lab), np.asarray(sc, dtype=float)
    auc = float(roc_auc_score(lab, sc))
    rng = np.random.default_rng(seed)
    if gruppi is None:
        unita = [np.array([i]) for i in range(len(lab))]
    else:
        idx: dict[Any, list[int]] = {}
        for i, g in enumerate(gruppi):
            idx.setdefault(g, []).append(i)
        unita = [np.array(v) for v in idx.values()]
    bs = []
    for _ in range(n_boot):
        scelte = rng.integers(0, len(unita), len(unita))
        i = np.concatenate([unita[j] for j in scelte])
        if len(set(lab[i])) == 2:
            bs.append(roc_auc_score(lab[i], sc[i]))
    return auc, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _p_auc(lab, sc) -> float:
    """p unilaterale per AUC > 0,5: test di Mann-Whitney (l'AUC e' U/(n1 n0)),
    senza il minimo 1/(n+1) di una permutazione finita (revisione 2026-10-16)."""
    from scipy.stats import mannwhitneyu
    lab, sc = np.asarray(lab), np.asarray(sc, dtype=float)
    return float(mannwhitneyu(sc[lab == 1], sc[lab == 0], alternative="greater").pvalue)


def _holm(p: dict[str, float]) -> dict[str, float]:
    """p corretti con Holm (famiglia: le metriche di questa run)."""
    ordine = sorted(p, key=p.get)
    m = len(ordine)
    out, corrente = {}, 0.0
    for i, k in enumerate(ordine):
        corrente = max(corrente, min(1.0, (m - i) * p[k]))
        out[k] = corrente
    return out


def _commit() -> str | None:
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config/experiment.yaml"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--meccanismo", default="nodp", choices=sorted(MECCANISMI))
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--min-sessioni", type=int, default=10)
    ap.add_argument("--seme", default="3,10", help="sessioni note per persona (lista)")
    ap.add_argument("--ritrovate", default="10,20,50", help="sessioni ritrovate (lista)")
    ap.add_argument("--out", default=str(ROOT / "risultati/mia_persona"))
    ap.add_argument("--solo-riferimento", action="store_true",
                    help="calcola solo il modello di riferimento del seed e termina")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mec = MECCANISMI[a.meccanismo]

    cfg = rx.load_config(Path(a.config), {"rounds": a.rounds, "epochs": a.epochs})
    cfg["experiment"]["seed"] = a.seed
    cfg_t = copy.deepcopy(cfg)
    for k, v in (mec.get("ml") or {}).items():
        cfg_t["ml"][k] = copy.deepcopy(v)
    if "epsilon" in mec:
        cfg_t["experiment"]["epsilon"] = mec["epsilon"]
    if min(int(x) for x in str(a.seme).split(",")) >= a.min_sessioni:
        print("ATTENZIONE: un seme >= --min-sessioni coincide con l'oracolo per alcune persone")
    _semina(a.seed)
    sessions = rx.enrich_sessions(rx.load_sessions(cfg))
    membri, non_membri, persone_rif, per_utente = _dividi_persone(sessions, a.min_sessioni, a.seed)
    valutati = membri | non_membri
    escluse_sfondo = valutati | persone_rif
    sfondo_tutto = [s for s in sessions if str(s.get("user_id")) not in escluse_sfondo]
    # riserva per i profili di abitudine: sessioni di sfondo senza userID, mai
    # nel training (servono solo a stimare i profili della DP per soggetto)
    senza_id = [i for i, s in enumerate(sfondo_tutto) if s.get("user_id") in (None, "")]
    random.Random(a.seed + 4242).shuffle(senza_id)
    ris_i = set(senza_id[:int(round(0.3 * len(senza_id)))])
    riserva = [s for i, s in enumerate(sfondo_tutto) if i in ris_i]
    sfondo = [s for i, s in enumerate(sfondo_tutto) if i not in ris_i]
    train = sfondo + [s for u in sorted(membri) for s in per_utente[u]]
    train_rif = sfondo + [s for u in sorted(persone_rif) for s in per_utente[u]]
    feats = rx._mia_feature_names(cfg)
    # limiti pubblici, non statistiche dei membri (Segnalazioni R voce 76)
    stats, clip, info_norm = rx.statistiche_normalizzazione(cfg, train, feats)
    norm_list = rx.normalize_sessions(sessions, stats, feats, clip=clip)
    pos = {id(s): i for i, s in enumerate(sessions)}
    train_n = [norm_list[pos[id(s)]] for s in train]
    rif_n = [norm_list[pos[id(s)]] for s in train_rif]
    riserva_n = [norm_list[pos[id(s)]] for s in riserva]
    profili = None
    sdp = rx._subject_dp_config(cfg_t["ml"])
    if sdp["enabled"] and sdp["senza_chiave"] == "profili":
        from ml.profili_abitudini import addestra, assegna
        pp = sdp.get("profili") or {}
        n_sito: dict[str, int] = {}
        for s in train_n:
            n_sito[str(s.get("site_id") or "")] = n_sito.get(str(s.get("site_id") or ""), 0) + 1
        mod = addestra(riserva_n, dimensione_media=float(pp.get("dimensione_media", 10.0)),
                       peso_gruppo=float(pp.get("peso_gruppo", 1.0)), seed=a.seed,
                       fonte="riserva dello sfondo", n_training_per_sito=n_sito)
        assegna(mod, norm_list)
        profili = mod.descrizione()

    # riferimento: stesso meccanismo del bersaglio, persone di riferimento +
    # sfondo; chiave della cache = contenuto di tutto cio' che lo determina
    import hashlib
    codice = hashlib.sha256(b"".join(
        (ROOT / f).read_bytes() for f in (
            "scripts/run_experiments.py", "scripts/mia_persona.py", "src/ml/autoencoder_trainer.py",
            "src/ml/subject_dp.py", "src/ml/gradient_manager.py", "src/ml/fedavg_aggregator.py",
            "src/ml/profili_abitudini.py", "src/core/autoencoder.py"))).hexdigest()[:16]
    chiave = hashlib.sha256(json.dumps({
        "codice": codice,
        "cfg": cfg_t, "meccanismo": a.meccanismo, "seed": a.seed, "min": a.min_sessioni,
        "feats": feats, "norm": info_norm, "n_sessioni": len(sessions),
        "train_rif": sorted(str(s.get("session_id")) for s in train_rif),
    }, sort_keys=True, default=str).encode()).hexdigest()[:16]
    tag = Path(a.config).stem
    cache = out / f"riferimento_{tag}_{a.meccanismo}_s{a.seed}_{chiave}.npy"
    if cache.exists():
        loss_ref = np.load(cache)
        if len(loss_ref) != len(sessions):
            raise RuntimeError(f"cache {cache} non coerente: {len(loss_ref)} != {len(sessions)}")
    else:
        _semina(a.seed + 1)
        w_ref, _ = _allena(copy.deepcopy(cfg_t), rif_n, mec["no_dp"], mec.get("dp_mode", "dp-fedavg"))
        loss_ref = _perdite(cfg_t, w_ref, norm_list, feats)
        tmp = cache.with_name(cache.stem + f".{os.getpid()}.tmp.npy")
        np.save(tmp, loss_ref)
        tmp.replace(cache)  # scrittura atomica
        cache.with_suffix(".json").write_text(json.dumps(
            {"meccanismo": a.meccanismo, "seed": a.seed, "config": a.config, "min": a.min_sessioni,
             "sessioni_train_rif": len(train_rif), "chiave": chiave}, indent=1))
    if a.solo_riferimento:
        print(f"riferimento pronto: {cache}")
        return

    # bersaglio
    _semina(a.seed)
    t1 = time.time()
    w_t, cs = _allena(cfg_t, train_n, mec["no_dp"], mec.get("dp_mode", "dp-fedavg"))
    t_train = time.time() - t1
    loss_t = _perdite(cfg_t, w_t, norm_list, feats)

    # costo: loss del bersaglio sulle sessioni dei non membri (mai viste)
    idx_nm = [pos[id(s)] for u in non_membri for s in per_utente[u]]
    costo = float(np.mean(loss_t[idx_nm]))

    # attacchi
    semi = [int(x) for x in str(a.seme).split(",")]
    ritr_n = [int(x) for x in str(a.ritrovate).split(",")]
    c, k = _feature_profilo(sessions)
    n_staz = [int(k[:, j].max()) + 1 for j in range(k.shape[1])]
    utente_di = np.array([str(s.get("user_id")) if s.get("user_id") not in (None, "") else ""
                          for s in sessions])
    ordine = {u: sorted((pos[id(s)] for s in per_utente[u]), key=lambda i: sessions[i]["start_time"])
              for u in valutati}
    persone = sorted(valutati)
    lab = [1 if u in membri else 0 for u in persone]
    ris: dict[str, Any] = {}
    prec: dict[str, float] = {}

    pval: dict[str, float] = {}

    def _due(gruppi, nome):
        for versione in ("grezzo", "calibrato"):
            sc = [(-float(np.mean(loss_t[g])) if versione == "grezzo"
                   else float(np.mean(loss_ref[g] - loss_t[g]))) for g in gruppi]
            ris[f"{nome}_{versione}"] = _auc_ci(lab, sc)
            pval[f"{nome}_{versione}"] = _p_auc(lab, sc)

    _due([np.asarray(ordine[u]) for u in persone], "oracolo")
    for s_n in semi:
        semi_u = {u: ordine[u][:s_n] for u in persone}
        _due([np.asarray(semi_u[u]) for u in persone], f"seme{s_n}")
        max_n = max(ritr_n)
        trovate = {u: _ritrova(c, k, np.array(semi_u[u]), max_n, n_staz=n_staz) for u in persone}
        for n in ritr_n:
            # le n piu' simili fra le max_n trovate
            gr = {}
            for u in persone:
                idx_s = np.array(semi_u[u])
                cand = trovate[u]
                lk = -0.5 * ((c[idx_s][:, None, :] - c[cand][None, :, :]) ** 2).sum(-1) / (0.15 ** 2)
                for j, nv in enumerate(n_staz):
                    ug = k[idx_s, j][:, None] == k[cand, j][None, :]
                    lk = lk + np.where(ug, math.log(0.7), math.log(0.3 / max(nv - 1, 1)))
                m = lk.max(0)
                sc_c = m + np.log(np.exp(lk - m).sum(0))
                scelte = cand[np.argsort(-sc_c)[:n]]
                gr[u] = np.concatenate([idx_s, scelte])
                prec.setdefault(f"seme{s_n}_rit{n}", 0.0)
                prec[f"seme{s_n}_rit{n}"] += float(np.mean(utente_di[scelte] == u)) / len(persone)
            _due([gr[u] for u in persone], f"seme{s_n}_rit{n}")
    # livello sessione (riferimento): bootstrap sulle persone
    idx_val = [(i, 1 if u in membri else 0, u) for u in persone for i in ordine[u]]
    lab_s = [y for _, y, _ in idx_val]
    gr_s = [u for _, _, u in idx_val]
    ris["sessione_grezzo"] = _auc_ci(lab_s, [-float(loss_t[i]) for i, _, _ in idx_val], gruppi=gr_s)
    ris["sessione_calibrato"] = _auc_ci(lab_s, [float(loss_ref[i] - loss_t[i]) for i, _, _ in idx_val],
                                        gruppi=gr_s)
    # contabilita' della cella, sui dati di training di questa run
    n_per_client = {cid: len(v) for cid, v in cs.items()}
    contab: dict[str, Any] = {}
    contab.update(rx._record_dp_fields(cfg_t, len(train_n), n_per_client))
    if sdp["enabled"]:
        contab.update(rx._subject_dp_fields(cfg_t, rx._subject_stats(cs, str(sdp["subject_key"]), feats)))
    contab.update(rx._client_dp_fields(cfg_t, mec.get("dp_mode", "dp-fedavg"), mec["no_dp"], n_per_client))
    # metrica primaria dichiarata: p senza correzione; Holm sulle secondarie
    primaria = "oracolo_calibrato"
    holm = _holm({k: v for k, v in pval.items() if k != primaria})
    holm[primaria] = pval.get(primaria)
    esito = {
        "seed": a.seed, "meccanismo": a.meccanismo, "config": a.config, "commit": _commit(),
        "min_sessioni": a.min_sessioni, "rounds": cfg["experiment"]["fl_rounds"],
        "epochs": cfg["ml"]["epochs"], "membri": len(membri), "non_membri": len(non_membri),
        "persone_riferimento": len(persone_rif), "sessioni_train": len(train),
        "sessioni_riserva_profili": len(riserva), "seme": semi, "ritrovate": ritr_n,
        "precisione_reid": prec, "costo_loss_non_membri": costo,
        "metrica_primaria": "oracolo_calibrato",
        "auc": {k2: {"auc": v[0], "ci95": [v[1], v[2]], "p": pval.get(k2), "p_holm": holm.get(k2)}
                for k2, v in ris.items()},
        "riferimento": {"meccanismo": "come il bersaglio", "dati": "persone di riferimento + sfondo",
                        "cache": cache.name},
        "normalizzazione": info_norm, "profili": profili, "contabilita": contab,
        "secondi_training": t_train, "secondi_totali": time.time() - t0,
    }
    f = out / f"mia_persona_{tag}_{a.meccanismo}_s{a.seed}_r{cfg['experiment']['fl_rounds']}_e{cfg['ml']['epochs']}.json"
    f.write_text(json.dumps(esito, indent=2))
    print(json.dumps(esito, indent=2))


if __name__ == "__main__":
    main()
