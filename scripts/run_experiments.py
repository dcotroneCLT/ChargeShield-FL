#!/usr/bin/env python3
# scripts/run_experiments.py
# ChargeShield-FL — Sprint 5: Experiment Runner
#
# Esegue il ciclo completo:
#   1. Carica ACN-Data JPL 2019 + 2020
#   2. Esegue FL rounds via ML Plane (AutoencoderTrainer + GradientManager + FedAvgAggregator)
#   3. Lancia FedMIA attack per ogni round
#   4. Valuta IDS come baseline defense
#   5. Misura AUC-ROC e privacy/utility trade-off (epsilon vs AUC-ROC)
#   6. Salva risultati in experiments/
#
# Usage:
#   python scripts/run_experiments.py --config config/experiment.yaml
#   python scripts/run_experiments.py --epsilon 0.5 --rounds 10
#   python scripts/run_experiments.py --dry-run

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapters.acn_dataset import ACNDataset
from adapters.chargeplace_scotland_adapter import ChargePlaceScotlandDataset
from auditor.privacy_auditor import PrivacyAuditor
from auditor.privacy_auditor_subscriber import PrivacyAuditorSubscriber
from core.autoencoder import Autoencoder
from ids.charging_ids import ByzantineDetector
from ml.autoencoder_trainer import AutoencoderTrainer
from ml.fedavg_aggregator import FedAvgAggregator
from ml.gradient_manager import GradientManager
from ml.ml_plane import FLArtifactCollector, MLPlane
from ml.base_ml import MLPlaneEvent
from ml.record_dp_accounting import record_dp_fields as _record_dp_fields_impl
from ml.subject_dp import (
    cross_client_weights as _subject_cross_client_weights,
    subject_dp_config as _subject_dp_config,
    subject_dp_fields as _subject_dp_fields,
    subject_stats as _subject_stats,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_experiment")


# ── Architettura modello (capacità configurabile, Sprint 10jj 2026-08-28) ───────

def _auditor_telemetry(reports: dict) -> dict:
    """
    Telemetria del Privacy Auditor per un round (Sprint 10zz+119).

    reports: node_id -> AuditReport (src/core/base_auditor.py), prodotto dal
    subscriber a ogni round. Prima di questo sprint veniva consumato da
    ids.analyze_round() e poi scartato: nel JSON restava solo l'overhead.

    NOTA INTERPRETATIVA, da riportare nel paper. privacy_score ed epsilon qui
    sono un PROXY calcolato dalla norma degli aggiornamenti, non una misura del
    rischio di appartenenza e non una contabilita' DP formale (epsilon per
    round e' scalato sulla mediana dei peer, quindi dipende dai dati, mentre la
    contabilita' DP e' data-independent per costruzione). Servono a verificare
    che un nodo resti dentro i parametri dichiarati, non a stimare quanta
    informazione esce. La distinzione e' il motivo per cui questo lavoro
    instanzia avversari invece di leggere il budget.
    """
    if not reports:
        return None
    per_node = {}
    for node_id, r in reports.items():
        per_node[node_id] = {
            "privacy_score":    getattr(r, "privacy_score", None),
            "epsilon":          getattr(r, "epsilon", None),
            "threats_detected": list(getattr(r, "threats_detected", []) or []),
            "metadata":         dict(getattr(r, "metadata", {}) or {}),
        }
    scores = [v["privacy_score"] for v in per_node.values() if v["privacy_score"] is not None]
    eps    = [v["epsilon"]       for v in per_node.values() if v["epsilon"]       is not None]
    return {
        "per_node":            per_node,
        "n_nodes":             len(per_node),
        "privacy_score_min":   min(scores) if scores else None,
        "privacy_score_mean":  (sum(scores) / len(scores)) if scores else None,
        "epsilon_round_max":   max(eps) if eps else None,
        "epsilon_round_mean":  (sum(eps) / len(eps)) if eps else None,
        "n_threats":           sum(len(v["threats_detected"]) for v in per_node.values()),
        "note": ("privacy_score/epsilon sono un proxy data-dependent dalla norma "
                 "degli update, NON una misura di rischio di membership ne' una "
                 "contabilita' DP formale."),
    }

def _record_dp_fields(
    cfg: dict,
    n_sessions: int | None = None,
    n_per_client: dict[str, int] | None = None,
) -> dict:
    """
    Campi record-DP per il JSON risultato (Sprint 10zz+117).

    Senza questi, una cella con DP per-record e' indistinguibile da una
    no-DP: il campo "epsilon" del JSON e' il budget CLIENT-level e non si
    applica al meccanismo per-record, che ha un budget suo calcolato via
    accountant RDP.

    Correzione della segnalazione 4 (2026-09-24): il calcolo e' spostato in
    src/ml/record_dp_accounting.py. Prima leggeva fl_rounds e delta dalla
    radice di cfg (vivono in cfg["experiment"]: rounds valeva sempre 1),
    usava il numero globale di sessioni invece di quello di ogni client e
    dichiarava un campionamento di Poisson senza avvertire che il training
    usa shuffle. Ora: round e delta effettivi, epsilon per client e massimo,
    piu' il limite valido con lo shuffle. Vedi la docstring del modulo.
    """
    return _record_dp_fields_impl(cfg, n_sessions=n_sessions, n_per_client=n_per_client)

def _autoencoder_arch_kwargs(cfg: dict) -> dict[str, Any]:
    """
    Estrae hidden_dims/latent_dim opzionali da cfg['ml'] per istanziare
    Autoencoder() con la STESSA architettura usata dal trainer FL reale
    (AutoencoderTrainer legge le stesse chiavi — vedi src/ml/autoencoder_trainer.py).

    Necessario in ogni punto di questo file che ricostruisce un Autoencoder()
    per caricare global_weights via load_state_dict(strict=True) (FedMIA,
    Shadow MIA, LiRA) o che addestra un modello shadow "gemello" del client
    reale: un mismatch di architettura tra trainer e ricostruzione qui
    causerebbe un RuntimeError di shape mismatch (weights) o un confronto
    shadow/target non comparabile (LiRA).

    Default: hidden_dims=None, latent_dim=4 → architettura storica (16, 8)/4,
    570 parametri, invariata per ogni config YAML che non imposta
    esplicitamente hidden_dims (cioè tutti i run esistenti/pubblicati).
    """
    ml_cfg = cfg.get("ml", {})
    hidden_dims = ml_cfg.get("hidden_dims")
    return {
        "hidden_dims": tuple(hidden_dims) if hidden_dims is not None else None,
        "latent_dim": ml_cfg.get("latent_dim", 4),
        # norm (Sprint 10zz+115): parametro ARCHITETTURALE a tutti gli effetti,
        # e quindi di competenza di questa funzione. BatchNorm1d contribuisce
        # 4 tensori per layer allo state_dict (weight, bias, running_mean,
        # running_var, + num_batches_tracked), GroupNorm solo 2: con
        # hidden_dims=(32,16) sono 22 voci contro 16. Se il target gira in
        # "group" (obbligatorio con record_dp, vedi src/core/autoencoder.py) e
        # i modelli ricostruiti qui nascono in "batch", ogni attacco salta con
        # "global_weights ha 16 elementi, state_dict ne richiede 22" e il run
        # non produce alcuna misura. E' esattamente la classe di mismatch che
        # questa funzione esiste per prevenire.
        # Segnalazione 49 (2026-09-24): il trainer forza "group" quando record_dp
        # e' attivo (src/ml/autoencoder_trainer.py); qui si ricostruiva invece
        # "batch" se il config non lo diceva, e FedMIA/Shadow/LiRA saltavano
        # ogni round ("global_weights ha 16 elementi, state_dict ne richiede
        # 22"). Stessa regola del trainer, cosi' le due architetture coincidono.
        # 2026-10-13: stessa regola per la DP per soggetto (il trainer forza
        # "group" anche li', src/ml/autoencoder_trainer.py).
        # subject_dp letto con la stessa funzione del trainer (che tratta la
        # stringa "false" come falso): un bool() diretto darebbe "group" qui e
        # "batch" nel trainer, e gli attacchi salterebbero per shape mismatch.
        "norm": ("group" if ((ml_cfg.get("record_dp") or {}).get("enabled")
                             or _subject_dp_config(ml_cfg)["enabled"])
                 else str(ml_cfg.get("norm", "batch"))),
    }


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: Path | None, overrides: dict) -> dict:
    """Carica config da YAML e applica override da CLI."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        raise FileNotFoundError(
            f"Config non trovata: {config_path}. "
            "Specifica --config o crea config/experiment.yaml"
        )
    if overrides.get("epsilon") is not None:
        cfg["experiment"]["epsilon"] = overrides["epsilon"]
    if overrides.get("rounds") is not None:
        cfg["experiment"]["fl_rounds"] = overrides["rounds"]
    if overrides.get("epochs") is not None:
        # Override cfg["ml"]["epochs"] (default 50) — usato sia per il training
        # locale reale dei client (run_fl_rounds()) sia come base della formula
        # shadow_epochs in run_lira()/run_shadow() (min(epochs*rounds, 500)
        # / min(epochs*max(rounds//4,5), 300)), quindi alzarlo alza anche il
        # training degli shadow, non solo del modello target — coerente col resto
        # della pipeline, nessun trattamento speciale necessario qui.
        # Aggiunto 2026-08-27 per la calibrazione empirica del sanity-check
        # positivo (README docs/TestRoadmap_DSN2027.md #2): prima non esisteva un
        # modo per variare epochs da riga di comando, solo editando il config.
        cfg["ml"]["epochs"] = overrides["epochs"]
    return cfg


# ── Dataset ────────────────────────────────────────────────────────────────────

# Mappa siteID ACN-Data (campo "site_id" già estratto da ACNDataset.get_sample(),
# vedi src/adapters/acn_dataset.py) → nome sito leggibile usato come cluster_id
# nell'FL. Verificata empiricamente 2026-07-22 confrontando il numero di stazioni
# uniche per siteID coi valori ufficiali pubblicati su https://ev.caltech.edu/dataset
# (Caltech 54 EVSE, JPL 50 EVSE, Office 1 8 EVSE): siteID="0002"→54 stazioni→Caltech,
# siteID="0001"→52 stazioni (≈50 ufficiali)→JPL, siteID="0019"→8 stazioni→Office 1.
# FIX 2026-07-22 (scoperta importante): il progetto ha SEMPRE chiamato "jpl" il
# dataset con siteID="0002" (54 stazioni) — che in realtà è Caltech, non JPL (50
# EVSE). Ogni esperimento precedente (nodp-sweep1/dp-sweep1 inclusi) ha quindi
# usato dati Caltech mal etichettati come "jpl" — vedi README/CaseStudies.md per
# la correzione della documentazione storica. Questa mappa ora usa l'identità
# corretta, verificata via conteggio stazioni, non il nome storico del file.
_SITE_ID_TO_NAME = {
    "0002": "caltech",
    "0001": "jpl",
    "0019": "office1",
}


def load_sessions_chargeplace_scotland(cfg: dict) -> list[dict[str, Any]]:
    """
    Carica sessioni EV da ChargePlace Scotland (task #37, adapter aggiunto
    2026-09-09 — vedi src/adapters/chargeplace_scotland_adapter.py).

    A differenza di ACN-Data (un file per sito), ChargePlace Scotland ha un
    set di file MENSILI condivisi che coprono tutte le 32 council area
    scozzesi insieme — il raggruppamento per sito (site_id = local_authority)
    avviene DOPO il caricamento, filtrando a `cfg["chargeplace_scotland"]["clients"]`
    (lista di nomi di local_authority esatti, es. "Glasgow City"). Sessioni di
    council area non in questa lista vengono scartate qui, PRIMA che arrivino
    a group_sessions_by_site() — altrimenti risulterebbero in 32 client FL
    invece dei client scelti.

    LIMITE NOTO (da tenere presente leggendo i risultati): questo dataset non
    ha equivalenti di kwh_requested/minutes_available (sempre 0.0/0, vedi
    adapter) — 2 delle 6 feature di input (input_dim=6, vedi cfg["ml"]) sono
    quindi costanti per ogni sessione. compute_feature_stats() gestisce già
    fmax==fmin senza errori (fallback fmin+1.0), quindi non c'è un crash o un
    NaN, ma il modello ha di fatto solo 4 feature informative invece di 6 —
    diverso da ACN-Data, da menzionare se questi risultati finiscono nel paper.
    """
    section = cfg.get("chargeplace_scotland") or {}
    session_paths = section.get("session_files") or []
    metadata_dir = section.get("metadata_dir")
    clients = set(section.get("clients") or [])
    if not session_paths or not metadata_dir:
        raise ValueError(
            "cfg['chargeplace_scotland'] deve avere 'session_files' e 'metadata_dir' "
            "quando cfg['dataset_adapter'] == 'chargeplace_scotland'"
        )

    dataset = ChargePlaceScotlandDataset()
    resolved_paths = [str(PROJECT_ROOT / p) for p in session_paths]
    dataset.load_with_metadata(
        session_paths=resolved_paths,
        metadata_dir=str(PROJECT_ROOT / metadata_dir),
    )
    all_sessions = [dataset.get_sample(i) for i in range(len(dataset))]
    logger.info(f"ChargePlace Scotland: {len(all_sessions)} sessioni totali caricate (tutte le council area)")

    if clients:
        sessions = [s for s in all_sessions if s.get("site_id") in clients]
        logger.info(
            f"ChargePlace Scotland: {len(sessions)} sessioni dopo il filtro client "
            f"({sorted(clients)}) — {len(all_sessions) - len(sessions)} scartate (altre council area)"
        )
    else:
        sessions = all_sessions
        logger.warning(
            "cfg['chargeplace_scotland']['clients'] vuoto — usando TUTTE le 32 council "
            "area come client FL separati (probabilmente non voluto, verifica il config)"
        )

    if not sessions:
        raise ValueError(
            f"Nessuna sessione trovata per i client richiesti: {sorted(clients)} — "
            "controlla che i nomi in cfg['chargeplace_scotland']['clients'] combacino "
            "esattamente con i valori di local_authority in CPID_and_local_authority.xlsx"
        )
    return sessions


def load_sessions(cfg: dict) -> list[dict[str, Any]]:
    """
    Carica sessioni EV — da ACN-Data (default, comportamento storico) o da
    ChargePlace Scotland (task #37) se cfg["dataset_adapter"] ==
    "chargeplace_scotland". Il default resta "acn" per retrocompatibilità
    totale: nessun config esistente (config/experiment.yaml e derivati) è
    affetto da questo dispatch, perché nessuno di essi imposta questa chiave.

    Per ACN-Data: carica da TUTTI i siti/anni elencati in cfg["sites"]
    (2026-07-22: sostituisce il precedente cfg["datasets"] — un solo dataset
    condiviso affettato arbitrariamente in 4 "cluster" fittizi).
    cfg["sites"] è {nome_sito: [path_anno1, path_anno2, ...]} — ogni sito reale
    combina tutti gli anni disponibili in un unico pool (stessa logica già
    usata per jpl_2019+jpl_2020 prima di oggi, ora per sito invece che globale).

    Restituisce una lista PIATTA (stesso contratto di sempre — compute_feature_stats/
    normalize_sessions/il train-holdout split lavorano su questa lista intera).
    L'appartenenza al sito reale di ciascuna sessione resta comunque disponibile
    nel campo "site_id" (estratto da ACNDataset/ChargePlaceScotlandDataset) —
    vedi group_sessions_by_site() per il raggruppamento usato da
    run_fl_rounds()/run_lira().
    """
    if cfg.get("dataset_adapter") == "chargeplace_scotland":
        return load_sessions_chargeplace_scotland(cfg)

    sessions: list[dict[str, Any]] = []
    sites_cfg = cfg.get("sites") or cfg.get("datasets") or {}
    for site_name, paths in sites_cfg.items():
        path_list = paths if isinstance(paths, list) else [paths]
        site_count = 0
        for path_str in path_list:
            p = PROJECT_ROOT / path_str
            if not p.exists():
                logger.warning(f"Dataset non trovato: {p} — skip")
                continue
            dataset = ACNDataset()
            dataset.load(str(p))
            loaded = [dataset.get_sample(i) for i in range(len(dataset))]
            sessions.extend(loaded)
            site_count += len(loaded)
        logger.info(f"{site_name}: {site_count} sessioni caricate (tutti gli anni)")
    if not sessions:
        raise FileNotFoundError(
            "Nessun dataset trovato. Scarica ACN-Data da "
            "https://ev.caltech.edu/dataset e posizionalo in datasets/acn/<sito>/"
        )
    logger.info(f"Totale sessioni (tutti i siti): {len(sessions)}")
    return sessions


def _git_commit_corrente() -> str | None:
    """Commit HEAD del repository, con marcatore 'dirty' se ci sono modifiche.

    Aggiunto 2026-09-21 (Fase A della guida scientifica). Restituisce None se
    git non e' disponibile o la cartella non e' un repository: meglio un campo
    assente che un valore inventato, come impone la riga 150 della guida.
    """
    import subprocess
    try:
        radice = Path(__file__).resolve().parent.parent
        sha = subprocess.run(
            ["git", "-C", str(radice), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if sha.returncode != 0:
            return None
        commit = sha.stdout.strip()
        stato = subprocess.run(
            ["git", "-C", str(radice), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if stato.returncode == 0 and stato.stdout.strip():
            return f"{commit}-dirty"
        return commit
    except Exception:
        return None


def group_sessions_by_site(sessions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    Raggruppa le sessioni per SITO REALE, usando il campo "site_id" già presente
    in ogni sessione (estratto da ACNDataset dal siteID di ACN-Data) — non per
    slicing posizionale/contiguo come in precedenza.

    FIX 2026-07-22 (review indipendente su scripts/run_nvflare_mia.py, stesso
    principio applicato qui): ricostruire l'appartenenza a un cluster affettando
    una lista per indice (es. "le prime N/4 sessioni sono il cluster A") è
    fragile — dipende dall'ordine/dalla lunghezza esatta della lista a monte, e
    un mismatch tra chi produce la lista e chi la riaffetta produce risultati
    sbagliati SENZA errori visibili (esattamente il bug trovato oggi). Usare il
    campo site_id proprio di ogni sessione elimina questa intera classe di bug:
    il raggruppamento è corretto qualunque sia l'ordine/la provenienza della
    lista di sessioni.

    Sessioni con site_id sconosciuto (non in _SITE_ID_TO_NAME) sono raggruppate
    sotto il loro site_id grezzo invece di essere scartate — permette di
    scoprire nuovi siti aggiunti in futuro senza modificare questa funzione.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for s in sessions:
        site_id = s.get("site_id", "")
        name = _SITE_ID_TO_NAME.get(site_id, site_id or "unknown")
        groups.setdefault(name, []).append(s)
    return groups


def group_indices_by_site(sessions: list[dict[str, Any]]) -> dict[str, list[int]]:
    """
    Come group_sessions_by_site(), ma restituisce gli INDICI GLOBALI in
    `sessions` invece delle sessioni stesse — usato da run_lira() (fase MIA),
    che deve campionare gli shadow model da un pool di indici nello stesso
    spazio di train_sessions (per costruire, dato un round, i sotto-tensori
    IN/OUT), non dalle sessioni stesse. Vedi cluster_membership in run_lira().
    """
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(sessions):
        site_id = s.get("site_id", "")
        name = _SITE_ID_TO_NAME.get(site_id, site_id or "unknown")
        groups.setdefault(name, []).append(i)
    return groups


def inject_synthetic_client_indices(
    real_index_groups: dict[str, list[int]],
    n_synthetic: int = 2,
    seed: int = 42,
) -> dict[str, list[int]]:
    """
    Aggiunge n_synthetic client FITTIZI a real_index_groups (l'output di
    group_indices_by_site()) — SOLO per lo sweep IDS/Byzantine (config/
    experiment.yaml: byzantine_attack.enabled=True). MAI usato per l'esperimento
    privacy/FedMIA/LiRA principale, che deve vedere SOLO i 3 client reali
    (caltech/jpl/office1) — questa funzione va chiamata unicamente sul percorso
    codice dello sweep Byzantine, mai su quello di default.

    Perché servono client fittizi: Krum (usato da ByzantineDetector per il rilevamento
    Byzantine) garantisce di rilevare f nodi Byzantine solo con n≥2f+3 nodi
    totali. Con f=1 (un solo attaccante, unico scenario oggi supportato)
    servono n≥5 client totali — i soli 3 siti reali non bastano su basi
    teoriche solide. Nessun 4°/5° sito REALE esiste in ACN-Data (solo Caltech/
    JPL/Office1, verificato su https://ev.caltech.edu/dataset, sezione "Sites")
    — un vero 4°/5° sito richiederebbe un dataset EV completamente diverso.

    Come sono costruiti: pool = concatenazione di TUTTI gli indici reali (3
    siti, nell'ordine di iterazione di real_index_groups), shuffle con seed
    fisso, poi affettato in n_synthetic parti aggiuntive. Le sessioni
    referenziate possono sovrapporsi con quelle già assegnate ai client reali —
    accettabile SOLO per lo scopo di validazione IDS (Krum verifica un
    comportamento geometrico locale — un gradiente scalato artificialmente —
    non richiede popolazioni disgiunte come farebbe invece un esperimento di
    privacy/generalizzazione). Per questo motivo l'uso di questi client
    fittizi in FedMIA/LiRA sarebbe metodologicamente invalido e va evitato.
    """
    import random as _random

    pooled_indices: list[int] = [i for idxs in real_index_groups.values() for i in idxs]
    rng = _random.Random(seed + 271828)  # offset arbitrario, separato dal seed sperimentale
    shuffled = pooled_indices[:]
    rng.shuffle(shuffled)

    expanded: dict[str, list[int]] = dict(real_index_groups)
    chunk_size = max(1, len(shuffled) // n_synthetic)
    for i in range(n_synthetic):
        start = i * chunk_size
        end = len(shuffled) if i == n_synthetic - 1 else start + chunk_size
        expanded[f"synthetic_{i + 1}"] = shuffled[start:end]
    return expanded

# ── Session enrichment ─────────────────────────────────────────────────────────
def enrich_sessions(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Aggiunge feature derivate dai timestamp ACN-Data.
    - hour_of_day: ora di connessione LOCALE al sito (0–23), pattern comportamentale
    - duration_hours: durata sessione in ore, correlata all'energia

    Fix 2026-07-22 (review indipendente fresh-pass, bug reale confermato
    empiricamente): start_time/end_time (da ACNDataset) sono UTC nonostante il
    suffisso "GMT" nel dato grezzo ACN-Data sia fuorviante — vedi commento su
    "timezone" in src/adapters/acn_dataset.py per la verifica empirica (picco
    orario grezzo di office1 implausibile per charging da ufficio, plausibile
    dopo conversione a America/Los_Angeles). hour_of_day va quindi calcolato
    sull'ora LOCALE del sito, non sui pesi grezzi di start.hour (che erano UTC,
    sistematicamente sfasati di 7-8h da quanto la feature dichiara di
    rappresentare — "ora di connessione, pattern comportamentale"). Non
    tocchiamo start_time/end_time stessi (restano UTC — usati altrove per
    year-splitting/holdout, che non deve essere influenzato da questo fix) né
    duration_hours (invariante a un offset UTC uniforme, entrambi i lati
    dell'intervallo si spostano della stessa quantità).
    """
    from zoneinfo import ZoneInfo

    enriched = []
    for s in sessions:
        try:
            start = datetime.fromisoformat(s["start_time"])
            end   = datetime.fromisoformat(s["end_time"])

            tz_name = s.get("timezone")
            if tz_name:
                try:
                    local_start = start.replace(tzinfo=ZoneInfo("UTC")).astimezone(
                        ZoneInfo(tz_name)
                    )
                    hour_of_day = float(local_start.hour)
                except Exception:
                    # Timezone IANA sconosciuta/malformata — fallback all'ora
                    # grezza (comportamento pre-fix) invece di scartare la
                    # sessione: un hour_of_day leggermente sfasato è preferibile
                    # a perdere il campione.
                    hour_of_day = float(start.hour)
            else:
                hour_of_day = float(start.hour)  # nessun timezone noto — fallback

            s["hour_of_day"]    = hour_of_day
            # hour_of_day_sin/_cos (2026-08-31, Fase 8 — opt-in, non tocca il
            # default): encoding circolare di hour_of_day, così 23h e 0h
            # risultano vicine nello spazio delle feature quanto lo sono
            # comportamentalmente (hour_of_day lineare le tratta come le più
            # lontane possibili, 23 vs 0). Calcolate SEMPRE (costo
            # trascurabile), ma usate dal modello SOLO se esplicitamente
            # elencate in ml.feature_names al posto di "hour_of_day" — stesso
            # meccanismo opt-in già usato per start_time_epoch (Sprint 10kk,
            # vedi config/experiment_overfit_calibration_richfeat.yaml).
            # Nessuna config esistente/pubblicata le referenzia: nessun run
            # già eseguito è affetto, l'architettura storica a 6 feature/570
            # parametri resta il default invariato.
            s["hour_of_day_sin"] = math.sin(2.0 * math.pi * hour_of_day / 24.0)
            s["hour_of_day_cos"] = math.cos(2.0 * math.pi * hour_of_day / 24.0)
            s["duration_hours"] = max(0.0, (end - start).total_seconds() / 3600.0)
            # start_time_epoch (2026-08-28, Sprint 10kk — escalation feature-entropy
            # del sanity-check LiRA, vedi docs/TestRoadmap_DSN2027.md #2, passo 2).
            # Timestamp Unix (secondi, UTC — start è naive-UTC, vedi commento sopra)
            # dell'inizio sessione: una feature reale, non un ID opaco iniettato, ma
            # a risoluzione abbastanza fine da essere quasi univoca per sessione
            # (collisioni al secondo estremamente rare su ~1300+ sessioni per sito).
            # NON usata di default — entra nel tensore solo se esplicitamente elencata
            # in ml.feature_names (vedi config/experiment_overfit_calibration_richfeat.yaml).
            # Ogni config esistente/pubblicato non la referenzia: nessun run è affetto.
            s["start_time_epoch"] = start.replace(tzinfo=ZoneInfo("UTC")).timestamp()
            enriched.append(s)
        except (KeyError, ValueError):
            pass  # scarta sessioni con timestamp malformati
    return enriched


# ── Entity-aware split (Fase 8, 2026-08-31) — opt-in, non il default ───────────

def entity_aware_split(
    sessions: list[dict[str, Any]],
    entity_key: str,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split train/holdout che garantisce l'indipendenza dei non-membri: tutte
    le sessioni con lo stesso `entity_key` (es. "node_id" = stazione EVSE, o
    "user_id" se popolato) finiscono INTERAMENTE in train O in holdout, mai
    divise tra i due.

    Perché: lo split di default (random.shuffle su sessioni individuali, poi
    80/20) può assegnare due sessioni della STESSA stazione/utente
    rispettivamente a train e holdout — violando l'assunzione di indipendenza
    statistica tra membri e non-membri su cui si basa la valutazione MIA
    (obiezione standard di un revisore: "i tuoi non-membri sono davvero
    indipendenti, o solo sessioni diverse della stessa entità già vista in
    training?"). Questo split la chiude per costruzione.

    OPT-IN, non il default: cambia QUALI sessioni specifiche sono
    membri/non-membri, quindi non è direttamente comparabile con la campagna
    5-seed×8-config già completata (Sprint 10tt, split random). Attivabile
    via cfg["split"]["strategy"] = "entity_aware" nel config YAML — il
    default resta "random" (comportamento storico invariato, usato da ogni
    risultato già pubblicato). Pensato come robustness experiment separato:
    se il leakage misurato resta ≈0 anche con questo split più severo,
    rafforza — non sostituisce — il risultato principale.

    Algoritmo: raggruppa le sessioni per entity_key, mescola l'ORDINE dei
    gruppi (non delle sessioni singole) con il seed dato, poi assegna gruppi
    interi a holdout finché la frazione target non è raggiunta (greedy —
    con molti gruppi piccoli converge vicino alla frazione esatta; con pochi
    gruppi grandi può discostarsene, loggato esplicitamente così lo scarto
    non passa inosservato).

    Args:
        sessions:         lista di sessioni enrichite (post enrich_sessions())
        entity_key:       campo su cui raggruppare (es. "node_id", "user_id")
        holdout_fraction: frazione approssimativa di sessioni per l'holdout
        seed:             seed per lo shuffle dei gruppi (riproducibilità)

    Returns:
        (train_sessions, holdout_sessions)
    """
    groups: dict[Any, list[dict[str, Any]]] = {}
    ungrouped: list[dict[str, Any]] = []
    for s in sessions:
        key = s.get(entity_key)
        if key is None:
            # Nessun valore per entity_key (es. user_id spesso assente in
            # ACN-Data pubblico) — trattata come entità a sé stante, non
            # unita ad altre sessioni senza key (evita di raggruppare
            # falsamente sessioni non correlate sotto la stessa chiave None).
            ungrouped.append(s)
            continue
        groups.setdefault(key, []).append(s)

    rng = random.Random(seed)
    group_items: list[tuple[Any, list[dict[str, Any]]]] = list(groups.items())
    group_items.extend((id(s), [s]) for s in ungrouped)
    rng.shuffle(group_items)

    total = len(sessions)
    target_holdout = int(total * holdout_fraction)

    holdout_sessions: list[dict[str, Any]] = []
    train_sessions: list[dict[str, Any]] = []
    for _key, group_sessions in group_items:
        if len(holdout_sessions) < target_holdout:
            holdout_sessions.extend(group_sessions)
        else:
            train_sessions.extend(group_sessions)

    _achieved = (len(holdout_sessions) / total) if total else 0.0
    logger.info(
        f"[SPLIT entity_aware] entity_key={entity_key!r} — {len(groups)} entità "
        f"({len(ungrouped)} sessioni senza {entity_key}, trattate come entità "
        f"singole) — train={len(train_sessions)}, holdout={len(holdout_sessions)} "
        f"(target holdout_fraction={holdout_fraction:.2f}, ottenuto {_achieved:.2f})"
    )
    return train_sessions, holdout_sessions


# ── Canary Positive Control (Sprint 10vv, 2026-08-31) ───────────────────────────

def inject_canaries(
    train_sessions: list[dict[str, Any]],
    holdout_sessions: list[dict[str, Any]],
    cfg: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Inietta un piccolo gruppo di sessioni "canary" per un vero positive
    control sull'harness LiRA — richiesto esplicitamente dall'utente
    (2026-08-31) dopo aver notato che il sanity-check a 5 assi (Sprint
    10ee-10nn) prova solo che la memorizzazione "naturale" è difficile da
    indurre su questi dati, NON che l'harness sarebbe in grado di rilevare
    una violazione di privacy se ci fosse davvero (vedi Limitazione #7 del
    paper). Tecnica standard in letteratura DP/MIA ("canary insertion" —
    cfr. Carlini "The Secret Sharer" 2019, Jagielski et al. "Auditing
    Differentially Private Machine Learning" 2020): sceglie n_templates
    sessioni REALI dal training set di un singolo client come "template" e
    ne inserisce n_duplicates copie ESATTE nel training set di quello
    stesso client, amplificando il contributo di quel record al gradiente
    di un fattore n_duplicates — la manipolazione più diretta e aggressiva
    possibile per indurre memorizzazione, più diretta di qualunque dei 5
    assi già testati (epoche/capacità/feature/sito). Un gruppo di controllo
    non-membro di pari sito — sessioni REALI distinte campionate
    dall'holdout dello stesso client, MAI copie dei template membro — serve
    da confronto pulito membro-vs-non-membro ristretto ai soli canary
    (canary_auc_roc in run_lira()).

    Correzione (2026-09-15, errata "config legacy fuorvianti" §5 —
    autocontraddizione nella docstring trovata da un feedback esterno
    verificato): questo paragrafo diceva "copie singole degli stessi
    template" per il gruppo non-membro. Falso — vedi il Fix 2026-08-31 più
    sotto e il codice (`nonmember_templates = rng.sample(site_holdout_sessions,
    n_nonmember_templates)`): il lato non-membro è sempre stato sessioni
    reali indipendenti dall'holdout, mai copie dei template membro. Il paper
    (§6.2) descriveva questo gruppo come "held-out sibling sessions", che
    suggerisce lo stesso errore — corretto anche lì (Sprint 10zz+87) per
    dire esplicitamente "independently-sampled ... ordinary holdout
    sessions, not copies of the duplicated templates".

    Attivo SOLO se cfg["canary"]["enabled"] è True — default assente/False,
    quindi no-op per ogni config/run esistente, inclusa l'intera campagna
    5-seed×8-config appena conclusa (Sprint 10tt): zero rischio di
    invalidare risultati già pubblicati.

    Ogni sessione canary/gemella porta due campi di bookkeeping,
    "_canary_group" (stringa) e "_canary_role" ("member"/"nonmember") — MAI
    usati come feature (esclusi da _mia_feature_names, che legge solo
    cfg["ml"]["feature_names"]), letti solo da run_lira() per calcolare
    canary_auc_roc separatamente dall'AUC principale. Nessuna formula/
    soglia/pooling esistente viene toccata.

    Fix 2026-08-31 (dopo il primo run con canary reali: canary_auc_roc
    instabile in segno tra round — 0.21/0.38/0.71 — con un outlier evidente,
    t=0.865 contro un range normale di 0.0001-0.002, che da solo può
    ribaltare un AUC calcolato su soli 5 non-membri): n_nonmember_templates
    ora separato da n_templates (che resta il conteggio SOLO lato membro,
    duplicato n_duplicates volte). Un lato non-membro più numeroso (es. 20
    invece di 5, nessuna duplicazione necessaria lì — bastano sessioni reali
    distinte) riduce l'effetto di un singolo outlier sull'AUC canary senza
    toccare affatto la logica di iniezione lato training. canary_auc_roc in
    run_lira() non richiede un accoppiamento 1:1 membro↔gemello per gruppo —
    aggrega semplicemente tutti i membri taggati contro tutti i non-membri
    taggati — quindi i due lati possono avere numerosità diverse senza
    alcuna modifica alla logica di scoring.

    Fix 2026-09-15 (Sprint 10zz+96, task #155/#121 — root cause del
    `shadow_canary_auc_roc` invertito 0.26/0.32/0.32 osservato nel primo run
    reale di Blocker 2 a tre attacchi): l'occorrenza REALE originale di ogni
    template membro (l'elemento di `site_train_sessions` scelto da
    `rng.sample()`, prima di questo fix mai taggata) viene ora sostituita
    dentro `injected_train` con una copia taggata (`_canary_group`/
    `_canary_role="member"`, stessi valori dei suoi cloni) invece di restare
    una sessione membro ordinaria indistinguibile. Prima del fix, quella
    sessione poteva finire nel `shadow_train` di `run_shadow()` (la
    guardia anti-contaminazione di Sprint 10zz+93 la cerca per tag, non per
    contenuto) e nell'universo shadow non filtrato di `run_lira()` (la
    stessa classe di contaminazione già corretta per i cloni TAGGATI in
    `_sample_preserving_canary_groups`, Sprint 10zz+16, ma non per
    quest'occorrenza priva di tag) — allenando lo shadow model direttamente
    sul contenuto del canary. Ogni gruppo membro conta ora `n_duplicates + 1`
    occorrenze taggate identiche (non `n_duplicates` cloni + 1 originale
    invisibile), un'unità atomica genuina per qualunque guardia/filtro
    esistente basato su `_canary_role`/`_canary_group`. INVALIDA i numeri
    canary già raccolti con canary attivo (Sprint 10zz+18 LiRA composed
    0.6875, Sprint 10zz+95 Blocker 2 a 3 attacchi) — vanno rilanciati.
    """
    canary_cfg = cfg.get("canary", {})
    if not canary_cfg.get("enabled", False):
        return train_sessions, holdout_sessions

    site                  = canary_cfg.get("site", "office1")
    n_templates           = int(canary_cfg.get("n_templates", 5))
    n_duplicates          = int(canary_cfg.get("n_duplicates", 30))
    n_nonmember_templates = int(canary_cfg.get("n_nonmember_templates", n_templates))

    # Offset di seed dedicato (271828, cifre di 'e') — indipendente da ogni
    # altro uso di `seed` in questa pipeline (split train/holdout, shuffle
    # sessioni, ecc.), per non alterare quei campionamenti quando i canary
    # sono disattivati o attivi con parametri diversi.
    rng = random.Random(seed + 271828)

    # Fix 2026-08-31 (bug reale, trovato dal primo run della Fase 0 sulla
    # macchina dell'utente: "train=0, holdout=0" nonostante 1344/336 sessioni
    # office1 caricate): s["site_id"] porta il CODICE ACN-Data grezzo (es.
    # "0019"), non il nome leggibile ("office1") — confrontarlo direttamente
    # con `site` (che arriva da cfg["canary"]["site"], un nome leggibile)
    # non trovava mai corrispondenza. Fix: risolvere il nome esattamente come
    # group_indices_by_site() fa già (_SITE_ID_TO_NAME), invece di reinventare
    # una logica di risoluzione diversa qui.
    def _resolved_site_name(s: dict[str, Any]) -> str:
        raw = s.get("site_id", "")
        return _SITE_ID_TO_NAME.get(raw, raw or "unknown")

    site_train_sessions   = [s for s in train_sessions   if _resolved_site_name(s) == site]
    site_holdout_sessions = [s for s in holdout_sessions if _resolved_site_name(s) == site]

    # ------------------------------------------------------------------
    # Fix 2026-09-15 (Sprint 10zz+109, task #160 — confondente
    # membership/difficolta' di ricostruzione, diagnosticato su
    # _d1_canary_realdp/experiment_20260915_155515.json da una review
    # esterna). Fino a questo fix i due lati del controllo positivo
    # venivano estratti da POOL DIVERSI: i template membro da
    # site_train_sessions, i gemelli non-membro da site_holdout_sessions.
    # Nessuna procedura garantiva che i due gruppi fossero equiparabili
    # per difficolta' intrinseca di ricostruzione, e nei dati reali non
    # lo erano: shadow_canary_debug_group_raw riporta, sotto modelli
    # shadow PULITI (mai rumorizzati), una loss media di ~0.00014 per i
    # cinque template membro contro ~0.00097 per i venti non-membri, un
    # fattore ~7x. Qualunque attacco che soglia sulla loss separa i due
    # gruppi a prescindere dall'appartenenza, quindi canary_auc_roc
    # misurava membership CONFUSA con difficolta' della sessione.
    #
    # NOTA su una soluzione scartata: usare come gemelli copie ESATTE dei
    # template membro tenute fuori dal training non funziona. A parita' di
    # vettore di feature il modello produce la stessa ricostruzione e
    # quindi la stessa loss per entrambi i lati, e l'AUC vale 0.5 per
    # costruzione qualunque sia la memorizzazione. Il controllo positivo
    # diventerebbe incapace di rilevare alcunche'.
    #
    # Correzione adottata (disegno standard in letteratura canary): un
    # POOL UNICO e una ASSEGNAZIONE CASUALE. Si estraggono
    # n_templates + n_nonmember_templates sessioni dal training set del
    # sito con un'unica chiamata a rng.sample(), le si mescola, e si
    # assegna la prima parte al gruppo membro (duplicato, resta in train)
    # e la seconda al gruppo non-membro (RIMOSSO da train e spostato in
    # holdout, cosi' che sia genuinamente non-membro). I due gruppi sono
    # ora campioni scambiabili della stessa distribuzione, estratti dalla
    # stessa procedura, quindi la difficolta' intrinseca e' bilanciata in
    # aspettazione e si media via su piu' seed.
    #
    # canary_cfg["swap_assignment"] (default False) scambia i due gruppi.
    # Eseguire lo stesso seed con swap=False e swap=True e' inteso come
    # controllo negativo: se l'AUC resta sopra 0.5 in entrambi i versi il
    # segnale e' appartenenza, se si ribalta sotto 0.5 e' difficolta'
    # intrinseca.
    #
    # ATTENZIONE (2026-09-16, Sprint 10zz+113) — questo controllo e' valido
    # SOLO se n_templates == n_nonmember_templates. Con dimensioni diverse
    # (es. 5 e 20, la configurazione storica) non c'e' alcuno scambio di
    # ruoli: swap=False prende member=drawn[:5] e swap=True prende
    # member=drawn[20:25], cioe' due insiemi di membri DISGIUNTI, mentre
    # 15 dei 25 template restano non-membri in entrambi i bracci. I due run
    # sono quindi due esperimenti su record diversi, non lo stesso
    # esperimento a ruoli invertiti, e un esito discordante fra i due NON
    # dimostra che il segnale sia difficolta' intrinseca. Il commento
    # precedente ("il controllo decisivo") era falso per quel caso ed e'
    # stato corretto qui. La guardia sotto avvisa quando la condizione non
    # e' soddisfatta.
    #
    # canary_cfg["nonmember_source"]:
    #   "paired_split" (default) — comportamento corretto descritto sopra
    #   "holdout"                — comportamento precedente, conservato
    #                              solo per riprodurre i run storici
    # ------------------------------------------------------------------
    nonmember_source = str(canary_cfg.get("nonmember_source", "paired_split"))
    swap_assignment  = bool(canary_cfg.get("swap_assignment", False))

    # Guardia (2026-09-16, Sprint 10zz+113): swap_assignment ha senso come
    # controllo negativo SOLO a gruppi bilanciati — vedi il commento esteso
    # sopra. Non solleviamo un'eccezione (spezzerebbe i config storici che
    # hanno gia' prodotto risultati citati nel README) ma l'avviso deve
    # comparire nel log del run, non solo nella documentazione, perche' e'
    # esattamente il punto su cui l'interpretazione del risultato si rompe.
    if swap_assignment and n_templates != n_nonmember_templates:
        logger.warning(
            "[CANARY] swap_assignment=True con n_templates=%d != "
            "n_nonmember_templates=%d: NON e' uno scambio di ruoli. I due "
            "bracci useranno insiemi di membri DISGIUNTI e condivideranno "
            "%d template non-membro, quindi un esito discordante fra "
            "swap=False e swap=True NON dimostra difficolta' intrinseca. "
            "Per un controllo negativo valido servono gruppi di pari "
            "dimensione (n_templates == n_nonmember_templates).",
            n_templates, n_nonmember_templates,
            max(0, n_nonmember_templates - n_templates),
        )

    if nonmember_source == "paired_split":
        n_needed = n_templates + n_nonmember_templates
        if len(site_train_sessions) < n_needed:
            logger.warning(
                f"[CANARY] site={site} non ha abbastanza sessioni di training per "
                f"{n_templates} template membro + {n_nonmember_templates} gemelli "
                f"non-membro estratti dallo stesso pool (train={len(site_train_sessions)}, "
                f"servono {n_needed}) — canary NON iniettati, run prosegue come se "
                "cfg['canary']['enabled'] fosse False."
            )
            return train_sessions, holdout_sessions

        drawn = rng.sample(site_train_sessions, n_needed)
        rng.shuffle(drawn)
        if swap_assignment:
            nonmember_templates = drawn[:n_nonmember_templates]
            member_templates    = drawn[n_nonmember_templates:]
        else:
            member_templates    = drawn[:n_templates]
            nonmember_templates = drawn[n_templates:]

        logger.info(
            f"[CANARY] pool unificato: {n_needed} template estratti da "
            f"site_train_sessions({site}), assegnazione casuale "
            f"{len(member_templates)} membro / {len(nonmember_templates)} non-membro"
            + (" [SWAP ATTIVO]" if swap_assignment else "")
        )
    else:
        if len(site_train_sessions) < n_templates or len(site_holdout_sessions) < n_nonmember_templates:
            logger.warning(
                f"[CANARY] site={site} non ha abbastanza sessioni per {n_templates} "
                f"template membro / {n_nonmember_templates} gemelli non-membro "
                f"(train={len(site_train_sessions)}, holdout={len(site_holdout_sessions)}) "
                "— canary NON iniettati, run prosegue come se cfg['canary']['enabled'] fosse False."
            )
            return train_sessions, holdout_sessions

        logger.warning(
            "[CANARY] nonmember_source='holdout' — modalita' LEGACY, i due lati del "
            "controllo positivo escono da pool diversi e membership resta confusa con "
            "difficolta' di ricostruzione. Usare solo per riprodurre run storici."
        )
        member_templates    = rng.sample(site_train_sessions, n_templates)
        nonmember_templates = rng.sample(site_holdout_sessions, n_nonmember_templates)

    injected_train   = list(train_sessions)
    injected_holdout = list(holdout_sessions)

    for i, template in enumerate(member_templates):
        group = f"canary_m{i}"

        # Fix (2026-09-15, Sprint 10zz+96, task #155/#121 — root cause
        # diagnosticato leggendo il codice dopo che l'utente ha rilanciato
        # Blocker 2 e ottenuto shadow_canary_auc_roc invertito: 0.26/0.32/0.32,
        # sotto 0.5, non spiegabile da campione piccolo). Fino a questo fix,
        # `template` restava nel pool SENZA tag — la stessa identica sessione
        # reale di cui sotto vengono inseriti n_duplicates cloni taggati.
        # `run_shadow()` sposta le sessioni `_canary_role=="member"` da
        # shadow_train a eval_members PRIMA di addestrare lo shadow model
        # (Sprint 10zz+93) — ma quella guardia non vedeva l'originale, perché
        # non portava il tag. Se l'originale finiva per caso in shadow_train
        # (50% di probabilità), lo shadow model si allenava direttamente sullo
        # stesso vettore di feature dei 30 duplicati canary, arrivando a una
        # loss bassa quanto o più bassa di quella del target — invertendo lo
        # score calibrato (shadow_loss - target_loss) per quel gruppo.
        # `_sample_preserving_canary_groups()` (LiRA, Sprint 10zz+16) aveva lo
        # stesso buco per lo stesso motivo, solo diluito su n_shadow=8 modelli
        # invece che su 1 solo, quindi meno visibile nei numeri LiRA.
        # Fix: sostituire l'occorrenza originale dentro injected_train con una
        # COPIA taggata (stesso _canary_group/_canary_role dei cloni), invece
        # di mutare l'oggetto condiviso con train_sessions (che potrebbe
        # essere riusato altrove dal chiamante). Il gruppo diventa così
        # genuinamente atomico: n_duplicates + 1 occorrenze identiche, tutte
        # taggate, non n_duplicates+1 di cui una invisibile alle guardie
        # esistenti. Zero impatto se canary è disattivato (default).
        for idx, s in enumerate(injected_train):
            if s is template:
                injected_train[idx] = dict(template, _canary_group=group, _canary_role="member")
                break

        for _ in range(n_duplicates):
            clone = dict(template)
            clone["_canary_group"] = group
            clone["_canary_role"]  = "member"
            injected_train.append(clone)

    for j, template in enumerate(nonmember_templates):
        group = f"canary_n{j}"
        # Prefisso "canary_n" (non "canary_m") — deliberatamente NON
        # accoppiato 1:1 ai gruppi membro sopra: canary_auc_roc in run_lira()
        # aggrega tutti i membri taggati contro tutti i non-membri taggati,
        # non richiede corrispondenza di gruppo per indice.

        # Fix (2026-09-15, Sprint 10zz+108 — stesso bug e stessa correzione
        # del lato membro sopra, Sprint 10zz+96, mai portato qui: trovato da
        # una review indipendente richiesta esplicitamente dall'utente prima
        # di rilanciare qualunque campagna canary). Fino a questo fix,
        # `template` restava in injected_holdout SENZA tag — la stessa
        # sessione reale di cui sotto viene aggiunto UN SOLO clone taggato
        # (qui non ci sono duplicati, n_nonmember_templates è indipendente da
        # n_duplicates). _sample_preserving_canary_groups() tratta le
        # sessioni non taggate come unità indipendenti, quindi l'originale
        # non taggato e il suo clone taggato potevano finire l'uno IN e
        # l'altro OUT per lo stesso shadow model — stessa classe di
        # contaminazione già diagnosticata e corretta lato membro (shadow che
        # si allena sullo stesso vettore di feature del gemello canary
        # "out"), qui sul lato non-membro. Fix identico: sostituire
        # l'occorrenza originale in injected_holdout con una copia taggata
        # invece di mutare l'oggetto condiviso con holdout_sessions.
        # Sotto nonmember_source="paired_split" il template e' stato estratto
        # da site_train_sessions, quindi vive in injected_train e NON in
        # injected_holdout: va RIMOSSO dal training (altrimenti non sarebbe
        # un non-membro) e aggiunto all'holdout come copia taggata. Sotto
        # "holdout" (legacy) vive gia' in injected_holdout e si applica la
        # sostituzione in loco di sempre.
        tagged_nm = dict(template, _canary_group=group, _canary_role="nonmember")
        if nonmember_source == "paired_split":
            for idx, s in enumerate(injected_train):
                if s is template:
                    del injected_train[idx]
                    break
            injected_holdout.append(tagged_nm)
        else:
            for idx, s in enumerate(injected_holdout):
                if s is template:
                    injected_holdout[idx] = tagged_nm
                    break

        clone = dict(template)
        clone["_canary_group"] = group
        clone["_canary_role"]  = "nonmember"
        injected_holdout.append(clone)

    logger.info(
        f"[CANARY] Iniettati {n_templates} template × {n_duplicates} duplicati "
        f"({n_templates * n_duplicates} record membro nuovi, + {n_templates} occorrenze "
        f"originali ora ri-taggate, Sprint 10zz+96 — totale {n_templates * (n_duplicates + 1)} "
        f"record taggati _canary_role='member') nel training di '{site}', "
        f"+ {n_nonmember_templates} gemelli non-membro nell'holdout — positive control "
        "(Sprint 10vv, vedi docs/TestRoadmap_DSN2027.md)."
    )
    return injected_train, injected_holdout


# ── Sampling shadow universe rispettando i gruppi canary (Sprint 10zz+16) ───────

def _sample_preserving_canary_groups(
    rng: random.Random,
    pool: list[dict[str, Any]],
    n: int,
) -> list[dict[str, Any]]:
    """
    Come rng.sample(pool, n), ma se `pool` contiene sessioni canary (taggate
    `_canary_group`, vedi inject_canaries()) tratta ogni gruppo come
    un'unità atomica indivisibile — o TUTTI i suoi duplicati finiscono nel
    campione IN di uno shadow, o NESSUNO.

    Fix (Sprint 10zz+16, 2026-09-02) alla contaminazione degli shadow
    confermata in Sprint 10zz+15: prima di questo fix, i 30 duplicati di un
    canary membro (`inject_canaries()`, gruppo "canary_m{i}") erano oggetti
    Python indipendenti campionati singolarmente da `rng.sample()`. Per un
    dato duplicato x, uno shadow "OUT" per x (x stesso non campionato)
    campionava comunque, quasi certamente, alcuni dei suoi ~29 gemelli
    identici (stesso valore di ogni feature) — lo shadow "OUT" imparava
    quindi il pattern di x tramite i gemelli, ottenendo una loss bassa su x
    tanto quanto uno shadow "IN" vero. La separazione IN/OUT su cui si basa
    la calibrazione Gaussiana di LiRA collassava per costruzione (dimostrato
    matematicamente, non solo osservato: con 149 gemelli e n_in≈metà
    universo, P(tutti esclusi) è trascurabile), non per un bug numerico.

    ZERO impatto quando nessuna sessione in `pool` ha `_canary_group` (ogni
    run reale/pubblicato — canary disattivato di default): in quel caso
    ogni "gruppo" è un singleton da 1 elemento e la funzione richiama
    direttamente `rng.sample(pool, n)` — stessa sequenza di estrazioni
    casuali, stesso risultato, bit-per-bit, di prima di questo fix.

    La dimensione del campione risultante è approssimata a `n` (non esatta)
    quando esistono gruppi con più di un elemento — accettabile per un
    diagnostico opt-in (canary positive control), mai rilevante per LiRA
    reale (dove ogni pool è già fatto solo di singleton per costruzione).
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    units: list[list[dict[str, Any]]] = []
    for s in pool:
        g = s.get("_canary_group")
        if g is None:
            units.append([s])
        else:
            groups.setdefault(g, []).append(s)
    for g_sessions in groups.values():
        units.append(g_sessions)

    if all(len(u) == 1 for u in units):
        return rng.sample(pool, n)

    shuffled_units = units[:]
    rng.shuffle(shuffled_units)
    sampled: list[dict[str, Any]] = []
    for unit in shuffled_units:
        if len(sampled) >= n:
            break
        sampled.extend(unit)
    return sampled


# ── Feature Normalization ──────────────────────────────────────────────────────

def compute_feature_stats(
    sessions: list[dict[str, Any]],
    features: list[str],
) -> dict[str, tuple[float, float]]:
    """
    Calcola min e max per ogni feature dalle sessioni di training.
    Chiamare SOLO su train_sessions per evitare data leakage dal hold-out.
    """
    stats: dict[str, tuple[float, float]] = {}
    for feat in features:
        vals = []
        for s in sessions:
            raw = s.get(feat)
            if raw is None:
                continue
            try:
                vals.append(float(raw))
            except (ValueError, TypeError):
                continue  # salta valori non numerici (corrotti o stringhe errate)
        if not vals:
            stats[feat] = (0.0, 1.0)
            continue
        fmin, fmax = min(vals), max(vals)
        stats[feat] = (fmin, fmax if fmax != fmin else fmin + 1.0)
    return stats


def normalize_sessions(
    sessions: list[dict[str, Any]],
    stats: dict[str, tuple[float, float]],
    features: list[str],
) -> list[dict[str, Any]]:
    """
    Applica min-max scaling [0,1] alle feature continue.
    stats deve provenire da compute_feature_stats(train_sessions, ...).
    """
    normalized = []
    for s in sessions:
        s = dict(s)  # shallow copy — non modificare l'originale
        for feat in features:
            val = s.get(feat)
            if val is None:
                continue
            fmin, fmax = stats[feat]
            s[feat] = (float(val) - fmin) / (fmax - fmin)
        normalized.append(s)
    return normalized

# ── FL Experiment ──────────────────────────────────────────────────────────────

# ── Ablazioni del meccanismo DP per client (2026-10-05, segnalazione 38) ────────
# "full" e' la DP per client di sempre (taglio del delta + rumore sigma per
# client). "clip-only" taglia senza rumore, "noise-only" aggiunge lo stesso
# rumore (sigma tarato su max_grad_norm) senza tagliare: servono a separare il
# costo del taglio da quello del rumore (linea per il paper DSN, punto 2 della
# coda). Nessuna delle due ablazioni da' una garanzia DP. Valgono solo per il
# meccanismo per client (dp-fedavg, local), non per central ne' con --no-dp.
DP_ABLATIONS = ("full", "clip-only", "noise-only")


def _check_dp_ablation(dp_ablation: str, no_dp: bool, dp_mode: str) -> None:
    """Solleva ValueError se l'ablazione non e' valida o non e' combinabile."""
    if dp_ablation not in DP_ABLATIONS:
        raise ValueError(f"dp_ablation non valida: {dp_ablation!r} (attese: {DP_ABLATIONS})")
    if dp_ablation != "full" and (no_dp or dp_mode not in ("dp-fedavg", "local")):
        raise ValueError(
            f"dp_ablation={dp_ablation!r} vale solo con la DP per client "
            f"(dp_mode dp-fedavg o local) e senza --no-dp; ricevuto no_dp={no_dp}, "
            f"dp_mode={dp_mode!r}"
        )


def _privatize_per_client(
    gm: "GradientManager",
    update: Any,
    dp_ablation: str = "full",
    weight_keys: list[str] | None = None,
    reference_weights: list[Any] | None = None,
) -> Any:
    """Meccanismo per client (dp-fedavg/local) secondo l'ablazione scelta."""
    if dp_ablation == "clip-only":
        return gm.clip_no_noise(update, weight_keys=weight_keys, reference_weights=reference_weights)
    if dp_ablation == "noise-only":
        return gm.noise_no_clip(update, weight_keys=weight_keys)
    return gm.privatize(update, weight_keys=weight_keys, reference_weights=reference_weights)


def run_fl_rounds(
    cfg: dict,
    sessions: list[dict[str, Any]],
    no_dp: bool = False,
    dp_mode: str = "dp-fedavg",
    cluster_sessions: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[int, dict[str, Any]]:
    """
    Esegue FL rounds via ML Plane.
    Ogni round: train locale → [DP opzionale] → FedAvg → global model.
    Restituisce gradient history per round.

    Args:
        cfg:     configurazione esperimento
        sessions: sessioni di training (usato SOLO come fallback se
                 cluster_sessions non è fornito — vedi sotto).
        cluster_sessions: sessioni GIÀ raggruppate per cluster/sito reale
                 (2026-07-22, vedi group_sessions_by_site()) — {cluster_id:
                 [sessioni]}. Se fornito, `sessions` viene ignorato e i cluster
                 sono esattamente quelli passati (es. 3 siti reali: caltech/
                 jpl/office1, o 5 per lo sweep IDS/Byzantine — vedi
                 inject_synthetic_clients()). Se None (default, retrocompatibile
                 coi test esistenti che usano dati sintetici senza site_id
                 reale), affetta `sessions` in 4 parti uguali fittizie — stesso
                 comportamento storico pre-2026-07-22.
        no_dp:   se True, salta il rumore DP (σ=0) — usato per baseline experiment.
                 Permette di distinguere:
                   Scenario A: DP funziona → AUC > 0.5 senza DP, ≈0.5 con DP
                   Scenario B: modello non memorizza → AUC ≈ 0.5 in entrambi i casi
        dp_mode: quale placement DP usare quando no_dp=False (2026-07-22, vedi
                 docs/CaseStudies.md §2.4.3 per la tassonomia completa):
                   "dp-fedavg" (default, comportamento storico) — il server
                     riceve l'update raw di ogni client e lo clippa+rumorizza
                     PRIMA di aggregarlo. Questo placement (rumore per-client
                     prima dell'aggregazione) è una variante più restrittiva e
                     non-standard, non descritta letteralmente nell'Algoritmo 1
                     di McMahan et al. 2018 — vedi "central" sotto per il mode
                     a cui quel paper corrisponde davvero. Il server/IDS vede
                     transitoriamente l'update raw (usato oggi da run_ids()).
                   "central" [McMahan et al. 2018 — meccanismo esatto del loro
                     Algoritmo 1 (DP-FedAvg): clip lato client, un solo draw di
                     rumore lato server sull'aggregato] — ogni client CLIPPA il
                     proprio update (bound la sensitività) ma NON lo rumorizza;
                     il server (trusted) aggrega gli update puliti-ma-clippati
                     con FedAvg, poi aggiunge UN SOLO rumore Gaussiano
                     all'aggregato (scalato 1/n_participants). I singoli update
                     non sono mai rumorizzati: un attacco sul singolo update
                     (LiRA) non dovrebbe mostrare alcuna soppressione — è il
                     risultato atteso, non un bug.
                   "local" — stesso meccanismo per-client di "dp-fedavg", ma il
                     server/IDS non deve MAI vedere l'update raw, nemmeno
                     transitoriamente: run_ids() userà `updates` (rumorizzati)
                     invece di `raw_updates` (che in questa modalità non viene
                     salvato in fl_results — vedi sotto).
                 Ignorato se no_dp=True (nessun rumore in nessun caso).
    """
    exp_cfg  = cfg["experiment"]
    ml_cfg   = cfg["ml"]
    fl_rounds = exp_cfg["fl_rounds"]
    _dp_ablation = exp_cfg.get("dp_ablation", "full")
    _check_dp_ablation(_dp_ablation, no_dp=no_dp, dp_mode=dp_mode)
    if _dp_ablation != "full":
        logger.warning(
            f"[ABLAZIONE DP] dp_ablation={_dp_ablation}: "
            + ("taglio del delta a C, NESSUN rumore" if _dp_ablation == "clip-only"
               else "rumore sigma tarato su C, NESSUN taglio")
            + " — nessuna garanzia DP, solo per separare taglio e rumore."
        )

    if cluster_sessions is not None:
        # Sessioni già raggruppate per sito/cluster reale (2026-07-22) — vedi
        # group_sessions_by_site()/inject_synthetic_clients(). `sessions` (il
        # parametro posizionale) viene ignorato in questo ramo.
        cluster_ids = list(cluster_sessions.keys())
    else:
        # Fallback storico (pre-2026-07-22, retrocompatibile coi test esistenti
        # che usano dati sintetici senza site_id reale): affetta `sessions` in
        # 4 parti uguali fittizie, come sempre fatto finora.
        cluster_ids = ["highway", "urban", "residential", "corporate"]
        cluster_size = max(1, len(sessions) // len(cluster_ids))
        cluster_sessions = {}
        for i, cid in enumerate(cluster_ids):
            start = i * cluster_size
            end   = None if i == len(cluster_ids) - 1 else start + cluster_size
            cluster_sessions[cid] = sessions[start:end]

    # Inizializza trainer per ogni cluster
    #
    # LIMITE NOTO, NON RISOLTO (osservazione review indipendente round 4,
    # 2026-07-24): a differenza di run_lira()/run_shadow() (che
    # chiamano torch.manual_seed() esplicitamente subito prima di ogni
    # Autoencoder(...)), i 3 trainer reali qui sotto vengono costruiti uno
    # dopo l'altro senza un manual_seed() dedicato per ciascuno — l'init dei
    # pesi dipende dalla sequenza corrente del generatore RNG globale (seedato
    # una volta sola in main(), vedi torch.manual_seed(seed) più sopra nel
    # flusso). Funziona correttamente OGGI (verificato: i pesi del round 1
    # sono bit-identici tra experiment_20260724_111109.json [ε=1.0] e
    # experiment_20260724_144952.json [ε=0.1], stesso seed=42, stesso ordine
    # di operazioni prima di questo punto) ma è fragile: qualunque futura
    # modifica che inserisca un draw casuale in più prima di questo loop
    # desincronizzerebbe silenziosamente i pesi iniziali dei 3 siti, senza
    # sollevare errori. Non corretto qui (aggiungere un manual_seed() per
    # cluster cambierebbe i pesi iniziali rispetto a OGNI esperimento già
    # pubblicato, invalidando silenziosamente la comparabilità storica, senza
    # possibilità di verificarne l'effetto per esecuzione reale in questo
    # sandbox) — lasciato come nota per un refactor deliberato, non un
    # blind-fix.
    trainers: dict[str, AutoencoderTrainer] = {}
    for cid in cluster_ids:
        # Propaga il seed sperimentale nella config ml così AutoencoderTrainer
        # lo usa per il DataLoader generator → shuffle deterministico per seed.
        trainer_cfg = {**ml_cfg, "seed": exp_cfg.get("seed", 42)}
        trainers[cid] = AutoencoderTrainer(
            config=trainer_cfg,
            node_id=f"{cid}-01",
            cluster_id=cid,
        )
        logger.info(f"Cluster {cid}: {len(cluster_sessions[cid])} sessioni")

    # Segnalazione 48 (2026-09-24): i trainer qui sopra nascono con
    # inizializzazioni DIVERSE (estrazioni successive dal generatore globale),
    # quindi nel round 1 FedAvg media tre reti indipendenti, a differenza del
    # protocollo standard in cui il server distribuisce un unico modello
    # iniziale. Nei JSON di nodp-sweep2 il modello globale dopo il round 1 ha
    # loss sull'holdout 0.057-0.065 contro 0.0012 di loss locale; al round 3 e'
    # recuperato. Con ml.common_init: true tutti i client partono dai pesi del
    # primo trainer. Default False: il comportamento di tutte le run gia'
    # prodotte resta identico e confrontabile. _global_weights torna a None
    # perche' il termine prossimale FedProx resti attivo solo dal round 2, come
    # nelle run esistenti: cosi' la cella di controllo differisce SOLO per
    # l'inizializzazione (nel FedProx standard sarebbe attivo anche al round 1,
    # verso il modello iniziale). Nessuna estrazione casuale in piu': il resto
    # della run consuma il generatore come prima.
    if ml_cfg.get("common_init", False) and cluster_ids:
        _w0 = trainers[cluster_ids[0]].get_weights()
        for cid in cluster_ids[1:]:
            trainers[cid].set_weights(_w0)
        for _t in trainers.values():
            _t._global_weights = None
        logger.info(
            f"[COMMON INIT] {len(cluster_ids)} client inizializzati con i pesi di "
            f"{cluster_ids[0]}-01 (ml.common_init: true, segnalazione 48)"
        )

    # DP a livello di soggetto (2026-10-13, src/ml/subject_dp.py): una cella
    # per volta, come la DP per record. I soggetti presenti in piu' client
    # ricevono in ogni client un peso w = sessioni nel client / totali, cosi'
    # che la somma dei loro contributi tagliati sia al piu' C.
    _sdp = _subject_dp_config(ml_cfg)
    if _sdp["enabled"]:
        if not no_dp:
            raise ValueError(
                "ml.subject_dp attivo senza --no-dp: la DP per soggetto e quella "
                "per client non si combinano in questa pipeline (una cella per volta)."
            )
        _sint = [c for c in cluster_ids if str(c).startswith("synthetic")]
        if _sint:
            raise ValueError(
                f"ml.subject_dp con client sintetici {_sint}: le loro sessioni sono "
                "copie di quelle reali, quindi lo stesso soggetto comparirebbe in "
                "piu' client senza pesi ne' composizione. Disattivare lo sweep IDS."
            )
        _sfeat = list(trainers[cluster_ids[0]].CONTINUOUS_FEATURES) if cluster_ids else None
        _sw = _subject_cross_client_weights(cluster_sessions, str(_sdp["subject_key"]), _sfeat)
        _sstats_attese = _subject_stats(cluster_sessions, str(_sdp["subject_key"]), _sfeat)
        for cid in cluster_ids:
            trainers[cid].set_subject_weights(_sw.get(cid, {}))
        _n_multi = len({s for per in _sw.values() for s in per})
        logger.info(
            f"[SUBJECT-DP] chiave={_sdp['subject_key']!r}, sigma={_sdp['noise_multiplier']}, "
            f"C={_sdp['max_grad_norm']}, soggetti in piu' client: {_n_multi}"
        )

    gm = GradientManager({
        "epsilon":       exp_cfg["epsilon"],
        "delta":         exp_cfg["delta"],
        "max_grad_norm": exp_cfg["max_grad_norm"],
    })

    agg = FedAvgAggregator({"min_participants": len(cluster_ids)})

    # ML Plane (2026-07-22, richiesta esplicita — vedi README "Relation to
    # Prior Work" e src/ml/ml_plane.py per il contesto completo): PRIMA di
    # questo fix, AutoencoderTrainer/GradientManager/FedAvgAggregator
    # emettevano già eventi ML Plane reali (`emit_event()`), ma nessuno
    # chiamava mai `subscribe()` nella pipeline — gli eventi finivano nel
    # vuoto e `results[round_num]` sotto veniva costruito leggendo le
    # variabili Python locali `raw_updates`/`round_updates`/`aggregated`
    # direttamente, mai attraverso il ML Plane. Ora un singolo `MLPlane`
    # collega tutti e tre i componenti (`wire()`) a un `FLArtifactCollector`
    # reale, che raccoglie gli stessi oggetti (stessa identità, non copie)
    # nel punto in cui il paper QRS 2026 li colloca: "client updates are
    # temporarily available... before the execution of FedAvg". Da qui in
    # poi, `results[round_num]` viene popolato leggendo dal collector, non
    # dalle variabili locali — il ML Plane è quindi il meccanismo realmente
    # usato per raccogliere il flusso di artefatti FL, non un'osservazione
    # parallela mai consultata.
    mlplane   = MLPlane()
    collector = FLArtifactCollector()
    mlplane.subscribe(collector)
    mlplane.wire(*trainers.values(), gm, agg)

    results: dict[int, dict[str, Any]] = {}

    # ── Byzantine attack config ────────────────────────────────────────────────
    # Legge la sezione byzantine_attack dal config. Se assente o disabled, nessun attacco.
    _byz_cfg      = cfg.get("byzantine_attack", {})
    _byz_enabled  = _byz_cfg.get("enabled", False)
    # Default aggiornato 2026-07-22 (3 siti reali + 2 client sintetici per lo
    # sweep IDS): "highway" non esiste più come cluster_id — l'attaccante di
    # default è ora uno dei client sintetici (mai un sito reale, per non
    # implicare che un sito reale specifico sia "malevolo" nella narrazione).
    _byz_node     = _byz_cfg.get("byzantine_node", "synthetic_1")   # cluster attaccante
    _byz_type     = _byz_cfg.get("attack_type", "gradient_scaling")
    _byz_scale    = float(_byz_cfg.get("scale_factor", 10.0))

    if _byz_enabled and _byz_node not in cluster_ids:
        logger.error(
            f"[BYZANTINE ATTACK] byzantine_node={_byz_node!r} non è tra i cluster "
            f"attivi {cluster_ids} — l'attacco NON verrà applicato a nessun client "
            "(nessun errore verrà sollevato più avanti, il confronto cid==_byz_node "
            "semplicemente non scatterà mai mai). Verifica byzantine_node nel config."
        )

    if _byz_enabled:
        logger.warning(
            f"[BYZANTINE ATTACK] abilitato — nodo={_byz_node}, "
            f"tipo={_byz_type}, scale={_byz_scale}"
        )

    if no_dp:
        logger.warning(
            "[NO-DP BASELINE] Rumore Differential Privacy DISABILITATO (σ=0). "
            "Il modello si addestra senza privacy noise. "
            "Confronta AUC con esperimento DP per disambiguare: "
            "AUC>0.5 → Scenario A (DP sopprime MIA); "
            "AUC≈0.5 → Scenario B (modello non memorizza)."
        )

    # Baseline per IDS al round 1: pesi del modello inizializzato (prima del training).
    # Consente di calcolare il delta round 1 = post_training - init_model invece di
    # usare pesi assoluti (che causano GRADIENT_EXPLOSION falso per L2 >> max_grad_norm).
    _init_weights = trainers[cluster_ids[0]].get_weights() if cluster_ids else None
    results[0] = {"raw_global_weights": _init_weights}

    for round_num in range(1, fl_rounds + 1):
        logger.info(f"=== FL Round {round_num}/{fl_rounds} ===")
        _delta_norms: dict[str, float] = {}
        # Attaccante che riscala l'update (2026-10-09, Sprint 10zz+170): modello
        # ricevuto da ogni client a inizio round, riferimento di
        # ref + s*(update osservato - ref) in run_lira(). Solo in memoria, non
        # va nel JSON; get_weights() restituisce copie.
        _pre_round_per_client: dict[str, Any] = {}

        for cid, trainer in trainers.items():
            # Fix 2026-07-22 (review B1): cattura i pesi del modello PRIMA del
            # training locale di questo round — sono il "modello ricevuto" da
            # usare come riferimento per il clipping del DELTA (non del vettore
            # assoluto) in privatize()/clip_only() più sotto. Per il round 1
            # coincide con l'init casuale del trainer (nessun modello globale
            # ancora applicato); per i round successivi coincide col modello
            # applicato da apply_global_model() alla fine del round precedente.
            pre_round_weights = trainer.get_weights()
            _pre_round_per_client[cid] = pre_round_weights

            # Training locale
            update = trainer.train_local(cluster_sessions[cid], round_num)

            # DP per soggetto: il numero di soggetti usato dal trainer deve
            # coincidere con quello dell'accountant, altrimenti l'epsilon nel
            # JSON non corrisponde a quello che e' stato eseguito.
            if _sdp["enabled"] and trainer.subject_dp_last is not None:
                _n_tr = trainer.subject_dp_last["n_soggetti"]
                _n_acc = _sstats_attese["per_client_subjects"].get(cid)
                if _n_tr != _n_acc:
                    raise RuntimeError(
                        f"[SUBJECT-DP] {cid}: il trainer ha {_n_tr} soggetti, "
                        f"l'accountant {_n_acc}. Epsilon non affidabile: run fermata."
                    )

            # Norma L2 del delta del client (2026-09-24, solo misura: nessun
            # effetto sul training ne' sulla DP). Serve a scegliere la soglia di
            # clipping C (max_grad_norm) sui valori reali: stessa definizione di
            # GradientManager._clip_weights(), cioe' parametri float esclusi i
            # buffer BatchNorm, rispetto al modello ricevuto a inizio round.
            _nd_sq = 0.0
            for _k, _w, _r in zip(trainer.get_weight_keys(), update.weights or [],
                                  pre_round_weights or []):
                if _k.split(".")[-1] in ("running_mean", "running_var",
                                         "num_batches_tracked"):
                    continue
                _wt = _w if isinstance(_w, torch.Tensor) else torch.tensor(_w)
                _rt = _r if isinstance(_r, torch.Tensor) else torch.tensor(_r)
                if not _wt.is_floating_point():
                    continue
                _nd_sq += float(((_wt.float() - _rt.float()) ** 2).sum())
            _delta_norms[cid] = _nd_sq ** 0.5
            logger.info(
                f"[NORMA DELTA] Round {round_num} {cid}: ||delta||_2 = "
                f"{_delta_norms[cid]:.4f} (C = {exp_cfg['max_grad_norm']})"
            )

            # ── Gradient scaling attack ──────────────────────────────────────
            # Se questo cluster è il nodo Byzantine e l'attacco è abilitato,
            # moltiplica tutti i pesi per scale_factor.
            # Effetto: l'update è geometricamente ~scale_factor× più distante
            # dagli altri nodi → Krum score >> 1.5 → alert reale.
            # L'attacco avviene sui pesi raw (pre-DP), che è ciò che l'IDS analizza;
            # poi passa anche nella privatizzazione → distorce FedAvg.
            if _byz_enabled and cid == _byz_node and _byz_type == "gradient_scaling":
                scaled_weights = [
                    (w if isinstance(w, torch.Tensor) else torch.tensor(float(w))) * _byz_scale
                    for w in (update.weights or [])
                ]
                from ml.base_ml import GradientUpdate as _GU
                update = _GU(
                    node_id=update.node_id,
                    cluster_id=update.cluster_id,
                    round_num=update.round_num,
                    weights=scaled_weights,
                    gradients=update.gradients,
                    loss=update.loss,
                    n_samples=update.n_samples,
                    metadata={**update.metadata, "byzantine_attack": True,
                               "scale_factor": _byz_scale},
                )
                logger.warning(
                    f"[BYZANTINE] Round {round_num}: {cid} — "
                    f"gradient scaling ×{_byz_scale} applicato"
                )
                # Ri-emissione ML Plane (2026-07-22, wiring reale — vedi
                # src/ml/ml_plane.py): train_local() ha già emesso un evento
                # purdue_level=1 con l'update ORIGINALE (pre-scaling) qualche
                # riga fa. Senza questa ri-emissione, FLArtifactCollector
                # continuerebbe a restituire da raw_updates() il peso non
                # scalato per questo nodo — sbagliato, perché ciò che
                # "attraversa davvero il confine" verso l'aggregatore (e verso
                # l'IDS) è la versione scalata. Semantica "ultimo vince" per
                # (round, node_id) nel collector gestisce correttamente questa
                # sovrascrittura.
                trainers[cid].emit_event(MLPlaneEvent(
                    event_type="gradient_upload",
                    purdue_level=1,
                    payload=update,
                    round_num=round_num,
                    metadata={**update.metadata, "byzantine_reemit": True},
                ))

            # Nota (2026-07-22, wiring ML Plane): non serve più appendere
            # `update` a una lista locale — train_local() (e, per il nodo
            # Byzantine, la ri-emissione sopra) ha già emesso l'evento
            # purdue_level=1 attraverso il ML Plane; sarà `collector.raw_updates()`,
            # letto sotto dopo il loop, a fornire la stessa informazione — ma
            # realmente "raccolta" dal ML Plane, non passata a mano.
            # Applica DP — passa le chiavi per escludere buffer BatchNorm dal rumore.
            # Con --no-dp, usa l'update raw direttamente (σ=0, nessun rumore).
            weight_keys = trainer.get_weight_keys()
            if no_dp:
                private_update = update  # baseline: nessuna privacy noise
            elif dp_mode == "central":
                # Central DP (2026-07-22): clip-only per client, NESSUN rumore
                # individuale — il rumore va sull'aggregato (vedi sotto, dopo
                # agg.aggregate()). Un attacco sul singolo update (LiRA) vedrà
                # quindi l'update clippato ma pulito — atteso, non un bug.
                # reference_weights=pre_round_weights (fix 2026-07-22, review
                # B1): clippa il DELTA rispetto al modello ricevuto, non il
                # vettore assoluto — vedi GradientManager._clip_weights().
                private_update = gm.clip_only(
                    update, weight_keys=weight_keys, reference_weights=pre_round_weights
                )
            else:
                # "dp-fedavg" (default) e "local" condividono lo stesso meccanismo
                # per-client (clip+noise prima dell'aggregazione) — la differenza
                # tra i due è SOLO nella visibilità di raw_updates per l'IDS
                # (vedi sotto, dopo il loop dei client).
                # Ablazioni (2026-10-05): "clip-only"/"noise-only" sostituiscono
                # privatize() con uno solo dei due pezzi, vedi _privatize_per_client().
                private_update = _privatize_per_client(
                    gm, update, _dp_ablation,
                    weight_keys=weight_keys, reference_weights=pre_round_weights,
                )
            agg.collect(private_update)
            # Nota (2026-07-22): idem — `private_update` è già stato emesso da
            # gm.privatize()/clip_only() (o è l'update raw stesso, se no_dp)
            # attraverso il ML Plane; niente più lista locale `round_updates`.

        # ── Lettura dal ML Plane (2026-07-22, wiring reale) ──────────────────
        # Questi due elenchi sono ORA la vera fonte di raw_updates/updates per
        # `results[round_num]` sotto — non più le liste locali `raw_updates`/
        # `round_updates` costruite a mano durante il loop qui sopra (rimosse).
        # `collector` (src/ml/ml_plane.py) le ha popolate osservando gli eventi
        # realmente emessi da train_local() (+ ri-emissione Byzantine sopra) e
        # da privatize()/clip_only(), con semantica "ultimo vince" per
        # (round, node_id) — quindi riflettono correttamente l'eventuale
        # scalatura Byzantine senza bisogno di logica speciale qui.
        _collected_raw = collector.raw_updates(round_num)
        _collected_updates = collector.privatized_updates(round_num)

        # Calcola raw_global_weights: media semplice dei pesi pre-DP.
        # Usato da IDS come riferimento per i delta (evita che il rumore DP
        # del global aggregato inquini l'analisi degli update locali).
        raw_global_weights: list[Any] | None = None
        if _collected_raw and _collected_raw[0].weights:
            n_w    = len(_collected_raw[0].weights)
            total  = sum(u.n_samples for u in _collected_raw) or len(_collected_raw)
            raw_global_weights = []
            for i in range(n_w):
                wavg = sum(
                    (u.weights[i] if isinstance(u.weights[i], torch.Tensor)
                     else torch.tensor(float(u.weights[i])))
                    * (u.n_samples / total)
                    for u in _collected_raw
                )
                raw_global_weights.append(wavg)

        # FedAvg
        aggregated = agg.aggregate(round_num)

        if aggregated is None:
            logger.warning(f"Round {round_num} saltato — partecipanti insufficienti")
            continue

        # Central DP (2026-07-22): UN SOLO rumore Gaussiano sull'aggregato pulito,
        # dopo FedAvg — non per client (vedi clip_only() sopra). Il modello
        # rumorizzato è quello effettivamente distribuito ai trainer per il
        # prossimo round E quello salvato in results (usato da LiRA per il
        # warm-start degli shadow — deve vedere lo stesso rumore del target).
        if not no_dp and dp_mode == "central" and aggregated.global_weights:
            _agg_weight_keys = (
                trainers[cluster_ids[0]].get_weight_keys() if cluster_ids else None
            )
            aggregated.global_weights = gm.privatize_aggregate(
                aggregated.global_weights,
                weight_keys=_agg_weight_keys,
                n_participants=aggregated.n_participants or len(cluster_ids),
                # FASE 8 (2026-08-31, fix sensibilità pesata): n_samples reali
                # per client di QUESTO round, popolati da
                # FedAvgAggregator.aggregate() — vedi privatize_aggregate()
                # per il perché (limite Office1, sensitività max_i(n_i/N) non
                # 1/n_participants).
                participant_n_samples=list(
                    aggregated.metadata.get("participant_n_samples", {}).values()
                ),
            )
            # Ri-emissione ML Plane (2026-07-22, wiring reale): agg.aggregate()
            # sopra ha già emesso un evento "aggregation" con l'aggregato
            # PULITO (pre-rumore). privatize_aggregate() modifica `aggregated`
            # in-place aggiungendo il rumore centrale — senza questa
            # ri-emissione, FLArtifactCollector.aggregation(round_num)
            # restituirebbe l'aggregato sbagliato (senza rumore), mentre è
            # quello rumorizzato che viene davvero distribuito ai trainer e
            # salvato in results per il warm-start degli shadow di LiRA.
            agg.emit_event(MLPlaneEvent(
                event_type="aggregation",
                purdue_level=3,
                payload=aggregated,
                round_num=round_num,
            ))

        # Distribuisci modello globale ai trainer
        for trainer in trainers.values():
            trainer.apply_global_model(aggregated)

        loss_str = f"{aggregated.mean_loss:.6f}" if aggregated.mean_loss is not None else "N/A"
        logger.info(f"Round {round_num} — loss globale: {loss_str}")

        # Local DP (2026-07-22): il server/IDS non deve MAI vedere l'update raw,
        # nemmeno transitoriamente — a differenza di dp-fedavg/central, dove un
        # server "honest-but-curious"/trusted vede comunque il raw update prima
        # di clippare+rumorizzare (dp-fedavg) o prima di aggregare (central).
        # Non salviamo raw_updates in questa modalità: run_ids() (vedi sotto,
        # "Preferisci raw_updates... Fallback su updates per retrocompatibilità")
        # userà automaticamente `updates` (già rumorizzati) — modellando
        # correttamente il fatto che sotto local DP non esiste nessun momento in
        # cui il server osserva il valore pulito.
        _store_raw = _collected_raw if not (dp_mode == "local" and not no_dp) else None
        _store_raw_global = (
            raw_global_weights if not (dp_mode == "local" and not no_dp) else None
        )

        results[round_num] = {
            "mean_loss":         aggregated.mean_loss,
            "delta_norm_per_client": dict(_delta_norms),
            "n_participants":    aggregated.n_participants,
            "updates":           _collected_updates,  # privatized — usati da FedMIA — dal ML Plane
            "raw_updates":       _store_raw,          # pre-DP — usati da IDS (None sotto local DP) — dal ML Plane
            "raw_global_weights": _store_raw_global,  # media raw — riferimento IDS (idem)
            "global_weights":    aggregated.global_weights,
            # Sprint 10zz+170: vedi _pre_round_per_client sopra.
            "pre_round_weights_per_client": _pre_round_per_client,
        }

    return results

# ── FedMIA Attack ──────────────────────────────────────────────────────────────

# Feature ACN usate per FedMIA — allineate con AutoencoderTrainer
_MIA_FEATURES = [
    "total_energy_kwh", "max_power_kw", "kwh_requested",
    "minutes_available", "hour_of_day", "duration_hours",
]


# ── TPR@low-FPR (roadmap #4, Sprint 10pp 2026-08-28) ────────────────────────────

# FPR fissi a cui riportare la TPR — 0.1%/1% coprono il regime "attacco reale"
# che conta di più secondo Carlini et al. 2022 (AUC-ROC medio su tutta la curva
# può nascondere un segnale concentrato a FPR bassissimo); 5% è un punto più
# permissivo per confronto. Chiavi arrotondate per essere leggibili nel JSON
# (es. "tpr_at_fpr_0.01" invece di "tpr_at_fpr_0.010000000000000002").
_TPR_AT_FPR_TARGETS = (0.001, 0.01, 0.05)


def _tpr_at_fixed_fpr(
    labels: list[int],
    scores: list[float],
    fpr_targets: tuple[float, ...] = _TPR_AT_FPR_TARGETS,
) -> dict[str, float | None]:
    """
    Calcola TPR a FPR fissi dalle stesse coppie score/label già usate per
    l'AUC-ROC — nessun nuovo esperimento richiesto, solo un'aggregazione
    diversa sui dati già raccolti (vedi docs/TestRoadmap_DSN2027.md #4).

    Motivazione (Carlini et al. 2022, si veda anche docs/ReadingList_DSN2027.md,
    nota 2026-08-14): gli autori di LiRA stessi argomentano che l'AUC-ROC,
    mediata su tutta la curva, è una metrica MIA inadeguata — un attacco reale
    opera a FPR basso (un attaccante che sbaglia troppo spesso su chi NON è
    membro non è utilizzabile), quindi un AUC~0.5 potrebbe comunque nascondere
    un segnale concentrato a FPR bassissimo, o viceversa un AUC>0.5 potrebbe
    essere trainato da FPR alti irrilevanti in pratica. TPR@low-FPR risponde a
    questa domanda direttamente sugli stessi dati.

    Usa interpolazione lineare (np.interp) sulla curva ROC di sklearn — fpr è
    monotona non decrescente per costruzione di roc_curve, quindi interp è
    ben definita anche con valori ripetuti di fpr (thresholds ravvicinate).

    Returns:
        {"tpr_at_fpr_0.001": float|None, "tpr_at_fpr_0.01": float|None,
         "tpr_at_fpr_0.05": float|None} — None se roc_curve non è calcolabile
        (es. un'unica classe presente nel pool, stesso caso già gestito per
        roc_auc_score con un try/except ValueError altrove in questo file).
    """
    from sklearn.metrics import roc_curve

    result: dict[str, float | None] = {f"tpr_at_fpr_{t}": None for t in fpr_targets}
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
    except ValueError:
        return result
    for target in fpr_targets:
        result[f"tpr_at_fpr_{target}"] = round(float(np.interp(target, fpr, tpr)), 6)
    return result


# ── MIA Advantage (task #41, Sprint 10zz+13, 2026-09-02) ────────────────────────

def _advanced_composition_epsilon(
    epsilon_per_round: float,
    delta_per_round: float,
    rounds: int,
    delta_prime: float | None = None,
) -> tuple[float, float]:
    """
    Advanced composition bound (Dwork & Roth 2014, "The Algorithmic
    Foundations of Differential Privacy", Theorem 3.20) — closed-form,
    no external DP-accounting library, no new experimental run required.
    Task #117/#118 (Sprint 10zz+81/83, 2026-09-14): implements "opzione A"
    of the three remediation options discussed with the user for the
    naive-composition gap ("epsilon_cumulative_naive" above) — (A) this
    bound, cheapest, computed retroactively from already-logged
    epsilon/delta/fl_rounds; (B) a closed-form RDP accountant specific to
    the Gaussian mechanism, tighter than (A), NOT implemented; (C) a full
    external DP-accounting library (Opacus/TF Privacy/dp-accounting) with
    true per-sample DP-SGD, most rigorous, left as future work.

    For k-fold adaptive composition of (ε, δ)-DP mechanisms, for any
    δ' > 0, the composition is (ε', kδ + δ')-DP, where:

        ε' = ε · sqrt(2k · ln(1/δ')) + k · ε · (e^ε − 1)

    δ' is a free slack parameter traded against ε': smaller δ' loosens
    ε' (grows as sqrt(ln(1/δ'))) while barely changing δ_tot (still
    dominated by k·δ for the δ this project uses, 1e-5). We default
    δ' = δ_per_round — the same order of magnitude as the per-round δ
    already in the config, an unremarkable choice not tuned to make
    either number look better.

    IMPORTANT — this bound is NOT always tighter than naive composition
    (ε_tot = ε × rounds): advanced composition is a large-k/small-ε
    asymptotic improvement (its dominant term grows as sqrt(k) instead
    of k), so for small k or ε not small it can be looser than the
    trivial ε×k bound. This function does not decide which is tighter —
    see epsilon_cumulative_best_known (= min of the two) where it is
    wired in below.

    Returns:
        (epsilon_prime, delta_total) as floats. rounds<=0 returns
        (0.0, delta_prime or delta_per_round) rather than raising.
    """
    if rounds <= 0:
        return 0.0, float(delta_prime if delta_prime is not None else delta_per_round)
    if delta_prime is None:
        delta_prime = delta_per_round
    epsilon_prime = (
        epsilon_per_round * math.sqrt(2 * rounds * math.log(1.0 / delta_prime))
        + rounds * epsilon_per_round * (math.exp(epsilon_per_round) - 1)
    )
    delta_total = rounds * delta_per_round + delta_prime
    return float(epsilon_prime), float(delta_total)


def _mia_advantage(labels: list[int], scores: list[float]) -> float | None:
    """
    Empirical membership advantage (Yeom, Fredrikson, Jha, "Privacy Risk in
    Machine Learning: Analyzing the Connection to Overfitting," IEEE CSF
    2018): Adv = max_t( TPR(t) - FPR(t) ), il massimo sulla curva ROC — la
    statistica J di Youden. È la definizione standard di "attacker
    advantage" nella letteratura MIA: a differenza dell'AUC (media su ogni
    soglia possibile), è ancorata a UNA soglia — la migliore che
    l'attaccante potrebbe scegliere — coerente con l'argomento di Carlini
    et al. 2022 che un attaccante realistico opera a una soglia fissa, non
    mediata (stesso principio già applicato a _tpr_at_fixed_fpr() sopra).

    Stesse coppie score/label già usate per AUC-ROC/TPR@low-FPR — nessun
    nuovo esperimento richiesto, solo un'aggregazione diversa sui dati già
    raccolti. NOTA: a differenza di PES v1 (calcolabile retroattivamente da
    mean_lira_auc_roc già salvato in ogni JSON esistente), Adv richiede la
    curva ROC completa — mai salvata nei JSON storici (solo gli aggregati
    auc_roc/tpr_at_fpr_* lo sono) — quindi è disponibile SOLO per run
    eseguiti dopo questa modifica, non retroattivamente sulla campagna
    5-seed×8-config già completata (Sprint 10tt) né su qualunque run
    precedente. Vedi scripts/compute_pes.py per la parte retroattiva.

    Returns:
        Adv in [0.0, 1.0], o None se roc_curve non è calcolabile (stesso
        contratto di _tpr_at_fixed_fpr()/roc_auc_score altrove nel file).
    """
    from sklearn.metrics import roc_curve

    try:
        fpr, tpr, _ = roc_curve(labels, scores)
    except ValueError:
        return None
    return round(float(np.max(tpr - fpr)), 6)


# ── MIA Confusion Matrix at Best Threshold (Sprint 10zz+25, 2026-09-03) ─────────

def _mia_confusion_at_best_threshold(
    labels: list[int], scores: list[float]
) -> dict[str, float | int | None]:
    """
    Conteggi assoluti (veri positivi/falsi negativi/falsi positivi/veri
    negativi) dell'attacco alla soglia che massimizza l'Advantage (Youden
    J — stessa soglia di _mia_advantage() sopra), letti come numeri
    assoluti invece che come tasso aggregato.

    Motivazione (chat 2026-09-03, domanda esplicita dell'utente: "calcoliamo
    il numero di veri positivi e falsi negativi dell'attacco?"): AUC-ROC,
    TPR@fixed-FPR e Advantage sono tutte metriche "a tasso" — nessuna
    riporta quanti campioni sono stati effettivamente classificati
    correttamente/erroneamente in cifra assoluta. Utile soprattutto per i
    pool piccoli (es. canary, n_nonmember~19-20 per round), dove un
    conteggio concreto ("rilevati X canary membri su 150, mancati Y") è
    più leggibile e citabile di un tasso nel testo del paper.

    Convenzione soglia: un campione è predetto "membro" se score >=
    threshold — stessa convenzione di sklearn.roc_curve (i thresholds
    restituiti sono già ordinati in modo che scorrerli con >= riproduca
    esattamente le coppie fpr/tpr calcolate). threshold è preso da
    thresholds[argmax(tpr-fpr)], lo stesso indice che _mia_advantage()
    usa per calcolare il massimo — richiede una seconda chiamata a
    roc_curve() (stesso costo, nessun nuovo esperimento) invece di
    condividere l'array con _mia_advantage() per mantenere le due funzioni
    indipendenti e testabili separatamente, a costo trascurabile (roc_curve
    su array di poche migliaia di elementi, già ricalcolato più volte per
    round in questo file).

    Returns:
        {"threshold": float, "advantage": float, "tp": int, "fp": int,
         "tn": int, "fn": int, "n_members": int, "n_nonmembers": int} —
        tutti None se roc_curve non è calcolabile (stesso contratto di
        _mia_advantage()/_tpr_at_fixed_fpr() sopra, es. un'unica classe
        presente nel pool).
    """
    from sklearn.metrics import roc_curve

    none_result: dict[str, float | int | None] = {
        "threshold": None, "advantage": None,
        "tp": None, "fp": None, "tn": None, "fn": None,
        "n_members": None, "n_nonmembers": None,
    }
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores)
    except ValueError:
        return none_result

    best_idx = int(np.argmax(tpr - fpr))
    best_threshold = float(thresholds[best_idx])
    advantage = round(float(tpr[best_idx] - fpr[best_idx]), 6)

    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    predicted_member = scores_arr >= best_threshold

    n_members = int(np.sum(labels_arr == 1))
    n_nonmembers = int(np.sum(labels_arr == 0))
    tp = int(np.sum(predicted_member & (labels_arr == 1)))
    fp = int(np.sum(predicted_member & (labels_arr == 0)))
    fn = n_members - tp
    tn = n_nonmembers - fp

    return {
        "threshold": round(best_threshold, 6),
        "advantage": advantage,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_members": n_members, "n_nonmembers": n_nonmembers,
    }


# ── Curva ROC completa per plotting log-log (task #54, Sprint 10zz+29, 2026-09-03) ──

def _sablayrolles_score(mu_in: float, mu_out: float, target_loss: float) -> float:
    """
    Attacco di Sablayrolles et al. 2019 [56] — task #58, Sprint 10zz+33
    (2026-09-03), su richiesta esplicita dell'utente dopo aver letto Carlini
    et al. 2022 §V-C/Table I: "the most direct influence for LiRA", che
    nonostante sia più vecchio (2019) e usi una soglia NON-parametrica per
    ogni esempio (a differenza del fit Gaussiano di LiRA) batte quasi tutti
    gli altri attacchi pre-LiRA a basso FPR.

    Formula originale del paper: A'(x,y) = ℓ(f(x),y) - τ_{x,y}, con
    τ_{x,y} = (μ_in(x,y)+μ_out(x,y))/2 stimato via shadow models (stessi
    μ_in/μ_out già calcolati da run_lira() per il proprio fit Gaussiano —
    nessuno shadow model aggiuntivo). Qui restituiamo il NEGATIVO di quella
    quantità (τ - loss, non loss - τ) per allinearci alla convenzione di
    segno già usata in questo file per lira_score e per
    lira_debug_raw_mse_auc_roc: punteggio più alto = più probabile membro
    (coerente con Yeom -ℓ(x,y) > τ — loss più bassa della soglia → membro).

    Returns:
        τ_{x,y} - target_loss. Nessun clipping (a differenza di lira_score,
        che è un log-likelihood-ratio non limitato da natura) — questa è una
        differenza lineare su una scala già bounded dalla loss stessa (MSE
        di ricostruzione), non necessita di clip per instabilità numeriche.
    """
    return float(((mu_in + mu_out) / 2.0) - target_loss)


def _lira_log_score(
    in_losses: list[float],
    out_losses: list[float],
    target_loss: float,
    eps: float = 1e-8,
    sigma_floor: float = 0.05,
) -> float:
    """
    Variante ESPLORATIVA di LiRA — task #61, Sprint 10zz+36 (2026-09-03), su
    richiesta esplicita dell'utente dopo il risultato reale del task #57
    (MSE grezza fortemente non-Gaussiana su dati reali: skewness~10-11,
    Jarque-Bera~16-24 milioni contro soglia 5.99 — vedi
    docs/MetricsReference_DSN2027.md §3). Stesso principio del
    log-likelihood-ratio Gaussiano di LiRA (Carlini et al. 2022, Eq. 2),
    applicato a `log(x + eps)` invece che a `x` grezzo — analogo, nel nostro
    dominio, al logit-scaling che Carlini applica alla confidenza (qui non
    applicabile: nessuna probabilità in [0,1] da cui partire per una MSE di
    ricostruzione, log() è il candidato naturale per un valore semi-illimitato
    positivo, stesso ragionamento di check_gaussian_fit.py).

    SEMPLIFICAZIONE DICHIARATA rispetto al fit raw di run_lira() (μ_in/σ_in/
    μ_out/σ_out sopra): quel fit raw incorpora settimane di fix empirici
    (floor scale-adattivo, floor simmetrico, ancoraggio per-cluster di μ_in
    quando mancano osservazioni IN reali, esclusione 8σ) scoperti e
    corretti uno alla volta su run reali. Riprodurre TUTTA quella logica in
    scala logaritmica avrebbe richiesto lo stesso ciclo di scoperta/fix
    (rischiando di introdurre bug nuovi e non ancora scoperti) prima di
    poter rispondere alla domanda specifica posta qui ("il log-transform
    riduce il floor-hit-rate e/o migliora AUC/TPR?"). Questa funzione usa
    invece due semplificazioni esplicite, entrambe conservative (nel senso
    che NON possono gonfiare artificialmente il segnale a favore della
    variante log):
    - fallback quando mancano ≥2 osservazioni IN reali: `μ_in_log =
      μ_out_log` (nessun vantaggio assunto, a differenza dell'ancoraggio
      per-cluster raffinato del fit raw, che stima esplicitamente il gap
      tipico IN/OUT) — un fallback più debole, non più forte, quindi non
      può produrre un AUC artificialmente alto per bias di formula (lo
      stesso tipo di rischio verificato e escluso per il fit raw via
      `lira_debug_matched_formula_auc`, non riprodotto qui).
    - floor fisso `sigma_floor` invece di scale-adattivo: ragionevole in
      scala logaritmica perché il log-transform comprime già la scala
      grezza (che spazia su più ordini di grandezza) in un intervallo
      comparabile tra campioni — un floor scale-adattivo ha meno
      giustificazione qui che nel caso raw.

    Non ancora sottoposta allo stesso rigore (worst-case check §10c, canary
    positive-control §6) del fit raw prima di essere promossa a metrica
    primaria — vedi §3 di docs/MetricsReference_DSN2027.md.

    Returns:
        score = log_p_in_log - log_p_out_log, clippato a ±20 (stesso
        contratto di lira_score in run_lira() — oltre questo range il
        log-likelihood-ratio non ha valore discriminativo pratico
        aggiuntivo). Richiede len(out_losses) >= 2 (contratto del
        chiamante, stesso di run_lira() per il fit raw — non validato qui
        per evitare un doppio controllo ridondante nel call site).
    """
    log_out = [math.log(x + eps) for x in out_losses]
    log_target = math.log(target_loss + eps)

    mu_out_log = float(np.mean(log_out))
    sigma_out_log = max(float(np.std(log_out)), sigma_floor)

    if len(in_losses) >= 2:
        log_in = [math.log(x + eps) for x in in_losses]
        mu_in_log = float(np.mean(log_in))
        sigma_in_log = max(float(np.std(log_in)), sigma_floor)
    else:
        mu_in_log = mu_out_log
        sigma_in_log = sigma_out_log

    log_p_in = (-0.5 * ((log_target - mu_in_log) / sigma_in_log) ** 2) - math.log(sigma_in_log)
    log_p_out = (-0.5 * ((log_target - mu_out_log) / sigma_out_log) ** 2) - math.log(sigma_out_log)
    return float(np.clip(log_p_in - log_p_out, -20.0, 20.0))


def _full_roc_curve(labels: list[int], scores: list[float]) -> dict[str, list[float]] | None:
    """
    Restituisce l'intera curva ROC (fpr, tpr) — non un singolo numero
    aggregato (AUC, §1) né un punto a soglia fissa (TPR@low-FPR,
    _tpr_at_fixed_fpr) né alla soglia ottimale (Advantage/Confusion,
    _mia_advantage/_mia_confusion_at_best_threshold) — pensata per essere
    salvata e poi plottata in scala log-log da scripts/plot_roc_log_scale.py,
    su richiesta esplicita dell'utente dopo un'altra citazione di Carlini et
    al. 2022: "la bontà di un attacco di privacy si misura unicamente
    osservando cosa accade quando il FPR è prossimo allo zero" — un singolo
    numero (anche TPR@0.01 fisso) mostra un solo punto di quella regione, la
    curva completa in scala log-log mostra l'intero comportamento a FPR
    piccolissimo, non solo un campione discreto.

    Stesse coppie label/score già usate per AUC/TPR@low-FPR/Advantage/
    Confusion — nessun nuovo esperimento richiesto, solo un'estrazione
    diversa (l'intero array invece di un aggregato) dagli stessi dati già
    raccolti. Arrotondato a 8 decimali (più di quanto serva per un plot, ma
    sufficiente a non introdurre artefatti visibili in scala log su valori
    piccoli come 1e-4/1e-5).

    Returns:
        {"fpr": [...], "tpr": [...]} (stessa lunghezza, monotoni non
        decrescenti per costruzione di sklearn.roc_curve), o None se
        roc_curve non è calcolabile (stesso contratto delle altre funzioni
        di questa famiglia — es. un'unica classe presente nel pool).
    """
    from sklearn.metrics import roc_curve

    try:
        fpr, tpr, _ = roc_curve(labels, scores)
    except ValueError:
        return None
    return {
        "fpr": [round(float(x), 8) for x in fpr],
        "tpr": [round(float(x), 8) for x in tpr],
    }


def _write_diagnostic_dump(path: str, payload: dict[str, Any]) -> None:
    """
    Scrittura JSON condivisa dai dump diagnostici opt-in di questo file —
    roc_curve_dump_path (Yeom/Shadow/LiRA, task #54) e raw_loss_dump_path
    (solo LiRA, task #57) — stesso pattern del dump per-campione di
    run_lira() (task #50), qui centralizzata perché usata da più punti
    diversi invece che uno solo. Il campo "attack" nel payload distingue il
    tipo di dump per chi legge il file (non usato da questa funzione).

    Fix 2026-09-03 (task #60, Sprint 10zz+35) — bug scoperto da un run reale
    dell'utente (comando --raw-loss-dump raccomandato in
    docs/MetricsReference_DSN2027.md §3, task #57): la directory genitore di
    `path` (es. `experiments/_check_gaussian_fit/`) viene creata SOLO alla
    fine dell'intera pipeline, quando `main()` salva il JSON/Excel finale
    dell'esperimento (`output_dir.mkdir(parents=True, exist_ok=True)`,
    altrove in questo file) — ma questa funzione scrive PRIMA, durante
    `run_registered_attacks()`. Se la directory non esiste ancora (run
    lanciato a mano senza `mkdir -p` preventivo — come nel comando
    raccomandato in §3, che non lo includeva), `open(path, "w")` falla con
    FileNotFoundError, l'eccezione viene loggata e ingoiata da
    `run_registered_attacks()` (try/except per-attacco, § sopra) — il resto
    dell'esperimento completa comunque, ma il dump richiesto va perso in
    silenzio (solo un log ERROR, facile da non notare in un output lungo).
    Successo finora con la campagna Makefile/DUMP_EXTRAS (task #55) solo
    perché quei target creano lo sweep-dir con `mkdir -p` prima di invocare
    python — non una garanzia strutturale di questo file. `Path(path).parent
    .mkdir(parents=True, exist_ok=True)` qui rende il fix strutturale:
    funziona sia da Makefile sia da comando manuale, senza richiedere
    all'utente di ricordarsi `mkdir -p`.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Dump diagnostico scritto: {path}")


def _mia_feature_names(cfg: dict) -> list[str]:
    """
    Come _autoencoder_arch_kwargs() ma per la lista di feature (Sprint 10kk,
    2026-08-28 — escalation feature-entropy del sanity-check LiRA).

    Deve restituire ESATTAMENTE la stessa lista, nello stesso ordine, usata da
    AutoencoderTrainer per costruire il tensore di training (vedi
    src/ml/autoencoder_trainer.py, che legge la stessa chiave
    config['feature_names']) — altrimenti gli score MIA verrebbero calcolati
    su una proiezione diversa dello stesso vettore di pesi, invalidando il
    confronto. Default None → _MIA_FEATURES (le 6 feature storiche), invariato
    per ogni config YAML che non imposta esplicitamente ml.feature_names.
    """
    names = cfg.get("ml", {}).get("feature_names")
    return list(names) if names else _MIA_FEATURES


def run_yeom(
    cfg: dict,
    members: list[dict[str, Any]],
    non_members: list[dict[str, Any]],
    fl_results: dict[int, dict[str, Any]],
    roc_curve_dump_path: str | None = None,
    no_dp: bool = False,
    dp_mode: str = "dp-fedavg",
) -> dict[int, dict[str, Any]]:
    """
    Loss-based Membership Inference Attack per FL con autoencoder.

    Per ogni round FL carica i pesi globali aggregati in un Autoencoder
    locale e misura l'errore di ricostruzione su membri e non-membri.
    Principio (Yeom et al., 2018): il modello FL produce errore basso sui
    campioni visti nel training (membri) e alto sui non visti (non-membri).
    La DP riduce questa gap → AUC-ROC → 0.5 (attacco non migliore del random).

    IMPORTANTE: members deve contenere sessioni effettivamente usate per il
    training FL; non_members deve essere un hold-out set mai visto da nessun
    FL node. Usare lo stesso pool per entrambi invalida la misura di AUC-ROC.

    L'AUC varia per round: round iniziali → modello non converge → AUC ≈ 0.5;
    round finali → modello memorizza → AUC cresce se DP insufficiente.

    Args:
        cfg:         configurazione esperimento
        members:     sessioni usate per FL training (vere member)
        non_members: sessioni hold-out mai viste durante training (vere non-member)
        fl_results:  dict round → {"global_weights": [...], ...}
        roc_curve_dump_path: (task #54, Sprint 10zz+29, 2026-09-03) percorso
            file opzionale — se impostato, scrive un JSON con la curva ROC
            COMPLETA (fpr/tpr, non un aggregato) di ogni round, per il plot
            log-log richiesto dall'utente (scripts/plot_roc_log_scale.py).
            Default None, zero impatto se omesso.

    Returns:
        {round_num: {"auc_roc": float, "member_score_mean": float,
        "non_member_score_mean": float, "yeom_tpr_at_fpr_0.001"/"...0.01"/"...0.05":
        float|None, "advantage": float|None, "confusion": dict|None — questi
        ultimi 5 aggiunti Sprint 10zz+28 (2026-09-03, task #53), stesso
        pattern/motivazione di _tpr_at_fixed_fpr()/_mia_advantage()/
        _mia_confusion_at_best_threshold() già usati in run_lira(): un
        confronto Yeom-vs-Shadow-vs-LiRA basato solo su auc_roc cadrebbe
        nella stessa "fallacia delle medie" di Carlini et al. 2022 che
        TPR@low-FPR/Advantage risolvono già dentro un singolo attacco. Non
        retroattivo sui JSON storici (richiede la curva ROC completa).
        NOTA: i campi tpr_at_fpr_* sono prefissati "yeom_" dal Sprint 10zz+34
        (2026-09-03, task #59) — bug di collisione di chiavi con Shadow/LiRA
        nel merge di run_registered_attacks(), vedi commento lì.}}
    """
    from sklearn.metrics import roc_auc_score

    logger.info(f"Yeom — members: {len(members)}, non-members: {len(non_members)}")

    # Observation surface: global (default, invariato) vs client (Sprint 10zz+94,
    # 2026-09-15) — opt-in richiesto esplicitamente dall'utente per "chiudere il
    # cerchio" tra i tre attacchi: LiRA attacca già gli update per-client
    # (round_data["updates"]/["raw_updates"], selezione dp_mode-aware, Strada B);
    # "client" fa attaccare anche a Yeom quella stessa superficie invece del
    # modello aggregato round_data["global_weights"] (superficie "global",
    # INVARIATA, resta il default). Sotto "client", ogni membro è valutato SOLO
    # dal modello del client del proprio sito (group_sessions_by_site(), già
    # usato altrove per lo stesso scopo) — una rilevazione positiva attribuisce
    # l'appartenenza a un operatore specifico, non più "da qualche parte nella
    # federazione" (vedi §5 del paper). Il pool non-membro resta lo stesso
    # holdout condiviso, valutato con CIASCUN modello client (concatenato nel
    # pool finale) — semplificazione dichiarata: non ancora verificato con un
    # run reale (nessun torch in questo sandbox), vedi TestRoadmap_DSN2027.md.
    _yeom_observation_surface = cfg.get("yeom", {}).get("observation_surface", "global")
    if _yeom_observation_surface not in ("global", "client"):
        raise ValueError(
            f"cfg['yeom']['observation_surface'] non valido: {_yeom_observation_surface!r} "
            "(atteso 'global' o 'client')"
        )
    _members_by_site_yeom = (
        group_sessions_by_site(members) if _yeom_observation_surface == "client" else {}
    )

    # ── Bilanciamento pool MIA ──────────────────────────────────────────────────
    # ACN-Data: 10,458 members vs 2,615 non-members (split 80/20).
    # Un pool sbilanciato 4:1 non invalida l'AUC-ROC (che è rank-based) ma produce
    # soglie di classificazione asimmetriche. Si usa un subsample fisso dei members
    # per garantire pool identici in dimensione e confrontabilità tra esperimenti.
    _pool_rng        = random.Random(cfg.get("experiment", {}).get("seed", 42))
    _n_bal           = min(len(members), len(non_members))
    members_balanced = _pool_rng.sample(members, _n_bal)
    logger.info(
        f"Yeom pool bilanciato — members: {len(members_balanced)}, "
        f"non-members: {len(non_members)} "
        f"(members originali: {len(members)}, campionati con seed fisso)"
    )

    input_dim = cfg["ml"]["input_dim"]

    def _score_batch(model: Autoencoder, sess_list: list[dict]) -> list[float]:
        """Calcola membership scores (-MSE) sulle sessioni con feature complete.
        Coerente con _sessions_to_tensor: scarta sessioni con None nelle feature."""
        rows: list[list[float]] = []
        for s in sess_list:
            try:
                row = [float(s[f]) for f in _mia_feature_names(cfg)]
                rows.append(row)
            except (KeyError, TypeError, ValueError):
                continue
        if not rows:
            return []
        tensor = torch.tensor(rows, dtype=torch.float32)
        results_: list[float] = []
        with torch.no_grad():
            for i in range(0, len(tensor), 256):
                batch = tensor[i : i + 256]
                recon  = model(batch)
                errors = torch.mean((recon - batch) ** 2, dim=1)
                # Score = -errore: basso errore → membro → score alto
                results_.extend(-e.item() for e in errors)
        return results_

    mia_results: dict[int, dict[str, Any]] = {}
    # Sprint 10zz+29 (2026-09-03, task #54) — curve ROC complete per round,
    # accumulate solo se roc_curve_dump_path è impostato (nessun costo se
    # omesso). Vedi _full_roc_curve() per la motivazione.
    _roc_curves_per_round: dict[int, dict[str, list[float]]] = {}

    for round_num, round_data in sorted(fl_results.items()):
        # Sprint 10zz+94 — "client" costruisce N modelli (uno per client
        # sottomittente in questo round) da round_data["updates"]/["raw_updates"]
        # (stessa selezione dp_mode-aware di run_lira(), Strada B), invece di UN
        # modello da round_data["global_weights"]. `model` sotto resta l'ultimo
        # modello costruito (riusato dal blocco canary più sotto SOLO in
        # modalità "global" — vedi commento lì per il perché in "client" il
        # canary non è ancora supportato).
        model = None
        if _yeom_observation_surface == "client":
            if dp_mode == "dp-fedavg" and not no_dp:
                _yeom_client_updates = round_data.get("raw_updates") or []
            else:
                _yeom_client_updates = round_data.get("updates", [])
            if not _yeom_client_updates:
                logger.warning(
                    f"Round {round_num}: observation_surface='client' ma nessun "
                    "update — skip FedMIA"
                )
                continue

            member_scores: list[float] = []
            non_member_scores: list[float] = []
            for _update in _yeom_client_updates:
                if _update is None or not _update.weights:
                    continue
                _client_model = Autoencoder(input_dim=input_dim, **_autoencoder_arch_kwargs(cfg))
                _orig_state = _client_model.state_dict()
                _keys = list(_orig_state.keys())
                if len(_update.weights) != len(_keys):
                    logger.warning(
                        f"Round {round_num} {getattr(_update, 'cluster_id', '?')}: "
                        "weights shape mismatch — skip client"
                    )
                    continue
                _state = {
                    k: (w if isinstance(w, torch.Tensor) else torch.tensor(w)).to(_orig_state[k].dtype)
                    for k, w in zip(_keys, _update.weights)
                }
                _client_model.load_state_dict(_state, strict=True)
                for _buf_name, _buf in _client_model.named_buffers():
                    if "running_var" in _buf_name:
                        _buf.clamp_(min=1e-8)
                _client_model.eval()
                model = _client_model

                _site_name = getattr(_update, "cluster_id", None)
                _site_members = _members_by_site_yeom.get(_site_name, [])
                if not _site_members:
                    continue
                member_scores.extend(_score_batch(_client_model, _site_members))
                non_member_scores.extend(_score_batch(_client_model, non_members))

            if not member_scores or not non_member_scores:
                logger.warning(f"Round {round_num}: score batch vuoto — skip AUC")
                continue
        else:
            global_weights = round_data.get("global_weights")
            if global_weights is None:
                logger.warning(f"Round {round_num}: global_weights assenti — skip FedMIA")
                continue

            # Carica pesi globali FL in un autoencoder locale (inference only).
            # load_state_dict trasferisce anche i buffer BatchNorm (running_mean/var).
            # global_weights è una lista con lo stesso ordine di state_dict().values():
            # sia AutoencoderTrainer.get_weights() che questo zip usano state_dict()
            # sulla stessa architettura Autoencoder, quindi l'ordine è garantito.
            model = Autoencoder(input_dim=input_dim, **_autoencoder_arch_kwargs(cfg))
            orig_state = model.state_dict()
            keys = list(orig_state.keys())
            if len(global_weights) != len(keys):
                logger.error(
                    f"Round {round_num}: global_weights ha {len(global_weights)} elementi, "
                    f"state_dict ne richiede {len(keys)} — skip FedMIA"
                )
                continue
            state = {
                k: (w if isinstance(w, torch.Tensor) else torch.tensor(w)).to(orig_state[k].dtype)
                for k, w in zip(keys, global_weights)
            }
            model.load_state_dict(state, strict=True)
            # Clamp BatchNorm running_var a valori positivi: il rumore DP con σ grande
            # (es. σ=48 per ε=0.1) può rendere running_var negativa, causando NaN in
            # sqrt(running_var + eps) durante la forward pass in eval mode.
            # Questo guard è difensivo; con il fix in GradientManager._add_noise() i
            # buffer BN non ricevono più rumore, quindi running_var sarà già positiva.
            for buf_name, buf in model.named_buffers():
                if "running_var" in buf_name:
                    buf.clamp_(min=1e-8)
            model.eval()

            member_scores     = _score_batch(model, members_balanced)
            non_member_scores = _score_batch(model, non_members)

            if not member_scores or not non_member_scores:
                logger.warning(f"Round {round_num}: score batch vuoto — skip AUC")
                continue

        labels = [1] * len(member_scores) + [0] * len(non_member_scores)
        scores = member_scores + non_member_scores

        # Filtra NaN residui (belt-and-suspenders: senza il BN fix potrebbe ancora
        # verificarsi NaN per epsilon molto piccoli con sigma >> 1)
        scores_arr = np.array(scores)
        labels_arr = np.array(labels)
        valid_mask = ~np.isnan(scores_arr) & ~np.isinf(scores_arr)
        if valid_mask.sum() < 10:
            logger.warning(
                f"Round {round_num}: troppi score NaN/Inf "
                f"({(~valid_mask).sum()}/{len(scores_arr)}) — probabile corruzione pesi DP "
                f"(σ grande >> norma pesi). AUC impostato a 0.5 (baseline random)."
            )
            mia_results[round_num] = {
                "auc_roc":               0.5,
                "member_score_mean":     float("nan"),
                "non_member_score_mean": float("nan"),
                "nan_fraction":          float((~valid_mask).sum()) / len(scores_arr),
                # Sprint 10zz+28 (2026-09-03, task #53) — vedi sotto per la
                # motivazione; None qui perché non c'è nessuna curva ROC
                # valida da cui derivarli (stesso caso limite di auc_roc=0.5
                # fallback sopra). Chiavi prefissate "yeom_" (fix task #59,
                # Sprint 10zz+34) per coerenza col ramo non-NaN sotto.
                **{f"yeom_tpr_at_fpr_{t}": None for t in _TPR_AT_FPR_TARGETS},
                "advantage":             None,
                "confusion":             None,
            }
            continue
        if not valid_mask.all():
            logger.warning(
                f"Round {round_num}: {(~valid_mask).sum()} score NaN/Inf filtrati "
                f"su {len(scores_arr)} totali"
            )
        labels_arr = labels_arr[valid_mask]
        scores_arr = scores_arr[valid_mask]
        auc = roc_auc_score(labels_arr, scores_arr)
        # Segnalazione 54 (2026-09-24): era "FedMIA AUC-ROC", residuo del vecchio
        # nome run_fedmia. E' l'attacco di Yeom, non FedMIA.
        logger.info(f"Round {round_num} — Yeom AUC-ROC: {auc:.4f}")

        # Sprint 10zz+28 (2026-09-03, task #53) — TPR@low-FPR/Advantage/
        # Confusion Matrix, finora cablati SOLO su LiRA (_tpr_at_fixed_fpr/
        # _mia_advantage/_mia_confusion_at_best_threshold), estesi qui a
        # Yeom. Motivazione: la "fallacia delle medie" di Carlini et al.
        # 2022 (un attacco chirurgico su un piccolo sottogruppo e un
        # attacco uniformemente mediocre possono avere lo stesso AUC-ROC)
        # si applica non solo al confronto membro/non-membro DENTRO un
        # attacco, ma anche al confronto TRA attacchi diversi (Yeom vs
        # Shadow vs LiRA) — se quel confronto usasse solo `auc_roc`,
        # cadrebbe nella stessa fallacia. Stesse coppie label/score già
        # usate per `auc`, nessun costo aggiuntivo. NON retroattivo sui
        # JSON storici (richiede la curva ROC completa, mai salvata prima).
        # Fix 2026-09-03 (task #59, Sprint 10zz+34) — bug scoperto durante la
        # verifica richiesta dall'utente sulla "fallacia delle medie" TRA
        # attacchi (task #53): _tpr_at_fixed_fpr() restituisce SEMPRE le
        # stesse chiavi generiche ("tpr_at_fpr_0.001" ecc.), identiche a
        # quelle già usate da run_lira() (bare, da prima di task #53) e da
        # run_shadow() (introdotte insieme a queste, stesso bug).
        # run_registered_attacks() fonde yeom→shadow→lira nello STESSO dict
        # per round con .update() — l'ultimo scrittore vince. Senza prefisso,
        # il TPR@low-FPR di Yeom calcolato qui viene silenziosamente
        # sovrascritto da quello di Shadow e poi da quello di LiRA prima di
        # essere salvato: il campo "tpr_at_fpr_*" nel JSON finale è SEMPRE
        # quello di LiRA, mai quello di Yeom, anche se Yeom lo calcola
        # correttamente qui. advantage/confusion non hanno questo problema
        # (Yeom li salva bare per convenzione, Shadow/LiRA/canary li salvano
        # già con prefisso "shadow_"/"lira_"/"canary_" — solo TPR mancava il
        # prefisso). Qui prefissato "yeom_" per coerenza con l'unico altro
        # campo Yeom-specifico che rischiava la stessa collisione.
        tpr_fields = {
            f"yeom_{k}": v for k, v in _tpr_at_fixed_fpr(list(labels_arr), list(scores_arr)).items()
        }
        advantage = _mia_advantage(list(labels_arr), list(scores_arr))
        confusion = _mia_confusion_at_best_threshold(list(labels_arr), list(scores_arr))
        if roc_curve_dump_path is not None:
            _curve = _full_roc_curve(list(labels_arr), list(scores_arr))
            if _curve is not None:
                _roc_curves_per_round[round_num] = _curve

        # Canary positive control (Sprint 10zz+93, 2026-09-15) — stessa idea
        # già in uso in run_lira() (Sprint 10vv), estesa qui su richiesta
        # esplicita dell'utente (Blocker 2): verificare un vero leak di
        # membership con TUTTI e tre gli attacchi già implementati (Yeom,
        # Shadow, LiRA), non solo LiRA. AUC calcolato SOLO sui campioni
        # canary (membri duplicati vs gemelli non-membro mai visti in
        # training), None se i canary sono disabilitati o un lato del pool
        # è vuoto in questo round. Usa `members` (pool completo, non il
        # sotto-campione bilanciato `members_balanced`) per non dipendere
        # dalla varianza di quel campionamento casuale — stesso modello
        # `model` già caricato per questo round, nessun training aggiuntivo.
        # Chiavi prefissate "yeom_" (stesso motivo/bug di collisione già
        # corretto per tpr_at_fpr_* in task #59, Sprint 10zz+34):
        # run_registered_attacks() fonde yeom→shadow→lira con .update()
        # nello stesso dict per round — senza prefisso, il canary_auc_roc
        # (bare) già pubblicato da LiRA sovrascriverebbe silenziosamente
        # quello di Yeom.
        #
        # Sprint 10zz+94 — limitato a observation_surface=="global": in
        # modalità "client" `model` è solo l'ultimo modello-client costruito
        # nel round (non è per-sito), quindi valutarci sopra TUTTI i canary
        # (che appartengono a siti eterogenei) darebbe un numero fuorviante.
        # Semplificazione dichiarata: canary sotto "client" non ancora
        # supportato, vedi TestRoadmap_DSN2027.md.
        canary_members    = [s for s in members if s.get("_canary_role") == "member"]
        canary_nonmembers = [s for s in non_members if s.get("_canary_role") == "nonmember"]
        yeom_canary_auc_roc = None
        yeom_canary_advantage = None
        yeom_canary_confusion = None
        if _yeom_observation_surface == "global" and canary_members and canary_nonmembers:
            _canary_member_scores    = _score_batch(model, canary_members)
            _canary_nonmember_scores = _score_batch(model, canary_nonmembers)
            if _canary_member_scores and _canary_nonmember_scores:
                _c_labels = [1] * len(_canary_member_scores) + [0] * len(_canary_nonmember_scores)
                _c_scores = _canary_member_scores + _canary_nonmember_scores
                try:
                    yeom_canary_auc_roc = round(float(roc_auc_score(_c_labels, _c_scores)), 6)
                except ValueError:
                    yeom_canary_auc_roc = None
                yeom_canary_advantage = _mia_advantage(_c_labels, _c_scores)
                yeom_canary_confusion = _mia_confusion_at_best_threshold(_c_labels, _c_scores)

        mia_results[round_num] = {
            "auc_roc":               auc,
            "member_score_mean":     float(np.nanmean(member_scores)),
            "non_member_score_mean": float(np.nanmean(non_member_scores)),
            **tpr_fields,
            "advantage":             advantage,
            "confusion":             confusion,
            "yeom_canary_auc_roc":   yeom_canary_auc_roc,
            "yeom_canary_advantage": yeom_canary_advantage,
            "yeom_canary_confusion": yeom_canary_confusion,
        }

    if roc_curve_dump_path is not None and _roc_curves_per_round:
        _write_diagnostic_dump(roc_curve_dump_path, {
            "attack": "yeom",
            "seed": cfg.get("experiment", {}).get("seed"),
            "epsilon": cfg.get("experiment", {}).get("epsilon"),
            "no_dp": cfg.get("experiment", {}).get("no_dp", False),
            "dp_mode": cfg.get("experiment", {}).get("dp_mode"),
            "per_round": _roc_curves_per_round,
        })

    return mia_results


# Alias di compatibilità (Sprint 10zz+107, 2026-09-15, richiesto esplicitamente
# dall'utente prima di scrivere il paper: "run_yeom" era il nome storico
# dell'attacco Yeom 2018 — da quando FedMIA-gradient (run_fedmia_gradient()
# sotto, un attacco DIVERSO) è stato aggiunto, "fedmia" nel nome di QUESTA
# funzione è fuorviante, non solo storico: un lettore del codice affianco al
# paper (che chiama questo attacco "Yeom" ovunque) troverebbe due funzioni
# "fedmia" per due attacchi diversi. Il nome canonico è ora run_yeom(); questo
# alias resta SOLO per non rompere test/chiamate esterne esistenti (es.
# tests/test_run_experiments_integration.py, che chiama run_yeom()
# direttamente e non può essere eseguito in questo sandbox — niente torch —
# quindi non lo si tocca senza necessità). Nessun nuovo codice deve usare
# questo alias: usare run_yeom().
run_yeom = run_yeom


# ── Shadow Model MIA Attack ────────────────────────────────────────────────────

def run_shadow(
    cfg: dict,
    train_sessions: list[dict[str, Any]],
    holdout_sessions: list[dict[str, Any]],
    fl_results: dict[int, dict[str, Any]],
    roc_curve_dump_path: str | None = None,
    no_dp: bool = False,
    dp_mode: str = "dp-fedavg",
) -> dict[int, dict[str, Any]]:
    """
    Calibrated Shadow-Model MIA Attack (ispirato a LiRA, Carlini et al. 2022).

    Motivazione:
        L'attacco Yeom (loss-based) fallisce quando il modello FL generalizza bene:
        la loss su membro e non-membro è simile → AUC ≈ 0.5. Il shadow attack
        controlla per la "difficoltà intrinseca" di ogni campione confrontando
        il modello FL target con un modello shadow addestrato su dati simili ma
        diversi: il segnale di membership emerge dalla *differenza* di loss, non
        dal valore assoluto.

    Setup:
        - shadow_train (50% di train_sessions) → addestra shadow model (no FL, no DP)
        - eval_members (50% di train_sessions) → valutazione membro:
            * FL ha visto questi campioni → loss_target bassa
            * shadow NON li ha visti     → loss_shadow alta
            * score = loss_shadow - loss_target > 0  ✓
        - holdout_sessions → valutazione non-membro:
            * né FL né shadow li hanno visti
            * score = loss_shadow - loss_target ≈ 0  ✓

    Score di membership:
        score(x) = MSE(shadow_model, x) − MSE(target_model, x)
        Maggiore score → più probabile che x sia stato nel training FL.

    Riferimenti:
        Carlini et al., "Membership Inference Attacks From First Principles",
        IEEE S&P 2022. https://arxiv.org/abs/2112.03570

    Args:
        cfg:              configurazione esperimento
        train_sessions:   sessioni usate nel FL training (tutti i membri)
        holdout_sessions: sessioni mai viste durante FL (non-membri)
        fl_results:       dict round → {"global_weights": [...], ...}

    Returns:
        {round_num: {
            "shadow_auc_roc": float,
            "shadow_member_score_mean": float,
            "shadow_non_member_score_mean": float,
            "shadow_score_gap": float,   # differenza media membro - non-membro
            "n_eval_members": int,
            "n_non_members": int,
            # Sprint 10zz+28 (2026-09-03, task #53) — stesso motivo di
            # run_yeom(), vedi lì: non retroattivo sui JSON storici.
            # Prefisso "shadow_" dal Sprint 10zz+34 (task #59, fix collisione
            # di chiavi con Yeom/LiRA nel merge di run_registered_attacks()).
            "shadow_tpr_at_fpr_0.001"/"...0.01"/"...0.05": float | None,
            "shadow_advantage": float | None,
            "shadow_confusion": dict | None,
        }}
    """
    from sklearn.metrics import roc_auc_score

    seed       = cfg.get("experiment", {}).get("seed", 42)
    input_dim  = cfg["ml"]["input_dim"]
    ml_cfg     = cfg["ml"]
    local_epochs = ml_cfg.get("epochs", 3)
    total_rounds = cfg.get("experiment", {}).get("fl_rounds", 100)

    # ── Step 1: split train → shadow_train (50%) + eval_members (50%) ──────────
    rng = random.Random(seed + 999)       # seed diverso da quello del pool Yeom
    shuffled = list(train_sessions)
    rng.shuffle(shuffled)
    mid           = max(1, len(shuffled) // 2)
    shadow_train  = shuffled[:mid]
    eval_members  = shuffled[mid:]

    # Sprint 10zz+93 (2026-09-15) — canary positive control esteso a Shadow
    # (richiesto dall'utente per Blocker 2: verificare un vero leak di
    # membership con TUTTI e tre gli attacchi già implementati, non solo
    # LiRA). Se canary sono abilitati, i duplicati canary non devono MAI
    # finire in shadow_train: lo shadow model verrebbe addestrato sugli
    # stessi record esatti che poi valutiamo come "membro", contaminando
    # la calibrazione shadow-vs-target — stessa causa radice del bug già
    # corretto per LiRA in _sample_preserving_canary_groups() (Sprint
    # 10zz+16): uno split casuale che non tratta i gruppi canary come
    # atomici. Qui la correzione è più semplice perché i canary sono
    # synthetic controls, non organici: spostarli TUTTI in eval_members
    # (mai usati per addestrare lo shadow model) è corretto per costruzione,
    # non solo un workaround. Nessun impatto se canary è disabilitato
    # (nessuna sessione ha "_canary_role").
    _canary_in_shadow_train = [s for s in shadow_train if s.get("_canary_role") == "member"]
    if _canary_in_shadow_train:
        shadow_train = [s for s in shadow_train if s.get("_canary_role") != "member"]
        eval_members = eval_members + _canary_in_shadow_train
        logger.info(
            f"[CANARY] {len(_canary_in_shadow_train)} sessioni canary spostate da "
            "shadow_train a eval_members per evitare contaminazione della "
            "calibrazione shadow (Sprint 10zz+93)."
        )

    logger.info(
        f"Shadow MIA — shadow_train: {len(shadow_train)}, "
        f"eval_members: {len(eval_members)}, non-members: {len(holdout_sessions)}"
    )

    # Sprint 10zz+94 (2026-09-15, task #155) — observation_surface opt-in,
    # stesso significato/design di Yeom sopra: "global" (default, invariato)
    # confronta shadow_model contro un UNICO target_model per round, costruito
    # da round_data["global_weights"]; "client" confronta invece contro un
    # target_model per-client costruito da round_data["updates"]/["raw_updates"]
    # (stessa selezione dp_mode-aware di run_lira()/run_yeom(), Strada B),
    # valutando ogni sito solo sui propri eval_members. Il shadow_model di
    # calibrazione resta UNICO e globale in entrambe le modalità: è il
    # riferimento "non ha mai visto questi dati", non dipende dal FL round,
    # quindi non ha un analogo "per-client" sensato. Semplificazione
    # dichiarata: il canary block più sotto resta limitato a "global" (stesso
    # motivo di Yeom — un target_model per-client non è rappresentativo di
    # tutti i canary, che appartengono a siti eterogenei); non ancora
    # verificato con un run reale (nessun torch in questo sandbox), vedi
    # TestRoadmap_DSN2027.md.
    _shadow_observation_surface = cfg.get("shadow", {}).get("observation_surface", "global")
    if _shadow_observation_surface not in ("global", "client"):
        raise ValueError(
            f"cfg['shadow']['observation_surface'] non valido: {_shadow_observation_surface!r} "
            "(atteso 'global' o 'client')"
        )
    _eval_members_by_site_shadow = (
        group_sessions_by_site(eval_members) if _shadow_observation_surface == "client" else {}
    )

    # ── Step 2: addestra il shadow model ──────────────────────────────────────
    # Autoencoder locale (non FL, no DP) addestrato sul shadow_train.
    # Epoche totali = local_epochs × total_rounds (equivalente al training FL),
    # capped a 500 per non rallentare troppo il sweep.
    shadow_epochs = min(local_epochs * total_rounds, 500)

    def _build_tensor(sess_list: list[dict]) -> torch.Tensor | None:
        rows = []
        for s in sess_list:
            try:
                rows.append([float(s[f]) for f in _mia_feature_names(cfg)])
            except (KeyError, TypeError, ValueError):
                continue
        return torch.tensor(rows, dtype=torch.float32) if rows else None

    shadow_tensor = _build_tensor(shadow_train)
    if shadow_tensor is None or len(shadow_tensor) == 0:
        logger.warning("Shadow MIA: shadow_train vuoto dopo feature extraction — skip")
        return {}

    # Fix (2026-07-24, review indipendente — stessa classe di bug già corretta in
    # run_lira() il 2026-07-21d): seed PRIMA di istanziare il modello. Autoencoder()
    # pesca l'init casuale dei pesi dall'RNG globale di torch — se il seed viene
    # fissato solo qui sotto (dopo la costruzione), l'init dipende da qualunque cosa
    # abbia consumato l'RNG globale prima (draw di rumore DP, init di altri modelli,
    # ecc.), rendendo shadow_auc_roc/shadow_score_gap non riproducibili "a parità di
    # seed" e — più grave — confondendo il confronto no-DP vs DP: il path no-DP non
    # fa mai draw di rumore prima di questo punto, il path DP sì, quindi i due rami
    # partivano da pesi iniziali diversi anche a seed identico.
    torch.manual_seed(seed + 999)
    shadow_model = Autoencoder(input_dim=input_dim, **_autoencoder_arch_kwargs(cfg))
    shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=ml_cfg.get("lr", 1e-3))
    shadow_criterion = torch.nn.MSELoss()
    batch_size = ml_cfg.get("batch_size", 32)

    if len(shadow_tensor) < batch_size:
        logger.warning(
            f"Shadow MIA: shadow_train ({len(shadow_tensor)} sessioni) < "
            f"batch_size ({batch_size}) — shadow model non addestrato con drop_last=True. "
            "shadow_auc_roc non significativo. Aumentare il dataset o ridurre batch_size."
        )
        return {}

    shadow_model.train()
    torch.manual_seed(seed + 999)
    shadow_ds = torch.utils.data.TensorDataset(shadow_tensor)
    shadow_loader_gen = torch.Generator()
    shadow_loader_gen.manual_seed(seed + 999)
    shadow_loader = torch.utils.data.DataLoader(
        shadow_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, generator=shadow_loader_gen,
    )

    for epoch in range(shadow_epochs):
        for (batch,) in shadow_loader:
            shadow_optimizer.zero_grad()
            recon = shadow_model(batch)
            loss  = shadow_criterion(recon, batch)
            loss.backward()
            shadow_optimizer.step()

    shadow_model.eval()
    logger.info(f"Shadow model addestrato — {shadow_epochs} epoche su {len(shadow_train)} sessioni")

    # ── Step 3: compute scores per-sample con entrambi i modelli ───────────────

    def _mse_batch(model: Autoencoder, sess_list: list[dict]) -> list[float]:
        """Calcola MSE per sessione (non negato: valore grezzo per calibrazione)."""
        tensor = _build_tensor(sess_list)
        if tensor is None:
            return []
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(tensor), 256):
                batch = tensor[i : i + 256]
                recon  = model(batch)
                errors = torch.mean((recon - batch) ** 2, dim=1)
                scores.extend(e.item() for e in errors)
        return scores

    # Pre-computa shadow scores una volta sola (non dipende dal round FL)
    shadow_scores_members     = _mse_batch(shadow_model, eval_members)
    shadow_scores_nonmembers  = _mse_batch(shadow_model, holdout_sessions)
    # Preservata (ordine = holdout_sessions) per observation_surface="client"
    # sotto — lì ogni client-model valuta l'intero pool non-membro, senza il
    # sotto-campionamento bilanciato usato dal ramo "global".
    _shadow_scores_nonmembers_unbalanced = list(shadow_scores_nonmembers)

    # Bilanciamento: stessa dimensione per eval_members e non-members
    _bal_rng       = random.Random(seed + 999)
    _n_bal         = min(len(shadow_scores_members), len(shadow_scores_nonmembers))
    shadow_scores_members    = _bal_rng.sample(shadow_scores_members, _n_bal)
    shadow_scores_nonmembers = _bal_rng.sample(shadow_scores_nonmembers, _n_bal)

    # Sprint 10zz+94 — precomputa shadow scores per sito (shadow_model è unico
    # e non dipende dal round FL, quindi questo va fatto una sola volta, come
    # sopra). Ordine di ogni lista = ordine di _eval_members_by_site_shadow[sito],
    # cosi' lo zip con i target scores per-client nel round loop resta allineato
    # per indice senza ricorrere al trucco "stesso seed" usato dal ramo "global".
    _shadow_scores_by_site: dict[str, list[float]] = {}
    if _shadow_observation_surface == "client":
        for _site_name, _site_members in _eval_members_by_site_shadow.items():
            if _site_members:
                _shadow_scores_by_site[_site_name] = _mse_batch(shadow_model, _site_members)

    # ── Step 4: per ogni round FL, calcola score calibrato ─────────────────────
    shadow_results: dict[int, dict[str, Any]] = {}
    # Sprint 10zz+29 (2026-09-03, task #54) — vedi run_yeom() sopra.
    _roc_curves_per_round: dict[int, dict[str, list[float]]] = {}

    for round_num, round_data in sorted(fl_results.items()):
        # Sprint 10zz+94 — "client" costruisce N target model (uno per client
        # sottomittente nel round, da round_data["updates"]/["raw_updates"],
        # stessa selezione dp_mode-aware di run_lira()/run_yeom(), Strada B)
        # invece di UN target model da round_data["global_weights"]. Ogni
        # client-model valuta SOLO i propri eval_members (via
        # _eval_members_by_site_shadow) contro i propri shadow scores
        # pre-calcolati (_shadow_scores_by_site), e l'intero pool non-membro
        # condiviso; i risultati di tutti i client sono poi concatenati.
        target_model = None
        if _shadow_observation_surface == "client":
            if dp_mode == "dp-fedavg" and not no_dp:
                _shadow_client_updates = round_data.get("raw_updates") or []
            else:
                _shadow_client_updates = round_data.get("updates", [])
            if not _shadow_client_updates:
                logger.warning(
                    f"Shadow MIA round {round_num}: observation_surface='client' ma "
                    "nessun update — skip round"
                )
                continue

            calibrated_members: list[float] = []
            calibrated_nonmembers: list[float] = []
            for _update in _shadow_client_updates:
                if _update is None or not _update.weights:
                    continue
                _client_target = Autoencoder(input_dim=input_dim, **_autoencoder_arch_kwargs(cfg))
                _orig_state = _client_target.state_dict()
                _keys = list(_orig_state.keys())
                if len(_update.weights) != len(_keys):
                    logger.warning(
                        f"Shadow MIA round {round_num} {getattr(_update, 'cluster_id', '?')}: "
                        "weights shape mismatch — skip client"
                    )
                    continue
                _state = {
                    k: (w if isinstance(w, torch.Tensor) else torch.tensor(w)).to(_orig_state[k].dtype)
                    for k, w in zip(_keys, _update.weights)
                }
                _client_target.load_state_dict(_state, strict=True)
                for _buf_name, _buf in _client_target.named_buffers():
                    if "running_var" in _buf_name:
                        _buf.clamp_(min=1e-8)
                _client_target.eval()
                target_model = _client_target

                _site_name = getattr(_update, "cluster_id", None)
                _site_shadow_scores = _shadow_scores_by_site.get(_site_name)
                _site_members = _eval_members_by_site_shadow.get(_site_name, [])
                if not _site_shadow_scores or not _site_members:
                    continue
                _site_target_scores = _mse_batch(_client_target, _site_members)
                calibrated_members.extend(
                    s - t for s, t in zip(_site_shadow_scores, _site_target_scores)
                )

                _client_nonmember_target_scores = _mse_batch(_client_target, holdout_sessions)
                calibrated_nonmembers.extend(
                    s - t for s, t in zip(_shadow_scores_nonmembers_unbalanced, _client_nonmember_target_scores)
                )

            if not calibrated_members or not calibrated_nonmembers:
                logger.warning(f"Shadow MIA round {round_num}: score batch vuoto — skip AUC")
                continue
        else:
            global_weights = round_data.get("global_weights")
            if global_weights is None:
                continue

            # Carica pesi globali FL nel target model
            target_model = Autoencoder(input_dim=input_dim, **_autoencoder_arch_kwargs(cfg))
            orig_state   = target_model.state_dict()
            keys         = list(orig_state.keys())
            if len(global_weights) != len(keys):
                logger.error(
                    f"Shadow MIA round {round_num}: global_weights {len(global_weights)} "
                    f"!= state_dict {len(keys)} — skip"
                )
                continue
            state = {
                k: (w if isinstance(w, torch.Tensor) else torch.tensor(w)).to(orig_state[k].dtype)
                for k, w in zip(keys, global_weights)
            }
            target_model.load_state_dict(state, strict=True)
            for buf_name, buf in target_model.named_buffers():
                if "running_var" in buf_name:
                    buf.clamp_(min=1e-8)
            target_model.eval()

            # Scores target model
            target_scores_members    = _mse_batch(target_model, eval_members)
            target_scores_nonmembers = _mse_batch(target_model, holdout_sessions)

            # Bilancia anche i target scores allo stesso indice del shadow
            _bal_rng2 = random.Random(seed + 999)
            target_scores_members    = _bal_rng2.sample(target_scores_members, _n_bal)
            target_scores_nonmembers = _bal_rng2.sample(target_scores_nonmembers, _n_bal)

            # score calibrato = loss_shadow − loss_target
            # Positivo → target conosce il campione meglio del shadow → membro
            calibrated_members    = [
                s - t for s, t in zip(shadow_scores_members,    target_scores_members)
            ]
            calibrated_nonmembers = [
                s - t for s, t in zip(shadow_scores_nonmembers, target_scores_nonmembers)
            ]

        _n_members_out    = len(calibrated_members)
        _n_nonmembers_out = len(calibrated_nonmembers)
        labels = [1] * len(calibrated_members) + [0] * len(calibrated_nonmembers)
        scores = calibrated_members + calibrated_nonmembers

        scores_arr = np.array(scores)
        labels_arr = np.array(labels)
        valid_mask = ~np.isnan(scores_arr) & ~np.isinf(scores_arr)
        if valid_mask.sum() < 10:
            logger.warning(f"Shadow MIA round {round_num}: troppi NaN — skip")
            continue
        scores_arr = scores_arr[valid_mask]
        labels_arr = labels_arr[valid_mask]

        try:
            auc = roc_auc_score(labels_arr, scores_arr)
        except ValueError:
            auc = 0.5

        score_gap = float(np.nanmean(calibrated_members) - np.nanmean(calibrated_nonmembers))
        logger.info(
            f"Round {round_num} — Shadow MIA AUC: {auc:.4f} "
            f"(gap={score_gap:.6f})"
        )

        # Sprint 10zz+28 (2026-09-03, task #53) — stesso motivo/pattern di
        # run_yeom() sopra e run_lira() sotto: TPR@low-FPR/Advantage/
        # Confusion sulle stesse coppie label/score già usate per `auc`,
        # cosi' un confronto Yeom/Shadow/LiRA non deve appoggiarsi solo su
        # auc_roc (fallacia delle medie di Carlini applicata al confronto
        # TRA attacchi, non solo dentro un attacco). Non retroattivo.
        # Fix 2026-09-03 (task #59, Sprint 10zz+34) — vedi commento gemello
        # in run_yeom() sopra: stesso bug di collisione chiavi
        # "tpr_at_fpr_*" con LiRA nel merge di run_registered_attacks(),
        # stessa correzione (prefisso "shadow_", coerente con
        # shadow_advantage/shadow_confusion già prefissati qui sotto).
        tpr_fields = {
            f"shadow_{k}": v for k, v in _tpr_at_fixed_fpr(list(labels_arr), list(scores_arr)).items()
        }
        advantage = _mia_advantage(list(labels_arr), list(scores_arr))
        confusion = _mia_confusion_at_best_threshold(list(labels_arr), list(scores_arr))
        if roc_curve_dump_path is not None:
            _curve = _full_roc_curve(list(labels_arr), list(scores_arr))
            if _curve is not None:
                _roc_curves_per_round[round_num] = _curve

        # Canary positive control (Sprint 10zz+93, 2026-09-15) — vedi
        # run_yeom() sopra per motivazione/pattern identico (Blocker 2).
        # canary_members è sempre un sottoinsieme di eval_members (mai di
        # shadow_train, vedi guardia nello split sopra). Ricalcolato qui sul
        # pool canary completo (non sul sotto-campione bilanciato
        # shadow_scores_members/target_scores_members) per non dipendere
        # dalla varianza di quel bilanciamento casuale. Chiavi prefissate
        # "shadow_" per lo stesso motivo di yeom_canary_auc_roc in
        # run_yeom() — evitare la collisione con il canary_auc_roc (bare)
        # già pubblicato da LiRA nel merge yeom→shadow→lira.
        #
        # Sprint 10zz+94 — limitato a observation_surface=="global", stesso
        # motivo del canary block gemello in run_yeom(): in modalità
        # "client" `target_model` è solo l'ultimo client-model costruito nel
        # round, non rappresentativo di tutti i siti a cui i canary
        # appartengono. Semplificazione dichiarata, vedi TestRoadmap_DSN2027.md.
        canary_members    = [s for s in eval_members if s.get("_canary_role") == "member"]
        canary_nonmembers = [s for s in holdout_sessions if s.get("_canary_role") == "nonmember"]
        shadow_canary_auc_roc = None
        shadow_canary_advantage = None
        shadow_canary_confusion = None
        # Sprint 10zz+97 (2026-09-15) — diagnostica additiva, richiesta dal
        # secondo run reale di Blocker 2: il fix Sprint 10zz+96 (tag
        # dell'originale non taggato) non ha risolto l'inversione di
        # shadow_canary_auc_roc (anzi 0.21/0.24/0.25, leggermente PEGGIO di
        # 0.26/0.32/0.32 prima del fix) — l'ipotesi "contaminazione
        # shadow_train" non spiega (da sola) il fenomeno. Ipotesi alternativa
        # da verificare con un run reale: con solo 5 template distinti (anche
        # se replicati 31 volte ciascuno dopo il fix), l'AUC canary di Shadow
        # potrebbe essere dominato da una varianza di campione effettivo
        # piccolissimo (5 valori distinti lato membro contro 20 lato
        # non-membro), non da un bug — un singolo shadow model senza
        # normalizzazione per varianza (a differenza degli 8 shadow Gaussiani
        # di LiRA) è un estimatore molto più fragile su n_templates=5. Questi
        # campi espongono la media calibrata PER GRUPPO canary (lato membro)
        # e le statistiche del lato non-membro, per distinguere le due ipotesi
        # senza bisogno di un nuovo dump per-campione. Puramente additivo,
        # nessun impatto su shadow_canary_auc_roc/advantage/confusion sopra.
        shadow_canary_debug_group_means: dict[str, float] | None = None
        shadow_canary_debug_nonmember_stats: dict[str, float] | None = None
        shadow_canary_debug_group_raw: dict[str, dict[str, float]] | None = None
        shadow_canary_debug_nonmember_raw: dict[str, float] | None = None
        if _shadow_observation_surface == "global" and canary_members and canary_nonmembers:
            _c_shadow_m = _mse_batch(shadow_model, canary_members)
            _c_shadow_n = _mse_batch(shadow_model, canary_nonmembers)
            _c_target_m = _mse_batch(target_model, canary_members)
            _c_target_n = _mse_batch(target_model, canary_nonmembers)
            if _c_shadow_m and _c_shadow_n and _c_target_m and _c_target_n:
                _c_cal_m = [s - t for s, t in zip(_c_shadow_m, _c_target_m)]
                _c_cal_n = [s - t for s, t in zip(_c_shadow_n, _c_target_n)]
                _c_labels = [1] * len(_c_cal_m) + [0] * len(_c_cal_n)
                _c_scores = _c_cal_m + _c_cal_n
                try:
                    shadow_canary_auc_roc = round(float(roc_auc_score(_c_labels, _c_scores)), 6)
                except ValueError:
                    shadow_canary_auc_roc = None
                shadow_canary_advantage = _mia_advantage(_c_labels, _c_scores)
                shadow_canary_confusion = _mia_confusion_at_best_threshold(_c_labels, _c_scores)

                _group_scores: dict[str, list[float]] = {}
                for _s, _cal in zip(canary_members, _c_cal_m):
                    _group_scores.setdefault(_s.get("_canary_group", "?"), []).append(_cal)
                shadow_canary_debug_group_means = {
                    g: round(float(np.mean(vals)), 6) for g, vals in sorted(_group_scores.items())
                }
                shadow_canary_debug_nonmember_stats = {
                    "mean": round(float(np.mean(_c_cal_n)), 6),
                    "std":  round(float(np.std(_c_cal_n)), 6),
                    "min":  round(float(np.min(_c_cal_n)), 6),
                    "max":  round(float(np.max(_c_cal_n)), 6),
                }

                # Sprint 10zz+99 (2026-09-15) — ultimo dump richiesto
                # dall'utente prima di chiudere la diagnosi: i campi sopra
                # mostrano solo la DIFFERENZA calibrata (shadow_loss -
                # target_loss), che esclude la varianza di campione ma non
                # dice se il "pavimento" ipotizzato in
                # docs/TestRoadmap_DSN2027.md (sia shadow che target vicini al
                # proprio errore minimo per sessioni tipiche di questo sito,
                # rendendo la differenza assoluta dominata dalla difficoltà
                # intrinseca del campione) sia reale — serve il valore
                # ASSOLUTO di shadow_loss e target_loss separatamente, non solo
                # la loro differenza. Se l'ipotesi regge: target_mean sarà
                # molto più basso della loss "tipica" (conferma memorizzazione,
                # coerente con yeom_canary_auc_roc alto) E shadow_mean sarà
                # QUASI ALTRETTANTO basso (non "tipico") sugli stessi 5
                # campioni — a differenza dei non-membri, dove ci si aspetta
                # shadow_mean e target_mean più simili tra loro E più alti in
                # assoluto. Puramente additivo, nessun impatto sui campi
                # esistenti.
                _group_raw: dict[str, dict[str, float]] = {}
                for _s, _sh, _tg in zip(canary_members, _c_shadow_m, _c_target_m):
                    _grp = _s.get("_canary_group", "?")
                    _acc = _group_raw.setdefault(_grp, {"shadow": [], "target": []})
                    _acc["shadow"].append(_sh)
                    _acc["target"].append(_tg)
                shadow_canary_debug_group_raw = {
                    g: {
                        "shadow_mean": round(float(np.mean(v["shadow"])), 6),
                        "target_mean": round(float(np.mean(v["target"])), 6),
                        "n": len(v["shadow"]),
                    }
                    for g, v in sorted(_group_raw.items())
                }
                shadow_canary_debug_nonmember_raw = {
                    "shadow_mean": round(float(np.mean(_c_shadow_n)), 6),
                    "shadow_std":  round(float(np.std(_c_shadow_n)), 6),
                    "target_mean": round(float(np.mean(_c_target_n)), 6),
                    "target_std":  round(float(np.std(_c_target_n)), 6),
                }

        shadow_results[round_num] = {
            "shadow_auc_roc":               round(auc, 6),
            "shadow_member_score_mean":     round(float(np.nanmean(calibrated_members)), 6),
            "shadow_non_member_score_mean": round(float(np.nanmean(calibrated_nonmembers)), 6),
            "shadow_score_gap":             round(score_gap, 6),
            "n_eval_members":               _n_members_out,
            "n_non_members":                _n_nonmembers_out,
            **tpr_fields,
            "shadow_advantage":             advantage,
            "shadow_confusion":             confusion,
            "shadow_canary_auc_roc":        shadow_canary_auc_roc,
            "shadow_canary_advantage":      shadow_canary_advantage,
            "shadow_canary_confusion":      shadow_canary_confusion,
            "shadow_canary_debug_group_means":     shadow_canary_debug_group_means,
            "shadow_canary_debug_nonmember_stats": shadow_canary_debug_nonmember_stats,
            "shadow_canary_debug_group_raw":       shadow_canary_debug_group_raw,
            "shadow_canary_debug_nonmember_raw":   shadow_canary_debug_nonmember_raw,
        }

    if roc_curve_dump_path is not None and _roc_curves_per_round:
        _write_diagnostic_dump(roc_curve_dump_path, {
            "attack": "shadow",
            "seed": cfg.get("experiment", {}).get("seed"),
            "epsilon": cfg.get("experiment", {}).get("epsilon"),
            "no_dp": cfg.get("experiment", {}).get("no_dp", False),
            "dp_mode": cfg.get("experiment", {}).get("dp_mode"),
            "per_round": _roc_curves_per_round,
        })

    return shadow_results


# ── LiRA Attack (Carlini et al. 2022) ──────────────────────────────────────────

def _rescale_update_weights(
    observed: list[Any],
    reference: list[Any],
    s: float,
    weight_keys: list[str],
) -> list[Any] | None:
    """Pesi ref + s*(observed - ref) per l'attaccante che riscala l'update.

    Aggiunto il 2026-10-09 (Sprint 10zz+170). Con il taglio a C l'update del
    client arriva ridotto (Office 1: norma circa 16 portata a 1 al round 1) e il
    modello osservato resta vicino al riferimento; un attaccante che conosce il
    riferimento (il modello inviato dal server) puo' riscalare la direzione
    dell'update. s = 1 riproduce il modello osservato. Stessa convenzione di
    GradientManager._clip_weights(): buffer BatchNorm e tensori non float non
    si riscalano (si tiene il valore osservato). None se le liste non hanno la
    stessa lunghezza.
    """
    if len(observed) != len(reference) or len(observed) != len(weight_keys):
        return None
    _bn = {"running_mean", "running_var", "num_batches_tracked"}
    out: list[Any] = []
    for k, w, r in zip(weight_keys, observed, reference):
        wt = w if isinstance(w, torch.Tensor) else torch.tensor(w)
        rt = r if isinstance(r, torch.Tensor) else torch.tensor(r)
        if k.split(".")[-1] in _bn or not wt.is_floating_point():
            out.append(wt.clone())
            continue
        rt = rt.to(wt.dtype)
        out.append(rt + s * (wt - rt))
    return out


# ── RMIA (2026-10-13) ───────────────────────────────────────────────────────────
# Zarifzadeh, Liu, Bhatt, Shokri, "Low-Cost High-Power Membership Inference
# Attacks", ICML 2024. Calcolato dentro run_lira() sugli stessi modelli shadow
# (nessun addestramento in piu'), ma su TUTTI i campioni con almeno un
# riferimento: LiRA scarta quelli con meno di 2 shadow OUT, RMIA no.
#
# Adattamento all'autoencoder: RMIA usa Pr(x | theta), che per un classificatore
# e' una probabilita' confrontabile fra modelli. Qui Pr(x | theta) =
# exp(-L(x) / tau_theta), con L l'errore di ricostruzione e tau_theta la mediana
# delle loss DI QUEL MODELLO sui campioni del client nel round: ogni modello
# (bersaglio e ogni shadow) e' normalizzato sulla propria scala. Senza questa
# normalizzazione, con shadow addestrati meno del bersaglio (--shadow-epochs-cap)
# lo score diventa una differenza di loss dominata dalla scala degli shadow, e
# si inverte (verificato su dati sintetici). Per ogni x:
#   LR(x) = Pr(x | theta) / Pr(x),  Pr(x) = media su TUTTI i riferimenti (IN e
#           OUT) ("online"), oppure 1/2((1+a) media_OUT + (1-a)) ("offline",
#           solo OUT, a = RMIA_A_OFFLINE);
#   score(x) = frazione della popolazione z (i non membri dello stesso client e
#           round, x escluso) con LR(x) / LR(z) >= gamma.
# Con gamma = 1 lo score e' monotono in LR(x): l'AUC coincide con quella di
# LR(x), il confronto con z serve per TPR a FPR bassi e per gamma != 1.
# Composizione su piu' round: somma di log LR(x) per campione.
RMIA_A_OFFLINE = 0.3
RMIA_GAMMA = 1.0


def _rmia_scores(rows: list[dict[str, Any]], gamma: float = RMIA_GAMMA,
                 a: float = RMIA_A_OFFLINE) -> dict[str, list[tuple]]:
    """Score RMIA per campione. rows: dict con target, refs (lista di
    (indice shadow, loss, x nel training dello shadow)), member, client,
    canary, sid. Restituisce {"online": [...], "offline": [...]},
    liste di (sid, member, canary, score, log_lr)."""
    from collections import defaultdict as _dd
    per_client: dict[Any, list[dict[str, Any]]] = _dd(list)
    for r in rows:
        per_client[r["client"]].append(r)
    out: dict[str, list[tuple]] = {"online": [], "offline": []}
    lg = math.log(gamma)
    for _cid, rr in per_client.items():
        t = np.array([r["target"] for r in rr], dtype=float)
        tau = max(float(np.median(t)), 1e-12)
        lp_t = -t / tau
        # scala di ogni shadow: mediana delle sue loss su questi campioni
        per_mod: dict[Any, list[float]] = _dd(list)
        for r in rr:
            for (si, mse, _inn) in r["refs"]:
                per_mod[si].append(mse)
        tau_m = {si: max(float(np.median(v)), 1e-12) for si, v in per_mod.items()}

        def _lmean(refs: list[tuple]) -> float:
            v = np.array([-mse / tau_m[si] for (si, mse, _inn) in refs], dtype=float)
            m = float(v.max())
            return m + math.log(float(np.mean(np.exp(v - m))))

        lp_on = np.array([_lmean(r["refs"]) if r["refs"] else np.nan for r in rr])
        lp_off = np.array([
            math.log(0.5 * ((1 + a) * math.exp(_lmean([x for x in r["refs"] if not x[2]])) + (1 - a)))
            if any(not x[2] for x in r["refs"]) else np.nan for r in rr])
        memb = np.array([bool(r["member"]) for r in rr])
        for kind, lp_ref in (("online", lp_on), ("offline", lp_off)):
            llr = lp_t - lp_ref
            ok = np.isfinite(llr)
            pop = np.sort(llr[ok & ~memb])
            for i, r in enumerate(rr):
                if not ok[i] or len(pop) == 0:
                    continue
                cnt = int(np.searchsorted(pop, llr[i] - lg, side="right"))
                n_pop = len(pop)
                if not memb[i]:
                    # x stesso e' nella popolazione: si esclude
                    if llr[i] - lg >= llr[i]:
                        cnt -= 1
                    n_pop -= 1
                if n_pop <= 0:
                    continue
                out[kind].append((r["sid"], bool(r["member"]), r["canary"], cnt / n_pop, float(llr[i])))
    return out


def _rmia_round_fields(rows: list[dict[str, Any]], cumulative: dict[int, float],
                       cum_member: dict[int, bool], cum_canary: dict[int, Any]) -> dict[str, Any]:
    """Campi RMIA di un round per lira_results; aggiorna la composizione."""
    from sklearn.metrics import roc_auc_score as _auc
    res: dict[str, Any] = {}
    sc = _rmia_scores(rows)
    for kind, pref in (("online", "rmia"), ("offline", "rmia_offline")):
        lst = sc[kind]
        labels = [1 if m else 0 for (_s, m, _c, _sc, _l) in lst]
        scores = [s for (_s, _m, _c, s, _l) in lst]
        res[f"{pref}_n_samples"] = len(lst)
        if len(set(labels)) == 2:
            res[f"{pref}_auc_roc"] = round(float(_auc(labels, scores)), 6)
            for k, v in _tpr_at_fixed_fpr(labels, scores).items():
                res[f"{pref}_{k}"] = v
            res[f"{pref}_advantage"] = _mia_advantage(labels, scores)
        else:
            res[f"{pref}_auc_roc"] = None
        can = [(m, s) for (_s, m, c, s, _l) in lst if c is not None]
        cl = [1 if m else 0 for m, _ in can]
        res[f"canary_{pref}_auc_roc"] = (round(float(_auc(cl, [s for _, s in can])), 6)
                                         if len(set(cl)) == 2 else None)
        if kind == "online":
            for (sid, m, c, _s, l) in lst:
                cumulative[sid] = cumulative.get(sid, 0.0) + l
                cum_member[sid] = m
                cum_canary[sid] = c
    return res


def run_lira(
    cfg: dict,
    train_sessions: list[dict[str, Any]],
    holdout_sessions: list[dict[str, Any]],
    fl_results: dict[int, dict[str, Any]],
    n_shadow: int = 8,
    shadow_epochs_cap: int | None = None,
    no_dp: bool = False,
    dp_mode: str = "dp-fedavg",
    cluster_membership: dict[str, list[int]] | None = None,
    composed_output: dict[str, Any] | None = None,
    capture_shadow_weights: bool = False,
    controlled_composition: bool = False,
    per_sample_dump_path: str | None = None,
    roc_curve_dump_path: str | None = None,
    raw_loss_dump_path: str | None = None,
) -> dict[int, dict[str, Any]]:
    """
    LiRA — Likelihood Ratio Attack, server-side, on each client's per-round update
    as actually submitted for aggregation (Carlini et al., IEEE S&P 2022).

    Threat model: semi-honest aggregator receives each client's local model update
    and runs MIA before aggregating them. This is stronger than attacking the
    global model, because FedAvg averaging destroys per-cluster memorisation that
    is still present in the individual client updates.

    Attack flow:
        1. Reconstruct per-cluster session assignment (deterministic — must match
           run_fl_rounds() to identify which sessions each client trained on).
        2. Fix, once, the IN/OUT sample assignment for n_shadow PER-CLUSTER shadow
           models (attacker's fixed auxiliary knowledge: which subset of that
           cluster's sessions each shadow "would have seen").
        3. For EACH ROUND, retrain every shadow model warm-started from that
           round's real global weights (the same starting point real clients use)
           for local_epochs epochs on its fixed IN subset, then — if DP is enabled
           — apply the SAME clip+noise privatisation a real client's update
           receives (see "Fix — DP must be observable" below).
        4. For each round and each client's submitted update:
           a. Load client's update weights.
           b. Compute target_loss = MSE(client_model, x) for every eval sample.
           c. Split THAT CLIENT'S CLUSTER shadow losses (this round) into IN
              (shadow saw x) and OUT (didn't see x).
           d. score(x) = log P(loss | IN dist) − log P(loss | OUT dist)  [Gaussian log-LR]
        5. Pool scores across all clients per round → AUC-ROC.

    Fix — shadow/target distribution mismatch (2026-07-21a):
        Shadow models were trained on random 50% subsets drawn from ALL
        train_sessions (all 4 clusters mixed), while the attacked model is a
        per-cluster specialist trained only on its own cluster's ~2600 sessions.
        This violates LiRA's core assumption (shadow ≈ target's training
        distribution) and produced a systematically INVERTED score (lira_auc_roc
        as low as 0.14–0.32, lira_non_member_score_mean saturating near +20).
        Fix: one shadow ensemble PER CLUSTER, sampled only from that cluster's
        own index range.

    Fix — shadow/target TRAINING PROCEDURE mismatch (2026-07-21b):
        The per-cluster fix alone was NOT sufficient: round 1 improved
        (0.558→0.75) but rounds 2–10 stayed broken and flat (~0.26), confirmed on
        10/10 runs across nodp-sweep1 + dp-sweep1. Root cause: from round 2
        onward, a real client does NOT train from scratch — it starts from the
        previous round's shared GLOBAL weights and does only local_epochs (50)
        epochs of local fine-tuning. The shadow ensemble, however, was trained
        ONCE, from random init, for a fixed 250-epoch budget — a completely
        different trajectory. Round 1 (no shared init yet) had no such mismatch,
        which is why it alone looked healthy.
        Fix: shadows are now retrained EVERY ROUND, warm-started from
        fl_results[round-1]["global_weights"] (round 1 uses random init, matching
        what real clients do), then fine-tuned for exactly local_epochs epochs —
        mirroring the real per-round client procedure. The IN/OUT sample
        assignment per shadow stays fixed across rounds (fixed auxiliary
        knowledge); only the model weights are retrained each round.

    Fix — DP must be observable by the attack it's supposed to defend against
    (2026-07-21c):
        LiRA previously attacked `raw_updates`, captured in run_fl_rounds()
        BEFORE gm.privatize() is called. Since privatize() always returns a new
        object (never mutates its input), raw_updates NEVER carries DP noise,
        regardless of --no-dp / --epsilon. Verified empirically: lira_auc_roc at
        round 1 was bit-for-bit identical between nodp-sweep1 and dp-sweep1 for
        matching seeds (e.g. seed=42 → 0.750451 in both). By construction, DP
        could never suppress this attack — which defeats the purpose of the
        no-DP vs DP comparison (the goal is to measure how much DP degrades each
        attack's strength).
        Fix: LiRA now attacks `updates` (the actual per-client update submitted
        for aggregation — post-privatize when DP is enabled, identical to raw
        when --no-dp). Shadows are privatised with the same GradientManager
        clip+noise procedure when DP is enabled, so the IN/OUT calibration
        reflects the same noise regime the target went through — otherwise a
        noisy target would be miscalibrated against clean shadows, which is a
        mismatch of its own kind.

    Non-member handling:
        Non-members are NEVER in any shadow model's training set → in_losses is
        always empty. We fall back to the PER-CLUSTER, PER-ROUND global IN
        distribution (pooled from that cluster's members' IN shadow losses this
        round) as the IN reference. Non-members' target_loss is typically high
        (model never saw them), so log_p_in << log_p_out → negative score → correct.

    Fix — asymmetric IN/OUT variance estimation for non-members (2026-07-21e):
        Discovered from real sweep data (nodp-sweep2, seeds 42/123, fixes a-d
        already applied): lira_non_member_score_mean spiked to +17.9 at round 2
        (clip ceiling is ±20) then decayed slowly toward +1.4 by round 10, while
        lira_member_score_mean stayed flat near 0 throughout — an AUC inversion
        (0.15-0.27 at rounds 2-4) that is NOT explained by fixes a-d.
        Root cause: for a genuine non-member, out_losses is the per-sample,
        cross-shadow spread over just n_shadow (~8) values on that ONE point.
        Right after a shared warm-start (round >= 2), all shadows for a cluster
        start from the identical global_weights and have only briefly diverged,
        so on a point none of them trained on they produce nearly identical
        reconstructions → σ_out collapses toward the μ×0.05 floor. Meanwhile the
        IN side for a non-member uses the pooled, cluster-wide
        global_in_stats_per_cluster fallback (hundreds of samples, naturally
        much wider σ). Comparing a near-collapsed per-point σ_out against a wide
        pooled σ_in makes the Gaussian log-likelihood-ratio dominated by the
        1/σ² term rather than by genuine membership signal. Member scoring does
        NOT show this because both its IN and OUT sides are already
        per-sample/small-N (symmetric, so the collapse — when it happens —
        affects both terms similarly and partly cancels).
        Fix: compute an analogous pooled, per-cluster/per-round GLOBAL OUT
        distribution (global_out_stats_per_cluster — same construction as
        global_in_stats_per_cluster, pooling the complementary OUT
        observations) and use it as a floor for σ_out — and, symmetrically, for
        σ_in — so neither side's variance can collapse below what is typically
        observed across the whole cluster this round. μ_in/μ_out (the actual
        discriminative signal) are untouched by this fix; only the
        variance floor changes.
        Contributing factor: n_shadow=8 ("fast demo" per config/experiment.yaml)
        makes any per-sample cross-shadow variance estimate inherently noisy;
        the config itself recommends 16-32 for "paper quality" — increasing
        n_shadow would also reduce reliance on this floor.

    Fix — asimmetria strutturale IN/OUT σ floor sotto --no-dp (2026-08-15,
    CONFERMATA con test diretto, non più solo ipotesi):
        nodp-sweep3 (post fix 2026-08-11/12) mostrava member e non-member
        score muoversi INSIEME invece di separarsi (salto round1→round2 da
        ~+3 a ~-6.5, poi decadimento verso 0). Test di conferma (2026-08-15,
        sweep breve --no-dp, 3 round, seed 42, experiments/_diag_nodp_sigma):
        `lira_debug_raw_loss_gap` è risultato PICCOLO ma COERENTE IN SEGNO
        su tutti e 3 i round (-0.00029, -0.00037, -0.00038 — member sempre
        con loss più bassa, la direzione corretta per vera memorizzazione)
        — quindi NON uno Scenario B genuino (il modello non memorizza
        affatto). Il problema è invece che `lira_debug_sigma_out_mean`
        (~0.016-0.018) è risultato sistematicamente ~30× più grande di
        `lira_debug_sigma_in_mean` (~0.0005-0.0011), con
        `lira_debug_sigma_out_floor_hit_rate`≈0.993-0.998 (quasi sempre il
        fallback pooled vince) contro `lira_debug_sigma_in_floor_hit_rate`
        ≈0.49-0.50. Causa: `global_in_stats_per_cluster` pooling SOLO le
        coppie (membro, shadow che l'ha visto) — una popolazione ristretta e
        omogenea — mentre `global_out_stats_per_cluster` pooling OGNI
        non-membro (sempre OUT per costruzione, per tutti gli shadow) — una
        popolazione molto più ampia e variabile tra sessioni diverse. Non un
        collasso del floor assoluto 1e-4 (che infatti non è mai il termine
        dominante osservato) — un disallineamento strutturale tra due
        popolazioni pooled di scala diversa. Conseguenza: log_p_in (diviso
        per un σ minuscolo) domina il punteggio ed è instabile, mascherando
        il segnale vero ma piccolo. Confermato anche da
        `lira_debug_raw_mse_auc_roc` (AUC calcolata direttamente sulla MSE
        grezza, bypassando la normalizzazione log-ratio) nello stesso test.
        Fix implementato: `_cluster_sigma_symmetric_floor` (il maggiore tra
        `_cluster_sigma_in_fb` e `_cluster_sigma_out_fb`, vedi dove è
        calcolato e applicato più sotto nel corpo della funzione) — floor
        CONDIVISO applicato simmetricamente a σ_in e σ_out. Non tocca
        μ_in/μ_out (il segnale discriminativo, invariato) e non cambia il
        lato che aveva già il floor più alto (resta esattamente come prima);
        alza solo il lato strutturalmente più piccolo. ATTENZIONE: si applica
        a OGNI esperimento LiRA, non solo al no-DP — invalida i numeri di
        central-sweep/dp-sweep/local-sweep raccolti finora (decisione
        esplicita dell'utente, 2026-08-15). Vedi Task #1 (già in sospeso):
        l'intera campagna sperimentale va ripetuta con questo fix applicato,
        non solo con i due fix di pooling cross-cluster di agosto.

    Fix — campioni "non calibrabili" esclusi dal punteggio (2026-08-20,
    stesso giorno, terzo round di questa indagine):
        Dopo il fix del floor simmetrico sopra, la campagna completa (5 seed
        × 7 config, 2026-08-16/20) ha mostrato nodp-sweep1 stabilmente e
        SIGNIFICATIVAMENTE sotto 0.5 (CI bootstrap [0.3675, 0.3742]) mentre
        `lira_debug_raw_mse_auc_roc` restava ~0.50 in ogni round/seed —
        un'inversione forte e riproducibile, non spiegabile dal solo
        controllo aggregato di `lira_debug_mu_in_minus_out` (che infatti
        risultava CORRETTAMENTE negativo, μ_in<μ_out, smentendo l'ipotesi
        iniziale di "warm-start destabilizzato"). Un dump per-campione
        (t, μ_in, μ_out, σ_in, σ_out, log_p_in, log_p_out, score — vedi dove
        è loggato più sotto nel corpo della funzione) ha rivelato la causa
        reale: alcune sessioni hanno target_loss REALE fino a ~20 deviazioni
        standard lontano da ENTRAMBE μ_in e μ_out (es. t=0.0279 contro
        μ_in=0.0007, μ_out=0.0009, σ=0.0013) — né l'ensemble shadow IN né
        quello OUT spiegano minimamente quell'osservazione. In questo regime
        il segno di log_p_in−log_p_out è deciso da quale delle due medie
        minuscole è per puro caso una frazione più vicina a un valore
        comunque enorme e lontanissimo da entrambe — rumore geometrico col
        segno arbitrario, non segnale di membership. Bastano 2-3 di questi
        outlier su 15 campioni per trascinare la media del round di diversi
        punti interi, dominando l'intero aggregato. Il fenomeno è specifico
        al regime di loss bassissima e σ molto stretto raggiunto solo da
        no-DP (sotto DP attivo, rumore/clipping tengono tutto su una scala
        comparabile — questi casi sono rari o assenti), esattamente come i
        due fix precedenti di questa stessa indagine (floor simmetrico
        2026-08-15, pooling cross-cluster 2026-08-11/12).
        Fix: se un campione è oltre `_UNCALIBRATED_Z_THRESHOLD`=8 deviazioni
        standard da ENTRAMBE le distribuzioni (min dei due |z-score| > 8), il
        modello gaussiano di LiRA non ha alcuna informazione calibrata su di
        esso — viene escluso dal punteggio (skip, non forzato con un segno
        arbitrario), invece di essere incluso e contaminare l'aggregato. Il
        tasso di esclusione è salvato in `lira_debug_uncalibrated_skip_rate`/
        `_skipped_n` per trasparenza — se risultasse alto (>20-30%) andrebbe
        rivisto n_shadow o la costruzione di μ/σ, non solo accettato. Non
        tocca il calcolo di μ_in/μ_out/σ_in/σ_out per i campioni NON esclusi
        — solo filtra quali campioni entrano nel pool finale.
        ATTENZIONE: come i due fix precedenti, questo si applica a OGNI
        esperimento LiRA, non solo al no-DP — invalida ANCHE i numeri della
        campagna 2026-08-16/20 (già post floor-fix). Vedi Task #1.

    Fix — ancoraggio di μ_in dei non-membri al proprio μ_out (2026-08-21,
    stesso giorno del fix precedente, quarto round di questa indagine):
        Con l'esclusione outlier sopra applicata, `_verify_outlier_fix`
        (3 round, seed 42) ha mostrato lira_auc_roc ancora invertito
        (~0.32-0.44) con tasso di esclusione solo del 2.37% — l'ipotesi degli
        outlier a ~20σ era vera ma NON la causa dominante dell'inversione,
        solo un contributo minore. Ri-analizzando lo stesso dump per-campione
        (`_diag_persample`) con questo in mente è emersa la causa reale:
        per costruzione, un non-membro NON ha mai una vera calibrazione IN
        (nessuno shadow retraining lo tratta mai come incluso), quindi il
        codice ricorre al ramo `else` sotto — μ_in diventava la costante
        ASSOLUTA `_cluster_mu_in_fb` (pooled su tutto il cluster/round),
        mentre μ_out restava sempre una stima PER CAMPIONE, sensibile alla
        difficoltà di ricostruzione di quella specifica sessione. Le sessioni
        "facili" (μ_out piccolo, la maggioranza) risultavano quindi più vicine
        alla costante _cluster_mu_in_fb (relativamente più grande, essendo un
        pooled) che al proprio μ_out — cioè log_p_in > log_p_out per un
        artefatto di scala, non per vera membership. Il dump del 2026-08-20
        confermava infatti che la maggioranza dei non-membri con t basso
        (sessioni "facili") aveva score positivo (falso "sembra membro"),
        esattamente il pattern previsto da questa asimmetria costante-vs-
        per-campione.
        Fix: nel ramo `else` (nessuna calibrazione IN reale disponibile),
        μ_in non è più la costante grezza `_cluster_mu_in_fb`, ma viene
        ancorata al μ_out di QUESTO campione specifico, spostato del gap
        TIPICO osservato tra IN e OUT nei membri reali dello stesso
        cluster/round: `μ_in = μ_out + (_cluster_mu_in_fb - _cluster_mu_out_fb)`.
        Così un non-membro "facile" ottiene una stima simulata di μ_in
        coerente con la propria scala di difficoltà (non un valore assoluto
        scollegato), pur mantenendo il vantaggio IN tipico osservato
        empiricamente nel cluster/round. Non tocca il ramo `if` (calibrazione
        IN reale, len(in_losses)>=2), né σ_in/σ_out, né l'esclusione outlier
        del fix precedente — modifica solo la stima puntuale di μ_in quando
        non c'è alternativa migliore.
        ATTENZIONE: come i tre fix precedenti, questo si applica a OGNI
        esperimento LiRA — invalida ANCHE i numeri della campagna
        2026-08-16/20 raccolti dopo il fix dell'esclusione outlier. Vedi
        Task #1: la campagna va ripetuta di nuovo, per la terza volta questa
        settimana, ora con tutti e quattro i fix applicati insieme.

    Fix strutturale — universo shadow simmetrico membri+non-membri
    (2026-08-21, stesso giorno, quinto/sesto round di questa indagine — il
    più profondo di tutti):
        Il test di verifica del fix precedente (`_verify_mu_anchor`, no-DP,
        3 round, seed 42) ha mostrato lira_auc_roc passare da 0.32-0.44
        (invertito) a 0.72-0.82 — molto più alto di raw_mse_auc/Yeom (~0.50).
        Ipotesi: l'ancoraggio crea un'asimmetria strutturale opposta a quella
        appena corretta. Test decisivo, `lira_debug_matched_formula_auc`
        (diagnostica puramente additiva, vedi sua inizializzazione più
        sotto): per ogni membro con calibrazione IN reale, calcola ANCHE lo
        score che avrebbe ricevuto forzato nella stessa formula fallback dei
        non-membri — se l'AUC risultante (stessa formula su entrambi i lati)
        collassa verso 0.5, l'AUC alto visto sopra è un artefatto di formule
        diverse, non segnale reale. Risultato (`_diag_matched_formula`,
        no-DP, 3 round, seed 42): matched_formula_auc = 0.494, 0.485, 0.486
        — collassa esattamente a 0.5, confermando l'artefatto.
        Più grave: lo stesso test ripetuto su `--dp-mode central --epsilon
        1.0` (`_diag_matched_formula_central`, 3 round, seed 42) — la
        configurazione "flagship" del paper (McMahan-style noising non
        sopprime il leakage, storicamente AUC 0.65-0.88 su 5 seed, CI
        bootstrap che escludeva 0.5) — ha mostrato matched_formula_auc
        ANCH'ESSO vicino a 0.5 in ogni round (0.506, 0.498, 0.500), e persino
        lira_auc_roc COMPOSTO (la metrica multi-round statisticamente
        corretta, non la semplice media per-round) è sceso a 0.5022 —
        sostanzialmente al caso, classificato "Privacy risk: LOW" dal
        pipeline stesso. Conclusione: il claim centrale del paper (leakage
        sotto Central DP) era, almeno in parte sostanziale, lo stesso
        artefatto — non specifico al no-DP, non risolto dai tre fix
        precedenti, perché la sua causa non è nella formula del punteggio ma
        nel POOL da cui gli shadow campionano.
        Causa reale (verificata leggendo il codice, non solo ipotizzata):
        `cluster_idx_pools` (Step 1) e quindi il pool da cui ogni shadow
        campiona il proprio subset IN (Step 2, prima del fix) conteneva SOLO
        indici in `train_sessions` — gli `holdout_sessions` (i veri
        non-membri) non potevano MAI comparire nel subset IN di nessuno
        shadow, in nessun round, per costruzione architetturale. Un
        non-membro non può quindi MAI avere una calibrazione IN reale — non
        per una formula sbagliata (i tre fix precedenti hanno tutti provato,
        con successo parziale, ad approssimare meglio quella calibrazione
        mancante), ma perché il pool stesso esclude strutturalmente l'intera
        popolazione dei non-membri. Qualunque approssimazione di μ_in per un
        non-membro resta quindi sistematicamente diversa dalla calibrazione
        REALE usata per i membri (che invece, per costruzione, finiscono IN
        per circa metà degli shadow e OUT per l'altra metà) — e quella
        differenza sistematica, non la vera memorizzazione del modello,
        gonfiava l'AUC.
        Fix: il pool da cui ogni shadow campiona il proprio subset IN è ora
        `cluster_shadow_universe[cid]` — l'unione di `cluster_members[cid]`
        (membri, dati FL reali) e `cluster_holdout[cid]` (non-membri,
        sessioni mai viste dal training FL) dello stesso cluster — con
        assegnazione IN/OUT casuale per ogni shadow, indipendente dalla VERA
        membership rispetto al modello target. È esattamente il disegno
        originale di LiRA (Carlini et al. 2022, si veda anche la nota su
        questa stessa deviazione metodologica in
        docs/ReadingList_DSN2027.md): gli shadow si addestrano su split
        casuali di un pool più ampio del training set del modello attaccato,
        non sul solo insieme di training di quel modello — sia membri sia
        non-membri del target possono quindi finire IN per alcuni shadow e
        OUT per altri, ottenendo ENTRAMBI una calibrazione bidirezionale
        reale. I fix precedenti (floor simmetrico 2026-08-15, esclusione
        outlier 2026-08-20, ancoraggio μ_in 2026-08-21) NON sono stati
        rimossi — restano corretti e necessari per i casi (ora rari per
        ENTRAMBE le classi, non più sistematici per una sola) in cui un
        campione ha comunque troppo poche osservazioni IN reali.
        Dettaglio implementativo: `shadow_in_idx_sets_per_cluster[cid]` ora
        contiene insiemi di id() Python delle sessioni campionate (non più
        posizioni numeriche in `train_sessions`) — il controllo di
        appartenenza (`id(sample) in in_set`) è quindi identico per membri e
        non-membri in ogni punto del codice che lo usa (Step 2, i due pool
        globali di fallback, il loop di scoring per-campione), eliminando
        l'asimmetria alla radice invece di continuare ad approssimarla.
        Attivo solo quando `cluster_membership` è reale — quando è None
        (path storico per i test con dati sintetici) `cluster_holdout`
        resta vuoto per costruzione, l'universo collassa al solo pool
        membri, preservando ESATTAMENTE il comportamento pre-fix per quel
        path (verificato: tutti gli 83 test non-torch passano invariati).
        ATTENZIONE — la più seria di tutta questa catena: invalida OGNI
        numero LiRA mai raccolto in questo progetto, incluso il claim
        flagship "Central DP non sopprime il leakage" (central-sweep1/2,
        AUC storico 0.65-0.88). Task #1 va ripetuto da zero con questo fix;
        la tesi centrale del paper va rivalutata sui numeri reali che ne
        usciranno, non assunta.

    Why this differs from run_yeom / run_shadow:
        Both previous attacks use the GLOBAL aggregated model, which is itself a
        cross-cluster blend — so a cross-cluster, one-shot shadow ensemble is the
        correct reference for those two (no mismatch there; these fixes only
        apply to LiRA). LiRA uses each CLIENT's individual update — the signal
        before FedAvg averaging — which is why its shadows must mirror both the
        per-cluster data AND the per-round training procedure of that client.

    Args:
        cfg:              experiment configuration dict
        train_sessions:    sessions used in FL training (members)
        holdout_sessions:  hold-out sessions never seen by FL (non-members)
        fl_results:        per-round FL data, must contain "updates" and
                           "global_weights" (previous round used as warm-start)
        n_shadow:          number of shadow models PER CLUSTER (8 = fast demo;
                           ≥32 = paper quality)
        shadow_epochs_cap: override for shadow training epochs (default:
                           local_epochs, matching the real client). Use a small
                           value (e.g. 20) only for smoke tests.
        no_dp:             must match the flag used for this experiment — controls
                           whether shadows are privatised like real clients.
        dp_mode:           must match the dp_mode used in run_fl_rounds() for this
                           experiment (2026-07-22) — controls HOW shadows are
                           privatised to mirror the target's actual noise regime:
                             "dp-fedavg"/"local" → gm.privatize() per shadow (clip
                               + noise), matching what a real client submitted.
                             "central" → gm.clip_only() per shadow (clip, NO
                               noise) — matching that under central DP, individual
                               client updates are never noised, only the aggregate
                               is. LiRA on individual updates should therefore show
                               NO suppression under central DP, at any ε — this is
                               the expected empirical result the CS4 comparison
                               (docs/CaseStudies.md §2.4.3) is designed to surface,
                               not a bug to fix.
        cluster_membership: {cluster_id: [indici GLOBALI in train_sessions]}
                           (2026-07-22) — deve rispecchiare ESATTAMENTE i cluster
                           usati da run_fl_rounds() per lo stesso esperimento
                           (es. via group_sessions_by_site() per i 3 siti reali,
                           o inject_synthetic_clients() per i 5 client dello
                           sweep IDS). Se None (default, retrocompatibile coi
                           test con dati sintetici senza site_id reale), affetta
                           train_sessions in 4 parti contigue uguali — stesso
                           comportamento storico pre-2026-07-22.
        capture_shadow_weights: (Sprint 10zz, 2026-09-01) default False → ZERO
                           impatto sul comportamento esistente (nessun nuovo
                           codice eseguito nel path di default). Se True,
                           registra anche il vettore di peso COMPLETO
                           (flatten di state_dict().values(), nessun
                           troncamento) di ogni shadow già addestrato in
                           questa funzione, per round/cluster, sotto
                           lira_results[r]["_fedmia_shadow_weights"] — usato
                           SOLO da run_fedmia_gradient() per riusare
                           l'ensemble shadow di LiRA (universo membri+
                           non-membri, warm-start, privatizzazione DP — tutti
                           già validati sopra) invece di riaddestrarne uno
                           parallelo con una calibrazione a rumore gaussiano
                           come faceva la classe FedMIA originale (vedi
                           src/plugins/attacks/fedmia.py).
        controlled_composition: (Sprint 10zz+5, 2026-09-01) default False →
                           ZERO impatto sul comportamento esistente (LiRA
                           reale, quello registrato in ATTACK_REGISTRY, non
                           passa mai True qui). Se True, cambia SOLO come
                           viene scelto il subset IN di ogni shadow al Step 2
                           sotto: invece di campionare uniformemente
                           dall'universo membri+non-membri combinato (il
                           comportamento di LiRA, deliberato — rispecchia la
                           composizione reale), ogni shadow riceve una
                           frazione-membri BERSAGLIO, distribuita
                           uniformemente su [0.1, 0.9] in base al suo indice.
                           Trovato necessario (2026-09-01, da un run reale):
                           con l'universo storico ~80:20 membri:non-membri,
                           un campione ampio (metà universo) converge quasi
                           deterministicamente vicino a quella proporzione —
                           uno shadow "a maggioranza non-membro" è
                           essenzialmente impossibile per pura varianza
                           campionaria su cluster grandi (caltech/jpl,
                           decine di migliaia di sessioni), a QUALUNQUE
                           n_shadow. run_fedmia_gradient() (unico chiamante
                           di questo flag) ha bisogno di entrambe le classi
                           ben rappresentate per calibrare/valutare — LiRA
                           stesso non lo richiede mai (il suo disegno
                           originale, campionamento neutro, resta invariato
                           quando controlled_composition=False, cioè
                           sempre, per ogni chiamante reale).

    Returns:
        {round_num: {
            "lira_auc_roc":               float,
            "lira_member_score_mean":     float,
            "lira_non_member_score_mean": float,
            "lira_score_gap":             float,
            "n_shadow":                   int,
            "tpr_at_fpr_0.001":           float | None,  # Sprint 10pp 2026-08-28
            "tpr_at_fpr_0.01":            float | None,  # — vedi _tpr_at_fixed_fpr()
            "tpr_at_fpr_0.05":            float | None,
            "lira_advantage":             float | None,  # task #41, Sprint 10zz+13
            "canary_advantage":           float | None,  # 2026-09-02 — vedi _mia_advantage()
            "lira_confusion":             dict | None,   # task #49, Sprint 10zz+25 — vedi
            "canary_confusion":           dict | None,   # _mia_confusion_at_best_threshold()
        }}
        composed_output (se fornito) riceve anche gli stessi tre campi TPR@FPR
        calcolati sul punteggio cumulativo multi-round, sotto chiavi
        "composed_tpr_at_fpr_0.001"/"...0.01"/"...0.05" (prefisso "composed_"
        dal Sprint 10zz+41, 2026-09-04, task #66 — prima erano bare, identiche
        a quelle del solo ultimo round: il merge in
        src/plugins/attacks/lira.py sovrascriveva silenziosamente il
        tpr_at_fpr_* del round finale col valore composto, bug live dal
        Sprint 10pp), più "composed_lira_advantage"/"canary_composed_advantage"
        (task #41) e "composed_lira_confusion"/"canary_composed_confusion"
        (task #49).
        *_confusion è un dict {"threshold", "advantage", "tp", "fp", "tn",
        "fn", "n_members", "n_nonmembers"} (o tutti None se non calcolabile)
        — conteggi assoluti alla soglia Youden-ottimale, complementari al
        tasso già dato da *_advantage.
        NOTA: lira_advantage/canary_advantage/composed_lira_advantage/
        *_confusion sono calcolabili SOLO per run eseguiti dopo questa
        modifica (2026-09-02/03) — richiedono la curva ROC completa, mai
        salvata nei JSON storici. Per PES v1/v1.1 su dati già esistenti,
        vedi scripts/compute_pes.py.

        per_sample_dump_path (Sprint 10zz+27, 2026-09-03, task worst-case):
        se fornito INSIEME a composed_output (entrambi richiesti — opt-in
        doppio, default None su entrambi, zero impatto su ogni chiamante
        esistente), scrive un JSON con un record per campione realmente
        scorato (session_id reale da ACNDataset, is_member, is_canary,
        composed_score) — NON un aggregato, il punteggio di OGNI singolo
        campione. Costruito per scripts/analyze_worst_case_vulnerability.py:
        confrontando i dump di seed diversi per la STESSA config, verifica
        se specifici record reali sono ripetutamente ad alta confidenza
        attraverso seed indipendenti (vulnerabilità worst-case genuina) o
        se il tail della distribuzione cambia record ogni volta (rumore,
        coerente con l'assenza di leakage già trovata a livello di
        popolazione — vedi docs/DSN2027_Positioning.md, sezione "Worst-case
        vs. average-case privacy evaluation"). session_id assente/None nel
        dataset sorgente → record escluso dall'analisi cross-seed (loggato
        come warning, non un errore).
    """
    from sklearn.metrics import roc_auc_score

    from ml.base_ml import GradientUpdate as _GU

    seed         = cfg.get("experiment", {}).get("seed", 42)
    input_dim    = cfg["ml"]["input_dim"]
    ml_cfg       = cfg["ml"]
    exp_cfg      = cfg.get("experiment", {})
    lr           = ml_cfg.get("lr", 1e-3)
    batch_size   = ml_cfg.get("batch_size", 32)
    local_epochs = ml_cfg.get("epochs", 50)
    # Epoche di training per gli shadow ad ogni round: di default = local_epochs,
    # esattamente come i client reali (fix 2026-07-21b). shadow_epochs_cap permette
    # di ridurle per gli smoke test (es. 20) senza alterare il regime dei run reali.
    shadow_epochs = shadow_epochs_cap if shadow_epochs_cap is not None else local_epochs

    # Floor simmetrico σ_in/σ_out (Sprint 10zz+87, 2026-09-14) — flag diagnostico
    # opt-in, su richiesta esplicita dell'utente dopo un feedback esterno
    # verificato: lira_debug_matched_formula_auc mostra che su `central`
    # (floor-hit-rate 95-99%) il punteggio calibrato collassa algebricamente
    # nella forma di _sablayrolles_score() quando σ_in=σ_out=floor condiviso
    # (vedi _cluster_sigma_symmetric_floor sotto), e il "null result" su
    # central potrebbe essere la cancellazione tra un'inversione reale
    # (formula-matched, 0.21-0.34) e l'ancoraggio μ_in dei non-membri, non
    # un'assenza genuina di segnale. Default "symmetric" = comportamento
    # ESATTAMENTE invariato (fix 2026-08-15, mai toccato) per ogni config/run
    # esistente — zero impatto se questa chiave non è presente. Se
    # "independent", _cluster_sigma_symmetric_floor viene posto a 0.0 più
    # sotto (un solo punto di modifica, riga dove è calcolato) invece di
    # max(sigma_in_fb, sigma_out_fb) — ciascun lato torna al proprio floor
    # indipendente (max_scala/1e-4/fallback pooled), riproducendo il regime
    # PRE-fix che aveva il problema opposto (σ_out_fb ~30x più grande di
    # σ_in_fb, instabilità round-a-round) — usare SOLO per il confronto
    # diagnostico A/B richiesto (floor-hit-rate/matched_formula_auc/lira_auc
    # prima e dopo), non come nuovo default di produzione senza prima aver
    # verificato se n_shadow=16 è sufficiente a stimare σ_in per-campione
    # senza affidarsi a un floor condiviso.
    _lira_floor_mode = cfg.get("lira", {}).get("floor_mode", "symmetric")
    if _lira_floor_mode not in ("symmetric", "independent"):
        raise ValueError(
            f"cfg['lira']['floor_mode'] non valido: {_lira_floor_mode!r} "
            "(atteso 'symmetric' o 'independent')"
        )

    # Observation surface: client (default) vs global (Sprint 10zz+94, 2026-09-15) —
    # opt-in richiesto esplicitamente dall'utente per "chiudere il cerchio" tra i tre
    # attacchi: Yeom/Shadow attaccano round_data["global_weights"] (il modello
    # aggregato, superficie "global"), LiRA attacca invece per-client
    # round_data["updates"]/["raw_updates"] (superficie "client", INVARIATA, resta il
    # default). "global" fa attaccare a LiRA lo STESSO modello aggregato di
    # Yeom/Shadow, riusando comunque la calibrazione shadow esistente per-cluster
    # (μ_in/μ_out/σ_in/σ_out) — cambia SOLO la fonte di target_loss (un modello
    # condiviso per round invece del modello specifico del client), non quali
    # campioni vengono valutati né come vengono calibrati. Vedi il punto di
    # applicazione più sotto (subito prima del loop `for _client_idx, update in
    # enumerate(client_updates)`) per il motivo per cui questo è un cambiamento
    # chirurgico: la selezione dei membri per cluster e tutta la diagnostica restano
    # identiche, cambia solo quale modello produce target_loss.
    # Attaccante che riscala l'update osservato (2026-10-09, Sprint 10zz+170,
    # --canary-rescale): lista di fattori s; vuota = disattivato, nessun costo.
    # Solo sui canary e solo con observation_surface == "client".
    _canary_rescale: list[float] = [
        float(_s) for _s in (cfg.get("lira", {}).get("canary_rescale") or [])
    ]
    _lira_observation_surface = cfg.get("lira", {}).get("observation_surface", "client")
    if _lira_observation_surface not in ("client", "global"):
        raise ValueError(
            f"cfg['lira']['observation_surface'] non valido: {_lira_observation_surface!r} "
            "(atteso 'client' o 'global')"
        )

    # Shadow init: warm-start vs cold-start (Sprint 10zz+88, 2026-09-15) — flag
    # diagnostico opt-in, Blocker 1 del feedback esterno verificato (errata
    # punto §5 "Cosa farei ora"): la nostra Adattamento #1 a §3.5 (shadow
    # retrained OGNI round, warm-started dai pesi globali reali di quel round)
    # è una modifica sostanziale rispetto alla costruzione originale di
    # Carlini et al. (shadow addestrati indipendentemente, una sola volta, da
    # init casuale, su sottoinsiemi partizionati casualmente) — mai isolata
    # con un'ablation. Questo flag isola SOLO la variabile warm-start/cold-init
    # mantenendo invariata la cadenza "retrain ogni round" (già di per sé una
    # necessità dovuta al fatto che un client FL reale continua ad allenarsi
    # round dopo round, non un artefatto arbitrario) — un'ablation più fedele
    # alla costruzione originale (shadow addestrati UNA SOLA volta e mai più
    # ri-addestrati) richiederebbe un cambiamento strutturale più ampio, non
    # fatto qui. Default "warm" = comportamento ESATTAMENTE invariato per ogni
    # config/run esistente. Se "cold", più sotto _warm_start viene forzato a
    # None ad ogni round (init casuale ad ogni retrain, mai i pesi globali del
    # round precedente) — isola se è il warm-start stesso, e non l'assunzione
    # Gaussiana o il floor di varianza, a produrre il null result su LiRA.
    _lira_shadow_init = cfg.get("lira", {}).get("shadow_init", "warm")
    if _lira_shadow_init not in ("warm", "cold"):
        raise ValueError(
            f"cfg['lira']['shadow_init'] non valido: {_lira_shadow_init!r} "
            "(atteso 'warm' o 'cold')"
        )

    # Simmetria dello scoring μ_in membri vs non-membri (Sprint 10zz+90,
    # 2026-09-15) — flag diagnostico opt-in, feedback esterno verificato,
    # errata punto #3 ("Il valore di `central` è la cancellazione di due
    # artefatti"): per i MEMBRI, μ_in è quasi sempre una stima REALE
    # per-campione (calibrazione shadow effettiva, ramo `if len(in_losses) >=
    # 2` sopra); per i NON-membri, μ_in è invece SEMPRE la formula di
    # ancoraggio `μ_out + (μ_in_fb - μ_out_fb)` (nessuno shadow si allena mai
    # su dati hold-out, quindi il ramo `else` è l'UNICO percorso possibile per
    # loro). Il punteggio ufficiale confronta quindi due classi con
    # informazione strutturalmente diversa. `lira_debug_matched_formula_auc`
    # (sopra) misura già cosa succede forzando ANCHE i membri nella stessa
    # formula di ancoraggio (via `_diag_counterfactual_member_scores`) — ma
    # solo come diagnostica di sola lettura, mai usata per il punteggio
    # effettivo (vedi commento lì: "non modifica μ_in/σ_in/log_p_* reali").
    # Questo flag PROMUOVE quella stessa formula a percorso di scoring
    # ufficiale per i membri quando è "matched_formula", rendendo il
    # confronto IN/OUT simmetrico per costruzione. Default "real" =
    # comportamento ESATTAMENTE invariato per ogni config/run esistente (i
    # membri continuano a usare la loro calibrazione reale quando disponibile,
    # come sempre). Confrontare i due regimi con `compare_floor_mode.py`
    # (stesso schema before/after già usato per floor_mode/shadow_init) dice
    # se il ~0.50 ufficiale su `central` sopravvive anche a scoring simmetrico
    # o è, come sospettato nell'errata, la cancellazione di due artefatti
    # asimmetrici.
    _lira_member_scoring = cfg.get("lira", {}).get("member_scoring", "real")
    if _lira_member_scoring not in ("real", "matched_formula"):
        raise ValueError(
            f"cfg['lira']['member_scoring'] non valido: {_lira_member_scoring!r} "
            "(atteso 'real' o 'matched_formula')"
        )

    # GradientManager per privatizzare gli shadow ESATTAMENTE come i client reali
    # (stesso clipping + stesso meccanismo di rumore) — fix 2026-07-21c: senza
    # questo, un target rumoroso (DP on) verrebbe calibrato contro shadow puliti,
    # un mismatch che si aggiungerebbe a quelli già corretti sopra.
    gm = GradientManager({
        "epsilon":       exp_cfg.get("epsilon", 1.0),
        "delta":         exp_cfg.get("delta", 1e-5),
        "max_grad_norm": exp_cfg.get("max_grad_norm", 1.0),
    })

    # ── Step 1: Reconstruct per-cluster membership — must match run_fl_rounds() ─
    # FIX 2026-07-22 (coerente col fix di run_nvflare_mia.py/group_sessions_by_site()):
    # cluster_idx_pools ora è {cluster_id: [indici GLOBALI in train_sessions]} —
    # se cluster_membership è fornito (3 siti reali o 5 per lo sweep IDS), viene
    # usato direttamente; altrimenti fallback storico (4 fette contigue fittizie,
    # per compatibilità coi test con dati sintetici senza site_id).
    if cluster_membership is not None:
        _CLUSTER_IDS = list(cluster_membership.keys())
        cluster_idx_pools: dict[str, list[int]] = cluster_membership
    else:
        _CLUSTER_IDS = ["highway", "urban", "residential", "corporate"]
        cluster_size = max(1, len(train_sessions) // len(_CLUSTER_IDS))
        cluster_idx_pools = {}
        for i, cid in enumerate(_CLUSTER_IDS):
            start = i * cluster_size
            end   = len(train_sessions) if i == len(_CLUSTER_IDS) - 1 else start + cluster_size
            cluster_idx_pools[cid] = list(range(start, end))

    cluster_members: dict[str, list[dict[str, Any]]] = {
        cid: [train_sessions[i] for i in idxs] for cid, idxs in cluster_idx_pools.items()
    }

    # Reverse map: sample Python id → cluster (for correct client-member matching).
    # Members must be evaluated ONLY against their home cluster's client.
    # Cross-cluster evaluation (highway sample vs urban client) gives high target_loss
    # → looks like a non-member → contaminates member pool with false negatives
    # → AUC drops below 0.5 when 3/4 of evaluations are cross-cluster (4 clusters).
    _sample_to_cluster: dict[int, str] = {
        id(s): cid
        for cid, sessions in cluster_members.items()
        for s in sessions
    }

    # Fix 2026-08-11: simmetrico a _sample_to_cluster ma per i NON-member
    # (holdout_sessions). Trovato investigando un'anomalia reale: il nodp-sweep1
    # in corso mostrava lira_score_gap CRESCENTE (separazione IN/OUT reale e
    # forte) ma AUC-ROC pooled piatto vicino al caso — la firma di un pool
    # eterogeneo, non di segnale assente. Causa: il guard cross-cluster sopra
    # (poche righe più in basso, "if is_member: ... skip mismatched pairs") è
    # applicato SOLO ai member — ogni non-member viene invece valutato contro
    # TUTTI i client/cluster senza restrizione, quindi entra fino a 3 volte in
    # round_nonmember_scores, mescolando client con scale di loss potenzialmente
    # diverse. Sotto DP con clipping per-client (central/dp-fedavg/local) tutti
    # i client sono forzati in una scala di delta comparabile (max_grad_norm),
    # quindi la mescolanza è quasi inerte — ma sotto --no-dp, senza alcun vincolo
    # di norma, il client più piccolo (office1, ~1341 sessioni contro
    # ~25-27k di caltech/jpl) può divergere di scala in modo scorrelato
    # round-per-round, e quel rumore extra nel pool diluisce un ranking che a
    # livello per-cluster è in realtà ben calibrato. Fix: ogni non-member ha
    # comunque un site_id reale proprio (proviene da una site FL prima dello
    # split, non "non è di nessun sito") — usiamo la stessa risoluzione
    # _SITE_ID_TO_NAME di group_indices_by_site() per restringere anche i
    # non-member al proprio cluster di origine, esattamente come i member.
    # Guard applicato SOLO quando cluster_membership è reale (3+ siti reali/
    # sweep IDS n=5) — quando è None (fallback storico a 4 fette fittizie,
    # usato dai test con dati sintetici senza site_id reale), holdout_sessions
    # non è mai stato partizionato in quelle 4 fette fittizie: risolvere
    # site_id qui darebbe "unknown" per ogni non-member, mai uguale a una delle
    # 4 fette fittizie lato client → guard sempre vero → round_nonmember_scores
    # vuoto per ogni round → stesso genere di regressione silenziosa che questo
    # fix vuole eliminare, ma sui test invece che sui dati reali. Dict vuoto in
    # quel caso preserva esattamente il comportamento pre-fix (nessuna restrizione
    # sui non-member), identico a prima per tutta la suite di test esistente.
    _holdout_sample_to_cluster: dict[int, str] = {
        id(s): _SITE_ID_TO_NAME.get(s.get("site_id", ""), s.get("site_id", "") or "unknown")
        for s in holdout_sessions
    } if cluster_membership is not None else {}

    # Fix strutturale 2026-08-21 (quinto/sesto round di questa stessa indagine
    # — il più profondo dei fix di questa catena, vedi docstring "Fix —
    # universo shadow simmetrico" più sotto per l'analisi completa). I
    # non-membri qui sotto vengono raggruppati per cluster ESATTAMENTE come i
    # membri sopra (`cluster_members`), per costruire più sotto un pool
    # UNICO membri+non-membri da cui ogni shadow campiona il proprio subset
    # IN — invece del solo pool membri usato finora, che rendeva impossibile
    # per un non-membro avere MAI una calibrazione IN reale, qualunque
    # formula di fallback si usasse per approssimarla. Attivo solo quando
    # cluster_membership è reale — stessa condizione di _holdout_sample_to_cluster
    # sopra, per lo stesso motivo (senza site_id reale, "unknown" per ogni
    # non-member non permetterebbe un raggruppamento per cluster sensato).
    cluster_holdout: dict[str, list[dict[str, Any]]] = {cid: [] for cid in _CLUSTER_IDS}
    if cluster_membership is not None:
        for s in holdout_sessions:
            _hc = _holdout_sample_to_cluster.get(id(s))
            if _hc in cluster_holdout:
                cluster_holdout[_hc].append(s)
            # non-member il cui site_id non risolve a nessuno dei cluster noti
            # (dato reale ma sito non mappato in _SITE_ID_TO_NAME) — esclusa
            # dall'universo shadow per quel cluster, non forzata altrove.

    # Balanced eval pool: subsample members to match hold-out size
    _pool_rng      = random.Random(seed + 31415)
    _n_bal         = min(len(train_sessions), len(holdout_sessions))
    members_bal    = _pool_rng.sample(train_sessions, _n_bal)
    nonmembers_bal = _pool_rng.sample(holdout_sessions, min(_n_bal, len(holdout_sessions)))

    # Canary positive control (Sprint 10vv, 2026-08-31): garantisce che OGNI
    # sessione canary (train_sessions/holdout_sessions taggate _canary_group
    # da inject_canaries()) finisca nel pool di valutazione, invece di
    # dipendere dal campionamento casuale sopra — con solo poche centinaia di
    # canary su decine di migliaia di membri totali, il sottocampionamento
    # casuale ne includerebbe solo una frazione variabile, rendendo
    # canary_auc_roc più rumoroso del necessario per un positive control
    # pensato per essere un test pulito e diretto. Puramente additivo: le
    # sessioni già presenti non vengono duplicate (dedup per id()); zero
    # effetto se nessuna sessione ha _canary_group (ogni run esistente/
    # pubblicato, inclusa l'intera campagna 5-seed×8-config — Sprint 10tt).
    _canary_train_extra = [
        s for s in train_sessions
        if s.get("_canary_group") is not None
        and id(s) not in {id(x) for x in members_bal}
    ]
    _canary_holdout_extra = [
        s for s in holdout_sessions
        if s.get("_canary_group") is not None
        and id(s) not in {id(x) for x in nonmembers_bal}
    ]
    if _canary_train_extra or _canary_holdout_extra:
        members_bal    = members_bal + _canary_train_extra
        nonmembers_bal = nonmembers_bal + _canary_holdout_extra
        logger.info(
            f"[CANARY] Pool di valutazione esteso: +{len(_canary_train_extra)} "
            f"membri, +{len(_canary_holdout_extra)} non-membri canary garantiti."
        )

    logger.info(
        f"LiRA — n_shadow={n_shadow}/cluster, shadow_epochs={shadow_epochs}/round, "
        f"eval pool: {len(members_bal)} members, {len(nonmembers_bal)} non-members"
    )

    # Concatenate into a single ordered list: members first, then non-members.
    # Index j < len(members_bal) → member; j >= len(members_bal) → non-member.
    eval_samples = members_bal + nonmembers_bal
    n_eval       = len(eval_samples)

    # Accumulatore opzionale per l'attacco "composto" multi-round (2026-08-12,
    # vedi ComposedLiRAAttack in src/plugins/attacks/lira.py): somma il
    # log-likelihood-ratio dello STESSO campione su tutti i round invece di
    # mediare 10 AUC-ROC indipendenti — combinazione Neyman-Pearson ottima per
    # evidenza indipendente ripetuta, pensata per sfruttare il vero budget ε
    # COMPOSTO (vedi "epsilon_cumulative_naive" in config) invece del singolo
    # ε per round che ogni round-AUC indipendente vede. Piggyback sulla STESSA
    # retraining degli shadow già fatta da questa funzione, per non raddoppiare
    # il costo computazionale con una funzione separata. Attivo SOLO se il
    # chiamante passa composed_output (non None) — default None, quindi zero
    # impatto sul comportamento/output esistente per ogni chiamante attuale.
    _cumulative_scores: dict[int, float] = {}
    # Sprint 10zz+90 (2026-09-15) — conta in QUANTI round ogni campione ha
    # effettivamente ricevuto un punteggio (non tutti i round ne danno uno per
    # ogni campione: `continue` sopra per calibrazione insufficiente, o per lo
    # skip 8σ di _UNCALIBRATED_Z_THRESHOLD). _cumulative_scores è una SOMMA di
    # log-likelihood ratio — sommare evidenza indipendente round su round è
    # la costruzione corretta di Carlini et al. 2022 quando ogni campione è
    # scorato lo stesso numero di volte, ma se due campioni sono scorati un
    # numero DIVERSO di volte, la magnitudo del loro punteggio composto
    # riflette in parte "quante volte è stato possibile scorarlo", non solo
    # "quanta evidenza di membership c'è" — un potenziale confondente se il
    # tasso di skip correlasse con la classe (membro/non-membro). Puramente
    # diagnostico: non modifica composed_lira_auc_roc (la statistica ufficiale
    # del paper, invariata), aggiunge solo un campo di confronto opt-in
    # (composed_lira_auc_roc_mean_per_round, sotto) quando composed_output è
    # fornito — stesso costo zero-impatto delle altre diagnostiche already
    # esistenti in questa funzione.
    _cumulative_score_counts: dict[int, int] = {}
    # RMIA (2026-10-13): composizione su piu' round, vedi _rmia_round_fields()
    _rmia_cumulative: dict[int, float] = {}
    _rmia_cum_member: dict[int, bool] = {}
    _rmia_cum_canary: dict[int, Any] = {}
    _sample_is_member:  dict[int, bool]  = {id(s): True for s in members_bal}
    _sample_is_member.update({id(s): False for s in nonmembers_bal})

    # Canary positive control (Sprint 10vv, 2026-08-31): mappa id(sample) →
    # _canary_group, per isolare canary_auc_roc dall'AUC principale sia a
    # livello di round sia nell'accumulo composto, senza toccare nessuna
    # formula/soglia/pooling esistente. Vuota se nessuna sessione ha il tag
    # (ogni run esistente/pubblicato) — zero impatto in quel caso.
    _sample_canary_group: dict[int, str] = {
        id(s): s["_canary_group"]
        for s in members_bal + nonmembers_bal
        if s.get("_canary_group") is not None
    }

    # Fix strutturale 2026-08-21: la vecchia mappa `_train_idx` (id(sample) →
    # posizione in train_sessions) è stata rimossa — serviva solo a
    # verificare `train_idx in in_set` quando `in_set` conteneva posizioni
    # numeriche nel SOLO pool membri. Ora `in_set` contiene id() Python di
    # sessioni campionate dall'universo shadow COMBINATO (membri+non-membri,
    # vedi Step 2 sotto) — il controllo di appartenenza IN/OUT si fa
    # direttamente con `id(sample) in in_set`, uniforme per entrambe le
    # classi, senza bisogno di questa mappa intermedia.

    def _build_tensor(sess_list: list[dict]) -> torch.Tensor | None:
        rows = []
        for s in sess_list:
            try:
                rows.append([float(s[f]) for f in _mia_feature_names(cfg)])
            except (KeyError, TypeError, ValueError):
                continue
        return torch.tensor(rows, dtype=torch.float32) if rows else None

    def _load_weights_into(model: Autoencoder, weights: list) -> bool:
        """Carica una lista di pesi (stesso ordine di state_dict().values()) in
        `model`. Restituisce False se le shape non combaciano (nessuna eccezione)."""
        orig_state = model.state_dict()
        keys = list(orig_state.keys())
        if len(weights) != len(keys):
            return False
        state = {
            k: (w if isinstance(w, torch.Tensor) else torch.tensor(w)).to(orig_state[k].dtype)
            for k, w in zip(keys, weights)
        }
        model.load_state_dict(state, strict=True)
        for buf_name, buf in model.named_buffers():
            if "running_var" in buf_name:
                buf.clamp_(min=1e-8)
        return True

    # ── Step 2: IN/OUT sample assignment per shadow — FISSO tra i round ────────
    # Rappresenta la conoscenza ausiliaria fissa dell'attaccante (stesso subset di
    # sessioni per ogni cluster/shadow). Solo i PESI dello shadow vengono
    # riaddestrati ogni round (Step 3), non il subset di campioni.
    # Offset grande e primo per cluster, per garantire seed distinti tra cluster senza
    # usare hash(str) (non deterministico tra processi Python — PYTHONHASHSEED random).
    #
    # Fix strutturale 2026-08-21 — "universo shadow simmetrico" (il fix più
    # profondo di questa indagine, vedi analisi completa nel docstring più
    # sotto). PRIMA di questo fix, `cluster_idx_pool` (e quindi il pool da
    # cui ogni shadow campiona il proprio subset IN) conteneva SOLO indici in
    # `train_sessions` — gli `holdout_sessions` (i veri non-membri) non
    # potevano MAI comparire nel subset IN di nessuno shadow, in nessun
    # round. Conseguenza: un non-membro non poteva MAI avere una
    # calibrazione IN reale — non per una formula sbagliata, ma perché il
    # POOL da cui gli shadow attingono escludeva strutturalmente l'intera
    # popolazione dei non-membri. Qualunque approssimazione di μ_in per un
    # non-membro (costante 2026-08-15..08-20, poi ancorata al proprio μ_out
    # 2026-08-21) restava quindi sistematicamente diversa dalla calibrazione
    # REALE usata per i membri — un'asimmetria di formula tra le due classi
    # che il test `lira_debug_matched_formula_auc` (stesso giorno) ha
    # confermato essere la causa dominante dell'AUC gonfiato (0.72-0.82 no-DP,
    # fino a 0.66 anche sotto Central DP ε=1.0 — contro ~0.50 di
    # raw_mse_auc/Yeom in ENTRAMBI i casi).
    # Fix: il pool da cui ogni shadow campiona il proprio subset IN è ora
    # `cluster_shadow_universe[cid]` — l'UNIONE di membri (`cluster_members`,
    # dati FL reali) e non-membri (`cluster_holdout`, sessioni mai viste dal
    # training FL) dello stesso cluster — con assegnazione IN/OUT casuale,
    # indipendente dalla VERA membership rispetto al modello target. Questo è
    # esattamente il disegno originale di LiRA (Carlini et al. 2022): gli
    # shadow vengono addestrati su split casuali di un pool più ampio, non
    # sul solo insieme di training del modello attaccato — sia membri sia
    # non-membri del target possono quindi finire IN per alcuni shadow e OUT
    # per altri, ottenendo entrambi una calibrazione bidirezionale reale.
    # `in_indices`/`in_set` ora contiene id() Python delle sessioni
    # campionate (non più posizioni numeriche in train_sessions) — il
    # controllo di appartenenza più sotto (`id(sample) in in_set`) è quindi
    # identico per membri e non-membri, eliminando l'asimmetria alla radice.
    # Attivo solo quando cluster_membership è reale — quando è None (path
    # storico per i test con dati sintetici, `cluster_holdout` resta vuoto
    # per costruzione, vedi sopra) l'universo collassa al solo pool membri,
    # preservando ESATTAMENTE il comportamento pre-fix per quel path e quindi
    # la suite di 83 test non-torch.
    _CLUSTER_SEED_OFFSET = 104729
    shadow_in_idx_sets_per_cluster: dict[str, list[set[int]]] = {}
    shadow_tensors_per_cluster: dict[str, list[torch.Tensor | None]] = {}
    # Sprint 10zz (2026-09-01), attivo solo se capture_shadow_weights=True:
    # per ogni shadow, "è la maggioranza del suo subset IN composta da membri
    # reali (cluster_members) o da non-membri (cluster_holdout)?" — usato da
    # run_fedmia_gradient() per separare i pesi finali degli shadow in due
    # gruppi di calibrazione REALI (non simulati a rumore gaussiano). Non
    # influisce in alcun modo sul comportamento esistente di LiRA (solo
    # lettura di in_ids già calcolato sopra).
    shadow_in_majority_member_per_cluster: dict[str, list[bool]] = {}

    for cluster_idx, cid in enumerate(_CLUSTER_IDS):
        cluster_shadow_universe = cluster_members[cid] + cluster_holdout.get(cid, [])

        # FIX 2026-09-11 (bug reale trovato durante un audit richiesto
        # dall'utente, non da un run fallito): quando controlled_composition=True
        # (solo run_fedmia_gradient(), mai la vera LiRA registrata), il ramo IN
        # sopra in questo loop NON passa da _sample_preserving_canary_groups()
        # (Sprint 10zz+16) — usa due rng.sample() diretti su cluster_members/
        # cluster_holdout, perché la composizione IN deliberata per-shadow
        # (10%-90% membri) è incompatibile con la nozione di "gruppo canary
        # atomico" di quella funzione. Se cfg["canary"]["enabled"] fosse MAI
        # True insieme a --include-fedmia-gradient, questo riaprirebbe
        # silenziosamente la stessa contaminazione shadow-canary già trovata e
        # corretta per la vera LiRA (Sprint 10zz+15/16) — ma solo qui, in un
        # diagnostico esplicitamente "primo draft, mai validato, mai nel
        # registro" (vedi docstring di run_fedmia_gradient()). Invece di
        # lasciare la lacuna silenziosa, la rendiamo rumorosa: fallisce subito
        # e in modo esplicito piuttosto che produrre un canary_auc_roc
        # silenziosamente inaffidabile in questo path specifico.
        if controlled_composition and any(
            s.get("_canary_group") is not None for s in cluster_shadow_universe
        ):
            raise ValueError(
                f"[{cid}] controlled_composition=True con canary abilitato non è "
                "supportato: il campionamento a composizione deliberata bypassa "
                "_sample_preserving_canary_groups() (Sprint 10zz+16), riaprendo la "
                "contaminazione shadow-canary già corretta per la vera LiRA. "
                "Disabilita cfg['canary']['enabled'] per questo run, oppure non "
                "usare --include-fedmia-gradient insieme al canary."
            )

        _member_id_set = {id(s) for s in cluster_members[cid]}
        cluster_in_idx_sets: list[set[int]] = []
        cluster_tensors: list[torch.Tensor | None] = []
        cluster_in_majority_member: list[bool] = []

        for shadow_idx in range(n_shadow):
            _s = seed + cluster_idx * _CLUSTER_SEED_OFFSET + shadow_idx * 31337
            shadow_rng = random.Random(_s)
            n_in       = max(batch_size + 1, len(cluster_shadow_universe) // 2)
            n_in       = min(n_in, len(cluster_shadow_universe))

            # Fix (2026-09-01, controlled_composition — vedi Args nel
            # docstring per il razionale completo): campionamento a
            # composizione deliberata invece che uniforme, SOLO quando
            # richiesto esplicitamente (mai per LiRA reale) e SOLO se
            # esistono davvero non-membri per questo cluster.
            if controlled_composition and cluster_holdout.get(cid):
                _frac_member = 0.1 + 0.8 * (shadow_idx / max(1, n_shadow - 1))
                _n_member_target = min(
                    round(n_in * _frac_member), len(cluster_members[cid])
                )
                _n_nonmember_target = min(
                    n_in - _n_member_target, len(cluster_holdout[cid])
                )
                in_sessions_sampled = (
                    shadow_rng.sample(cluster_members[cid], _n_member_target)
                    + shadow_rng.sample(cluster_holdout[cid], _n_nonmember_target)
                )
            else:
                # Sprint 10zz+16: gruppi canary campionati come unità
                # atomiche — no-op (identico a shadow_rng.sample()) per
                # ogni pool senza sessioni canary (ogni run reale/pubblicato).
                in_sessions_sampled = _sample_preserving_canary_groups(
                    shadow_rng, cluster_shadow_universe, n_in
                )

            in_ids = set(id(s) for s in in_sessions_sampled)
            cluster_in_idx_sets.append(in_ids)
            cluster_in_majority_member.append(
                len(in_ids & _member_id_set) > len(in_ids) / 2
            )

            shadow_tensor = _build_tensor(in_sessions_sampled)
            if shadow_tensor is None or len(shadow_tensor) < batch_size:
                logger.warning(
                    f"LiRA[{cid}] shadow {shadow_idx}: {len(in_sessions_sampled)} sessioni < "
                    f"batch_size={batch_size} — shadow skippato in ogni round"
                )
            cluster_tensors.append(shadow_tensor)

        shadow_in_idx_sets_per_cluster[cid] = cluster_in_idx_sets
        shadow_tensors_per_cluster[cid]     = cluster_tensors
        shadow_in_majority_member_per_cluster[cid] = cluster_in_majority_member

    # ── Step 3 & 4: per round, riaddestra gli shadow (warm-start) e valuta i client ─
    lira_results: dict[int, dict[str, Any]] = {}
    # Sprint 10zz+29 (2026-09-03, task #54) — curve ROC complete (fpr/tpr,
    # non solo AUC/TPR@fixed/Advantage) per il plot log-log richiesto
    # dall'utente. Popolato solo se roc_curve_dump_path è impostato.
    _roc_curves_per_round: dict[int, dict[str, Any]] = {}
    # Sprint 10zz+32 (2026-09-03, task #57) — dump delle liste COMPLETE
    # (non solo la media, già in _diag_fields sopra) di _diag_raw_loss_members/
    # _diag_raw_loss_nonmembers per round, per verificare empiricamente se la
    # MSE grezza è approssimativamente Gaussiana (assunzione richiesta dal fit
    # parametrico di LiRA, §3 di docs/MetricsReference_DSN2027.md) — Carlini
    # et al. la verificano SOLO dopo un logit-scaling della confidenza,
    # trasformazione che non abbiamo un equivalente naturale per applicare a
    # una MSE di ricostruzione. Popolato solo se raw_loss_dump_path è
    # impostato — riusa dati già raccolti internamente (_diag_raw_loss_*),
    # zero costo computazionale aggiuntivo, solo I/O.
    _raw_loss_per_round: dict[int, dict[str, list[float]]] = {}

    for round_num, round_data in sorted(
        (item for item in fl_results.items() if item[0] > 0), key=lambda x: x[0]
    ):
        # Sprint 10zz+92 (2026-09-15) — CORREZIONE del threat model per
        # dp_mode="dp-fedavg", non un flag diagnostico opt-in: decisione
        # esplicita dell'utente dopo l'errata esterna verificata (punto
        # "Three DP placements — SMENTITO"). Il commento su `_store_raw` in
        # run_fl_rounds() (~riga 1114) documenta GIA' che sotto dp-fedavg un
        # server "honest-but-curious"/trusted vede l'update RAW (pre-clip,
        # pre-noise) prima di clippare+rumorizzare esso stesso — a differenza
        # di "local", dove il client applica clip+noise PRIMA di trasmettere,
        # quindi il server non vede mai il valore pulito, nemmeno
        # transitoriamente. Fino a questo fix LiRA attaccava "updates" (post-
        # privatize) IDENTICAMENTE per dp-fedavg e local, rendendo le due
        # modalità indistinguibili per costruzione — root cause delle 3 righe
        # duplicate in Tabella 2 (dp-fedavg == local su ogni ε/seed/attacco,
        # bit per bit). Sotto dp-fedavg CON DP attivo, l'attaccante ora
        # preleva "raw_updates" (l'update così com'è stato sottomesso, PRIMA
        # che il server stesso lo clippi/rumorizzi) invece di "updates" —
        # central e local NON cambiano (central attacca già clip-only per una
        # ragione strutturale diversa e già documentata riga ~1010 sopra;
        # local non ha mai un raw_updates disponibile, per costruzione, vedi
        # `_store_raw` sopra). Conseguenza attesa, non un effetto collaterale
        # da correggere: il numero LiRA di dp-fedavg diventa insensibile a ε
        # per costruzione (identico al caso no-DP) — dp-fedavg misura ora "se
        # un server che vede il dato grezzo prima di applicare la propria
        # stessa privatizzazione offre ancora protezione" (risposta: nessuna,
        # per costruzione), una domanda DIVERSA da quella che central/local
        # misurano, non un indebolimento della stessa domanda. INVALIDA ogni
        # numero LiRA già pubblicato per dp-fedavg con DP attivo (non per
        # no-DP dp-fedavg, né per central/local, che non cambiano codice qui).
        if dp_mode == "dp-fedavg" and not no_dp:
            client_updates = round_data.get("raw_updates") or []
            if not client_updates:
                logger.warning(
                    f"LiRA round {round_num}: raw_updates assente per "
                    "dp_mode=dp-fedavg (atteso presente, vedi _store_raw in "
                    "run_fl_rounds()) — skip"
                )
                continue
        else:
            # Fix 2026-07-21c: attacca "updates" (post-privatize, ciò che viene
            # davvero sottoposto ad aggregazione) invece di "raw_updates" (pre-DP
            # per costruzione) — invariato per central/local/no-DP.
            client_updates = round_data.get("updates", [])
            if not client_updates:
                logger.warning(f"LiRA round {round_num}: nessun update — skip")
                continue

        # Observation surface "global" (Sprint 10zz+94) — costruisce UN SOLO modello
        # condiviso per questo round da round_data["global_weights"], esattamente
        # come fanno run_yeom()/run_shadow(). Costruito una volta sola qui
        # fuori dal loop per-client sotto; se "client" (default), resta None e non è
        # usato — zero costo/impatto per ogni run/config esistente.
        _lira_global_model = None
        if _lira_observation_surface == "global":
            _global_weights_this_round = round_data.get("global_weights")
            if _global_weights_this_round is None:
                logger.warning(
                    f"LiRA round {round_num}: observation_surface='global' ma "
                    "global_weights assenti — skip round"
                )
                continue
            _lira_global_model = Autoencoder(input_dim=input_dim, **_autoencoder_arch_kwargs(cfg))
            if not _load_weights_into(_lira_global_model, _global_weights_this_round):
                logger.warning(
                    f"LiRA round {round_num}: observation_surface='global' — "
                    "global_weights shape mismatch — skip round"
                )
                continue
            _lira_global_model.eval()

        # Warm-start per gli shadow di QUESTO round: stesso punto di partenza usato
        # dai client reali per il training locale del round (fix 2026-07-21b).
        # Round 1 → init casuale (nessun round precedente, come i client reali).
        # Sprint 10zz+88: se cfg["lira"]["shadow_init"]=="cold", _warm_start resta
        # SEMPRE None (init casuale ad ogni round, mai i pesi globali reali) —
        # vedi commento sopra su _lira_shadow_init per il razionale dell'ablation.
        _warm_start = (
            fl_results.get(round_num - 1, {}).get("global_weights")
            if (round_num > 1 and _lira_shadow_init == "warm") else None
        )

        shadow_mse_matrix_per_cluster: dict[str, list[list[float | None]]] = {}
        # Sprint 10zz (2026-09-01), popolato solo se capture_shadow_weights=True
        # (altrimenti resta vuoto, zero costo/impatto): vettore di peso FINALE
        # (post-training, post-privatizzazione se DP attiva) di ogni shadow,
        # flatten completo — vedi nota su capture_shadow_weights nel docstring.
        shadow_weight_vectors_per_cluster: dict[str, list[list[float] | None]] = {}

        for cluster_idx, cid in enumerate(_CLUSTER_IDS):
            in_sets  = shadow_in_idx_sets_per_cluster[cid]
            tensors  = shadow_tensors_per_cluster[cid]
            cluster_mse_matrix: list[list[float | None]] = []
            cluster_weight_vectors: list[list[float] | None] = []

            for shadow_idx, (in_indices, shadow_tensor) in enumerate(zip(in_sets, tensors)):
                if shadow_tensor is None:
                    cluster_mse_matrix.append([None] * n_eval)
                    if capture_shadow_weights:
                        cluster_weight_vectors.append(None)
                    continue

                _s = (
                    seed + round_num * 1_000_003
                    + cluster_idx * _CLUSTER_SEED_OFFSET + shadow_idx * 31337
                )

                # Fix (review indipendente 2026-07-21d): seed PRIMA di istanziare il
                # modello — altrimenti, quando _warm_start è None (round 1), l'init
                # casuale di Autoencoder() dipende dallo stato ambientale del RNG
                # globale di torch invece che da _s, rendendo il round 1 non
                # riproducibile in isolamento.
                torch.manual_seed(_s)
                shadow_model = Autoencoder(input_dim=input_dim, **_autoencoder_arch_kwargs(cfg))
                if _warm_start is not None and not _load_weights_into(shadow_model, _warm_start):
                    logger.warning(
                        f"LiRA[{cid}] round {round_num} shadow {shadow_idx}: "
                        "warm-start non applicabile (shape mismatch) — init casuale"
                    )

                # Fix 2026-07-22 (review B1): cattura i pesi PRIMA del training
                # dello shadow — riferimento per il clip del DELTA più sotto,
                # simmetrico a pre_round_weights in run_fl_rounds(). Coincide
                # col warm-start se applicato con successo, altrimenti con
                # l'init casuale appena fatto (round 1 / shape mismatch).
                _shadow_pretrain_weights = [
                    w.detach().clone() for w in shadow_model.state_dict().values()
                ]

                shadow_opt  = torch.optim.Adam(shadow_model.parameters(), lr=lr)
                shadow_crit = torch.nn.MSELoss()
                shadow_ds  = torch.utils.data.TensorDataset(shadow_tensor)
                shadow_gen = torch.Generator()
                shadow_gen.manual_seed(_s)
                shadow_loader = torch.utils.data.DataLoader(
                    shadow_ds, batch_size=batch_size, shuffle=True,
                    drop_last=True, generator=shadow_gen,
                )

                shadow_model.train()
                for _ in range(shadow_epochs):
                    for (batch,) in shadow_loader:
                        shadow_opt.zero_grad()
                        recon = shadow_model(batch)
                        loss  = shadow_crit(recon, batch)
                        loss.backward()
                        shadow_opt.step()
                shadow_model.eval()

                # Fix 2026-07-21c: privatizza lo shadow ESATTAMENTE come un client
                # reale, cosi' la calibrazione IN/OUT riflette il vero effetto della
                # DP quando abilitata (altrimenti un target rumoroso verrebbe
                # confrontato con shadow puliti — mismatch aggiuntivo).
                # Fix 2026-07-22: sotto "central" DP, il client NON rumorizza il
                # proprio update (solo clip — il rumore va sull'aggregato, mai
                # osservabile da LiRA che attacca il singolo update) — lo shadow
                # deve rispecchiare la STESSA cosa, non gm.privatize() completo.
                if not no_dp:
                    _shadow_keys = list(shadow_model.state_dict().keys())
                    _shadow_update = _GU(
                        node_id=f"shadow-{cid}-{shadow_idx}",
                        cluster_id=cid,
                        round_num=round_num,
                        weights=[w.detach().clone() for w in shadow_model.state_dict().values()],
                        gradients=None,
                        loss=None,
                        n_samples=len(shadow_tensor),
                        metadata={},
                    )
                    # reference_weights=_shadow_pretrain_weights (fix 2026-07-22,
                    # review B1): clippa il DELTA dello shadow rispetto al proprio
                    # punto di partenza, come per i client reali in run_fl_rounds().
                    if dp_mode == "central":
                        _privatized = gm.clip_only(
                            _shadow_update, weight_keys=_shadow_keys,
                            reference_weights=_shadow_pretrain_weights,
                        )
                    else:
                        # Ablazioni (2026-10-05): lo shadow usa lo stesso meccanismo
                        # dei client reali, anche con clip-only/noise-only.
                        _privatized = _privatize_per_client(
                            gm, _shadow_update, exp_cfg.get("dp_ablation", "full"),
                            weight_keys=_shadow_keys,
                            reference_weights=_shadow_pretrain_weights,
                        )
                    _load_weights_into(shadow_model, _privatized.weights)
                    shadow_model.eval()

                if capture_shadow_weights:
                    with torch.no_grad():
                        cluster_weight_vectors.append(
                            torch.cat(
                                [w.detach().flatten() for w in shadow_model.state_dict().values()]
                            ).tolist()
                        )

                mse_row: list[float | None] = []
                for sample in eval_samples:
                    try:
                        row    = [float(sample[f]) for f in _mia_feature_names(cfg)]
                        tensor = torch.tensor([row], dtype=torch.float32)
                        with torch.no_grad():
                            recon = shadow_model(tensor)
                            mse_row.append(float(torch.mean((recon - tensor) ** 2).item()))
                    except (KeyError, TypeError, ValueError):
                        mse_row.append(None)
                cluster_mse_matrix.append(mse_row)

            shadow_mse_matrix_per_cluster[cid] = cluster_mse_matrix
            if capture_shadow_weights:
                shadow_weight_vectors_per_cluster[cid] = cluster_weight_vectors

        logger.info(
            f"LiRA round {round_num}: {n_shadow}×{len(_CLUSTER_IDS)} shadow "
            f"riaddestrati ({shadow_epochs} epoche/shadow, "
            f"warm_start={'sì' if _warm_start is not None else 'no (init casuale)'}, "
            f"dp_su_shadow={'no (--no-dp)' if no_dp else f'sì (dp_mode={dp_mode})'})"
        )

        # Per-cluster, per-round global IN distribution — fallback per i rari
        # campioni (di ENTRAMBE le classi, dal fix strutturale 2026-08-21
        # sopra — non più "sempre i non-membri") con troppo poche
        # osservazioni IN reali. Ricalcolata ogni round perché gli shadow
        # sono stati appena riaddestrati.
        # Fix strutturale 2026-08-21: prima pooling solo su `members_bal`
        # (range(len(members_bal))) con lookup in `_train_idx` — un
        # non-membro non poteva mai contribuire qui, per lo stesso motivo
        # strutturale spiegato sopra. Ora itera su TUTTI gli eval_samples e
        # usa id(sample) in in_set direttamente (in_set contiene id() Python
        # dall'universo shadow combinato) — un non-membro che uno shadow ha
        # per caso campionato come IN contribuisce al pool esattamente come
        # un membro nella stessa situazione.
        global_in_stats_per_cluster: dict[str, tuple[float, float]] = {}
        for cid in _CLUSTER_IDS:
            in_sets    = shadow_in_idx_sets_per_cluster[cid]
            mse_matrix = shadow_mse_matrix_per_cluster.get(cid, [])
            pooled: list[float] = []
            for j in range(n_eval):
                _sid = id(eval_samples[j])
                for si, in_set in enumerate(in_sets):
                    if _sid in in_set and si < len(mse_matrix):
                        mse = mse_matrix[si][j]
                        if mse is not None:
                            pooled.append(mse)

            mu    = float(np.mean(pooled)) if pooled else 0.05
            sigma = max(float(np.std(pooled)), max(mu * 0.05, 1e-4)) if pooled else 0.01
            global_in_stats_per_cluster[cid] = (mu, sigma)

        # Fix — asymmetric IN/OUT variance for non-members (2026-07-21e):
        # For a genuine non-member, out_losses is built from n_shadow (~8) MSE
        # values on THAT ONE sample — a per-sample, cross-shadow spread. Right
        # after a shared warm-start (round ≥ 2), all shadows for a cluster begin
        # from the identical global_weights and have only briefly diverged, so
        # on a point none of them trained on they tend to produce nearly
        # identical reconstructions → σ_out collapses toward the μ×0.05 floor
        # (observed: as low as ~1e-4). Meanwhile the IN side for a non-member
        # uses the pooled, cluster-wide global_in_stats_per_cluster fallback
        # above (hundreds of samples, naturally much wider σ). Comparing a
        # near-collapsed per-point σ_out against a wide pooled σ_in makes the
        # Gaussian log-likelihood-ratio dominated by the 1/σ² term rather than
        # by genuine membership signal, producing non-member scores that spike
        # to the clip ceiling (empirically +17.9 at round 2, seed 42,
        # nodp-sweep2) while member scores stay flat near 0 (member scoring is
        # symmetric: both its IN and OUT sides are already per-sample/small-N).
        # Fix: compute an analogous pooled, per-cluster/per-round GLOBAL OUT
        # distribution (same construction as global_in_stats_per_cluster, just
        # pooling the complementary — OUT — observations) and use it as a
        # floor for σ_out (and, symmetrically, for σ_in) so neither side's
        # variance can collapse below what is typically observed across the
        # whole cluster this round. This does not touch μ_in/μ_out (the
        # discriminative signal itself) — only prevents an under-estimated
        # per-sample σ from artificially amplifying the ratio.
        global_out_stats_per_cluster: dict[str, tuple[float, float]] = {}
        for cid in _CLUSTER_IDS:
            in_sets    = shadow_in_idx_sets_per_cluster[cid]
            mse_matrix = shadow_mse_matrix_per_cluster.get(cid, [])
            pooled_out: list[float] = []
            for j in range(n_eval):
                is_mem = j < len(members_bal)
                _sample = eval_samples[j]
                # Fix 2026-08-12: il fix 2026-08-11 (guard cross-cluster
                # simmetrico) era applicato solo nel loop di scoring per-sample
                # (righe ~1731-1742), non qui — questo pool "floor" continuava a
                # mescolare member/non-member di QUALSIASI sito nella statistica
                # OUT di OGNI cluster, reintroducendo un livello più in profondità
                # la stessa contaminazione cross-cluster che il fix precedente
                # doveva eliminare (un member/non-member fuori sito valutato
                # contro shadow di un cluster che non ha mai visto quel tipo di
                # dato dà una loss innaturalmente alta, gonfiando σ_out_fb in modo
                # diverso per cluster in base a quanta contaminazione riceve —
                # inerte sotto clipping, che allinea le scale tra client, ma reale
                # sotto --no-dp). Stesso guard usato sotto, con la stessa mappa
                # _sample_to_cluster/_holdout_sample_to_cluster.
                _home_cluster = (
                    _sample_to_cluster.get(id(_sample)) if is_mem
                    else _holdout_sample_to_cluster.get(id(_sample))
                )
                if _home_cluster is not None and _home_cluster != cid:
                    continue
                # Fix strutturale 2026-08-21: prima questo controllo era
                # ristretto ai membri (`is_mem and train_idx...`) — un
                # non-membro non poteva mai risultare "IN" per definizione
                # del vecchio pool (solo train_sessions), quindi ogni sua
                # osservazione finiva sempre in pooled_out, corretto per
                # costruzione ma per il motivo sbagliato. Ora, con l'universo
                # shadow combinato, un non-membro PUÒ essere IN per alcuni
                # shadow — va escluso da pooled_out esattamente come un
                # membro nella stessa situazione, stesso controllo id()-based
                # per entrambe le classi.
                _sid = id(_sample)
                for si, in_set in enumerate(in_sets):
                    if si >= len(mse_matrix):
                        continue
                    mse = mse_matrix[si][j]
                    if mse is None:
                        continue
                    if _sid in in_set:
                        continue  # this (shadow, sample) pair is an IN observation, skip
                    pooled_out.append(mse)

            mu_o    = float(np.mean(pooled_out)) if pooled_out else 0.05
            sigma_o = max(float(np.std(pooled_out)), max(mu_o * 0.05, 1e-4)) if pooled_out else 0.01
            global_out_stats_per_cluster[cid] = (mu_o, sigma_o)

        round_member_scores:    list[float] = []
        _rmia_rows: list[dict[str, Any]] = []  # RMIA (2026-10-13)
        round_nonmember_scores: list[float] = []
        # Sprint 10zz+33 (2026-09-03, task #58) — attacco di Sablayrolles
        # et al. 2019 [56] come secondo scorer post-hoc, in parallelo a
        # round_member_scores/round_nonmember_scores sopra. Verificato via
        # Carlini et al. 2022 §V-C/Table I/II: A'(x,y) = τ_{x,y} - ℓ(f(x),y)
        # con τ_{x,y} = (μ_in(x,y)+μ_out(x,y))/2 — soglia NON parametrica
        # per-esempio (a differenza del fit Gaussiano di LiRA), "the most
        # direct influence for LiRA" secondo gli stessi autori. Riusa
        # ESATTAMENTE gli stessi μ_in/μ_out/target_loss già calcolati per
        # lira_score qualche riga sotto — nessuno shadow model aggiuntivo,
        # nessun training aggiuntivo, zero costo computazionale extra.
        round_sablayrolles_member_scores:    list[float] = []
        round_sablayrolles_nonmember_scores: list[float] = []
        # Sprint 10zz+36 (2026-09-03, task #61) — variante ESPLORATIVA di
        # LiRA con fit Gaussiano su log(MSE+eps) invece che su MSE grezza,
        # per testare se riduce il floor-hit-rate osservato nel fit raw
        # (task #57 ha misurato su dati reali: MSE grezza fortemente
        # non-Gaussiana — skewness~10-11, Jarque-Bera~16-24 milioni contro
        # soglia 5.99; log-transform migliora di 4-5 ordini di grandezza ma
        # non elimina formalmente la non-normalità a questa numerosità —
        # vedi docs/MetricsReference_DSN2027.md §3 per il dettaglio
        # completo). SEMPLIFICAZIONE DICHIARATA rispetto al fit raw sopra
        # (vedi commento al punto di calcolo, più sotto, per il perché):
        # fallback μ_in_log≈μ_out_log invece dell'ancoraggio per-cluster
        # raffinato in settimane di fix sul fit raw, e floor fisso invece di
        # scale-adattivo. Questo è un ablation esplorativo per rispondere
        # alla domanda specifica "il log-transform riduce il floor-hit-rate
        # e/o migliora AUC/TPR?" — non ancora sottoposto allo stesso rigore
        # (worst-case check, canary sanity positive-control) del fit raw
        # prima di essere promosso a metrica primaria.
        round_lira_log_member_scores:    list[float] = []
        round_lira_log_nonmember_scores: list[float] = []
        _diag_sigma_in_log_values:  list[float] = []
        _diag_sigma_out_log_values: list[float] = []
        _diag_sigma_in_log_floor_hits  = 0
        _diag_sigma_out_log_floor_hits = 0
        # Canary positive control (Sprint 10vv): accumulatori paralleli,
        # popolati SOLO per campioni taggati _canary_group — restano vuoti
        # per ogni run esistente/pubblicato senza canary iniettati.
        round_canary_member_scores:    list[float] = []
        round_canary_nonmember_scores: list[float] = []
        # Gruppi canary DISTINTI effettivamente entrati nell'AUC (2026-09-16,
        # Sprint 10zz+113). Motivo: i conteggi di record sopra sovrastimano
        # di molto la potenza statistica della metrica. Ogni gruppo membro e'
        # un template replicato n_duplicates+1 volte con feature IDENTICHE,
        # quindi loss identica e — poiche' _sample_preserving_canary_groups()
        # tiene il gruppo atomico fra gli shadow — lira_score identico; ogni
        # gruppo non-membro compare 2 volte. L'AUC su 155x40 record coincide
        # numericamente con quella su 5x20 valori distinti, ma la sua
        # varianza e' quella del secondo numero: riportare solo i record
        # implica un errore standard ~sqrt(31*2)≈7.9 volte piu' stretto del
        # vero. Questi campi rendono il denominatore reale leggibile dal JSON
        # invece che ricostruibile solo a mano.
        round_canary_member_groups:    set[str] = set()
        round_canary_nonmember_groups: set[str] = set()
        # Gruppi distinti del pool RAW (Sprint 10zz+119, 2026-09-16). Separati
        # dai due sopra perche' dal fix di questo sprint il pool raw NON e' piu'
        # filtrato da `insufficient_calibration`, quindi i due denominatori
        # possono differire e vanno riportati distinti.
        round_canary_member_raw_groups:    set[str] = set()
        round_canary_nonmember_raw_groups: set[str] = set()
        # Diagnostico raw-loss canary (Sprint 10ww) — target_loss grezzo,
        # a monte della calibrazione shadow μ/σ, stesso principio di
        # _diag_raw_loss_members/nonmembers ma ristretto ai canary.
        round_canary_member_raw_loss:    list[float] = []
        round_canary_nonmember_raw_loss: list[float] = []
        # Sprint 10zz+170: loss dei canary sul modello riscalato, per s:
        # (membri, non-membri). Vuoto se --canary-rescale non e' dato.
        round_canary_rescaled_loss: dict[float, tuple[list[float], list[float]]] = {
            _s: ([], []) for _s in _canary_rescale
        } if _lira_observation_surface == "client" else {}
        _rescale_ref_per_client = (
            (round_data.get("pre_round_weights_per_client") or {})
            if round_canary_rescaled_loss else {}
        )
        # Diagnostica 2026-09-13 (richiesta esplicita dell'utente dopo il run
        # experiment_canary_positive_control_caltech_highdensity.yaml:
        # canary_n_member=2490 invece dei 2520 attesi — 84×30 — e
        # canary_n_nonmember=19 invece di 20; run precedenti mostravano
        # deficit diversi e non multipli di n_duplicates, es. office1 -30,
        # caltech originale -5, jpl -1 sul lato non-membro). Esclusa
        # sperimentalmente la soglia _UNCALIBRATED_Z_THRESHOLD (skip_rate=0.0
        # in ogni round di quel run) e l'estrazione feature (enrich_sessions
        # assegna hour_of_day/duration_hours incondizionatamente). Resta da
        # verificare se il colpevole sia uno dei tre `continue` per-campione
        # qui sotto (estrazione tensore, guard cross-cluster, calibrazione
        # insufficiente out_losses<2) — puramente additivo, incrementa un
        # contatore SOLO per campioni già taggati _canary_group, zero impatto
        # su qualunque score/soglia/formula esistente o su run senza canary.
        _diag_canary_skip_reason: dict[str, int] = {
            "tensor_extraction": 0,
            "cross_cluster_guard": 0,
            "insufficient_calibration": 0,
        }

        # DIAGNOSTICA 2026-08-15 (indagine anomalia no-DP AUC≈0.5, vedi
        # docs/ReadingList_DSN2027.md e README Sprint-log 2026-08-15):
        # nodp-sweep3 mostra un pattern sospetto — member e non-member NON si
        # separano mai, si muovono INSIEME (quasi identici) con un salto enorme
        # round1→round2 (da ~+3 a ~-6.5) poi un decadimento graduale verso 0 nei
        # round successivi — molto diverso dalla separazione stabile e piccola
        # osservata sotto central DP (~+0.2/-0.4, costante per tutti i round).
        # Ipotesi di lavoro: senza clipping (SOLO no-DP ne è privo — anche
        # central clippa via clip_only()), il modello converge a una loss molto
        # più bassa (fl.mean_loss≈0.0012 osservato al round 1 di nodp-sweep3,
        # un ordine di grandezza sotto le modalità con DP) — questo può far
        # collassare σ_in/σ_out sul floor minimo (1e-4 o μ×0.05), e con un σ
        # minuscolo anche una differenza minima tra target_loss e μ viene
        # amplificata a dismisura da /σ² (righe ~1838-1839 sotto), producendo
        # punteggi enormi e instabili PER ENTRAMBI i gruppi simmetricamente —
        # rumore di stima amplificato meccanicamente, non segnale di membership
        # vero. Queste 4 liste catturano i dati grezzi necessari per confermare
        # o smentire l'ipotesi SENZA dover strumentare di nuovo dopo un altro
        # sweep completo di ore: raw_loss_members/nonmembers (il segnale vero,
        # a monte di qualunque normalizzazione — se qui c'è già separazione
        # pulita ma il punteggio finale no, conferma l'amplificazione da σ; se
        # anche qui sono indistinguibili, il modello davvero non memorizza
        # abbastanza per LiRA — Scenario B genuino, non un artefatto) e
        # sigma_in_values/sigma_out_values (il σ EFFETTIVAMENTE usato — se
        # sistematicamente vicino al floor 1e-4 sotto no-DP e molto più alto
        # sotto central/dp-fedavg, conferma il floor collapse). Puramente
        # additivo — non altera lo scoring esistente, solo raccoglie dati in
        # parallelo per la diagnosi.
        _diag_raw_loss_members:    list[float] = []
        _diag_raw_loss_nonmembers: list[float] = []
        _diag_sigma_in_values:     list[float] = []
        _diag_sigma_out_values:    list[float] = []
        _diag_sigma_in_floor_hits  = 0  # conta quante volte il floor (non lo std grezzo) ha vinto il max()
        _diag_sigma_out_floor_hits = 0
        # DIAGNOSTICA 2026-08-20 (round 2 dell'indagine no-DP): dopo il fix
        # del floor simmetrico, la campagna completa (5 seed × 7 config) mostra
        # nodp-sweep1 stabilmente E SIGNIFICATIVAMENTE sotto 0.5 (CI bootstrap
        # [0.3675, 0.3742], non un caso di rumore) mentre lira_debug_raw_mse_auc_roc
        # resta ~0.50 in ogni round/seed controllato — il segnale è nullo a
        # livello grezzo ma l'inversione nel punteggio finale è forte e
        # riproducibile. Ipotesi: μ_in (calibrazione "IN" dagli shadow) risulta
        # sistematicamente MAGGIORE di μ_out (non minore, come l'intuizione
        # LiRA richiederebbe) — possibile artefatto del warm-start: uno shadow
        # "IN" continua per altre shadow_epochs (50) epoche NON vincolate
        # (nessun clipping/rumore sotto no-DP) un modello già ben convergente,
        # e può destabilizzarsi/overfittare in modo rumoroso proprio sui punti
        # che ha visto, mentre uno shadow "OUT" generalizza normalmente su
        # dati molto simili (stesso cluster, sessioni correlate). Se μ_in>μ_out
        # è reale, la formula del rapporto di verosimiglianza (invariata,
        # matematicamente corretta per μ_in<μ_out) si inverte di segno.
        # Questi due accumulatori (puramente additivi) isolano μ_in/μ_out
        # medi per confermarlo prima di decidere il fix.
        _diag_mu_in_values:  list[float] = []
        _diag_mu_out_values: list[float] = []
        # DIAGNOSTICA 2026-08-20 (round 3): μ_in<μ_out in aggregato (direzione
        # CORRETTA, confermato — smentisce l'ipotesi warm-start destabilizzato)
        # eppure lira_auc_roc resta invertito (~0.32-0.34) — l'algebra sui soli
        # valori medi aggregati non spiega l'inversione, serve vedere i valori
        # REALI per singolo campione (t, μ_in, μ_out, score), non solo le medie,
        # per capire se membri e non-membri vengono penalizzati in modo
        # strutturalmente diverso (es. μ_in dei non-membri è SEMPRE il
        # fallback per-cluster costante — mai una stima per-campione — mentre
        # per i membri μ_in è quasi sempre una stima reale per quel campione
        # specifico: un'asimmetria strutturale che le medie da sole non
        # rivelano). Dump limitato ai primi 15 membri + 15 non-membri
        # incontrati nel PRIMO client processato in questo round (evita di
        # sommergere il log su 3 client × 10 round).
        _diag_dump_mem_count    = 0
        _diag_dump_nonmem_count = 0
        _DIAG_DUMP_CAP = 15

        # DIAGNOSTICA 2026-08-21 (test mirato, vedi commento esteso al punto
        # in cui viene popolata più sotto): dopo il fix dell'ancoraggio μ_in
        # (2026-08-21), lira_auc_roc è passato da 0.32-0.44 (invertito) a
        # 0.72-0.82 — MOLTO più alto di raw_mse_auc/Yeom (~0.50, nessun
        # segnale a livello grezzo). Ipotesi da verificare: l'ancoraggio crea
        # un'asimmetria strutturale opposta a quella appena corretta — i
        # non-membri finiscono SEMPRE nel ramo fallback (per costruzione),
        # i membri finiscono quasi sempre nel ramo con calibrazione reale;
        # se le due formule hanno un comportamento sistematicamente diverso
        # anche in ASSENZA di vera memorizzazione, l'AUC alto sarebbe un
        # artefatto del ramo usato, non segnale reale. Test: per ogni membro
        # con calibrazione reale (len(in_losses)>=2), calcolo ANCHE lo score
        # "controfattuale" che avrebbe ricevuto se fosse stato forzato nel
        # ramo fallback (stessa formula usata per i non-membri) — puramente
        # per logging, non tocca lo score reale usato nel pool. Se questi
        # controfattuali sono sistematicamente negativi quanto i punteggi
        # reali dei non-membri, conferma l'artefatto di formula.
        _diag_counterfactual_member_scores: list[float] = []

        # Fix 2026-08-20 — campioni "non calibrabili" esclusi dal punteggio.
        # Il dump per-campione (2026-08-20, sopra) ha mostrato la causa reale
        # dell'inversione no-DP: alcune sessioni hanno target_loss REALE fino
        # a ~20 deviazioni standard (σ_in/σ_out, ormai correttamente
        # simmetrici dal fix del 2026-08-15) lontano da ENTRAMBE μ_in e μ_out
        # — cioè né gli shadow "IN" né quelli "OUT" spiegano minimamente
        # quella osservazione. In questo regime il segno di log_p_in-log_p_out
        # è deciso da quale delle due medie minuscole è per puro caso una
        # frazione più vicina a un valore comunque enorme e lontanissimo da
        # entrambe — rumore geometrico col segno arbitrario, non segnale di
        # membership. Sotto DP attivo σ è molto più grande (rumore/clipping
        # tengono tutto su una scala comparabile), quindi questi casi sono
        # rari o assenti — il problema è specifico al regime di loss
        # bassissima raggiunto solo da no-DP, esattamente come i due fix
        # precedenti di questa stessa indagine (floor 2026-08-15, pooling
        # 2026-08-11/12). Soglia: se il campione è oltre 8σ da ENTRAMBE le
        # distribuzioni (min dei due z-score assoluti > 8), il modello
        # gaussiano di LiRA non ha alcuna informazione calibrata su di esso —
        # va escluso, non forzato con un segno arbitrario. 8σ è scelta per
        # separare nettamente gli outlier osservati (~20σ) dai campioni
        # tipici (~0-3σ), lasciando un margine ampio. Contatore diagnostico
        # per trasparenza — se una frazione grande venisse esclusa, andrebbe
        # rivisto n_shadow o la costruzione stessa di μ/σ, non solo la soglia.
        _UNCALIBRATED_Z_THRESHOLD = 8.0
        _diag_uncalibrated_skipped = 0
        _diag_scored_total = 0

        _diag_mu_in_clamped = 0
        for _client_idx, update in enumerate(client_updates):
            if update is None or not update.weights:
                continue

            if _lira_observation_surface == "global":
                # Sprint 10zz+94: riusa lo STESSO modello aggregato per ogni
                # cluster in questo round invece del modello specifico del
                # client — vedi commento al punto di costruzione sopra. Tutto
                # il resto (selezione membri per cluster, calibrazione shadow,
                # diagnostica) resta invariato: cambia solo la fonte di
                # target_loss.
                client_model = _lira_global_model
            else:
                # Load client's submitted update (post-privatize when DP enabled).
                client_model = Autoencoder(input_dim=input_dim, **_autoencoder_arch_kwargs(cfg))
                if not _load_weights_into(client_model, update.weights):
                    logger.warning(
                        f"LiRA round {round_num} {update.cluster_id}: "
                        f"weights shape mismatch — skip client"
                    )
                    continue
                client_model.eval()

            # Sprint 10zz+170: modelli riscalati ref + s*(osservato - ref).
            _rescaled_models: dict[float, Any] = {}
            if round_canary_rescaled_loss:
                _ref_w = _rescale_ref_per_client.get(getattr(update, "cluster_id", None))
                if _ref_w is None:
                    logger.warning(
                        f"LiRA round {round_num} {getattr(update, 'cluster_id', None)}: "
                        "riferimento di inizio round assente, niente modelli riscalati"
                    )
                else:
                    _ae_keys = list(client_model.state_dict().keys())
                    for _s in _canary_rescale:
                        _w_s = _rescale_update_weights(update.weights, _ref_w, _s, _ae_keys)
                        _m_s = Autoencoder(input_dim=input_dim, **_autoencoder_arch_kwargs(cfg))
                        if _w_s is not None and _load_weights_into(_m_s, _w_s):
                            _m_s.eval()
                            _rescaled_models[_s] = _m_s

            # Fix: usa l'ensemble shadow del cluster di QUESTO client — non un ensemble
            # cross-cluster globale — per calibrare IN/OUT sotto lo stesso regime di
            # training del modello attaccato (vedi docstring "shadow/target mismatch").
            _client_cluster_id = getattr(update, "cluster_id", None)
            _cluster_shadow_mse     = shadow_mse_matrix_per_cluster.get(_client_cluster_id, [])
            _cluster_shadow_in_sets = shadow_in_idx_sets_per_cluster.get(_client_cluster_id, [])
            _cluster_mu_in_fb, _cluster_sigma_in_fb = global_in_stats_per_cluster.get(
                _client_cluster_id, (0.05, 0.01)
            )
            # Fix 2026-07-21e: pooled OUT floor — see comment above global_out_stats_per_cluster.
            _cluster_mu_out_fb, _cluster_sigma_out_fb = global_out_stats_per_cluster.get(
                _client_cluster_id, (0.05, 0.01)
            )
            # Fix 2026-08-15 — floor simmetrico adattivo (indagine anomalia
            # no-DP, confermata da test diretto lo stesso giorno: vedi
            # lira_debug_* nel docstring di run_lira, sezione "Diagnostica").
            # _cluster_sigma_in_fb e _cluster_sigma_out_fb sono costruiti da
            # popolazioni STRUTTURALMENTE diverse: quello IN pooling solo le
            # coppie (campione membro, shadow che l'ha visto) — un sottoinsieme
            # più ristretto e omogeneo — quello OUT pooling OGNI non-membro
            # (che per costruzione è sempre OUT per tutti gli shadow) più i
            # membri sul loro lato OUT — una popolazione molto più ampia e
            # variabile tra sessioni diverse. Il test diagnostico 2026-08-15
            # (sweep breve --no-dp, 3 round, seed 42) ha misurato σ_out_fb
            # sistematicamente ~30× più grande di σ_in — non un collasso del
            # floor assoluto (1e-4), ma un disallineamento strutturale tra le
            # due popolazioni pooled. Conseguenza: log_p_in (diviso per un σ
            # minuscolo) domina il punteggio finale ed è instabile — piccole
            # fluttuazioni di target_loss producono oscillazioni enormi da un
            # round all'altro (osservato: gap da +3 a -6.5), mascherando il
            # segnale vero ma piccolo (raw_loss_gap, confermato coerente in
            # segno per tutti e 3 i round del test). Fix: un floor CONDIVISO,
            # il maggiore tra i due fallback esistenti, applicato SIMMETRICAMENTE
            # a σ_in e σ_out — non tocca μ_in/μ_out (il segnale discriminativo
            # vero, invariato) e non tocca il lato che ha già il floor più alto
            # (rimane esattamente come prima); alza solo il lato strutturalmente
            # più piccolo, impedendo che UN SOLO lato domini il rapporto /σ².
            # ATTENZIONE: questo fix si applica a OGNI esperimento LiRA, non
            # solo al no-DP — invalida i numeri di central-sweep/dp-sweep/
            # local-sweep raccolti finora (decisione esplicita dell'utente,
            # 2026-08-15, dopo aver visto la conferma diagnostica). Vedi
            # Task #1 (già in sospeso) — l'intera campagna va ripetuta.
            _cluster_sigma_symmetric_floor = (
                max(_cluster_sigma_in_fb, _cluster_sigma_out_fb)
                if _lira_floor_mode == "symmetric"
                else 0.0
            )  # vedi cfg["lira"]["floor_mode"], Sprint 10zz+87 — 0.0 rende inerte
               # ogni max(..., _cluster_sigma_symmetric_floor) sotto (le 4 sedi
               # d'uso restano invariate), riportando ciascun lato al proprio
               # floor indipendente per il confronto diagnostico A/B richiesto
            if not _cluster_shadow_mse:
                logger.warning(
                    f"LiRA round {round_num} {_client_cluster_id}: nessun ensemble shadow "
                    "per questo cluster — skip client"
                )
                continue

            for j, sample in enumerate(eval_samples):
                try:
                    row         = [float(sample[f]) for f in _mia_feature_names(cfg)]
                    tensor      = torch.tensor([row], dtype=torch.float32)
                    with torch.no_grad():
                        recon       = client_model(tensor)
                        target_loss = float(torch.mean((recon - tensor) ** 2).item())
                except (KeyError, TypeError, ValueError):
                    if id(sample) in _sample_canary_group:
                        _diag_canary_skip_reason["tensor_extraction"] += 1
                    continue

                is_member = (j < len(members_bal))
                # Fix strutturale 2026-08-21: `_train_idx`/`sample_train_idx`
                # rimossi — vedi commento alla loro rimozione sopra. Il
                # controllo IN/OUT più sotto usa direttamente id(sample) in
                # in_set, valido per entrambe le classi.

                # Cross-cluster guard: only evaluate a member against its home cluster's client.
                # A highway sample vs an urban client always gives high target_loss (never seen)
                # → looks like a non-member → 3/4 of member evaluations are false negatives
                # → AUC < 0.5 even without DP. Skip mismatched (sample, client) pairs.
                if is_member:
                    _member_cluster = _sample_to_cluster.get(id(sample))
                    _client_cluster = getattr(update, "cluster_id", None)
                    if _member_cluster is not None and _member_cluster != _client_cluster:
                        if id(sample) in _sample_canary_group:
                            _diag_canary_skip_reason["cross_cluster_guard"] += 1
                        continue
                else:
                    # Fix 2026-08-11: guard simmetrico — vedi commento su
                    # _holdout_sample_to_cluster sopra. Senza questo, ogni
                    # non-member entra nel pool fino a una volta per cluster
                    # (3x), mescolando client con scale di loss potenzialmente
                    # diverse — inerte sotto clipping per-client, ma diluisce
                    # l'AUC pooled sotto --no-dp dove non c'è alcun vincolo di
                    # norma a mantenere le scale dei client comparabili.
                    _nonmember_cluster = _holdout_sample_to_cluster.get(id(sample))
                    _client_cluster = getattr(update, "cluster_id", None)
                    if _nonmember_cluster is not None and _nonmember_cluster != _client_cluster:
                        if id(sample) in _sample_canary_group:
                            _diag_canary_skip_reason["cross_cluster_guard"] += 1
                        continue

                # Split shadow losses: IN = shadows (di QUESTO cluster, QUESTO round)
                # che hanno visto il campione; OUT = resto.
                # Fix strutturale 2026-08-21: prima `is_member and ... in
                # in_set` — un non-membro finiva SEMPRE in out_losses, mai in
                # in_losses, per costruzione del vecchio pool (solo membri).
                # Ora `in_set` contiene id() Python campionati dall'universo
                # shadow COMBINATO (membri+non-membri, Step 2 sopra) — il
                # controllo `id(sample) in in_set` è identico per entrambe le
                # classi: un non-membro campionato come IN da uno shadow
                # finisce correttamente in in_losses, esattamente come un
                # membro nella stessa situazione.
                # ------------------------------------------------------------
                # Raccolta raw-loss dei canary — DEVE stare qui (Sprint
                # 10zz+119, 2026-09-16). Bug reale trovato confrontando la
                # cella a 32 shadow con quella a 8: stesso modello target
                # (traiettoria di loss identica), raw AUC media 0.6475 contro
                # 0.6926. Causa: gli append erano piu' in basso nel loop, DOPO
                # il `continue` di `insufficient_calibration`, quindi un canary
                # scartato dal filtro di calibrazione LiRA spariva anche dalla
                # metrica raw. Con n_shadow=8 il filtro toglieva 4 record (2
                # template non-membro) su s42: 20x18=360 coppie invece di
                # 20x20=400. La metrica raw risultava percio' indipendente
                # dagli shadow nel PUNTEGGIO ma non nel POOL DI VALUTAZIONE —
                # esattamente il contrario di quanto dichiarato nei commenti
                # dei config, in docs/CanaryPositiveControl.md e nel paper.
                # Effetto sui numeri: i baseline a init casuale girano senza
                # shadow, quindi su 400 coppie piene, mentre i run post-training
                # su 360-400: il Delta non era appaiato e risultava gonfiato
                # (s42: +0.165 riportato contro +0.120 reale).
                #
                # Il guard cross-cluster sopra resta PRIMA di questa raccolta,
                # ed e' corretto che sia cosi': valutare un membro contro il
                # modello di un altro client produce una loss priva di
                # significato, non un dato filtrato arbitrariamente.
                if id(sample) in _sample_canary_group:
                    if is_member:
                        round_canary_member_raw_loss.append(target_loss)
                        round_canary_member_raw_groups.add(_sample_canary_group[id(sample)])
                    else:
                        round_canary_nonmember_raw_loss.append(target_loss)
                        round_canary_nonmember_raw_groups.add(_sample_canary_group[id(sample)])
                    # Sprint 10zz+170: stesse coppie del pool raw, modello riscalato.
                    for _s, _m_s in _rescaled_models.items():
                        with torch.no_grad():
                            _rec_s = _m_s(tensor)
                            _loss_s = float(torch.mean((_rec_s - tensor) ** 2).item())
                        round_canary_rescaled_loss[_s][0 if is_member else 1].append(_loss_s)

                _sample_id = id(sample)
                in_losses:  list[float] = []
                out_losses: list[float] = []
                _rmia_refs: list[tuple] = []  # RMIA (2026-10-13)
                for si, in_set in enumerate(_cluster_shadow_in_sets):
                    if si >= len(_cluster_shadow_mse):
                        continue
                    mse = _cluster_shadow_mse[si][j]
                    if mse is None:
                        continue
                    if _sample_id in in_set:
                        in_losses.append(mse)
                    else:
                        out_losses.append(mse)
                    _rmia_refs.append((si, float(mse), _sample_id in in_set))

                # RMIA (2026-10-13): raccolto PRIMA del filtro di
                # calibrazione di LiRA; basta un riferimento.
                if _rmia_refs:
                    _rmia_rows.append({
                        "target": target_loss, "refs": _rmia_refs,
                        "member": bool(is_member),
                        "client": _client_idx, "sid": id(sample),
                        "canary": _sample_canary_group.get(id(sample)),
                    })

                if len(out_losses) < 2:
                    if id(sample) in _sample_canary_group:
                        _diag_canary_skip_reason["insufficient_calibration"] += 1
                    continue  # insufficient calibration data

                μ_out = float(np.mean(out_losses))
                # σ floor (fix 2026-07-21e): the per-sample cross-shadow std alone can
                # collapse to ~0 when all n_shadow (~8) shadows agree on this one point
                # (typically right after a shared warm-start, before they've diverged —
                # see comment on global_out_stats_per_cluster above). Floor it with the
                # larger of: the μ×0.05 scale-adaptive floor (original guard, prevents
                # 1e-8 collapse), and the pooled per-cluster/per-round OUT spread
                # (reflects the actual point-to-point + shadow-to-shadow variability
                # observed this round, not just this one sample's near-zero spread).
                σ_out = max(
                    float(np.std(out_losses)),
                    max(μ_out * 0.05, 1e-4),
                    _cluster_sigma_out_fb,
                    _cluster_sigma_symmetric_floor,  # fix 2026-08-15 — vedi commento sopra
                )
                # DIAGNOSTICA 2026-08-15 (vedi commento su _diag_* sopra):
                # σ_out effettivo (post-floor) + se il floor (non lo std
                # grezzo cross-shadow) ha vinto il max() — un floor-hit-rate
                # sistematicamente più alto sotto no-DP confermerebbe il
                # floor collapse ipotizzato.
                _diag_sigma_out_values.append(σ_out)
                if σ_out > float(np.std(out_losses)):
                    _diag_sigma_out_floor_hits += 1
                # DIAGNOSTICA 2026-08-20: μ_out per QUESTO campione (che sia
                # membro o non-membro) — confrontato sotto con μ_in SOLO per i
                # membri con calibrazione IN reale (non il fallback), per
                # testare l'ipotesi μ_in>μ_out (vedi commento sopra).
                if is_member:
                    _diag_mu_out_values.append(μ_out)

                # Use per-sample IN distribution if available; fall back to this
                # client's cluster-specific global IN stats (not a cross-cluster global).
                # Same σ floor rationale as σ_out above — applied symmetrically so
                # neither side's variance can be artificially tighter than the other
                # purely due to small-N (n_shadow≈8) per-sample estimation noise.
                _mu_in_is_real = len(in_losses) >= 2  # DIAGNOSTICA 2026-08-20, vedi dump sotto
                if len(in_losses) >= 2:
                    μ_in = float(np.mean(in_losses))
                    σ_in = max(
                        float(np.std(in_losses)),
                        max(μ_in * 0.05, 1e-4),
                        _cluster_sigma_in_fb,
                        _cluster_sigma_symmetric_floor,  # fix 2026-08-15 — vedi commento sopra
                    )
                    # DIAGNOSTICA 2026-08-15 — vedi commento su σ_out sopra,
                    # stesso ragionamento. Il ramo else (fallback cluster,
                    # riga sotto) non ha uno std grezzo da confrontare (usa
                    # direttamente _cluster_sigma_in_fb) — floor-hit non
                    # applicabile in quel caso, ma σ_in viene comunque
                    # registrato in _diag_sigma_in_values in entrambi i rami.
                    if σ_in > float(np.std(in_losses)):
                        _diag_sigma_in_floor_hits += 1
                    # DIAGNOSTICA 2026-08-20: solo qui μ_in è una stima REALE
                    # per QUESTO campione (non il fallback _cluster_mu_in_fb) —
                    # confronto diretto e onesto con μ_out dello stesso campione.
                    _diag_mu_in_values.append(μ_in)
                    # DIAGNOSTICA 2026-08-21 (test mirato, vedi inizializzazione
                    # di _diag_counterfactual_member_scores sopra): SOLO per
                    # logging, calcolo lo score che questo membro avrebbe
                    # ricevuto se — invece della sua calibrazione IN reale —
                    # fosse stato forzato nella stessa formula di ancoraggio
                    # usata per i non-membri (μ_out + gap tipico). Non
                    # modifica μ_in/σ_in/log_p_* reali usati sotto per il
                    # punteggio effettivo del pool.
                    if is_member:
                        _cf_mu_in = μ_out + (_cluster_mu_in_fb - _cluster_mu_out_fb)
                        if _cf_mu_in <= 0.0:   # stesso clamp del ramo reale (Sprint 10zz+111)
                            _cf_mu_in = max(μ_out * 0.01, 1e-12)
                        _cf_sigma_in = max(_cluster_sigma_in_fb, _cluster_sigma_symmetric_floor)
                        _cf_log_p_in = (
                            -0.5 * ((target_loss - _cf_mu_in) / _cf_sigma_in) ** 2
                        ) - np.log(_cf_sigma_in)
                        _cf_log_p_out = (
                            -0.5 * ((target_loss - μ_out) / σ_out) ** 2
                        ) - np.log(σ_out)
                        _diag_counterfactual_member_scores.append(
                            float(np.clip(_cf_log_p_in - _cf_log_p_out, -20.0, 20.0))
                        )
                else:
                    # Fix 2026-08-21 — quarto/quinto round di questa stessa
                    # indagine (l'esclusione outlier dell'8σ, 2026-08-20, ha
                    # tolto solo il 2.37% dei campioni e NON ha risolto
                    # l'inversione: lira_auc_roc restava a 0.32-0.44 col
                    # warning ANOMALY del pipeline). Il dump per-campione ha
                    # mostrato la causa dominante: per i non-membri (che non
                    # hanno MAI una calibrazione IN reale, per definizione —
                    # nessuno shadow addestra mai su dati hold-out) μ_in era
                    # la costante ASSOLUTA _cluster_mu_in_fb, mentre μ_out era
                    # sempre una stima PER CAMPIONE — un confronto non
                    # simmetrico rispetto alla difficoltà di ricostruzione
                    # specifica di quella sessione (alcune sessioni sono
                    # intrinsecamente più difficili di altre; μ_out lo
                    # rifletteva correttamente per-campione, la costante no).
                    # Risultato osservato: sessioni "facili" (μ_out piccolo)
                    # finivano sistematicamente più vicine alla costante
                    # (relativamente più alta) che al proprio μ_out — punteggio
                    # positivo, "sembra un membro", per un artefatto di scala,
                    # non per vera membership (quasi tutti i non-membri nel
                    # dump del 2026-08-20 mostravano score positivo).
                    # Fix: ancora μ_in al μ_out DI QUESTO campione specifico
                    # (che riflette la sua vera difficoltà), spostato del gap
                    # TIPICO osservato tra IN e OUT nei membri reali di questo
                    # cluster/round (_cluster_mu_in_fb - _cluster_mu_out_fb,
                    # entrambi già pooled — normalmente negativo: IN più
                    # basso di OUT, la direzione della vera memorizzazione).
                    # Così un non-membro "facile" ottiene un μ_in stimato
                    # coerentemente basso (vicino al proprio μ_out, meno il
                    # tipico vantaggio IN), non un valore assoluto scollegato
                    # dalla propria scala. Se questo campione HA una
                    # calibrazione IN reale (ramo sopra) non viene toccato.

                    # Fix 2026-09-15 (Sprint 10zz+111, task #161 — bug reale
                    # osservato nei dump del run `_canary_pairedsplit_nodp_swap`:
                    # `mu_in=-0.000103(FALLBACK)`, `-0.000162`, `-0.000149`).
                    # μ_in e' una media di loss QUADRATICHE e non puo' essere
                    # negativa. L'ancoraggio relativo
                    # μ_out + (_cluster_mu_in_fb - _cluster_mu_out_fb) va sotto
                    # zero ogni volta che il delta di cluster e' negativo e
                    # supera in modulo il μ_out del campione — cioe' proprio sui
                    # campioni con loss piu' bassa, che sono quelli che LiRA
                    # dovrebbe classificare come membri. Una gaussiana centrata
                    # su un valore impossibile sposta sistematicamente lo score
                    # di quei campioni, in una direzione che dipende dal segno
                    # del delta e non dall'appartenenza. Clamp a un valore
                    # positivo piccolo: il floor e' relativo a μ_out cosi' da non
                    # introdurre una costante arbitraria estranea alla scala del
                    # cluster, con un pavimento assoluto per il caso μ_out=0.
                    # Contatore diagnostico esposto come
                    # `lira_debug_mu_in_fallback_clamp_rate` per quantificare
                    # quanto spesso il problema si presentava.
                    μ_in = μ_out + (_cluster_mu_in_fb - _cluster_mu_out_fb)
                    if μ_in <= 0.0:
                        _diag_mu_in_clamped += 1
                        μ_in = max(μ_out * 0.01, 1e-12)
                    # fix 2026-08-15: anche il fallback diretto deve rispettare
                    # il floor simmetrico, altrimenti un client con troppo
                    # pochi in_losses (<2) tornerebbe silenziosamente al vecchio
                    # comportamento asimmetrico.
                    σ_in = max(_cluster_sigma_in_fb, _cluster_sigma_symmetric_floor)
                _diag_sigma_in_values.append(σ_in)

                # Fix 2026-08-20 — vedi commento esteso all'inizializzazione
                # di _UNCALIBRATED_Z_THRESHOLD sopra. Se target_loss è oltre
                # 8σ da ENTRAMBE le distribuzioni, né l'ensemble IN né quello
                # OUT spiegano l'osservazione — il campione va escluso, non
                # forzato con un segno arbitrario (guarda l'algebra: quando
                # |t-μ_in| e |t-μ_out| sono entrambi enormi rispetto a σ, il
                # segno di log_p_in-log_p_out dipende da quale dei due è per
                # puro caso marginalmente più vicino, non da un vero segnale).
                _diag_scored_total += 1
                _z_in  = abs(target_loss - μ_in)  / σ_in  if σ_in  > 0 else 0.0
                _z_out = abs(target_loss - μ_out) / σ_out if σ_out > 0 else 0.0
                if min(_z_in, _z_out) > _UNCALIBRATED_Z_THRESHOLD:
                    _diag_uncalibrated_skipped += 1
                    continue

                # Gaussian log-likelihood ratio (Carlini 2022, Eq. 2):
                # score > 0 → loss matches IN distribution → member
                #
                # Sprint 10zz+90 (2026-09-15) — se cfg["lira"]["member_scoring"]
                # == "matched_formula" (vedi commento all'inizializzazione di
                # _lira_member_scoring sopra), un MEMBRO con calibrazione IN
                # reale (_mu_in_is_real True) viene comunque forzato nella
                # STESSA formula di ancoraggio usata per i non-membri (μ_out di
                # QUESTO campione + il gap tipico IN/OUT del cluster/round) —
                # esattamente la stessa formula già calcolata come diagnostica
                # di sola lettura in _cf_mu_in/_cf_sigma_in sopra, qui promossa
                # a punteggio ufficiale. Un non-membro non cambia mai (è già
                # sempre nel ramo formula-anchored per costruzione). Default
                # "real": nessun cambiamento, questo blocco non viene mai
                # eseguito per config/run esistenti.
                if _lira_member_scoring == "matched_formula" and _mu_in_is_real:
                    _mf_mu_in = μ_out + (_cluster_mu_in_fb - _cluster_mu_out_fb)
                    if _mf_mu_in <= 0.0:   # stesso clamp del ramo reale (Sprint 10zz+111)
                        _mf_mu_in = max(μ_out * 0.01, 1e-12)
                    _mf_sigma_in = max(_cluster_sigma_in_fb, _cluster_sigma_symmetric_floor)
                    log_p_in = (
                        -0.5 * ((target_loss - _mf_mu_in) / _mf_sigma_in) ** 2
                    ) - np.log(_mf_sigma_in)
                else:
                    log_p_in = (-0.5 * ((target_loss - μ_in) / σ_in) ** 2) - np.log(σ_in)
                log_p_out = (-0.5 * ((target_loss - μ_out) / σ_out) ** 2) - np.log(σ_out)
                # DIAGNOSTICA 2026-08-20 (round 3, dump per-campione): il
                # round 2 (aggregate μ_in<μ_out corretto ma AUC ancora
                # invertito) non si spiega con le sole medie — serve vedere i
                # valori REALI congiunti (t, μ_in, μ_out, log_p_*, score) per
                # singolo campione, limitato al primo client processato in
                # questo round e ai primi 15 membri + 15 non-membri incontrati
                # (vedi _diag_dump_* inizializzati sopra). is_member non è
                # ancora definito qui (arriva 3 righe sotto) — lo calcolo
                # localmente identico a quello che verrà usato.
                if _client_idx == 0:
                    _dump_is_member = (j < len(members_bal))
                    _dump_this = (
                        (_dump_is_member and _diag_dump_mem_count < _DIAG_DUMP_CAP) or
                        (not _dump_is_member and _diag_dump_nonmem_count < _DIAG_DUMP_CAP)
                    )
                    if _dump_this:
                        if _dump_is_member:
                            _diag_dump_mem_count += 1
                        else:
                            _diag_dump_nonmem_count += 1
                        logger.info(
                            f"[DUMP r{round_num}] is_member={_dump_is_member} "
                            f"t={target_loss:.6f} mu_in={μ_in:.6f}"
                            f"({'reale' if _mu_in_is_real else 'FALLBACK'}) "
                            f"mu_out={μ_out:.6f} sigma_in={σ_in:.6f} sigma_out={σ_out:.6f} "
                            f"log_p_in={log_p_in:.4f} log_p_out={log_p_out:.4f} "
                            f"score={log_p_in - log_p_out:.4f}"
                        )
                # Clip to ±20: log-LR beyond this range has no practical discriminative
                # value and only amplifies numerical instabilities in edge cases.
                lira_score = float(np.clip(log_p_in - log_p_out, -20.0, 20.0))

                if np.isnan(lira_score) or np.isinf(lira_score):
                    continue

                # Sprint 10zz+33 (2026-09-03, task #58) — vedi commento
                # all'inizializzazione di round_sablayrolles_member_scores
                # sopra. score>0 → loss sotto la soglia (μ_in+μ_out)/2 →
                # membro, stessa convenzione di segno di lira_score (usa gli
                # stessi μ_in/μ_out/σ già validati dal controllo 8σ sopra —
                # nessun campione "non calibrabile" entra qui che non sia
                # già stato escluso anche per LiRA).
                sablayrolles_score = _sablayrolles_score(μ_in, μ_out, target_loss)
                if np.isnan(sablayrolles_score) or np.isinf(sablayrolles_score):
                    sablayrolles_score = None

                # Sprint 10zz+36 (2026-09-03, task #61) — vedi commento
                # all'inizializzazione di round_lira_log_member_scores sopra
                # e _lira_log_score() per la spiegazione completa. Usa
                # in_losses/out_losses (raw, già raccolti sopra per il fit
                # normale) — nessuna raccolta dati aggiuntiva. Floor-hit
                # tracciato qui (non dentro _lira_log_score(), che non
                # espone lo stato interno) per il confronto diretto con
                # _diag_sigma_in_floor_hits/_diag_sigma_out_floor_hits del
                # fit raw sopra — la domanda specifica posta dall'utente
                # ("il log-transform riduce il floor-hit-rate?").
                _LIRA_LOG_EPS = 1e-8
                _LIRA_LOG_SIGMA_FLOOR = 0.05
                _log_out_losses = [math.log(x + _LIRA_LOG_EPS) for x in out_losses]
                _sigma_out_log_raw = float(np.std(_log_out_losses))
                _diag_sigma_out_log_values.append(max(_sigma_out_log_raw, _LIRA_LOG_SIGMA_FLOOR))
                if _sigma_out_log_raw < _LIRA_LOG_SIGMA_FLOOR:
                    _diag_sigma_out_log_floor_hits += 1
                if len(in_losses) >= 2:
                    _log_in_losses = [math.log(x + _LIRA_LOG_EPS) for x in in_losses]
                    _sigma_in_log_raw = float(np.std(_log_in_losses))
                    _diag_sigma_in_log_values.append(max(_sigma_in_log_raw, _LIRA_LOG_SIGMA_FLOOR))
                    if _sigma_in_log_raw < _LIRA_LOG_SIGMA_FLOOR:
                        _diag_sigma_in_log_floor_hits += 1
                else:
                    # Fallback: sigma_in_log = sigma_out_log (vedi
                    # _lira_log_score) — stesso floor-hit del lato out, non
                    # double-counted separatamente.
                    _diag_sigma_in_log_values.append(max(_sigma_out_log_raw, _LIRA_LOG_SIGMA_FLOOR))
                    if _sigma_out_log_raw < _LIRA_LOG_SIGMA_FLOOR:
                        _diag_sigma_in_log_floor_hits += 1

                lira_log_score = _lira_log_score(
                    in_losses, out_losses, target_loss,
                    eps=_LIRA_LOG_EPS, sigma_floor=_LIRA_LOG_SIGMA_FLOOR,
                )
                if math.isnan(lira_log_score) or math.isinf(lira_log_score):
                    lira_log_score = None

                if is_member:
                    round_member_scores.append(lira_score)
                    # DIAGNOSTICA 2026-08-15: target_loss GREZZO, a monte di
                    # qualunque normalizzazione μ/σ — il test più diretto per
                    # distinguere "il modello davvero non memorizza" (anche
                    # qui indistinguibile) da "il punteggio finale è un
                    # artefatto di amplificazione /σ²" (qui separato, ma il
                    # log-ratio no).
                    _diag_raw_loss_members.append(target_loss)
                    if sablayrolles_score is not None:
                        round_sablayrolles_member_scores.append(sablayrolles_score)
                    if lira_log_score is not None:
                        round_lira_log_member_scores.append(lira_log_score)
                else:
                    round_nonmember_scores.append(lira_score)
                    _diag_raw_loss_nonmembers.append(target_loss)
                    if sablayrolles_score is not None:
                        round_sablayrolles_nonmember_scores.append(sablayrolles_score)
                    if lira_log_score is not None:
                        round_lira_log_nonmember_scores.append(lira_log_score)

                # Canary positive control (Sprint 10vv): bucketing puramente
                # additivo, in parallelo a round_member_scores/
                # round_nonmember_scores sopra — stesso lira_score già
                # calcolato, nessuna formula diversa. No-op se sample non è
                # taggato (id(sample) assente da _sample_canary_group).
                # NOTA (Sprint 10zz+119): qui restano SOLO gli score LiRA e i
                # gruppi distinti del pool LiRA. La raccolta raw-loss e' stata
                # spostata piu' in alto, prima del filtro
                # `insufficient_calibration` — vedi il commento esteso li'.
                # Non reintrodurre gli append raw in questo blocco: li' e' dopo
                # il `continue`, e il pool raw tornerebbe a dipendere dagli
                # shadow.
                if id(sample) in _sample_canary_group:
                    if is_member:
                        round_canary_member_scores.append(lira_score)
                        round_canary_member_groups.add(_sample_canary_group[id(sample)])
                    else:
                        round_canary_nonmember_scores.append(lira_score)
                        round_canary_nonmember_groups.add(_sample_canary_group[id(sample)])

                if composed_output is not None:
                    _sid = id(sample)
                    _cumulative_scores[_sid] = _cumulative_scores.get(_sid, 0.0) + lira_score
                    _cumulative_score_counts[_sid] = _cumulative_score_counts.get(_sid, 0) + 1

        if not round_member_scores or not round_nonmember_scores:
            logger.warning(
                f"LiRA round {round_num}: score pool vuoto "
                f"(m={len(round_member_scores)}, nm={len(round_nonmember_scores)}) — skip"
            )
            continue

        labels = [1] * len(round_member_scores) + [0] * len(round_nonmember_scores)
        scores = round_member_scores + round_nonmember_scores

        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = 0.5

        # TPR@low-FPR (roadmap #4, Sprint 10pp 2026-08-28) — stessa coppia
        # labels/scores già usata per l'AUC, nessun costo aggiuntivo.
        _tpr_fields = _tpr_at_fixed_fpr(labels, scores)

        # MIA Advantage (task #41, Sprint 10zz+13, 2026-09-02) — stessa
        # coppia labels/scores, vedi _mia_advantage() per la formula/motivazione.
        lira_advantage = _mia_advantage(labels, scores)
        # Conteggi TP/FP/TN/FN alla stessa soglia ottimale (task #49,
        # Sprint 10zz+25, 2026-09-03) — vedi _mia_confusion_at_best_threshold().
        lira_confusion = _mia_confusion_at_best_threshold(labels, scores)

        # Sprint 10zz+33 (2026-09-03, task #58) — Sablayrolles et al. 2019
        # [56], vedi commento all'inizializzazione di
        # round_sablayrolles_member_scores sopra. Stesso identico set di
        # campioni di round_member_scores/round_nonmember_scores (calcolato
        # nello stesso passaggio del loop, dopo lo stesso filtro 8σ) — quindi
        # None solo se anche il pool LiRA fosse vuoto (già escluso dal guard
        # sopra), non serve un controllo separato.
        sablayrolles_auc_roc = None
        sablayrolles_advantage = None
        sablayrolles_confusion = None
        _sablayrolles_tpr_fields: dict[str, Any] = {}
        if round_sablayrolles_member_scores and round_sablayrolles_nonmember_scores:
            _sab_labels = [1] * len(round_sablayrolles_member_scores) + [0] * len(round_sablayrolles_nonmember_scores)
            _sab_scores = round_sablayrolles_member_scores + round_sablayrolles_nonmember_scores
            try:
                sablayrolles_auc_roc = round(float(roc_auc_score(_sab_labels, _sab_scores)), 6)
            except ValueError:
                sablayrolles_auc_roc = None
            sablayrolles_advantage = _mia_advantage(_sab_labels, _sab_scores)
            sablayrolles_confusion = _mia_confusion_at_best_threshold(_sab_labels, _sab_scores)
            _sablayrolles_tpr_fields = {
                f"sablayrolles_{k}": v for k, v in _tpr_at_fixed_fpr(_sab_labels, _sab_scores).items()
            }

        # Sprint 10zz+36 (2026-09-03, task #61) — variante esplorativa LiRA
        # su log(MSE), vedi commento all'inizializzazione di
        # round_lira_log_member_scores sopra e _lira_log_score() per il
        # dettaglio/le semplificazioni dichiarate. Stesso pattern di
        # Sablayrolles sopra — AUC/Advantage/Confusion/TPR@low-FPR sullo
        # stesso tipo di pool, più il floor-hit-rate log-space (la domanda
        # specifica posta dall'utente) per il confronto diretto con
        # lira_debug_sigma_in_floor_hit_rate/lira_debug_sigma_out_floor_hit_rate
        # (fit raw, sopra).
        lira_log_auc_roc = None
        lira_log_advantage = None
        lira_log_confusion = None
        _lira_log_tpr_fields: dict[str, Any] = {}
        if round_lira_log_member_scores and round_lira_log_nonmember_scores:
            _log_labels = [1] * len(round_lira_log_member_scores) + [0] * len(round_lira_log_nonmember_scores)
            _log_scores = round_lira_log_member_scores + round_lira_log_nonmember_scores
            try:
                lira_log_auc_roc = round(float(roc_auc_score(_log_labels, _log_scores)), 6)
            except ValueError:
                lira_log_auc_roc = None
            lira_log_advantage = _mia_advantage(_log_labels, _log_scores)
            lira_log_confusion = _mia_confusion_at_best_threshold(_log_labels, _log_scores)
            _lira_log_tpr_fields = {
                f"lira_log_{k}": v for k, v in _tpr_at_fixed_fpr(_log_labels, _log_scores).items()
            }
        _lira_log_diag_fields: dict[str, Any] = {}
        if _diag_sigma_in_log_values:
            _lira_log_diag_fields["lira_log_debug_sigma_in_mean"] = round(
                float(np.mean(_diag_sigma_in_log_values)), 8
            )
            _lira_log_diag_fields["lira_log_debug_sigma_in_floor_hit_rate"] = round(
                _diag_sigma_in_log_floor_hits / len(_diag_sigma_in_log_values), 4
            )
        if _diag_sigma_out_log_values:
            _lira_log_diag_fields["lira_log_debug_sigma_out_mean"] = round(
                float(np.mean(_diag_sigma_out_log_values)), 8
            )
            _lira_log_diag_fields["lira_log_debug_sigma_out_floor_hit_rate"] = round(
                _diag_sigma_out_log_floor_hits / len(_diag_sigma_out_log_values), 4
            )

        # Canary positive control (Sprint 10vv): AUC calcolato SOLO sui
        # campioni canary (membri duplicati vs gemelli non-membro mai
        # visti in training) — None se questo round non ha canary
        # taggati in entrambe le classi (ogni run esistente/pubblicato,
        # o un round in cui il pool canary risultasse per qualche motivo
        # sbilanciato). Stessa formula roc_auc_score dell'AUC principale,
        # solo su un sottoinsieme diverso di (label, score) — nessuna
        # soglia/pooling/formula nuova.
        canary_auc_roc = None
        canary_advantage = None
        canary_confusion = None
        if round_canary_member_scores and round_canary_nonmember_scores:
            _canary_labels = [1] * len(round_canary_member_scores) + [0] * len(round_canary_nonmember_scores)
            _canary_scores = round_canary_member_scores + round_canary_nonmember_scores
            try:
                canary_auc_roc = round(float(roc_auc_score(_canary_labels, _canary_scores)), 6)
            except ValueError:
                canary_auc_roc = None
            # Advantage (task #41) anche sul pool canary — ancorato a UNA
            # soglia invece che mediato su tutte, può divergere dall'AUC
            # instabile già osservato qui (Sprint 10yy) e dare un secondo
            # angolo diagnostico sulla stessa domanda, a costo zero.
            canary_advantage = _mia_advantage(_canary_labels, _canary_scores)
            # Conteggi TP/FP/TN/FN sul pool canary (task #49, Sprint 10zz+25) —
            # qui il conteggio assoluto è particolarmente leggibile: n piccolo
            # (n_member=150, n_nonmember~19-20/round), un tasso da solo dice
            # meno di "rilevati X canary su 150".
            canary_confusion = _mia_confusion_at_best_threshold(_canary_labels, _canary_scores)

        # Diagnostico raw-loss canary (Sprint 10ww): stessa idea di
        # lira_debug_raw_mse_auc_roc sopra (score = -loss, loss più bassa =
        # più "membro-simile"), ristretta ai soli canary — a monte di
        # qualunque calibrazione shadow, quindi immune a un'eventuale
        # contaminazione degli shadow dagli stessi duplicati canary.
        canary_raw_mse_auc_roc = None
        canary_raw_advantage = None
        canary_raw_confusion = None
        if round_canary_member_raw_loss and round_canary_nonmember_raw_loss:
            _canary_raw_labels = [1] * len(round_canary_member_raw_loss) + [0] * len(round_canary_nonmember_raw_loss)
            _canary_raw_scores = [-x for x in round_canary_member_raw_loss] + [-x for x in round_canary_nonmember_raw_loss]
            try:
                canary_raw_mse_auc_roc = round(float(roc_auc_score(_canary_raw_labels, _canary_raw_scores)), 6)
            except ValueError:
                canary_raw_mse_auc_roc = None
            # Sprint 10zz+28 (2026-09-03, task #53) — stesso motivo di
            # canary_advantage/canary_confusion sopra, qui applicato al
            # diagnostico raw-loss (a monte della calibrazione shadow,
            # quindi il confronto "che soglia userebbe l'attaccante" è
            # significativo anche qui, non solo sul punteggio calibrato).
            canary_raw_advantage = _mia_advantage(_canary_raw_labels, _canary_raw_scores)
            canary_raw_confusion = _mia_confusion_at_best_threshold(
                _canary_raw_labels, _canary_raw_scores
            )

        # Sprint 10zz+29 (2026-09-03, task #54) — vedi _full_roc_curve().
        if roc_curve_dump_path is not None:
            _round_curves: dict[str, Any] = {}
            _lira_curve = _full_roc_curve(labels, scores)
            if _lira_curve is not None:
                _round_curves["lira"] = _lira_curve
            if round_canary_member_scores and round_canary_nonmember_scores:
                _c = _full_roc_curve(_canary_labels, _canary_scores)
                if _c is not None:
                    _round_curves["canary"] = _c
            if round_canary_member_raw_loss and round_canary_nonmember_raw_loss:
                _cr = _full_roc_curve(_canary_raw_labels, _canary_raw_scores)
                if _cr is not None:
                    _round_curves["canary_raw"] = _cr
            # Sprint 10zz+33 (2026-09-03, task #58) — curva ROC anche per
            # Sablayrolles, stessa infrastruttura di _full_roc_curve() già
            # usata sopra, cosi' scripts/plot_roc_log_scale.py può confrontare
            # le due curve (parametrica vs non-parametrica) sullo stesso
            # grafico log-log — proprio il tipo di confronto che Carlini et
            # al. fanno nella loro Table I/II.
            if round_sablayrolles_member_scores and round_sablayrolles_nonmember_scores:
                _sab_curve = _full_roc_curve(_sab_labels, _sab_scores)
                if _sab_curve is not None:
                    _round_curves["sablayrolles"] = _sab_curve
            # Sprint 10zz+36 (2026-09-03, task #61) — curva ROC anche per la
            # variante esplorativa lira_log, stesso motivo di sablayrolles
            # sopra: confronto diretto raw-vs-log sullo stesso grafico
            # log-log via --subkey lira_log.
            if round_lira_log_member_scores and round_lira_log_nonmember_scores:
                _log_curve = _full_roc_curve(_log_labels, _log_scores)
                if _log_curve is not None:
                    _round_curves["lira_log"] = _log_curve
            if _round_curves:
                _roc_curves_per_round[round_num] = _round_curves

        score_gap = float(np.mean(round_member_scores) - np.mean(round_nonmember_scores))
        logger.info(
            f"Round {round_num} — LiRA AUC: {auc:.4f} "
            f"(gap={score_gap:.6f}, n_shadow={n_shadow}, "
            f"TPR@1%FPR={_tpr_fields.get('tpr_at_fpr_0.01')})"
        )
        # Sprint 10zz+170: AUC della loss grezza dei canary sul modello riscalato.
        canary_rescaled_raw_auc_roc: dict[str, float | None] | None = None
        canary_rescaled_mean_loss: dict[str, list[float | None]] | None = None
        if round_canary_rescaled_loss:
            canary_rescaled_raw_auc_roc, canary_rescaled_mean_loss = {}, {}
            for _s, (_ml, _nl) in round_canary_rescaled_loss.items():
                _k = f"{_s:g}"
                _auc_s = None
                if _ml and _nl:
                    try:
                        _auc_s = round(float(roc_auc_score(
                            [1] * len(_ml) + [0] * len(_nl),
                            [-x for x in _ml] + [-x for x in _nl],
                        )), 6)
                    except ValueError:
                        _auc_s = None
                canary_rescaled_raw_auc_roc[_k] = _auc_s
                canary_rescaled_mean_loss[_k] = [
                    round(float(np.mean(_ml)), 8) if _ml else None,
                    round(float(np.mean(_nl)), 8) if _nl else None,
                ]
            logger.info(
                f"Round {round_num} — [CANARY RISCALATO] raw_loss_auc per s: "
                + ", ".join(f"s={k}: {v}" for k, v in canary_rescaled_raw_auc_roc.items())
                + f" (raw_loss_auc osservato={canary_raw_mse_auc_roc})"
            )
            if "1" in canary_rescaled_raw_auc_roc and canary_rescaled_raw_auc_roc["1"] != canary_raw_mse_auc_roc:
                logger.warning(
                    f"Round {round_num} — [CANARY RISCALATO] s=1 diverso dal raw_loss_auc "
                    f"osservato ({canary_rescaled_raw_auc_roc['1']} contro {canary_raw_mse_auc_roc})"
                )
        if canary_auc_roc is not None:
            logger.info(
                f"Round {round_num} — [CANARY] AUC: {canary_auc_roc:.4f} "
                f"(n_member={len(round_canary_member_scores)}, "
                f"n_nonmember={len(round_canary_nonmember_scores)}; "
                f"DISTINTI {len(round_canary_member_groups)}x"
                f"{len(round_canary_nonmember_groups)} = "
                f"{len(round_canary_member_groups) * len(round_canary_nonmember_groups)} "
                f"coppie reali) — positive control "
                f"| RAW {len(round_canary_member_raw_groups)}x"
                f"{len(round_canary_nonmember_raw_groups)} = "
                f"{len(round_canary_member_raw_groups) * len(round_canary_nonmember_raw_groups)} coppie "
                f"| raw_loss_auc={canary_raw_mse_auc_roc} "
                f"(mean_loss_member={round(float(np.mean(round_canary_member_raw_loss)), 8) if round_canary_member_raw_loss else 'N/A'}, "
                f"mean_loss_nonmember={round(float(np.mean(round_canary_nonmember_raw_loss)), 8) if round_canary_nonmember_raw_loss else 'N/A'})"
            )
        # Diagnostica 2026-09-13 (vedi commento all'inizializzazione di
        # _diag_canary_skip_reason sopra) — logga solo se almeno un canary è
        # stato effettivamente saltato in questo round, per non sporcare il
        # log di ogni run senza canary o senza skip.
        if _sample_canary_group and sum(_diag_canary_skip_reason.values()) > 0:
            logger.info(
                f"Round {round_num} — [CANARY DIAG] campioni canary saltati per motivo: "
                f"{_diag_canary_skip_reason} (su {len(_sample_canary_group)} taggati totali)"
            )

        # DIAGNOSTICA 2026-08-15 — vedi commento all'inizializzazione dei
        # _diag_* sopra (indagine anomalia no-DP AUC≈0.5, richiesta esplicita
        # dell'utente 2026-08-15 dopo aver visto member/non-member muoversi
        # insieme invece di separarsi in nodp-sweep3). Campi puramente
        # additivi, salvati SOLO se raccolti almeno un valore questo round —
        # non alterano né invalidano nessun campo pre-esistente. Il test
        # decisivo è "lira_debug_raw_loss_gap": se è chiaramente separato
        # (≠0, segno coerente round dopo round) MA lira_score_gap oscilla
        # vicino a 0 con segno instabile, il colpevole è l'amplificazione
        # /σ² con σ vicino al floor (guardare lira_debug_sigma_*_floor_hit_rate
        # in quel caso — atteso vicino a 1.0 se l'ipotesi è corretta). Se
        # invece anche lira_debug_raw_loss_gap è ≈0 e instabile, il modello
        # genuinamente non memorizza abbastanza per LiRA sotto no-DP — non
        # un bug, un risultato reale (Scenario B).
        _diag_fields: dict[str, Any] = {}
        if _diag_raw_loss_members and _diag_raw_loss_nonmembers:
            _diag_fields["lira_debug_raw_loss_member_mean"] = round(
                float(np.mean(_diag_raw_loss_members)), 8
            )
            _diag_fields["lira_debug_raw_loss_nonmember_mean"] = round(
                float(np.mean(_diag_raw_loss_nonmembers)), 8
            )
            _diag_fields["lira_debug_raw_loss_gap"] = round(
                float(np.mean(_diag_raw_loss_members) - np.mean(_diag_raw_loss_nonmembers)), 8
            )
            # Test mirato di conferma (2026-08-15, richiesto esplicitamente
            # dall'utente prima di implementare il fix del floor simmetrico
            # sopra): AUC calcolata DIRETTAMENTE sulla MSE grezza (target_loss),
            # bypassando interamente la normalizzazione log-likelihood-ratio
            # (μ_in/μ_out/σ_in/σ_out) — usa solo il segnale già confermato in
            # lira_debug_raw_loss_gap. Punteggio = -target_loss (loss più bassa
            # → più probabile membro → punteggio più alto, convenzione
            # roc_auc_score). Se questo recupera un AUC>0.5 stabile mentre
            # lira_auc_roc (sopra, via log-ratio) resta vicino a 0.5/instabile,
            # conferma che il segnale esiste ma la normalizzazione lo
            # distrugge — esattamente l'ipotesi diagnosticata. Puramente
            # informativo: NON sostituisce lira_auc_roc come metrica
            # ufficiale, serve solo a confermare/smentire prima di fidarsi
            # del fix del floor simmetrico applicato sopra.
            try:
                _raw_labels = [1] * len(_diag_raw_loss_members) + [0] * len(_diag_raw_loss_nonmembers)
                _raw_scores = [-x for x in _diag_raw_loss_members] + [-x for x in _diag_raw_loss_nonmembers]
                _diag_fields["lira_debug_raw_mse_auc_roc"] = round(
                    float(roc_auc_score(_raw_labels, _raw_scores)), 6
                )
            except ValueError:
                _diag_fields["lira_debug_raw_mse_auc_roc"] = None
            if raw_loss_dump_path is not None:
                _raw_loss_per_round[round_num] = {
                    "member_losses": list(_diag_raw_loss_members),
                    "nonmember_losses": list(_diag_raw_loss_nonmembers),
                }
        if _diag_sigma_in_values:
            _diag_fields["lira_debug_sigma_in_mean"] = round(float(np.mean(_diag_sigma_in_values)), 8)
            _diag_fields["lira_debug_sigma_in_floor_hit_rate"] = round(
                _diag_sigma_in_floor_hits / len(_diag_sigma_in_values), 4
            )
        if _diag_sigma_out_values:
            _diag_fields["lira_debug_sigma_out_mean"] = round(float(np.mean(_diag_sigma_out_values)), 8)
            _diag_fields["lira_debug_sigma_out_floor_hit_rate"] = round(
                _diag_sigma_out_floor_hits / len(_diag_sigma_out_values), 4
            )
        # DIAGNOSTICA 2026-08-20 — test decisivo per l'ipotesi "μ_in>μ_out
        # invertito" (vedi commento all'inizializzazione dei _diag_mu_*
        # sopra). Solo membri con calibrazione IN reale (non fallback) in
        # entrambe le liste, stesso identico insieme di campioni per μ_in e
        # μ_out (vedi dove sono popolati: entrambi dentro `if len(in_losses)
        # >= 2` / subito prima, stesso `is_member` branch) — confronto lecito.
        if _diag_mu_in_values and _diag_mu_out_values:
            _diag_fields["lira_debug_mu_in_mean"] = round(float(np.mean(_diag_mu_in_values)), 8)
            _diag_fields["lira_debug_mu_out_mean"] = round(float(np.mean(_diag_mu_out_values)), 8)
            _diag_fields["lira_debug_mu_in_minus_out"] = round(
                float(np.mean(_diag_mu_in_values) - np.mean(_diag_mu_out_values)), 8
            )
        # Fix 2026-08-20 — tasso di esclusione per "non calibrabilità" (vedi
        # _UNCALIBRATED_Z_THRESHOLD sopra). Trasparenza: se questo tasso
        # fosse alto (es. >20-30%), andrebbe rivisto n_shadow o σ, non solo
        # accettato — qui salvato sempre (non solo se _diag_fields è già
        # popolato) perché è parte integrante del fix, non solo diagnostica.
        if _diag_scored_total > 0:
            _diag_fields["lira_debug_uncalibrated_skip_rate"] = round(
                _diag_uncalibrated_skipped / _diag_scored_total, 4
            )
            # Sprint 10zz+111: quota di campioni il cui μ_in di fallback sarebbe
            # stato negativo (impossibile per una media di loss quadratiche) e
            # che il clamp ha riportato a un valore positivo.
            _diag_fields["lira_debug_mu_in_fallback_clamp_rate"] = round(
                _diag_mu_in_clamped / _diag_scored_total, 4
            )
            _diag_fields["lira_debug_uncalibrated_skipped_n"] = _diag_uncalibrated_skipped
        # DIAGNOSTICA 2026-08-21 (test mirato, vedi inizializzazione di
        # _diag_counterfactual_member_scores sopra): il fix dell'ancoraggio
        # μ_in ha portato lira_auc_roc da 0.32-0.44 (invertito) a 0.72-0.82,
        # molto sopra raw_mse_auc/Yeom (~0.50) — sospetto che l'AUC alto sia
        # un artefatto del fatto che i non-membri usano SEMPRE la formula
        # fallback mentre i membri usano quasi sempre quella con calibrazione
        # reale. Test decisivo: `lira_debug_matched_formula_auc` confronta i
        # membri FORZATI nella stessa formula fallback dei non-membri contro
        # i non-membri reali (anch'essi fallback) — stessa formula da
        # entrambi i lati. Se questo AUC resta vicino a 0.5, l'AUC alto visto
        # sopra è un artefatto di formule diverse (mismatch), NON segnale
        # reale. Se resta alto anche qui, il segnale è genuino (sopravvive
        # anche a formula identica), e l'anomalia rispetto a raw_mse_auc
        # andrebbe spiegata diversamente (LiRA più sensibile del solo
        # threshold sulla loss, come atteso da Carlini et al. 2022).
        if _diag_counterfactual_member_scores and round_nonmember_scores:
            _diag_fields["lira_debug_counterfactual_member_mean"] = round(
                float(np.mean(_diag_counterfactual_member_scores)), 6
            )
            _diag_fields["lira_debug_real_nonmember_mean"] = round(
                float(np.mean(round_nonmember_scores)), 6
            )
            try:
                _mf_labels = [1] * len(_diag_counterfactual_member_scores) + [0] * len(round_nonmember_scores)
                _mf_scores = list(_diag_counterfactual_member_scores) + list(round_nonmember_scores)
                _diag_fields["lira_debug_matched_formula_auc"] = round(
                    float(roc_auc_score(_mf_labels, _mf_scores)), 6
                )
            except ValueError:
                _diag_fields["lira_debug_matched_formula_auc"] = None
        if _diag_fields:
            # Log a INFO (non DEBUG) cosi' e' visibile subito con un
            # `tail -f` durante un test breve, senza dover aspettare la fine
            # dell'esperimento e aprire il JSON.
            logger.info(
                f"Round {round_num} — LiRA diagnostica: "
                f"raw_loss_gap={_diag_fields.get('lira_debug_raw_loss_gap', 'N/A')} "
                f"raw_mse_auc={_diag_fields.get('lira_debug_raw_mse_auc_roc', 'N/A')} "
                f"mu_in-mu_out={_diag_fields.get('lira_debug_mu_in_minus_out', 'N/A')} "
                f"σ_in_mean={_diag_fields.get('lira_debug_sigma_in_mean', 'N/A')} "
                f"(floor_hit={_diag_fields.get('lira_debug_sigma_in_floor_hit_rate', 'N/A')}) "
                f"σ_out_mean={_diag_fields.get('lira_debug_sigma_out_mean', 'N/A')} "
                f"(floor_hit={_diag_fields.get('lira_debug_sigma_out_floor_hit_rate', 'N/A')}) "
                f"uncalibrated_skip_rate={_diag_fields.get('lira_debug_uncalibrated_skip_rate', 'N/A')} "
                f"matched_formula_auc={_diag_fields.get('lira_debug_matched_formula_auc', 'N/A')} "
                f"(cf_member_mean={_diag_fields.get('lira_debug_counterfactual_member_mean', 'N/A')} "
                f"real_nonmember_mean={_diag_fields.get('lira_debug_real_nonmember_mean', 'N/A')})"
            )

        try:
            _rmia_fields = _rmia_round_fields(
                _rmia_rows, _rmia_cumulative, _rmia_cum_member, _rmia_cum_canary)
        except Exception as _exc:  # noqa: BLE001 - RMIA non deve far perdere LiRA
            logger.warning(f"RMIA round {round_num} fallito: {_exc}")
            _rmia_fields = {"rmia_auc_roc": None}
        logger.info(
            f"Round {round_num} — RMIA AUC: {_rmia_fields.get('rmia_auc_roc')} "
            f"(offline {_rmia_fields.get('rmia_offline_auc_roc')}, "
            f"n={_rmia_fields.get('rmia_n_samples')}, "
            f"TPR@1%FPR={_rmia_fields.get('rmia_tpr_at_fpr_0.01')}); "
            f"canary {_rmia_fields.get('canary_rmia_auc_roc')}"
        )
        lira_results[round_num] = {
            **_rmia_fields,
            "lira_auc_roc":               round(auc, 6),
            "lira_member_score_mean":     round(float(np.mean(round_member_scores)),    6),
            "lira_non_member_score_mean": round(float(np.mean(round_nonmember_scores)), 6),
            "lira_score_gap":             round(score_gap, 6),
            "n_shadow":                   n_shadow,
            # Canary positive control (Sprint 10vv) — None per ogni run senza
            # canary iniettati (cfg["canary"]["enabled"] non True).
            "canary_auc_roc":             canary_auc_roc,
            "canary_n_member":            len(round_canary_member_scores),
            "canary_n_nonmember":         len(round_canary_nonmember_scores),
            # Denominatore REALE dell'AUC canary (Sprint 10zz+113) — vedi il
            # commento su round_canary_member_groups. Usare QUESTI, non i due
            # conteggi di record sopra, per qualunque intervallo di confidenza
            # o test di significativita' sul canary.
            "canary_n_member_distinct":    len(round_canary_member_groups),
            "canary_n_nonmember_distinct": len(round_canary_nonmember_groups),
            # Denominatore del pool RAW (Sprint 10zz+119). Dal fix di questo
            # sprint puo' essere PIU' GRANDE dei due sopra: il pool raw non e'
            # filtrato da `insufficient_calibration`, quello LiRA si'. E' il
            # numero da citare accanto a canary_raw_mse_auc_roc, ed e' anche
            # quello confrontabile con la baseline a init casuale, che gira
            # senza shadow e quindi su tutti i template.
            "canary_raw_n_member_distinct":    len(round_canary_member_raw_groups),
            "canary_raw_n_nonmember_distinct": len(round_canary_nonmember_raw_groups),
            # Diagnostico raw-loss (Sprint 10ww) — immune a un'eventuale
            # contaminazione degli shadow dai duplicati canary, vedi sopra.
            "canary_raw_mse_auc_roc":     canary_raw_mse_auc_roc,
            # Sprint 10zz+28 (2026-09-03, task #53) — vedi commento sopra.
            "canary_raw_advantage":       canary_raw_advantage,
            "canary_raw_confusion":       canary_raw_confusion,
            # Sprint 10zz+170 (--canary-rescale): AUC e loss medie (membri,
            # non-membri) dei canary su ref + s*(osservato - ref), chiave s.
            "canary_rescaled_raw_auc_roc": canary_rescaled_raw_auc_roc,
            "canary_rescaled_mean_loss":   canary_rescaled_mean_loss,
            # MIA Advantage (task #41, Sprint 10zz+13) — vedi _mia_advantage().
            "lira_advantage":             lira_advantage,
            "canary_advantage":           canary_advantage,
            # Conteggi TP/FP/TN/FN alla soglia ottimale (task #49, Sprint
            # 10zz+25, 2026-09-03) — vedi _mia_confusion_at_best_threshold().
            "lira_confusion":             lira_confusion,
            "canary_confusion":           canary_confusion,
            # Sprint 10zz+33 (2026-09-03, task #58) — attacco di Sablayrolles
            # et al. 2019 [56], soglia non-parametrica per-esempio, calcolato
            # in parallelo a LiRA sugli stessi μ_in/μ_out/target_loss (vedi
            # docs/MetricsReference_DSN2027.md §3 per il confronto completo
            # con LiRA e la motivazione — Carlini et al. 2022 lo trovano
            # "sorprendentemente" competitivo nonostante sia più semplice).
            "sablayrolles_auc_roc":       sablayrolles_auc_roc,
            "sablayrolles_advantage":     sablayrolles_advantage,
            "sablayrolles_confusion":     sablayrolles_confusion,
            "sablayrolles_n_member":      len(round_sablayrolles_member_scores),
            "sablayrolles_n_nonmember":   len(round_sablayrolles_nonmember_scores),
            # Sprint 10zz+36 (2026-09-03, task #61) — variante esplorativa
            # LiRA su log(MSE), vedi commento sopra e docs/MetricsReference_
            # DSN2027.md §3 per il confronto completo col fit raw
            # (semplificazioni dichiarate — non ancora stesso rigore).
            "lira_log_auc_roc":          lira_log_auc_roc,
            "lira_log_advantage":        lira_log_advantage,
            "lira_log_confusion":        lira_log_confusion,
            "lira_log_n_member":         len(round_lira_log_member_scores),
            "lira_log_n_nonmember":      len(round_lira_log_nonmember_scores),
            **_tpr_fields,
            **_sablayrolles_tpr_fields,
            **_lira_log_tpr_fields,
            **_lira_log_diag_fields,
            **_diag_fields,
        }
        if capture_shadow_weights:
            lira_results[round_num]["_fedmia_shadow_weights"] = {
                cid: {
                    "vectors": shadow_weight_vectors_per_cluster.get(cid, []),
                    "is_member_majority": shadow_in_majority_member_per_cluster.get(cid, []),
                }
                for cid in _CLUSTER_IDS
            }

    # Sprint 10zz+29 (2026-09-03, task #54) — inizializzato qui (non dentro
    # "if _cumulative_scores:" sotto) cosi' resta definito anche se il pool
    # composto risulta vuoto, per la scrittura finale del dump più sotto.
    _composed_roc_curves: dict[str, Any] = {}
    if composed_output is not None:
        if _cumulative_scores:
            _labels = [1 if _sample_is_member[_sid] else 0 for _sid in _cumulative_scores]
            _scores = list(_cumulative_scores.values())
            try:
                _composed_auc = roc_auc_score(_labels, _scores)
            except ValueError:
                _composed_auc = 0.5
            _member_vals    = [s for _sid, s in _cumulative_scores.items() if _sample_is_member[_sid]]
            _nonmember_vals = [s for _sid, s in _cumulative_scores.items() if not _sample_is_member[_sid]]
            composed_output["composed_lira_auc_roc"] = round(float(_composed_auc), 6)
            # RMIA composto (2026-10-13): somma dei log LR per campione sui round
            try:
                _rl = [1 if _rmia_cum_member[_s] else 0 for _s in _rmia_cumulative]
                _rs = [_rmia_cumulative[_s] for _s in _rmia_cumulative]
                composed_output["composed_rmia_auc_roc"] = (
                    round(float(roc_auc_score(_rl, _rs)), 6) if len(set(_rl)) == 2 else None)
                for _k, _v in _tpr_at_fixed_fpr(_rl, _rs).items():
                    composed_output[f"composed_rmia_{_k}"] = _v
                _rc = [_s for _s in _rmia_cumulative if _rmia_cum_canary.get(_s) is not None]
                _rcl = [1 if _rmia_cum_member[_s] else 0 for _s in _rc]
                composed_output["canary_composed_rmia_auc_roc"] = (
                    round(float(roc_auc_score(_rcl, [_rmia_cumulative[_s] for _s in _rc])), 6)
                    if len(set(_rcl)) == 2 else None)
                logger.info(
                    f"RMIA composto (somma dei log LR sui round) — AUC-ROC: "
                    f"{composed_output['composed_rmia_auc_roc']}, canary "
                    f"{composed_output['canary_composed_rmia_auc_roc']}"
                )
            except Exception as _exc:  # noqa: BLE001
                logger.warning(f"RMIA composto fallito: {_exc}")
            composed_output["composed_lira_score_gap"] = round(
                float(np.mean(_member_vals) - np.mean(_nonmember_vals))
                if _member_vals and _nonmember_vals else 0.0,
                6,
            )
            composed_output["n_samples_scored"]   = len(_cumulative_scores)
            composed_output["n_rounds_aggregated"] = len(lira_results)
            # Sprint 10zz+90 (2026-09-15) — diagnostica di sola lettura, vedi
            # commento all'inizializzazione di _cumulative_score_counts sopra:
            # NON sostituisce composed_lira_auc_roc (la somma resta la
            # statistica ufficiale), calcola in aggiunta l'AUC sulla media
            # per-round-scorato di ogni campione (sum/count invece di sum),
            # che rimuove l'effetto "quante volte è stato scorato" dalla
            # magnitudo del punteggio. Se le due AUC divergono in modo
            # sostanziale, il numero di round scorati per campione correla
            # con la classe — un confondente da investigare prima di citare
            # composed_lira_auc_roc come statistica "pulita". Se sono vicine,
            # il confondente non è presente in pratica per questi dati.
            _mean_per_round_scores = {
                _sid: (_cumulative_scores[_sid] / _cumulative_score_counts[_sid])
                for _sid in _cumulative_scores
                if _cumulative_score_counts.get(_sid, 0) > 0
            }
            if _mean_per_round_scores:
                _mpr_labels = [1 if _sample_is_member[_sid] else 0 for _sid in _mean_per_round_scores]
                _mpr_scores = list(_mean_per_round_scores.values())
                try:
                    composed_output["composed_lira_auc_roc_mean_per_round"] = round(
                        float(roc_auc_score(_mpr_labels, _mpr_scores)), 6
                    )
                except ValueError:
                    composed_output["composed_lira_auc_roc_mean_per_round"] = None
                composed_output["composed_score_count_member_mean"] = round(
                    float(np.mean([
                        _cumulative_score_counts[_sid] for _sid in _cumulative_scores
                        if _sample_is_member[_sid]
                    ])), 4,
                ) if any(_sample_is_member[_sid] for _sid in _cumulative_scores) else None
                composed_output["composed_score_count_nonmember_mean"] = round(
                    float(np.mean([
                        _cumulative_score_counts[_sid] for _sid in _cumulative_scores
                        if not _sample_is_member[_sid]
                    ])), 4,
                ) if any(not _sample_is_member[_sid] for _sid in _cumulative_scores) else None
            # TPR@low-FPR (roadmap #4, Sprint 10pp 2026-08-28) sul composto —
            # è la metrica "headline" citata nei Sprint-log (vedi README), non
            # solo il per-round, quindi merita la stessa lettura a FPR fisso.
            #
            # Fix (Sprint 10zz+41, 2026-09-04, task #66 — bug scoperto durante
            # un audit del codice, non da un run fallito): _tpr_at_fixed_fpr()
            # restituisce SEMPRE le chiavi bare "tpr_at_fpr_0.001" ecc.,
            # indipendentemente da chi la chiama (stesso meccanismo del bug
            # Yeom/Shadow del task #59). Qui il risultato finiva in
            # composed_output con .update() SENZA prefisso, mentre ogni altro
            # campo composto in questa funzione usa "composed_lira_*"
            # (composed_lira_auc_roc/_advantage/_confusion sopra/sotto) — le
            # uniche chiavi bare. src/plugins/attacks/lira.py poi fa
            # `results[_final_round].update(_composed)`: quel merge sovrascriveva
            # silenziosamente il TPR@fixed-FPR del SOLO ultimo round (calcolato
            # correttamente qualche riga sopra in questa stessa funzione, sulle
            # sole evidenze di quel round) con il valore CUMULATIVO multi-round
            # — un dato diverso, non un duplicato innocuo. A differenza del
            # bug #59 (mai innescato, nessun JSON storico aveva quelle chiavi),
            # questo è live da quando "LiRA composto" esiste (Sprint 10pp,
            # 2026-08-28): ogni run con LiRA nel registro (sempre, è
            # nell'ATTACK_REGISTRY di default) ha l'ultimo round con
            # tpr_at_fpr_* che è in realtà il valore composto, non quello del
            # round. Fix: prefisso "composed_" per coerenza con gli altri
            # campi composti — elimina la collisione di chiavi alla radice.
            for _k, _v in _tpr_at_fixed_fpr(_labels, _scores).items():
                composed_output[f"composed_{_k}"] = _v
            # MIA Advantage (task #41, Sprint 10zz+13) sul composto — stessa
            # motivazione di TPR@low-FPR sopra, vedi _mia_advantage().
            composed_output["composed_lira_advantage"] = _mia_advantage(_labels, _scores)
            # Conteggi TP/FP/TN/FN sul composto (task #49, Sprint 10zz+25).
            composed_output["composed_lira_confusion"] = _mia_confusion_at_best_threshold(
                _labels, _scores
            )
            # Curva ROC completa sul composto (task #54, Sprint 10zz+29) —
            # il composto è la metrica "headline" (vedi commento TPR@low-FPR
            # sopra), quindi merita anche il plot log-log completo, non solo
            # AUC/TPR@fixed/Advantage. Struttura {"lira": {...}, "canary":
            # {...}|assente} — _composed_roc_curves inizializzato PRIMA di
            # questo "if composed_output is not None:" (resta definito anche
            # a pool composto vuoto), qui solo popolato.
            if roc_curve_dump_path is not None:
                _composed_curve = _full_roc_curve(_labels, _scores)
                if _composed_curve is not None:
                    _composed_roc_curves["lira"] = _composed_curve

            # Canary positive control (Sprint 10vv, 2026-08-31): stesso
            # accumulo composto sopra, ristretto ai soli campioni taggati
            # _canary_group (_sample_canary_group, costruita prima del loop
            # round). None se nessun canary è stato iniettato in questo run
            # (ogni run esistente/pubblicato) — zero impatto in quel caso.
            _canary_sids = [_sid for _sid in _cumulative_scores if _sid in _sample_canary_group]
            _canary_composed_member    = [_cumulative_scores[_sid] for _sid in _canary_sids if _sample_is_member[_sid]]
            _canary_composed_nonmember = [_cumulative_scores[_sid] for _sid in _canary_sids if not _sample_is_member[_sid]]
            if _canary_composed_member and _canary_composed_nonmember:
                _canary_composed_labels = [1] * len(_canary_composed_member) + [0] * len(_canary_composed_nonmember)
                _canary_composed_scores = _canary_composed_member + _canary_composed_nonmember
                try:
                    composed_output["canary_composed_auc_roc"] = round(
                        float(roc_auc_score(_canary_composed_labels, _canary_composed_scores)), 6
                    )
                except ValueError:
                    composed_output["canary_composed_auc_roc"] = None
                composed_output["canary_composed_advantage"] = _mia_advantage(
                    _canary_composed_labels, _canary_composed_scores
                )
                # Conteggi TP/FP/TN/FN sul canary composto (task #49,
                # Sprint 10zz+25) — n piccolo, il conteggio assoluto conta
                # più del tasso qui (vedi nota su canary_confusion sopra).
                composed_output["canary_composed_confusion"] = _mia_confusion_at_best_threshold(
                    _canary_composed_labels, _canary_composed_scores
                )
                composed_output["canary_composed_n_member"]    = len(_canary_composed_member)
                composed_output["canary_composed_n_nonmember"] = len(_canary_composed_nonmember)
                if roc_curve_dump_path is not None:
                    _canary_composed_curve = _full_roc_curve(
                        _canary_composed_labels, _canary_composed_scores
                    )
                    if _canary_composed_curve is not None:
                        _composed_roc_curves["canary"] = _canary_composed_curve
                logger.info(
                    f"LiRA composto — [CANARY] AUC: {composed_output['canary_composed_auc_roc']} "
                    f"(n_member={len(_canary_composed_member)}, "
                    f"n_nonmember={len(_canary_composed_nonmember)}) — positive control"
                )
            else:
                composed_output["canary_composed_auc_roc"] = None
            logger.info(
                f"LiRA composto (multi-round, evidenza sommata su "
                f"{len(lira_results)} round) — AUC-ROC: {_composed_auc:.4f} "
                f"(gap={composed_output['composed_lira_score_gap']:.6f}, "
                f"n_samples={len(_cumulative_scores)}, "
                # Fix 2026-09-11 (trovato preparando la ri-analisi statistica
                # TPR@low-FPR, non da un run fallito): questa chiave era rimasta
                # "tpr_at_fpr_0.01" (bare) da prima del fix Sprint 10zz+41 sopra,
                # che ha rinominato il campo effettivamente scritto in
                # composed_output a "composed_tpr_at_fpr_0.01" per eliminare la
                # collisione di chiavi. Da allora questo log stampava sempre
                # "None" (la chiave bare non viene più scritta) — nessun dato
                # nei JSON è mai stato affetto, solo la riga di log a schermo.
                f"TPR@1%FPR={composed_output.get('composed_tpr_at_fpr_0.01')})"
            )
        else:
            composed_output["composed_lira_auc_roc"] = None
            logger.warning("LiRA composto: nessuno score accumulato — pool vuoto")

        # Dump per-campione (task worst-case, Sprint 10zz+27, 2026-09-03) —
        # richiede composed_output (da cui viene _cumulative_scores, vedi
        # sopra) E per_sample_dump_path esplicito: opt-in doppio, zero
        # impatto su ogni chiamante che non passa entrambi. Usa session_id
        # REALE (ACNDataset, campo "session_id" — vedi src/adapters/
        # acn_dataset.py) invece di id(sample), che è un indirizzo di
        # memoria Python valido solo per la durata di QUESTO processo:
        # senza session_id stabile, non ci sarebbe modo di riconoscere lo
        # STESSO record reale in run/seed diversi per il controllo
        # worst-case cross-seed (scripts/analyze_worst_case_vulnerability.py).
        if per_sample_dump_path is not None and _cumulative_scores:
            _sid_to_session_id: dict[int, str | None] = {
                id(_s): _s.get("session_id") for _s in members_bal + nonmembers_bal
            }
            _dump_records = [
                {
                    "session_id": _sid_to_session_id.get(_sid),
                    "is_member": bool(_sample_is_member[_sid]),
                    "is_canary": _sid in _sample_canary_group,
                    "composed_score": _score,
                }
                for _sid, _score in _cumulative_scores.items()
            ]
            _n_missing_session_id = sum(1 for r in _dump_records if r["session_id"] is None)
            if _n_missing_session_id:
                logger.warning(
                    f"Dump per-campione: {_n_missing_session_id}/{len(_dump_records)} record "
                    f"senza session_id (campo assente/None nel dataset sorgente) — non "
                    f"riconoscibili cross-seed, esclusi automaticamente dall'analisi worst-case "
                    f"(scripts/analyze_worst_case_vulnerability.py li scarta)."
                )
            # Fix 2026-09-03 (task #60, Sprint 10zz+35) — stesso bug/fix di
            # _write_diagnostic_dump() sopra: la directory genitore non è
            # garantita esistente a questo punto della pipeline.
            Path(per_sample_dump_path).parent.mkdir(parents=True, exist_ok=True)
            with open(per_sample_dump_path, "w") as _dump_f:
                json.dump(
                    {
                        "seed": cfg.get("experiment", {}).get("seed"),
                        "epsilon": cfg.get("experiment", {}).get("epsilon"),
                        "no_dp": no_dp,
                        "dp_mode": dp_mode,
                        "n_rounds": len(lira_results),
                        "n_records": len(_dump_records),
                        "records": _dump_records,
                    },
                    _dump_f,
                )
            logger.info(
                f"Dump per-campione scritto: {per_sample_dump_path} "
                f"({len(_dump_records)} record, {_n_missing_session_id} senza session_id)"
            )

    # Scrittura dump curve ROC (task #54, Sprint 10zz+29, 2026-09-03) — FUORI
    # dal blocco "if composed_output is not None:" cosi' funziona anche
    # quando composed_output non è passato (i per-round esistono comunque);
    # _composed_roc_curves è {} in quel caso (mai popolato), quindi la
    # chiave "composed" viene semplicemente omessa dal file.
    if roc_curve_dump_path is not None and (_roc_curves_per_round or _composed_roc_curves):
        _write_diagnostic_dump(roc_curve_dump_path, {
            "attack": "lira",
            "seed": cfg.get("experiment", {}).get("seed"),
            "epsilon": cfg.get("experiment", {}).get("epsilon"),
            "no_dp": no_dp,
            "dp_mode": dp_mode,
            "per_round": _roc_curves_per_round,
            "composed": _composed_roc_curves or None,
        })

    # Scrittura dump raw-loss (task #57, Sprint 10zz+32, 2026-09-03) — FUORI
    # dal blocco "if composed_output is not None:" (i _diag_raw_loss_* sono
    # raccolti ogni round indipendentemente da composed_output).
    if raw_loss_dump_path is not None and _raw_loss_per_round:
        _write_diagnostic_dump(raw_loss_dump_path, {
            "attack": "lira_raw_loss",
            "seed": cfg.get("experiment", {}).get("seed"),
            "epsilon": cfg.get("experiment", {}).get("epsilon"),
            "no_dp": no_dp,
            "dp_mode": dp_mode,
            "per_round": _raw_loss_per_round,
        })

    return lira_results


def run_fedmia_gradient(
    cfg: dict,
    train_sessions: list[dict[str, Any]],
    holdout_sessions: list[dict[str, Any]],
    fl_results: dict[int, dict[str, Any]],
    n_shadow: int = 8,
    shadow_epochs_cap: int | None = None,
    no_dp: bool = False,
    dp_mode: str = "dp-fedavg",
    cluster_membership: dict[str, list[int]] | None = None,
    normalize_vectors: bool = False,
    composed_output: dict[str, Any] | None = None,
    **_ignored: Any,
) -> dict[int, dict[str, Any]]:
    """
    FedMIA-gradient — Sprint 10zz (2026-09-01), PRIMO DRAFT, NON validato con
    un run reale (torch non eseguibile nell'ambiente di sviluppo — questa
    funzione è stata scritta e py_compile-verificata, ma mai eseguita).
    Aspettati iterazione reale, come per run_lira() (6 round di fix trovati
    SOLO eseguendo il codice — vedi il suo docstring). Deciso esplicitamente
    dall'utente (2026-09-01) dopo un'analisi che ha rifiutato due alternative
    più semplici (fix di solo troncamento sulla classe FedMIA originale;
    non implementarlo affatto) — vedi README Sprint 10zz.

    Cosa NON è: NON è l'attacco per-nodo di
    src/plugins/attacks/fedmia.py::FedMIA come originariamente concepito
    ("membro"/"non-membro" a livello di nodo — etichetta priva di senso in
    questo framework: ogni client FL reale ha sempre davvero partecipato al
    training). NON è un sostituto di LiRA a livello di sessione — resta lo
    strumento primario del paper.

    Cosa È: una domanda distinta, a granularità round+cluster — "il vettore
    di peso FINALE di uno shadow rivela se il suo subset IN era a
    maggioranza sessioni membro o non-membro?" — usando la classe FedMIA
    (autoencoder) con FedMIA.calibrate_from_vectors() (Sprint 10zz), invece
    della calibrazione a rumore gaussiano dell'implementazione originale
    (mai stata eseguita da nessun esperimento di questo progetto).

    Riusa INTERAMENTE l'ensemble shadow di run_lira() (universo membri+
    non-membri, warm-start, privatizzazione DP — 6 fix reali, vedi
    run_lira.__doc__) via capture_shadow_weights=True: chiama run_lira() una
    SECONDA volta (stesso costo computazionale di un run LiRA — raddoppia il
    tempo se entrambi vengono eseguiti sullo stesso esperimento), zero
    duplicazione di quella logica, zero rischio per run_lira()/LiRA
    (capture_shadow_weights resta False in ogni altro chiamante — vedi
    run_registered_attacks() — comportamento LiRA esistente invariato).

    Metodo per (round, cluster):
      1. Split deterministico dei vettori di peso shadow validi: indici pari
         → calibrazione (train), indici dispari → valutazione (test). Non
         leave-one-out — più semplice, sufficiente per un primo draft con
         n_shadow piccolo (default 8).
      2. FedMIA(input_dim=len(vettore)) — lunghezza REALE del vettore di
         peso appiattito, NESSUN troncamento (fix esplicitamente richiesto
         dall'utente il 2026-09-01, invece del troncamento a 6 valori
         dell'implementazione originale).
      3. calibrate_from_vectors(member_vectors=shadow di train a maggioranza
         membri, non_member_vectors=shadow di train a maggioranza
         non-membri) — calibrazione su dati REALI, non rumore gaussiano.
      4. AUC-ROC sugli shadow di test: -MSE di ricostruzione (calibrato sui
         membri → errore basso atteso per un vettore "member-like"; AUC-ROC
         è invariante a trasformazioni monotone dello score, quindi l'MSE
         grezzo con segno invertito basta, non serve il punteggio 0-1
         normalizzato) vs is_member_majority.

    LIMITI NOTI (dichiarati prima di qualunque run reale, non dopo):
      - n_shadow=8 (default "fast demo") → ~4 shadow di test per
        cluster/round dopo lo split — AUC-ROC su ~4 punti per cluster è
        estremamente rumorosa. Serve n_shadow≥16-32 ("paper quality", stessa
        soglia raccomandata da run_lira()) prima di interpretare il numero
        come risultato, non solo come smoke test.
      - Fix (review indipendente, 2026-09-01): con n_shadow piccolo,
        train_member (metà dispari degli shadow validi di un cluster/round,
        filtrati per maggioranza membri) ha spesso lunghezza <2 — sotto
        quella soglia il cluster/round viene SALTATO (non forzato con
        troppo pochi dati, vedi il check esplicito sotto), quindi con
        n_shadow=8 aspettati "fedmia_gradient_auc_roc": null per MOLTI
        round/cluster, non solo un numero rumoroso — un motivo in più per
        usare n_shadow≥16-32 in qualunque run i cui risultati contano.
      - Il segnale ipotizzato (la composizione membri/non-membri del subset
        IN di uno shadow altera il suo vettore di peso finale in modo
        distinguibile da un piccolo autoencoder) NON è garantito essere più
        forte del rumore stocastico tra shadow diversi (init casuale, ordine
        batch, warm-start condiviso che fa convergere gli shadow verso pesi
        simili indipendentemente dal subset IN — stesso fenomeno di
        collasso già documentato per LiRA, fix 2026-07-21e). Un risultato
        nullo (AUC≈0.5) qui è un esito scientificamente valido, non
        necessariamente un bug — stesso principio già applicato a
        hour_of_day circular encoding (Sprint 10eee, non promosso per lo
        stesso tipo di onestà nel criterio di decisione).

    NOTA METODOLOGICA (Sprint 10zz+8, 2026-09-01, trovata da una review statica
    dopo il primo run reale — vedi README): il primo smoke test post-
    controlled_composition (n_shadow=16) ha prodotto un AUC-ROC POOLED
    identico (0.222222) in entrambi i round, sospetto perché i vettori di
    peso shadow sono genuinamente diversi round-su-round (seed include
    round_num, verificato). Causa più probabile trovata per ispezione: ogni
    cluster viene calibrato con la SUA PROPRIA istanza FedMIA (scale di MSE
    diverse — office1 ha ~1300 sessioni contro le ~25-27mila di caltech/jpl,
    quindi vettori di peso e relativi errori di ricostruzione su scale
    strutturalmente diverse per ragioni indipendenti dalla membership), ma i
    punteggi dei 3 cluster venivano poi RIUNITI in un'unica lista prima di
    calcolare un solo AUC-ROC pooled — mescolando classificatori non
    comparabili. Poiché le dimensioni relative dei 3 siti sono stabili da un
    round all'altro, questo può produrre un AUC pooled molto simile o
    identico indipendentemente dal vero segnale round-su-round. Fix: l'AUC
    per-cluster è ora la metrica PRIMARIA (results[round]["...auc_roc_per_cluster"]),
    calcolato separatamente per ogni cluster con i propri punteggi, prima di
    qualunque mescolamento. Il valore pooled resta nel risultato per
    compatibilità/diagnostica ma è esplicitamente marcato come secondario —
    non usarlo da solo per interpretare il segnale.

    Args (aggiuntivo):
        normalize_vectors: (Sprint 10zz+17, 2026-09-02) default False → ZERO
            impatto sul comportamento esistente. Test diagnostico mirato,
            deciso dopo Sprint 10zz+14 (il round 2 di office1 mostra norme
            membro/non-membro quasi identiche — 1.01× — eppure AUC resta
            0.0, la sola scala non basta a spiegare la separazione
            perfetta). Se True, ogni vettore di peso (train_member,
            test_member, train_non_member, test_non_member) viene
            normalizzato alla propria norma L2 unitaria PRIMA della
            diagnosi L2 (che a quel punto riporterà ~1.0 per costruzione,
            atteso) e prima di calibrate_from_vectors()/
            reconstruction_error(). Se l'AUC scende verso 0.5 con questo
            attivo, la scala ERA la causa dominante (→ un fix di
            normalizzazione permanente sarebbe sufficiente). Se l'AUC resta
            vicino a 0.0 anche con vettori unit-norm, conferma il
            confondimento strutturale ipotizzato in Sprint 10zz+14 (shadow
            "a maggioranza membri" vs "a maggioranza non-membri" allenati
            su dati REALMENTE diversi, quindi con pesi strutturalmente
            diversi indipendentemente dalla scala) — in quel caso un fix
            di normalizzazione non risolverebbe nulla, e il disegno
            round+cluster andrebbe abbandonato, non solo corretto.
        composed_output: (Sprint 10zz+21, 2026-09-02) default None → ZERO
            impatto sul comportamento esistente. Deciso dopo Sprint 10zz+20:
            con n_shadow=16 il test set per cluster/round è ~8 punti (~4 per
            classe) — troppo poco per distinguere un segnale reale dal
            rumore campionario (AUC osservate 0.0-0.93 nello stesso run,
            nessuna convergenza pulita). A differenza di LiRA (dove il
            "composto" somma l'evidenza dello STESSO campione su più round),
            qui ogni round produce shadow/punti DIVERSI — la composizione
            corretta è quindi un pooling: accumula le coppie (label, score)
            di OGNI round in un pool per-cluster, poi calcola UN SOLO
            AUC-ROC finale per cluster sul pool intero — stessa idea di
            "più osservazioni indipendenti riducono il rumore", applicata al
            livello giusto per questo disegno, non una copia meccanica del
            pattern LiRA. Se fornito un dict (stesso pattern by-reference di
            composed_output in run_lira(), vedi LiRAAttack.run() in
            src/plugins/attacks/lira.py), viene popolato con
            "composed_auc_roc_per_cluster"/"composed_n_test_per_cluster"
            dopo l'ultimo round.

    Returns:
        {round_num: {
            "fedmia_gradient_auc_roc":  float | None,  # POOLED sui 3 cluster —
                                                          diagnostico, vedi
                                                          "Nota metodologica"
                                                          sopra: NON la metrica
                                                          primaria.
            "fedmia_gradient_n_test":   int,   # shadow di test totali usati (pooled)
            "fedmia_gradient_n_shadow": int,   # n_shadow richiesto (config)
            "fedmia_gradient_auc_roc_per_cluster": dict[str, float | None],
                                                # METRICA PRIMARIA — AUC calcolato
                                                # separatamente per cluster, mai
                                                # mescolato con score di altri
                                                # cluster su scale diverse.
            "fedmia_gradient_n_test_per_cluster":  dict[str, int],
            "fedmia_gradient_advantage_per_cluster":  dict[str, float | None],   # Sprint 10zz+70
            "fedmia_gradient_confusion_per_cluster":  dict[str, dict],          # Sprint 10zz+70
            "fedmia_gradient_tpr_at_fpr_per_cluster": dict[str, dict],          # Sprint 10zz+70
        }}
        composed_output (se fornito) riceve, dopo l'ultimo round:
            "fedmia_gradient_composed_auc_roc_per_cluster": dict[str, float | None],
                                                # METRICA PRIMARIA per campagne
                                                # multi-round (task #47/Sprint
                                                # 10zz+21) — un solo AUC-ROC per
                                                # cluster sul pool di TUTTI i
                                                # round, riduce il rumore
                                                # campionario di n_test~8/round.
            "fedmia_gradient_composed_n_test_per_cluster": dict[str, int],
            "fedmia_gradient_composed_advantage_per_cluster":  dict[str, float | None],  # Sprint 10zz+70
            "fedmia_gradient_composed_confusion_per_cluster":  dict[str, dict],          # Sprint 10zz+70
            "fedmia_gradient_composed_tpr_at_fpr_per_cluster": dict[str, dict],          # Sprint 10zz+70
    """
    from sklearn.metrics import roc_auc_score

    from plugins.attacks.fedmia import FedMIA

    lira_side_channel = run_lira(
        cfg, train_sessions, holdout_sessions, fl_results,
        n_shadow=n_shadow, shadow_epochs_cap=shadow_epochs_cap,
        no_dp=no_dp, dp_mode=dp_mode, cluster_membership=cluster_membership,
        capture_shadow_weights=True,
        # Fix (2026-09-01, trovato da un run reale — n_test=0 in ogni round
        # con lo split stratificato "onesto" appena aggiunto): senza questo,
        # gli shadow "a maggioranza non-membro" sono essenzialmente
        # impossibili da ottenere per varianza campionaria su cluster grandi
        # (universo storico ~80:20 membri:non-membri) — vedi il docstring di
        # controlled_composition in run_lira() per l'analisi completa. Attivo
        # SOLO in questa chiamata interna, mai per la vera LiRA registrata.
        controlled_composition=True,
    )

    results: dict[int, dict[str, Any]] = {}
    # Sprint 10zz+21: pool cross-round per cluster, riempito round dopo
    # round SOLO se composed_output è stato passato (default None → zero
    # costo extra, nessuna lista accumulata a vuoto). Vedi Args sopra.
    _pooled_cluster_labels: dict[str, list[int]] = {}
    _pooled_cluster_scores: dict[str, list[float]] = {}

    for round_num, round_data in lira_side_channel.items():
        _payload = round_data.get("_fedmia_shadow_weights")
        if not _payload:
            continue

        all_labels: list[int] = []
        all_scores: list[float] = []
        # Sprint 10zz+8: AUC/n_test per cluster — metrica primaria, vedi
        # "Nota metodologica" nel docstring sopra.
        per_cluster_auc: dict[str, float | None] = {}
        per_cluster_n_test: dict[str, int] = {}
        # Sprint 10zz+70: stesse metriche già calcolate per Yeom/Shadow/LiRA
        # (vedi _mia_advantage()/_mia_confusion_at_best_threshold()/
        # _tpr_at_fixed_fpr()), qui per cluster invece che sul pool intero —
        # coerente con "l'AUC per-cluster è la metrica primaria" già stabilito
        # per questo attacco (Sprint 10zz+8, vedi "Nota metodologica" sopra).
        per_cluster_advantage: dict[str, float | None] = {}
        per_cluster_confusion: dict[str, dict[str, float | int | None]] = {}
        per_cluster_tpr_at_fpr: dict[str, dict[str, float | None]] = {}

        for cid, cluster_data in _payload.items():
            vectors = cluster_data.get("vectors", [])
            is_member_majority = cluster_data.get("is_member_majority", [])
            # Scarta shadow skippati (vettore None — es. cluster con troppo
            # poche sessioni per riempire batch_size, vedi run_lira()).
            paired = [
                (v, m) for v, m in zip(vectors, is_member_majority) if v is not None
            ]
            if len(paired) < 4:
                # Troppo pochi shadow validi per uno split train/test onesto
                # (train_member/train_non_member rischierebbero di restare
                # vuoti) — cluster/round saltato, non forzato con dati
                # insufficienti.
                continue

            # Fix (2026-09-01, trovato da un run reale — non solo
            # ipotizzato): split stratificato PER CLASSE, non sull'elenco
            # combinato. Il primo smoke test con n_shadow=16 ha prodotto
            # n_test=24 (tutti e 3 i cluster ammessi) ma auc=None in
            # ENTRAMBI i round — con l'universo membri:non-membri sbilanciato
            # ~4:1 (split 80/20), lo split 0::2/1::2 sull'elenco intero
            # lascia spesso il test set con UNA SOLA classe (i pochi shadow
            # "a maggioranza non-membro" cadono per caso tutti in posizione
            # pari o dispari) — non un problema di n_shadow insufficiente,
            # ma di uno split non stratificato. Dividere membri e non-membri
            # SEPARATAMENTE in metà garantisce entrambe le classi nel test
            # set ogni volta che sono presenti almeno 2 shadow per classe.
            members     = [v for v, m in paired if m]
            non_members = [v for v, m in paired if not m]
            train_member     = members[0::2]
            test_member      = members[1::2]
            train_non_member = non_members[0::2]
            test_non_member  = non_members[1::2]
            # FedMIA.calibrate_from_vectors() richiede ALMENO 2
            # member_vectors (nn.BatchNorm1d con batch_size=1 solleva un
            # errore torch — vedi il suo guard esplicito). Il test set deve
            # contenere ENTRAMBE le classi, altrimenti l'AUC non è
            # definibile per questo cluster/round — skip esplicito in
            # entrambi i casi, mai un crash o un "quasi tutto una classe".
            if len(train_member) < 2 or not test_member or not test_non_member:
                continue

            # Test mirato scala-vs-confondimento-strutturale (Sprint 10zz+17,
            # 2026-09-02) — vedi Args nel docstring per il razionale completo.
            # Normalizza OGNI vettore alla propria norma L2 unitaria PRIMA di
            # ogni uso a valle (diagnosi L2 inclusa, che dopo questo passo
            # riporterà ~1.0 per costruzione — atteso, non un bug). Applicato
            # qui (non prima) cosi' non altera in alcun modo lo split
            # train/test stratificato appena fatto sopra.
            if normalize_vectors:
                def _unit_norm(vec: list[float]) -> list[float]:
                    n = sum(x * x for x in vec) ** 0.5
                    return [x / n for x in vec] if n > 0 else vec

                train_member     = [_unit_norm(v) for v in train_member]
                test_member      = [_unit_norm(v) for v in test_member]
                train_non_member = [_unit_norm(v) for v in train_non_member]
                test_non_member  = [_unit_norm(v) for v in test_non_member]

            # Diagnosi economica (Sprint 10zz+10, 2026-09-02, su richiesta
            # esplicita — "fai prima una diagnosi più economica" prima di
            # investire in un redesign): norma L2 media dei vettori di peso
            # GREZZI, calcolata PRIMA di toccare FedMIA/l'autoencoder interno
            # — se membri e non-membri differiscono già in norma a questo
            # punto, conferma che il segnale (o l'AUC=0.0 spurio) è guidato
            # dalla scala del vettore, non da qualcosa che l'autoencoder
            # "impara" — indipendentemente da qualunque problema di
            # capacità/training di FedMIA stesso. Nessun costo aggiuntivo:
            # riusa i vettori già in memoria, nessuna chiamata torch in più.
            def _l2_norm(vec: list[float]) -> float:
                return sum(x * x for x in vec) ** 0.5

            _norm_train_member     = sum(_l2_norm(v) for v in train_member) / len(train_member)
            _norm_test_member      = sum(_l2_norm(v) for v in test_member) / len(test_member)
            _norm_train_non_member = (
                sum(_l2_norm(v) for v in train_non_member) / len(train_non_member)
                if train_non_member else None
            )
            _norm_test_non_member  = sum(_l2_norm(v) for v in test_non_member) / len(test_non_member)
            logger.info(
                f"Round {round_num} — FedMIA-gradient [{cid}] DIAGNOSI norma L2 "
                f"(prima della calibrazione): train_member={_norm_train_member:.4f} "
                f"test_member={_norm_test_member:.4f} "
                f"train_non_member={_norm_train_non_member if _norm_train_non_member is None else round(_norm_train_non_member, 4)} "
                f"test_non_member={_norm_test_non_member:.4f}"
            )

            test_pairs = [(v, True) for v in test_member] + [(v, False) for v in test_non_member]
            input_dim = len(members[0]) if members else len(non_members[0])
            fedmia = FedMIA(input_dim=input_dim)
            fedmia.calibrate_from_vectors(train_member, train_non_member)

            # Sprint 10zz+8: punteggi tenuti ANCHE separati per cluster, prima
            # di finire nel pool combinato — vedi "Nota metodologica" sopra.
            _cluster_labels: list[int] = []
            _cluster_scores: list[float] = []
            for v, is_member in test_pairs:
                mse = fedmia.reconstruction_error(v)
                label = 1 if is_member else 0
                score = -mse
                _cluster_labels.append(label)
                _cluster_scores.append(score)
                all_labels.append(label)
                all_scores.append(score)

            _cluster_auc = None
            if len(set(_cluster_labels)) == 2:
                try:
                    _cluster_auc = round(float(roc_auc_score(_cluster_labels, _cluster_scores)), 6)
                except ValueError:
                    _cluster_auc = None
            per_cluster_auc[cid] = _cluster_auc
            per_cluster_n_test[cid] = len(_cluster_labels)

            # Fix (2026-09-14, Sprint 10zz+70 — gap strutturale segnalato in
            # README Sprint 10zz+34: "run_fedmia_gradient() produce SOLO
            # fedmia_gradient_auc_roc [...], MAI Advantage/Confusion/
            # TPR@low-FPR (mai esteso, a differenza di Yeom/Shadow/canary
            # nel task #53)"). Necessario per confrontare FedMIA-gradient a
            # parità di metriche con Yeom/Shadow/LiRA quando promosso da
            # diagnostico opt-in ad attacco citabile nel paper (vedi README
            # Sprint 10zz+69/+70) — stesse funzioni pure (labels, scores) →
            # dict già usate ovunque nel file, nessuna nuova formula.
            per_cluster_advantage[cid] = _mia_advantage(_cluster_labels, _cluster_scores)
            per_cluster_confusion[cid] = _mia_confusion_at_best_threshold(
                _cluster_labels, _cluster_scores
            )
            per_cluster_tpr_at_fpr[cid] = _tpr_at_fixed_fpr(_cluster_labels, _cluster_scores)

            if composed_output is not None:
                _pooled_cluster_labels.setdefault(cid, []).extend(_cluster_labels)
                _pooled_cluster_scores.setdefault(cid, []).extend(_cluster_scores)

            _n_member_test = sum(_cluster_labels)
            _n_non_member_test = len(_cluster_labels) - _n_member_test
            _mean_member = round(
                sum(s for s, l in zip(_cluster_scores, _cluster_labels) if l == 1) / _n_member_test, 6
            ) if _n_member_test else None
            _mean_non_member = round(
                sum(s for s, l in zip(_cluster_scores, _cluster_labels) if l == 0) / _n_non_member_test, 6
            ) if _n_non_member_test else None
            logger.info(
                f"Round {round_num} — FedMIA-gradient [{cid}] AUC-ROC: {_cluster_auc} "
                f"(n_test={len(_cluster_labels)}, score_medio_membri={_mean_member}, "
                f"score_medio_non_membri={_mean_non_member}, "
                f"normalize_vectors={normalize_vectors})"
            )

        auc = None
        if len(set(all_labels)) == 2:
            try:
                auc = round(float(roc_auc_score(all_labels, all_scores)), 6)
            except ValueError:
                auc = None

        results[round_num] = {
            "fedmia_gradient_auc_roc":  auc,   # POOLED — diagnostico, vedi "Nota metodologica" sopra
            "fedmia_gradient_n_test":   len(all_labels),
            "fedmia_gradient_n_shadow": n_shadow,
            "fedmia_gradient_auc_roc_per_cluster": per_cluster_auc,   # METRICA PRIMARIA
            "fedmia_gradient_n_test_per_cluster":  per_cluster_n_test,
            # Sprint 10zz+70: stesse tre metriche di Yeom/Shadow/LiRA, per
            # cluster (coerente con la metrica primaria di questo attacco).
            "fedmia_gradient_advantage_per_cluster":     per_cluster_advantage,
            "fedmia_gradient_confusion_per_cluster":     per_cluster_confusion,
            "fedmia_gradient_tpr_at_fpr_per_cluster":    per_cluster_tpr_at_fpr,
        }
        # Fix (2026-09-01, trovato dal primo smoke test reale — mancava,
        # a differenza di OGNI altro attacco in questo file, che logga
        # sempre il proprio AUC per round): senza questa riga il risultato
        # è visibile solo aprendo il JSON, zero segnale in console/log.
        logger.info(
            f"Round {round_num} — FedMIA-gradient AUC-ROC pooled (diagnostico, NON primario): {auc} "
            f"(n_test={len(all_labels)}, n_shadow={n_shadow}) — per-cluster: {per_cluster_auc}"
        )

    if composed_output is not None:
        # Sprint 10zz+21: un solo AUC-ROC per cluster sul pool di TUTTI i
        # round accumulati sopra — vedi Args nel docstring per il razionale
        # (pooling, non somma di log-likelihood come il "composto" di LiRA).
        _composed_auc_per_cluster: dict[str, float | None] = {}
        _composed_n_test_per_cluster: dict[str, int] = {}
        # Sprint 10zz+70: stesse tre metriche aggiunte sopra per il per-round,
        # qui sul pool composto — coerente col fatto che la Tabella 2 del
        # paper riporta Advantage/Confusion/TPR@low-FPR sul COMPOSTO
        # multi-round per Yeom/Shadow/LiRA, non solo per round singolo.
        _composed_advantage_per_cluster: dict[str, float | None] = {}
        _composed_confusion_per_cluster: dict[str, dict[str, float | int | None]] = {}
        _composed_tpr_at_fpr_per_cluster: dict[str, dict[str, float | None]] = {}
        for cid, labels in _pooled_cluster_labels.items():
            scores = _pooled_cluster_scores[cid]
            _c_auc = None
            if len(set(labels)) == 2:
                try:
                    _c_auc = round(float(roc_auc_score(labels, scores)), 6)
                except ValueError:
                    _c_auc = None
            _composed_auc_per_cluster[cid] = _c_auc
            _composed_n_test_per_cluster[cid] = len(labels)
            _composed_advantage_per_cluster[cid] = _mia_advantage(labels, scores)
            _composed_confusion_per_cluster[cid] = _mia_confusion_at_best_threshold(labels, scores)
            _composed_tpr_at_fpr_per_cluster[cid] = _tpr_at_fixed_fpr(labels, scores)
        composed_output["fedmia_gradient_composed_auc_roc_per_cluster"] = _composed_auc_per_cluster
        composed_output["fedmia_gradient_composed_n_test_per_cluster"] = _composed_n_test_per_cluster
        composed_output["fedmia_gradient_composed_advantage_per_cluster"] = _composed_advantage_per_cluster
        composed_output["fedmia_gradient_composed_confusion_per_cluster"] = _composed_confusion_per_cluster
        composed_output["fedmia_gradient_composed_tpr_at_fpr_per_cluster"] = _composed_tpr_at_fpr_per_cluster
        logger.info(
            f"FedMIA-gradient composto (pool di {len(results)} round) — "
            f"AUC-ROC per cluster: {_composed_auc_per_cluster} "
            f"(n_test: {_composed_n_test_per_cluster})"
        )

    return results


def run_centralized_control(
    cfg: dict,
    train_sessions: list[dict[str, Any]],
    holdout_sessions: list[dict[str, Any]],
    rounds: int,
    local_epochs: int,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Esperimento di controllo — Sprint 10zz+1 (2026-09-01), su richiesta
    esplicita dell'utente in risposta a un commento in stile revisore:
    "l'assenza di leakage è dovuta a capacità limitata del modello o
    all'architettura FL che agisce essa stessa da regolarizzatore?".

    I 5 esperimenti di escalation già fatti (epoche 50→1000, capacità
    570→1870 parametri, feature quasi-uniche, combinati, sito diverso — vedi
    docs/TestRoadmap_DSN2027.md Sprint 10jj-10mm) rispondono SOLO alla prima
    metà della domanda ("non è solo capacità/esposizione limitata" — tutti e
    5 restano AUC 0.48-0.54). Nessuno di quei 5 isola la variabile
    FEDERAZIONE: sono tutti run FL standard (partizionati per cluster,
    FedAvg ogni round), solo con capacità/epoche/feature diverse. Questo
    esperimento isola quella variabile: STESSA architettura, STESSI dati di
    training (pooled, non partizionati per cluster), STESSO budget totale di
    epoche (rounds × local_epochs — identico a quanto un client FL vedrebbe
    cumulativamente), ma addestrati con UN SOLO modello centralizzato — nessun
    FedAvg, nessuna media periodica verso un punto di partenza condiviso.

    Se anche qui l'AUC resta ≈0.5: l'assenza di leakage non dipende dalla
    federazione (il modello centralizzato, esposto agli stessi dati/epoche,
    non memorizza comunque) — rafforza l'ipotesi "il modello/i dati non
    danno margine di memorizzazione", indipendentemente dalla FL.
    Se invece l'AUC centralizzato sale sensibilmente sopra 0.5: la FL
    (partizionamento per cluster + media periodica verso pesi condivisi) è
    essa stessa un regolarizzatore che riduce la memorizzazione rispetto al
    centralizzato — un risultato positivo e citabile, non solo un controllo
    negativo.

    Valutazione: stessa metrica loss-based di Yeom/run_yeom() (MSE di
    ricostruzione per-campione, score=-MSE, membri=train_sessions vs
    non-membri=holdout_sessions, stesso _mia_feature_names(cfg)) — comparabile
    DIRETTAMENTE con mean_auc_roc del run federato sullo stesso esperimento
    (stesso train/holdout split, stessa normalizzazione, già calcolati da
    main() prima di chiamare questa funzione).

    NON tocca run_fl_rounds()/FedAvgAggregator/GradientManager/NVFLARE/
    run_lira() — usa solo AutoencoderTrainer direttamente, in un percorso
    completamente separato. Zero rischio per la pipeline federata già
    validata.

    Returns:
        {
            "centralized_control_auc_roc":       float | None,
            "centralized_control_n_member":      int,
            "centralized_control_n_non_member":  int,
            "centralized_control_total_epochs":  int,
        }
    """
    from sklearn.metrics import roc_auc_score

    total_epochs = rounds * local_epochs
    _ml_cfg = {**cfg["ml"], "seed": seed, "epochs": total_epochs}
    if _subject_dp_config(_ml_cfg)["enabled"]:
        logger.warning(
            "[CENTRALIZED CONTROL] ml.subject_dp attivo: il controllo centralizzato "
            "gira con la DP per soggetto ma senza accounting proprio."
        )
    trainer = AutoencoderTrainer(
        config=_ml_cfg, node_id="centralized-control", cluster_id="centralized-control",
    )
    logger.info(
        f"[CENTRALIZED CONTROL] training centralizzato su {len(train_sessions)} "
        f"sessioni pooled (nessun partizionamento per cluster, nessun FedAvg), "
        f"{total_epochs} epoche totali ({rounds}×{local_epochs} — stesso budget "
        f"cumulativo del run federato)"
    )
    trainer.train_local(train_sessions, round_num=1)
    model = trainer.model
    model.eval()

    def _score(sessions: list[dict[str, Any]]) -> list[float]:
        rows = []
        for s in sessions:
            try:
                rows.append([float(s[f]) for f in _mia_feature_names(cfg)])
            except (KeyError, TypeError, ValueError):
                continue
        if not rows:
            return []
        tensor = torch.tensor(rows, dtype=torch.float32)
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(tensor), 256):
                batch  = tensor[i : i + 256]
                recon  = model(batch)
                errors = torch.mean((recon - batch) ** 2, dim=1)
                # Score = -errore: basso errore → membro → score alto (stessa
                # convenzione di run_yeom()/_score_batch()).
                scores.extend(-e.item() for e in errors)
        return scores

    member_scores     = _score(train_sessions)
    non_member_scores = _score(holdout_sessions)

    result: dict[str, Any] = {
        "centralized_control_auc_roc":      None,
        "centralized_control_n_member":     len(member_scores),
        "centralized_control_n_non_member": len(non_member_scores),
        "centralized_control_total_epochs": total_epochs,
    }
    if member_scores and non_member_scores:
        labels_arr = np.array([1] * len(member_scores) + [0] * len(non_member_scores))
        scores_arr = np.array(member_scores + non_member_scores)
        valid_mask = ~np.isnan(scores_arr) & ~np.isinf(scores_arr)
        # Fix (review indipendente, 2026-09-01): il controllo originale era
        # solo sul TOTALE combinato (>=10) — su un dataset piccolo (es. una
        # riproduzione futura per singolo sito) 9 membri + 1 non-membro
        # passerebbe questo controllo e produrrebbe un AUC reale ma
        # rumorosissimo, senza alcun avviso — stesso tipo di gap di
        # bilanciamento per classe già trovato e corretto oggi in
        # run_fedmia_gradient(). Guardia esplicita per classe.
        _n_valid_member     = int((valid_mask & (labels_arr == 1)).sum())
        _n_valid_non_member = int((valid_mask & (labels_arr == 0)).sum())
        if _n_valid_member >= 5 and _n_valid_non_member >= 5:
            try:
                result["centralized_control_auc_roc"] = round(
                    float(roc_auc_score(labels_arr[valid_mask], scores_arr[valid_mask])), 6
                )
            except ValueError:
                pass
        else:
            logger.warning(
                f"[CENTRALIZED CONTROL] troppo pochi score validi per classe "
                f"(membri={_n_valid_member}, non-membri={_n_valid_non_member}, "
                f"minimo 5 per classe) — AUC non calcolato, resta None."
            )

    logger.info(
        f"[CENTRALIZED CONTROL] AUC-ROC: {result['centralized_control_auc_roc']} "
        f"(membri={result['centralized_control_n_member']}, "
        f"non-membri={result['centralized_control_n_non_member']})"
    )
    return result


# ── Dispatch pluggable degli attacchi (src/plugins/attacks/) ───────────────────

def run_registered_attacks(
    cfg: dict,
    train_sessions: list[dict[str, Any]],
    holdout_sessions: list[dict[str, Any]],
    fl_results: dict[int, dict[str, Any]],
    extra_attacks: dict[str, type] | None = None,
    skip_attacks: set[str] | None = None,
    **attack_kwargs: Any,
) -> dict[int, dict[str, Any]]:
    """
    Esegue tutti gli attacchi in ATTACK_REGISTRY (src/plugins/attacks/) e fonde
    i risultati per round in un unico dict.

    Fix 2026-07-24 — implementazione reale della "pluggable attack interface"
    descritta in docs/DSN2027_Positioning.md e docs/DeveloperGuide.md (prima
    di questo fix quei documenti descrivevano un meccanismo di plugin/registro
    che non esisteva nel codice — src/plugins/attacks/ conteneva solo un file
    inutilizzato, senza classe base né registro; vedi le correzioni datate
    2026-07-24 in entrambi i documenti per la storia completa).

    Sostituisce due copie quasi-identiche della stessa logica di dispatch che
    esistevano prima: questa in main() sotto, e un'altra in
    scripts/run_nvflare_mia.py::main() (analisi post-hoc sui dump NVFLARE).
    Entrambe ora chiamano questa funzione — stesso comportamento garantito
    da un solo punto di verità, non due copie che potevano divergere.

    Aggiungere un nuovo attacco: registralo in
    src/plugins/attacks/__init__.py::ATTACK_REGISTRY — non serve toccare
    questa funzione né i suoi due call site.

    Comportamento preservato IDENTICO alla versione pre-refactor (chiamate
    dirette a run_yeom()/run_shadow()/run_lira()): stesso ordine di
    esecuzione (yeom, shadow, lira), stessa policy in caso di eccezione
    (logga e continua con gli altri, non solleva — un attacco fallito non
    deve impedire il salvataggio degli altri risultati), stessa logica di
    merge per round (round assente → assegna il dict; round presente →
    update in-place, i.e. shadow/lira aggiungono campi a un round già
    popolato da yeom senza sovrascriverlo). Zero cambio di comportamento
    numerico: ogni classe wrapper in src/plugins/attacks/ chiama la funzione
    esistente, invariata, con gli stessi argomenti (vedi src/core/base_attack.py
    per il razionale — queste funzioni hanno anni di fix empirici dietro,
    non riscritte qui).

    Args:
        extra_attacks: (Sprint 10zz, 2026-09-01) default None → ZERO impatto
            sul comportamento esistente (stesso identico ATTACK_REGISTRY di
            sempre). Se fornito, {nome: classe} aggiuntivi eseguiti INSIEME
            al registro di default, senza modificarlo — usato da main() per
            --include-fedmia-gradient, senza toccare ATTACK_REGISTRY (che
            resta solo Yeom/Shadow/LiRA, i tre attacchi validati).
        attack_kwargs: passati a TUTTI gli attacchi registrati; ciascuna
            classe legge solo le chiavi che le servono (vedi
            src/plugins/attacks/lira.py per l'esempio — n_shadow,
            shadow_epochs_cap, no_dp, dp_mode, cluster_membership sono usati
            solo da LiRA, ignorati da Yeom/Shadow).
    """
    from plugins.attacks import ATTACK_REGISTRY

    # Fix 2026-07-24 (review indipendente, stesso giorno): questo loop iterava
    # una tupla hardcoded ("yeom", "shadow", "lira") invece di ATTACK_REGISTRY
    # stesso — la docstring sopra e il commento in
    # src/plugins/attacks/__init__.py promettono entrambi che registrare un
    # nuovo attacco lì basta, "non serve toccare questa funzione". Con la
    # tupla hardcoded era falso: un quarto attacco registrato non sarebbe mai
    # stato eseguito, in silenzio. Iterare ATTACK_REGISTRY.items() mantiene lo
    # stesso ordine di oggi (yeom, shadow, lira — un dict Python preserva
    # l'ordine di inserimento, e __init__.py li registra in quest'ordine) e
    # rende la promessa vera per davvero.
    _registry = ATTACK_REGISTRY if not extra_attacks else {**ATTACK_REGISTRY, **extra_attacks}
    mia_results: dict[int, dict[str, Any]] = {}
    for attack_name, attack_cls in _registry.items():
        # --skip-attacks (2026-10-05): attacchi saltati per le run di sola
        # utilita' (ablazioni, prove di meccanismo). Default None: nessun cambio.
        if skip_attacks and attack_name in skip_attacks:
            logger.warning(f"Attacco {attack_name} saltato (--skip-attacks)")
            continue
        attack = attack_cls()
        try:
            attack_results = attack.run(
                cfg, train_sessions, holdout_sessions, fl_results, **attack_kwargs
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"{attack_cls.__name__} ({attack_name}) fallito: {exc}. "
                "I risultati degli altri attacchi (e FL) vengono comunque salvati. "
                "Controllare il log per la causa (es. NaN negli score MIA).",
                exc_info=True,
            )
            continue
        for rnd, data in attack_results.items():
            if rnd in mia_results:
                mia_results[rnd].update(data)
            else:
                mia_results[rnd] = data
    return mia_results


# ── IDS Evaluation ─────────────────────────────────────────────────────────────

def run_ids(
    cfg: dict,
    fl_results: dict[int, dict[str, Any]],
    no_dp: bool = False,
) -> dict[int, dict[str, Any]]:
    """
    Valuta ByzantineDetector su ogni round FL.

    Usa PrivacyAuditor per generare AuditReport reali con threats_detected
    popolato (GRADIENT_EXPLOSION, PRIVACY_BUDGET_EXHAUSTED, ecc.).
    Un singolo auditor persiste tra i round per tracciare l'epsilon cumulativo.

    Fix IDS (Sprint 9, 2026-07-16):
      1. GRADIENT_EXPLOSION — normalizzazione peer-relative:
         Con 50 epoch di training locale, la norma L2 delle differenze di peso
         supera sempre la soglia assoluta (max_grad_norm + 3σ ≈ 15.5).
         Fix: normalizza ogni delta per la norma mediana dei peer nello stesso round,
         portando la mediana = max_grad_norm. In questo modo solo gli outlier
         statistici (incl. Byzantine ×10) superano la soglia.
      2. Krum false positive — soglia calibrata per 50 epoch:
         Con 50 epoch, la varianza naturale inter-cluster produce score Krum fino a
         3.3 anche senza attacco. L'attacco Byzantine (scale_factor=10) dà score ≈4.0.
         Fix: soglia = 3.5 (Byzantine ≥ 4.0 > soglia > max FP osservato 3.27).
      3. Budget esaurito con no_dp=True — falso allarme:
         Quando DP è disabilitato (no_dp=True), il budget non viene consumato.
         Fix: passa epsilon=1000.0 all'auditor per eliminare i BUDGET_EXHAUSTED alert.

    Nota — degradazione attesa sotto dp_mode="local" (2026-07-22):
        Sotto vera local DP, run_fl_rounds() non salva raw_updates/
        raw_global_weights in fl_results (il server non deve mai vedere il
        valore pulito, nemmeno transitoriamente — vedi run_fl_rounds()
        docstring). Questa funzione usa già `round_data.get("raw_updates") or
        round_data.get("updates", [])`, quindi sotto local DP userà
        automaticamente gli update rumorizzati; ma senza un raw_global_weights
        di riferimento, il calcolo del delta peer-relative (fix 1 sopra) degrada
        a confronto sui pesi ASSOLUTI (rumorizzati) invece che su un delta
        round-su-round, il che può produrre più falsi GRADIENT_EXPLOSION. Questo
        NON è un bug da correggere: è la conseguenza reale e attesa del fatto che
        un server che non vede mai il gradiente pulito ha una difesa IDS
        strutturalmente più debole — un punto di discussione legittimo per il
        paper (motiva secure aggregation o central DP come alternative più
        IDS-compatibili quando serve sia privacy sia intrusion detection).
    """
    # Sprint 10zz+120: sovrascrivibile via CHARGESHIELD_AUDITOR_CONFIG per
    # provare configurazioni dell'Auditor (es. total_rounds_budget allineato
    # ai round reali) senza toccare il default usato da tutti gli altri run.
    import os
    config_path = os.environ.get(
        "CHARGESHIELD_AUDITOR_CONFIG",
        str(PROJECT_ROOT / "config" / "auditor.yaml"),
    )
    max_grad_norm = cfg["experiment"]["max_grad_norm"]

    # byzantine_tolerance/krum_threshold (2026-07-22, 3 siti reali + sweep IDS n=5):
    # PRIMA di questo fix, byzantine_tolerance era SEMPRE 0, anche durante lo
    # sweep dedicato con byzantine_attack.enabled=true (n=5 = 3 client reali +
    # 2 sintetici synthetic_1/synthetic_2, vedi inject_synthetic_client_indices()
    # in main()). Questo era sbagliato: la garanzia teorica di Krum di rilevare
    # f nodi Byzantine richiede n≥2f+3 — i 2 sintetici sono stati aggiunti
    # ESATTAMENTE per soddisfare n=5≥2·1+3 con f=1, quindi Krum va calcolato
    # con byzantine_tolerance=1 in quel caso (neighbors=n-f-2=2), non con f=0
    # (che userebbe neighbors=n-f-2=3 e non offrirebbe alcuna garanzia formale
    # per f=1). Nell'esperimento principale (n=3, solo client reali, mai
    # attaccato) byzantine_tolerance resta 0: con solo 3 nodi Krum funge da
    # trimmed-mean (esclude il più isolato) senza garanzia formale, il che è
    # accettabile perché quel run non inietta mai un attacco.
    _byz_cfg = cfg.get("byzantine_attack", {})
    _byz_enabled = _byz_cfg.get("enabled", False)
    _byz_tolerance = 1 if _byz_enabled else 0

    # FIX 2026-07-22 (review indipendente pre-push): byzantine_tolerance sopra
    # è derivato SOLO dal flag di config, non dal numero di client realmente
    # presenti in fl_results. Questo è un problema concreto per
    # scripts/run_nvflare_mia.py: i 2 client sintetici esistono SOLO nella
    # simulazione (mai portati su NVFLARE, per scelta esplicita — vedi
    # docs/NVFlareIntegration.md), quindi un dump NVFLARE reale ha SEMPRE e
    # SOLO 3 client anche se byzantine_attack.enabled=true è rimasto nel
    # config usato per l'analisi post-hoc. Con byzantine_tolerance=1 e solo 3
    # client, KrumDetector.compute_scores() (src/ids/charging_ids.py) applica
    # il guard n<2f+3 (3<5) e azzera silenziosamente TUTTI gli score (con un
    # solo warning nei log, facile da perdere) — non un crash, ma un IDS che
    # sembra funzionare e in realtà non rileva nulla. Cross-check: conta i
    # node_id distinti effettivamente presenti in fl_results e, se
    # byzantine_tolerance=1 non è supportato dal numero reale di client
    # (n<2·1+3=5), ricade su 0 con un warning esplicito — più sicuro che
    # lasciare che il guard di Krum lo faccia silenziosamente più a valle.
    if _byz_tolerance > 0:
        _observed_node_ids: set[str] = set()
        for _round_data in fl_results.values():
            for _upd in (_round_data or {}).get("raw_updates") or (_round_data or {}).get("updates", []):
                _nid = getattr(_upd, "node_id", None)
                if _nid is not None:
                    _observed_node_ids.add(_nid)
        _n_observed = len(_observed_node_ids)
        if _n_observed and _n_observed < 2 * _byz_tolerance + 3:
            logger.warning(
                f"[IDS] byzantine_attack.enabled=true ma solo {_n_observed} client "
                f"osservati in fl_results ({sorted(_observed_node_ids)}) — insufficienti "
                f"per la garanzia Krum n≥2f+3 con f={_byz_tolerance} (serve n≥{2*_byz_tolerance+3}). "
                "Probabile analisi di un run NVFLARE reale (dove i client sintetici non "
                "esistono, vedi docs/NVFlareIntegration.md) con un config di simulazione "
                "lasciato a byzantine_attack.enabled=true. Ricado su byzantine_tolerance=0 "
                "per evitare che Krum azzeri silenziosamente tutti gli score."
            )
            _byz_tolerance = 0

    # krum_threshold=3.5: calibrato empiricamente (2026-07-16) su n=4 "cluster"
    # fittizi — che in realtà erano 4 fette CONTIGUE dello STESSO singolo sito
    # reale (varianza inter-client dovuta solo a rumore di campionamento, MAI
    # a eterogeneità reale tra siti). Con 50 epoch, varianza naturale → score
    # Krum fino a ~3.3 (FP osservato). Attacco Byzantine ×10 → score ≈4.0.
    # Soglia 3.5: rileva Byzantine, non FP. Precedente 1.5 era calibrato per
    # 3 epoch (varianza bassa, score legittimi ≤1.1).
    #
    # ATTENZIONE — NON ANCORA RI-VALIDATA per n=5 (2026-07-22): lo sweep IDS
    # con n=5 usa 3 siti reali GENUINAMENTE eterogenei (Caltech/JPL/Office1 —
    # EVSE count, popolazione, pattern di ricarica diversi) più 2 client
    # sintetici che sono ri-affettature del pool COMBINATO (quindi vicini alla
    # "media" globale). Questo cambia la natura della varianza naturale rispetto
    # al vecchio caso (4 fette identiche dello stesso sito): un sito reale
    # potrebbe ora apparire geometricamente isolato rispetto ai 2 sintetici
    # anche SENZA alcun attacco, producendo falsi positivi Byzantine su un
    # client legittimo — oppure, al contrario, la soglia 3.5 potrebbe restare
    # valida se la varianza inter-sito è comunque dominata dal training (50
    # epoch). Non è stato possibile verificarlo in questa sessione (training
    # reale richiede torch/nvflare, non disponibili in questo sandbox).
    # PRIMA di fidarsi degli alert Krum di un run byzantine_attack.enabled=true,
    # eseguire lo sweep una volta con scale_factor=10 e ispezionare gli score
    # Krum reali in experiments/.../ids_audit_results (o l'export NVFLARE) per
    # i 3 client reali SENZA attacco attivo (o nei round prima che l'attaccante
    # venga scelto) — se il loro score naturale supera già ~3.0-3.5, alzare
    # krum_threshold di conseguenza prima di considerare i risultati attendibili.
    krum_threshold = 3.5

    ids = ByzantineDetector(
        config_path=config_path,
        byzantine_tolerance=_byz_tolerance,
        cosine_threshold=0.3,
        krum_threshold=krum_threshold,
    )
    # Fix 3a: budget — se no_dp=True, epsilon enorme → budget_ratio resta ~0 (no BUDGET_EXHAUSTED).
    # Fix 3b: explosion threshold — con epsilon=1000, sigma≈0.005 → threshold≈1.015.
    #   Dopo peer-relative normalisation, median_norm=max_grad_norm=1.0 → ~50% client sopra 1.015
    #   → GRADIENT_EXPLOSION falsi in OGNI round della baseline no-DP.
    #   Con float("inf") il check è disabilitato: sensato perché senza DP non c'è sigma di rumore
    #   su cui basare la soglia; il Byzantine check rimane affidato a Krum.
    # Sprint 10zz+120: il ramo binario qui sopra presuppone che no_dp implichi
    # "nessun rumore", il che non vale piu' da quando esiste la DP a livello di
    # record: --no-dp disattiva il meccanismo client-level ma NON record_dp, che
    # applica rumore gaussiano per esempio. Prima di questo fix una cella
    # record-DP finiva nel ramo float("inf") e l'Auditor girava con il controllo
    # di esplosione disabilitato e un budget dimensionato per 1000 round su 3
    # effettivi (budget_ratio ~0.3%), quindi non poteva segnalare nulla in
    # nessuna condizione. Verificato su _audit_ref_s42 / _audit_rdp_s42:
    # telemetria identica a sei cifre nelle due celle, sensitivity saturata a
    # 1.0, explosion_threshold = Infinity in entrambe.
    _rdp_cfg = (cfg.get("ml", {}) or {}).get("record_dp") or {}
    _rdp_on  = bool(_rdp_cfg.get("enabled"))
    # 2026-10-13: la DP per soggetto ha lo stesso problema (rumore anche con
    # --no-dp) e si tratta allo stesso modo, con il suo C e il suo sigma.
    _sdp_cfg = _subject_dp_config(cfg.get("ml", {}) or {})
    if _sdp_cfg["enabled"] and not _rdp_on:
        _rdp_cfg = _sdp_cfg
        _rdp_on = True

    if _rdp_on:
        # Rumore per-esempio: la soglia va calibrata sul clipping PER RECORD e
        # sul sigma effettivamente applicato, non sul sigma client-level (che
        # con --no-dp non esiste). Stessa forma della formula Gaussian 3-sigma
        # del ramo client-level, con C e sigma del meccanismo giusto.
        _rdp_C     = float(_rdp_cfg.get("max_grad_norm", 1.0))
        _rdp_sigma = float(_rdp_cfg.get("noise_multiplier", 0.0))
        _auditor_epsilon  = 1000.0 if no_dp else cfg["experiment"]["epsilon"]
        _explosion_thresh = _rdp_C + 3.0 * _rdp_sigma * _rdp_C
    elif no_dp:
        _auditor_epsilon  = 1000.0
        _explosion_thresh = float("inf")
    else:
        _auditor_epsilon  = cfg["experiment"]["epsilon"]
        _explosion_thresh = None   # None → formula Gaussian 3-sigma client-level

    logger.info(
        "[AUDITOR] epsilon=%s explosion_threshold=%s (record_dp=%s, subject_dp=%s, no_dp=%s)",
        _auditor_epsilon, _explosion_thresh,
        bool(((cfg.get("ml", {}) or {}).get("record_dp") or {}).get("enabled")),
        _sdp_cfg["enabled"], no_dp,
    )
    auditor = PrivacyAuditor(
        config_path=config_path,
        epsilon=_auditor_epsilon,
        explosion_threshold=_explosion_thresh,
    )

    ids_results: dict[int, dict[str, Any]] = {}

    # FASE 8 (2026-08-31) — Privacy Auditor come vero subscriber ML Plane, non
    # più invocato imperativamente con un model_update calcolato a mano (vedi
    # PrivacyAuditorSubscriber, src/auditor/privacy_auditor.py, per la formula
    # completa — Fix 1 GRADIENT_EXPLOSION Sprint 9, invariata). run_ids() resta
    # un'analisi POST-HOC su fl_results già salvato (stessa scelta di design
    # di sempre, coerente con run_lira() — vedi docstring di modulo in
    # chargeshield_aggregator.py per il perché): "ri-riproduciamo" qui gli
    # eventi ML Plane dal dict salvato invece di ricalcolare le delta a mano —
    # stessa identica formula, stessi input, stessi output; cambia SOLO il
    # meccanismo di attivazione dell'Auditor, nessun numero già pubblicato
    # (campagna 5-seed×8-config, Sprint 10tt) cambia.
    #
    # IDS usa pesi PRE-DP (raw_updates) quando disponibili, altrimenti updates
    # (rumorizzati) — stessa preferenza "vista più raw disponibile" di sempre,
    # replicata qui scegliendo il livello Purdue dell'evento simulato (1=raw,
    # 2=privatizzato) in base a quale campo di round_data è popolato.
    mlplane    = MLPlane()
    collector  = FLArtifactCollector()
    subscriber = PrivacyAuditorSubscriber(auditor, collector, max_grad_norm)
    mlplane.subscribe(collector)
    mlplane.subscribe(subscriber)
    # Inizializza la baseline con i pesi del modello iniziale (round 0),
    # salvati in run_fl_rounds() prima dell'inizio del training loop — stesso
    # ruolo di prev_raw_global prima di questo refactor: elimina il falso
    # GRADIENT_EXPLOSION al round 1 dovuto ai pesi assoluti non ancora aggiornati.
    subscriber.set_initial_baseline((fl_results.get(0) or {}).get("raw_global_weights"))

    for round_num, round_data in sorted(
        (item for item in fl_results.items() if item[0] > 0), key=lambda x: x[0]
    ):
        _raw = round_data.get("raw_updates")
        _view = _raw if _raw else round_data.get("updates", [])

        if not _view:
            ids_results[round_num] = {
                "alerts": [], "byzantine_detected": False, "drift_detected": False,
                "auditor_overhead_seconds": 0.0,  # subscriber mai attivato per questo round
            }
            continue

        # Livello Purdue dell'evento simulato: 1 (raw) solo se raw_updates era
        # popolato per questo round, altrimenti 2 (privatizzato) — determina
        # se PrivacyAuditorSubscriber userà questo round per avanzare la
        # baseline (solo da raw, mai da privatizzato — vedi
        # _handle_round_complete() per il perché: preserva la degradazione
        # intenzionale già documentata sotto dp_mode="local").
        _purdue_level = 1 if _raw else 2
        for update in _view:
            if not update or not update.node_id:
                continue
            mlplane.on_ml_event(MLPlaneEvent(
                event_type="gradient_upload",
                purdue_level=_purdue_level,
                payload=update,
                round_num=round_num,
            ))
        mlplane.on_ml_event(MLPlaneEvent(
            event_type="aggregation", purdue_level=3, payload=None, round_num=round_num,
        ))

        reports   = subscriber.reports_for_round(round_num)
        gradients = subscriber.gradients_for_round(round_num)
        # Letto SEMPRE dopo l'evento "aggregation" sopra, che ha già fatto
        # avanzare _handle_round_complete() per questo round — vale sia nel
        # ramo "reports vuoti" sotto sia in quello pieno (D1 corretto, Sprint
        # 10zz+106: overhead ML Plane attribuibile all'Auditor, misurato per
        # round invece che con due run A/B separati — vedi commento in
        # PrivacyAuditorSubscriber.__init__ per il perché).
        _auditor_overhead = subscriber.overhead_seconds_for_round(round_num)

        if not reports:
            ids_results[round_num] = {
                "alerts": [], "byzantine_detected": False, "drift_detected": False,
                "auditor_overhead_seconds": _auditor_overhead,
                # Sprint 10zz+119: telemetria Auditor. Vedi _auditor_telemetry()
                # per la nota sul fatto che privacy_score/epsilon sono un proxy.
                "auditor": _auditor_telemetry(reports),
            }
            continue

        analysis = ids.analyze_round(
            round_id=round_num,
            reports=reports,
            gradients=gradients,
        )

        ids_results[round_num] = {
            "alerts": [
                {
                    "node_id":            a.node_id,
                    "severity":           a.severity,
                    "reasons":            a.reasons,
                    "recommended_action": a.recommended_action,
                }
                for a in (analysis.alerts if analysis else [])
            ],
            "byzantine_detected":   len(analysis.byzantine_nodes) > 0 if analysis else False,
            "drift_detected":       False,
            "low_similarity_nodes": analysis.low_similarity_nodes if analysis else [],
            "auditor_overhead_seconds": _auditor_overhead,
            # Sprint 10zz+119: telemetria Auditor. Vedi _auditor_telemetry()
            # per la nota sul fatto che privacy_score/epsilon sono un proxy.
            "auditor": _auditor_telemetry(reports),
        }

    return ids_results


# ── Save Results ───────────────────────────────────────────────────────────────

def save_results(
    cfg: dict,
    mia_results: dict[int, dict[str, Any]],
    ids_results: dict[int, dict[str, Any]],
    fl_results: dict[int, dict[str, Any]] | None = None,
    sweep_dir: Path | None = None,
    extra_summary: dict[str, Any] | None = None,
    # Sprint 10zz+118: serve a _record_dp_fields() per chiamare
    # l'accountant RDP. Opzionale: senza, epsilon_record_dp resta
    # None e la nota nel JSON lo dichiara.
    n_train_sessions: int | None = None,
    # Segnalazione 4 (2026-09-24): sessioni di training di OGNI client,
    # {cluster_id: n}. L'accountant record-DP e' per client (q = B/n_i);
    # senza, epsilon_record_dp resta None invece di usare il totale.
    n_train_per_client: dict[str, int] | None = None,
    # 2026-10-13: statistiche dei soggetti (src/ml/subject_dp.py) per
    # l'accountant della DP per soggetto. None se la cella non la usa.
    subject_stats: dict[str, Any] | None = None,
) -> Path:
    """
    Salva risultati in experiments/ (o sweep_dir) con timestamp.

    Se sweep_dir è fornita, i JSON vengono salvati in quella directory
    e l'Excel verrà nominato come la directory (es.
    experiments/nodp-sweep1/nodp-sweep1.xlsx).
    Questo garantisce che ogni sweep abbia il proprio file Excel separato.

    extra_summary: (Sprint 10zz+1, 2026-09-01) default None → ZERO impatto
        sul JSON esistente. Se fornito, le chiavi vengono unite dentro
        summary["summary"] (stesso dict di mean_auc_roc/mean_lira_auc_roc/...)
        — usato da main() per centralized_control_auc_roc/_n_member/
        _n_non_member/_total_epochs (--centralized-control), senza introdurre
        un nuovo livello nello schema JSON già consumato da
        generate_excel_report.py.
    """
    if sweep_dir is not None:
        output_dir = sweep_dir
    else:
        output_dir = PROJECT_ROOT / cfg["output"]["experiments_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"experiment_{timestamp}.json"

    auc_values = [
        r["auc_roc"]
        for r in mia_results.values()
        if r.get("auc_roc") is not None
    ]
    shadow_auc_values = [
        r["shadow_auc_roc"]
        for r in mia_results.values()
        if r.get("shadow_auc_roc") is not None
    ]
    lira_auc_values = [
        r["lira_auc_roc"]
        for r in mia_results.values()
        if r.get("lira_auc_roc") is not None
    ]

    # Privacy risk basato sull'attacco più forte disponibile:
    # priority: LiRA (intercepts raw local models, strongest) > Shadow > Yeom.
    _primary_auc  = lira_auc_values if lira_auc_values else (
                    shadow_auc_values if shadow_auc_values else auc_values)
    _primary_mean = float(np.mean(_primary_auc)) if _primary_auc else None
    _primary_min  = float(np.min(_primary_auc))  if _primary_auc else None

    # Fix 2026-07-21 (review Excel/privacy_risk): un AUC-ROC molto SOTTO 0.5 non
    # è "privacy sicura" — è quasi sempre il sintomo di un attacco rotto/invertito
    # (visto in experiments/exp3 pre-fix: lira_auc_roc fino a 0.14, classificato
    # "LOW risk" dalla logica precedente). Controlliamo il MINIMO per round, non
    # solo la media: un'inversione isolata in pochi round può restare nascosta
    # nella media ma è comunque un segnale di bug da investigare, non da ignorare.
    _ANOMALY_LOW_AUC = 0.40
    _is_anomalous = _primary_min is not None and _primary_min < _ANOMALY_LOW_AUC

    # Sprint 10zz+75 (2026-09-14, fix del gap trovato dalla deep review round 5,
    # task #112/README): finché FedMIA-gradient restava un diagnostico opt-in,
    # non avere un aggregato in summary["summary"] (a differenza di Yeom/Shadow/
    # LiRA sopra) era una scelta deliberata — nessun risultato lo richiedeva.
    # Ora che è promosso a 4° attacco citabile (task #107), i suoi numeri
    # restavano visibili SOLO nel JSON grezzo per-round (per_round.<ultimo
    # round>.mia.fedmia_gradient_composed_*_per_cluster) — invisibili a
    # generate_excel_report.py, che legge solo summary["summary"] (vedi
    # load_experiments()). Qui si estrae, SE presente, il pool composto
    # cross-round per cluster (Sprint 10zz+21 — la metrica meno rumorosa,
    # coerente con "l'AUC per-cluster composto è la metrica primaria" già
    # stabilito per questo attacco) dall'UNICO round in cui
    # FedMIAGradientAttack.run() li scrive (results[max(results.keys())],
    # vedi src/plugins/attacks/fedmia_gradient.py). Zero impatto su ogni run
    # senza --include-fedmia-gradient: mia_results non contiene mai queste
    # chiavi, quindi _fedmia_gradient_composed resta None e i 3 campi sotto
    # restano None — stesso comportamento di sempre.
    _fedmia_gradient_composed: dict[str, Any] | None = None
    for _r in mia_results.values():
        if "fedmia_gradient_composed_auc_roc_per_cluster" in _r:
            _fedmia_gradient_composed = _r
            break

    # epsilon_cumulative_advanced / _best_known (Sprint 10zz+83, 2026-09-14,
    # task #118, "opzione A"): closed-form advanced-composition bound
    # (Dwork & Roth 2014, Theorem 3.20) computed alongside the naive bound
    # above, from the exact same already-logged epsilon/delta/fl_rounds —
    # no new experimental run needed, retroactively computable on every
    # past result too (see scripts/compute_advanced_composition.py). None
    # under no_dp, same guard as epsilon_cumulative_naive.
    _no_dp = cfg["experiment"].get("no_dp", False)
    # Ablazioni (2026-10-05): senza rumore o senza taglio non c'e' garanzia DP,
    # quindi nessuna contabilita' cumulata (segnalazione 38).
    _dp_ablation = cfg["experiment"].get("dp_ablation", "full")
    _no_guarantee = _no_dp or _dp_ablation != "full"
    _epsilon_advanced: float | None = None
    _delta_advanced: float | None = None
    _epsilon_best_known: float | None = None
    if not _no_guarantee:
        _epsilon_advanced, _delta_advanced = _advanced_composition_epsilon(
            epsilon_per_round=cfg["experiment"]["epsilon"],
            delta_per_round=cfg["experiment"]["delta"],
            rounds=cfg["experiment"]["fl_rounds"],
        )
        _epsilon_naive = cfg["experiment"]["epsilon"] * cfg["experiment"]["fl_rounds"]
        _epsilon_best_known = min(_epsilon_naive, _epsilon_advanced)

    summary = {
        "experiment_name": cfg["experiment"]["name"],
        "timestamp":       timestamp,
        "config": {
            "epsilon":    cfg["experiment"]["epsilon"],
            "delta":      cfg["experiment"]["delta"],
            "fl_rounds":  cfg["experiment"]["fl_rounds"],
            "proximal_mu": cfg["ml"]["proximal_mu"],
            # common_init (2026-09-24, segnalazione 48): True solo nella cella di
            # controllo; assente o False = client con inizializzazioni diverse,
            # come in tutte le run prodotte fino a quella data.
            "common_init": bool(cfg["ml"].get("common_init", False)),
            # max_grad_norm (2026-09-24, segnalazione 53): la soglia di clipping C
            # client-level. Prima non era salvata, quindi due run con C diversi e
            # stesso (dp_mode, epsilon) erano indistinguibili dal JSON. Tutti i
            # config presenti nel repository a quella data hanno C = 1.0.
            "max_grad_norm": cfg["experiment"].get("max_grad_norm"),
            # epochs (2026-08-27): aggiunto per poter distinguere risultati di una
            # sweep di calibrazione epochs (sanity-check positivo, vedi
            # docs/TestRoadmap_DSN2027.md #2) leggendo il JSON, invece di doversi
            # fidare del nome della sweep-dir o della memoria di chi ha lanciato il
            # comando — stessa classe di provenance-bug gia' trovata piu' volte in
            # questo progetto (Sprint 10r, e il fix di check_significance.py dello
            # stesso giorno).
            "epochs":     cfg["ml"]["epochs"],
            # hidden_dims/latent_dim (2026-08-28, Sprint 10jj): stessa motivazione
            # di provenance di "epochs" sopra — la sweep di escalation capacità
            # (docs/TestRoadmap_DSN2027.md, seguito alla sweep epochs che non ha
            # mostrato overfitting fino a 1000 epoche) deve poter distinguere i
            # propri risultati leggendo il JSON. None/4 = architettura storica.
            "hidden_dims": cfg["ml"].get("hidden_dims"),
            "latent_dim":  cfg["ml"].get("latent_dim", 4),
            # feature_names (2026-08-28, Sprint 10kk): stessa motivazione di
            # provenance — la sweep di escalation feature-entropy (seguita
            # all'escalation di capacità, Sprint 10jj, anch'essa senza segnale)
            # deve poter distinguere i propri risultati leggendo il JSON.
            # None = 6 feature storiche.
            "feature_names": cfg["ml"].get("feature_names"),
            # no_dp=True → baseline senza rumore DP; usato per disambiguare AUC≈0.5
            "no_dp":      cfg["experiment"].get("no_dp", False),
            # dp_mode (2026-07-22): quale placement DP — "dp-fedavg" (default),
            # "central" o "local". Vedi docs/CaseStudies.md §2.4.3 per la
            # tassonomia. Irrilevante quando no_dp=True.
            "dp_mode":    cfg["experiment"].get("dp_mode", "dp-fedavg"),
            # dp_ablation / skipped_attacks (2026-10-05): ablazioni del meccanismo
            # DP e attacchi saltati, vedi --dp-ablation e --skip-attacks.
            "dp_ablation": _dp_ablation,
            "skipped_attacks": cfg["experiment"].get("skipped_attacks"),
            # Sprint 10zz+170: fattori dell'attaccante che riscala (None = spento).
            "canary_rescale": cfg.get("lira", {}).get("canary_rescale"),
            # seed: necessario per multi-seed aggregation (mean±std) — fix M1
            "seed":       cfg["experiment"].get("seed", 42),
            # canary (2026-09-16, Sprint 10zz+113): provenance-bug trovato
            # confrontando i JSON di _canary_maxmemo e _canary_maxmemo_swap —
            # erano INDISTINGUIBILI fra loro dal solo file di output, perché
            # il blocco cfg["canary"] (enabled/site/n_templates/n_duplicates/
            # n_nonmember_templates/nonmember_source/swap_assignment) non
            # veniva salvato da nessuna parte. Due run con iniezione canary
            # diversa producevano lo stesso `config`, quindi l'unico modo di
            # sapere quale fosse quale era il nome della sweep-dir o la
            # memoria di chi aveva lanciato il comando — esattamente la
            # classe di bug che "epochs"/"hidden_dims" sopra hanno già chiuso
            # due volte. None quando i canary non sono abilitati (ogni run
            # pubblicato non-canary), quindi nessun cambiamento per quelli.
            "canary":     (cfg.get("canary") if cfg.get("canary", {}).get("enabled") else None),
            # partition_strategy / split_strategy (2026-09-21, Fase A della
            # guida scientifica). Stessa classe di bug di "canary" qui sopra,
            # trovata dall'inventario: NESSUNA delle 235 run registrava con
            # quale partizione fosse stata eseguita, quindi non esisteva modo
            # di distinguere a posteriori un riferimento IID da uno per sito —
            # e infatti la matrice di confrontabilita' ha dovuto dichiarare
            # RQ2 non costruibile. Salvarle entrambe rende ogni run futura
            # auto-descrittiva su questo asse.
            #   partition_strategy: come le sessioni sono assegnate ai CLIENT
            #                       ('per_site' = non-IID naturale, 'iid' =
            #                        rimescolate fra i siti a numerosita' fissa)
            #   split_strategy:     come si separa train da holdout
            #                       ('random' oppure 'entity_aware')
            "partition_strategy": (cfg.get("partition", {}) or {}).get("strategy", "per_site"),
            "split_strategy":     (cfg.get("split", {}) or {}).get("strategy", "random"),
            # git_commit (2026-09-21, Fase A). L'inventario ha mostrato che
            # NESSUNA delle 235 run registra il commit con cui e' stata
            # prodotta, quindi la tracciabilita' chiesta dalla riga 150 della
            # guida non e' ricostruibile a posteriori per nulla di gia' fatto:
            # dedurla dalla data sarebbe l'analogia che la guida vieta. Da qui
            # in avanti la si registra. `dirty` segnala che l'albero di lavoro
            # aveva modifiche non committate al momento del run, nel qual caso
            # il solo hash NON identifica il codice eseguito.
            "git_commit": _git_commit_corrente(),
            # epsilon_cumulative_naive (2026-07-22, review indipendente pre-push):
            # PRIMA di questo fix, solo l'epsilon NOMINALE per-round veniva
            # esportato — mai il budget cumulativo reale su tutti i round.
            # gradient_manager.py::_compute_sigma() documenta che (a) la
            # garanzia (ε,δ)-DP formale vale solo per epochs=1 (qui epochs=50),
            # e (b) sotto composizione naive (nessuna amplificazione Rényi/
            # sub-sampling) il budget cumulativo reale su T round è
            # ε_tot ≈ T × ε_per_round — MOLTO più permissivo del solo ε
            # nominale riportato finora. Vedi docs/CaseStudies.md §2.4.3 per
            # la tabella e la spiegazione completa. Questo numero è la
            # spiegazione più rigorosa del perché LiRA continua a rilevare
            # membership anche con "DP attiva" a ε nominale basso: riportarlo
            # esplicitamente nei risultati (non solo in un commento nel
            # codice) rende il claim "c'è fuga di privacy reale nonostante la
            # DP" verificabile numero alla mano, non solo qualitativo.
            # None quando no_dp=True (nessun budget DP consumato).
            "epsilon_cumulative_naive": (
                None if _no_guarantee
                else cfg["experiment"]["epsilon"] * cfg["experiment"]["fl_rounds"]
            ),
            # epsilon_cumulative_advanced / delta_cumulative_advanced /
            # epsilon_cumulative_best_known (Sprint 10zz+83, 2026-09-14,
            # task #118): advanced-composition bound (Dwork & Roth 2014,
            # Theorem 3.20, see _advanced_composition_epsilon() above) —
            # "opzione A" of the naive-composition remediation discussed
            # with the user (Sprint 10zz+81/82). best_known = min(naive,
            # advanced): advanced composition is NOT always tighter (it's
            # a large-k/small-ε asymptotic improvement), so we report
            # whichever closed-form bound is actually smaller for this
            # specific (ε, δ, rounds) rather than assume advanced wins.
            # None under no_dp, same guard as epsilon_cumulative_naive.
            "epsilon_cumulative_advanced": _epsilon_advanced,
            "delta_cumulative_advanced":   _delta_advanced,
            "epsilon_cumulative_best_known": _epsilon_best_known,
            # Sprint 10zz+117: record_dp/norm/epsilon_record_dp. Senza questi un
            # JSON record-DP e' indistinguibile da uno no-DP, e il campo
            # 'epsilon' qui sopra (client-level) viene letto come budget della
            # cella, che e' falso: a sigma=1.0 il budget reale vale ~88, non 1.0.
            **_record_dp_fields(cfg, n_train_sessions, n_train_per_client),
            # 2026-10-13: DP per soggetto. Se attiva, epsilon_subject_dp (non
            # il campo 'epsilon') e' il budget della cella.
            **_subject_dp_fields(cfg, subject_stats),
        },
        "summary": {
            # Yeom 2018 — loss-based MIA sul modello globale (baseline debole)
            "mean_auc_roc": float(np.mean(auc_values)) if auc_values else None,
            "max_auc_roc":  float(np.max(auc_values))  if auc_values else None,
            "min_auc_roc":  float(np.min(auc_values))  if auc_values else None,
            # Shadow calibrated — Carlini 2022 stile, modello globale
            "mean_shadow_auc_roc": float(np.mean(shadow_auc_values)) if shadow_auc_values else None,
            "max_shadow_auc_roc":  float(np.max(shadow_auc_values))  if shadow_auc_values else None,
            "min_shadow_auc_roc":  float(np.min(shadow_auc_values))  if shadow_auc_values else None,
            # LiRA — Carlini 2022, server-side sul singolo update di ogni client
            # PRE-aggregazione FedAvg, POST-privatizzazione DP (fix 2026-07-21c) — attacco primario
            "mean_lira_auc_roc": float(np.mean(lira_auc_values)) if lira_auc_values else None,
            # RMIA (2026-10-13), sugli stessi shadow di LiRA
            "mean_rmia_auc_roc": (float(np.mean([r["rmia_auc_roc"] for r in mia_results.values()
                                                 if r.get("rmia_auc_roc") is not None]))
                                  if any(r.get("rmia_auc_roc") is not None for r in mia_results.values())
                                  else None),
            "max_lira_auc_roc":  float(np.max(lira_auc_values))  if lira_auc_values else None,
            "min_lira_auc_roc":  float(np.min(lira_auc_values))  if lira_auc_values else None,
            # Privacy risk: usa l'attacco più forte (LiRA > Shadow > Yeom)
            "primary_attack": (
                "LiRA"   if lira_auc_values else
                "Shadow" if shadow_auc_values else
                "Yeom"
            ),
            # ANOMALY ha priorità su tutto il resto: un min_*_auc_roc < 0.40 indica
            # quasi certamente un bug nell'attacco (score invertito), non un dato
            # di privacy risk affidabile — va investigato, non riportato come LOW.
            #
            # FIX 2026-09-11 (bug reale trovato durante un audit richiesto
            # dall'utente, non da un run fallito): questa soglia usava
            # HIGH>0.7/MEDIUM>0.6, ma scripts/generate_excel_report.py
            # (_auc_risk_color(), Fix 2026-07-21 successivo a questo — mai
            # riportato qui) affina la banda a BAD>0.60/WARN>0.52/GOOD<=0.52
            # per lo stesso identico _primary_mean. Le due soglie erano
            # divergute silenziosamente: un AUC in (0.52, 0.7] veniva
            # riportato "LOW"/"MEDIUM" qui ma colorato WARN/BAD (giallo/rosso)
            # nello stesso report Excel — due verdetti incoerenti per lo
            # stesso numero. Nessun risultato già pubblicato è interessato
            # (ogni campagna reale è ~0.49-0.52, ben dentro "LOW" in
            # entrambi gli schemi) — allineato ora alla banda più fine
            # dell'Excel, l'unica delle due già corretta con un fix datato.
            "privacy_risk": (
                "ANOMALY" if _is_anomalous else
                "HIGH"    if _primary_mean is not None and _primary_mean > 0.60 else
                "MEDIUM"  if _primary_mean is not None and _primary_mean > 0.52 else
                "LOW"
            ),
            # FedMIA-gradient (Sprint 10zz+75) — None su ogni run senza
            # --include-fedmia-gradient, vedi commento sopra _fedmia_gradient_composed.
            # Diagnostico/4° attacco, non usato per primary_attack/privacy_risk
            # sopra (che restano Yeom/Shadow/LiRA, invariati) — solo per
            # renderlo visibile a generate_excel_report.py.
            "fedmia_gradient_auc_roc_per_cluster": (
                _fedmia_gradient_composed["fedmia_gradient_composed_auc_roc_per_cluster"]
                if _fedmia_gradient_composed else None
            ),
            "fedmia_gradient_advantage_per_cluster": (
                _fedmia_gradient_composed["fedmia_gradient_composed_advantage_per_cluster"]
                if _fedmia_gradient_composed else None
            ),
            "fedmia_gradient_n_test_per_cluster": (
                _fedmia_gradient_composed["fedmia_gradient_composed_n_test_per_cluster"]
                if _fedmia_gradient_composed else None
            ),
        },
        "per_round": {
            # Itera sull'unione di tutti i round: FL, MIA e IDS.
            # round 0 è escluso: contiene solo raw_global_weights (init model) per IDS.
            str(r): {
                "fl":  {"mean_loss": (fl_results or {}).get(r, {}).get("mean_loss"),
                        # 2026-09-24: norma L2 del delta di ogni client prima della
                        # DP, per scegliere max_grad_norm sui valori reali.
                        "delta_norm_per_client": (fl_results or {}).get(r, {}).get(
                            "delta_norm_per_client")},
                "mia": mia_results.get(r, {}),
                "ids": ids_results.get(r, {}),
            }
            for r in sorted(
                set(mia_results.keys())
                | {k for k in (fl_results or {}).keys() if k > 0}
                | set(ids_results.keys())
            )
        },
    }

    if extra_summary:
        summary["summary"].update(extra_summary)

    with open(result_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Fix 2026-07-22 (trovato analizzando i risultati nodp/dp reali, non da una
    # review del codice): questo log stampava mean_auc_roc (la media di Yeom,
    # l'attacco PIÙ DEBOLE — quasi sempre ≈0.50, a prescindere da ε) accanto al
    # verdetto privacy_risk, che invece è calcolato da _primary_mean (LiRA
    # quando disponibile — vedi "Privacy risk basato sull'attacco più forte"
    # sopra). Il log dava l'impressione fuorviante che il numero mostrato
    # spiegasse il verdetto, mentre erano due metriche diverse (visto dal vivo:
    # "AUC-ROC medio: 0.4988 — Privacy risk: ANOMALY", con LiRA mean reale
    # ≈0.49 ma min 0.39 — il numero giusto per capire l'ANOMALY non era quello
    # stampato). Ora mostra esplicitamente l'attacco primario usato per il
    # verdetto, con Yeom a fianco per contesto.
    mean_str = f"{_primary_mean:.4f}" if _primary_mean is not None else "N/A"
    yeom_mean_str = f"{summary['summary']['mean_auc_roc']:.4f}" \
                    if summary["summary"]["mean_auc_roc"] is not None else "N/A"
    logger.info(f"Risultati salvati: {result_file.name}")
    logger.info(
        f"{summary['summary']['primary_attack']} AUC-ROC medio: {mean_str} "
        f"(Yeom: {yeom_mean_str}) — "
        f"Privacy risk: {summary['summary']['privacy_risk']}"
    )
    if summary["summary"]["privacy_risk"] == "ANOMALY":
        logger.warning(
            f"[ANOMALY] {summary['summary']['primary_attack']} ha un min AUC-ROC "
            f"< {_ANOMALY_LOW_AUC} in almeno un round — probabile bug nell'attacco "
            "(score sistematicamente invertito), NON privacy sicura. "
            "Controllare per_round prima di usare questi risultati nel paper."
        )

    # Aggiorna automaticamente il report Excel del sweep corrente
    _update_excel_report(output_dir, named_sweep=(sweep_dir is not None))

    return result_file


def _update_excel_report(sweep_dir: Path, named_sweep: bool = False) -> None:
    """
    Rigenera il report Excel a 11 sheet per il sweep corrente.

    Il nome del file Excel dipende dalla modalità:
    - sweep_dir nominata (es. experiments/nodp-sweep1) → nodp-sweep1.xlsx
    - fallback (experiments/) → ChargeShield_FL_Results.xlsx (retro-compatibilità,
      usata da `make experiment-nodp`/`experiment-dp` — run singolo-seed senza
      --sweep-dir, non parte della metodologia multi-seed)

    Usa un import Python standard invece di exec_module() per evitare il rischio
    di arbitrary code execution se il file fosse modificato da un attacker con accesso
    al filesystem. Con import standard il modulo viene caricato una sola volta e
    cachato in sys.modules — sicuro e idempotente.

    NON SOVRASCRITTURA (fix 2026-07-24, richiesta esplicita dell'utente: "i file
    excel non si devono sovrascrivere... un foglio o un file per ogni esperimento
    lanciato"): questa funzione viene chiamata una volta per ogni singolo run/seed
    completato all'interno di uno sweep. Il file "{sweep}.xlsx" continua a essere
    sovrascritto a ogni chiamata — di proposito, è la vista aggregata sempre
    aggiornata su tutti i seed raccolti finora (lo stesso principio già accettato
    per il foglio "Seed Aggregation" al suo interno). Oltre a quello, ora viene
    salvato anche uno snapshot permanente in <sweep_dir>/history/, MAI
    sovrascritto — uno per ogni esperimento lanciato, esattamente come richiesto.
    Vedi generate_excel_report.py::save_report_with_history().

    Args:
        sweep_dir:   directory dove sono i JSON del sweep (e dove salvare l'Excel)
        named_sweep: True se sweep_dir è una directory nominata (es. exp1),
                     False per backward compatibility con experiments/
    """
    try:
        from openpyxl import Workbook

        # Import diretto: generate_excel_report.py è nella stessa directory di questo script.
        # Se il file non esiste, ImportError viene catturato sotto.
        _scripts_dir = str(Path(__file__).parent)
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        import generate_excel_report as gen  # noqa: PLC0415

        records = gen.load_experiments(sweep_dir)
        if not records:
            return

        wb = Workbook()
        wb.remove(wb.active)
        gen.build_raw_data(wb.create_sheet("Raw Data"),                   records)
        gen.build_heat_map(wb.create_sheet("Heat Map"),                   records)
        gen.build_per_rounds(wb.create_sheet("Per Rounds"),               records)
        gen.build_per_epsilon(wb.create_sheet("Per Epsilon"),             records)
        gen.build_comparison(wb.create_sheet("Comparison"),               records)
        gen.build_auc_progression(wb.create_sheet("AUC Progression"),     records)
        gen.build_attack_comparison(wb.create_sheet("Attack Comparison"), records)
        gen.build_yeom_per_round(wb.create_sheet("Yeom Per Round"),         records)
        gen.build_shadow_per_round(wb.create_sheet("Shadow Per Round"),   records)
        gen.build_lira_per_round(wb.create_sheet("LiRA Per Round"),       records)
        gen.build_seed_aggregation(wb.create_sheet("Seed Aggregation"),   records)
        wb.properties.title   = "ChargeShield-FL Experiment Results"
        wb.properties.subject = "FedMIA vs Differential Privacy — DSN 2027"

        # Nome file Excel: sweep nominato → "{nome}.xlsx", fallback → nome storico
        if named_sweep:
            output_path = sweep_dir / f"{sweep_dir.name}.xlsx"
        else:
            output_path = sweep_dir / "ChargeShield_FL_Results.xlsx"

        snapshot_path = gen.save_report_with_history(wb, output_path)
        logger.info(
            f"Report Excel aggiornato: {output_path.name} "
            f"(snapshot permanente: {snapshot_path.relative_to(sweep_dir)})"
        )
    except ImportError:
        logger.warning(
            "openpyxl non trovato — report Excel non generato. "
            "Installa con: pip install openpyxl"
        )
    except Exception as exc:
        logger.warning(f"Report Excel non generato: {exc}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChargeShield-FL — FedMIA Experiment Runner"
    )
    parser.add_argument("--config",   type=Path, default=Path("config/experiment.yaml"))
    parser.add_argument("--epsilon",  type=float, default=None)
    parser.add_argument("--rounds",   type=int,   default=None)
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=(
            "Override cfg['ml']['epochs'] (default 50). Aggiunto 2026-08-27 per "
            "la calibrazione del sanity-check positivo (docs/TestRoadmap_DSN2027.md "
            "#2) — permette di variare le epoche locali da riga di comando invece "
            "di editare il config. Alza anche shadow_epochs in LiRA/Shadow MIA "
            "(derivato da questo valore), non solo il training del modello target."
        ),
    )
    parser.add_argument("--skip-ids", action="store_true")
    parser.add_argument(
        "--include-fedmia-gradient", action="store_true",
        help=(
            "Aggiunge FedMIAGradientAttack (Sprint 10zz, 2026-09-01) alla run "
            "degli attacchi — NON in ATTACK_REGISTRY di default perché è un "
            "primo draft mai eseguito con torch reale (vedi docstring di "
            "run_fedmia_gradient()). Raddoppia il costo computazionale di "
            "LiRA (chiama run_lira() una seconda volta internamente)."
        ),
    )
    parser.add_argument(
        "--fedmia-gradient-normalize", action="store_true",
        help=(
            "Test diagnostico mirato (Sprint 10zz+17, 2026-09-02, deciso dopo "
            "Sprint 10zz+14): normalizza ogni vettore di peso shadow alla "
            "propria norma L2 unitaria PRIMA della calibrazione FedMIA — "
            "distingue se l'AUC=0.0 osservato è dominato dalla scala assoluta "
            "del vettore (AUC dovrebbe salire verso 0.5 con questo attivo) o "
            "da un confondimento strutturale del disegno round+cluster (AUC "
            "resterebbe vicino a 0.0 anche con vettori unit-norm). Richiede "
            "--include-fedmia-gradient; no-op altrimenti. Vedi Args di "
            "run_fedmia_gradient()."
        ),
    )
    parser.add_argument(
        "--centralized-control", action="store_true",
        help=(
            "Esperimento di controllo (Sprint 10zz+1, 2026-09-01): addestra un "
            "secondo modello, stessa architettura/dati/budget di epoche ma "
            "centralizzato (no FedAvg, no partizionamento per cluster) — isola "
            "l'effetto 'FL come regolarizzatore' da 'capacità limitata del "
            "modello'. Aggiunge centralized_control_auc_roc/_n_member/"
            "_n_non_member/_total_epochs al JSON risultato, invariato per il "
            "resto. Vedi docstring di run_centralized_control()."
        ),
    )
    parser.add_argument(
        "--per-sample-dump", type=str, default=None,
        help=(
            "Percorso file (Sprint 10zz+27, 2026-09-03, task worst-case): "
            "scrive un JSON con lo score composto di OGNI singolo campione "
            "scorato da LiRA (session_id reale, is_member, is_canary, "
            "composed_score) — non un aggregato. Costruito per confrontare "
            "seed diversi della stessa config con "
            "scripts/analyze_worst_case_vulnerability.py e verificare se "
            "record specifici sono ripetutamente ad alta confidenza "
            "(vulnerabilità worst-case genuina) o se cambia ogni volta "
            "(rumore). Letto solo da LiRAAttack; no-op per gli altri "
            "attacchi. Richiede la modalità composta (sempre attiva per "
            "LiRA in questo script); nessun impatto se omesso (default None)."
        ),
    )
    parser.add_argument(
        "--roc-curve-dump-dir", type=str, default=None,
        help=(
            "Directory (task #54, Sprint 10zz+29, 2026-09-03): scrive per "
            "ogni attacco eseguito (Yeom/Shadow/LiRA) un JSON con la curva "
            "ROC COMPLETA (fpr/tpr per ogni soglia, non un aggregato come "
            "AUC/TPR@fixed/Advantage) — richiesto dall'utente dopo la "
            "discussione su Carlini et al. 2022 sul comportamento a FPR "
            "vicino a zero, visualizzabile solo con la curva completa in "
            "scala log-log, non con un singolo numero. Un file per attacco "
            "dentro la directory (roc_curves_yeom.json/_shadow.json/"
            "_lira.json), letto poi da scripts/plot_roc_log_scale.py. "
            "Nessun impatto se omesso (default None)."
        ),
    )
    parser.add_argument(
        "--raw-loss-dump", type=str, default=None,
        help=(
            "Percorso file (task #57, Sprint 10zz+32, 2026-09-03): scrive un "
            "JSON con le liste COMPLETE di MSE grezza (member/nonmember, "
            "pooled su tutti i campioni, per ogni round) usate internamente "
            "da run_lira() per il fit Gaussiano (μ_in/σ_in, vedi "
            "docs/MetricsReference_DSN2027.md §3) — non un aggregato (la "
            "media è già salvata in lira_debug_raw_loss_member_mean). "
            "Richiesto dall'utente per verificare empiricamente se la MSE "
            "grezza è approssimativamente Gaussiana (Carlini et al. 2022 lo "
            "verificano solo DOPO un logit-scaling della confidenza, "
            "trasformazione senza equivalente naturale per una MSE di "
            "ricostruzione) — vedi scripts/check_gaussian_fit.py per "
            "l'analisi (skewness/curtosi/statistica Jarque-Bera, MSE grezza "
            "vs log-trasformata). Letto solo da LiRAAttack; no-op per gli "
            "altri attacchi. Nessun impatto se omesso (default None)."
        ),
    )
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument(
        "--sweep-dir", type=Path, default=None,
        help=(
            "Directory del sweep corrente (es. experiments/nodp-sweep1). "
            "Se fornita, JSON e Excel vengono salvati qui con nome del sweep "
            "(es. nodp-sweep1.xlsx). Permette di isolare i risultati di sweep distinti."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Seed riproducibilità (override experiment.yaml). "
            "Usato per: shuffle sessioni, DataLoader, init modello. "
            "Valori consigliati per sweep: 42 123 456 789 1234."
        ),
    )
    parser.add_argument(
        "--byzantine", action="store_true", default=False,
        help="Abilita Byzantine attack (gradient scaling) sul nodo configurato in experiment.yaml.",
    )
    parser.add_argument(
        "--byzantine-node", type=str, default=None,
        help=(
            "Override cluster attaccante (2026-07-22: synthetic_1/synthetic_2 "
            "per lo sweep IDS a 5 client — mai un sito reale caltech/jpl/office1). "
            "Default: valore da config."
        ),
    )
    parser.add_argument(
        "--scale-factor", type=float, default=None,
        help="Override scale_factor dell'attacco. Default: valore da config (10.0).",
    )
    parser.add_argument(
        "--no-dp", action="store_true", default=False,
        help=(
            "Disabilita rumore Differential Privacy (σ=0). "
            "BASELINE CRITICO per DSN 2027: distingue "
            "Scenario A (DP sopprime MIA → AUC>0.5 senza DP) da "
            "Scenario B (modello non memorizza → AUC≈0.5 anche senza DP). "
            "Usare sempre prima del full sweep per capire in quale scenario siamo."
        ),
    )
    parser.add_argument(
        "--dp-mode", type=str, default="dp-fedavg",
        choices=["dp-fedavg", "central", "local"],
        help=(
            "Placement del meccanismo DP quando --no-dp non è passato "
            "(2026-07-22, vedi docs/CaseStudies.md §2.4.3): "
            "'dp-fedavg' (default, storico) = server clippa+rumorizza ogni "
            "update PRIMA di aggregare — placement per-client non-standard, "
            "NON quello descritto dall'Algoritmo 1 di McMahan 2018; server/IDS "
            "vede l'update raw transitoriamente. "
            "'central' [McMahan 2018, Algoritmo 1 (DP-FedAvg): clip lato "
            "client, un solo draw di rumore lato server sull'aggregato] = "
            "client clippano SENZA rumorizzare, il server aggrega "
            "pulito e aggiunge UN SOLO rumore all'aggregato — atteso: LiRA sul "
            "singolo update non mostra alcuna soppressione, a nessun ε. "
            "'local' = stesso meccanismo per-client di dp-fedavg, ma server/IDS "
            "non vede MAI l'update raw (nemmeno transitoriamente) — l'IDS perde "
            "l'accesso a raw_updates, un tradeoff reale della local DP, non un bug."
        ),
    )
    parser.add_argument(
        "--dp-ablation", type=str, default="full", choices=list(DP_ABLATIONS),
        help=(
            "Ablazione del meccanismo DP per client (2026-10-05, segnalazione 38): "
            "'full' (default) = taglio del delta + rumore, come sempre; "
            "'clip-only' = solo taglio a max_grad_norm, nessun rumore; "
            "'noise-only' = solo rumore (sigma tarato su max_grad_norm), nessun taglio. "
            "Solo con dp-fedavg o local e senza --no-dp. Nessuna garanzia DP per le "
            "ablazioni: i campi epsilon_cumulative_* del JSON restano None."
        ),
    )
    parser.add_argument(
        "--skip-attacks", type=str, default=None,
        help=(
            "Attacchi da saltare, separati da virgola, fra yeom, shadow, lira "
            "(2026-10-05): per run di sola utilita'. Es.: --skip-attacks lira. "
            "Senza LiRA mancano i numeri sui canary e il dump per campione."
        ),
    )
    parser.add_argument(
        "--canary-rescale", type=str, default=None,
        help=(
            "Fattori s separati da virgola, es. 1,2,4,8,16,32 (2026-10-09, Sprint "
            "10zz+170): oltre alla loss grezza dei canary sul modello osservato, "
            "la calcola su ref + s*(osservato - ref), con ref il modello ricevuto "
            "dal client a inizio round. Attaccante che riscala l'update tagliato. "
            "Richiede LiRA (non con --skip-attacks lira) e observation_surface client."
        ),
    )
    parser.add_argument(
        "--n-shadow", type=int, default=None,
        help=(
            "Numero di shadow models per LiRA (override config lira.n_shadow). "
            "8 = fast demo (~10 min CPU); 16 = buona qualità; ≥32 = paper quality. "
            "Più shadow = IN/OUT distributions più stabili → AUC più affidabile."
        ),
    )
    parser.add_argument(
        "--shadow-epochs-cap", type=int, default=None,
        help=(
            "Cap massimo per shadow_epochs in LiRA (override formula automatica). "
            "Formula default: min(local_epochs × max(rounds//4, 5), 300). "
            "Usare un valore basso (es. 20) nel smoke test per ridurre il tempo: "
            "con epochs=50 e rounds=5, la formula darebbe 250 — troppo per un test rapido. "
            "Non usare nei run sperimentali reali: i shadow models sarebbero sottoadatti."
        ),
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("ChargeShield-FL — Sprint 5 Experiment")
    logger.info("=" * 60)

    cfg = load_config(args.config, {"epsilon": args.epsilon, "rounds": args.rounds, "epochs": args.epochs})

    # Override seed da CLI (--seed)
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed

    # Override Byzantine attack da CLI (--byzantine, --byzantine-node, --scale-factor)
    if args.byzantine:
        cfg.setdefault("byzantine_attack", {})["enabled"] = True
    if args.byzantine_node:
        cfg.setdefault("byzantine_attack", {})["byzantine_node"] = args.byzantine_node
    if args.scale_factor is not None:
        cfg.setdefault("byzantine_attack", {})["scale_factor"] = args.scale_factor

    # Guard: --byzantine e --no-dp sono esperimenti concettualmente separati.
    # --no-dp = baseline MIA pulito (nessun attacco, nessun rumore DP).
    # --byzantine = validazione IDS (attacco attivo, non misura di privacy risk).
    # Combinarli produce un risultato privo di significato per entrambi gli obiettivi.
    if args.byzantine and args.no_dp:
        logger.error(
            "Combinazione non valida: --byzantine e --no-dp non possono essere usati insieme.\n"
            "  --no-dp   = baseline MIA pulito (solo per misurare privacy risk senza DP)\n"
            "  --byzantine = validazione IDS (solo per testare rilevamento attacchi)\n"
            "Eseguirli come esperimenti separati:\n"
            "  Privacy baseline: make experiment-nodp\n"
            "  IDS validation:   make experiment-byzantine-sweep"
        )
        sys.exit(1)

    # No-DP baseline flag: disabilita rumore DP, rinomina esperimento per distinzione
    if args.no_dp:
        cfg["experiment"]["no_dp"] = True
        cfg["experiment"]["name"] = cfg["experiment"]["name"] + "_nodp_baseline"
        if args.dp_mode != "dp-fedavg":
            logger.warning(
                f"--dp-mode {args.dp_mode} ignorato: --no-dp disabilita ogni rumore, "
                "indipendentemente dal placement DP scelto."
            )
        # Fix 2026-07-22 (review A3): normalizza args.dp_mode PRIMA che venga
        # persistito qui sotto in cfg["experiment"]["dp_mode"] — senza questo,
        # una riga con no_dp=True poteva comunque riportare dp_mode="central"/
        # "local" nel JSON/Excel, incoerente col fatto che no_dp disabilita
        # ogni rumore (il valore era innocuo a runtime — no_dp cortocircuita
        # run_fl_rounds() prima che dp_mode sia consultato — ma fuorviante nei
        # risultati salvati).
        args.dp_mode = "dp-fedavg"

    # dp_mode (2026-07-22): rinomina esperimento per distinguere le 3 modalità DP
    # nel nome/nei log — importante perché tutte e tre producono un modello con
    # rumore (auc<0.6 plausibile per tutte), ma per ragioni architetturali diverse.
    cfg["experiment"]["dp_mode"] = args.dp_mode
    if not args.no_dp and args.dp_mode != "dp-fedavg":
        cfg["experiment"]["name"] = cfg["experiment"]["name"] + f"_{args.dp_mode}_dp"

    # Ablazioni del meccanismo DP e attacchi saltati (2026-10-05).
    try:
        _check_dp_ablation(args.dp_ablation, no_dp=args.no_dp, dp_mode=args.dp_mode)
    except ValueError as exc:
        logger.error(f"--dp-ablation: {exc}")
        sys.exit(1)
    cfg["experiment"]["dp_ablation"] = args.dp_ablation
    if args.dp_ablation != "full":
        cfg["experiment"]["name"] = cfg["experiment"]["name"] + f"_{args.dp_ablation}"
    _skip_attacks: set[str] = set()
    if args.skip_attacks:
        _skip_attacks = {a.strip() for a in args.skip_attacks.split(",") if a.strip()}
        _known_attacks = {"yeom", "shadow", "lira", "fedmia_gradient"}
        _unknown = _skip_attacks - _known_attacks
        if _unknown:
            logger.error(f"--skip-attacks: nomi sconosciuti {sorted(_unknown)} (attesi fra {sorted(_known_attacks)})")
            sys.exit(1)
        if "lira" in _skip_attacks and args.per_sample_dump:
            logger.warning("--per-sample-dump ignorato: il dump per campione viene da LiRA, saltato")
    cfg["experiment"]["skipped_attacks"] = sorted(_skip_attacks) or None
    if args.canary_rescale:
        try:
            _rescale = [float(x) for x in args.canary_rescale.split(",") if x.strip()]
        except ValueError:
            logger.error(f"--canary-rescale: valori non numerici {args.canary_rescale!r}")
            sys.exit(1)
        if not _rescale or any(x <= 0 for x in _rescale):
            logger.error(f"--canary-rescale: servono fattori positivi, ricevuto {args.canary_rescale!r}")
            sys.exit(1)
        if "lira" in _skip_attacks:
            logger.error("--canary-rescale richiede LiRA: togliere lira da --skip-attacks")
            sys.exit(1)
        cfg.setdefault("lira", {})["canary_rescale"] = _rescale

    # Warning esplicito se Byzantine è attivo senza --sweep-dir: rischio di mischiare
    # risultati IDS con risultati MIA nella directory experiments/ principale.
    if args.byzantine and not args.sweep_dir:
        logger.warning(
            "[BYZANTINE] Attacco attivo senza --sweep-dir esplicita. "
            "I risultati Byzantine andrebbero in experiments/ids_validation/ "
            "(usa 'make experiment-byzantine-sweep' per garantire la separazione). "
            "I risultati MIA in questa run NON sono validi per il privacy sweep."
        )

    exp_cfg = cfg["experiment"]
    logger.info(
        f"Config: epsilon={exp_cfg['epsilon']}, "
        f"rounds={exp_cfg['fl_rounds']}, "
        f"proximal_mu={cfg['ml']['proximal_mu']}"
    )

    sessions = load_sessions(cfg)
    sessions = enrich_sessions(sessions)
    logger.info(f"Sessioni dopo enrichment: {len(sessions)}")

    # Split hold-out PRIMA del training FL: 80% train, 20% hold-out (mai visti dai nodi FL).
    # Seed fisso per riproducibilità dei risultati — fondamentale per DSN 2027.
    seed = exp_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Determinismo GPU (no overhead su CPU-only, ignorato silenziosamente se no CUDA)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Split strategy (Fase 8, 2026-08-31): "random" (default, storico,
    # invariato) o "entity_aware" (opt-in — vedi entity_aware_split() per il
    # perché: garantisce che nessuna stazione/utente compaia sia in train che
    # in holdout, chiudendo l'obiezione "i tuoi non-membri sono davvero
    # indipendenti?"). Attivabile via split.strategy nel config YAML — nessun
    # config esistente/pubblicato lo imposta, quindi ogni run già eseguito
    # resta invariato.
    _split_cfg = cfg.get("split", {})
    _split_strategy = _split_cfg.get("strategy", "random")
    if _split_strategy == "entity_aware":
        _entity_key = _split_cfg.get("entity_key", "node_id")
        _holdout_fraction = _split_cfg.get("holdout_fraction", 0.2)
        train_sessions, holdout_sessions = entity_aware_split(
            sessions, entity_key=_entity_key, holdout_fraction=_holdout_fraction, seed=seed,
        )
    else:
        random.shuffle(sessions)
        split = max(1, int(len(sessions) * 0.8))
        train_sessions   = sessions[:split]
        holdout_sessions = sessions[split:]
    logger.info(
        f"Split ({_split_strategy}) — train: {len(train_sessions)}, "
        f"hold-out: {len(holdout_sessions)}"
    )

    # Canary positive control (Sprint 10vv, 2026-08-31) — no-op se
    # cfg["canary"]["enabled"] non è True (ogni config esistente/pubblicato).
    # DEVE avvenire DOPO lo split (i canary non alterano il taglio 80/20) e
    # PRIMA di compute_feature_stats/group_indices_by_site sotto, così i
    # canary iniettati in train_sessions vengono automaticamente assegnati
    # al cluster corretto via il loro site_id reale (ereditato dal template),
    # senza bisogno di toccare group_indices_by_site().
    train_sessions, holdout_sessions = inject_canaries(train_sessions, holdout_sessions, cfg, seed)

    # Normalizzazione min-max: calcolata SOLO su train_sessions (no leakage dal hold-out).
    # Stessa trasformazione applicata a holdout_sessions per la FedMIA.
    # feature_names opzionale (Sprint 10kk, 2026-08-28): stessa chiave letta da
    # AutoencoderTrainer e da _mia_feature_names(cfg) — garantisce che normalizzazione,
    # training reale e tutti gli attacchi MIA usino esattamente lo stesso vettore di
    # feature nello stesso ordine. Default None → le 6 feature storiche (invariato).
    _FEATURES = _mia_feature_names(cfg)
    feature_stats    = compute_feature_stats(train_sessions, _FEATURES)
    train_sessions   = normalize_sessions(train_sessions,   feature_stats, _FEATURES)
    holdout_sessions = normalize_sessions(holdout_sessions, feature_stats, _FEATURES)
    logger.info(f"Feature normalizzate [0,1]: {list(feature_stats.keys())}")
    if args.dry_run:
        logger.info("Dry run completato — uscita.")
        return

    # ── Cluster reali (2026-07-22) — sostituisce lo slicing contiguo in 4 fette
    # fittizie di un unico dataset (mai stato realmente eterogeneo — vedi fix
    # mislabeling jpl/Caltech in README/CaseStudies.md). Ogni sessione porta già
    # il proprio site_id reale (ACNDataset) — group_indices_by_site() raggruppa
    # per quello, non per posizione. Vedi group_indices_by_site()/
    # inject_synthetic_client_indices() per il contratto completo.
    #
    # IMPORTANTE: i 2 client sintetici (synthetic_1/synthetic_2) esistono
    # SOLO per rendere valido il rilevamento Byzantine di Krum (che richiede
    # n≥2f+3 nodi — con f=1 servono 5, i 3 siti reali non bastano). Vengono
    # aggiunti ESCLUSIVAMENTE quando byzantine_attack.enabled=True (sweep IDS
    # dedicato, separato dallo sweep privacy). L'esperimento principale
    # (FedMIA/Shadow/LiRA, privacy_risk, tutto ciò che finisce nel paper come
    # misura di privacy) vede SEMPRE e SOLO i 3 client reali (caltech/jpl/
    # office1) — mai i sintetici, che userebbero sessioni duplicate/sovrapposte
    # tra client e invaliderebbero qualunque misura di privacy o utility.
    real_cluster_membership = group_indices_by_site(train_sessions)

    # ── Partizione IID come RIFERIMENTO SPERIMENTALE (2026-09-21, RQ2) ───────
    # Fino a oggi l'unica partizione possibile era quella per site_id reale,
    # quindi non-IID PER NATURA e non per costruzione: l'eterogeneita' non era
    # un fattore manipolabile e RQ2 non aveva un braccio di controllo. Questo
    # blocco costruisce il riferimento omogeneo che la guida chiede alla Fase E.
    #
    # DISEGNO (guida, Fase E): "Costruire un riferimento circa IID e una
    # partizione eterogenea, inizialmente mantenendo le stesse numerosita' per
    # client: cosi' la differenza riguarda principalmente la distribuzione
    # delle feature." Quindi si rimescolano le sessioni FRA i siti ma si
    # conservano ESATTAMENTE le cardinalita' per client della partizione
    # naturale. L'unica cosa che cambia e' quali sessioni finiscono dove.
    #
    # COSA ALTERA, da dichiarare nel paper: la partizione IID rompe la
    # corrispondenza client<->sito, quindi rompe anche la coerenza temporale e
    # la separazione fra utenti che il raggruppamento per sito garantiva. E'
    # un riferimento sperimentale, NON un deployment realistico.
    #
    # ATTENZIONE PER LiRA: la verita' di membership per-client cambia con la
    # partizione. Qui cambia solo l'assegnazione ai client, non lo split
    # train/holdout, quindi il pool dei non-membri resta lo stesso; ma gli
    # shadow universe per-cluster cambiano, ed e' corretto che cambino.
    _part_cfg = cfg.get("partition", {}) or {}
    _part_strategy = _part_cfg.get("strategy", "per_site")
    if _part_strategy == "iid":
        _rng_part = random.Random(seed)
        _tutti = [i for idxs in real_cluster_membership.values() for i in idxs]
        _rng_part.shuffle(_tutti)
        _nuovo: dict[str, list[int]] = {}
        _pos = 0
        for cid, idxs in real_cluster_membership.items():
            _nuovo[cid] = _tutti[_pos:_pos + len(idxs)]
            _pos += len(idxs)
        assert _pos == len(_tutti), "partizione IID: indici persi nel rimescolamento"
        assert {len(v) for v in _nuovo.values()} == {
            len(v) for v in real_cluster_membership.values()
        }, "partizione IID: le numerosita' per client devono restare identiche"
        logger.warning(
            "[PARTIZIONE IID] Le sessioni sono state rimescolate FRA i siti "
            "conservando le numerosita' per client (%s). Questo e' un "
            "RIFERIMENTO SPERIMENTALE per RQ2, non un deployment reale: la "
            "corrispondenza client-sito, la coerenza temporale e la "
            "separazione fra utenti sono deliberatamente rotte.",
            {c: len(v) for c, v in _nuovo.items()},
        )
        real_cluster_membership = _nuovo
    elif _part_strategy != "per_site":
        raise ValueError(
            f"partition.strategy sconosciuta: {_part_strategy!r}. "
            "Valori ammessi: 'per_site' (default, non-IID naturale) o 'iid'."
        )

    _byz_enabled_main = cfg.get("byzantine_attack", {}).get("enabled", False)
    if _byz_enabled_main:
        cluster_membership = inject_synthetic_client_indices(
            real_cluster_membership, n_synthetic=2, seed=seed,
        )
        logger.warning(
            "[BYZANTINE/IDS SWEEP] 2 client sintetici aggiunti (synthetic_1/"
            "synthetic_2) per validare Krum con n=5 — NON usare questo run per "
            "misure di privacy/FedMIA/LiRA, solo per validazione IDS."
        )
    else:
        cluster_membership = real_cluster_membership
    cluster_sessions = {
        cid: [train_sessions[i] for i in idxs] for cid, idxs in cluster_membership.items()
    }
    logger.info(f"Client attivi ({len(cluster_sessions)}): {list(cluster_sessions.keys())}")

    fl_results = run_fl_rounds(
        cfg, train_sessions, no_dp=args.no_dp, dp_mode=args.dp_mode,
        cluster_sessions=cluster_sessions,
    )

    # ── FedMIA/Shadow/LiRA: SOLO client reali, MAI durante uno sweep Byzantine/IDS ──
    # Istruzione esplicita 2026-07-22: i 2 client sintetici (synthetic_1/
    # synthetic_2) servono ESCLUSIVAMENTE a rendere valido il rilevamento Krum
    # (n≥2f+3). Il modello FL viene addestrato su di essi quando
    # byzantine_attack.enabled=True (sopra, cluster_sessions), ma qualunque
    # attacco di privacy calcolato su quel run includerebbe update di client le
    # cui sessioni si sovrappongono arbitrariamente tra loro e coi client reali
    # (vedi inject_synthetic_client_indices()) — un numero privo di significato
    # per qualunque claim di privacy, non solo "da usare con cautela". Per
    # questo FedMIA/Shadow/LiRA vengono saltati DEL TUTTO quando
    # byzantine_attack.enabled=True, invece di girare comunque su dati che poi
    # nessuno dovrebbe interpretare. Il run Byzantine resta quindi
    # esclusivamente uno strumento di validazione IDS (Krum/cosine/alert),
    # mai una fonte di numeri di privacy — coerente con quanto già indicato
    # dal warning "I risultati MIA in questa run NON sono validi per il privacy
    # sweep" più sopra in questa stessa funzione, reso qui vincolante invece
    # che solo informativo.
    mia_results: dict = {}
    if _byz_enabled_main:
        logger.warning(
            "[BYZANTINE/IDS SWEEP] FedMIA/Shadow/LiRA SALTATI — questo run usa "
            "client sintetici (synthetic_1/synthetic_2) solo per validare Krum, "
            "non è un esperimento di privacy valido. Per i numeri di privacy "
            "usare un run con byzantine_attack.enabled=false (solo 3 client reali)."
        )
    else:
        # Fix 2026-07-24: i tre attacchi (Yeom, Shadow, LiRA) non vengono più
        # chiamati per nome qui — main() itera ATTACK_REGISTRY tramite
        # run_registered_attacks() (src/plugins/attacks/, vedi la sua docstring
        # per il razionale completo e la garanzia di comportamento identico
        # alla versione pre-refactor). run_ids() resta chiamato direttamente
        # sotto: non è un "attacco" nel senso di questa interfaccia (produce
        # audit/detection, non un punteggio di membership), quindi resta fuori
        # dal registro — coerente con come i due concetti erano già separati
        # prima di questo fix.
        n_shadow = args.n_shadow if args.n_shadow is not None else cfg.get("lira", {}).get("n_shadow", 8)
        shadow_cap = args.shadow_epochs_cap  # None → local_epochs; int → override per smoke
        # Sprint 10zz (2026-09-01): --include-fedmia-gradient aggiunge
        # FedMIAGradientAttack SENZA toccare ATTACK_REGISTRY (resta solo
        # Yeom/Shadow/LiRA per ogni altro run) — vedi extra_attacks in
        # run_registered_attacks() e il docstring di run_fedmia_gradient()
        # per i limiti noti (primo draft, mai eseguito con torch reale).
        _extra_attacks = None
        if args.include_fedmia_gradient:
            from plugins.attacks.fedmia_gradient import FedMIAGradientAttack
            _extra_attacks = {FedMIAGradientAttack.name: FedMIAGradientAttack}
        mia_results = run_registered_attacks(
            cfg, train_sessions, holdout_sessions, fl_results,
            extra_attacks=_extra_attacks,
            skip_attacks=_skip_attacks or None,
            n_shadow=n_shadow, shadow_epochs_cap=shadow_cap, no_dp=args.no_dp,
            dp_mode=args.dp_mode, cluster_membership=cluster_membership,
            # Sprint 10zz+17: passato a TUTTI gli attacchi registrati (stesso
            # meccanismo di n_shadow/dp_mode sopra) ma letto SOLO da
            # FedMIAGradientAttack.run() — ogni altro wrapper lo ignora
            # tramite il proprio **kwargs, nessun impatto altrove.
            fedmia_gradient_normalize=args.fedmia_gradient_normalize,
            # Sprint 10zz+27: stesso meccanismo — passato a tutti gli
            # attacchi registrati, letto solo da LiRAAttack.run().
            per_sample_dump_path=args.per_sample_dump,
            # Sprint 10zz+29 (task #54): stesso meccanismo — passato a tutti
            # gli attacchi registrati, letto da YeomAttack/ShadowAttack/
            # LiRAAttack.run() (ognuno costruisce il proprio path dentro la
            # directory), ignorato dagli altri wrapper tramite **kwargs.
            roc_curve_dump_dir=args.roc_curve_dump_dir,
            # Sprint 10zz+32 (task #57): stesso meccanismo — passato a tutti
            # gli attacchi registrati, letto solo da LiRAAttack.run().
            raw_loss_dump_path=args.raw_loss_dump,
        )

    ids_results: dict = {}
    if not args.skip_ids:
        try:
            ids_results = run_ids(cfg, fl_results, no_dp=args.no_dp)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"run_ids() fallita: {exc}. Continuazione senza risultati IDS.", exc_info=True)

    # Sprint 10zz+1 (2026-09-01): esperimento di controllo opt-in, sullo STESSO
    # train_sessions/holdout_sessions già usato dal run federato sopra (stesso
    # split, stessa normalizzazione) — confronto diretto, non un run separato
    # con dati potenzialmente diversi. Vedi docstring di run_centralized_control().
    _extra_summary = None
    if args.centralized_control:
        try:
            _extra_summary = run_centralized_control(
                cfg, train_sessions, holdout_sessions,
                rounds=cfg["experiment"]["fl_rounds"],
                local_epochs=cfg["ml"]["epochs"],
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"run_centralized_control() fallita: {exc}. "
                "Continuazione senza il controllo centralizzato.",
                exc_info=True,
            )

    # sweep_dir: se fornita via --sweep-dir, i risultati vanno in quella directory
    # con Excel nominato come il sweep (es. exp1.xlsx). Altrimenti usa experiments/.
    sweep_dir = args.sweep_dir.resolve() if args.sweep_dir else None
    save_results(
        cfg, mia_results, ids_results, fl_results,
        sweep_dir=sweep_dir, extra_summary=_extra_summary,
        n_train_sessions=len(train_sessions),
        n_train_per_client={cid: len(s) for cid, s in cluster_sessions.items()},
        subject_stats=(
            _subject_stats(
                cluster_sessions,
                str(_subject_dp_config(cfg.get("ml", {}))["subject_key"]),
                # stessa regola del trainer: feature_names se presente (anche
                # vuoto), altrimenti le 6 storiche
                list((cfg.get("ml", {}) or {}).get("feature_names")
                     if (cfg.get("ml", {}) or {}).get("feature_names") is not None
                     else AutoencoderTrainer.CONTINUOUS_FEATURES),
            )
            if _subject_dp_config(cfg.get("ml", {}))["enabled"] else None
        ),
    )

    logger.info("=" * 60)
    logger.info("Esperimento completato.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
