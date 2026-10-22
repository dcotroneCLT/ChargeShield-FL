#!/usr/bin/env python3
# scripts/run_nvflare_mia.py
# ChargeShield-FL — analisi MIA/IDS post-hoc su un run NVFLARE reale (fase 5, 2026-07-22)
#
# STATO: scritto in un sandbox dove nvflare/torch non sono installabili — vedi
# docs/NVFlareIntegration.md. NON eseguito, NON testato (a differenza di
# run_experiments.py, che ha girato su dati reali). Verificare su una macchina
# con le dipendenze reali prima di fidarsi dell'output.
#
# COSA FA QUESTO SCRIPT E PERCHÉ ESISTE SEPARATO DA run_experiments.py:
#   ChargeShieldAggregator (nvflare/jobs/chargeshield_poc/app/custom/
#   chargeshield_aggregator.py) esporta, dopo ogni round di un vero job
#   NVFLARE, un dump pickle (experiments/nvflare_fl_results_<timestamp>.pkl —
#   nome univoco per run dal 2026-07-24, per non sovrascrivere run precedenti;
#   prima di quel fix era un nome fisso) con ESATTAMENTE la stessa struttura
#   dati che run_fl_rounds() produce in memoria durante la simulazione
#   single-process (mean_loss, n_participants, updates, raw_updates,
#   raw_global_weights, global_weights — un dict per round).
#
#   run_lira()/run_ids()/run_yeom()/run_shadow()/save_results() sono
#   già scritte per consumare esattamente quella struttura — e sono già state
#   validate empiricamente su nodp-sweep1/dp-sweep1 (5 round di fix per LiRA,
#   vedi la sua docstring). Riscriverle per girare "dal vivo" dentro
#   ChargeShieldAggregator.aggregate(), alla cieca (nessuna esecuzione
#   possibile in questo sandbox), sarebbe un secondo tentativo con altissima
#   probabilità di introdurre bug nuovi e silenziosi.
#
#   Questo script invece CARICA il dump prodotto da un job NVFLARE reale e
#   chiama quelle stesse funzioni, INVARIATE — zero rischio di regressione
#   sulla logica di attacco già validata. È l'equivalente, per NVFLARE, di
#   "main()" in run_experiments.py, con fl_results letto da disco invece che
#   prodotto da run_fl_rounds().
#
# LIMITE NOTO (non risolto qui): l'Aggregator/Executor NVFLARE non hanno un
# equivalente di --no-dp (bypass completo del rumore) — dp_mode è sempre uno
# dei 3 valori. Questo script chiama sempre run_ids()/run_lira() con
# no_dp=False; per un run NVFLARE "senza DP" servirebbe un epsilon molto
# grande in config_fed_server.json/config_fed_client.json (approssimazione,
# non equivalente esatto al bypass di run_fl_rounds()).
#
# Usage:
#   python scripts/run_nvflare_mia.py --config config/experiment.yaml
#       # (senza --fl-results: usa il dump nvflare_fl_results_*.pkl più recente)
#   python scripts/run_nvflare_mia.py \
#       --fl-results experiments/nvflare_fl_results_20260724_162431.pkl \
#       --n-shadow 8 --sweep-dir experiments/nvflare-run1

from __future__ import annotations

import argparse
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ── Path setup — identico a run_experiments.py ──────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # per "import run_experiments"

from adapters.acn_dataset import ACNDataset  # noqa: E402
from ml.autoencoder_trainer import AutoencoderTrainer  # noqa: E402

from run_experiments import (  # noqa: E402
    compute_feature_stats,
    statistiche_normalizzazione,
    enrich_sessions,
    load_config,
    normalize_sessions,
    run_ids,
    run_registered_attacks,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_nvflare_mia")


def load_nvflare_fl_results(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """
    Carica il dump pickle prodotto da ChargeShieldAggregator._export_fl_results()
    (fase 5). Restituisce (fl_results, meta) dove fl_results ha la stessa forma
    prodotta da run_fl_rounds() nella simulazione — vedi quella funzione per il
    contratto esatto dei campi.

    Nota: manca sempre l'entry per il round 0 (l'Aggregator non vede mai i pesi
    di init casuale, generati da persistor/shareable_generator prima del round
    1) — run_ids()/run_lira() gestiscono già questo caso (fallback a None).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dump NVFLARE non trovato: {path}. "
            "Deve essere generato da un job NVFLARE reale con ChargeShieldAggregator "
            "(vedi nvflare/jobs/chargeshield_poc/app/config/config_fed_server.json, "
            "campo 'fl_results_export_path') — questo script non lo genera da solo."
        )
    with open(path, "rb") as f:
        payload = pickle.load(f)

    meta = payload.get("meta", {})
    rounds = payload.get("rounds", {})
    # Le chiavi sono già int (scritte così da ChargeShieldAggregator) — nessuna
    # conversione da stringa necessaria (a differenza del JSON di fase 4).
    fl_results = {int(r): data for r, data in rounds.items()}
    logger.info(
        f"Caricato dump NVFLARE: {len(fl_results)} round, "
        f"dp_mode={meta.get('dp_mode')!r}, epsilon={meta.get('epsilon')!r}"
    )
    return fl_results, meta


def _inject_canaries_for_site(
    site_train: list[dict[str, Any]],
    site_holdout: list[dict[str, Any]],
    canary_cfg: dict[str, Any] | None,
    site_name: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Ricostruisce, offline, la STESSA iniezione canary che
    chargeshield_executor.py::_inject_canary_members() applica dal vivo
    (2026-09-11) — stesso blocco "canary" letto dal medesimo config_fed_
    client.json, stesso seed, stesso offset RNG (+271828). Aggiunge in più
    i gemelli non-membro (n_nonmember_templates), che l'executor non calcola
    mai (non mantiene una lista esplicita del proprio hold-out — vedi
    _inject_canary_members()) ma servono qui per canary_auc_roc/
    canary_raw_mse_auc_roc in run_lira().

    Ordine dei draw dallo stesso oggetto random.Random — DEVE combaciare con
    inject_canaries() in run_experiments.py (member_templates prima,
    nonmember_templates dopo) e con _inject_canary_members() lato executor
    (che pesca SOLO member_templates, stesso primo draw): la sequenza resta
    quindi allineata finché train/holdout ricostruiti qui coincidono con
    quelli reali — già garantito dallo split seed-based 80/20 (fix
    2026-08-03), invariato da questo cambiamento.

    No-op (restituisce gli input invariati) se canary_cfg è None/disabled o
    se "site" non è questo sito — stesso identico criterio di
    _inject_canary_members(), quindi zero rischio per ogni job/analisi
    esistente che non usa canary.
    """
    if not canary_cfg or not canary_cfg.get("enabled", False):
        return site_train, site_holdout
    target_site = canary_cfg.get("site", "office1")
    if target_site != site_name:
        return site_train, site_holdout

    n_templates           = int(canary_cfg.get("n_templates", 5))
    n_duplicates          = int(canary_cfg.get("n_duplicates", 30))
    n_nonmember_templates = int(canary_cfg.get("n_nonmember_templates", n_templates))

    if len(site_train) < n_templates or len(site_holdout) < n_nonmember_templates:
        logger.warning(
            f"[CANARY] site={site_name} non ha abbastanza sessioni ricostruite per "
            f"{n_templates} template membro / {n_nonmember_templates} gemelli "
            f"non-membro (train={len(site_train)}, holdout={len(site_holdout)}) — "
            "canary NON ricostruiti, la rianalisi prosegue senza canary_auc_roc."
        )
        return site_train, site_holdout

    rng = random.Random(seed + 271828)
    member_templates = rng.sample(site_train, n_templates)
    nonmember_templates = rng.sample(site_holdout, n_nonmember_templates)

    injected_train = list(site_train)
    for i, template in enumerate(member_templates):
        group = f"canary_m{i}"

        # Fix (2026-09-15, Sprint 10zz+105) — stessa correzione applicata a
        # _inject_canary_members() in chargeshield_executor.py e a
        # inject_canaries() in run_experiments.py (Sprint 10zz+96): questa
        # ricostruzione DEVE restare bit-per-bit equivalente all'iniezione
        # live del client, incluso questo fix, altrimenti i gruppi canary
        # ricostruiti qui non corrisponderebbero a quelli realmente iniettati.
        for idx, s in enumerate(injected_train):
            if s is template:
                injected_train[idx] = dict(template, _canary_group=group, _canary_role="member")
                break

        for _ in range(n_duplicates):
            clone = dict(template)
            clone["_canary_group"] = group
            clone["_canary_role"] = "member"
            injected_train.append(clone)

    injected_holdout = list(site_holdout)
    for j, template in enumerate(nonmember_templates):
        group = f"canary_n{j}"

        # Fix (2026-09-15, Sprint 10zz+108) — stesso bug del lato membro sopra
        # (Sprint 10zz+96/105), mai portato al lato non-membro qui né in
        # inject_canaries() di run_experiments.py (corretto nello stesso
        # giro, vedi lì): `template` restava in injected_holdout senza tag,
        # invisibile a _sample_preserving_canary_groups()/alle guardie
        # anti-contaminazione dello shadow, mentre il suo unico clone (qui
        # non ci sono duplicati) portava il tag — stessa classe di
        # contaminazione shadow, lato non-membro. Fix identico: sostituire
        # l'occorrenza originale con una copia taggata.
        for idx, s in enumerate(injected_holdout):
            if s is template:
                injected_holdout[idx] = dict(template, _canary_group=group, _canary_role="nonmember")
                break

        clone = dict(template)
        clone["_canary_group"] = group
        clone["_canary_role"] = "nonmember"
        injected_holdout.append(clone)

    logger.info(
        f"[CANARY] site={site_name}: ricostruiti {n_templates}x{n_duplicates} "
        f"record membro + {n_nonmember_templates} gemelli non-membro (seed={seed}) "
        "— stessa iniezione applicata dal client reale."
    )
    return injected_train, injected_holdout


def _load_client_sessions_chargeplace_scotland(
    client_args: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[int]], list[dict[str, Any]]]:
    """
    Ricostruzione offline per dataset_adapter="chargeplace_scotland" (2026-09-11)
    — mirror di chargeshield_executor.py::_load_chargeplace_scotland_sessions(),
    esteso a TUTTI e 3 i siti (l'executor ne carica solo uno per processo client
    reale; questa funzione, chiamata una volta sola offline, deve ricostruire
    l'intera campagna multi-sito per popolare cluster_membership, esattamente
    come il ramo ACN-Data sopra).

    A differenza di ACN-Data (un file per sito), i mensili ChargePlace Scotland
    coprono tutte le 32 council area insieme — si carica quindi il pool
    condiviso UNA sola volta (session_files/metadata_dir da
    client_args["chargeplace_scotland"]) e si filtra, per ciascuna identità
    NVFLARE fissa (caltech/jpl/office1), a site_id == site_mapping[identità] —
    stessa logica dell'executor, non una re-implementazione diversa.
    """
    from adapters.chargeplace_scotland_adapter import ChargePlaceScotlandDataset

    cfg = client_args.get("chargeplace_scotland")
    if not cfg:
        raise ValueError(
            "dataset_adapter='chargeplace_scotland' ma 'chargeplace_scotland' "
            "non è configurato in config_fed_client.json (servono "
            "metadata_dir/session_files/site_mapping)."
        )
    site_mapping: dict[str, str] = cfg.get("site_mapping", {})
    if not site_mapping:
        raise ValueError(
            "'chargeplace_scotland.site_mapping' è vuoto o assente — non è "
            "possibile sapere quale council area corrisponde a quale identità "
            "NVFLARE (caltech/jpl/office1)."
        )

    ds = ChargePlaceScotlandDataset()
    ds.load_with_metadata(
        session_paths=[str(PROJECT_ROOT / p) for p in cfg.get("session_files", [])],
        metadata_dir=str(PROJECT_ROOT / cfg["metadata_dir"]),
    )
    all_pool_sessions = [ds.get_sample(i) for i in range(len(ds))]

    train_sessions: list[dict[str, Any]] = []
    holdout_sessions: list[dict[str, Any]] = []
    cluster_membership: dict[str, list[int]] = {}
    canary_cfg = client_args.get("canary")

    for cluster_id, local_authority in site_mapping.items():
        site_sessions = [
            s for s in all_pool_sessions if s.get("site_id") == local_authority
        ]
        if not site_sessions:
            logger.warning(
                f"[{cluster_id}] Nessuna sessione trovata per la council area "
                f"'{local_authority}' — sito saltato."
            )
            continue
        # Stesso ordine di operazioni di _load_acn_sessions sopra:
        # enrichment prima dello split, split seed-based 80/20 per sito,
        # poi (opzionale) canary — identico al ramo ACN-Data.
        site_sessions = enrich_sessions(site_sessions)
        random.seed(seed)
        random.shuffle(site_sessions)
        split = max(1, int(len(site_sessions) * 0.8))
        site_train = site_sessions[:split]
        site_holdout = site_sessions[split:]

        site_train, site_holdout = _inject_canaries_for_site(
            site_train, site_holdout, canary_cfg, cluster_id, seed,
        )

        start = len(train_sessions)
        train_sessions.extend(site_train)
        cluster_membership[cluster_id] = list(range(start, len(train_sessions)))
        holdout_sessions.extend(site_holdout)

    logger.info(
        f"Sessioni ChargePlace Scotland ricostruite — {len(train_sessions)} train / "
        f"{len(holdout_sessions)} hold-out su {len(cluster_membership)} siti reali "
        f"(seed={seed}): " + ", ".join(f"{k}={len(v)}" for k, v in cluster_membership.items())
    )
    return train_sessions, cluster_membership, holdout_sessions


def load_client_sessions(
    client_config_path: Path,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, list[int]] | None, list[dict[str, Any]]]:
    """
    Ricostruisce ESATTAMENTE le sessioni (enriched) che i client NVFLARE
    reali hanno visto in TRAINING — leggendo "dataset_path" da
    config_fed_client.json invece di assumerlo — e, separatamente, l'insieme
    hold-out riservato da ChargeShieldExecutor._setup() e MAI passato al
    training.

    FIX 2026-08-03 (bug reale trovato al primo tentativo di eseguire questo
    script su un dump NVFLARE reale completato: ogni sito aveva già usato
    TUTTI gli anni ACN-Data scaricati per il training — nessun hold-out
    genuino esisteva su disco, e scaricarne uno aggiuntivo si è rivelato non
    praticabile, vedi chargeshield_executor.py). Invece di richiedere un file
    esterno mai visto da alcun client, questa funzione ricostruisce lo STESSO
    split seed-based 80/20 che ChargeShieldExecutor._setup() applica ora
    prima del training (stesso ordine di caricamento file, stessa
    enrichment, stesso seed) — il 20% escluso dal training è per costruzione
    lo stesso hold-out che il client reale ha riservato, senza bisogno di
    alcun dato esterno. Richiede che il dump NVFLARE analizzato provenga da
    un job lanciato DOPO questo fix — un dump del vecchio comportamento
    (100% training, nessun hold-out riservato) non ha un hold-out genuino da
    ricostruire, qualunque split si applichi qui sarebbe usato anche nel
    training reale (data leakage silenzioso).

    FIX 2026-07-22 (review indipendente): la prima stesura di questo script
    caricava le sessioni via load_sessions(cfg) da config/experiment.yaml
    (che combina jpl_2019+jpl_2020) E le shuffle-ava prima dello split
    train/hold-out — un dataset e un ordinamento DIVERSI da quelli che
    chargeshield_executor.py::_setup() usa davvero (SOLO il file indicato
    da "dataset_path" in config_fed_client.json, MAI shuffled, split
    contiguo per indice — vedi quella funzione). run_lira()/run_yeom()
    ricostruiscono l'appartenenza ai cluster assumendo che train_sessions
    sia nello STESSO ordine/split usato dai client reali (stesso principio
    di run_fl_rounds() nella simulazione) — con un dataset/ordine diverso,
    la ground truth membership (chi ha allenato su cosa) sarebbe scorrelata
    da quella reale, rendendo gli AUC di LiRA/Shadow/Yeom non significativi
    SENZA generare alcun errore visibile (fallimento silenzioso, il più
    pericoloso). Fix: legge lo stesso "dataset_path" dal config del client
    reale e applica la stessa pipeline (ACNDataset → enrich). Nota
    (aggiornata 2026-08-03): all'epoca di questo fix non c'era ancora alcuno
    shuffle in nessuno dei due percorsi — lo split seed-based 80/20 introdotto
    sopra è arrivato dopo, e ha reso questa frase storica ("NESSUNO shuffle")
    non più accurata: oggi lo shuffle c'è, deliberatamente, in entrambi i
    file, con lo stesso seed.

    FIX 2026-07-22 (review indipendente pre-push, 3 siti reali): questa
    funzione assumeva ancora "dataset_path" = UN singolo file JSON condiviso
    da tutti i client (design pre-2026-07-22). Da quando
    chargeshield_executor.py carica invece TUTTI i file sotto la directory
    PADRE "datasets/acn/<cluster_id>/" (un cluster_id per sito reale), questa
    funzione andava in `IsADirectoryError` aprendo quella directory come se
    fosse un file — mai eseguita/testata da nessuno nel frattempo (nvflare/
    torch non disponibili in questo sandbox), quindi il bug non era stato
    notato. Fix: se "dataset_path" è una directory, la tratta come la radice
    multi-sito (stessa struttura di chargeshield_executor.py) e carica OGNI
    sottodirectory sito trovata (non solo quella del cluster_id di default
    nel config — quel valore è comunque solo un fallback per-sito, mai letto
    a runtime da un vero deployment NVFLARE, vedi commento in
    config_fed_client.json), costruendo anche un `cluster_membership`
    (site_name -> lista indici) — stesso schema di
    scripts/run_experiments.py::group_indices_by_site() — così run_lira()
    può raggruppare per sito reale invece di ricadere silenziosamente sul
    fallback a 4 fette fittizie (comportamento di default se
    cluster_membership=None). Se invece "dataset_path" è ancora un file
    singolo (design pre-2026-07-22, o un dump storico), il comportamento
    originale è preservato invariato e restituisce cluster_membership=None.
    """
    import json as _json

    with open(client_config_path) as f:
        client_cfg = _json.load(f)
    client_args = client_cfg["executors"][0]["executor"]["args"]

    # FIX 2026-09-11 (bug reale trovato durante un audit richiesto
    # dall'utente, non da un run fallito): questa funzione assumeva sempre
    # "dataset_path" (l'adapter ACN) — un dump NVFLARE reale prodotto con
    # dataset_adapter="chargeplace_scotland" (task #37/#93,
    # chargeshield_executor.py::_load_chargeplace_scotland_sessions(), quel
    # config non ha affatto la chiave "dataset_path") sollevava un
    # KeyError non gestito qui sotto, prima ancora di provare a caricare
    # qualunque sessione — la rianalisi MIA offline era quindi
    # completamente inutilizzabile per quell'adapter. Fix: stesso dispatch
    # già presente lato executor, mirror di
    # scripts/run_experiments.py::load_sessions().
    if client_args.get("dataset_adapter") == "chargeplace_scotland":
        return _load_client_sessions_chargeplace_scotland(client_args, seed)

    dataset_path_str = client_args["dataset_path"]
    dataset_path = PROJECT_ROOT / dataset_path_str

    if dataset_path.is_dir():
        site_dirs = sorted(p for p in dataset_path.iterdir() if p.is_dir())
        if not site_dirs:
            raise ValueError(
                f"'dataset_path' ({dataset_path}) è una directory ma non contiene "
                "nessuna sottodirectory sito (es. caltech/jpl/office1) — non è "
                "possibile ricostruire le sessioni dei client reali."
            )
        train_sessions: list[dict[str, Any]] = []
        holdout_sessions: list[dict[str, Any]] = []
        cluster_membership: dict[str, list[int]] = {}
        for site_dir in site_dirs:
            json_files = sorted(site_dir.glob("*.json"))
            if not json_files:
                logger.warning(f"Nessun file .json in {site_dir} — sito saltato.")
                continue
            site_sessions: list[dict[str, Any]] = []
            for jf in json_files:
                ds = ACNDataset()
                ds.load(str(jf))
                site_sessions.extend(ds.get_sample(i) for i in range(len(ds)))
            # Enrichment PRIMA dello split, per-sito — stesso ordine di
            # operazioni di chargeshield_executor.py::_setup() (2026-08-03).
            site_sessions = enrich_sessions(site_sessions)
            # FIX 2026-08-03: stesso split seed-based 80/20 di
            # ChargeShieldExecutor._setup() — vedi docstring della funzione.
            # Applicato PER SITO con lo stesso seed, perché ogni client reale
            # fa lo split indipendentemente sulle proprie sole sessioni.
            random.seed(seed)
            random.shuffle(site_sessions)
            split = max(1, int(len(site_sessions) * 0.8))
            site_train = site_sessions[:split]
            site_holdout = site_sessions[split:]

            # Canary positive control (2026-09-11) — ricostruisce offline la
            # stessa iniezione che chargeshield_executor.py applica dal vivo
            # lato training, leggendo lo stesso blocco "canary" da questo
            # config del client reale. No-op se non configurato/disabilitato
            # o se questo non è il sito bersaglio — vedi _inject_canaries_for_site().
            canary_cfg = client_cfg["executors"][0]["executor"]["args"].get("canary")
            site_train, site_holdout = _inject_canaries_for_site(
                site_train, site_holdout, canary_cfg, site_dir.name, seed,
            )

            start = len(train_sessions)
            train_sessions.extend(site_train)
            cluster_membership[site_dir.name] = list(range(start, len(train_sessions)))
            holdout_sessions.extend(site_holdout)

        logger.info(
            f"Sessioni multi-sito ricostruite da {dataset_path} — "
            f"{len(train_sessions)} train / {len(holdout_sessions)} hold-out "
            f"su {len(cluster_membership)} siti reali (seed={seed}): "
            + ", ".join(f"{k}={len(v)}" for k, v in cluster_membership.items())
        )
        return train_sessions, cluster_membership, holdout_sessions

    ds = ACNDataset()
    ds.load(str(dataset_path))
    sessions = [ds.get_sample(i) for i in range(len(ds))]
    sessions = enrich_sessions(sessions)
    logger.info(
        f"Sessioni client ricostruite da {dataset_path.name} (stesso file, stesso "
        f"ordine, nessuno shuffle — design a file singolo pre-2026-07-22, hold-out "
        f"da passare esplicitamente via --holdout-dataset): {len(sessions)}"
    )
    return sessions, None, []


def load_holdout_sessions(holdout_dataset_path: Path) -> list[dict[str, Any]]:
    """
    Sessioni hold-out (non-member) per MIA — devono provenire da un file MAI
    visto da nessun client NVFLARE reale (non da uno split random dello STESSO
    file, che qui non avrebbe senso: load_client_sessions() restituisce l'intero
    dataset del client, senza nessuna porzione riservata — a differenza della
    simulazione, dove main() fa un 80/20 split PRIMA di passare i dati a
    run_fl_rounds(), qui TUTTE le sessioni del file client vengono usate per
    il training reale, quindi un vero hold-out deve venire da un anno/file
    diverso, mai toccato da alcun client — es. acndata_sessions_2020.json
    quando i client hanno usato acndata_sessions_2019.json).
    """
    ds = ACNDataset()
    ds.load(str(holdout_dataset_path))
    sessions = [ds.get_sample(i) for i in range(len(ds))]
    sessions = enrich_sessions(sessions)
    logger.info(f"Sessioni hold-out da {holdout_dataset_path.name}: {len(sessions)}")
    return sessions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ChargeShield-FL — esegue LiRA/Shadow/Yeom/IDS su un dump prodotto "
            "da un vero job NVFLARE (fase 5), riusando run_experiments.py invariato."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config/experiment.yaml"))
    parser.add_argument(
        "--fl-results", type=Path,
        default=None,
        help=(
            "Path al dump pickle prodotto da ChargeShieldAggregator (fase 5). "
            "Se omesso, usa il file 'experiments/nvflare_fl_results_*.pkl' più "
            "recente (fix 2026-07-24: da quando ogni run NVFLARE genera un "
            "nome univoco con timestamp — vedi chargeshield_aggregator.py — "
            "non esiste più un unico 'nvflare_fl_results.pkl' fisso da usare "
            "come default)."
        ),
    )
    parser.add_argument(
        "--sweep-dir", type=Path, default=None,
        help="Directory di output (come in run_experiments.py --sweep-dir).",
    )
    parser.add_argument(
        "--client-config", type=Path,
        default=Path("nvflare/jobs/chargeshield_poc/app/config/config_fed_client.json"),
        help=(
            "config_fed_client.json del job NVFLARE reale — usato per leggere "
            "'dataset_path' e ricostruire ESATTAMENTE le sessioni viste dai "
            "client (stesso file, nessuno shuffle, vedi load_client_sessions())."
        ),
    )
    parser.add_argument(
        "--holdout-dataset", type=Path, default=None,
        help=(
            "Override esplicito del pool non-member per LiRA/Shadow/Yeom. "
            "Di default (2026-08-03) NON serve passarlo: load_client_sessions() "
            "ricostruisce da sola l'hold-out 80/20 che ChargeShieldExecutor "
            "riserva ora prima del training (stesso seed, stesso ordine di "
            "caricamento). Passa questo argomento solo se vuoi usare un file "
            "diverso (es. un anno genuinamente mai scaricato)."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-shadow", type=int, default=None)
    parser.add_argument("--shadow-epochs-cap", type=int, default=None)
    parser.add_argument(
        "--include-fedmia-gradient", action="store_true",
        help=(
            "Sprint 10zz+71 (2026-09-14): aggiunge FedMIAGradientAttack alla "
            "rianalisi NVFLARE, stesso meccanismo extra_attacks di "
            "run_experiments.py::main() (vedi --include-fedmia-gradient lì e "
            "run_registered_attacks()). RISTRETTO a dp_mode 'central'/'local' "
            "(scelta esplicita dell'utente) — con dp_mode='dp-fedavg' (letto "
            "dal dump NVFLARE stesso, non da questo flag) solleva un errore "
            "esplicito invece di eseguire silenziosamente un attacco su un "
            "placement mai validato in questo contesto. Raddoppia il costo "
            "computazionale (richiama run_lira() una seconda volta) — non "
            "ancora eseguito con torch reale in questo script, stesso stato "
            "'non testato' del resto del file (vedi docstring in testa)."
        ),
    )
    parser.add_argument(
        "--fedmia-gradient-normalize", action="store_true",
        help=(
            "Passato a FedMIAGradientAttack se --include-fedmia-gradient è "
            "attivo (vedi run_fedmia_gradient() in run_experiments.py per il "
            "razionale) — no-op altrimenti."
        ),
    )
    return parser.parse_args()


def _round_count_or_reason(path: Path) -> str:
    """Legge un dump pickle NVFLARE e ritorna il numero di round contenuti
    (o il motivo per cui non è stato possibile leggerlo) — usato solo per
    logging diagnostico, non cambia quale file viene selezionato."""
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return f"{len(payload)} round"
    except Exception as e:  # file corrotto/troncato/formato inatteso
        return f"illeggibile ({e.__class__.__name__})"


def _resolve_fl_results_path(explicit: Path | None) -> Path:
    """Se --fl-results non è passato esplicitamente, sceglie il dump pickle
    NVFLARE più recente per data di modifica — fix 2026-07-24: da quando
    ChargeShieldAggregator inserisce un timestamp univoco nel nome (per non
    sovrascrivere i risultati di run precedenti, vedi il suo __init__), non
    esiste più un singolo 'nvflare_fl_results.pkl' fisso da usare come
    default.

    Fix 2026-07-24 (review indipendente, stesso giorno): la sola mtime NON
    distingue un run completato da uno interrotto a metà — ChargeShieldAggregator
    riscrive lo stesso file dopo OGNI round, quindi un run avviato dopo uno
    completato ma interrotto al round 2 ha comunque una mtime più recente, e
    verrebbe scelto in silenzio al posto del run completo precedente. Non
    risolto scegliendo un file diverso (quale euristica sostituire a "il più
    recente" è una scelta che cambia il comportamento di default, non ovvia
    da prendere alla cieca) — mitigato invece rendendo sempre visibile nel log
    il numero di round di ogni candidato, cosicché un run interrotto sia
    immediatamente riconoscibile prima di fidarsi dei risultati a valle."""
    if explicit is not None:
        return explicit
    candidates = sorted(
        Path("experiments").glob("nvflare_fl_results_*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "Nessun 'experiments/nvflare_fl_results_*.pkl' trovato — esegui prima "
            "'make nvflare-sim-smoke'/'make nvflare-sim', oppure passa --fl-results "
            "esplicitamente."
        )
    chosen = candidates[0]
    logger.info(
        f"Dump NVFLARE selezionato (più recente per mtime): {chosen.name} — "
        f"{_round_count_or_reason(chosen)}. Verifica che il conteggio round sia "
        "quello atteso prima di usare questi risultati — un run interrotto ha "
        "comunque mtime più recente di uno completato prima."
    )
    if len(candidates) > 1:
        others = ", ".join(
            f"{p.name} ({_round_count_or_reason(p)})" for p in candidates[1:]
        )
        logger.warning(
            f"Trovati {len(candidates)} dump NVFLARE — uso il più recente per mtime "
            f"(NON necessariamente il più completo): {chosen.name}. Altri: {others}"
        )
    return chosen


def main() -> None:
    args = parse_args()
    args.fl_results = _resolve_fl_results_path(args.fl_results)

    logger.info("=" * 60)
    logger.info("ChargeShield-FL — Analisi post-hoc su dump NVFLARE (fase 5)")
    logger.info(f"Dump NVFLARE: {args.fl_results}")
    logger.info("=" * 60)

    cfg = load_config(args.config, {"epsilon": None, "rounds": None})
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed

    fl_results, meta = load_nvflare_fl_results(args.fl_results)

    # dp_mode effettivamente usato dal job NVFLARE (registrato nel dump da
    # ChargeShieldAggregator) — sovrascrive quello di config/experiment.yaml,
    # che descrive la simulazione single-process, non necessariamente lo
    # stesso run NVFLARE che ha prodotto questo dump.
    dp_mode = meta.get("dp_mode", cfg["experiment"].get("dp_mode", "dp-fedavg"))
    cfg["experiment"]["dp_mode"] = dp_mode

    # Sprint 10zz+71 (2026-09-14): --include-fedmia-gradient su NVFLARE è
    # ristretto a dp_mode 'central'/'local', per scelta esplicita dell'utente
    # (dp-fedavg escluso dallo scope di questa promozione — nessuna analisi
    # empirica ancora fatta su quel placement con questo attacco). dp_mode qui
    # viene dal dump reale (meta, sopra), non da un argomento scelto a mano —
    # fail-fast invece di eseguire silenziosamente un attacco fuori scope.
    if args.include_fedmia_gradient and dp_mode not in ("central", "local"):
        raise ValueError(
            f"--include-fedmia-gradient su NVFLARE è supportato solo per "
            f"dp_mode in {{'central', 'local'}} (scelta esplicita "
            f"dell'utente, Sprint 10zz+71) — il dump {args.fl_results.name} "
            f"ha dp_mode={dp_mode!r}. Rilanciare senza --include-fedmia-"
            "gradient per questo dump, o usare un dump central/local."
        )
    cfg["experiment"]["name"] = cfg["experiment"]["name"] + "_nvflare"
    if meta.get("epsilon") is not None:
        cfg["experiment"]["epsilon"] = meta["epsilon"]
    if meta.get("delta") is not None:
        cfg["experiment"]["delta"] = meta["delta"]
    if meta.get("max_grad_norm") is not None:
        cfg["experiment"]["max_grad_norm"] = meta["max_grad_norm"]

    # LIMITE NOTO (vedi docstring modulo): nessun bypass --no-dp lato NVFLARE.
    no_dp = False

    # ── Dataset: ricostruisce ESATTAMENTE cosa hanno visto i client reali ───
    # FIX 2026-07-22 (review indipendente — vedi load_client_sessions() per il
    # dettaglio del bug originale): NON riusa load_sessions(cfg)/shuffle come
    # nella simulazione — quella pipeline combina un dataset diverso (2019+2020,
    # da config/experiment.yaml) e mischia l'ordine, disallineando la ground
    # truth membership da quella reale. Legge invece "dataset_path" da
    # --client-config (config_fed_client.json del job reale).
    seed = cfg["experiment"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # FIX 2026-08-04 (review indipendente): il seed per RICOSTRUIRE lo split
    # 80/20 train/hold-out deve essere quello che ChargeShieldExecutor._setup()
    # ha realmente usato — letto da --client-config (l'unico che il client reale
    # legge, via "args"."seed" in config_fed_client.json) — non quello di
    # config/experiment.yaml o --seed (che qui governano solo la randomness del
    # resto di questa analisi offline: init degli shadow model, ecc.). Prima di
    # questo fix i due valori coincidevano solo per coincidenza (entrambi 42):
    # se in futuro cambia l'uno senza l'altro, la ricostruzione del train/hold-out
    # diverge silenziosamente da quella reale — la stessa classe di bug "nessun
    # errore visibile" che load_client_sessions() già documenta di voler evitare.
    import json as _json
    with open(args.client_config) as _f:
        _client_split_seed = (
            _json.load(_f)["executors"][0]["executor"]["args"].get("seed", 42)
        )
    if _client_split_seed != seed:
        logger.warning(
            f"Seed di config/experiment.yaml (o --seed, ora {seed}) diverso dal "
            f"seed reale del client NVFLARE in {args.client_config} "
            f"({_client_split_seed}) — uso quest'ultimo per ricostruire lo split "
            "80/20 train/hold-out (deve combaciare con quanto il client ha "
            "realmente fatto), l'altro resta usato per il resto di questa analisi."
        )

    train_sessions, cluster_membership, holdout_sessions = load_client_sessions(
        args.client_config, seed=_client_split_seed,
    )

    if args.holdout_dataset is not None:
        # Override esplicito: ignora l'hold-out ricostruito sopra e usa un
        # file passato a mano (utile per confrontare contro un anno/periodo
        # genuinamente esterno, se in futuro ne esiste uno disponibile).
        logger.warning(
            "--holdout-dataset esplicito passato — ignoro l'hold-out 80/20 "
            "ricostruito da load_client_sessions() e uso questo file al suo posto."
        )
        holdout_sessions = load_holdout_sessions(args.holdout_dataset)
    elif not holdout_sessions:
        # Design a file singolo (pre-2026-07-22) o dump precedente al fix
        # 2026-08-03: nessun hold-out ricostruibile automaticamente da
        # load_client_sessions() — serve un file esplicito, non lo si deduce
        # più "alla cieca" (vedi FIX 2026-08-03 nel docstring di
        # load_client_sessions() per il perché).
        raise ValueError(
            "Nessun hold-out disponibile: load_client_sessions() non ne ha "
            "ricostruito uno (design a file singolo, o dump di un job NVFLARE "
            "precedente al fix 2026-08-03 che allenava sul 100% dei dati). "
            "Specifica --holdout-dataset esplicitamente con un file mai usato "
            "per il training di nessun client."
        )

    logger.info(
        f"Train (client reali): {len(train_sessions)} sessioni — "
        f"Hold-out (non-member): {len(holdout_sessions)} sessioni"
    )

    _FEATURES = AutoencoderTrainer.CONTINUOUS_FEATURES
    # Stats calcolate su train_sessions (le sessioni realmente usate per il
    # training), NON su holdout — stessa regola anti-leakage di main() nella
    # simulazione.
    #
    # ATTENZIONE — discrepanza nota, non "corretta" alla cieca (2026-07-24,
    # review indipendente round 3): questa riga calcola UNA sola feature_stats
    # globale sull'intero pool multi-sito combinato. Il commento precedente qui
    # affermava che questo fosse "equivalente" a chargeshield_executor.py::
    # _setup() — non è più vero nel design attuale (post-migrazione ai 3 siti
    # reali, Sprint 10): lì ogni client calcola le proprie stats SOLO sulle
    # sessioni del proprio sito (vedi il commento a chargeshield_executor.py
    # righe 364-373, che riconosce esplicitamente che min/max variano tra siti —
    # es. kWh massimo osservato a Caltech vs JPL — invece di un'unica scala
    # globale condivisa come nella simulazione single-process).
    #
    # Non correggiamo qui ricalcolando le stats per-sito "alla cieca": farlo
    # richiederebbe anche decidere con quali stats normalizzare holdout_sessions
    # (che non appartiene in modo univoco a nessun sito di training), una scelta
    # metodologica che non possiamo validare senza poter eseguire davvero
    # NVFLARE in questo sandbox (torch/nvflare non installabili). Finché questo
    # script resta "non eseguito, non testato" (vedi docstring in testa al
    # file), qualunque numero prodotto da questa pipeline offline contro un vero
    # dump NVFLARE va trattato come indicativo, non come replica esatta della
    # normalizzazione realmente vista da ciascun client in training.
    # limiti pubblici come la simulazione (2026-10-16)
    feature_stats, _clip, _ = statistiche_normalizzazione(cfg, train_sessions, _FEATURES)
    train_sessions = normalize_sessions(train_sessions, feature_stats, _FEATURES, clip=_clip)
    holdout_sessions = normalize_sessions(holdout_sessions, feature_stats, _FEATURES, clip=_clip)

    # ── IDS — stessa funzione, stessa logica, usata dalla simulazione ───────
    ids_results: dict = {}
    try:
        ids_results = run_ids(cfg, fl_results, no_dp=no_dp)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"run_ids() fallita: {exc}", exc_info=True)

    # ── Yeom + Shadow + LiRA — tramite il registro pluggable ────────────────
    # Fix 2026-07-24: questo blocco chiamava run_yeom()/run_shadow()/
    # run_lira() direttamente, duplicando (quasi identica) la stessa logica di
    # dispatch/merge/error-handling di scripts/run_experiments.py::main().
    # Ora entrambi i punti chiamano run_registered_attacks() (definita in
    # run_experiments.py, itera src/plugins/attacks/ATTACK_REGISTRY) — un solo
    # punto di verità invece di due copie che potevano divergere. Stesso
    # comportamento di prima: yeom, poi shadow, poi lira, merge per round,
    # un attacco fallito non blocca gli altri né il salvataggio finale.
    n_shadow = args.n_shadow if args.n_shadow is not None else cfg.get("lira", {}).get("n_shadow", 8)
    # Sprint 10zz+71 (2026-09-14): stesso meccanismo extra_attacks di
    # run_experiments.py::main() — vedi il commento lì per il razionale
    # completo. ATTACK_REGISTRY resta invariato (solo Yeom/Shadow/LiRA); il
    # guard su dp_mode sopra ha già bloccato il caso dp-fedavg prima di
    # arrivare qui.
    _extra_attacks = None
    if args.include_fedmia_gradient:
        from plugins.attacks.fedmia_gradient import FedMIAGradientAttack
        _extra_attacks = {FedMIAGradientAttack.name: FedMIAGradientAttack}
    mia_results = run_registered_attacks(
        cfg, train_sessions, holdout_sessions, fl_results,
        extra_attacks=_extra_attacks,
        n_shadow=n_shadow, shadow_epochs_cap=args.shadow_epochs_cap,
        no_dp=no_dp, dp_mode=dp_mode, cluster_membership=cluster_membership,
        # Sprint 10zz+71: letto solo da FedMIAGradientAttack.run(), ignorato
        # dagli altri wrapper tramite il proprio **kwargs — stesso pattern di
        # run_experiments.py::main().
        fedmia_gradient_normalize=args.fedmia_gradient_normalize,
    )

    result_file = save_results(cfg, mia_results, ids_results, fl_results=fl_results, sweep_dir=args.sweep_dir)
    logger.info(f"Analisi NVFLARE completata — risultati in {result_file}")


if __name__ == "__main__":
    main()
