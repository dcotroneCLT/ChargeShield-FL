"""
LiRA e RMIA, versione 2 (2026-10-16): revisione profonda, Segnalazioni R,
voci 65-73.

Perche' una versione nuova invece di correggere `run_lira()`: la versione 1
(scripts/run_experiments.py) ha accumulato in tre mesi decine di correzioni e
opzioni diagnostiche intrecciate; i difetti trovati il 16 ottobre stanno nella
costruzione degli shadow e nella stima della varianza, cioe' al centro della
funzione. Qui c'e' solo il metodo, scritto da capo e coperto da test:
`run_lira()` resta per riprodurre le run vecchie (`lira.version: 1`).

Metodo (Carlini et al., "Membership Inference Attacks From First Principles",
IEEE S&P 2022; RMIA: Zarifzadeh et al., "Low-Cost High-Power Membership
Inference Attacks", ICML 2024), adattato all'osservatore federato:

1. Shadow. Per ogni client c e ogni k = 1..n_shadow uno shadow con un proprio
   `AutoencoderTrainer`, PERSISTENTE fra i round come il client reale: stesso
   codice di training (`train_local`), quindi stesso FedProx, stesso stato di
   Adam fra i round, stessa normalizzazione, stessa DP per record o per
   soggetto. Dal round 2 lo shadow riparte dal modello globale del round
   precedente (`set_weights`, come `apply_global_model` del client), al round
   1 da un'inizializzazione propria o, con `ml.common_init`, da quella comune.
   L'insieme IN ha la stessa taglia dei dati del client (`lira.in_size:
   client`), estratto dall'universo membri + non membri del client.
2. Osservazione. Lo shadow subisce la stessa trasformazione dell'update che il
   server osserva per il bersaglio: update grezzo senza DP per client e con
   dp-fedavg (LiRA attacca `raw_updates`), update tagliato e rumorizzato con
   local, solo tagliato con central. Per la superficie "global" lo shadow
   prende il posto del client c nell'aggregazione del round, insieme agli
   update osservati degli altri client, con lo stesso peso del client e, in
   central, con il rumore sull'aggregato.
3. Statistica. Log della MSE per campione (piu' vicina a una gaussiana della
   MSE). Medie IN e OUT per campione dagli shadow; varianza FISSA per client e
   round (media delle varianze per campione, Carlini et al. sez. VI-B), al
   posto delle soglie della versione 1, che valevano quasi sempre
   (floor_hit circa 0,997). Punteggio = log N(t; mu_in, s_in) - log N(t;
   mu_out, s_out). Si valutano solo i campioni con almeno uno shadow IN e uno
   OUT; il loro numero e quelli scartati vanno nel JSON.
4. RMIA online: Pr(x|theta) = exp(-mse/tau), tau = mediana della MSE del
   bersaglio sulla popolazione z del client e del round; Pr(x) = media sugli
   shadow; punteggio = quota della popolazione z con LR(x)/LR(z) > gamma. La
   popolazione z e' un sottoinsieme dei non membri del client riservato e mai
   valutato, scelto senza guardare le etichette. Stessi campioni di LiRA.
5. Composizione sui round: somma dei log-rapporti dello stesso campione sui
   round in cui e' valutato in tutti (LiRA e RMIA).

Parallelismo: ogni traiettoria di shadow (client, k) e' indipendente e
deterministica dato il seed; con `workers > 1` gira in processi separati, un
thread ciascuno, e i numeri coincidono con l'esecuzione sequenziale.
"""

from __future__ import annotations

import logging
import math
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

LOG_EPS = 1e-12
_SEED_CLUSTER = 104729
_SEED_SHADOW = 31337
_SEED_ROUND = 1_000_003


# ── Funzioni pure (testate in tests/test_lira2.py) ─────────────────────────────

def statistica(mse: np.ndarray) -> np.ndarray:
    """Statistica di LiRA: log della MSE per campione."""
    return np.log(np.asarray(mse, dtype=float) + LOG_EPS)


def stime_in_out(
    stat_shadow: np.ndarray, in_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Medie IN/OUT per campione e varianza fissa.

    stat_shadow, in_mask: forma (n_shadow, n_campioni).
    Restituisce mu_in, mu_out (nan dove mancano shadow), n_in, n_out per
    campione, e le due deviazioni standard fisse: radice della media, sui
    campioni con almeno due shadow di quel lato, della varianza campionaria
    (ddof = 1). Senza alcun campione con due shadow di un lato, la deviazione
    di quel lato e' quella dell'altro lato; senza nessuno dei due, nan.
    """
    s = np.asarray(stat_shadow, dtype=float)
    m = np.asarray(in_mask, dtype=bool)
    n_in = m.sum(axis=0)
    n_out = (~m).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_in = np.where(n_in > 0, np.where(m, s, 0.0).sum(axis=0) / np.maximum(n_in, 1), np.nan)
        mu_out = np.where(n_out > 0, np.where(~m, s, 0.0).sum(axis=0) / np.maximum(n_out, 1), np.nan)
        dev_in = np.where(m, s - mu_in, 0.0)
        dev_out = np.where(~m, s - mu_out, 0.0)
        var_in = (dev_in ** 2).sum(axis=0) / np.maximum(n_in - 1, 1)
        var_out = (dev_out ** 2).sum(axis=0) / np.maximum(n_out - 1, 1)
    vi = var_in[n_in >= 2]
    vo = var_out[n_out >= 2]
    s_in = float(np.sqrt(vi.mean())) if vi.size else float("nan")
    s_out = float(np.sqrt(vo.mean())) if vo.size else float("nan")
    if not np.isfinite(s_in):
        s_in = s_out
    if not np.isfinite(s_out):
        s_out = s_in
    return mu_in, mu_out, n_in, n_out, s_in, s_out


def _log_norm(x: np.ndarray, mu: np.ndarray, s: float) -> np.ndarray:
    return -0.5 * ((x - mu) / s) ** 2 - math.log(s) - 0.5 * math.log(2 * math.pi)


def punteggio_lira(
    t: np.ndarray, stat_shadow: np.ndarray, in_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Log-rapporto di verosimiglianza di LiRA online con varianza fissa.

    t: statistica del bersaglio per campione. Restituisce il punteggio (nan
    per i campioni senza almeno uno shadow IN e uno OUT) e la diagnostica.
    """
    mu_in, mu_out, n_in, n_out, s_in, s_out = stime_in_out(stat_shadow, in_mask)
    t = np.asarray(t, dtype=float)
    ok = (n_in >= 1) & (n_out >= 1) & np.isfinite(t)
    score = np.full(t.shape, np.nan)
    if ok.any() and np.isfinite(s_in) and np.isfinite(s_out) and s_in > 0 and s_out > 0:
        score[ok] = _log_norm(t[ok], mu_in[ok], s_in) - _log_norm(t[ok], mu_out[ok], s_out)
    diag = {
        "sigma_in": s_in, "sigma_out": s_out,
        "n_valutati": int(np.isfinite(score).sum()),
        "n_senza_in": int((n_in == 0).sum()), "n_senza_out": int((n_out == 0).sum()),
        "mu_in_meno_out": float(np.nanmean(mu_in[ok] - mu_out[ok])) if ok.any() else None,
    }
    return score, diag


def _logmeanexp(a: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.mean(np.exp(a - m), axis=axis))


def log_lr_rmia(mse_target: np.ndarray, mse_shadow: np.ndarray, tau: float) -> np.ndarray:
    """log Pr(x|theta) - log Pr(x), con Pr(x|theta) = exp(-mse/tau) e Pr(x)
    media sugli shadow (RMIA online). mse_shadow: (n_shadow, n)."""
    tau = max(float(tau), LOG_EPS)
    lp_t = -np.asarray(mse_target, dtype=float) / tau
    lp_ref = _logmeanexp(-np.asarray(mse_shadow, dtype=float) / tau, axis=0)
    return lp_t - lp_ref


def punteggio_rmia(llr_x: np.ndarray, llr_z: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Quota della popolazione z con LR(x)/LR(z) > gamma (Zarifzadeh et al.,
    eq. 4). llr_z: popolazione riservata, mai fra i campioni valutati."""
    z = np.sort(np.asarray(llr_z, dtype=float)[np.isfinite(llr_z)])
    x = np.asarray(llr_x, dtype=float)
    out = np.full(x.shape, np.nan)
    if z.size == 0:
        return out
    ok = np.isfinite(x)
    # LR(x)/LR(z) > gamma  <=>  llr_z < llr_x - log(gamma)
    out[ok] = np.searchsorted(z, x[ok] - math.log(gamma), side="left") / z.size
    return out


# ── Traiettoria di uno shadow (eseguibile in un processo separato) ─────────────

@dataclass
class SpecShadow:
    """Tutto cio' che serve a una traiettoria; serializzabile per i processi."""
    cid: str
    k: int
    seed: int
    rounds: list[int]
    in_sessions: list[dict[str, Any]]
    eval_x: np.ndarray                      # (n_eval, n_feature), gia' normalizzato
    ml_cfg: dict[str, Any]
    arch: dict[str, Any]
    input_dim: int
    gm_cfg: dict[str, Any]
    no_dp: bool
    dp_mode: str
    dp_ablation: str
    init_weights: list[Any] | None          # common_init: pesi iniziali comuni
    global_prev: dict[int, list[Any]]        # round r -> pesi globali del round r-1 (r >= 2)
    subject_weights: dict[str, float] | None
    superficie: str                          # "client" o "global"
    altri_update: dict[int, list[Any]] = field(default_factory=dict)   # solo "global"; None = posto dello shadow
    n_samples_client: dict[int, int] = field(default_factory=dict)     # solo "global"
    n_partecipanti: int = 1


def _modello(arch: dict[str, Any], input_dim: int, weights: list[Any]):
    from core.autoencoder import Autoencoder
    m = Autoencoder(input_dim=input_dim, **arch)
    sd = m.state_dict()
    keys = list(sd.keys())
    if len(keys) != len(weights):
        raise RuntimeError(f"pesi: {len(weights)} tensori, modello: {len(keys)}")
    m.load_state_dict({
        k: (w if isinstance(w, torch.Tensor) else torch.tensor(w)).to(sd[k].dtype).reshape(sd[k].shape)
        for k, w in zip(keys, weights)
    }, strict=True)
    m.eval()
    return m


def mse_per_campione(arch: dict[str, Any], input_dim: int, weights: list[Any],
                     x: np.ndarray) -> np.ndarray:
    """MSE di ricostruzione per campione, modello in eval. Stessa funzione per
    bersaglio e shadow."""
    m = _modello(arch, input_dim, weights)
    with torch.no_grad():
        xt = torch.as_tensor(x, dtype=torch.float32)
        r = m(xt)
        return ((r - xt) ** 2).mean(dim=1).double().numpy()


def _seme_round(seed: int, r: int, extra: int = 0) -> int:
    return (seed + _SEED_ROUND * r + 7 * extra) % (2 ** 31 - 1)


def traiettoria_shadow(spec: SpecShadow) -> dict[int, np.ndarray]:
    """Allena uno shadow su tutti i round e restituisce la MSE per campione del
    modello osservato a ogni round (superficie scelta)."""
    from ml.autoencoder_trainer import AutoencoderTrainer
    from ml.base_ml import GradientUpdate
    from ml.fedavg_aggregator import FedAvgAggregator
    from ml.gradient_manager import GradientManager

    torch.set_num_threads(1)
    torch.manual_seed(spec.seed)
    tr = AutoencoderTrainer(
        config={**spec.ml_cfg, "seed": spec.seed},
        node_id=f"shadow-{spec.cid}-{spec.k}", cluster_id=spec.cid,
    )
    if spec.subject_weights is not None:
        tr.set_subject_weights(spec.subject_weights)
    if spec.init_weights is not None:
        tr.set_weights(spec.init_weights)
        tr._global_weights = None          # come run_fl_rounds con common_init
    keys = tr.get_weight_keys()
    gm = GradientManager(spec.gm_cfg)
    privato_server = (not spec.no_dp)
    out: dict[int, np.ndarray] = {}
    for r in spec.rounds:
        if r > 1:
            tr.set_weights(spec.global_prev[r])  # = apply_global_model del client
        pre = tr.get_weights()
        torch.manual_seed(_seme_round(spec.seed, r))
        upd = tr.train_local(spec.in_sessions, r)
        # update che entra nell'aggregazione (come run_fl_rounds)
        torch.manual_seed(_seme_round(spec.seed, r, 1))
        if not privato_server:
            aggregabile = upd
        elif spec.dp_mode == "central":
            aggregabile = gm.clip_only(upd, weight_keys=keys, reference_weights=pre)
        elif spec.dp_ablation == "clip-only":       # come _privatize_per_client
            aggregabile = gm.clip_no_noise(upd, weight_keys=keys, reference_weights=pre)
        elif spec.dp_ablation == "noise-only":
            aggregabile = gm.noise_no_clip(upd, weight_keys=keys)
        else:
            aggregabile = gm.privatize(upd, weight_keys=keys, reference_weights=pre)
        if spec.superficie == "client":
            # Il server osserva il grezzo senza DP e con dp-fedavg (LiRA attacca
            # raw_updates), il privatizzato con local, il tagliato con central.
            osservato = upd if (not privato_server or spec.dp_mode == "dp-fedavg") else aggregabile
            pesi = osservato.weights
        else:
            agg = FedAvgAggregator({"min_participants": spec.n_partecipanti})
            proprio = GradientUpdate(
                node_id=f"shadow-{spec.cid}-{spec.k}", cluster_id=spec.cid, round_num=r,
                weights=aggregabile.weights, gradients=None, loss=aggregabile.loss,
                n_samples=spec.n_samples_client.get(r, aggregabile.n_samples), metadata={},
            )
            # stesso ordine di raccolta del round reale: lo shadow al posto del client
            for u in spec.altri_update.get(r, []):
                agg.collect(proprio if u is None else u)
            aggregato = agg.aggregate(r)
            if aggregato is None:
                raise RuntimeError(f"shadow {spec.cid}/{spec.k} round {r}: aggregazione vuota")
            pesi = aggregato.global_weights
            if privato_server and spec.dp_mode == "central":
                torch.manual_seed(_seme_round(spec.seed, r, 2))
                pesi = gm.privatize_aggregate(
                    pesi, weight_keys=keys,
                    n_participants=aggregato.n_participants or spec.n_partecipanti,
                    participant_n_samples=list(
                        aggregato.metadata.get("participant_n_samples", {}).values()),
                )
        out[r] = mse_per_campione(spec.arch, spec.input_dim, pesi, spec.eval_x)
    return out


def esegui_traiettorie(specs: list[SpecShadow], workers: int = 1) -> list[dict[int, np.ndarray]]:
    """Esegue le traiettorie, in sequenza o in processi separati. L'ordine dei
    risultati segue quello di `specs` in entrambi i casi."""
    if workers <= 1 or len(specs) <= 1:
        # Nel processo principale: si ripristinano thread e generatore globale,
        # cosi' gli attacchi successivi non dipendono da LiRA (voce 73).
        stato = torch.get_num_threads()
        rng = torch.get_rng_state()
        try:
            return [traiettoria_shadow(s) for s in specs]
        finally:
            torch.set_num_threads(stato)
            torch.set_rng_state(rng)
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(traiettoria_shadow, specs))


# ── Attacco completo ───────────────────────────────────────────────────────────

def _metriche(labels: list[int], scores: list[float], pref: str) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    from run_experiments import _mia_advantage, _tpr_at_fixed_fpr
    res: dict[str, Any] = {f"{pref}_n_member": int(sum(labels)),
                           f"{pref}_n_nonmember": int(len(labels) - sum(labels))}
    if len(set(labels)) < 2:
        res[f"{pref}_auc_roc"] = None
        return res
    res[f"{pref}_auc_roc"] = round(float(roc_auc_score(labels, scores)), 6)
    for k, v in _tpr_at_fixed_fpr(labels, scores).items():
        res[k if pref == "lira" else f"{pref}_{k}"] = v
    res[f"{pref}_advantage"] = _mia_advantage(labels, scores)
    return res


def assegna_non_membri(
    holdout_sessions: list[dict[str, Any]], cluster_ids: list[str],
    cfg: dict, cluster_sizes: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """Client di origine dei non membri, coerente con la partizione dei membri
    (Segnalazioni, voce 72): per sito con `per_site`, a caso con le stesse
    proporzioni dei membri con `iid` o quando i siti non sono noti."""
    from run_experiments import _SITE_ID_TO_NAME
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    strategia = (cfg.get("partition") or {}).get("strategy", "per_site")
    out: dict[str, list[dict[str, Any]]] = {c: [] for c in cluster_ids}
    per_sito = strategia == "per_site"
    if per_sito:
        nomi = [_SITE_ID_TO_NAME.get(s.get("site_id", ""), s.get("site_id", "")) for s in holdout_sessions]
        per_sito = any(n in out for n in nomi)
    if per_sito:
        for s, n in zip(holdout_sessions, nomi):
            if n in out:
                out[n].append(s)
        return out
    rng = random.Random(seed + 271828)
    idx = list(range(len(holdout_sessions)))
    rng.shuffle(idx)
    tot = sum(cluster_sizes.values()) or 1
    pos = 0
    for j, c in enumerate(cluster_ids):
        n = (len(idx) - pos) if j == len(cluster_ids) - 1 else round(len(idx) * cluster_sizes[c] / tot)
        out[c] = [holdout_sessions[i] for i in idx[pos:pos + n]]
        pos += n
    return out


def run_lira_v2(
    cfg: dict,
    train_sessions: list[dict[str, Any]],
    holdout_sessions: list[dict[str, Any]],
    fl_results: dict[int, dict[str, Any]],
    n_shadow: int = 16,
    shadow_epochs_cap: int | None = None,
    no_dp: bool = False,
    dp_mode: str = "dp-fedavg",
    cluster_membership: dict[str, list[int]] | None = None,
    composed_output: dict[str, Any] | None = None,
    per_sample_dump_path: str | None = None,
    workers: int = 1,
) -> dict[int, dict[str, Any]]:
    """LiRA e RMIA versione 2 su ogni round di `fl_results`. Vedi il docstring
    del modulo per il metodo."""
    import json
    from pathlib import Path

    from run_experiments import (
        _autoencoder_arch_kwargs, _mia_feature_names, _sample_preserving_canary_groups,
        _subject_cross_client_weights, _subject_dp_config,
    )

    lira_cfg = cfg.get("lira", {}) or {}
    if lira_cfg.get("shadow_init", "warm") != "warm":
        raise ValueError("lira v2: shadow_init diverso da 'warm' non e' supportato")
    if lira_cfg.get("canary_rescale"):
        raise ValueError("lira v2: canary_rescale non e' supportato (usare lira.version: 1)")
    superficie = lira_cfg.get("observation_surface", "client")
    if superficie not in ("client", "global"):
        raise ValueError(f"observation_surface non valido: {superficie!r}")
    in_size = lira_cfg.get("in_size", "client")
    if in_size not in ("client", "half"):
        raise ValueError(f"lira.in_size non valido: {in_size!r} (atteso 'client' o 'half')")
    gamma = float(lira_cfg.get("rmia_gamma", 1.0))
    frazione_z = float(lira_cfg.get("rmia_population_fraction", 0.2))

    exp_cfg = cfg.get("experiment", {})
    ml_cfg = dict(cfg["ml"])
    seed = int(exp_cfg.get("seed", 42))
    feats = list(_mia_feature_names(cfg))
    arch = _autoencoder_arch_kwargs(cfg)
    input_dim = int(ml_cfg["input_dim"])
    if shadow_epochs_cap is not None and shadow_epochs_cap != ml_cfg.get("epochs"):
        logger.warning(
            "LiRA v2: --shadow-epochs-cap %s diverso dalle epoche del client (%s): "
            "LiRA RIDOTTA, numeri non validi (ESPERIMENTI.md)", shadow_epochs_cap, ml_cfg.get("epochs"))
        ml_cfg["epochs"] = int(shadow_epochs_cap)
    dp_ablation = exp_cfg.get("dp_ablation", "full")
    gm_cfg = {"epsilon": exp_cfg.get("epsilon", 1.0), "delta": exp_cfg.get("delta", 1e-5),
              "max_grad_norm": exp_cfg.get("max_grad_norm", 1.0),
              "sigma_calibration": exp_cfg.get("sigma_calibration", "analitica")}
    rounds = sorted(r for r in fl_results if r >= 1)

    # Partizione dei membri: la stessa di run_fl_rounds
    if cluster_membership is not None:
        cluster_ids = list(cluster_membership.keys())
        pools = cluster_membership
    else:
        cluster_ids = ["highway", "urban", "residential", "corporate"]
        size = max(1, len(train_sessions) // len(cluster_ids))
        pools = {c: list(range(i * size, len(train_sessions) if i == len(cluster_ids) - 1 else (i + 1) * size))
                 for i, c in enumerate(cluster_ids)}
    membri = {c: [train_sessions[i] for i in pools[c]] for c in cluster_ids}
    non_membri = assegna_non_membri(holdout_sessions, cluster_ids, cfg,
                                    {c: len(v) for c, v in membri.items()})

    sdp = _subject_dp_config(ml_cfg)
    pesi_soggetti = (_subject_cross_client_weights(membri, str(sdp["subject_key"]), feats)
                     if sdp["enabled"] else None)

    common_init = bool(ml_cfg.get("common_init", False))
    init_comune = (fl_results.get(0) or {}).get("raw_global_weights") if common_init else None
    global_prev = {r: fl_results[r - 1]["global_weights"] for r in rounds if r > 1}

    def _vettore(sess: list[dict[str, Any]]) -> np.ndarray:
        return np.array([[float(s[f]) for f in feats] for s in sess], dtype=np.float32)

    def _valida(s: dict[str, Any]) -> bool:
        try:
            return all(np.isfinite(float(s[f])) for f in feats)
        except (KeyError, TypeError, ValueError):
            return False

    risultati: dict[int, dict[str, Any]] = {r: {} for r in rounds}
    righe: dict[int, list[dict[str, Any]]] = {r: [] for r in rounds}
    specs: list[SpecShadow] = []
    info_client: dict[str, dict[str, Any]] = {}

    for ci, c in enumerate(cluster_ids):
        rng = random.Random(seed + 9001 + ci * _SEED_CLUSTER)
        nm = [s for s in non_membri[c] if _valida(s)]
        mb = [s for s in membri[c] if _valida(s)]
        if not nm or not mb:
            logger.warning("LiRA v2 [%s]: membri o non membri assenti, client saltato", c)
            continue
        # popolazione z di RMIA: non membri (non canary) riservati, mai valutati,
        # scelti a caso senza guardare punteggi o perdite
        canarini = [s for s in mb + nm if s.get("_canary_group") is not None]
        canarini_ids = {id(s) for s in canarini}
        nm_shuf = [s for s in nm if id(s) not in canarini_ids]
        rng.shuffle(nm_shuf)
        n_z = max(1, int(round(frazione_z * len(nm_shuf)))) if len(nm_shuf) > 1 else 0
        z = nm_shuf[:n_z]
        nm_eval = nm_shuf[n_z:]
        mb_libere = [s for s in mb if id(s) not in canarini_ids]
        mb_eval = rng.sample(mb_libere, min(len(nm_eval), len(mb_libere)))
        valutati = mb_eval + nm_eval + canarini
        mb_ids = {id(s) for s in mb}
        membro = np.array([id(s) in mb_ids for s in valutati], dtype=bool)
        tutti = valutati + z                                   # z in coda
        x = _vettore(tutti)
        # universo e insiemi IN degli shadow
        universo = mb + nm
        n_in = len(mb) if in_size == "client" else max(1, len(universo) // 2)
        if n_in >= len(universo):
            raise ValueError(f"LiRA v2 [{c}]: insieme IN ({n_in}) >= universo ({len(universo)})")
        in_mask = np.zeros((n_shadow, len(tutti)), dtype=bool)
        pos = {id(s): j for j, s in enumerate(tutti)}
        for k in range(n_shadow):
            sseed = seed + ci * _SEED_CLUSTER + k * _SEED_SHADOW
            ins = _sample_preserving_canary_groups(random.Random(sseed), universo, n_in)
            for s in ins:
                j = pos.get(id(s))
                if j is not None:
                    in_mask[k, j] = True
            altri: dict[int, list[Any]] = {}
            n_cli: dict[int, int] = {}
            if superficie == "global":
                for r in rounds:
                    upd = fl_results[r].get("updates") or []
                    altri[r] = [None if u.cluster_id == c else u for u in upd]
                    propri = [u for u in upd if u.cluster_id == c]
                    if propri:
                        n_cli[r] = propri[0].n_samples
            specs.append(SpecShadow(
                cid=c, k=k, seed=sseed, rounds=rounds, in_sessions=ins, eval_x=x,
                ml_cfg=ml_cfg, arch=arch, input_dim=input_dim, gm_cfg=gm_cfg,
                no_dp=no_dp, dp_mode=dp_mode, dp_ablation=dp_ablation,
                init_weights=init_comune, global_prev=global_prev,
                subject_weights=(pesi_soggetti or {}).get(c, {}) if pesi_soggetti is not None else None,
                superficie=superficie, altri_update=altri, n_samples_client=n_cli,
                n_partecipanti=len(cluster_ids),
            ))
        info_client[c] = {"valutati": valutati, "membro": membro, "n_z": len(z), "x": x,
                          "in_mask": in_mask, "n_in": n_in, "universo": len(universo)}

    logger.info("LiRA v2: %d traiettorie di shadow (%d per client), %s worker, superficie %s, IN = %s",
                len(specs), n_shadow, workers, superficie, in_size)
    esiti = esegui_traiettorie(specs, workers=workers)
    per_client: dict[str, list[dict[int, np.ndarray]]] = {}
    for spec, es in zip(specs, esiti):
        per_client.setdefault(spec.cid, []).append(es)

    cumul_lira: dict[int, float] = {}
    cumul_rmia: dict[int, float] = {}
    conta: dict[int, int] = {}
    meta: dict[int, tuple[bool, Any, Any]] = {}
    for r in rounds:
        rd = fl_results[r]
        if superficie == "global":
            pesi_target = {c: rd["global_weights"] for c in info_client}
        else:
            fonte = (rd.get("raw_updates") if (not no_dp and dp_mode == "dp-fedavg") else rd.get("updates")) or []
            pesi_target = {u.cluster_id: u.weights for u in fonte}
        lab_l: list[int] = []
        sc_l: list[float] = []
        lab_r: list[int] = []
        sc_r: list[float] = []
        raw_lab: list[int] = []
        raw_sc: list[float] = []
        can = {"l": [], "r": [], "raw": []}
        diag_c: dict[str, Any] = {}
        for c, inf in info_client.items():
            if c not in pesi_target:
                logger.warning("LiRA v2 round %d: bersaglio del client %s assente", r, c)
                continue
            mse_t = mse_per_campione(arch, input_dim, pesi_target[c], inf["x"])
            mse_s = np.stack([es[r] for es in per_client[c]])          # (n_shadow, n)
            nv = len(inf["valutati"])
            t = statistica(mse_t[:nv])
            score, diag = punteggio_lira(t, statistica(mse_s[:, :nv]), inf["in_mask"][:, :nv])
            tau = float(np.median(mse_t[nv:])) if inf["n_z"] else float(np.median(mse_t))
            llr_all = log_lr_rmia(mse_t, mse_s, tau)
            rmia = punteggio_rmia(llr_all[:nv], llr_all[nv:], gamma)
            ok = np.isfinite(score) & np.isfinite(rmia)
            m = inf["membro"]
            for j in np.flatnonzero(ok):
                s = inf["valutati"][j]
                lab = int(m[j])
                canary = s.get("_canary_group") is not None
                # i canary entrano solo nelle metriche canary (revisione 2026-10-16)
                if not canary:
                    lab_l.append(lab); sc_l.append(float(score[j]))
                    lab_r.append(lab); sc_r.append(float(rmia[j]))
                key = id(s)
                cumul_lira[key] = cumul_lira.get(key, 0.0) + float(score[j])
                cumul_rmia[key] = cumul_rmia.get(key, 0.0) + float(llr_all[j])
                conta[key] = conta.get(key, 0) + 1
                meta[key] = (bool(m[j]), s.get("_canary_group"), s.get("session_id"))
                if s.get("_canary_group") is not None:
                    can["l"].append((lab, float(score[j])))
                    can["r"].append((lab, float(rmia[j])))
            for j in range(nv):
                if inf["valutati"][j].get("_canary_group") is not None:
                    can["raw"].append((int(m[j]), -float(mse_t[j])))
                else:
                    raw_lab.append(int(m[j])); raw_sc.append(-float(mse_t[j]))
            ok_nc = [j for j in np.flatnonzero(ok) if inf["valutati"][j].get("_canary_group") is None]
            pc = _metriche([int(m[j]) for j in ok_nc], [float(score[j]) for j in ok_nc], "lira")
            diag_c[c] = {"auc": pc.get("lira_auc_roc"), **{k: v for k, v in diag.items()},
                         "n_scartati": int(nv - ok.sum()), "tau_rmia": tau,
                         "n_in_shadow": inf["n_in"], "universo": inf["universo"]}
        res = _metriche(lab_l, sc_l, "lira")
        res.update(_metriche(lab_r, sc_r, "rmia"))
        from sklearn.metrics import roc_auc_score
        res["lira_debug_raw_mse_auc_roc"] = (round(float(roc_auc_score(raw_lab, raw_sc)), 6)
                                             if len(set(raw_lab)) == 2 else None)
        if lab_l:
            mem = [s for l, s in zip(lab_l, sc_l) if l]
            non = [s for l, s in zip(lab_l, sc_l) if not l]
            if mem and non:
                res["lira_member_score_mean"] = round(float(np.mean(mem)), 6)
                res["lira_non_member_score_mean"] = round(float(np.mean(non)), 6)
                res["lira_score_gap"] = round(float(np.mean(mem) - np.mean(non)), 6)
        for nome, coppie in (("canary_auc_roc", can["l"]), ("canary_rmia_auc_roc", can["r"]),
                             ("canary_raw_mse_auc_roc", can["raw"])):
            ll = [a for a, _ in coppie]
            res[nome] = (round(float(roc_auc_score(ll, [b for _, b in coppie])), 6)
                         if len(set(ll)) == 2 else None)
        res.update({
            "lira_version": 2, "n_shadow": n_shadow, "lira_observation_surface": superficie,
            "lira_in_size": in_size, "rmia_gamma": gamma,
            "lira_per_client": diag_c,
        })
        risultati[r] = res
        logger.info("Round %d — LiRA v2 AUC %s, RMIA AUC %s (n=%d)", r, res.get("lira_auc_roc"),
                    res.get("rmia_auc_roc"), len(lab_l))

    # composizione sui round: solo i campioni valutati in tutti i round
    completi = [k for k, n in conta.items() if n == len(rounds) and meta[k][1] is None]
    if composed_output is not None and completi:
        lab = [int(meta[k][0]) for k in completi]
        composed_output.update({f"composed_{k}": v for k, v in
                                _metriche(lab, [cumul_lira[k] for k in completi], "lira").items()})
        # somma dei log-rapporti di RMIA sui round (non il punteggio a coppie di RMIA)
        composed_output.update({f"composed_{k}": v for k, v in
                                _metriche(lab, [cumul_rmia[k] for k in completi], "rmia_loglr").items()})
        composed_output["composed_n_samples"] = len(completi)
    if per_sample_dump_path is not None and completi:
        Path(per_sample_dump_path).parent.mkdir(parents=True, exist_ok=True)
        with open(per_sample_dump_path, "w") as f:
            json.dump({
                "seed": seed, "epsilon": exp_cfg.get("epsilon"), "no_dp": no_dp, "dp_mode": dp_mode,
                "lira_version": 2, "n_rounds": len(rounds), "n_records": len(completi),
                "records": [{"session_id": meta[k][2], "is_member": meta[k][0],
                             "is_canary": meta[k][1] is not None, "composed_score": cumul_lira[k],
                             "composed_rmia_log_lr": cumul_rmia[k]} for k in completi],
            }, f)
    return risultati
