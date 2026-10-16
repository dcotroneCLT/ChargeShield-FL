"""
DP a livello di SOGGETTO (2026-10-13).

Perche' esiste. La DP per record (`record_dp`) protegge la singola sessione; la
garanzia per una persona con k sessioni si ricava solo per privacy di gruppo e
cresce molto con k (`scripts/privacy_di_gruppo.py`). La DP per client protegge
l'intero sito. Qui l'unita' protetta e' il soggetto: tutte le sessioni che
condividono la stessa chiave (`user_id` in ACN-Data, la colonnina `node_id` in
ChargePlace Scotland, dove l'utente non esiste).

Meccanismo (DP-SGD a livello di utente, McMahan et al. 2018, "Learning
differentially private recurrent language models", applicato dentro ogni
client). A ogni passo locale:
  1. ogni soggetto del client entra nel lotto con probabilita' q
     (campionamento di Poisson vero, non shuffle: l'accountant e' una garanzia);
  2. per ogni soggetto estratto si prendono al piu' `max_records_per_subject`
     delle sue sessioni e si calcola il gradiente della loss media su quelle;
  3. il gradiente del soggetto e' tagliato a norma C * w (w = peso del
     soggetto in questo client, vedi sotto);
  4. si sommano i gradienti tagliati, si aggiunge rumore N(0, (sigma*C)^2) e
     si divide per il numero atteso di soggetti q*N.
Passi per epoca = ceil(N / soggetti attesi per passo), cosi' in media ogni
soggetto entra una volta per epoca.

Chi e' un soggetto (`subject_of`):
  - le copie di un canary (`_canary_group`) sono un unico soggetto: e' proprio
    il caso "una persona con molte sessioni simili" che la DP per record non
    copre;
  - altrimenti il valore della chiave scelta;
  - se la chiave manca (sessioni ACN senza userID): dal 2026-10-16 il profilo
    di abitudine della sessione (`senza_chiave: profili`, src/ml/
    profili_abitudini.py, profili stimati su una riserva dell'holdout); con
    `senza_chiave: singoli` la sessione e' un soggetto a se' stante (prima).

Soggetti presenti in piu' client (utenti ACN che ricaricano in piu' siti). Ogni
client aggiunge il proprio rumore, quindi la perdita di privacy del soggetto si
compone sui client in cui compare. Per limitarla, come in ULDP-FL (Kato et al.,
PVLDB 2024), il contributo del soggetto nel client c e' tagliato a C * w_c con
w_c = (sue sessioni in c) / (sue sessioni totali), cosi' che sum_c w_c = 1. In
ogni client il moltiplicatore di rumore effettivo per quel soggetto e'
sigma / w_c; l'accountant somma le RDP sui client. Ipotesi da dichiarare: i
client conoscono quante sessioni ha ogni soggetto negli altri client (in
ULDP-FL lo si ottiene con un protocollo di intersezione privata; qui la
simulazione lo calcola direttamente).

Non contabilizzati (da dichiarare insieme all'epsilon):
  - il numero di soggetti N di ogni client, trattato come pubblico: q e il
    numero di passi dipendono da N (come in Opacus);
  - la funzione sessione -> profilo, stimata su dati esterni al training.
Dal 2026-10-16 non sono piu' fra i non contabilizzati: la normalizzazione
(limiti pubblici, non dai dati), i pesi di aggregazione (pubblici e fissi) e
la loss di training (non inviata nelle celle con DP).
Validita' delle chiavi: i soggetti si contano sulle sole righe valide (feature
presenti e numeriche), con la stessa regola del trainer; se il trainer e
l'accountant vedessero N diversi, run_fl_rounds() si ferma con un errore.
Il denominatore del gradiente e' sempre il numero atteso di soggetti per passo
(costante), anche quando q = 1: dividere per N renderebbe la scala del passo
dipendente dai dati.

Il modulo e' puro Python (dp-accounting importato dentro le funzioni): i test
dell'accounting non richiedono torch.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "subject_key": "user_id",
    "max_grad_norm": 1.0,
    "noise_multiplier": 0.0,
    "expected_subjects_per_step": 32,
    "max_records_per_subject": 64,
    # Sessioni senza chiave (2026-10-16): "profili" = soggetto dal profilo di
    # abitudine (src/ml/profili_abitudini.py), "singoli" = ogni sessione un
    # soggetto (comportamento precedente).
    "senza_chiave": "profili",
    "profili": {"dimensione_media": 10.0, "peso_gruppo": 1.0},
}

_VUOTI = ("", "none", "nan", "null")


_VERI = ("1", "true", "yes", "si", "sì", "on")
_FALSI = ("0", "false", "no", "off", "")


def flag(v: Any) -> bool:
    """Interruttore di config letto in modo stretto (2026-10-16, Segnalazioni R
    voce 80): bool("false") e' True, quindi le stringhe si leggono per valore;
    un valore non riconosciuto e' un errore, non un "acceso" silenzioso."""
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _VERI:
            return True
        if s in _FALSI:
            return False
    raise ValueError(f"valore di interruttore non riconosciuto: {v!r}")


def record_dp_on(ml_cfg: dict | None) -> bool:
    """`ml.record_dp.enabled` letto con `flag` (unico punto per trainer,
    attacchi e accounting)."""
    return flag(((ml_cfg or {}).get("record_dp") or {}).get("enabled", False))


_enabled = flag


def subject_dp_config(ml_cfg: dict | None) -> dict[str, Any]:
    """Config `ml.subject_dp` completato con i default."""
    raw = (ml_cfg or {}).get("subject_dp") or {}
    out = {**DEFAULTS, **raw}
    out["enabled"] = _enabled(out["enabled"])
    if out["senza_chiave"] not in ("profili", "singoli"):
        raise ValueError(f"subject_dp.senza_chiave non valido: {out['senza_chiave']!r}")
    return out


def _chiave(v: Any) -> str | None:
    """Valore della chiave in forma canonica, None se assente.

    Gli id numerici scritti in modi diversi (22, "22", "000022", 22.0) danno
    lo stesso soggetto; NaN e stringhe vuote sono assenza di chiave."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if v.is_integer():
            return str(int(v))
    s = str(v).strip()
    if s.lower() in _VUOTI:
        return None
    if s.isdigit():
        return str(int(s))
    return s


def row_valid(session: dict[str, Any], features: list[str] | None) -> bool:
    """Stessa regola di AutoencoderTrainer._sessions_to_tensor: ogni feature
    presente e convertibile in float. features=None: nessun filtro."""
    if features is None:
        return True
    for f in features:
        val = session.get(f)
        if val is None:
            return False
        try:
            x = float(val)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(x):    # 2026-10-16: inf/nan aggirerebbero il taglio
            return False
    return True


def subject_of(session: dict[str, Any], key: str, fallback: str) -> str:
    """Identificativo del soggetto di una sessione (vedi docstring del modulo)."""
    g = _chiave(session.get("_canary_group"))
    if g is not None:
        return f"canary:{g}"
    v = _chiave(session.get(key))
    if v is not None:
        return f"{key}:{v}"
    p = session.get("_profilo")   # assegnato solo con senza_chiave: profili
    if p is not None:
        return f"profilo:{p}"
    return f"solo:{fallback}"


def subject_ids(sessions: list[dict[str, Any]], key: str, client_id: str) -> list[str]:
    """Soggetto di ogni sessione del client, nello stesso ordine.

    Il fallback dipende da client e posizione nella lista COMPLETA: deve
    coincidere fra `cross_client_weights` e il trainer, che ricevono la
    stessa lista."""
    return [subject_of(s, key, f"{client_id}:{i}") for i, s in enumerate(sessions)]


def valid_subject_ids(
    sessions: list[dict[str, Any]], key: str, client_id: str,
    features: list[str] | None,
) -> list[str]:
    """Soggetti delle sole righe valide, nell'ordine delle sessioni."""
    return [sid for s, sid in zip(sessions, subject_ids(sessions, key, client_id))
            if row_valid(s, features)]


def cross_client_weights(
    cluster_sessions: dict[str, list[dict[str, Any]]], key: str,
    features: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """{client: {soggetto: w}} con w = righe valide del soggetto nel client /
    totali.

    Si riportano solo i soggetti presenti in piu' client (w < 1): per gli altri
    il trainer usa w = 1."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for cid, ss in cluster_sessions.items():
        for subj in valid_subject_ids(ss, key, cid, features):
            counts[subj][cid] += 1
    weights: dict[str, dict[str, float]] = {cid: {} for cid in cluster_sessions}
    for subj, per in counts.items():
        if len(per) < 2:
            continue
        tot = sum(per.values())
        for cid, n in per.items():
            weights[cid][subj] = n / tot
    return weights


def sampling_schedule(n_subjects: int, expected_per_step: int) -> tuple[float, int]:
    """(q, passi per epoca). Stessa formula per il trainer e l'accountant."""
    if n_subjects <= 0:
        return 0.0, 0
    expected = max(1, int(expected_per_step))
    if expected >= n_subjects:
        return 1.0, 1
    return expected / n_subjects, math.ceil(n_subjects / expected)


def subject_stats(
    cluster_sessions: dict[str, list[dict[str, Any]]], key: str,
    features: list[str] | None = None,
) -> dict[str, Any]:
    """Numeri che servono all'accountant, calcolati sulle righe valide delle
    sessioni di training (stessa regola del trainer)."""
    per_client_subjects: dict[str, int] = {}
    per_client_singletons: dict[str, int] = {}
    for cid, ss in cluster_sessions.items():
        ids = set(valid_subject_ids(ss, key, cid, features))
        per_client_subjects[cid] = len(ids)
        per_client_singletons[cid] = sum(1 for x in ids if x.startswith("solo:"))
    w = cross_client_weights(cluster_sessions, key, features)
    vettori: dict[str, dict[str, float]] = defaultdict(dict)
    for cid, per in w.items():
        for subj, val in per.items():
            vettori[subj][cid] = val
    # Pesi arrotondati PER ECCESSO a 1e-3 (sensibilita' piu' alta, epsilon piu'
    # grande: prudente) per raggruppare i vettori uguali.
    unici: dict[tuple, int] = defaultdict(int)
    for vec in vettori.values():
        t = tuple(sorted((c, math.ceil(v * 1000) / 1000) for c, v in vec.items()))
        unici[t] += 1
    return {
        "subject_key": key,
        "features": list(features) if features is not None else None,
        "per_client_subjects": per_client_subjects,
        "per_client_singletons": per_client_singletons,
        "n_multi_client_subjects": len(vettori),
        "multi_client_weight_vectors": [
            {"weights": dict(t), "n_subjects": n} for t, n in sorted(unici.items())
        ],
    }


RELAZIONI = ("sostituzione", "add_remove")


def accountant(relazione: str = "sostituzione"):
    """Accountant per la relazione di adiacenza (2026-10-16, revisione).

    "sostituzione" (default): si sostituisce un record o un soggetto, il numero
    n resta fisso. E' la relazione coerente con quello che il server vede: n
    (e quindi q = B/n e i pesi di aggregazione) e' pubblico. PLD, che supporta
    la sostituzione con campionamento di Poisson.
    "add_remove": si aggiunge o toglie un record o un soggetto (RDP); vale
    solo se n non e' osservabile, qui riportato per confronto con la
    letteratura."""
    import dp_accounting
    from dp_accounting import pld, rdp
    if relazione == "sostituzione":
        return pld.PLDAccountant(neighboring_relation=dp_accounting.NeighboringRelation.REPLACE_ONE)
    if relazione == "add_remove":
        return rdp.RdpAccountant()
    raise ValueError(f"relazione di adiacenza non valida: {relazione!r}")


def _event(q: float, noise_multiplier: float):
    from dp_accounting import dp_event

    g = dp_event.GaussianDpEvent(noise_multiplier)
    return g if q >= 1.0 else dp_event.PoissonSampledDpEvent(q, g)


def epsilon_single_client(
    n_subjects: int, expected_per_step: int, epochs: int, rounds: int,
    noise_multiplier: float, delta: float, weight: float = 1.0,
    relazione: str = "sostituzione",
) -> tuple[float, float, int]:
    """(epsilon, q, passi) per un soggetto con peso `weight` in un client."""
    q, per_epoch = sampling_schedule(n_subjects, expected_per_step)
    steps = per_epoch * epochs * rounds
    if steps == 0 or weight <= 0.0:
        return 0.0, q, steps
    acc = accountant(relazione)
    acc.compose(_event(q, noise_multiplier / weight), steps)
    return float(acc.get_epsilon(target_delta=delta)), q, steps


def epsilon_multi_client(
    weights: dict[str, float], per_client_subjects: dict[str, int],
    expected_per_step: int, epochs: int, rounds: int,
    noise_multiplier: float, delta: float, relazione: str = "sostituzione",
) -> float:
    """Epsilon di un soggetto presente in piu' client: composizione sui client."""
    acc = accountant(relazione)
    for cid, w in weights.items():
        if w <= 0.0:
            continue
        q, per_epoch = sampling_schedule(per_client_subjects.get(cid, 0), expected_per_step)
        steps = per_epoch * epochs * rounds
        if steps:
            acc.compose(_event(q, noise_multiplier / w), steps)
    return float(acc.get_epsilon(target_delta=delta))


def subject_dp_accounting(
    stats: dict[str, Any], expected_per_step: int, epochs: int, rounds: int,
    noise_multiplier: float, delta: float,
) -> dict[str, Any]:
    """Epsilon per client (soggetti in un solo client) e per i soggetti
    presenti in piu' client; `epsilon_max` e' il peggiore fra tutti."""
    pcs = stats["per_client_subjects"]
    per_client: dict[str, dict[str, Any]] = {}
    for cid, n in pcs.items():
        eps, q, steps = epsilon_single_client(
            int(n), expected_per_step, epochs, rounds, noise_multiplier, delta)
        eps_ar, _, _ = epsilon_single_client(
            int(n), expected_per_step, epochs, rounds, noise_multiplier, delta, relazione="add_remove")
        per_client[str(cid)] = {"n_soggetti": int(n), "q": q, "passi": steps, "epsilon": eps,
                                "epsilon_add_remove": eps_ar}
    multi = []
    for item in stats.get("multi_client_weight_vectors") or []:
        eps = epsilon_multi_client(item["weights"], pcs, expected_per_step, epochs,
                                   rounds, noise_multiplier, delta)
        multi.append({**item, "epsilon": eps})
    worst_client = max(per_client, key=lambda c: per_client[c]["epsilon"]) if per_client else None
    eps_single = per_client[worst_client]["epsilon"] if worst_client else 0.0
    eps_multi = max((m["epsilon"] for m in multi), default=0.0)
    return {
        "per_client": per_client,
        "client_peggiore": worst_client,
        "epsilon_un_client_max": eps_single,
        "multi_client": multi,
        "epsilon_multi_client_max": eps_multi if multi else None,
        "epsilon_max": max(eps_single, eps_multi),
    }


def subject_dp_fields(cfg: dict, stats: dict[str, Any] | None) -> dict[str, Any]:
    """Campi del JSON risultato. Nessun numero con parametri mancanti."""
    ml = cfg.get("ml") if isinstance(cfg.get("ml"), dict) else cfg
    sd = subject_dp_config(ml)
    out: dict[str, Any] = {
        "subject_dp": sd if sd["enabled"] else None,
        "subject_dp_stats": stats if sd["enabled"] else None,
        "epsilon_subject_dp": None,
        "epsilon_subject_dp_per_client": None,
        "epsilon_subject_dp_multi_client": None,
        "subject_dp_accounting": None,
        "subject_dp_accounting_note": None,
    }
    if not sd["enabled"]:
        return out
    sigma = float(sd.get("noise_multiplier") or 0.0)
    exp = cfg.get("experiment") or {}
    epochs = ml.get("epochs")
    rounds = exp.get("fl_rounds", cfg.get("fl_rounds"))
    delta = exp.get("delta", cfg.get("delta"))
    mancanti = [n for n, v in (("epochs", epochs), ("fl_rounds", rounds),
                               ("delta", delta), ("statistiche dei soggetti", stats)) if v is None]
    if mancanti or sigma <= 0.0:
        motivo = (f"parametri mancanti: {', '.join(mancanti)}" if mancanti
                  else f"noise_multiplier={sigma}: nessuna garanzia")
        out["subject_dp_accounting_note"] = (
            f"DP per soggetto attiva ma epsilon non calcolato ({motivo}). "
            "Il campo 'epsilon' NON si applica a questa cella.")
        return out
    try:
        acc = subject_dp_accounting(stats, int(sd["expected_subjects_per_step"]),
                                    int(epochs), int(rounds), sigma, float(delta))
    except ImportError:
        out["subject_dp_accounting_note"] = (
            "DP per soggetto attiva; 'dp-accounting' non installato, epsilon da "
            "ricalcolare. Il campo 'epsilon' NON si applica.")
        return out
    except Exception as exc:  # pragma: no cover - non deve far perdere la run
        logger.warning("[SUBJECT-DP] accountant fallito: %s", exc)
        out["subject_dp_accounting_note"] = (
            f"DP per soggetto attiva; accountant fallito ({exc}). "
            "Il campo 'epsilon' NON si applica.")
        return out
    out["epsilon_subject_dp"] = acc["epsilon_max"]
    out["epsilon_subject_dp_per_client"] = {c: v["epsilon"] for c, v in acc["per_client"].items()}
    out["epsilon_subject_dp_multi_client"] = acc["epsilon_multi_client_max"]
    out["subject_dp_accounting"] = {
        "accountant": "RDP (dp-accounting), campionamento di Poisson dei soggetti, "
                      "adiacenza add/remove di un soggetto",
        "noise_multiplier": sigma, "max_grad_norm": sd["max_grad_norm"],
        "expected_subjects_per_step": sd["expected_subjects_per_step"],
        "max_records_per_subject": sd["max_records_per_subject"],
        "epochs": int(epochs), "fl_rounds": int(rounds), "delta": float(delta),
        **acc,
    }
    multi_txt = (f"; soggetti in piu' client: {acc['epsilon_multi_client_max']:.4g}"
                 if acc["epsilon_multi_client_max"] is not None else "")
    sing = stats.get("per_client_singletons") or {}
    sing_txt = (f" Soggetti singoli (sessioni senza chiave, protette solo come "
                f"sessione): {sum(sing.values())}." if sing else "")
    out["subject_dp_accounting_note"] = (
        f"epsilon_subject_dp = {acc['epsilon_max']:.4g} per soggetto "
        f"(chiave {stats.get('subject_key')!r}; peggior client "
        f"{acc['client_peggiore']}: {acc['epsilon_un_client_max']:.4g}{multi_txt}). "
        f"RDP, Poisson, sigma={sigma}, E={epochs}, T={rounds}, delta={delta}. "
        "Questo, NON il campo 'epsilon', e' il budget del meccanismo sui pesi; "
        "non contabilizzati: n_samples e loss media inviati al server senza "
        "rumore, statistiche min-max, numero di soggetti per client trattato "
        f"come pubblico.{sing_txt}")
    return out
