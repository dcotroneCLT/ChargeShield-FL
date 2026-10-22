"""
Calibrazione e contabilita' della DP per client (2026-10-16, Segnalazioni R
voce 78).

Prima: sigma = C * sqrt(2 ln(1,25/delta)) / epsilon, la formula classica del
meccanismo gaussiano, valida solo per epsilon < 1 (Dwork e Roth, teorema A.1);
i config usano epsilon per round da 2 a 256, dove la formula da' meno rumore
di quello che serve. Adiacenza mai dichiarata.

Ora:
- sigma dal meccanismo gaussiano analitico (Balle e Wang, ICML 2018, teorema 8):
  il piu' piccolo sigma per cui
      Phi(D/(2 s) - eps s/D) - e^eps Phi(-D/(2 s) - eps s/D) <= delta,
  valido per ogni epsilon > 0. sigma e' lineare nella sensibilita' D.
- adiacenza dichiarata: SOSTITUZIONE dei dati di un client, in tutte le
  modalita' (stessa relazione della DP per record e per soggetto, dove n e'
  pubblico). Il delta del client e' tagliato a norma C, due delta tagliati
  distano al piu' 2C:
  * dp-fedavg e local (rumore per client): D = 2C;
  * central (un rumore sull'aggregato, McMahan et al. 2018), pesi n_i/N
    pubblici e fissi, tutti i client presenti in ogni round (controllato in
    run_fl_rounds): D = 2 max_i(w_i) C. Con aggiunta/rimozione di un client il
    FedAvg rinormalizzato cambierebbe anche i pesi degli altri (revisione del
    2026-10-16), quindi non si dichiara.
- epsilon totale su T round: RDP del meccanismo gaussiano con moltiplicatore
  sigma/D, composto T volte (composizione adattiva), convertito in
  (epsilon, delta).
"""

from __future__ import annotations

import math
from typing import Any

ADIACENZE = ("sostituzione", "add_remove")


def _log_phi(x: float) -> float:
    from scipy.special import log_ndtr
    return float(log_ndtr(x))


def delta_gaussiano(sigma: float, eps: float, sens: float = 1.0) -> float:
    """delta esatto del meccanismo gaussiano (Balle e Wang 2018, teorema 8)."""
    a = sens / (2.0 * sigma)
    b = eps * sigma / sens
    t1 = math.exp(_log_phi(a - b))
    log_t2 = eps + _log_phi(-a - b)
    t2 = math.exp(log_t2) if log_t2 < 700 else float("inf")
    return max(0.0, t1 - t2)


def sigma_gaussiano_analitico(eps: float, delta: float, sens: float = 1.0,
                              tol: float = 1e-12) -> float:
    """Il piu' piccolo sigma con delta_gaussiano(sigma, eps, sens) <= delta."""
    if eps <= 0 or not (0 < delta < 1) or sens <= 0:
        raise ValueError(f"parametri non validi: eps={eps}, delta={delta}, sens={sens}")
    lo, hi = 1e-6 * sens, sens
    while delta_gaussiano(hi, eps, sens) > delta:
        hi *= 2.0
        if hi > 1e8 * sens:
            raise RuntimeError("sigma non trovato")
    while delta_gaussiano(lo, eps, sens) <= delta:
        lo /= 2.0
        if lo < 1e-15:
            return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if delta_gaussiano(mid, eps, sens) <= delta:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * hi:
            break
    return hi


def sigma_classico(eps: float, delta: float, sens: float = 1.0) -> float:
    """Formula classica (solo per riprodurre le run precedenti al 2026-10-16)."""
    return sens * math.sqrt(2.0 * math.log(1.25 / delta)) / eps


def epsilon_totale_rdp(noise_multiplier: float, rounds: int, delta: float) -> float:
    """epsilon su `rounds` composizioni del meccanismo gaussiano con
    moltiplicatore sigma/D (RDP, dp-accounting)."""
    from dp_accounting import dp_event, rdp
    acc = rdp.RdpAccountant()
    acc.compose(dp_event.GaussianDpEvent(noise_multiplier), int(rounds))
    return float(acc.get_epsilon(target_delta=delta))


def adiacenza_per_modo(dp_mode: str) -> str:
    return "sostituzione"


def client_dp_fields(cfg: dict, dp_mode: str, no_dp: bool,
                     pesi_pubblici: dict[str, float] | None) -> dict[str, Any]:
    """Campi del JSON per la DP per client."""
    exp = cfg.get("experiment", {}) or {}
    out: dict[str, Any] = {"client_dp_accounting": None}
    if no_dp:
        return out
    abl = exp.get("dp_ablation", "full")
    eps = float(exp["epsilon"])
    delta = float(exp["delta"])
    C = float(exp["max_grad_norm"])
    rounds = int(exp["fl_rounds"])
    calib = exp.get("sigma_calibration", "analitica")
    adj = adiacenza_per_modo(dp_mode)
    if dp_mode == "central":
        tot = float(sum((pesi_pubblici or {}).values())) or 1.0
        w_max = max((v / tot for v in (pesi_pubblici or {}).values()), default=1.0)
        sens = 2.0 * w_max * C
    else:
        sens = 2.0 * C
    unit = (sigma_gaussiano_analitico(eps, delta, 1.0) if calib == "analitica"
            else sigma_classico(eps, delta, 1.0))
    sigma = unit * sens
    garanzia = abl == "full"
    out["client_dp_accounting"] = {
        "calibrazione": calib,
        "adiacenza": adj,
        "sensibilita": sens,
        "sigma": sigma,
        "epsilon_round": eps if (garanzia and calib == "analitica") else None,
        "epsilon_totale_rdp": (epsilon_totale_rdp(sigma / sens, rounds, delta)
                               if garanzia else None),
        "delta": delta,
        "fl_rounds": rounds,
        "pesi_pubblici": dict(pesi_pubblici) if pesi_pubblici else None,
        "nota": ("nessuna garanzia: ablazione " + abl) if not garanzia else (
            "sostituzione dei dati di un client; dp-fedavg/local D = 2C, "
            "central D = 2 max_i w_i C con pesi pubblici e tutti i client presenti. "
            "Non coperta: la scelta di C."),
    }
    return out
