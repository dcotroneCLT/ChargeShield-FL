"""
Privacy di gruppo della DP per record: quale garanzia si puo' dichiarare per
un gruppo di k sessioni (la stessa persona, la stessa colonnina) quando il
meccanismo protegge la singola sessione.

Due limiti, entrambi validi; si riporta il minore.

(A) Teorema generico per (eps, delta)-DP: se il meccanismo e' (e, d)-DP per un
    record, per un gruppo di k record e' (k*e, k*exp((k-1)*e)*d)-DP. Si sceglie
    il delta per record d che rende k*exp((k-1)*e(d))*d <= delta_obiettivo e
    minimizza k*e(d). Se nessun d lo permette, il limite e' infinito.

(B) Privacy di gruppo in RDP (Mironov 2017, Prop. 2): se il meccanismo e'
    (alpha, r(alpha))-RDP e alpha >= 2^(c+1), per gruppi di 2^c record e'
    (alpha/2^c, 3^c * r(alpha))-RDP. Per k record si usa c = ceil(log2 k).
    Poi conversione standard in (eps, delta).

La curva RDP per record e' quella dell'accountant di Poisson gia' usato in
`src/ml/record_dp_accounting.py` (q = B/n, passi = floor(n/B)*E*T). Come li',
e' un'APPROSSIMAZIONE: il training usa lo shuffle con batch fissi. Il limite
valido con lo shuffle e senza amplificazione e' gia' molto grande per k = 1
(vedi `epsilon_shuffle_bound`), quindi non e' informativo per i gruppi.

Uso:
    python scripts/privacy_di_gruppo.py
"""

from __future__ import annotations

import math

from dp_accounting import dp_event
from dp_accounting.rdp import rdp_privacy_accountant as rdp

# Ordini RDP: per il limite (B) servono alpha = alpha' * 2^c con alpha' >= 2;
# alpha' fino a 64 basta per delta = 1e-5, c fino a 10 copre k <= 1024.
_BASE = [1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32,
         48, 64]
ORDERS = sorted({b * 2 ** c for b in _BASE for c in range(0, 11)})


def curva_rdp(n: int, batch: int, epochs: int, rounds: int, sigma: float):
    """Ordini e valori RDP per un client con n record di training."""
    steps = (n // batch) * epochs * rounds
    q = batch / n
    acc = rdp.RdpAccountant(orders=ORDERS)
    acc.compose(dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(sigma)), steps)
    return list(acc._orders), list(acc._rdp)


def eps_da_rdp(orders, valori, delta: float) -> float:
    eps, _ = rdp.compute_epsilon(orders, valori, delta)
    return float(eps)


def limite_generico(orders, valori, k: int, delta: float) -> float:
    """Limite (A). math.inf se non esiste un delta per record adatto."""
    if k == 1:
        return eps_da_rdp(orders, valori, delta)
    migliore = math.inf
    for j in range(5, 301):
        d = 10.0 ** (-j)
        e = eps_da_rdp(orders, valori, d)
        if math.log(k) + (k - 1) * e + math.log(d) <= math.log(delta):
            migliore = min(migliore, k * e)
    return migliore


def limite_rdp_mironov(orders, valori, k: int, delta: float) -> float:
    """Limite (B)."""
    if k == 1:
        return eps_da_rdp(orders, valori, delta)
    c = math.ceil(math.log2(k))
    g = 2 ** c
    nuovi_ord, nuovi_val = [], []
    for a, r in zip(orders, valori):
        if a >= 2 ** (c + 1) and math.isfinite(r):
            nuovi_ord.append(a / g)
            nuovi_val.append((3 ** c) * r)
    if not nuovi_ord:
        return math.inf
    return eps_da_rdp(nuovi_ord, nuovi_val, delta)


# Solo ordini >= 2: sotto 2 l'integrazione numerica e l'accountant di
# dp-accounting differiscono fino al 25% (valori molto piccoli). Escluderli
# rende il limite piu' grande, cioe' prudente, mai piu' ottimista.
ORDINI_C = [2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128,
            192, 256]


def rdp_gruppo_poisson(q: float, sigma: float, k: int, ordini=ORDINI_C,
                       max_punti: int = 4_000_000) -> tuple[list[float], list[float]]:
    """(C) RDP per passo di un gruppo di k record con campionamento di Poisson.

    Ogni record del gruppo entra nel lotto con probabilita' q, indipendente;
    se j record del gruppo sono nel lotto la somma clippata si sposta al piu'
    di j*C. Caso peggiore (contributi allineati, adiacenza add/remove del
    gruppo): confronto fra la miscela P = sum_j Bin(k, q)(j) N(j, sigma^2) e
    Q = N(0, sigma^2), in unita' di C (coppia dominante della privacy di gruppo
    con campionamento di Poisson). Si calcolano D_alpha(P||Q) e D_alpha(Q||P)
    per integrazione numerica in 1D e si prende il massimo.

    Tutti i k+1 termini della miscela sono tenuti: anche quelli con peso
    minuscolo possono dominare la divergenza agli ordini alti. Gli integrandi
    hanno il picco vicino a x = a*j e x = -(a-1)*j, quindi la griglia deve
    coprire (a+1)*k; gli ordini per cui servirebbero piu' di max_punti punti
    sono saltati. Saltare ordini rende epsilon piu' grande, mai piu' piccolo.
    Restituisce (ordini usati, RDP per passo). Con k = 1 coincide con il
    gaussiano sottocampionato di dp-accounting.
    """
    import numpy as np
    from scipy.stats import binom

    js = np.arange(k + 1).astype(float)
    logw = binom.logpmf(np.arange(k + 1), k, q)
    jmax = float(k)
    c0 = np.log(sigma * np.sqrt(2 * np.pi))
    usati, out = [], []
    for a in ordini:
        lo = -(a + 1) * jmax - 40.0 * sigma
        hi = (a + 1) * jmax + 40.0 * sigma
        npt = int(max(200_001, (hi - lo) / (0.02 * sigma)))
        if npt > max_punti:
            continue
        x = np.linspace(lo, hi, npt)
        dx = x[1] - x[0]
        logq = -0.5 * (x / sigma) ** 2 - c0
        logp = np.full_like(x, -np.inf)
        for j, lw in zip(js, logw):
            if np.isfinite(lw):
                logp = np.logaddexp(logp, lw - 0.5 * ((x - j) / sigma) ** 2 - c0)

        def _div(la, lb):
            tt = a * la + (1 - a) * lb
            mm = tt.max()
            return (mm + np.log(np.exp(tt - mm).sum() * dx)) / (a - 1)

        usati.append(a)
        out.append(float(max(_div(logp, logq), _div(logq, logp), 0.0)))
    return usati, out


def eps_gruppo_poisson(n: int, batch: int, epochs: int, rounds: int,
                       sigma: float, k: int, delta: float = 1e-5) -> float:
    """(C) epsilon per un gruppo di k record su tutto il training."""
    steps = (n // batch) * epochs * rounds
    q = batch / n
    ordini, per_passo = rdp_gruppo_poisson(q, sigma, k)
    if not ordini:
        return math.inf
    return eps_da_rdp(ordini, [steps * r for r in per_passo], delta)


def eps_gruppo_da_curva(o, v, k, delta=1e-5):
    return min(limite_generico(o, v, k, delta), limite_rdp_mironov(o, v, k, delta))


def eps_gruppo(n, batch, epochs, rounds, sigma, k, delta=1e-5):
    o, v = curva_rdp(n, batch, epochs, rounds, sigma)
    return eps_gruppo_da_curva(o, v, k, delta)


def _fmt(x: float) -> str:
    if not math.isfinite(x):
        return "inf"
    return f"{x:.3g}" if x < 1000 else f"{x:.2e}"


def main() -> None:
    # Parametri delle run rq1-recorddp-nm1/nm2 (JSON del Mac principale):
    # B = 32, E = 50, T = 10, delta = 1e-5; n di training per client.
    # sigma = 5 non ha run salvate: stessi parametri, solo calcolo.
    clients = {"caltech": 25160, "jpl": 26875, "office1": 1335}
    ks = [1, 2, 5, 14, 61, 142]
    print("Limite (C), Poisson, adiacenza add/remove del gruppo, delta = 1e-5")
    for sigma in (1.0, 2.0, 5.0):
        print(f"\nsigma = {sigma}")
        print("client   | " + " | ".join(f"k={k}" for k in ks))
        for cid, n in clients.items():
            vals = [eps_gruppo_poisson(n, 32, 50, 10, sigma, k) for k in ks]
            print(f"{cid:8s} | " + " | ".join(_fmt(x) for x in vals))
    # Controllo: i limiti generici (A) e (B) sono sempre piu' grandi di (C).
    o, v = curva_rdp(25160, 32, 50, 10, 2.0)
    print("\ncontrollo caltech, sigma 2, k = 14: (A) "
          f"{_fmt(limite_generico(o, v, 14, 1e-5))}, (B) "
          f"{_fmt(limite_rdp_mironov(o, v, 14, 1e-5))}, (C) "
          f"{_fmt(eps_gruppo_poisson(25160, 32, 50, 10, 2.0, 14))}")


if __name__ == "__main__":
    main()
