"""
Re-identificazione ottimizzata su ACN-Data (2026-10-13).

Stesso scenario di `scripts/reidentificazione_acn.py` (mondo chiuso: storico
noto = primo 70% delle sessioni di ogni utente, da riconoscere = il 30%
successivo), con tre attaccanti piu' forti e iperparametri scelti SENZA
guardare il test:

  - kNN: vicini piu' prossimi pesati per distanza (numero di vicini e peso
    delle feature categoriche scelti in validazione);
  - profilo: classificatore bayesiano con una densita' per utente. Per ogni
    utente u, p(x | u) e' una stima a nucleo congiunta sulle sue sessioni note
    (gaussiana sulle feature continue standardizzate, larghezza b; per le
    categoriche probabilita' 1 - e se coincidono e e/(V-1) altrimenti; per la
    data una gaussiana di larghezza b_t giorni). Punteggio
    log p(x | u) + beta * log n_u (n_u = sessioni note di u);
  - fusione: profilo + lambda * log(probabilita' kNN).

Validazione temporale: dentro lo storico di ogni utente il primo 80% fa da
riferimento e l'ultimo 20% da validazione. Gli iperparametri si scelgono sul
top-1 di validazione; poi si riaddestra sullo storico intero e si misura sul
test una volta sola.

Quattro insiemi di feature: i tre di `reidentificazione_acn.py` piu'
"dati del client + data", che aggiunge il giorno della sessione (l'attaccante
vede la data: gli utenti sono attivi in periodi diversi).

Uso:
    python scripts/reidentificazione_ottimizzata.py [cartella_acn] [--rapido]
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reidentificazione_acn as base  # noqa: E402

CATEGORICHE = {"station_code", "weekday"}
INSIEMI = {
    **base.INSIEMI,
    "dati del client + data": base.INSIEMI["dati del client"] + ["day"],
}
SEED = 42


# ── preparazione delle matrici ─────────────────────────────────────────────────

class Spazio:
    """Feature continue standardizzate sul riferimento (la data a parte) e
    categoriche come codici interi."""

    def __init__(self, rif: pd.DataFrame, feats: list[str]):
        self.cont = [f for f in feats if f not in CATEGORICHE and f != "day"]
        self.cat = [f for f in feats if f in CATEGORICHE]
        self.usa_data = "day" in feats
        self.mu = {f: float(rif[f].mean()) for f in self.cont}
        self.sd = {f: (float(rif[f].std()) or 1.0) for f in self.cont}
        self.nv = {f: int(max(rif[f].max(), 0)) + 1 for f in self.cat}

    def trasforma(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {
            "cont": np.stack([((df[f].to_numpy(float) - self.mu[f]) / self.sd[f]) for f in self.cont],
                             axis=1).astype(np.float32) if self.cont else np.zeros((len(df), 0), np.float32),
            "cat": np.stack([df[f].to_numpy(np.int64) for f in self.cat], axis=1)
            if self.cat else np.zeros((len(df), 0), np.int64),
        }
        out["day"] = df["day"].to_numpy(np.float32) if self.usa_data else None
        return out


def _ordina_per_utente(rif: pd.DataFrame):
    """Riferimento ordinato per utente: inizi dei segmenti e utenti."""
    rif = rif.sort_values("user", kind="stable")
    utenti, inizi, conta = np.unique(rif["user"].to_numpy(), return_index=True, return_counts=True)
    return rif, utenti, inizi, conta


def punteggi_profilo(rif: pd.DataFrame, q: pd.DataFrame, feats: list[str], b: float,
                     eps: float, bt: float, beta: float, chunk: int = 256):
    """Matrice (query x utenti) dei punteggi del profilo bayesiano."""
    rif, utenti, inizi, conta = _ordina_per_utente(rif)
    sp = Spazio(rif, feats)
    R, Q = sp.trasforma(rif), sp.trasforma(q)
    lsame = {f: np.log(1 - eps) for f in sp.cat}
    ldiff = {f: np.log(eps / max(sp.nv[f] - 1, 1)) for f in sp.cat}
    out = np.empty((len(q), len(utenti)), np.float64)
    prior = beta * np.log(conta) - np.log(conta)  # log p(x|u) = lse - log n_u
    for s in range(0, len(q), chunk):
        e = min(s + chunk, len(q))
        lk = np.zeros((e - s, len(rif)), np.float64)
        if R["cont"].shape[1]:
            d2 = ((Q["cont"][s:e, None, :] - R["cont"][None, :, :]) ** 2).sum(-1)
            lk -= 0.5 * d2 / (b * b)
        for j, f in enumerate(sp.cat):
            uguali = Q["cat"][s:e, j][:, None] == R["cat"][:, j][None, :]
            lk += np.where(uguali, lsame[f], ldiff[f])
        if sp.usa_data:
            lk -= 0.5 * ((Q["day"][s:e, None] - R["day"][None, :]) / bt) ** 2
        m = lk.max(axis=1, keepdims=True)
        somme = np.add.reduceat(np.exp(lk - m), inizi, axis=1)
        with np.errstate(divide="ignore"):
            out[s:e] = np.log(somme) + m + prior[None, :]
    return out, utenti


def punteggi_knn(rif: pd.DataFrame, q: pd.DataFrame, feats: list[str], k: int, peso: float):
    """Matrice (query x utenti) delle probabilita' kNN pesate per distanza."""
    from sklearn.neighbors import KNeighborsClassifier

    base.PESO_CATEGORICHE = peso
    feats_knn = list(feats)
    xr, xq = base._matrice(rif, feats_knn, rif), base._matrice(q, feats_knn, rif)
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance", algorithm="brute")
    clf.fit(xr, rif["user"].to_numpy())
    pr = np.vstack([clf.predict_proba(xq[s:s + 2000]) for s in range(0, len(xq), 2000)])
    return pr, clf.classes_


def ranghi(score: np.ndarray, utenti: np.ndarray, vero: np.ndarray) -> np.ndarray:
    """Posizione (0 = primo) del vero utente; a parita' il caso peggiore."""
    idx = np.searchsorted(utenti, vero)
    sv = score[np.arange(len(vero)), idx]
    sopra = (score > sv[:, None]).sum(axis=1)
    pari = (score == sv[:, None]).sum(axis=1) - 1
    return sopra + pari


def metriche(pos: np.ndarray, n_utenti: int) -> dict:
    return {"top1": float((pos == 0).mean()), "top5": float((pos < 5).mean()),
            "top10": float((pos < 10).mean()), "rango_mediano": float(np.median(pos) + 1),
            "utenti": n_utenti, "query": int(len(pos))}


def _allinea(score_u, utenti_u, utenti):
    """Riordina le colonne di score_u sull'elenco `utenti`."""
    idx = np.searchsorted(utenti_u, utenti)
    return score_u[:, idx]


def valuta(rif, q, feats, par):
    """Punteggi fusi per un insieme di iperparametri."""
    sp, ut = punteggi_profilo(rif, q, feats, par["b"], par["eps"], par["bt"], par["beta"])
    if par["lam"] > 0:
        pk, uk = punteggi_knn(rif, q, feats, par["k"], par["peso"])
        sp = sp + par["lam"] * np.log(_allinea(pk, uk, ut) + 1e-6)
    return sp, ut


def griglia(feats: list[str], rapido: bool):
    bs = [0.15, 0.3, 0.5] if not rapido else [0.3]
    epss = [0.02, 0.1, 0.3] if not rapido else [0.1]
    bts = ([10.0, 30.0, 120.0] if "day" in feats else [1.0]) if not rapido else \
        ([30.0] if "day" in feats else [1.0])
    betas = [0.0, 1.0]
    for b, eps, bt, beta in itertools.product(bs, epss, bts, betas):
        yield {"b": b, "eps": eps, "bt": bt, "beta": beta}


def scegli(storico: pd.DataFrame, feats: list[str], rapido: bool) -> tuple[dict, dict]:
    """Validazione temporale dentro lo storico."""
    rif, val = [], []
    for _, g in storico.groupby("user", sort=False):
        k = max(1, int(round(len(g) * 0.8)))
        rif.append(g.iloc[:k])
        val.append(g.iloc[k:])
    rif, val = pd.concat(rif), pd.concat(val)
    val = val.sample(n=min(2500, len(val)), random_state=SEED)
    vero = val["user"].to_numpy()
    # 1) profilo
    migliore, top = None, -1.0
    for par in griglia(feats, rapido):
        sp, ut = punteggi_profilo(rif, val, feats, **par)
        t1 = metriche(ranghi(sp, ut, vero), len(ut))["top1"]
        if t1 > top:
            migliore, top = par, t1
    # 2) kNN
    mk, topk = None, -1.0
    for k, peso in itertools.product([10, 25, 50, 100] if not rapido else [50], [0.5, 1.0, 2.0] if not rapido else [1.0]):
        pk, uk = punteggi_knn(rif, val, feats, k, peso)
        t1 = metriche(ranghi(pk, uk, vero), len(uk))["top1"]
        if t1 > topk:
            mk, topk = {"k": k, "peso": peso}, t1
    # 3) fusione
    sp, ut = punteggi_profilo(rif, val, feats, **migliore)
    pk, uk = punteggi_knn(rif, val, feats, mk["k"], mk["peso"])
    lpk = np.log(_allinea(pk, uk, ut) + 1e-6)
    ml, topf = 0.0, -1.0
    for lam in [0.0, 0.25, 0.5, 1.0, 2.0]:
        t1 = metriche(ranghi(sp + lam * lpk, ut, vero), len(ut))["top1"]
        if t1 > topf:
            ml, topf = lam, t1
    par = {**migliore, **mk, "lam": ml}
    return par, {"validazione_top1_profilo": top, "validazione_top1_knn": topk,
                 "validazione_top1_fusione": topf}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rapido = "--rapido" in sys.argv
    cartella = Path(args[0]) if args else Path("datasets/acn")
    df = base.carica(cartella)
    storico, test = base.dividi(df)
    print(f"{storico['user'].nunique()} utenti, storico {len(storico)}, da riconoscere {len(test)}")
    righe = []
    for nome, feats in INSIEMI.items():
        st = storico.dropna(subset=feats)
        te = test.dropna(subset=feats)
        te = te[te["user"].isin(set(st["user"]))]
        par, val = scegli(st, feats, rapido)
        vero = te["user"].to_numpy()
        sp, ut = punteggi_profilo(st, te, feats, par["b"], par["eps"], par["bt"], par["beta"])
        pk, uk = punteggi_knn(st, te, feats, par["k"], par["peso"])
        r_prof = metriche(ranghi(sp, ut, vero), len(ut))
        r_knn = metriche(ranghi(pk, uk, vero), len(uk))
        r_fus = metriche(ranghi(sp + par["lam"] * np.log(_allinea(pk, uk, ut) + 1e-6), ut, vero), len(ut))
        # controllo: etichette dello storico mescolate
        st2 = st.copy()
        st2["user"] = np.random.default_rng(SEED).permutation(st2["user"].to_numpy())
        sp2, ut2 = punteggi_profilo(st2, te, feats, par["b"], par["eps"], par["bt"], par["beta"])
        r_mesc = metriche(ranghi(sp2, ut2, vero), len(ut2))
        print(f"\n[{nome}] parametri {par}; validazione {val}")
        for et, r in (("kNN", r_knn), ("profilo", r_prof), ("fusione", r_fus), ("mescolate", r_mesc)):
            print(f"  {et:9s} top-1 {r['top1']:.3f}  top-5 {r['top5']:.3f}  top-10 {r['top10']:.3f}  "
                  f"rango mediano {r['rango_mediano']:.0f}/{r['utenti']}")
        righe.append({"insieme": nome, **{f"{k}_{m}": v for k, r in
                                         (("knn", r_knn), ("profilo", r_prof), ("fusione", r_fus),
                                          ("mescolate", r_mesc)) for m, v in r.items()},
                      "parametri": par})
    print(f"\ncaso: top-1 {1 / storico['user'].nunique():.4f}, top-5 {5 / storico['user'].nunique():.4f}")
    out = Path("risultati")
    out.mkdir(exist_ok=True)
    pd.DataFrame(righe).to_csv(out / "reidentificazione_ottimizzata.csv", index=False)


if __name__ == "__main__":
    main()
