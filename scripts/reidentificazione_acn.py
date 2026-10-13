"""
Attacco di re-identificazione su ACN-Data (2026-10-13).

Domanda: senza un identificativo utente, si possono ricondurre le sessioni di
ricarica alla stessa persona? Su ACN l'userID da' la risposta vera, quindi
l'attacco si puo' misurare.

Scenario (identificazione a mondo chiuso): l'attaccante conosce una parte
delle sessioni di ogni utente (per esempio le ha osservate o collegate in
passato) e riceve sessioni nuove senza identificativo. Deve dire di chi sono.
Metodo: vicini piu' prossimi (50, pesati per distanza) sullo storico, con le
feature continue standardizzate e quelle categoriche in one-hot.
Per ogni utente con almeno MIN_SESSIONI sessioni: in ordine di tempo, il primo
70% e' lo storico noto, il restante 30% sono le sessioni da riconoscere. Si misura in quanti casi
il vero utente e' il primo (top-1) o fra i primi 5 (top-5) indicati. Il caso
vale 1/U e 5/U con U utenti; si riporta anche la frequenza dell'utente piu'
comune fra le sessioni di test.

Tre insiemi di feature, dal piu' povero al piu' ricco:
  - "come Scotland": energia, potenza media, ora di arrivo, durata (le 4
    feature del config Scotland);
  - "feature del modello": le 6 storiche del modello ACN (le 4 sopra piu'
    kWh richiesti e minuti disponibili, che l'utente inserisce nell'app);
  - "dati del client": le 6 del modello piu' colonnina e giorno della
    settimana, cioe' quello che il gestore del sito ha in chiaro.

Uso:
    python scripts/reidentificazione_acn.py [cartella_acn]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_SESSIONI = 10
QUOTA_STORICO = 0.7
SEED = 42

INSIEMI = {
    "come Scotland": ["total_energy_kwh", "max_power_kw", "hour_sin", "hour_cos", "duration_hours"],
    "feature del modello": ["total_energy_kwh", "max_power_kw", "hour_sin", "hour_cos",
                            "duration_hours", "kwh_requested", "minutes_available"],
    "dati del client": ["total_energy_kwh", "max_power_kw", "hour_sin", "hour_cos",
                        "duration_hours", "kwh_requested", "minutes_available",
                        "station_code", "weekday"],
}


def carica(cartella: Path, solo_con_id: bool = True) -> pd.DataFrame:
    """Sessioni ACN con le feature dell'attacco. solo_con_id=False tiene anche
    le sessioni senza userID (colonna user = None)."""
    righe = []
    for f in sorted(glob.glob(str(cartella / "*" / "acndata_sessions_*.json"))):
        for s in json.load(open(f)).get("_items", []):
            uid = s.get("userID")
            if uid in (None, "") and solo_con_id:
                continue
            ui = (s.get("userInputs") or [{}])[-1] or {}
            righe.append({
                "session_id": s.get("sessionID"),
                "user": (None if uid in (None, "") else
                         str(int(uid)) if str(uid).isdigit() else str(uid)),
                "site": s.get("siteID"), "station": s.get("stationID"),
                "start": s.get("connectionTime"), "end": s.get("disconnectTime"),
                "done": s.get("doneChargingTime"), "kwh": s.get("kWhDelivered"),
                "kwh_requested": ui.get("kWhRequested"), "minutes_available": ui.get("minutesAvailable"),
            })
    df = pd.DataFrame(righe)
    for c in ("start", "end", "done"):
        df[c] = pd.to_datetime(df[c], format="%a, %d %b %Y %H:%M:%S GMT", utc=True, errors="coerce")
    df = df.dropna(subset=["start", "end", "kwh"]).copy()
    loc = df["start"].dt.tz_convert("America/Los_Angeles")
    ora = loc.dt.hour + loc.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * ora / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ora / 24)
    df["weekday"] = loc.dt.weekday
    df["duration_hours"] = (df["end"] - df["start"]).dt.total_seconds() / 3600.0
    carica_h = ((df["done"].fillna(df["end"]) - df["start"]).dt.total_seconds() / 3600.0).clip(lower=1 / 60)
    df["total_energy_kwh"] = df["kwh"].astype(float)
    df["max_power_kw"] = df["total_energy_kwh"] / carica_h
    df["station_code"] = df["station"].astype("category").cat.codes
    # giorni dall'inizio dei dati: l'attaccante vede la data della sessione
    df["day"] = (df["start"] - df["start"].min()).dt.total_seconds() / 86400.0
    for c in ("kwh_requested", "minutes_available"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("start").reset_index(drop=True)


def dividi(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    conta = df["user"].value_counts()
    df = df[df["user"].isin(conta[conta >= MIN_SESSIONI].index)]
    storico, test = [], []
    for _, g in df.groupby("user", sort=False):
        k = int(round(len(g) * QUOTA_STORICO))
        storico.append(g.iloc[:k])
        test.append(g.iloc[k:])
    return pd.concat(storico), pd.concat(test)


CATEGORICHE = {"station_code", "weekday"}
PESO_CATEGORICHE = 1.0  # scelto fra 0.5, 1, 2, 3 sugli stessi dati (lieve ottimismo, da dichiarare)


def _matrice(df: pd.DataFrame, feats: list[str], rif: pd.DataFrame) -> np.ndarray:
    """Continue standardizzate sullo storico; categoriche in one-hot pesato."""
    blocchi = []
    for f in feats:
        if f in CATEGORICHE:
            valori = sorted(rif[f].unique())
            oh = (df[f].to_numpy()[:, None] == np.array(valori)[None, :]).astype(np.float32)
            blocchi.append(oh * PESO_CATEGORICHE)
        else:
            mu, sd = float(rif[f].mean()), float(rif[f].std()) or 1.0
            blocchi.append(((df[f].to_numpy(dtype=np.float32) - mu) / sd)[:, None])
    return np.hstack(blocchi).astype(np.float32)


def attacco(storico: pd.DataFrame, test: pd.DataFrame, feats: list[str],
            vicini: int = 50) -> dict:
    """Vicini piu' prossimi pesati per distanza: ogni sessione nuova vota per
    gli utenti delle sessioni dello storico piu' simili. Gli utenti senza voti
    sono messi in fondo (rango peggiore possibile)."""
    from sklearn.neighbors import KNeighborsClassifier

    tr = storico.dropna(subset=feats)
    te = test.dropna(subset=feats)
    te = te[te["user"].isin(set(tr["user"]))]
    xtr, xte = _matrice(tr, feats, tr), _matrice(te, feats, tr)
    clf = KNeighborsClassifier(n_neighbors=vicini, weights="distance", algorithm="brute")
    clf.fit(xtr, tr["user"].to_numpy())
    classi = clf.classes_
    vero = te["user"].to_numpy()
    idx_vero = np.searchsorted(classi, vero)
    pos = np.empty(len(te), dtype=np.int64)
    for s in range(0, len(te), 2000):
        pr = clf.predict_proba(xte[s:s + 2000])
        pv = pr[np.arange(len(pr)), idx_vero[s:s + 2000]]
        piu_alti = (pr > pv[:, None]).sum(axis=1)
        # a parita' di punteggio (anche zero) si conta il caso peggiore
        pari = (pr == pv[:, None]).sum(axis=1) - 1
        pos[s:s + 2000] = piu_alti + np.where(pv > 0, 0, pari)
    n_utenti = len(classi)
    return {
        "utenti": n_utenti,
        "sessioni_test": len(te),
        "top1": float((pos == 0).mean()),
        "top5": float((pos < 5).mean()),
        "caso_top1": 1.0 / n_utenti,
        "caso_top5": min(1.0, 5.0 / n_utenti),
        "utente_piu_frequente": float(te["user"].value_counts(normalize=True).iloc[0]),
        "rango_mediano": float(np.median(pos) + 1),
    }


def main() -> None:
    cartella = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets/acn")
    df = carica(cartella)
    storico, test = dividi(df)
    print(f"Sessioni con userID: {len(df)}; utenti con >= {MIN_SESSIONI} sessioni: "
          f"{storico['user'].nunique()}; storico {len(storico)}, da riconoscere {len(test)}")
    for nome, feats in INSIEMI.items():
        r = attacco(storico, test, feats)
        print(f"\n[{nome}] {r['utenti']} utenti, {r['sessioni_test']} sessioni da riconoscere")
        print(f"  top-1 {r['top1']:.3f} (caso {r['caso_top1']:.4f}, utente piu' frequente "
              f"{r['utente_piu_frequente']:.3f})")
        print(f"  top-5 {r['top5']:.3f} (caso {r['caso_top5']:.4f}); rango mediano del vero utente "
              f"{r['rango_mediano']:.0f} su {r['utenti']}")


if __name__ == "__main__":
    main()
