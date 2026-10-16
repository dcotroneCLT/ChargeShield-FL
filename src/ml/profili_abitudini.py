"""
Soggetti ricostruiti dalle abitudini per le sessioni senza userID
(2026-10-16, Segnalazioni R voce 81; decisione della dottoranda).

Problema. Nella DP per soggetto una sessione ACN senza userID era un soggetto
a se' stante: per una persona senza userID la protezione valeva per sessione,
e il numero di soggetti (gonfiato dai singoli) abbassava l'epsilon dichiarato.

Idea. Le sessioni della stessa persona si somigliano (stesso gruppo di
colonnine, stessa ora di arrivo, giorni feriali, energia e durata simili). Si
raggruppano in PROFILI di abitudine e ogni profilo e' un soggetto: tutte le
sue sessioni sono protette insieme.

Perche' i profili si imparano FUORI dal training. Se il soggetto di una
sessione dipendesse dalle altre sessioni del training (per esempio un
clustering fatto sul training), aggiungere o togliere una persona cambierebbe
i soggetti degli altri e l'analisi di sensibilita' non varrebbe piu'. Qui il
modello dei profili (centroidi per sito) si stima su dati esterni al training
(di default i non membri, `holdout`) e poi ogni sessione va al profilo piu'
vicino guardando SOLO se stessa: la funzione sessione -> soggetto e' fissata
prima di vedere il training.

Garanzia per una persona vera. Il meccanismo protegge il profilo. Una persona
le cui sessioni cadono in k profili e' protetta per privacy di gruppo su k
soggetti. k si misura sulle persone con userID (scripts/misura_profili.py) e va
riportato insieme all'epsilon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

CAMPO = "_profilo"
FEATURE_NUMERICHE = ("total_energy_kwh", "duration_hours", "kwh_requested", "minutes_available")


def _giorno_feriale(s: dict[str, Any]) -> float:
    """1 se la sessione inizia in un giorno feriale (ora locale), 0 altrimenti,
    0,5 se non ricavabile."""
    try:
        t = datetime.fromisoformat(str(s["start_time"]))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        tz = s.get("timezone")
        if tz:
            from zoneinfo import ZoneInfo
            t = t.astimezone(ZoneInfo(str(tz)))
        return 1.0 if t.weekday() < 5 else 0.0
    except Exception:  # noqa: BLE001
        return 0.5


def _gruppo(s: dict[str, Any]) -> str:
    return str(s.get("cluster_id") or s.get("node_id") or "")


def vettore(s: dict[str, Any], gruppi: list[str], peso_gruppo: float = 1.0) -> np.ndarray:
    """Vettore di abitudine di una sessione (feature gia' normalizzate in [0, 1])."""
    ora = float(s.get("hour_of_day") or 0.0)
    # hour_of_day deve essere gia' normalizzato in [0, 1] (limiti pubblici)
    if not (-1e-9 <= ora <= 1.0 + 1e-9):
        raise ValueError(f"hour_of_day={ora}: i profili vogliono feature normalizzate in [0, 1]")
    ang = 2 * math.pi * ora
    v = [math.sin(ang), math.cos(ang), _giorno_feriale(s)]
    for f in FEATURE_NUMERICHE:
        x = s.get(f)
        try:
            x = float(x)
        except (TypeError, ValueError):
            x = 0.0
        v.append(x if math.isfinite(x) else 0.0)
    oh = [0.0] * len(gruppi)
    g = _gruppo(s)
    if g in gruppi:
        oh[gruppi.index(g)] = peso_gruppo
    return np.array(v + oh, dtype=float)


@dataclass
class ModelloProfili:
    """Centroidi per sito, stimati su dati esterni al training."""
    per_sito: dict[str, dict[str, Any]]   # sito -> {"gruppi": [...], "centroidi": array}
    dimensione_media: float
    peso_gruppo: float
    fonte: str

    def profilo(self, s: dict[str, Any]) -> str | None:
        sito = str(s.get("site_id") or "")
        m = self.per_sito.get(sito)
        if m is None:
            return None
        x = vettore(s, m["gruppi"], self.peso_gruppo)
        k = int(np.argmin(((m["centroidi"] - x) ** 2).sum(axis=1)))
        return f"{sito}:{k}"

    def descrizione(self) -> dict[str, Any]:
        return {"fonte": self.fonte, "dimensione_media": self.dimensione_media,
                "peso_gruppo": self.peso_gruppo,
                "profili_per_sito": {s: int(m["centroidi"].shape[0]) for s, m in self.per_sito.items()}}


def addestra(sessioni_esterne: list[dict[str, Any]], dimensione_media: float = 10.0,
             peso_gruppo: float = 1.0, seed: int = 0, fonte: str = "holdout",
             n_training_per_sito: dict[str, int] | None = None) -> ModelloProfili:
    """k-medie per sito sulle sessioni esterne. Con `n_training_per_sito`
    (numero di sessioni di training del sito, pubblico con l'adiacenza per
    sostituzione) k = n_training / dimensione_media, cosi' un profilo ha in
    media `dimensione_media` sessioni di TRAINING (revisione 2026-10-16);
    senza, k = sessioni esterne / dimensione_media. k e' fra 1 e il numero di
    sessioni esterne del sito."""
    from sklearn.cluster import KMeans
    per_sito: dict[str, list[dict[str, Any]]] = {}
    for s in sessioni_esterne:
        per_sito.setdefault(str(s.get("site_id") or ""), []).append(s)
    out: dict[str, dict[str, Any]] = {}
    for sito, ss in sorted(per_sito.items()):
        gruppi = sorted({_gruppo(s) for s in ss})
        X = np.stack([vettore(s, gruppi, peso_gruppo) for s in ss])
        base = (n_training_per_sito or {}).get(sito, len(ss))
        k = max(1, min(len(ss), int(round(base / float(dimensione_media)))))
        km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(X)
        out[sito] = {"gruppi": gruppi, "centroidi": km.cluster_centers_}
    return ModelloProfili(out, float(dimensione_media), float(peso_gruppo), fonte)


def assegna(modello: ModelloProfili, sessioni: list[dict[str, Any]]) -> None:
    """Scrive il profilo di ogni sessione nel campo `_profilo` (in place).
    Una sessione di un sito senza profili e' un errore, non un soggetto
    singolo silenzioso (revisione 2026-10-16)."""
    for s in sessioni:
        p = modello.profilo(s)
        if p is None:
            raise ValueError(f"nessun profilo per il sito {s.get('site_id')!r}: "
                             "la riserva non contiene sessioni di quel sito")
        s[CAMPO] = p
