#!/usr/bin/env python3
"""
Conteggi di ChargePlace Scotland per council e per mese (2026-10-14).

RQ3 e costo di RQ1, replica ridotta su un secondo dataset (ESPERIMENTI.md,
sezione "Scozia: replica ridotta"). Serve a scegliere, con la regola fissata
prima di vedere i risultati, due council grandi di taglia simile e uno piccolo
intorno a un ventesimo dei grandi, sugli stessi mesi consecutivi.

Legge i file mensili con lo stesso adapter degli esperimenti
(src/adapters/chargeplace_scotland_adapter.py), quindi i conteggi sono quelli
che il codice carica: stesse sessioni scartate, stesso fuso (Europe/London),
stessa durata (le durate oltre 24 ore arrivano come stringa e non vengono
troncate). Le sessioni senza council (CPID assente dai metadati) sono contate
a parte e scartate, come fa il filtro dei client.

Uscite in --out:
  council_mese.csv   una riga per council e mese: sessioni, punti di ricarica,
                     quota del punto piu' usato, energia, durata e ora mediane,
                     quota di giorni feriali, sessioni oltre 24 ore, quota DC
  council_totali.csv una riga per council sui 18 mesi, con la classe
                     urbano-rurale (UR8, 1 = grande citta', 8 = rurale remoto)
  mesi.csv           sessioni per mese, totale e senza council

Uso:  python scripts/conteggi_scozia.py --dati datasets/alt/chargeplace_scotland --out risultati/scozia_conteggi
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adapters.chargeplace_scotland_adapter import ChargePlaceScotlandDataset  # noqa: E402

LONDRA = ZoneInfo("Europe/London")
UTC = ZoneInfo("UTC")


def sessioni_di(dati: Path) -> pd.DataFrame:
    ds = ChargePlaceScotlandDataset()
    ds.load_metadata(str(dati / "CPID_information"))
    righe = []
    for f in sorted((dati / "Sessions_from_CPS").glob("*.xlsx")):
        rec = ds._parse_file(str(f))  # stesso parsing di load_multiple()
        print(f"{f.name}: {len(rec)} sessioni", flush=True)
        for s in rec:
            inizio = datetime.fromisoformat(s["start_time"]).replace(tzinfo=UTC).astimezone(LONDRA)
            fine = datetime.fromisoformat(s["end_time"])
            durata = (fine - datetime.fromisoformat(s["start_time"])).total_seconds() / 3600
            righe.append((f.name, s["site_id"], s["node_id"], inizio.strftime("%Y-%m"), inizio.hour,
                          inizio.weekday() < 5, s["total_energy_kwh"], durata, s["charging_mode"]))
    return pd.DataFrame(righe, columns=["file", "council", "cpid", "mese", "ora", "feriale",
                                        "kwh", "durata_h", "tipo"])


def riassunto(g: pd.DataFrame) -> pd.Series:
    per_punto = g["cpid"].value_counts()
    return pd.Series({
        "sessioni": len(g),
        "punti": per_punto.size,
        "quota_punto_max": round(per_punto.iloc[0] / len(g), 3),
        "kwh_mediana": round(g["kwh"].median(), 2),
        "durata_h_mediana": round(g["durata_h"].median(), 2),
        "ora_mediana": g["ora"].median(),
        "quota_serale_18_24": round((g["ora"] >= 18).mean(), 3),
        "quota_feriali": round(g["feriale"].mean(), 3),
        "oltre_24h": int((g["durata_h"] > 24).sum()),
        "quota_dc": round((g["tipo"].astype(str).str.upper() != "AC").mean(), 3),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dati", type=Path, default=ROOT / "datasets/alt/chargeplace_scotland")
    ap.add_argument("--out", type=Path, default=ROOT / "risultati/scozia_conteggi")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    df = sessioni_di(a.dati)
    mesi = df.groupby("mese").agg(sessioni=("cpid", "size"),
                                  senza_council=("council", lambda c: int((c == "").sum())))
    mesi.to_csv(a.out / "mesi.csv")

    con = df[df["council"] != ""]
    cm = con.groupby(["council", "mese"]).apply(riassunto, include_groups=False).reset_index()
    cm.to_csv(a.out / "council_mese.csv", index=False)

    ur = pd.read_excel(a.dati / "CPID_information" / "CPID_and_UR_Classification.xlsx")
    ur["CPID"] = ur["CPID"].astype(str)
    classe = con.drop_duplicates("cpid").merge(ur, left_on="cpid", right_on="CPID", how="left")
    ur_med = classe.groupby("council")["UR8Class"].median()
    tot = con.groupby("council").apply(riassunto, include_groups=False)
    tot["ur8_mediana_punti"] = ur_med
    tot.sort_values("sessioni", ascending=False).to_csv(a.out / "council_totali.csv")
    print(f"{len(df)} sessioni, {int((df['council'] == '').sum())} senza council -> {a.out}")


if __name__ == "__main__":
    main()
