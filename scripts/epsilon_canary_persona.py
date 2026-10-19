"""
Epsilon della persona canary nelle run di scripts/canary_persona.py con DP per
soggetto (2026-10-18, controllo aggiuntivo v2b).

canary_persona.py non scrive l'epsilon. Questo script lo ricava dopo, senza
toccare il codice che ha prodotto i risultati: legge dai log della coda le
righe "[SUBJECT-DP] [<client>] round r: n soggetti, ..., sigma=..." del client
del sito canary (campo `sito` del JSON del braccio), ricalcola q e i passi per
epoca con ml.subject_dp.sampling_schedule e compone gli eventi con
l'accountant di ml.subject_dp (sostituzione, PLD).

Le persone del pool hanno sessioni solo nel sito canary (canary_persona.py le
toglie dallo sfondo in tutti i siti), quindi peso 1 e un solo client.

Uso, dalla cartella ChargeShield_offline:
  ambiente/bin/python codice/scripts/epsilon_canary_persona.py \
      --log log --esiti esiti/canary_persona --out risultati/epsilon_canary_persona.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI.parent / "src"))

from ml import subject_dp  # noqa: E402

RIGA = re.compile(r"\[SUBJECT-DP\] \[(?P<nodo>[^\]]+)\] round (?P<r>\d+): (?P<n>\d+) soggetti, "
                  r"q=[^,]+, (?P<pe>\d+) passi per epoca, sigma=(?P<s>[0-9.eE+-]+)")
FILE_BRACCIO = re.compile(r"(canary_persona_\S+_[AB]_s\d+\.json)")


def analizza(log: Path, esiti: Path, per_step: int, delta: float) -> dict | None:
    testo = log.read_text(encoding="utf-8", errors="replace")
    nomi = FILE_BRACCIO.findall(testo)
    if not nomi:
        return None
    nome = nomi[-1]
    if not (esiti / nome).exists():
        print(f"attenzione: {log.name} cita {nome}, che non e' in {esiti}: saltato")
        return None
    j = json.loads((esiti / nome).read_text())
    sito, epoche = str(j["sito"]), int(j["epochs"])
    per_round: dict[int, tuple[int, int, float]] = {}
    for m in RIGA.finditer(testo):
        if m["nodo"].split("-")[0] != sito:
            continue
        # se il log contiene piu' tentativi, vale l'ultimo
        per_round[int(m["r"])] = (int(m["n"]), int(m["pe"]), float(m["s"]))
    base = {"log": log.name, "file": nome, "sito": sito, "epoche": epoche,
            "round": int(j["rounds"]), "delta": delta}
    if not per_round:
        return {**base, "sigma": "", "soggetti": "", "passi": 0, "epsilon": "senza DP per soggetto"}
    if len(per_round) != int(j["rounds"]):
        raise SystemExit(f"{log.name}: righe SUBJECT-DP per {sorted(per_round)} round, "
                         f"il JSON ne dichiara {j['rounds']}")
    sigma = {s for _, _, s in per_round.values()}
    if len(sigma) != 1:
        raise SystemExit(f"{log.name}: sigma diversi fra i round {sorted(sigma)}")
    sigma = sigma.pop()
    passi = 0
    acc = subject_dp.accountant("sostituzione")
    for r in sorted(per_round):
        n, pe, _ = per_round[r]
        q, pe2 = subject_dp.sampling_schedule(n, per_step)
        if pe2 != pe:
            raise SystemExit(f"{log.name}: round {r}, passi per epoca {pe} nel log, {pe2} ricalcolati "
                             f"(--soggetti-per-passo diverso dal config?)")
        if sigma > 0:
            acc.compose(subject_dp._event(q, sigma), pe * epoche)
        passi += pe * epoche
    eps = float(acc.get_epsilon(target_delta=delta)) if sigma > 0 else math.inf
    return {**base, "sigma": sigma, "soggetti": per_round[max(per_round)][0], "passi": passi,
            "epsilon": round(eps, 3) if math.isfinite(eps) else "inf"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="log")
    ap.add_argument("--esiti", default="esiti/canary_persona")
    ap.add_argument("--out", default="risultati/epsilon_canary_persona.csv")
    ap.add_argument("--soggetti-per-passo", type=int, default=32,
                    help="expected_subjects_per_step dei config canary")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--filtro", default="cps", help="solo i log il cui nome contiene questa stringa")
    a = ap.parse_args()
    righe = []
    for log in sorted(Path(a.log).glob("*.log")):
        if a.filtro not in log.name:
            continue
        r = analizza(log, Path(a.esiti), a.soggetti_per_passo, a.delta)
        if r:
            righe.append(r)
    if not righe:
        raise SystemExit("nessun log di canary_persona trovato")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(righe[0]))
        w.writeheader()
        w.writerows(righe)
    for r in righe:
        print(f"{r['log']:45s} sigma={r['sigma']!s:4s} epoche={r['epoche']:5d} eps={r['epsilon']}")
    print(f"{len(righe)} righe -> {out}")


if __name__ == "__main__":
    main()
