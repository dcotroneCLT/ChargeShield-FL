#!/usr/bin/env python3
"""Genera i tre config del controllo "solo taglio" della DP per soggetto
(2026-10-15) da config/experiment_canary_multisite.yaml.

Uso: python genera_config_subject_clip.py <radice del repository>
Scrive config/experiment_canary_multisite_gn.yaml,
       config/experiment_canary_multisite_subject_s0.yaml,
       config/experiment_canary_multisite_subject_s2.yaml.
Il corpo YAML resta quello del config di partenza, cambiano solo
experiment.name, ml.norm e (nei due bracci per soggetto) ml.subject_dp.
"""

from __future__ import annotations

import sys
from pathlib import Path

INTESTAZIONE = """\
# ChargeShield-FL — Canary persona nel regime indotto: {titolo} (2026-10-15)
#
# Derivato da experiment_canary_multisite.yaml (generato da
# genera_config_subject_clip.py): identico tranne experiment.name, ml.norm
# (group){extra}. Lo usa scripts/canary_persona.py (--config), che allena
# sempre con no_dp=True: la DP per soggetto si attiva solo da qui.
#
# SCOPO: controllo "solo taglio" della DP per soggetto, l'equivalente misurato
# del punto di osservazione A2 (docs/ESPERIMENTI.md, "DP per soggetto:
# controllo solo taglio"). Tre bracci con la stessa normalizzazione
# GroupNorm, richiesta dal taglio per soggetto: riferimento senza DP
# (experiment_canary_multisite_gn), solo taglio con sigma = 0
# (experiment_canary_multisite_subject_s0, nessuna garanzia formale) e DP
# per soggetto completa con sigma = 2 (experiment_canary_multisite_subject_s2,
# stessi parametri di experiment_acn_subjectdp.yaml).
#
# Soggetto = userID. Le sessioni senza userID sono soggetti singoli; le copie
# della persona costruita conservano il suo userID, quindi contano come un
# solo soggetto (al piu' 64 sessioni per passo).
#

"""

VARIANTI = {
    "gn": ("riferimento GroupNorm senza DP", "", None),
    "subject_s0": ("DP per soggetto, solo taglio (sigma = 0)",
                   " e ml.subject_dp con sigma = 0: taglio per soggetto,\n# nessun rumore", 0.0),
    "subject_s2": ("DP per soggetto completa (sigma = 2)",
                   " e ml.subject_dp con sigma = 2", 2.0),
}

BLOCCO_SOGGETTO = """\
  subject_dp:
    enabled: true
    subject_key: user_id
    max_grad_norm: 1.0
    noise_multiplier: {sigma}
    expected_subjects_per_step: 32
    max_records_per_subject: 64
"""


def main() -> None:
    radice = Path(sys.argv[1])
    sorgente = (radice / "config/experiment_canary_multisite.yaml").read_text(encoding="utf-8")
    corpo = sorgente[sorgente.index("experiment:\n"):]
    vecchio_nome = "  name: chargeshield_canary_multisite\n"
    ancora = "  latent_dim: 8\n"
    assert corpo.count(vecchio_nome) == 1 and corpo.count(ancora) == 1
    for suffisso, (titolo, extra, sigma) in VARIANTI.items():
        testo = corpo.replace(vecchio_nome, f"  name: chargeshield_canary_multisite_{suffisso}\n")
        aggiunta = "  norm: group\n"
        if sigma is not None:
            aggiunta += BLOCCO_SOGGETTO.format(sigma=sigma)
        testo = testo.replace(ancora, ancora + aggiunta)
        uscita = radice / f"config/experiment_canary_multisite_{suffisso}.yaml"
        uscita.write_text(INTESTAZIONE.format(titolo=titolo, extra=extra) + testo, encoding="utf-8")
        print(uscita)


if __name__ == "__main__":
    main()
