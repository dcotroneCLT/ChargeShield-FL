#!/usr/bin/env python3
"""
Code della campagna v2 (2026-10-16), dopo la revisione profonda (docs/
Segnalazioni_tecniche_2026-09-22.md, sezioni R e S; docs/ESPERIMENTI.md,
"Revisione del 16 ottobre").

Cosa cambia rispetto alle code del 13-15 ottobre:
- codice corretto: LiRA e RMIA v2, GroupNorm, DP per record con Poisson, sigma
  analitico, adiacenza per sostituzione, limiti pubblici, profili di abitudine;
- LiRA completa: 32 shadow, epoche del client (niente --shadow-epochs-cap),
  shadow in processi (--lira-workers), un thread (--threads 1);
- inizializzazione comune dei client (--common-init) in tutte le run FL;
- DP per soggetto: sigma = 5 cella principale (5 seed), sigma = 2 secondaria.

Ogni riga: id, priorita (1 = prima), ore (stima sul Mac della macchina, con i
worker indicati; servono a ordinare e a stimare la fine), gb, thread (core
occupati: worker + 1), dopo, comando. Una run alla volta per macchina
(esegui_coda.py --max-run 1): i core li usano i worker di LiRA.

Le stime vengono dai tempi misurati il 16 ottobre (un thread, 50 epoche, client
di 25 000 sessioni: 2,1 minuti senza DP, 6,2 con DP per record, 1,1 con DP per
soggetto e profili, su una macchina cloud circa 3 volte piu' lenta di un Mac M)
e vanno corrette con la prima run cronometrata di ogni macchina.

Uso: python genera_coda_v2.py [--workers-mac 10 --workers-domenico 14 --workers-windows 6]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

QUI = Path(__file__).resolve().parent
SEEDS5 = (42, 123, 456, 789, 1234)
SEEDS3 = (42, 123, 456)


class Coda:
    def __init__(self, nome: str, workers: int):
        self.nome, self.w, self.righe = nome, workers, []

    def run(self, id_, prio, ore_1w, gb, cfg, flags, seed, skip="shadow", dopo="", lira=True):
        """ore_1w: ore stimate con un solo worker; con LiRA si dividono per i worker."""
        attacchi = (f"--n-shadow 32 --lira-workers {self.w}" if lira and "lira" not in skip
                    else "")
        ore = ore_1w / self.w if attacchi else ore_1w
        self.righe.append({
            "id": id_, "priorita": prio, "ore": round(max(ore, 0.05), 2), "gb": gb,
            "thread": (self.w + 1) if attacchi else 1, "dopo": dopo,
            "comando": (f"{{PY}} scripts/run_experiments.py --config config/{cfg}.yaml {flags} "
                        f"--seed {seed} --common-init --threads 1 {attacchi} --skip-attacks {skip} "
                        f"--sweep-dir {{ESITI}}/{id_}").replace("  ", " "),
        })

    def cmd(self, id_, prio, ore, gb, comando, dopo="", thread=1):
        self.righe.append({"id": id_, "priorita": prio, "ore": ore, "gb": gb, "thread": thread,
                           "dopo": dopo, "comando": comando})

    def scrivi(self, cartella: Path) -> None:
        with open(cartella / self.nome, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "priorita", "ore", "gb", "thread", "dopo", "comando"])
            w.writeheader()
            w.writerows(self.righe)


def costruisci(w_mac: int, w_dom: int, w_win: int) -> list[Coda]:
    TEST = ("{PY} -m pytest tests/test_lira2.py tests/test_record_dp_poisson.py "
            "tests/test_client_dp_accounting.py tests/test_subject_dp.py tests/test_profili_abitudini.py -q")

    # ── Mac di Susy: blocco ACN con attacchi (RQ1, RQ3, DP per record e per soggetto, persona)
    mac = Coda("coda_v2_mac.csv", w_mac)
    mac.cmd("test_codice", 0, 0.2, 2, TEST)
    # prova cronometrata: la prima cella, le altre dopo
    mac.run("v2_acn_nodp_mu0.01_s42", 0, 9, 4, "experiment", "--no-dp", 42, dopo="test_codice")
    PROVA = "v2_acn_nodp_mu0.01_s42"
    celle = [  # nome, config, flag, ore con 1 worker
        ("nodp_mu0.01", "experiment", "--no-dp", 9),
        ("nodp_mu0", "experiment_rq3_mu0", "--no-dp", 9),
        ("eps64_mu0.01", "experiment_rq1_eps64", "--dp-mode local", 9),
        ("eps64_mu0", "experiment_rq3_mu0_eps64", "--dp-mode local", 9),
    ]
    for seed in SEEDS5:
        for nome, cfg, flags, ore in celle:
            id_ = f"v2_acn_{nome}_s{seed}"
            if id_ != PROVA:
                mac.run(id_, 1, ore, 4, cfg, flags, seed, dopo=PROVA)
    # DP per soggetto con profili: sigma 5 principale, sigma 2 secondaria
    for seed in SEEDS5:
        mac.run(f"v2_acn_subject_s5_s{seed}", 1, 4.5, 4, "experiment_acn_subjectdp_s5", "--no-dp", seed, dopo=PROVA)
    for seed in SEEDS3:
        mac.run(f"v2_acn_subject_s2_s{seed}", 2, 4.5, 4, "experiment_acn_subjectdp", "--no-dp", seed, dopo=PROVA)
    # DP per record con Poisson: sigma 1 e 2 (principali), 0 (solo taglio) e 5
    for nm, prio in (("nm1", 1), ("nm2", 1), ("nm0", 2), ("nm5", 2)):
        for seed in SEEDS3:
            mac.run(f"v2_acn_record_{nm}_s{seed}", prio, 27, 4, f"experiment_rq1_recorddp_{nm}",
                    "--no-dp", seed, dopo=PROVA)
    # riferimento GroupNorm senza DP: con GroupNorm ovunque coincide con nodp_mu0.01
    # central al punto operativo, FedProx
    for seed in SEEDS5:
        mac.run(f"v2_acn_central_eps64_s{seed}", 2, 9, 4, "experiment_rq1_eps64", "--dp-mode central",
                seed, dopo=PROVA)
    # curva in epsilon (RQ1, utilita'): sola utilita', FedProx, rumore lato client
    for e in (2, 4, 8, 16, 32, 256):
        for seed in SEEDS5:
            mac.run(f"v2_acn_eps{e}_s{seed}", 1, 0.25, 3, f"experiment_rq1_eps{e}", "--dp-mode local",
                    seed, skip="lira,shadow", dopo="test_codice", lira=False)
    # MIA per persona: riferimento con lo stesso meccanismo, quindi niente run di solo riferimento
    for seed in SEEDS5:
        for mec, prio, ore in (("nodp", 1, 0.8), ("client_eps64", 1, 0.8), ("subject_s5", 1, 0.8),
                               ("subject_s2", 2, 0.8), ("record_s2", 2, 3.0), ("record_s5", 2, 3.0)):
            mac.cmd(f"v2_persona_{mec}_s{seed}", prio, ore, 3,
                    f"{{PY}} scripts/mia_persona.py --seed {seed} --meccanismo {mec} "
                    f"--out {{ESITI}}/mia_persona", dopo="test_codice")
    mac.cmd("v2_privacy_di_gruppo", 3, 0.3, 2, "{PY} scripts/privacy_di_gruppo.py")
    mac.cmd("v2_misura_profili_s42", 1, 0.3, 3,
            "{PY} scripts/misura_profili.py --seed 42 --dimensioni 20,50,100,200,500 "
            "--out {ESITI}/misura_profili_s42.json")

    # ── Mac di Domenico: Scozia ridotta, controllo a quattro feature, curva di capacita',
    #    controllo solo taglio della DP per soggetto (canary persona)
    dom = Coda("coda_v2_domenico.csv", w_dom)
    dom.cmd("test_codice", 0, 0.2, 2, TEST)
    dom.cmd("v2_sc3_prova_s42", 0, 0.3, 4,
            "{PY} scripts/run_experiments.py --config config/experiment_sc3.yaml --no-dp --seed 42 "
            "--common-init --threads 1 --skip-attacks lira,shadow --sweep-dir {ESITI}/v2_sc3_prova_s42",
            dopo="test_codice")
    celle_sc3 = [
        ("nodp_mu0.01", "experiment_sc3", "--no-dp", True),
        ("nodp_mu0", "experiment_sc3_mu0", "--no-dp", True),
        ("eps64_mu0.01", "experiment_sc3_eps64", "--dp-mode local", True),
        ("eps64_mu0", "experiment_sc3_mu0_eps64", "--dp-mode local", True),
        ("clip_mu0.01", "experiment_sc3_eps64", "--dp-mode local --dp-ablation clip-only", False),
        ("clip_mu0", "experiment_sc3_mu0_eps64", "--dp-mode local --dp-ablation clip-only", False),
        ("noise_mu0.01", "experiment_sc3_eps64", "--dp-mode local --dp-ablation noise-only", False),
        ("noise_mu0", "experiment_sc3_mu0_eps64", "--dp-mode local --dp-ablation noise-only", False),
        ("eps64_mu0.001", "experiment_sc3_mu0.001_eps64", "--dp-mode local", False),
        ("eps64_mu0.1", "experiment_sc3_mu0.1_eps64", "--dp-mode local", False),
        ("record_s1_mu0.01", "experiment_sc3_recorddp", "--no-dp", False),
        ("subject_s5_mu0.01", "experiment_sc3_subjectdp", "--no-dp", False),
    ]
    for seed in SEEDS5:
        for nome, cfg, flags, attacchi in celle_sc3:
            if attacchi:
                dom.run(f"v2_sc3_{nome}_s{seed}", 1, 6, 5, cfg, flags, seed, dopo="v2_sc3_prova_s42")
            else:
                ore = 2.0 if "record" in nome else 0.3
                dom.run(f"v2_sc3_{nome}_s{seed}", 1, ore, 5, cfg, flags, seed, skip="lira,shadow",
                        dopo="v2_sc3_prova_s42", lira=False)
    celle_4f = [
        ("nodp_mu0.01", "experiment_acn4f", "--no-dp"), ("nodp_mu0", "experiment_acn4f_mu0", "--no-dp"),
        ("eps64_mu0.01", "experiment_acn4f_eps64", "--dp-mode local"),
        ("eps64_mu0", "experiment_acn4f_mu0_eps64", "--dp-mode local"),
        ("clip_mu0.01", "experiment_acn4f_eps64", "--dp-mode local --dp-ablation clip-only"),
        ("clip_mu0", "experiment_acn4f_mu0_eps64", "--dp-mode local --dp-ablation clip-only"),
        ("noise_mu0.01", "experiment_acn4f_eps64", "--dp-mode local --dp-ablation noise-only"),
        ("noise_mu0", "experiment_acn4f_mu0_eps64", "--dp-mode local --dp-ablation noise-only"),
    ]
    for seed in SEEDS5:
        for nome, cfg, flags in celle_4f:
            dom.run(f"v2_acn4f_{nome}_s{seed}", 1, 0.25, 3, cfg, flags, seed, skip="lira,shadow",
                    dopo="test_codice", lira=False)
    for seed in SEEDS5:
        for nome, cfg, ore in (("cap16_8", "experiment", 9), ("cap32_16", "experiment_cap32_16", 12),
                               ("cap64_32", "experiment_cap64_32", 18), ("cap128_64", "experiment_cap128_64", 30)):
            dom.run(f"v2_{nome}_s{seed}", 2, ore, 5, cfg, "--no-dp", seed, dopo="test_codice")
    CPS = "{PY} scripts/canary_persona.py --out {ESITI}/canary_persona --combina-se-pronto"
    for seed in SEEDS3:
        for braccio in ("gn", "subject_s0", "subject_s2"):
            for b in ("A", "B"):
                dom.cmd(f"v2_cps_{braccio}_{b}_s{seed}", 1, 2.5, 3,
                        f"{CPS} --config config/experiment_canary_multisite_{braccio}.yaml "
                        f"--etichetta indotto_{braccio} --copie 30 --spostamento 0.5 --braccio {b} --seed {seed}",
                        dopo="test_codice")

    # ── PC Windows: controllo positivo per persona (canary persona) e canary per sessione
    win = Coda("coda_v2_windows.csv", w_win)
    win.cmd("test_codice", 0, 0.2, 2, TEST)
    CP = "{PY} scripts/canary_persona.py --out {ESITI}/canary_persona --combina-se-pronto"
    for d in (0, 0.5):
        for seed in SEEDS5:
            for b in ("A", "B"):
                win.cmd(f"v2_cp_indotto_c30_d{d:g}_{b}_s{seed}", 1, 1.8, 3,
                        f"{CP} --config config/experiment_canary_multisite.yaml --etichetta indotto "
                        f"--copie 30 --spostamento {d:g} --braccio {b} --seed {seed}", dopo="test_codice")
    for seed in SEEDS3:
        for b in ("A", "B"):
            for c in (1, 10, 30):
                win.cmd(f"v2_cp_naturale_c{c}_{b}_s{seed}", 2, 0.7, 3,
                        f"{CP} --config config/experiment.yaml --etichetta naturale --copie {c} "
                        f"--braccio {b} --seed {seed}", dopo="test_codice")
    # canary per sessione su piu' siti con LiRA completa: 3 x 1000 epoche, costoso
    for seed in SEEDS3:
        for braccio, cfg in (("A", "experiment_canary_multisite"), ("B", "experiment_canary_multisite_swap")):
            win.run(f"v2_canary{braccio}_nodp_s{seed}", 2, 55, 4, cfg, "--no-dp", seed, dopo="test_codice")
            win.run(f"v2_canary{braccio}_A3eps64_s{seed}", 2, 55, 4, cfg, "--dp-mode local --epsilon 64",
                    seed, dopo="test_codice")
        for braccio, cfg in (("A", "experiment_canary_multisite_iid"), ("B", "experiment_canary_multisite_swap_iid")):
            win.run(f"v2_canary{braccio}_iid_nodp_s{seed}", 3, 55, 4, cfg, "--no-dp", seed, dopo="test_codice")
    # ── Mac di Domenico, controllo aggiuntivo v2b (2026-10-18). Nella v2 la DP per soggetto con
    #    sigma = 2 lascia vedere il canary persona (AUC 0,77-0,79 in 2 seed su 3), ma in quella
    #    configurazione (3 x 1000 epoche) l'epsilon della persona e' circa 112: nessuna garanzia utile.
    #    Qui: sigma 5 e 8 a 1000 epoche (epsilon circa 29 e 15,5) e, a 200 epoche, riferimento senza
    #    DP, solo taglio e sigma 5 (epsilon circa 10). Ultima riga: epsilon dai log.
    dom2 = Coda("coda_v2b_domenico.csv", w_dom)
    EPS = ("{PY} scripts/epsilon_canary_persona.py --log {ESITI}/../log --esiti {ESITI}/canary_persona "
           "--out {ESITI}/../risultati/epsilon_canary_persona.csv")
    dom2.cmd("v2b_verifica", 0, 0.05, 1, EPS)
    for seed in SEEDS3:
        for braccio, epoche, ore in (("gn", 200, 0.2), ("subject_s0", 200, 0.07), ("subject_s5", 200, 0.07),
                                     ("subject_s5", 1000, 0.3), ("subject_s8", 1000, 0.3)):
            for b in ("A", "B"):
                dom2.cmd(f"v2b_cps_{braccio}_e{epoche}_{b}_s{seed}", 1, ore, 3,
                         f"{CPS} --config config/experiment_canary_multisite_{braccio}.yaml "
                         f"--etichetta indotto_{braccio} --copie 30 --spostamento 0.5 --braccio {b} "
                         f"--seed {seed} --epochs {epoche}", dopo="v2b_verifica")
    dom2.cmd("v2b_epsilon_persona", 9, 0.05, 1, EPS, dopo="v2b_verifica")
    return [mac, dom, win, dom2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers-mac", type=int, default=10)
    ap.add_argument("--workers-domenico", type=int, default=14)
    ap.add_argument("--workers-windows", type=int, default=6)
    ap.add_argument("--out", default=str(QUI))
    a = ap.parse_args()
    code = costruisci(a.workers_mac, a.workers_domenico, a.workers_windows)
    ids = [r["id"] for c in code for r in c.righe if r["id"] != "test_codice"]
    assert len(ids) == len(set(ids)), "id duplicati fra le code"
    out = Path(a.out)
    for c in code:
        c.scrivi(out)
        tot = sum(r["ore"] for r in c.righe)
        print(f"{c.nome}: {len(c.righe)} run, circa {tot:.0f} ore (una run alla volta)")
        for p in sorted({r["priorita"] for r in c.righe}):
            rr = [r for r in c.righe if r["priorita"] == p]
            print(f"  priorita' {p}: {len(rr)} run, {sum(r['ore'] for r in rr):.0f} ore")


if __name__ == "__main__":
    main()
