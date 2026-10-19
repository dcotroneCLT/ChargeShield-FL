"""
Coda v2c (2026-10-19): le 110 run con LiRA e RMIA che alla v2 erano state messe
senza attacco (skip lira,shadow / lira=False). Stessi config e stessi seed della v2.

  - Mac principale (Susy): curva in epsilon su ACN, 6 valori x 5 seed = 30 run.
  - Mac di Domenico: Scozia (12 celle, 40 run con attacco) e ACN a quattro feature
    (8 celle, 40 run con attacco).

Le celle Scozia record e subject hanno anche LiRA e RMIA: stesso meccanismo della
coda ACN. Uso: python3 scripts/genera_coda_v2c.py --out <cartella>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
from genera_coda_v2 import SEEDS5, Coda  # noqa: E402

TEST = ("{PY} -m pytest tests/test_lira2.py tests/test_record_dp_poisson.py "
        "tests/test_client_dp_accounting.py tests/test_subject_dp.py tests/test_profili_abitudini.py -q")

CELLE_SC3 = [
    ("clip_mu0.01", "experiment_sc3_eps64", "--dp-mode local --dp-ablation clip-only", 6),
    ("clip_mu0", "experiment_sc3_mu0_eps64", "--dp-mode local --dp-ablation clip-only", 6),
    ("noise_mu0.01", "experiment_sc3_eps64", "--dp-mode local --dp-ablation noise-only", 6),
    ("noise_mu0", "experiment_sc3_mu0_eps64", "--dp-mode local --dp-ablation noise-only", 6),
    ("eps64_mu0.001", "experiment_sc3_mu0.001_eps64", "--dp-mode local", 6),
    ("eps64_mu0.1", "experiment_sc3_mu0.1_eps64", "--dp-mode local", 6),
    ("record_s1_mu0.01", "experiment_sc3_recorddp", "--no-dp", 12),
    ("subject_s5_mu0.01", "experiment_sc3_subjectdp", "--no-dp", 12),
]

CELLE_4F = [
    ("nodp_mu0.01", "experiment_acn4f", "--no-dp"),
    ("nodp_mu0", "experiment_acn4f_mu0", "--no-dp"),
    ("eps64_mu0.01", "experiment_acn4f_eps64", "--dp-mode local"),
    ("eps64_mu0", "experiment_acn4f_mu0_eps64", "--dp-mode local"),
    ("clip_mu0.01", "experiment_acn4f_eps64", "--dp-mode local --dp-ablation clip-only"),
    ("clip_mu0", "experiment_acn4f_mu0_eps64", "--dp-mode local --dp-ablation clip-only"),
    ("noise_mu0.01", "experiment_acn4f_eps64", "--dp-mode local --dp-ablation noise-only"),
    ("noise_mu0", "experiment_acn4f_mu0_eps64", "--dp-mode local --dp-ablation noise-only"),
]


def costruisci(w_mac: int, w_dom: int) -> list[Coda]:
    mac = Coda("coda_v2c_mac.csv", w_mac)
    mac.cmd("test_codice", 0, 0.2, 2, TEST)
    for seed in SEEDS5:
        for e in (2, 4, 8, 16, 32, 256):
            mac.run(f"v2c_acn_eps{e}_s{seed}", 1, 9, 4, f"experiment_rq1_eps{e}", "--dp-mode local",
                    seed, dopo="test_codice")

    dom = Coda("coda_v2c_domenico.csv", w_dom)
    dom.cmd("test_codice", 0, 0.2, 2, TEST)
    for seed in SEEDS5:
        for nome, cfg, flags, ore in CELLE_SC3:
            dom.run(f"v2c_sc3_{nome}_s{seed}", 1, ore, 5, cfg, flags, seed, dopo="test_codice")
    for seed in SEEDS5:
        for nome, cfg, flags in CELLE_4F:
            dom.run(f"v2c_acn4f_{nome}_s{seed}", 2, 6, 3, cfg, flags, seed, dopo="test_codice")
    return [mac, dom]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers-mac", type=int, default=10)
    ap.add_argument("--workers-domenico", type=int, default=14)
    ap.add_argument("--out", default=str(QUI))
    a = ap.parse_args()
    code = costruisci(a.workers_mac, a.workers_domenico)
    ids = [r["id"] for c in code for r in c.righe if r["id"] != "test_codice"]
    assert len(ids) == len(set(ids)), "id duplicati"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for c in code:
        c.scrivi(out)
        print(f"{c.nome}: {len(c.righe) - 1} run con attacco")
    print(f"totale run: {len(ids)}")


if __name__ == "__main__":
    main()
