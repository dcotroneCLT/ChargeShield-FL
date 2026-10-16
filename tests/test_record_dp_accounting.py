"""
Test dell'accountant per la DP a livello di record (src/ml/record_dp_accounting.py),
correzione della segnalazione 4 (2026-09-24).

Puro Python, nessuna dipendenza torch. Richiede `dp-accounting`, dichiarato in
pyproject.toml: se manca, questo file fallisce in raccolta invece di saltare,
perche' l'assenza silenziosa dell'accountant era proprio uno dei difetti della
segnalazione 4.

Valori di riferimento calcolati il 2026-09-24 con dp-accounting 0.6.0. Il 7.15
a sigma = 5 e l'88 a sigma = 1 sono i valori gia' riportati per le run canary
record-DP (Office 1, 1924 record, B = 32, E = 1000, T = 3): li riproduciamo, a
conferma che erano stati calcolati con i round giusti.
"""
import math

import dp_accounting  # noqa: F401  (vedi docstring: deve fallire se manca)
import pytest

from ml.record_dp_accounting import (
    epsilon_poisson,
    epsilon_shuffle_bound,
    record_dp_accounting,
    record_dp_fields,
)

DELTA = 1e-5
EB_CLIENTS = {"jpl": 26892, "caltech": 25143, "office1": 1335}


def _cfg_annidato(rounds=3, epochs=1000, sigma=5.0, enabled=True, delta=DELTA):
    exp = {"delta": delta}
    if rounds is not None:
        exp["fl_rounds"] = rounds
    return {
        "experiment": exp,
        "ml": {"batch_size": 32, "epochs": epochs,
               "record_dp": {"enabled": enabled, "noise_multiplier": sigma,
                             "max_grad_norm": 1.0}},
    }


def test_canary_poisson_riproduce_i_valori_pubblicati():
    # valori pubblicati prima del 2026-10-16: adiacenza add/remove (RDP)
    eps5, steps, q = epsilon_poisson(1924, 32, 1000, 3, 5.0, DELTA, relazione="add_remove")
    eps1, _, _ = epsilon_poisson(1924, 32, 1000, 3, 1.0, DELTA, relazione="add_remove")
    assert steps == (1924 // 32) * 1000 * 3 == 180000
    assert q == pytest.approx(32 / 1924)
    assert eps5 == pytest.approx(7.15, abs=0.02)
    assert eps1 == pytest.approx(88.2, abs=0.3)


def test_regressione_round_letti_da_experiment():
    # Il bug: fl_rounds letto dalla radice di cfg -> 1 round -> 3.79 invece di 7.15.
    f = record_dp_fields(_cfg_annidato(), n_per_client={"office1": 1924})
    eps_un_round, _, _ = epsilon_poisson(1924, 32, 1000, 1, 5.0, DELTA, relazione="add_remove")
    assert eps_un_round == pytest.approx(3.79, abs=0.02)
    pc = f["record_dp_accounting"]["per_client"]["office1"]
    assert pc["epsilon_add_remove"] == pytest.approx(7.15, abs=0.02)
    assert f["epsilon_record_dp"] == pytest.approx(
        epsilon_poisson(1924, 32, 1000, 3, 5.0, DELTA)[0])
    assert f["record_dp_accounting"]["fl_rounds"] == 3


def test_config_piatto_dei_json_da_lo_stesso_valore():
    piatto = {"fl_rounds": 3, "delta": DELTA, "batch_size": 32, "epochs": 1000,
              "record_dp": {"enabled": True, "noise_multiplier": 5.0}}
    a = record_dp_fields(piatto, n_per_client={"office1": 1924})
    b = record_dp_fields(_cfg_annidato(), n_per_client={"office1": 1924})
    assert a["epsilon_record_dp"] == pytest.approx(b["epsilon_record_dp"])


def test_round_mancanti_nessun_default_silenzioso():
    f = record_dp_fields(_cfg_annidato(rounds=None), n_per_client={"office1": 1924})
    assert f["epsilon_record_dp"] is None
    assert "fl_rounds" in f["record_dp_accounting_note"]


def test_senza_n_per_client_non_usa_il_totale():
    f = record_dp_fields(_cfg_annidato(), n_sessions=53370)
    assert f["epsilon_record_dp"] is None
    assert "n per client" in f["record_dp_accounting_note"]


def test_per_client_il_peggiore_e_il_client_piu_piccolo():
    acc = record_dp_accounting(EB_CLIENTS, 32, 50, 10, 1.0, DELTA)
    ear = {c: v["epsilon_add_remove"] for c, v in acc["per_client"].items()}
    assert ear["jpl"] == pytest.approx(4.82, abs=0.02)
    assert ear["caltech"] == pytest.approx(5.01, abs=0.02)
    assert ear["office1"] == pytest.approx(30.4, abs=0.1)
    eps = {c: v["epsilon"] for c, v in acc["per_client"].items()}
    # sostituzione (default dal 2026-10-16) piu' severa di add/remove
    assert all(eps[c] > ear[c] for c in eps)
    assert acc["client_peggiore"] == "office1"
    assert acc["epsilon_max"] == max(eps.values())
    f = record_dp_fields(_cfg_annidato(rounds=10, epochs=50, sigma=1.0),
                         n_sessions=sum(EB_CLIENTS.values()), n_per_client=EB_CLIENTS)
    assert f["epsilon_record_dp"] == pytest.approx(eps["office1"])
    assert set(f["epsilon_record_dp_per_client"]) == set(EB_CLIENTS)


def _rdp_gaussiano_classico(k, z, delta):
    """Conversione RDP -> (eps, delta) classica (Mironov 2017), implementazione
    indipendente: e' un limite superiore di quella, piu' stretta, della libreria."""
    return min(k * a / (2 * z * z) + math.log(1 / delta) / (a - 1)
               for a in (1 + i / 1000 for i in range(1, 20000)))


@pytest.mark.parametrize("epochs,rounds,sigma,atteso", [
    (1000, 3, 5.0, 342.9),
    (50, 10, 1.0, 1211.8),
    (50, 10, 2.0, 354.9),
])
def test_limite_shuffle(epochs, rounds, sigma, atteso):
    lib = epsilon_shuffle_bound(epochs, rounds, sigma, DELTA)
    classico = _rdp_gaussiano_classico(epochs * rounds, sigma / 2, DELTA)
    assert lib == pytest.approx(atteso, rel=1e-3)
    assert lib <= classico <= lib * 1.02


def test_limite_shuffle_non_dipende_da_n_ed_e_sopra_poisson():
    a = record_dp_accounting({"piccolo": 1335}, 32, 50, 10, 2.0, DELTA)
    b = record_dp_accounting({"grande": 26892}, 32, 50, 10, 2.0, DELTA)
    assert a["epsilon_shuffle_bound"] == b["epsilon_shuffle_bound"]
    assert a["epsilon_shuffle_bound"] > a["epsilon_max"] > b["epsilon_max"]


def test_record_dp_disattivo():
    f = record_dp_fields(_cfg_annidato(enabled=False), n_per_client={"x": 100})
    assert f["epsilon_record_dp"] is None
    assert f["epsilon_record_dp_shuffle_bound"] is None
    assert "disattivo" in f["record_dp_accounting_note"]


def test_sigma_zero_nessuna_garanzia():
    f = record_dp_fields(_cfg_annidato(sigma=0.0), n_per_client={"office1": 1924})
    assert f["epsilon_record_dp"] is None
    assert "nessuna garanzia" in f["record_dp_accounting_note"]


def test_client_con_meno_record_del_batch():
    eps, steps, _ = epsilon_poisson(20, 32, 50, 10, 1.0, DELTA)
    assert (eps, steps) == (0.0, 0)



def test_sostituzione_con_pld_come_dp_accounting():
    """Default dal 2026-10-16: sostituzione di un record, PLD."""
    import dp_accounting
    from dp_accounting import dp_event, pld
    eps, steps, q = epsilon_poisson(1341, 32, 50, 10, 1.0, DELTA)
    acc = pld.PLDAccountant(neighboring_relation=dp_accounting.NeighboringRelation.REPLACE_ONE)
    acc.compose(dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(1.0)), steps)
    assert eps == pytest.approx(acc.get_epsilon(DELTA))
