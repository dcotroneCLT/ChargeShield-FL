"""Soggetti dalle abitudini (2026-10-16)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
pytest.importorskip("sklearn")

from ml import profili_abitudini as pa  # noqa: E402
from ml.subject_dp import subject_dp_config, subject_of  # noqa: E402


def _s(i, rng, site="0002", uid=None):
    return {"site_id": site, "cluster_id": "0039", "user_id": uid,
            "start_time": f"2019-03-{1 + i % 28:02d}T15:00:00", "timezone": "America/Los_Angeles",
            "hour_of_day": rng.random(), "total_energy_kwh": rng.random(),
            "duration_hours": rng.random(), "kwh_requested": rng.random(),
            "minutes_available": rng.random()}


def test_assegnazione_dipende_solo_dalla_sessione():
    rng = random.Random(0)
    esterne = [_s(i, rng) for i in range(200)]
    m = pa.addestra(esterne, dimensione_media=10, seed=1)
    x = _s(5, random.Random(9))
    p1 = m.profilo(dict(x))
    # aggiungere o togliere altre sessioni dal training non cambia il profilo di x
    altre = [_s(i, rng) for i in range(50)]
    pa.assegna(m, altre)
    assert m.profilo(dict(x)) == p1
    assert m.descrizione()["profili_per_sito"]["0002"] == 20


def test_deterministico():
    rng = random.Random(0)
    esterne = [_s(i, rng) for i in range(100)]
    a = pa.addestra(esterne, dimensione_media=10, seed=3)
    b = pa.addestra(esterne, dimensione_media=10, seed=3)
    x = _s(1, random.Random(4))
    assert a.profilo(x) == b.profilo(x)


def test_soggetto_chiave_poi_profilo_poi_singolo():
    assert subject_of({"user_id": "7", "_profilo": "0002:3"}, "user_id", "f") == "user_id:7"
    assert subject_of({"user_id": None, "_profilo": "0002:3"}, "user_id", "f") == "profilo:0002:3"
    assert subject_of({"user_id": None}, "user_id", "f") == "solo:f"
    assert subject_of({"_canary_group": "g", "user_id": "7"}, "user_id", "f") == "canary:g"


def test_config():
    assert subject_dp_config({"subject_dp": {"enabled": True}})["senza_chiave"] == "profili"
    with pytest.raises(ValueError):
        subject_dp_config({"subject_dp": {"senza_chiave": "altro"}})
