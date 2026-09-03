# -*- coding: utf-8 -*-
"""Тесты truth-gate: каждая проверка П1-П6, порог, комбинации, мутации."""

from datetime import datetime, timedelta, timezone

import pytest

from mnemos.model import MemoryNode, make_node
from mnemos.truth_gate import (
    PASS_THRESHOLD,
    check_and_update,
    check_claim,
)

NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
FRESH_TS = NOW.isoformat(timespec="milliseconds")
OLD_TS = (NOW - timedelta(days=400)).isoformat(timespec="milliseconds")
FUTURE_TS = (NOW + timedelta(hours=1)).isoformat(timespec="milliseconds")


def node(**overrides):
    base = {
        "claim": "Выручка выросла на 12% в Q3 2025",
        "source": "финотчёт компании за Q3",
        "evidence": ["стр. 4 отчёта, таблица 2"],
        "context": "Квартальный обзор, сравниваем с Q2",
        "links": [],
        "ts": FRESH_TS,
    }
    base.update(overrides)
    return base


# --- П1 Свежесть ---------------------------------------------------------------

def test_p1_fresh_ts_passes():
    r = check_claim(node(), now=NOW)
    assert r.checks["P1"]["pass"] is True


def test_p1_missing_ts_fails():
    r = check_claim(node(ts=""), now=NOW)
    assert r.checks["P1"]["pass"] is False


def test_p1_old_ts_fails():
    r = check_claim(node(ts=OLD_TS), now=NOW)
    assert r.checks["P1"]["pass"] is False


def test_p1_future_ts_fails():
    r = check_claim(node(ts=FUTURE_TS), now=NOW)
    assert r.checks["P1"]["pass"] is False


# --- П2 Источник ---------------------------------------------------------------

def test_p2_empty_source_fails():
    r = check_claim(node(source=""), now=NOW)
    assert r.checks["P2"]["pass"] is False


def test_p2_source_present_passes():
    r = check_claim(node(), now=NOW)
    assert r.checks["P2"]["pass"] is True


# --- П3 Цифры ------------------------------------------------------------------

def test_p3_numbers_present_passes():
    r = check_claim(node(), now=NOW)
    assert r.checks["P3"]["pass"] is True


def test_p3_no_numbers_fails():
    r = check_claim(node(claim="Мы улучшили стабильность системы"), now=NOW)
    assert r.checks["P3"]["pass"] is False


# --- П4 Непротиворечивость ------------------------------------------------------

def test_p4_link_to_refuted_fails():
    registry = {"mn_bad": {"kind": "refuted"}}
    r = check_claim(node(links=["mn_bad"]), registry=registry, now=NOW)
    assert r.checks["P4"]["pass"] is False


def test_p4_link_to_fact_passes():
    registry = {"mn_ok": {"kind": "fact"}}
    r = check_claim(node(links=["mn_ok"]), registry=registry, now=NOW)
    assert r.checks["P4"]["pass"] is True


def test_p4_links_without_registry_pass():
    r = check_claim(node(links=["mn_unknown"]), registry=None, now=NOW)
    assert r.checks["P4"]["pass"] is True


def test_p4_broken_link_fails():
    r = check_claim(node(links=[""]), registry=None, now=NOW)
    assert r.checks["P4"]["pass"] is False


# --- П5 Воспроизводимость --------------------------------------------------------

def test_p5_empty_evidence_fails():
    r = check_claim(node(evidence=[]), now=NOW)
    assert r.checks["P5"]["pass"] is False


def test_p5_evidence_present_passes():
    r = check_claim(node(), now=NOW)
    assert r.checks["P5"]["pass"] is True


# --- П6 Полнота ------------------------------------------------------------------

def test_p6_empty_context_fails():
    r = check_claim(node(context=""), now=NOW)
    assert r.checks["P6"]["pass"] is False


def test_p6_context_present_passes():
    r = check_claim(node(), now=NOW)
    assert r.checks["P6"]["pass"] is True


# --- Порог и комбинации -----------------------------------------------------------

def test_full_node_passes_all_six():
    r = check_claim(node(), now=NOW)
    assert r.score == 6
    assert r.verdict == "pass"


def test_threshold_fail_at_three():
    # провалены P2 (source), P3 (цифры), P5 (evidence) -> score 3
    r = check_claim(
        node(source="", claim="Стабильность улучшилась", evidence=[]),
        now=NOW,
    )
    assert r.score == 3
    assert r.verdict == "fail"


def test_threshold_pass_at_four():
    # провалены P2 и P5 -> score 4, ровно порог
    r = check_claim(node(source="", evidence=[]), now=NOW)
    assert r.score == 4
    assert r.verdict == "pass"


def test_threshold_pass_at_five():
    # провален только P6 -> score 5
    r = check_claim(node(context=""), now=NOW)
    assert r.score == 5
    assert r.verdict == "pass"


def test_pass_threshold_constant_is_four():
    assert PASS_THRESHOLD == 4


def test_notes_cover_all_checks():
    r = check_claim(node(), now=NOW)
    assert len(r.notes) == 6
    assert all(isinstance(n, str) and n for n in r.notes)


def test_failing_node_notes_explain_reasons():
    r = check_claim(node(source="", evidence=[]), now=NOW)
    failed = [n for n in r.notes if "пусто" in n or "нет" in n]
    assert any("источник" in n for n in failed)


def test_result_as_dict_shape():
    r = check_claim(node(), now=NOW)
    d = r.as_dict()
    assert set(d) >= {"P1", "P2", "P3", "P4", "P5", "P6", "verdict", "score"}
    assert d["verdict"] == "pass"
    assert d["score"] == 6


def test_registry_as_callable():
    seen = []

    def reg(nid):
        seen.append(nid)
        return {"kind": "refuted"} if nid == "mn_bad" else {"kind": "fact"}

    r = check_claim(node(links=["mn_bad"]), registry=reg, now=NOW)
    assert r.checks["P4"]["pass"] is False
    assert "mn_bad" in seen


# --- check_and_update / модель -----------------------------------------------------

def test_check_and_update_writes_truth_check():
    n = make_node(claim="Выручка +12% в Q3 2025", source="отчёт", evidence=["табл.2"],
                  context="квартал", ts=FRESH_TS)
    r = check_and_update(n, now=NOW)
    assert n.truth_check["verdict"] == r.verdict
    assert n.truth_check["score"] == 6


def test_check_claim_accepts_memorynode_and_dict():
    n = make_node(claim="Выручка +12% в Q3 2025", source="отчёт", evidence=["табл.2"],
                  context="квартал", ts=FRESH_TS)
    r1 = check_claim(n, now=NOW)
    r2 = check_claim(n.to_dict(), now=NOW)
    assert r1.verdict == r2.verdict == "pass"
    assert r1.score == r2.score == 6


def test_check_claim_none_raises():
    with pytest.raises(ValueError):
        check_claim(None)


def test_registry_refuted_via_memorynode():
    target = make_node(claim="Старая гипотеза", kind="refuted", ts=FRESH_TS)
    registry = {target.id: target}
    r = check_claim(node(links=[target.id]), registry=registry, now=NOW)
    assert r.checks["P4"]["pass"] is False
