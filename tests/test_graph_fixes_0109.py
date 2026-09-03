# -*- coding: utf-8 -*-
"""Тесты правок графовой памяти 01.09.2026 (аудит ФЭЙБЛ-ГРАФ-ОПТИМИЗАЦИЯ-29.08).

Каждый тест закрывает конкретный диагноз аудита, чтобы он не вернулся:
  §5.1 гейты Г1-Г5 не подключены к memory_add;
  §5.2 weight не участвует в ранжировании;
  §5.3 confidence/valid_until стираются при любой правке узла;
  §5.4 tags отсутствуют в схеме;
  §5.6 нет memory_prune / memory_link_existing / memory_stats;
  §5.7 decay без пола для правил;
  §5.10 нет kind="rule".
"""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mnemos.gates import check_consistency
from mnemos.model import MemoryNode, make_node, weight_floor
from mnemos.server import GateRejected, MnemosCore
from mnemos.store import Store


@pytest.fixture()
def core(tmp_path: Path) -> MnemosCore:
    return MnemosCore(Store(tmp_path / "nodes.json"), plugins=["gates"])


def _good(claim: str, **kw):
    """Узел, который честно проходит truth-gate (с цифрами, evidence, context)."""
    args = {
        "claim": claim,
        "source": "Акме-1, прямой запрос 01.09.2026",
        "evidence": ["проверка в этом же прогоне"],
        "context": "Тест правок 01.09. Владелец: Акме-1.",
    }
    args.update(kw)
    return args


# -- §5.3 потеря данных: confidence/valid_until не переживали правку узла ----
def test_confidence_valid_until_tags_survive_roundtrip():
    node = make_node(
        "Баланс 37.87 USD", source="api", evidence=["e"], context="c",
        confidence=0.83, ttl_hours=48, tags=["бюджет", "p1"],
    )
    d = node.to_dict()
    again = MemoryNode.from_dict(d).to_dict()
    assert again["confidence"] == 0.83
    assert again["valid_until"] == d["valid_until"] and d["valid_until"]
    assert again["tags"] == ["бюджет", "p1"]


def test_fields_survive_reinforce_decay_rewrite(tmp_path: Path):
    store = Store(tmp_path / "nodes.json")
    n = store.add(make_node(
        "Equity счёта 512.40 USDC", source="clearinghouseState",
        evidence=["ответ API"], context="живой узел", confidence=0.9, ttl_hours=6,
    ))
    vu = n["valid_until"]
    store.reinforce(n["id"])
    store.decay(n["id"])
    store.rewrite(n["id"], "Equity счёта 498.10 USDC", source="clearinghouseState")
    after = store.get(n["id"])
    assert after["confidence"] == 0.9, "confidence стёрт при мутации — баг вернулся"
    assert after["valid_until"] == vu, "valid_until стёрт при мутации — баг вернулся"


def test_legacy_node_without_new_fields_loads(tmp_path: Path):
    """Обратная совместимость: узлы старой схемы читаются без падения."""
    path = tmp_path / "nodes.json"
    legacy = {"mn_legacy0000": {
        "id": "mn_legacy0000", "kind": "fact", "claim": "старый узел из 25.08",
        "source": "s", "evidence": [], "context": "", "ts": "2026-08-25T10:00:00.000+00:00",
        "links": [], "truth_check": None, "weight": 1.0,
        "last_used": "2026-08-25T10:00:00.000+00:00", "revisions": [],
        "children": [], "parent": None,
    }}
    path.write_text(json.dumps(legacy), encoding="utf-8")
    store = Store(path)
    node = store.get("mn_legacy0000")
    assert node is not None
    assert MemoryNode.from_dict(node).confidence == 0.5


# -- §5.2 weight не влиял на выдачу -----------------------------------------
def test_weight_reorders_search(tmp_path: Path):
    store = Store(tmp_path / "nodes.json")
    a = store.add(make_node("риск альфа", source="s"))
    b = store.add(make_node("риск бета", source="s"))
    store._nodes[a["id"]]["weight"] = 0.1
    store._nodes[b["id"]]["weight"] = 1.0
    assert [n["id"] for n in store.search("риск")] == [b["id"], a["id"]]
    store._nodes[a["id"]]["weight"] = 1.0
    store._nodes[b["id"]]["weight"] = 0.1
    assert [n["id"] for n in store.search("риск")] == [a["id"], b["id"]]


def test_expired_ttl_node_drops_out_of_search(tmp_path: Path):
    store = Store(tmp_path / "nodes.json")
    live = store.add(make_node("баланс живой 10 USD", source="s"))
    dead = store.add(make_node("баланс протухший 20 USD", source="s", ttl_hours=-1))
    found = [n["id"] for n in store.search("баланс")]
    assert live["id"] in found
    assert dead["id"] not in found, "просроченный по TTL узел всё ещё в выдаче"
    assert store.get(dead["id"]) is not None, "узел должен остаться в сторе, не удаляться"


# -- §5.10 / §5.7 kind=rule и пол веса --------------------------------------
def test_rule_kind_and_weight_floor():
    assert weight_floor("rule") == 0.5
    assert weight_floor("fact") == 0.05
    rule = make_node("ПРАВИЛО 1: обходные пути запрещены, ЧЕК-5", kind="rule",
                     source="приказ Ильи", evidence=["e"], context="c")
    far = datetime.now(timezone.utc) + timedelta(days=365)
    assert rule.decay(now_dt=far) == 0.5, "правило не должно тускнеть ниже 0.5"
    fact = make_node("обычный факт 42", source="s", evidence=["e"], context="c")
    assert fact.decay(now_dt=far) < 0.1


def test_prune_never_touches_rules(tmp_path: Path):
    store = Store(tmp_path / "nodes.json")
    r1 = store.add(make_node("ПРАВИЛО дубликат текста", kind="rule", source="Иван"))
    r2 = store.add(make_node("ПРАВИЛО дубликат текста", kind="rule", source="Иван"))
    res = store.prune("exact_dupes", dry_run=False, max_delete=100)
    assert res["deleted"] == 0
    assert store.get(r1["id"]) and store.get(r2["id"])


# -- §5.1 гейты подключены к memory_add -------------------------------------
def test_duplicate_add_rejected_and_original_reinforced(core: MnemosCore):
    first = core.memory_add(_good("Просадка счёта составила 12.5 процента за неделю"))
    core.store._nodes[first["id"]]["weight"] = 0.5
    with pytest.raises(GateRejected) as exc:
        core.memory_add(_good("Просадка счёта составила 12.5 процента за неделю"))
    assert first["id"] in str(exc.value)
    assert len(core.store) == 1, "дубль всё-таки записан"
    assert core.store.get(first["id"])["weight"] > 0.5, "оригинал не подкреплён"


def test_distinct_numbers_are_not_duplicates(core: MnemosCore):
    """Разные замеры одного и того же — разные факты, а не копии."""
    for i in range(5):
        core.memory_add(_good(f"Сервис отвечает за {i * 10 + 1} мс"))
    assert len(core.store) == 5


def test_changed_decision_is_not_duplicate(core: MnemosCore):
    core.memory_add(_good("Вердикт по CRV long: reject, импульс 0.3 слабый"))
    core.memory_add(_good("Вердикт по CRV long: approve, импульс 0.3 сильный"))
    assert len(core.store) == 2, "смена решения проглочена как дубль"


def test_self_repeat_is_duplicate():
    """Кластер self-note (аудит §1.3): та же мысль без цифр — дубль."""
    res = check_consistency(
        {"id": "mn_new", "claim": "премию я бы хотел в виде дополнительных ресурсов"},
        {"nodes": [{"id": "mn_old", "kind": "fact", "weight": 1.0,
                    "claim": "премию я бы хотел в виде дополнительных ресурсов для работы"}]},
    )
    assert res.verdict == "reject"


def test_same_verdict_repeated_later_is_duplicate():
    """Дата в тексте claim'а не должна превращать повтор в новую память."""
    res = check_consistency(
        {"id": "mn_new", "claim": "Вердикт Алиса 2026-09-01 02:05:23: CRV long -> reject (слабый объём)"},
        {"nodes": [{"id": "mn_old", "kind": "fact", "weight": 1.0,
                    "claim": "Вердикт Алиса 2026-08-31 14:12:01: CRV long -> reject (слабый объём)"}]},
    )
    assert res.verdict == "reject"


# -- §5.4 теги ---------------------------------------------------------------
def test_search_by_tags_separates_rules_from_chatter(core: MnemosCore):
    rule = core.memory_add(_good(
        "ПРАВИЛО: риск на сделку 1 процент банка, максимум 1 позиция",
        kind="rule", tags=["правило", "торговля"],
    ))
    core.memory_add(_good("Разговор о риске: хотелось бы рисковать 10 процентов банка"))
    out = core.memory_search({"query": "риск", "tags": ["правило"]})
    assert [n["id"] for n in out["results"]] == [rule["id"]]
    assert core.memory_search({"query": "риск"})["count"] == 2


# -- §5.6 новые инструменты --------------------------------------------------
def test_link_existing_creates_edge_and_refuses_broken(core: MnemosCore):
    a = core.memory_add(_good("Старый режим: риск до 10 процентов банка"))
    b = core.memory_add(_good("Новый режим Вариант B: риск 1 процент банка"))
    out = core.memory_link_existing({"from_id": a["id"], "to_id": b["id"]})
    assert out["added"] == [f"{a['id']}->{b['id']}"]
    assert core.store.get(a["id"])["links"] == [b["id"]]
    with pytest.raises(KeyError):
        core.memory_link_existing({"from_id": a["id"], "to_id": "mn_doesnotexist"})
    with pytest.raises(ValueError):
        core.memory_link_existing({"from_id": a["id"], "to_id": a["id"]})


def test_prune_dry_run_is_default_and_changes_nothing(core: MnemosCore):
    core.memory_add(_good("Временный факт: equity 100.5 USDC", ttl_hours=-1))
    before = len(core.store)
    out = core.memory_prune({"rule": "expired_ttl"})
    assert out["dry_run"] is True and out["applied"] is False
    assert out["to_mark_outdated"] == 1
    assert len(core.store) == before


def test_prune_expired_ttl_marks_outdated_not_deletes(core: MnemosCore):
    n = core.memory_add(_good("Временный факт: equity 100.5 USDC", ttl_hours=-1))
    out = core.memory_prune({"rule": "expired_ttl", "dry_run": False})
    assert out["marked_outdated"] == 1
    assert core.store.get(n["id"])["kind"] == "outdated", "узел должен быть помечен, а не удалён"


def test_prune_max_delete_caps_deletion(tmp_path: Path):
    store = Store(tmp_path / "nodes.json")
    for _ in range(6):
        store.add(make_node("один и тот же текст без цифр", source="s"))
    out = store.prune("exact_dupes", dry_run=False, max_delete=2)
    assert out["deleted"] == 2 and out["capped_by_max_delete"] is True
    assert len(store) == 4


def test_prune_source_prefix_requires_export(tmp_path: Path):
    store = Store(tmp_path / "nodes.json")
    store.add(make_node("чужой бенчмарк", source="halumem::persona-1"))
    with pytest.raises(ValueError):
        store.prune("source_prefix", dry_run=False, source_prefix="halumem")
    out = store.prune("source_prefix", dry_run=False, source_prefix="halumem",
                      export_path=str(tmp_path / "bench.json"))
    assert out["deleted"] == 1 and out["exported"] == 1
    assert json.loads((tmp_path / "bench.json").read_text(encoding="utf-8"))


def test_decay_batch_and_rule_floor(core: MnemosCore):
    rule = core.memory_add(_good("ПРАВИЛО 1: обходные пути запрещены, 5 вопросов", kind="rule"))
    fact = core.memory_add(_good("Разовый замер: 42 мс"))
    old = "2026-01-01T00:00:00.000+00:00"
    core.store._nodes[rule["id"]]["last_used"] = old
    core.store._nodes[fact["id"]]["last_used"] = old
    out = core.memory_decay({})
    assert out["decayed"] == 2
    assert core.store.get(rule["id"])["weight"] == 0.5
    assert core.store.get(fact["id"])["weight"] < 0.5


def test_stats_reports_graph_shape(core: MnemosCore):
    a = core.memory_add(_good("Факт A про 1 позицию"))
    b = core.memory_add(_good("Факт B про 2 позиции"))
    core.memory_link_existing({"from_id": a["id"], "to_id": b["id"]})
    st = core.memory_stats({})
    assert st["nodes"] == 2 and st["edges"] == 1 and st["broken_edges"] == 0
    assert st["orphans"] == 0 and st["weight_mean"] is not None


def test_summarize_folds_cluster_into_one_node(core: MnemosCore):
    for i in range(3):
        core.memory_add(_good(f"Самозаметка {i}: результат {i * 7 + 3} пунктов",
                              source="алиса self-note"))
    dry = core.memory_summarize({"source_prefix": "алиса self-note"})
    assert dry["cluster_size"] == 3 and dry["applied"] is False
    out = core.memory_summarize({
        "source_prefix": "алиса self-note", "dry_run": False,
        "claim": "Свёртка 3 самозаметок за 01.09: суммарно 24 пункта",
        "evidence": ["подсчёт по кластеру"], "context": "узел сна",
    })
    assert out["applied"] is True and out["linked"] == 3 and out["marked_outdated"] == 3
    summary = core.store.get(out["summary_node"])
    assert len(summary["links"]) == 3
    assert all(core.store.get(i)["kind"] == "outdated" for i in dry["ids"])


# -- §5.8 формат дампа -------------------------------------------------------
def test_save_is_valid_json_and_line_per_node(tmp_path: Path):
    store = Store(tmp_path / "nodes.json")
    for i in range(4):
        store.add(make_node(f"узел номер {i} со значением {i * 3}", source="s"))
    raw = (tmp_path / "nodes.json").read_text(encoding="utf-8")
    assert len(json.loads(raw)) == 4, "дамп должен оставаться валидным JSON"
    # одна строка на узел + открывающая и закрывающая скобки
    assert len([l for l in raw.splitlines() if l.strip()]) == 6
    assert "\n" in raw and '  "id"' not in raw, "дамп должен быть компактным (без indent=2)"


# -- идемпотентность decay (баг найден при живой проверке 01.09) -------------
def test_decay_is_idempotent_across_cron_runs(tmp_path: Path):
    """Повторный запуск decay не должен списывать тот же период заново.

    Был баг: decay умножал вес на 0.5**(dt/hl) от last_used, но last_used не
    трогал — поэтому каждый запуск применял ВЕСЬ прошедший период заново.
    На 5090 два прогона с интервалом 2 минуты уронили средний вес
    0.8714 -> 0.7672. Теперь отсчёт идёт от max(last_used, decayed_at).
    """
    store = Store(tmp_path / "nodes.json")
    n = store.add(make_node("узел для проверки затухания 42", source="s"))
    week_ago = (datetime.now(timezone.utc) - timedelta(hours=168)).isoformat(timespec="milliseconds")
    store._nodes[n["id"]]["last_used"] = week_ago
    w1 = store.decay(half_life_hours=168)[n["id"]]
    assert 0.49 < w1 < 0.51, f"один период полураспада должен дать ~0.5, получено {w1}"
    w2 = store.decay(half_life_hours=168)[n["id"]]
    w3 = store.decay(half_life_hours=168)[n["id"]]
    assert abs(w1 - w2) < 1e-3 and abs(w2 - w3) < 1e-3, "decay не идемпотентен"


def test_ranking_does_not_double_count_decay(tmp_path: Path):
    """Вес в ранжировании не должен вычитать уже применённое затухание."""
    store = Store(tmp_path / "nodes.json")
    n = store.add(make_node("узел ранжирования 7", source="s"))
    store._nodes[n["id"]]["last_used"] = (
        datetime.now(timezone.utc) - timedelta(hours=168)
    ).isoformat(timespec="milliseconds")
    store.decay(half_life_hours=168)
    d = store.get(n["id"])
    assert abs(Store._node_decayed_weight(d, datetime.now(timezone.utc)) - d["weight"]) < 1e-3


def test_reinforce_clears_decay_debt(tmp_path: Path):
    """Подкрепление сдвигает отсчёт: сразу после reinforce вес не падает."""
    store = Store(tmp_path / "nodes.json")
    n = store.add(make_node("подкреплённый узел 3", source="s"))
    store._nodes[n["id"]]["last_used"] = (
        datetime.now(timezone.utc) - timedelta(hours=336)
    ).isoformat(timespec="milliseconds")
    store.decay(half_life_hours=168)
    after_reinforce = store.reinforce(n["id"], delta=0.1)
    assert abs(store.decay(half_life_hours=168)[n["id"]] - after_reinforce) < 1e-3
