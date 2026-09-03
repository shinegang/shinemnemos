# -*- coding: utf-8 -*-
"""Бенчмарк-фикс 26.08: батч-запись Store.add_many (снимает O(N²) вставки).

Раньше каждый add переписывал весь nodes.json (O(N²) суммарно); add_many
валидирует ВСЕ узлы, добавляет ВСЕ и делает ОДИН _save в конце, атомарно:
при любой ошибке (невалидный узел, дубликат id) не добавляется ни один узел.
"""

import json
import time

import pytest

from mnemos import Store, make_node, now_iso


def _node(i, tag="batch"):
    """Простой dict-узел с явным id (как шлёт MCP-клиент)."""
    return {
        "id": f"mn_{tag}_{i}",
        "kind": "fact",
        "claim": f"факт номер {i}: сервис отвечает за {i * 10} мс",
        "source": "тесты",
        "context": "перф",
        "ts": now_iso(),
    }


def test_add_many_all_nodes_present_and_file_valid(tmp_path):
    store = Store(tmp_path / "nodes.json")
    nodes = [_node(i) for i in range(500)]
    added = store.add_many(nodes)
    assert len(added) == 500
    assert len(store) == 500
    for i in range(500):
        assert store.get(f"mn_batch_{i}") is not None
    # файл валиден и содержит все узлы
    with open(tmp_path / "nodes.json", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict) and len(data) == 500
    assert data["mn_batch_0"]["claim"].startswith("факт номер 0")
    assert data["mn_batch_499"]["claim"].startswith("факт номер 499")


def test_add_many_accepts_memorynode_objects(tmp_path):
    store = Store(tmp_path / "nodes.json")
    nodes = [make_node(claim=f"заметка {i}", source="блокнот") for i in range(10)]
    added = store.add_many(nodes)
    assert len(added) == 10
    assert all(isinstance(d, dict) for d in added)
    assert all(d["id"].startswith("mn_") for d in added)
    assert len(store) == 10


def test_add_many_duplicate_with_store_is_atomic(tmp_path):
    store = Store(tmp_path / "nodes.json")
    store.add(_node(0))
    before = len(store)
    with pytest.raises(ValueError):
        # n1, n2 — новые, но n0 уже в хранилище: НИЧЕГО не добавляется
        store.add_many([_node(1), _node(0), _node(2)])
    assert len(store) == before
    assert store.get("mn_batch_1") is None
    assert store.get("mn_batch_2") is None


def test_add_many_duplicate_within_batch_is_atomic(tmp_path):
    store = Store(tmp_path / "nodes.json")
    with pytest.raises(ValueError, match="дубликат"):
        store.add_many([_node(1), _node(1)])
    assert len(store) == 0


def test_add_many_invalid_node_adds_nothing(tmp_path):
    store = Store(tmp_path / "nodes.json")
    bad = {"id": "mn_bad", "kind": "fact", "claim": "   ",
           "ts": now_iso()}  # пустой claim — невалиден
    with pytest.raises(ValueError):
        store.add_many([_node(1), bad, _node(2)])
    assert len(store) == 0
    assert not (tmp_path / "nodes.json").exists()  # _save даже не вызывался


def test_add_many_empty_list(tmp_path):
    store = Store(tmp_path / "nodes.json")
    assert store.add_many([]) == []
    assert len(store) == 0


def test_add_many_faster_than_sequential_adds(tmp_path):
    """Бенчмарк-фикс 26.08: один _save вместо N — add_many обязан быть
    существенно быстрее N последовательных add (это и есть снятие O(N²))."""
    nodes = [_node(i, tag="seq") for i in range(300)]
    store = Store(tmp_path / "nodes_seq.json")
    t0 = time.perf_counter()
    for n in nodes:
        store.add(n)
    seq_time = time.perf_counter() - t0

    store2 = Store(tmp_path / "nodes_batch.json")
    nodes2 = [_node(i, tag="bat") for i in range(300)]
    t0 = time.perf_counter()
    store2.add_many(nodes2)
    batch_time = time.perf_counter() - t0

    assert batch_time < seq_time * 0.5, (
        f"add_many не снял O(N²): batch={batch_time:.4f}s, "
        f"seq×0.5={seq_time * 0.5:.4f}s"
    )
