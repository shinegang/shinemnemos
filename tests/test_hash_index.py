# -*- coding: utf-8 -*-
"""Хеш-индекс точного поиска (фикс перф 27.08).

Store.search_exact: O(1) поиск по точному claim через in-memory hash-индекс
(канонический ключ: strip + casefold) с ленивой синхронизацией с авторитетным
dict и fallback-скана при промахе (самолечение индекса). Плюс быстрый путь в
search(use_hash_index=True): query, равный точному claim, обслуживается за O(1).
"""

import pytest

from mnemos import Store, now_iso


def _node(nid, claim, ts=None):
    return {
        "id": nid,
        "kind": "fact",
        "claim": claim,
        "source": "тест",
        "context": "бенчмарк",
        "ts": ts or now_iso(),
    }


# -- попадание ---------------------------------------------------------------
def test_exact_hit_returns_node(tmp_path):
    store = Store(tmp_path / "nodes.json")
    store.add(_node("mn_a", "Курс BTC вырос на 3% за день"))
    res = store.search_exact("Курс BTC вырос на 3% за день")
    assert [n["id"] for n in res] == ["mn_a"]


def test_exact_ignores_substring(tmp_path):
    """Точность 100%: 'BTC' не находит 'Курс BTC вырос' — никаких substring."""
    store = Store(tmp_path / "nodes.json")
    store.add(_node("mn_a", "Курс BTC вырос на 3% за день"))
    store.add(_node("mn_b", "BTC"))
    assert [n["id"] for n in store.search_exact("BTC")] == ["mn_b"]
    assert store.search_exact("Курс BTC") == []
    assert store.search_exact("BTC вырос") == []


def test_exact_canonical_key_strip_and_casefold(tmp_path):
    store = Store(tmp_path / "nodes.json")
    store.add(_node("mn_a", "Выручка +12% в Q3"))
    # регистр и внешние пробелы не важны (канонический ключ)
    assert [n["id"] for n in store.search_exact("  выручка +12% в q3  ")] == ["mn_a"]
    assert [n["id"] for n in store.search_exact("ВЫРУЧКА +12% В Q3")] == ["mn_a"]


def test_exact_miss_returns_empty(tmp_path):
    store = Store(tmp_path / "nodes.json")
    store.add(_node("mn_a", "факт один"))
    assert store.search_exact("факт два") == []
    assert store.search_exact("") == []
    assert store.search_exact("   ") == []


def test_exact_top_k_and_duplicate_claims(tmp_path):
    store = Store(tmp_path / "nodes.json")
    store.add(_node("mn_a", "Один и тот же факт", ts="2025-01-01T00:00:00+00:00"))
    store.add(_node("mn_b", "Один и тот же факт", ts="2025-01-02T00:00:00+00:00"))
    store.add(_node("mn_c", "Один и тот же факт", ts="2025-01-03T00:00:00+00:00"))
    res = store.search_exact("Один и тот же факт", top_k=2)
    assert {n["id"] for n in res} == {"mn_b", "mn_c"}  # свежие первыми
    # свежие первыми (ts desc)
    assert [n["id"] for n in store.search_exact("Один и тот же факт", top_k=3)] == [
        "mn_c", "mn_b", "mn_a",
    ]


# -- промах -> fallback (защита от рассинхрона) ------------------------------
def test_exact_fallback_repairs_missing_index_entry(tmp_path):
    store = Store(tmp_path / "nodes.json")
    store.add(_node("mn_a", "Сервер отвечает за 100 мс"))
    # рассинхрон: запись пропала из индекса (например, ручная правка _nodes)
    del store._exact_index["сервер отвечает за 100 мс"]
    assert store.search_exact("Сервер отвечает за 100 мс")[0]["id"] == "mn_a"
    # самолечение: запись восстановлена
    assert store._exact_index.get("сервер отвечает за 100 мс") == ["mn_a"]


def test_exact_fallback_repairs_stale_index_entry(tmp_path):
    store = Store(tmp_path / "nodes.json")
    store.add(_node("mn_a", "Сервер отвечает за 100 мс"))
    # рассинхрон: индекс ссылается на несуществующий/чужой узел
    store._exact_index["сервер отвечает за 100 мс"] = ["mn_нет_такого"]
    res = store.search_exact("Сервер отвечает за 100 мс")
    assert [n["id"] for n in res] == ["mn_a"]
    assert store._exact_index.get("сервер отвечает за 100 мс") == ["mn_a"]


def test_exact_miss_when_really_absent_does_not_grow_index(tmp_path):
    store = Store(tmp_path / "nodes.json")
    store.add(_node("mn_a", "факт один"))
    assert store.search_exact("нет такого факта") == []
    assert "нет такого факта" not in store._exact_index


# -- синхронизация после записей ---------------------------------------------
def test_sync_after_add(tmp_path):
    store = Store(tmp_path / "nodes.json")
    assert store.search_exact("новый факт 42") == []
    store.add(_node("mn_a", "новый факт 42"))
    assert [n["id"] for n in store.search_exact("новый факт 42")] == ["mn_a"]


def test_sync_after_add_many(tmp_path):
    store = Store(tmp_path / "nodes.json")
    store.add_many([_node(f"mn_{i}", f"факт номер {i}") for i in range(100)])
    for i in range(100):
        assert [n["id"] for n in store.search_exact(f"факт номер {i}")] == [f"mn_{i}"]
    assert len(store._exact_index) == 100


def test_sync_after_update(tmp_path):
    store = Store(tmp_path / "nodes.json")
    store.add(_node("mn_a", "старый claim"))
    store.update(_node("mn_a", "новый claim"))
    assert store.search_exact("старый claim") == []
    assert [n["id"] for n in store.search_exact("новый claim")] == ["mn_a"]


def test_sync_after_rewrite(tmp_path):
    store = Store(tmp_path / "nodes.json")
    store.add(_node("mn_a", "старый claim"))
    store.rewrite("mn_a", "переписанный claim", source="новый источник")
    assert store.search_exact("старый claim") == []
    assert [n["id"] for n in store.search_exact("переписанный claim")] == ["mn_a"]


def test_sync_after_delete(tmp_path):
    store = Store(tmp_path / "nodes.json")
    store.add(_node("mn_a", "удаляемый claim"))
    assert store.search_exact("удаляемый claim")
    assert store.delete("mn_a") is True
    assert store.search_exact("удаляемый claim") == []
    assert "удаляемый claim" not in store._exact_index


def test_sync_after_add_child(tmp_path):
    store = Store(tmp_path / "nodes.json")
    parent = _node("mn_p", "родительский факт")
    store.add(parent)
    store.add_child("mn_p", _node("mn_c", "детский факт"))
    assert [n["id"] for n in store.search_exact("детский факт")] == ["mn_c"]
    assert [n["id"] for n in store.search_exact("родительский факт")] == ["mn_p"]


def test_index_rebuilt_on_reload(tmp_path):
    """При старте — полная загрузка индекса из nodes.json."""
    path = tmp_path / "nodes.json"
    store = Store(path)
    store.add(_node("mn_a", "персистентный факт"))
    store2 = Store(path)  # переоткрытие
    assert [n["id"] for n in store2.search_exact("персистентный факт")] == ["mn_a"]


def test_reindex_public_api(tmp_path):
    store = Store(tmp_path / "nodes.json")
    store.add(_node("mn_a", "факт"))
    store._nodes["mn_b"] = _node("mn_b", "факт вручную")  # обход API
    assert "факт вручную" not in store._exact_index        # индекс ещё не знает
    # даже до reindex search_exact находит узел — fallback-скан + самолечение
    assert [n["id"] for n in store.search_exact("факт вручную")] == ["mn_b"]
    store.reindex()
    assert store._exact_index.get("факт вручную") == ["mn_b"]


# -- быстрый путь в search() -------------------------------------------------
def test_search_fast_path_on_exact_claim(tmp_path):
    store = Store(tmp_path / "nodes.json")
    store.add(_node("mn_a", "Курс BTC вырос на 3% за день"))
    res = store.search("Курс BTC вырос на 3% за день", top_k=5)
    assert [n["id"] for n in res] == ["mn_a"]


def test_search_substring_still_works_with_fast_path(tmp_path):
    """Не-точные query уходят в прежний substring-путь (результаты те же)."""
    store = Store(tmp_path / "nodes.json")
    old = _node("mn_old", "Старый факт: 1 сервер", ts="2020-01-01T00:00:00+00:00")
    new = _node("mn_new", "Новый факт: 2 сервера", ts="2025-01-01T00:00:00+00:00")
    store.add(old)
    store.add(new)
    res = store.search("сервер", top_k=5)  # substring: свежие первыми
    assert [n["id"] for n in res] == ["mn_new", "mn_old"]


def test_search_use_hash_index_false_is_old_path(tmp_path):
    """use_hash_index=False — прежний полный проход, результат тот же."""
    store = Store(tmp_path / "nodes.json")
    store.add(_node("mn_a", "Выручка +12% в Q3"))
    assert [n["id"] for n in store.search("Выручка +12% в Q3", use_hash_index=False)] == ["mn_a"]
    # точный скан без индекса — тот же результат
    assert [n["id"] for n in store.search_exact("Выручка +12% в Q3", use_hash_index=False)] == ["mn_a"]


# -- Store(use_hash_index=False) ---------------------------------------------
def test_store_without_hash_index_correct_and_unchanged(tmp_path):
    store = Store(tmp_path / "nodes.json", use_hash_index=False)
    assert store._exact_index == {}  # индекса нет — нулевой оверхед
    # ts у всех узлов один и тот же ЯВНО (фикс флака 03.09): «факт 1» — это
    # substring-запрос, под него подходят и mn_1, и mn_10..mn_19, а тай-брейк
    # в search() — свежесть (ts desc). now_iso() тикает в миллисекундах, и если
    # граница миллисекунды падала внутрь этого списка, mn_1x оказывался свежее
    # mn_1 и выигрывал: тест падал ~4 раза на 100 прогонов (замер 03.09,
    # 11/300 на нетронутом /opt/mnemos). Ранжирование тут ни при чём — проверяем
    # мы fallback-скан, а не разрешение ничьих по времени.
    store.add_many([_node(f"mn_{i}", f"факт {i}", ts="2026-09-03T00:00:00.000+00:00")
                    for i in range(20)])
    # оба пути работают и возвращают правильный результат (fallback-скан)
    assert [n["id"] for n in store.search_exact("факт 7")] == ["mn_7"]
    assert [n["id"] for n in store.search("факт 7")] == ["mn_7"]
    # быстрый путь в search() отключён (индекса нет) — старый substring-путь
    assert store.search("факт 1")[0]["id"] == "mn_1"


# -- эквивалентность индекса и полного скана (свойство 100% точности) --------
def test_index_equals_full_scan_on_many_claims(tmp_path):
    """Для всех ключей результат с индексом == результат полного скана."""
    store = Store(tmp_path / "nodes.json")
    claims = [
        "Факт про выручку №1",
        "Факт про выручку №1",
        "Прибыль выросла на 5%",
        "Прибыль выросла на 5% в Q3",
        "BTC достиг 100000",
        "btc достиг 100000",
        "   с пробелами   ",
    ]
    store.add_many([_node(f"mn_{i}", c) for i, c in enumerate(claims)])
    probes = claims + [
        "Прибыль выросла на 5% в q3",
        "прибыль выросла на 5%",
        "BTC",
        "выручка",
        "нет такого",
    ]
    for q in probes:
        with_idx = store.search_exact(q, use_hash_index=True)
        without = store.search_exact(q, use_hash_index=False)
        assert [n["id"] for n in with_idx] == [n["id"] for n in without], q


# -- без fastembed / MCP-совместимость ---------------------------------------
def test_memory_search_substring_via_mcp_still_unchanged(rpc):
    """MCP memory_search (substring) с включённым по умолчанию индексом."""
    rpc("tools/call", {"name": "memory_add", "arguments": {
        "claim": "Курс BTC вырос на 3% за день", "source": "биржа"}})
    resp = rpc("tools/call", {"name": "memory_search", "arguments": {"query": "BTC"}})
    assert resp.get("error") is None, resp
    import json

    out = json.loads(resp["result"]["content"][0]["text"])
    assert out["mode"] == "substring"
    assert out["count"] == 1
    assert out["results"][0]["claim"].startswith("Курс BTC")


def test_memory_search_exact_claim_via_mcp(rpc):
    """Точный claim через MCP — тоже попадает (fast path внутри search)."""
    rpc("tools/call", {"name": "memory_add", "arguments": {
        "claim": "Выручка +12% в Q3", "source": "отчёт", "context": "квартал"}})
    resp = rpc("tools/call", {"name": "memory_search", "arguments": {
        "query": "Выручка +12% в Q3", "top_k": 5}})
    assert resp.get("error") is None, resp
    import json

    out = json.loads(resp["result"]["content"][0]["text"])
    assert out["count"] == 1
    assert out["results"][0]["claim"] == "Выручка +12% в Q3"
