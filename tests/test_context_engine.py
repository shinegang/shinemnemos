# -*- coding: utf-8 -*-
"""Тесты context_engine: иерархическое сжатие, канонический префикс,
дефрагментация сессий, гейты перед выгрузкой.

Проверяем требования задачи:
  1. сжатие сохраняет ключевые факты (поиск по Store находит сводку);
  2. префикс стабилен (повторный build_prefix идентичен);
  3. дефрагментация уменьшает размер сессии;
  4. выгрузка в память работает (узлы kind='context_summary' в Store);
  + гейты-хук перед выгрузкой (mnemos.gates).
"""

from datetime import datetime, timezone

from mnemos import Store
from mnemos.context_engine import (
    CanonicalPrefix,
    ContextDefragmenter,
    HierarchicalCompactor,
    estimate_tokens,
)
from mnemos.gates import (
    run_read_gates,
    run_write_gates,
)

NOW = datetime(2026, 2, 10, 12, 0, 0, tzinfo=timezone.utc)


def msg(i, text, role="user"):
    """Сообщение сессии в формате dict."""
    return {"id": f"m{i:03d}", "role": role, "content": text, "ts": ""}


SESSION_OLD = [
    "Курс BTC 97 000 долларов на бирже",
    "Выручка выросла на 12% в Q3 2025",
    "Релиз v2.4.1 выйдет 12 мая",
    "Сервер работает стабильно, аптайм 30 дней",
    "Мигрировали бэкенд на Python 3.12",
    "Пользователь просил добавить экспорт в CSV",
    "База PostgreSQL 16 — основная БД проекта",
    "Статус деплоя: успешно, заняло 14 минут",
]

# dict-версия (с id) — для проверок окна и рефов
SESSION = [msg(i, t) for i, t in enumerate(SESSION_OLD)]


def make_session(n=40, seed="сообщение"):
    """Сессия из n сообщений (для проверки иерархии и размера)."""
    return [msg(i, f"{seed} {i}: деталь {i * 7} и факт курса BTC {97000 + i}") for i in range(n)]


# ============================================================================
# HierarchicalCompactor
# ============================================================================

def test_compact_window_is_last_n(store):
    comp = HierarchicalCompactor(store=store)
    window, refs = comp.compact(SESSION, keep_recent=3)
    assert len(window) == 3
    assert [m["id"] for m in window] == ["m005", "m006", "m007"]
    assert [m["content"] for m in window] == SESSION_OLD[5:8]
    # все старые сообщения зареферированы как сырые
    raw_ids = [r["message_id"] for r in refs if r["type"] == "raw"]
    assert len(raw_ids) == 5
    assert raw_ids == ["m000", "m001", "m002", "m003", "m004"]
    assert any(r["type"] == "summary" for r in refs)


def test_compact_unloads_to_store(store):
    """Выгрузка в память работает: узлы kind='context_summary' в Store."""
    comp = HierarchicalCompactor(store=store)
    before = len(store)
    window, refs = comp.compact(SESSION, keep_recent=2)
    assert len(store) > before
    summaries = [r for r in refs if r["type"] == "summary"]
    assert summaries, "должны появиться рефы на сводки"
    kinds = {n["kind"] for n in store.all()}
    assert "context_summary" in kinds
    # node_id рефов реально лежат в Store
    ids = {n["id"] for n in store.all()}
    for s in summaries:
        assert s["node_id"] in ids
        assert s["kind"] == "context_summary"
    assert window[-1]["id"] == "m007"  # окно = последние 2


def test_compact_preserves_key_facts(store):
    """Сжатие сохраняет ключевые факты — поиск в Store их находит."""
    comp = HierarchicalCompactor(store=store)
    comp.compact(SESSION_OLD, keep_recent=2)
    # факты из старых сообщений должны находиться по сводкам
    for query in ("97000", "BTC", "12%", "v2.4.1"):
        hits = store.search(query, top_k=5)
        assert hits, f"факт {query!r} не найден в памяти после сжатия"
        assert any(
            n.get("kind") == "context_summary" for n in hits
        ), f"факт {query!r} найден, но не в сводке"
        assert any(query.lower() in str(n.get("claim", "")).lower() for n in hits)


def test_compact_hierarchy_levels(store):
    """При длинной сессии строится пирамида: есть сводки уровня >= 1."""
    comp = HierarchicalCompactor(store=store, chunk_size=3, hierarchy_cap=6)
    _, refs = comp.compact(make_session(60), keep_recent=4)
    levels = [r["level"] for r in refs if r["type"] == "summary"]
    assert levels, "нет узлов-сводок"
    assert max(levels) >= 1, f"нет верхних уровней иерархии, уровни: {sorted(set(levels))}"
    assert len(set(levels)) > 1  # уровней больше одного


def test_compact_without_store_returns_raw_refs():
    comp = HierarchicalCompactor()  # без Store
    window, refs = comp.compact(SESSION_OLD, keep_recent=3)
    assert len(window) == 3
    assert all(r["type"] == "raw" for r in refs)
    assert len(refs) == 5


def test_compact_keep_recent_ge_len(store):
    comp = HierarchicalCompactor(store=store)
    window, refs = comp.compact(SESSION_OLD, keep_recent=100)
    assert len(window) == len(SESSION_OLD)
    assert [m["content"] for m in window] == SESSION_OLD
    assert refs == []


def test_compact_empty(store):
    comp = HierarchicalCompactor(store=store)
    window, refs = comp.compact([])
    assert window == [] and refs == []


def test_compact_accepts_strings_and_nodes(store):
    """Вход — строки и MemoryNode: нормализация не падает."""
    from mnemos.model import make_node
    comp = HierarchicalCompactor(store=store)
    messages = ["просто строка", make_node(claim="узел памяти 42", source="тест")]
    window, refs = comp.compact(messages, keep_recent=0)
    assert len(window) == 0
    assert any(r["type"] == "summary" for r in refs)


def test_compact_accepts_memorynode_input(store):
    """Сессия из MemoryNode (память как источник истории)."""
    from mnemos.model import make_node
    comp = HierarchicalCompactor(store=store)
    nodes = [make_node(claim=f"факт {i} курса BTC {90000 + i}", source="лог") for i in range(12)]
    window, refs = comp.compact(nodes, keep_recent=2)
    assert len(window) == 2
    assert any(r["type"] == "summary" for r in refs)


# ============================================================================
# Гейты перед выгрузкой (хук в compactor)
# ============================================================================

def test_compact_runs_gates_before_store(store):
    """Каждая сводка проходит гейты; вердикт виден в рефах."""
    comp = HierarchicalCompactor(store=store, gates_enabled=True)
    _, refs = comp.compact(SESSION_OLD, keep_recent=2)
    summaries = [r for r in refs if r["type"] == "summary"]
    assert summaries
    for s in summaries:
        assert s["gate"] in ("pass", "flag"), f"гейт не пропустил сводку: {s}"


def test_gate_hook_rejects_bad_summary(store):
    """Хук гейтов: зыбкая сводка без источника отклоняется (reject)."""
    comp = HierarchicalCompactor(store=store, gates_enabled=True)
    verdict = comp._gate_node(
        {
            "claim": "Кажется, вчера деплой прошёл успешно",
            "kind": "context_summary",
            "source": "",
            "ts": NOW.isoformat(timespec="milliseconds"),
        }
    )
    assert verdict["verdict"] == "reject"


def test_gate_hook_off_when_disabled(store):
    comp = HierarchicalCompactor(store=store, gates_enabled=False)
    verdict = comp._gate_node(
        {
            "claim": "Кажется, вчера деплой прошёл успешно",
            "kind": "context_summary",
            "source": "",
            "ts": NOW.isoformat(timespec="milliseconds"),
        }
    )
    assert verdict["verdict"] == "pass"  # гейты отключены — запись не мешаем


# ============================================================================
# mnemos.gates: пайплайны записи и чтения
# ============================================================================

def test_run_write_gates_good_node_passes():
    node = {
        "claim": "Выручка выросла на 12% в Q3 2025",
        "kind": "fact",
        "source": "финотчёт компании за Q3",
        "ts": NOW.isoformat(timespec="milliseconds"),
        "links": [],
    }
    res = run_write_gates(node, registry={}, now=NOW)
    assert res.verdict == "pass"


def test_run_write_gates_hedge_flags():
    node = {
        "claim": "Возможно, перейдём на PostgreSQL 17",
        "kind": "hypothesis",
        "source": "обсуждение в команде",
        "ts": NOW.isoformat(timespec="milliseconds"),
    }
    res = run_write_gates(node, registry={}, now=NOW)
    assert res.verdict == "pass"  # hypothesis пропускается всегда


def test_run_write_gates_rumor_rejects():
    node = {
        "claim": "Конкурент подал на банкротство 2 млрд долга",
        "kind": "fact",
        "source": "кто-то сказал в чате",
        "ts": NOW.isoformat(timespec="milliseconds"),
    }
    res = run_write_gates(node, registry={}, now=NOW)
    assert res.verdict == "reject"


def test_run_read_gates_filters_irrelevant():
    now = NOW.isoformat(timespec="milliseconds")
    relevant = {"claim": "Курс BTC: 97 000$ на 12:00", "source": "биржа",
                "context": "утренний обзор", "ts": now}
    irrelevant = {"claim": "Погода в Москве: +15°C", "source": "метео",
                  "context": "прогноз на неделю", "ts": now}
    out = run_read_gates([relevant, irrelevant], query="курс BTC", now=NOW)
    assert out == [relevant]


# ============================================================================
# CanonicalPrefix
# ============================================================================

SYSTEM = "Ты — инженер ShineMnemos."
STATIC = ["Всегда отвечай на русском.", "Пиши компактно, без воды."]


def test_prefix_repeat_identical():
    """Префикс стабилен: повторный build_prefix байт-в-байт идентичен."""
    p = CanonicalPrefix()
    a = p.build_prefix(SYSTEM, STATIC)
    b = p.build_prefix(SYSTEM, STATIC)
    assert a == b
    assert a["hash"] == b["hash"]
    assert a["text"] == b["text"]
    assert a["tokens"] == b["tokens"] == estimate_tokens(a["text"])


def test_prefix_cross_instance_identical():
    p1, p2 = CanonicalPrefix(), CanonicalPrefix()
    assert p1.build_prefix(SYSTEM, STATIC)["hash"] == p2.build_prefix(SYSTEM, STATIC)["hash"]


def test_prefix_append_only_keeps_head():
    """Хвост растёт только добавлением; шапка не меняется."""
    p = CanonicalPrefix()
    first = p.build_prefix(SYSTEM, STATIC)
    p.append_tail("Память: пользователь предпочитает краткие ответы")
    second = p.build_prefix(SYSTEM, STATIC)
    assert second["head_hash"] == first["head_hash"]  # шапка неизменна
    assert second["tail_hash"] != first["tail_hash"]  # хвост вырос
    assert "Память: пользователь предпочитает краткие ответы" in second["text"]
    assert first["hash"] != second["hash"]


def test_prefix_check_stability_ok():
    p = CanonicalPrefix()
    p.build_prefix(SYSTEM, STATIC)
    st = p.check_stability(SYSTEM, STATIC)
    assert st["stable"] is True
    assert st["head_stable"] is True and st["tail_append_only"] is True


def test_prefix_check_stability_head_change():
    p = CanonicalPrefix()
    p.build_prefix(SYSTEM, STATIC)
    st = p.check_stability("Другой системный промпт", STATIC)
    assert st["stable"] is False and st["head_stable"] is False
    assert any("ШАПКА" in r for r in st["reasons"])


def test_prefix_check_stability_tail_mutation():
    p = CanonicalPrefix()
    p.build_prefix(SYSTEM, STATIC)
    p.append_tail("блок 1")
    p.append_tail("блок 2")
    st = p.check_stability(SYSTEM, STATIC, tail_blocks=["блок 1", "вставка", "блок 2"])
    assert st["stable"] is False and st["tail_append_only"] is False
    st2 = p.check_stability(SYSTEM, STATIC, tail_blocks=["блок 1", "блок 2", "блок 3"])
    assert st2["stable"] is True  # append-only — ок


def test_prefix_tokens_positive():
    p = CanonicalPrefix()
    out = p.build_prefix(SYSTEM, STATIC)
    assert out["tokens"] >= 1


# ============================================================================
# ContextDefragmenter
# ============================================================================

def test_defrag_reduces_size(store):
    """Дефрагментация уменьшает размер: новая сессия короче старой."""
    session = make_session(40)
    old_tokens = sum(estimate_tokens(m["content"]) for m in session)
    d = ContextDefragmenter()
    out = d.defragment(session, store=store, keep_recent=4)
    assert out["old_tokens"] == old_tokens
    assert out["new_tokens"] < out["old_tokens"]
    assert 0.0 < out["saved_ratio"] < 1.0


def test_defrag_returns_clean_starter_context(store):
    """Новый стартовый контекст: сводка-преамбула + компактное окно."""
    session = make_session(30)
    out = ContextDefragmenter().defragment(session, store=store, keep_recent=4)
    ctx = out["new_context"]
    assert ctx[0]["id"] == "memory_preamble"
    assert ctx[0]["content"].startswith("Память из предыдущей сессии")
    assert len(ctx) == 1 + 4  # преамбула + окно
    assert out["preamble"] == ctx[0]["content"]


def test_defrag_preamble_references_store_nodes(store):
    """Сводка-преамбула ссылается на узлы, реально лежащие в Store."""
    session = make_session(30)
    out = ContextDefragmenter().defragment(session, store=store, keep_recent=4)
    ids = {n["id"] for n in store.all()}
    summary_refs = [r for r in out["memory_refs"] if r["type"] == "summary"]
    assert summary_refs
    assert all(r["node_id"] in ids for r in summary_refs)
    # в преамбуле упомянут node_id хотя бы первой сводки (лимит строки)
    assert f"[id={summary_refs[0]['node_id']}]" in out["preamble"]


def test_defrag_short_session_no_op(store):
    """Короткая сессия (всё в окне) — память не трогается."""
    session = make_session(5)
    out = ContextDefragmenter().defragment(session, store=store, keep_recent=10)
    assert out["new_tokens"] == out["old_tokens"] + estimate_tokens(out["preamble"])
    assert len(out["memory_refs"]) == 0


# ============================================================================
# estimate_tokens
# ============================================================================

def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("привет") == 2  # 6 символов / 4 -> ceil
    assert estimate_tokens("a" * 100) == 25
