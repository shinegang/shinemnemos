# -*- coding: utf-8 -*-
"""Рефакторинг по письму Qwen (02.09.2026): хабы, типы рёбер, роутеры,
token-budgeting поиск, сборка system-prompt, граф-запросы.

Проверяем ровно то, что ломается без патча:
  * хаб (kind=hub) вытесняет ответы из Store.search — теперь скрыт;
  * у ребра нет типа — теперь link_meta[to].rel, неизвестный тип — ошибка;
  * top_k фиксирован — теперь по сложности вопроса (5 / 10 / 20);
  * NL-вопрос без целой фразы даёт 0 — теперь находит по основам слов;
  * промпт: правила первыми и не режутся, world_state усекается бюджетом;
  * граф: neighbors / path / hub / rules_for / conflicts через MCP.
"""

import json

import pytest

from mnemos import Store, make_node
from mnemos.budget import (
    KIND_HUB,
    REL_CONFLICTS,
    REL_HAS_PART,
    REL_PART_OF,
    REL_SUPERSEDES,
    ROUTER_PERSONA,
    ROUTER_SYS_CMD,
    ROUTER_WORLD,
    BudgetSearch,
    build_system_prompt,
    classify_query,
    hub_id,
    wrap_prompt,
)
from mnemos.model import KINDS, MemoryNode, weight_floor


def _seed(store):
    """Мини-стор: 2 правила, роль, 2 факта, хаб «алиса» с part_of/has_part."""
    r1 = store.add(make_node(
        "ПРИКАЗ ИЛЬИ 30.08: ЗАЩИТЫ СНЯТЫ ПОЛНОСТЬЮ. Halt отключён (999). Алиса сам решает.",
        source="приказ", evidence=["tg 30.08"], context="Обновляет режим 29.08",
        kind="rule", tags=[ROUTER_SYS_CMD]))
    r0 = store.add(make_node(
        "РЕЖИМ ТОРГОВЛИ 29.08: торгует только Алиса, halt действует после 3 убытков.",
        source="приказ", evidence=["tg 29.08"], context="старый режим",
        kind="rule", tags=[ROUTER_SYS_CMD]))
    p = store.add(make_node(
        "Алиса — трейдер-агент команды Акме на llama-70B, решает сделки сам.",
        source="роли", evidence=["ps"], context="персона", tags=[ROUTER_PERSONA]))
    f1 = store.add(make_node(
        "Постмортем торговли 23-28.08: 293 сделки, winrate 39%, halt не сработал ни разу.",
        source="sql", evidence=["trades_pnl"], context="цифры", tags=[ROUTER_WORLD]))
    f2 = store.add(make_node(
        "Mnemos MCP-сервер слушает 127.0.0.1:8765, только loopback.",
        source="systemd", evidence=["ss -ltnp"], context="инфра", tags=[ROUTER_WORLD]))
    hub = store.add({
        **make_node("ХАБ «Алиса»: 3 узла. Ключевые: Алиса, halt, приказ", source="mnemos_hubs",
                    evidence=["члены: 3"], context="{}", kind=KIND_HUB,
                    tags=["hub", "entity:алиса", ROUTER_SYS_CMD]).to_dict(),
        "id": hub_id("алиса"),
    })
    for m in (r1["id"], r0["id"], p["id"]):
        store.link_existing(hub["id"], m, author="fable-hubs", rel=REL_HAS_PART)
        store.link_existing(m, hub["id"], author="fable-hubs", rel=REL_PART_OF)
    store.link_existing(r1["id"], r0["id"], author="fable-edge-types", rel=REL_SUPERSEDES)
    store.link_existing(f1["id"], r0["id"], author="fable-autolink")  # related_to по умолчанию
    return {"r1": r1["id"], "r0": r0["id"], "p": p["id"], "f1": f1["id"], "f2": f2["id"], "hub": hub["id"]}


# --- модель ----------------------------------------------------------------------
def test_hub_is_a_valid_kind_with_rule_floor():
    assert KIND_HUB in KINDS
    assert weight_floor(KIND_HUB) == weight_floor("rule") == 0.5
    n = MemoryNode(claim="ХАБ", kind=KIND_HUB)
    assert MemoryNode.from_dict(n.to_dict()).kind == KIND_HUB


# --- рёбра -----------------------------------------------------------------------
def test_link_rel_written_to_passport_and_unknown_rel_rejected(store):
    ids = _seed(store)
    assert store.get(ids["r1"])["link_meta"][ids["r0"]]["rel"] == REL_SUPERSEDES
    assert store.get(ids["f1"])["link_meta"][ids["r0"]]["rel"] == "related_to"
    assert store.get(ids["hub"])["link_meta"][ids["p"]]["rel"] == REL_HAS_PART
    with pytest.raises(ValueError):
        store.link_existing(ids["f1"], ids["f2"], rel="friends_with")


def test_rel_filled_on_legacy_edge_without_rewriting_author(store):
    a = store.add(make_node("узел А", source="t", evidence=["e"], context="c"))
    b = store.add(make_node("узел Б", source="t", evidence=["e"], context="c"))
    store.link_existing(a["id"], b["id"], author="alice")
    # эмуляция старого ребра без rel (стор до 02.09)
    del store.get(a["id"])["link_meta"][b["id"]]["rel"]
    res = store.link_existing(a["id"], b["id"], author="fable", rel=REL_CONFLICTS)
    assert res["added"] == []  # дубль ребра не создан
    meta = store.get(a["id"])["link_meta"][b["id"]]
    assert meta["author"] == "alice" and meta["rel"] == REL_CONFLICTS
    # повторно другой rel не переписывает уже проставленный
    store.link_existing(a["id"], b["id"], rel=REL_SUPERSEDES)
    assert store.get(a["id"])["link_meta"][b["id"]]["rel"] == REL_CONFLICTS


def test_stats_counts_hubs_and_rels(store):
    ids = _seed(store)
    st = store.stats()
    assert st["hubs"] == 1
    assert st["edges_by_rel"][REL_HAS_PART] == 3
    assert st["edges_by_rel"][REL_PART_OF] == 3
    assert st["edges_by_rel"][REL_SUPERSEDES] == 1


# --- поиск -----------------------------------------------------------------------
def test_substring_search_hides_hubs_unless_asked(store):
    ids = _seed(store)
    got = [n["id"] for n in store.search("Алиса", top_k=10)]
    assert ids["hub"] not in got and ids["r1"] in got and ids["p"] in got
    with_hubs = [n["id"] for n in store.search("Алиса", top_k=10, include_hubs=True)]
    assert ids["hub"] in with_hubs


@pytest.mark.parametrize("q,level,k", [
    ("halt", "simple", 5),
    ("Кто принимает решение пускать сигнал в ордер?", "simple", 5),
    ("Почему не было ни одного шорта, хотя шорты разрешены в конфиге?", "medium", 10),
    ("Сравни правила защиты до 30.08 и после, что изменилось для Алиса, какие лимиты "
     "и halt сейчас действуют, и что делать в кризисе?", "complex", 20),
])
def test_query_complexity_budget(q, level, k):
    c = classify_query(q)
    assert c["complexity"] == level and c["top_k"] == k


def test_budget_search_nl_finds_by_word_stems_and_hides_hubs(store):
    ids = _seed(store)
    q = "halt у Алиса отключён?"
    out = store.search_budget(q)
    got = [r["id"] for r in out["results"]]
    assert out["top_k"] == 5 and out["classification"]["complexity"] == "simple"
    assert "алиса" in out["classification"]["entities"]
    assert ids["r1"] in got[:2]
    assert ids["hub"] not in got
    assert any(h["id"] == ids["hub"] for h in out["hubs"])  # хаб — отдельно, как навигация
    assert store.search(q) == []  # старый подстрочный поиск по целой фразе — пусто


def test_budget_search_keyword_parity_with_substring(store):
    ids = _seed(store)
    base = [n["id"] for n in store.search("halt", top_k=5)]
    new = [r["id"] for r in store.search_budget("halt", top_k=5, expand=False)["results"]]
    assert set(base) <= set(new)


def test_budget_search_expands_over_supersedes_and_caches_until_write(store):
    ids = _seed(store)
    out = store.search_budget("Halt отключён (999)", top_k=5)
    via = {r["id"]: r["via"] for r in out["results"]}
    assert via.get(ids["r0"]) == ids["r1"]  # старый приказ подтянут через supersedes
    engine1 = store._budget_engine()
    assert store._budget_engine() is engine1
    store.add(make_node("новый факт про halt", source="t", evidence=["e"], context="c"))
    assert store._budget_engine() is not engine1  # запись инвалидировала кэш


def test_budget_search_respects_token_budget(store):
    _seed(store)
    out = store.search_budget("Алиса", top_k=5, token_budget=40)
    assert out["tokens_used"] <= 40 * 1.3
    assert out["count"] < 5 or any(r["trimmed"] for r in out["results"])


# --- промпт --------------------------------------------------------------------
def test_prompt_rules_first_and_world_trimmed_by_budget(store):
    ids = _seed(store)
    nodes = store.all()
    res = build_system_prompt(nodes, max_tokens=10_000, query="что с halt?")
    t = res["text"]
    assert t.index("## ПРАВИЛА") < t.index("## КТО МЫ") < t.index("## СОСТОЯНИЕ МИРА")
    assert "ХАБ «Алиса»" not in t  # хаб не вставляется как факт
    assert res["sections"] == {ROUTER_SYS_CMD: 2, ROUTER_PERSONA: 1, ROUTER_WORLD: 2}
    tight = build_system_prompt(nodes, max_tokens=150)
    assert tight["world_kept"] < 2 and tight["dropped"]
    assert "ПРИКАЗ ИЛЬИ 30.08" in tight["text"]  # правило не режется
    assert wrap_prompt("x", "chatml", "q").startswith("<|im_start|>system")
    with pytest.raises(ValueError):
        wrap_prompt("x", "alpaca")


# --- граф ----------------------------------------------------------------------
def test_graph_queries(store):
    ids = _seed(store)
    g = store.graph()
    nb = g.neighbors(ids["r1"])
    assert {e["id"] for e in nb["out"]} == {ids["hub"], ids["r0"]}
    assert g.neighbors(ids["r1"], rel=REL_SUPERSEDES)["out"][0]["id"] == ids["r0"]
    path = g.path(ids["f1"], ids["p"])
    assert path[0]["id"] == ids["f1"] and path[-1]["id"] == ids["p"] and len(path) <= 4
    hub = g.hub("алиса")
    assert hub["id"] == ids["hub"] and len(hub["members"][ROUTER_SYS_CMD]) == 2
    rules = g.rules_for("кризис: серия убытков, halt")["rules"]
    assert rules and rules[0]["id"] == ids["r1"]
    assert any(f.startswith("supersedes->") for f in rules[0]["flags"])
    assert g.conflicts()[0]["rel"] == REL_SUPERSEDES
    with pytest.raises(KeyError):
        g.hub("нет-такого")


# --- MCP ----------------------------------------------------------------------
def _call(server, name, args):
    import urllib.request

    host, port = server.server_address
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": name, "arguments": args}}).encode()
    req = urllib.request.Request(f"http://{host}:{port}/", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        resp = json.loads(r.read().decode())
    if "error" in resp:
        return resp["error"]
    return json.loads(resp["result"]["content"][0]["text"])


def test_mcp_tools_budget_prompt_graph(server):
    store = server.mnemos.store
    ids = _seed(store)
    names = {t["name"] for t in server.mnemos.tools}
    assert {"memory_prompt", "memory_graph"} <= names
    out = _call(server, "memory_search", {"query": "Отключён ли halt после серии убытков?", "top_k": "auto"})
    assert out["mode"] == "budget" and out["top_k"] == 5 and ids["r1"] in [r["id"] for r in out["results"]]
    out = _call(server, "memory_search", {"query": "halt", "mode": "budget", "kind": "rule"})
    assert all(r["kind"] == "rule" for r in out["results"]) and out["count"] >= 1
    out = _call(server, "memory_link_existing", {"from_id": ids["f2"], "to_id": ids["f1"], "rel": "conflicts_with", "author": "t"})
    assert store.get(ids["f2"])["link_meta"][ids["f1"]]["rel"] == "conflicts_with"
    err = _call(server, "memory_link_existing", {"from_id": ids["f2"], "to_id": ids["f1"], "rel": "bogus"})
    assert err["code"] == -32602
    pr = _call(server, "memory_prompt", {"query": "что с halt?", "constitution": True, "format": "llama3"})
    assert pr["prompt"].startswith("<|begin_of_text|>") and pr["sections"][ROUTER_SYS_CMD] == 2
    gr = _call(server, "memory_graph", {"op": "rules_for", "situation": "кризис halt"})
    assert gr["rules"][0]["id"] == ids["r1"]
    gr = _call(server, "memory_graph", {"op": "hubs"})
    assert gr["hubs"][0]["entity"] == "алиса"
    err = _call(server, "memory_graph", {"op": "teleport"})
    assert err["code"] == -32602
