# -*- coding: utf-8 -*-
"""Тесты MCP-сервера: initialize, tools/list, tools/call (add/verify/search)."""

import json

from mnemos import Store, make_node
from mnemos.server import PROTOCOL_VERSION, SERVER_NAME, SERVER_VERSION


def test_initialize(rpc):
    resp = rpc("initialize", {"protocolVersion": PROTOCOL_VERSION,
                              "capabilities": {}, "clientInfo": {"name": "test"}})
    assert "error" not in resp
    result = resp["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == SERVER_NAME
    assert result["serverInfo"]["version"] == SERVER_VERSION


def test_tools_list_has_all_tools(rpc):
    resp = rpc("tools/list")
    tools = {t["name"]: t for t in resp["result"]["tools"]}
    # ядро памяти — подмножество: сверх него список зависит от включённых
    # плагинов (context_engine даёт context_*), а с 01.09 в ядро добавлены
    # инструменты уборки и связывания (аудит §5.6). Жёсткое равенство
    # ломалось при любом расширении и падало ещё до правок 01.09.
    assert {
        "memory_add", "memory_verify", "memory_search",
        "memory_rewrite", "memory_reinforce", "memory_link",
        "memory_link_existing", "memory_decay", "memory_prune",
        "memory_summarize", "memory_stats",
    } <= set(tools)
    assert tools["memory_add"]["inputSchema"]["required"] == ["claim"]
    assert tools["memory_prune"]["inputSchema"]["required"] == ["rule"]
    assert tools["memory_link_existing"]["inputSchema"]["required"] == ["from_id", "to_id"]
    assert tools["memory_verify"]["inputSchema"]["required"] == ["node_id"]
    assert tools["memory_search"]["inputSchema"]["required"] == ["query"]
    assert tools["memory_rewrite"]["inputSchema"]["required"] == ["node_id", "new_claim"]
    assert tools["memory_link"]["inputSchema"]["required"] == ["parent_id", "claim"]


def test_ping(rpc):
    resp = rpc("ping")
    assert resp["result"] == {}


def test_unknown_method_error(rpc):
    resp = rpc("no_such_method")
    assert resp["error"]["code"] == -32601


def test_memory_rewrite_via_rpc(rpc, good_claim):
    added = rpc("tools/call", {"name": "memory_add", "arguments": good_claim})
    node = json.loads(added["result"]["content"][0]["text"])
    resp = rpc("tools/call", {
        "name": "memory_rewrite",
        "arguments": {
            "node_id": node["id"],
            "new_claim": "Выручка выросла на 8% в Q3 2025",
            "reason": "уточнённый отчёт",
        },
    })
    assert "error" not in resp
    out = json.loads(resp["result"]["content"][0]["text"])
    assert out["id"] == node["id"]
    assert out["claim"] == "Выручка выросла на 8% в Q3 2025"
    assert len(out["revisions"]) == 1
    assert out["revisions"][0]["claim_before"] == good_claim["claim"]
    assert out["truth_check"]["verdict"] in ("pass", "fail")


def test_memory_link_and_reinforce_via_rpc(rpc, good_claim):
    added = rpc("tools/call", {"name": "memory_add", "arguments": good_claim})
    parent = json.loads(added["result"]["content"][0]["text"])
    link = rpc("tools/call", {
        "name": "memory_link",
        "arguments": {"parent_id": parent["id"], "claim": "Дочерний факт 42",
                      "source": "s", "context": "c"},
    })
    assert "error" not in link
    out = json.loads(link["result"]["content"][0]["text"])
    assert out["parent_id"] == parent["id"]
    assert out["depth"] == 1
    child_id = out["node"]["id"]
    # подкрепление ребёнка
    r = rpc("tools/call", {
        "name": "memory_reinforce",
        "arguments": {"node_id": child_id, "delta": 0.1},
    })
    assert "error" not in r
    assert json.loads(r["result"]["content"][0]["text"])["weight"] >= 1.0


def test_memory_link_too_deep_via_rpc(rpc, good_claim):
    added = rpc("tools/call", {"name": "memory_add", "arguments": good_claim})
    root = json.loads(added["result"]["content"][0]["text"])
    cur = root["id"]
    # строим цепочку до лимита глубины
    for i in range(5):
        resp = rpc("tools/call", {
            "name": "memory_link",
            "arguments": {"parent_id": cur, "claim": f"level {i} 1",
                          "source": "s", "context": "c"},
        })
        cur = json.loads(resp["result"]["content"][0]["text"])["node"]["id"]
    over = rpc("tools/call", {
        "name": "memory_link",
        "arguments": {"parent_id": cur, "claim": "level 6 1",
                      "source": "s", "context": "c"},
    })
    assert over["error"]["code"] == -32602


def test_notification_returns_no_id(server):
    import urllib.request

    port = server.server_address[1]
    body = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 202
        assert resp.read() == b""


def test_invalid_json_parse_error(server):
    import urllib.request

    port = server.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/", data=b"{not json",
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = json.loads(e.read().decode("utf-8"))
    assert body["error"]["code"] == -32700


def test_memory_add_returns_node_with_truth_check(rpc):
    resp = rpc("tools/call", {
        "name": "memory_add",
        "arguments": {
            "claim": "Выручка выросла на 12% в Q3 2025",
            "source": "финотчёт",
            "evidence": ["таблица 2"],
            "context": "квартальный обзор",
        },
    })
    assert "error" not in resp, resp
    node = json.loads(resp["result"]["content"][0]["text"])
    assert node["claim"].startswith("Выручка")
    assert node["id"].startswith("mn_")
    assert node["truth_check"]["verdict"] == "pass"
    assert node["truth_check"]["score"] >= 4


def test_memory_add_requires_claim(rpc):
    resp = rpc("tools/call", {"name": "memory_add", "arguments": {}})
    assert resp["error"]["code"] == -32602
    assert "claim" in resp["error"]["message"]


def test_memory_verify_returns_verdict(rpc):
    added = rpc("tools/call", {
        "name": "memory_add",
        "arguments": {"claim": "Цена выросла на 5%", "source": "биржа",
                      "evidence": ["тикер X"], "context": "дневной график"},
    })
    node_id = json.loads(added["result"]["content"][0]["text"])["id"]
    resp = rpc("tools/call", {"name": "memory_verify", "arguments": {"node_id": node_id}})
    tc = json.loads(resp["result"]["content"][0]["text"])
    assert tc["verdict"] in ("pass", "fail")
    assert 0 <= tc["score"] <= 6
    assert "P1" in tc and "summary" in tc


def test_memory_verify_unknown_node_error(rpc):
    resp = rpc("tools/call", {"name": "memory_verify",
                              "arguments": {"node_id": "mn_nonexistent"}})
    assert resp["error"]["code"] == -32002


def test_memory_verify_missing_param(rpc):
    resp = rpc("tools/call", {"name": "memory_verify", "arguments": {}})
    assert resp["error"]["code"] == -32602


def test_memory_search_substring(rpc):
    rpc("tools/call", {"name": "memory_add", "arguments": {
        "claim": "Курс BTC вырос на 3% за день", "source": "биржа",
        "evidence": ["свечи"], "context": "крипта"}})
    rpc("tools/call", {"name": "memory_add", "arguments": {
        "claim": "Команда провела ретроспективу", "source": "журнал",
        "evidence": ["заметки"], "context": "процесс"}})
    resp = rpc("tools/call", {"name": "memory_search",
                              "arguments": {"query": "BTC"}})
    out = json.loads(resp["result"]["content"][0]["text"])
    assert out["count"] == 1
    assert out["results"][0]["claim"].startswith("Курс BTC")


def test_memory_search_topk_limit(rpc):
    for i in range(5):
        rpc("tools/call", {"name": "memory_add", "arguments": {
            "claim": f"Факт номер {i}: сервис отвечает за {i * 10} мс",
            "source": "тесты", "evidence": ["лог"], "context": "перф"}})
    resp = rpc("tools/call", {"name": "memory_search",
                              "arguments": {"query": "Факт", "top_k": 3}})
    out = json.loads(resp["result"]["content"][0]["text"])
    assert out["count"] == 3
    assert len(out["results"]) == 3


def test_memory_search_no_results(rpc):
    resp = rpc("tools/call", {"name": "memory_search",
                              "arguments": {"query": "несуществующееслово"}})
    out = json.loads(resp["result"]["content"][0]["text"])
    assert out["count"] == 0
    assert out["results"] == []


def test_memory_search_requires_query(rpc):
    resp = rpc("tools/call", {"name": "memory_search", "arguments": {}})
    assert resp["error"]["code"] == -32602


def test_memory_search_invalid_mode(rpc):
    # бенчмарк-фикс 26.08: mode принимает только substring|semantic
    resp = rpc("tools/call", {"name": "memory_search", "arguments": {
        "query": "x", "mode": "vector"}})
    assert resp["error"]["code"] == -32602
    assert "substring" in resp["error"]["message"]
    assert "semantic" in resp["error"]["message"]


def test_unknown_tool_error(rpc):
    resp = rpc("tools/call", {"name": "memory_frobnicate", "arguments": {}})
    assert resp["error"]["code"] == -32602


def test_health_endpoint(server):
    import urllib.request

    port = server.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    assert data["ok"] is True
    assert data["name"] == SERVER_NAME
    assert data["nodes"] == 0


def test_persistence_across_server_restart(tmp_path):
    store = Store(tmp_path / "nodes.json")
    node = make_node(claim="Выручка +12% за Q3", source="отчёт",
                     evidence=["табл.2"], context="квартал")
    store.add(node.to_dict())
    store2 = Store(tmp_path / "nodes.json")  # переоткрытие
    assert store2.get(node.id)["claim"] == "Выручка +12% за Q3"


def test_store_search_fresh_first(tmp_path):
    from mnemos import Store

    store = Store(tmp_path / "nodes.json")
    old = make_node(claim="Старый факт: 1 сервер", source="архив",
                    evidence=["лог 2020"], context="история",
                    ts="2020-01-01T00:00:00+00:00")
    new = make_node(claim="Новый факт: 2 сервера", source="инфра",
                    evidence=["лог 2025"], context="текущее",
                    ts="2025-01-01T00:00:00+00:00")
    store.add(old.to_dict())
    store.add(new.to_dict())
    results = store.search("сервер", top_k=5)
    assert results[0]["claim"].startswith("Новый факт")
