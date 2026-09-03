# -*- coding: utf-8 -*-
"""Фиксы аудита 26.08: регрессионные тесты на топ-6 критичных багов.

Покрывают: B1 (RLock в Store — конкурентные add), B2 (битый nodes.json →
.corrupt-бэкап), B4 (дубликат id → ошибка), B5 (лимит HTTP-тела → 413),
B7 (truth_check: null → pending), B13 (отрицательная дельта reinforce → кламп).
"""

import json
import threading

import pytest

from mnemos import Store, make_node
from mnemos.model import WEIGHT_MAX, WEIGHT_MIN, MemoryNode
from mnemos.server import MAX_BODY_BYTES


# --- B1: конкурентные add не теряют узлы (RLock) -----------------------------

def test_concurrent_adds_lose_no_nodes(tmp_path):
    st = Store(tmp_path / "nodes.json")
    n_threads, per_thread = 4, 25
    errors = []

    def worker(wid):
        try:
            for i in range(per_thread):
                node = make_node(claim=f"w{wid}-{i}", source="s")
                st.add(node)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    expected = n_threads * per_thread
    assert len(st) == expected
    claims = {n["claim"] for n in st.all()}
    assert len(claims) == expected  # ни один узел не потерян и не задвоен
    # и на диске после переоткрытия — тоже всё на месте
    reloaded = Store(tmp_path / "nodes.json")
    assert len(reloaded) == expected


# --- B2: битый nodes.json → .corrupt-<timestamp> + пустой старт ---------------

def test_corrupt_file_backed_up_and_empty_start(tmp_path):
    p = tmp_path / "nodes.json"
    p.write_text("{not json", encoding="utf-8")

    st = Store(p)
    assert len(st) == 0  # старт с пустой памятью, а не с битыми данными

    corrupts = sorted(tmp_path.glob("nodes.json.corrupt-*"))
    assert len(corrupts) == 1
    assert corrupts[0].read_text(encoding="utf-8") == "{not json"  # данные целы
    assert not p.exists()  # битый файл убран с основного пути

    # первый же _save не затирает бэкап
    node = make_node(claim="новая память после аварии", source="s")
    st.add(node)
    assert st.get(node.id)["claim"] == "новая память после аварии"
    assert corrupts[0].read_text(encoding="utf-8") == "{not json"


def test_corrupt_file_non_dict_json_backed_up(tmp_path):
    p = tmp_path / "nodes.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")  # JSON валиден, но не объект
    st = Store(p)
    assert len(st) == 0
    corrupts = list(tmp_path.glob("nodes.json.corrupt-*"))
    assert len(corrupts) == 1
    assert corrupts[0].read_text(encoding="utf-8") == "[1, 2, 3]"


# --- B4: add с дублирующимся id — ошибка, а не тихая перезапись ---------------

def test_add_duplicate_id_raises(tmp_path):
    st = Store(tmp_path / "nodes.json")
    node = make_node(claim="оригинал", source="s")
    st.add(node)

    dup = MemoryNode(id=node.id, claim="подделка", source="s2")
    with pytest.raises(ValueError, match="уже существует"):
        st.add(dup)

    # исходный узел не тронут
    assert st.get(node.id)["claim"] == "оригинал"
    assert st.get(node.id)["source"] == "s"


# --- B5: лимит HTTP-тела → 413 ------------------------------------------------

def test_body_over_limit_returns_413(server):
    import http.client

    port = server.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest("POST", "/")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
    conn.endheaders()  # тело не шлём — сервер должен отказать до чтения
    try:
        resp = conn.getresponse()
        assert resp.status == 413
        body = json.loads(resp.read().decode("utf-8"))
        assert body["error"]["code"] == -32600
    finally:
        conn.close()


def test_body_negative_content_length_not_500(server):
    import http.client

    port = server.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest("POST", "/")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", "-1")
    conn.endheaders()
    try:
        resp = conn.getresponse()
        # отрицательный Content-Length трактуется как 0 → тело пустое → 400
        assert resp.status == 400
        body = json.loads(resp.read().decode("utf-8"))
        assert body["error"]["code"] == -32700
    finally:
        conn.close()


# --- B7: from_dict с truth_check: null не падает ------------------------------

def test_from_dict_truth_check_null_loads():
    data = {
        "id": "mn_nulltc",
        "kind": "fact",
        "claim": "утверждение без truth_check",
        "source": "s",
        "ts": "2026-08-01T00:00:00.000+00:00",
        "truth_check": None,
    }
    node = MemoryNode.from_dict(data)
    assert node.truth_check["verdict"] == "pending"
    assert node.truth_check["score"] == 0
    assert node.truth_check["P1"]["pass"] is None


def test_store_node_with_null_truth_check_roundtrips(tmp_path):
    p = tmp_path / "nodes.json"
    p.write_text(json.dumps({
        "mn_nulltc": {
            "id": "mn_nulltc",
            "kind": "fact",
            "claim": "из ручного JSON",
            "source": "s",
            "ts": "2026-08-01T00:00:00.000+00:00",
            "truth_check": None,
            "weight": 0.5,
            "last_used": "2026-08-01T00:00:00.000+00:00",
            "revisions": [],
            "children": [],
            "parent": None,
            "links": [],
            "evidence": [],
            "context": "",
        }
    }), encoding="utf-8")
    st = Store(p)
    assert len(st) == 1
    # загрузка не падает, узел снова сериализуется
    node = MemoryNode.from_dict(st.get("mn_nulltc"))
    assert node.truth_check["verdict"] == "pending"
    assert st.get("mn_nulltc")["claim"] == "из ручного JSON"


# --- B13: отрицательная дельта reinforce → кламп, а не битый вес --------------

def test_model_reinforce_negative_delta_clamped():
    node = make_node(claim="A", source="s")
    node.weight = 0.5
    w = node.reinforce(-2.0)
    assert w == 0.5  # кламп к 0: вес не изменился
    assert WEIGHT_MIN <= node.weight <= WEIGHT_MAX
    node.to_dict()  # сериализация не падает


def test_store_reinforce_negative_delta_not_crash(tmp_path):
    st = Store(tmp_path / "nodes.json")
    node = make_node(claim="A", source="s")
    node.weight = 0.5
    st.add(node)
    w = st.reinforce(node.id, delta=-2.0)
    assert w == 0.5
    stored = st.get(node.id)
    assert WEIGHT_MIN <= stored["weight"] <= WEIGHT_MAX
    assert stored["claim"] == "A"
