# -*- coding: utf-8 -*-
"""Brick-2 (пластичность узлов + «граф в узле»): тесты.

Идея брата 25.08: в мозге нейроны переписывают связи, а у статичных
ИИ-узлов нет. ShineMnemos: узлы живут — переписываются (rewrite),
подкрепляются (reinforce), затухают (decay) и содержат дочерние
под-графы (структурная рекурсия с ограниченной глубиной).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mnemos.model import (
    DEFAULT_HALF_LIFE_HOURS,
    WEIGHT_MAX,
    WEIGHT_MIN,
    MemoryNode,
    make_node,
)
from mnemos.store import MAX_DEPTH, Store


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


# --- rewrite: переписывание узла новым фактом -------------------------------

def test_rewrite_keeps_id_and_pushes_history():
    node = make_node(claim="VPS №1 жив", source="мониторинг")
    node_id, first_claim = node.id, node.claim
    node.rewrite("VPS №1 жив, но sshd под нагрузкой", source="мониторинг", reason="новый факт")
    assert node.id == node_id
    assert node.claim == "VPS №1 жив, но sshd под нагрузкой"
    assert len(node.revisions) == 1
    rev = node.revisions[0]
    assert rev["claim_before"] == first_claim
    assert rev["claim_after"] == "VPS №1 жив, но sshd под нагрузкой"
    assert rev["reason"] == "новый факт"
    assert rev["source"] == "мониторинг"
    assert _dt(node.ts) >= _dt(rev["ts"])  # ts узла обновлён


def test_rewrite_same_claim_is_noop():
    node = make_node(claim="A", source="s")
    node.rewrite("A")
    assert node.revisions == []


def test_rewrite_empty_claim_rejected():
    node = make_node(claim="A", source="s")
    with pytest.raises(ValueError):
        node.rewrite("   ")


def test_multiple_rewrites_keep_order():
    node = make_node(claim="v1", source="s")
    node.rewrite("v2", reason="r2")
    node.rewrite("v3", reason="r3")
    assert [r["claim_before"] for r in node.revisions] == ["v1", "v2"]
    assert [r["claim_after"] for r in node.revisions] == ["v2", "v3"]
    assert node.claim == "v3"


# --- reinforce/decay: вес как синапс ----------------------------------------

def test_reinforce_clamps_at_max():
    node = make_node(claim="A", source="s")
    node.weight = WEIGHT_MAX
    assert node.reinforce(0.5) == WEIGHT_MAX
    assert node.weight == WEIGHT_MAX


def test_reinforce_updates_last_used():
    node = make_node(claim="A", source="s")
    old = node.last_used
    node.reinforce(0.1)
    assert node.weight == 1.0  # уже на максимуме — вес не растёт
    assert node.last_used >= old


def test_decay_halves_weight_after_half_life():
    node = make_node(claim="A", source="s")
    node.weight = 1.0
    node.last_used = "2026-08-01T00:00:00.000+00:00"
    now = _dt("2026-08-01T00:00:00.000+00:00") + timedelta(hours=DEFAULT_HALF_LIFE_HOURS)
    w = node.decay(now_dt=now)
    assert abs(w - 0.5) < 1e-9
    assert abs(node.weight - 0.5) < 1e-9


def test_decay_floors_at_weight_min():
    node = make_node(claim="A", source="s")
    node.weight = 1.0
    node.last_used = "2026-01-01T00:00:00.000+00:00"
    w = node.decay(now_dt=_dt("2026-09-01T00:00:00.000+00:00"))
    assert w == WEIGHT_MIN


def test_decay_does_not_move_last_used():
    node = make_node(claim="A", source="s")
    lu = "2026-08-01T00:00:00.000+00:00"
    node.last_used = lu
    node.decay(now_dt=_dt("2026-08-10T00:00:00.000+00:00"))
    assert node.last_used == lu


def test_decay_with_past_now_is_noop():
    node = make_node(claim="A", source="s")
    node.weight = 0.8
    w = node.decay(now_dt=_dt(node.last_used) - timedelta(hours=1))
    assert w == 0.8


def test_weight_out_of_range_rejected():
    node = make_node(claim="A", source="s")
    node.weight = 1.5
    with pytest.raises(ValueError):
        node.validate()


# --- сериализация: совместимость со старыми узлами ---------------------------

def test_old_node_without_plasticity_fields_loads():
    data = {
        "id": "mn_old",
        "kind": "fact",
        "claim": "старое знание",
        "source": "s",
        "ts": "2026-08-01T00:00:00.000+00:00",
        "truth_check": {"verdict": "pass", "score": 5},
    }
    node = MemoryNode.from_dict(data)
    assert node.weight == WEIGHT_MAX
    assert node.revisions == []
    assert node.children == []
    assert node.parent is None
    assert node.last_used == data["ts"]


def test_to_dict_roundtrip_with_plasticity():
    node = make_node(claim="A", source="s")
    node.reinforce(0.0)
    node.rewrite("B", reason="r")
    child = make_node(claim="child", source="s")
    child.parent = node.id
    node.children.append(child.id)
    d = node.to_dict()
    assert set(d) >= {"weight", "last_used", "revisions", "children", "parent"}
    back = MemoryNode.from_dict(d)
    assert back.weight == node.weight
    assert back.revisions == node.revisions
    assert back.children == [child.id]
    assert back.parent is None


def test_self_child_rejected():
    node = make_node(claim="A", source="s")
    node.children.append(node.id)
    with pytest.raises(ValueError):
        node.validate()


# --- store: пластичность -----------------------------------------------------

def test_store_rewrite_persists(tmp_path):
    st = Store(str(tmp_path / "nodes.json"))
    node = make_node(claim="v1", source="s")
    st.add(node)
    st.rewrite(node.id, "v2", source="s2", reason="new")
    reloaded = Store(str(tmp_path / "nodes.json"))
    got = reloaded.get(node.id)
    assert got["claim"] == "v2"
    assert len(got["revisions"]) == 1
    assert got["revisions"][0]["claim_before"] == "v1"


def test_store_reinforce_and_decay_persist(tmp_path):
    st = Store(str(tmp_path / "nodes.json"))
    node = make_node(claim="A", source="s")
    node.weight = 0.5
    st.add(node)
    assert st.reinforce(node.id, 0.2) == 0.7
    st.decay(half_life_hours=1.0)
    reloaded = Store(str(tmp_path / "nodes.json"))
    w = reloaded.get(node.id)["weight"]
    assert WEIGHT_MIN <= w < 0.7  # затухло после подкрепления


def test_store_rewrite_unknown_raises(tmp_path):
    st = Store(str(tmp_path / "nodes.json"))
    with pytest.raises(KeyError):
        st.rewrite("mn_missing", "new")


# --- store: «граф в узле» ----------------------------------------------------

def test_add_child_sets_parent_and_children(tmp_path):
    st = Store(str(tmp_path / "nodes.json"))
    parent = make_node(claim="граф", source="s")
    st.add(parent)
    child = make_node(claim="под-узел", source="s")
    st.add_child(parent.id, child)
    assert st.get(child.id)["parent"] == parent.id
    assert child.id in st.get(parent.id)["children"]
    assert [c["id"] for c in st.children(parent.id)] == [child.id]
    assert st.depth(child.id) == 1
    assert st.ancestors(child.id) == [parent.id]


def test_add_child_cycle_rejected(tmp_path):
    st = Store(str(tmp_path / "nodes.json"))
    parent = make_node(claim="граф", source="s")
    st.add(parent)
    # поддельный узел с id предка — самопетля должна быть отклонена
    fake = MemoryNode(id=parent.id, claim="самопетля", source="s")
    with pytest.raises(ValueError, match="цикл"):
        st.add_child(parent.id, fake)
    # и «родитель ребёнка» тоже: узел с id ребёнка не может стать
    # родителем ребёнка (цикл длины 2) — через прямую самопетлю на ребёнке
    child = make_node(claim="под-узел", source="s")
    st.add_child(parent.id, child)
    fake2 = MemoryNode(id=child.id, claim="цикл", source="s")
    with pytest.raises(ValueError, match="цикл"):
        st.add_child(child.id, fake2)


def test_add_child_existing_node_rejected(tmp_path):
    st = Store(str(tmp_path / "nodes.json"))
    a = make_node(claim="A", source="s")
    b = make_node(claim="B", source="s")
    st.add(a)
    st.add(b)
    with pytest.raises(ValueError):
        st.add_child(a.id, b)


def test_add_child_depth_limit(tmp_path):
    st = Store(str(tmp_path / "nodes.json"))
    root = make_node(claim="root", source="s")
    st.add(root)
    cur = root.id
    for i in range(MAX_DEPTH):
        child = make_node(claim=f"level {i + 1}", source="s")
        st.add_child(cur, child)
        cur = child.id
    assert st.depth(cur) == MAX_DEPTH
    # следующий уровень — перебор
    too_deep = make_node(claim="too deep", source="s")
    with pytest.raises(ValueError, match="глубина"):
        st.add_child(cur, too_deep)


def test_add_child_unknown_parent_raises(tmp_path):
    st = Store(str(tmp_path / "nodes.json"))
    child = make_node(claim="сирота", source="s")
    with pytest.raises(KeyError):
        st.add_child("mn_missing", child)


def test_delete_repairs_parent_links(tmp_path):
    st = Store(str(tmp_path / "nodes.json"))
    parent = make_node(claim="граф", source="s")
    st.add(parent)
    child = make_node(claim="под-узел", source="s")
    st.add_child(parent.id, child)
    assert st.delete(child.id)
    assert st.get(parent.id)["children"] == []
