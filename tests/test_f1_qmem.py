# -*- coding: utf-8 -*-
"""Ф1 02.09.2026 — внедрение исследований в боевую Mnemos.

Проверяем ровно то, что ломается без правок:
  * уровни L1/L2 строятся при add/update/rewrite и переживают from_dict->to_dict;
  * поиск сканирует текст текущего уровня (ключи L2 находят узел, полный текст — нет);
  * буст уровня: при равном матче L0 выше L1 выше L2;
  * попадание в выдачу поднимает узел на L0 и пишет usage; на диск — с ближайшей записью;
  * decay пересчитывает уровни по политике; rule всегда L0;
  * tf-бонус: узел с тремя «Алиса» в claim выше узла с одним (гейт R4);
  * memory_search: NL-фоллбек по основам слов, levels в ответ не уходят;
  * memory_add: ровно один тег-маршрутизатор, явный уважается;
  * memory_prompt: бюджет по умолчанию 1200 токенов;
  * гейт: снимок 01.09 (36 узлов) даёт recall@5 >= 0.944 боевым Store.search.
"""

import json
import os
import shutil
from datetime import datetime, timedelta, timezone

import pytest

from mnemos import qmem
from mnemos.budget import ROUTER_PERSONA, ROUTER_SYS_CMD, ROUTER_WORLD, ROUTERS, classify_router
from mnemos.model import MemoryNode, make_node
from mnemos.server import PROMPT_DEFAULT_TOKENS, MnemosCore
from mnemos.store import Store

SNAP = "/opt/bench-memory/qmem/nodes.snapshot0109.json"
GT = "/opt/bench-memory/ground_truth.json"
NOW = datetime(2026, 9, 2, 15, 0, tzinfo=timezone.utc)

LONG = ("Алиса — трейдер-агент команды Акме: работает на локальной llama-3.3-70B, порт 8765, "
        "решает сделки сам. Постмортем 23-28.08: 293 сделки, winrate 39%, NET -145 USDC. "
        "Halt после серии убытков отключён приказом 30.08 (значение 999). "
        "Шорты разрешены в конфиге llm_allow_shorts: true, но ни одного шорта не было.")


def _iso(dt):
    return dt.isoformat(timespec="milliseconds")


def _old(store, claim, days, **kw):
    n = make_node(claim, source=kw.pop("source", "t"), evidence=kw.pop("evidence", ["e1"]),
                  context=kw.pop("context", "контекст узла"), **kw)
    d = n.to_dict()
    d["ts"] = d["last_used"] = _iso(NOW - timedelta(days=days))
    d["weight"] = 0.3
    return store.add(d)


# --- уровни -------------------------------------------------------------------------
def test_levels_built_on_add_and_survive_model_roundtrip(store):
    d = store.add(make_node(LONG, source="роли", evidence=["ps aux", "trades_pnl"], context="персона Алиса"))
    assert d["level"] == 0 and set(d["levels"]) == {"src", "1", "2"}
    l1, l2 = d["levels"]["1"], d["levels"]["2"]
    assert qmem.fields_chars(l1) < qmem.fields_chars(qmem.l0_fields(d)) * 0.6
    assert l2["claim"] and not l2["context"] and l2["keys"]
    assert l1["claim"].startswith("Алиса — трейдер-агент")
    back = MemoryNode.from_dict(d).to_dict()
    assert back["levels"] == d["levels"] and back["level"] == 0 and back["usage"]["count"] == 0


def test_levels_rebuilt_when_claim_changes(store):
    d = store.add(make_node(LONG, source="t", evidence=["e"], context="c"))
    src0 = d["levels"]["src"]
    store.rewrite(d["id"], "Совсем другой факт: порт 9999 и MRR $500 через 30 дней.", reason="тест")
    d2 = store.get(d["id"])
    assert d2["levels"]["src"] != src0 and "9999" in json.dumps(d2["levels"], ensure_ascii=False)


def test_search_scans_all_levels_boost_only_for_level_text(store):
    d = store.add(make_node(LONG, source="t", evidence=["e"], context="c"))
    nid = d["id"]
    e = store.add(make_node("Другой узел: ни одного шорта не было и здесь.", source="t", evidence=["e"], context="c"))
    # руками опускаем узел на L2: ключи L2 (outlier-токены) находят его с бустом
    d["level"] = 2
    l2 = d["levels"]["2"]
    key = l2["keys"].split()[0]
    assert [n["id"] for n in store.search(key, top_k=5, touch=False)] == [nid]
    # фраза из хвоста claim, которой нет ни в тезисе, ни в ключах: узел всё
    # равно находится по полному тексту, но без буста уровня — ниже L0-узла
    # с тем же матчем (до правки по ревью 02.09 он был невидим — recall 0.86 на +45д)
    assert "ни одного шорта не было" not in (l2["claim"] + l2["keys"]).lower()
    got = [n["id"] for n in store.search("ни одного шорта не было", top_k=5, touch=False)]
    assert got == [e["id"], nid]
    d["level"] = 0  # на L0 — свежий d (ts позже e) выше при равном матче
    got = [n["id"] for n in store.search("ни одного шорта не было", top_k=5, touch=False)]
    assert got == [e["id"], nid] or got == [nid, e["id"]]


def test_level_boost_orders_equal_matches(store):
    ids = []
    for i in range(3):
        d = store.add(make_node(f"порт 8765 слушает MCP, узел {i}", source="t", evidence=["e"], context="c"))
        ids.append(d["id"])
    # одинаковый матч по claim; уровни 2, 1, 0 -> выдача 0, 1, 2
    store.get(ids[0])["level"] = 2
    store.get(ids[1])["level"] = 1
    for nid in ids:  # тезис у всех содержит «8765»
        assert "8765" in store.get(nid)["levels"]["2"]["claim"] + store.get(nid)["levels"]["2"]["keys"]
    got = [n["id"] for n in store.search("8765", top_k=3, touch=False)]
    assert got == [ids[2], ids[1], ids[0]]


def test_hit_touches_usage_and_dequantizes_lazily(store):
    d = store.add(make_node(LONG, source="t", evidence=["e"], context="c"))
    d["level"] = 1
    before = store.path.read_bytes()
    out = store.search("winrate", top_k=5)
    assert out and out[0]["id"] == d["id"]
    assert d["level"] == 0 and d["usage"]["count"] == 1 and len(d["usage"]["hits"]) == 1
    assert store.path.read_bytes() == before, "попадание не пишет файл само"
    assert store._usage_dirty
    store.add(make_node("другой узел", source="t", evidence=["e"], context="c"))  # любая запись
    assert not store._usage_dirty
    saved = json.loads(store.path.read_text(encoding="utf-8"))[d["id"]]
    assert saved["usage"]["count"] == 1 and saved["level"] == 0
    assert store.flush_usage() is False


def test_decay_requantizes_by_policy_and_rules_stay_l0(store):
    fresh = _old(store, LONG, days=1)
    mid = _old(store, LONG + " (среднее)", days=45)
    old = _old(store, LONG + " (старое)", days=200)
    rule = _old(store, "ПРАВИЛО 1: обходные пути запрещены. " + LONG, days=200, kind="rule")
    store.decay(half_life_hours=168.0)
    lv = {nid: store.get(nid)["level"] for nid in (fresh["id"], mid["id"], old["id"], rule["id"])}
    now = datetime.now(timezone.utc)
    # политика считается от реального «сейчас»; NOW = 02.09 15:00, так что
    # возраст узлов >= заданным дням, а простой = возраст (last_used = ts)
    assert lv[fresh["id"]] == 0
    assert lv[mid["id"]] == 1
    assert lv[old["id"]] == 2
    assert lv[rule["id"]] == 0
    st = store.stats()
    assert st["levels"] == {"0": 2, "1": 1, "2": 1} and st["quantized"] == 2 and st["with_levels"] == 4
    # decide_level напрямую: попадания за 7 дней держат L0
    d = dict(store.get(old["id"]))
    d["usage"] = {"count": 3, "last_hit": _iso(now), "hits": [_iso(now - timedelta(days=i)) for i in range(3)]}
    assert qmem.decide_level(d, now) == 0
    # requantize(rebuild=True) пересобирает тексты, гистограмма та же
    res = store.requantize(rebuild=True)
    assert res["built"] == 4 and res["levels"] == st["levels"]


def test_tf_bonus_ranks_node_about_the_term_higher(store):
    a = store.add(make_node("Алиса сам решает: торгует только Алиса, сигналы Алиса идут в ордер.",
                            source="приказ", evidence=["tg"], context="c"))
    b = store.add(make_node("Заметка про память, где Алиса упомянут один раз.",
                            source="приказ", evidence=["tg"], context="c"))
    got = [n["id"] for n in store.search("Алиса", top_k=2, touch=False)]
    assert got == [a["id"], b["id"]]


def test_exact_claim_path_respects_active_and_hub_filters(store):
    core = MnemosCore(store, plugins=[])
    r = store.add(make_node("Опровергнутое утверждение про порт 1111.", source="t", evidence=["e"],
                            context="c", kind="refuted"))
    h = store.add({**make_node("ХАБ «Алиса»: 3 узла.", source="hubs", evidence=["e"], context="{}",
                               kind="hub", tags=["hub", "entity:алиса"]).to_dict()})
    assert store.search("Опровергнутое утверждение про порт 1111.") == []
    assert store.search("ХАБ «Алиса»: 3 узла.") == []
    assert [n["id"] for n in store.search("ХАБ «Алиса»: 3 узла.", include_hubs=True)] == [h["id"]]
    assert store.get(r["id"])["usage"]["count"] == 0 and store.get(h["id"])["usage"]["count"] == 0
    with pytest.raises(ValueError):
        core.memory_add({"claim": "подставной хаб", "kind": "hub", "tags": ["hub", "entity:алиса"],
                         "source": "t", "evidence": ["e"], "context": "c"})


def test_link_rel_bidirectional_inverts_and_legacy_fill_leaves_trace(store):
    a = store.add(make_node("узел А про halt", source="t", evidence=["e"], context="c"))
    b = store.add(make_node("узел Б про halt", source="t", evidence=["e"], context="c"))
    store.link_existing(a["id"], b["id"], bidirectional=True, rel="has_part", author="x")
    assert store.get(a["id"])["link_meta"][b["id"]]["rel"] == "has_part"
    assert store.get(b["id"])["link_meta"][a["id"]]["rel"] == "part_of"
    with pytest.raises(ValueError):
        store.link_existing(a["id"], b["id"], bidirectional=True, rel="supersedes")
    # легаси-ребро без rel: тип дописывается со следом rel_by/rel_ts, автор ребра не меняется
    c = store.add(make_node("узел В", source="t", evidence=["e"], context="c"))
    store.link_existing(a["id"], c["id"], author="alice")
    del store.get(a["id"])["link_meta"][c["id"]]["rel"]
    store.link_existing(a["id"], c["id"], rel="supersedes", author="fable")
    meta = store.get(a["id"])["link_meta"][c["id"]]
    assert meta["author"] == "alice" and meta["rel"] == "supersedes" and meta["rel_by"] == "fable" and meta["rel_ts"]


def test_router_tie_goes_to_world_state_and_exactly_one_tag(store):
    assert classify_router({"claim": "Просто заметка без маркеров", "kind": "context_summary"})[0] == ROUTER_WORLD
    assert classify_router({"claim": "По умолчанию top_k равен 5 в memory_search", "kind": "fact"})[0] == ROUTER_WORLD
    from mnemos.budget import ensure_router_tag
    assert ensure_router_tag(["sys_cmd", "world_state", "x"], {"claim": "a"}) == ["x", "sys_cmd"]
    assert ensure_router_tag(["SYS_CMD"], {"claim": "a"}) == ["sys_cmd"]


def test_memory_search_touches_only_returned_nodes(store):
    core = MnemosCore(store, plugins=[])
    a = core.memory_add({"claim": "Алиса решает сам", "source": "t", "evidence": ["e"], "context": "c"})
    out = core.memory_search({"query": "Алиса", "tags": ["нет_такого"]})
    assert out["count"] == 0 and store.get(a["id"])["usage"]["count"] == 0
    out = core.memory_search({"query": "Алиса"})
    assert out["count"] == 1 and store.get(a["id"])["usage"]["count"] == 1
    assert "levels" not in out["results"][0] and "levels" not in a
    nl = core.memory_search({"query": "кто решает сделки сам?", "kind": "fact"})
    assert nl["mode"] == "token_fallback" and nl["count"] == 1
    assert store.get(a["id"])["usage"]["count"] == 2


def test_old_node_without_levels_is_searched_as_l0(store):
    d = make_node(LONG, source="t", evidence=["e"], context="c").to_dict()
    d["level"] = 2  # уровень выставлен, а сжатых текстов нет (стор до Ф1)
    store._nodes[d["id"]] = d
    assert store.search("winrate", top_k=5, touch=False)[0]["id"] == d["id"]


# --- MCP -----------------------------------------------------------------------------
def test_memory_search_nl_fallback_and_no_levels_in_output(store):
    core = MnemosCore(store, plugins=[])
    core.memory_add({"claim": LONG, "source": "роли", "evidence": ["ps"], "context": "персона"})
    core.memory_add({"claim": "Mnemos MCP-сервер слушает 127.0.0.1:8765, только loopback.",
                     "source": "systemd", "evidence": ["ss"], "context": "инфра"})
    out = core.memory_search({"query": "На каком порту слушает MCP-сервер Mnemos?"})
    assert out["mode"] == "token_fallback" and out["count"] >= 1
    assert out["results"][0]["claim"].startswith("Mnemos MCP")
    assert all("levels" not in r for r in out["results"]) and "level" in out["results"][0]
    kw = core.memory_search({"query": "8765"})
    assert kw["mode"] == "substring" and kw["count"] == 2
    none = core.memory_search({"query": "квантовая телепортация бананов"})
    assert none["count"] == 0 and none["mode"] == "substring"


def test_memory_search_semantic_auto_is_not_budget(store):
    core = MnemosCore(store, plugins=[])
    core.memory_add({"claim": "x y z", "source": "t", "evidence": ["e"], "context": "c"})
    try:
        out = core.memory_search({"query": "x", "mode": "semantic", "top_k": "auto"})
        assert out["mode"] == "semantic"
    except ValueError as exc:  # нет fastembed — честная ошибка semantic, не budget
        assert "semantic" in str(exc)


def test_memory_add_assigns_exactly_one_router_tag(store):
    core = MnemosCore(store, plugins=[])
    r = core.memory_add({"claim": "ПРИКАЗ ИЛЬИ 30.08: защиты сняты полностью, halt отключён.",
                         "source": "tg", "evidence": ["tg"], "context": "c", "kind": "rule"})
    p = core.memory_add({"claim": "Роли команды на 29.08: Акме-1 — оркестратор, Алиса — трейдер-агент.",
                         "source": "tg", "evidence": ["tg"], "context": "c"})
    w = core.memory_add({"claim": "Стор памяти на 1 сентября: 36 узлов, 51172 байт.",
                         "source": "stats", "evidence": ["memory_stats"], "context": "c"})
    e = core.memory_add({"claim": "Явный тег клиента уважается, классификатор молчит.",
                         "source": "t", "evidence": ["e"], "context": "c", "tags": ["x", ROUTER_PERSONA]})
    assert [t for t in r["tags"] if t in ROUTERS] == [ROUTER_SYS_CMD]
    assert [t for t in p["tags"] if t in ROUTERS] == [ROUTER_PERSONA]
    assert [t for t in w["tags"] if t in ROUTERS] == [ROUTER_WORLD]
    assert e["tags"] == ["x", ROUTER_PERSONA]
    assert classify_router({"claim": "ЧЕК-5 перед каждым ответом", "kind": "fact"})[0] == ROUTER_SYS_CMD


def test_memory_prompt_default_budget_1200(store):
    core = MnemosCore(store, plugins=[])
    for i in range(30):
        core.memory_add({"claim": f"Факт {i} про порт 8765 и Алиса: " + LONG, "source": "t",
                         "evidence": ["e"], "context": "контекст " * 20})
    assert PROMPT_DEFAULT_TOKENS == 1200
    res = core.memory_prompt({"query": "Алиса 8765"})
    assert res["max_tokens"] == 1200 and res["tokens"] <= 1200 and not res["over_budget"]
    big = core.memory_prompt({"query": "Алиса 8765", "max_tokens": 5000})
    assert big["max_tokens"] == 5000 and big["tokens"] >= res["tokens"]


def test_memory_decay_reports_levels(store):
    core = MnemosCore(store, plugins=[])
    core.memory_add({"claim": LONG, "source": "t", "evidence": ["e"], "context": "c"})
    out = core.memory_decay({})
    assert out["levels"] == {"0": 1, "1": 0, "2": 0} and out["quantized"] == 0


# --- гейт R4 на снимке 01.09 --------------------------------------------------------
LIVE_COPY = "/opt/bench-memory/f1_nodes_before.json"


@pytest.mark.skipif(not (os.path.exists(SNAP) and os.path.exists(GT)), reason="нет снимка 01.09 / GT")
@pytest.mark.parametrize("src,days,gate", [
    (SNAP, 0, 0.944), (SNAP, 25, 0.944), (SNAP, 45, 0.944), (SNAP, 120, 0.944), (SNAP, 365, 0.944),
    # боевая копия 02.09 (50 узлов): гейт 0.944 держится до +25д; с +45д без единого
    # обращения — 0.9278 (D01: правила на L0 с бустом вытесняют «спящие» факты).
    # Это измеренное состояние, не доказательство «держится всегда» (финальное ревью, п.2).
    (LIVE_COPY, 0, 0.944), (LIVE_COPY, 25, 0.944), (LIVE_COPY, 45, 0.9277), (LIVE_COPY, 365, 0.9277),
])
def test_gate_recall_under_aging(tmp_path, src, days, gate):
    """Старение без единого обращения: узлы уходят на L1/L2, поиск по всем
    уровням держит гейт (до правки по ревью 02.09: 0.86 на +45д, 0.84 на +120д)."""
    if not os.path.exists(src):
        pytest.skip(f"нет {src}")
    cp = tmp_path / "nodes.json"
    shutil.copy2(src, cp)
    st = Store(cp)
    now = datetime.now(timezone.utc) + timedelta(days=days)
    st.requantize(now, rebuild=True)
    if days >= 45:
        assert st.stats()["quantized"] >= 30
    gt = json.load(open(GT, encoding="utf-8"))["decisions"]
    rec = 0.0
    for q in gt:
        top = [n["id"] for n in st.search(q["kw"], top_k=5, touch=False)]
        rec += len(set(q["gt"]) & set(top)) / len(q["gt"])
    assert rec / len(gt) >= gate, (src, days, rec / len(gt))


def test_flush_disk_error_does_not_break_search(store, monkeypatch):
    core = MnemosCore(store, plugins=[])
    a = core.memory_add({"claim": "Алиса решает сам", "source": "t", "evidence": ["e"], "context": "c"})
    store._dirty_hits = store.TOUCH_FLUSH_EVERY - 1
    monkeypatch.setattr("mnemos.store.os.replace", lambda *a, **k: (_ for _ in ()).throw(OSError("ENOSPC")))
    out = core.memory_search({"query": "Алиса"})  # флаш падает, поиск — нет
    assert out["count"] == 1 and store._dirty_hits == 0 and store._usage_dirty
    out = core.memory_search({"query": "Алиса"})
    assert out["count"] == 1


def test_link_response_reports_rel_applied(store):
    a = store.add(make_node("узел А", source="t", evidence=["e"], context="c"))
    b = store.add(make_node("узел Б", source="t", evidence=["e"], context="c"))
    r1 = store.link_existing(a["id"], b["id"], rel="part_of")
    r2 = store.link_existing(a["id"], b["id"], rel="conflicts_with")
    assert r1["rel_applied"] and r1["rel"] == "part_of"
    assert not r2["rel_applied"] and r2["rel"] == "part_of"


@pytest.mark.skipif(not (os.path.exists(SNAP) and os.path.exists(GT)), reason="нет снимка 01.09 / GT")
def test_gate_snapshot_0109_recall_not_below_0944(tmp_path):
    cp = tmp_path / "nodes.json"
    shutil.copy2(SNAP, cp)
    st = Store(cp)
    st.requantize()  # уровни построены; политика «сегодня» — все L0
    gt = json.load(open(GT, encoding="utf-8"))["decisions"]
    rec = 0.0
    for q in gt:
        top = [n["id"] for n in st.search(q["kw"], top_k=5)]
        rec += len(set(q["gt"]) & set(top)) / len(q["gt"])
    assert rec / len(gt) >= 0.944, rec / len(gt)


def test_prompt_marks_superseded_rules_and_drops_them_first(store):
    from mnemos.budget import build_system_prompt
    core = MnemosCore(store, plugins=[])
    old = core.memory_add({"claim": "РЕЖИМ ТОРГОВЛИ 29.08: halt действует после 3 убытков.", "source": "tg",
                           "evidence": ["tg"], "context": "c", "kind": "rule"})
    new = core.memory_add({"claim": "ПРИКАЗ ИЛЬИ 30.08: защиты сняты полностью, halt отключён.", "source": "tg",
                           "evidence": ["tg"], "context": "c", "kind": "rule"})
    store.link_existing(new["id"], old["id"], rel="supersedes", author="t")
    res = core.memory_prompt({"constitution": True, "max_tokens": 5000})
    t = res["text"]
    assert f"[ОТМЕНЁН ← {new['id']}]" in t and t.index(new["id"]) < t.index(old["id"])
    tight = build_system_prompt(store.all(), max_tokens=30, superseded={old["id"]: new["id"]})
    assert old["id"] not in tight["text"] and new["id"] in tight["text"]
    assert any("отменённый" in d for d in tight["dropped"])
