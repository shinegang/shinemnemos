# -*- coding: utf-8 -*-
"""Grounded по умолчанию + чистый (бланковый) граф для клиентов.

Приказ Ильи 03.09: «мы продаём инструменты, а не данные». Три проверки из
приказа:
  (а) свежая инстанция с blank — memory_search возвращает 0 узлов;
  (б) grounded по умолчанию — ответ без пред-прохода = ungrounded;
  (в) боевой стор не изменён (sha до/после полного прогона).

Плюс защита от того, чем такие фичи обычно и ломаются: blank, затирающий
живой граф клиента при рестарте; наши узлы, уехавшие в шаблон дистрибутива;
политика, о которой клиент узнаёт только из README.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from mnemos import grounding
from mnemos import store as store_mod
from mnemos.model import make_node
from mnemos.server import (
    GROUND_ENV,
    MCPHttpServer,
    MnemosCore,
    _env_flag,
)
from mnemos.store import (
    Store,
    blank_target,
    blank_template_path,
    create_blank_store,
    resolve_store_path,
)

PROD_STORE = os.environ.get("MNEMOS_GATE_STORE", "/opt/mnemos/mnemos/data/nodes.json")

# Маленький корпус: нужен только факт «ответ целиком из графа», качество
# сверки утверждений проверяет test_grounding.py.
CORPUS = [
    {
        "claim": "Боевой бот AcmeTrader работает на арендованной машине TW "
                 "с двумя картами 3090, инстанс vast 12345678",
        "source": "инвентаризация инфраструктуры 02.09",
        "evidence": ["vast.ai list instances 02.09"],
        "context": "Где физически исполняется боевой бот",
    },
]


@pytest.fixture(autouse=True)
def clean_policy_env(monkeypatch):
    """Политика по умолчанию не должна зависеть от окружения хоста."""
    monkeypatch.delenv(GROUND_ENV, raising=False)
    monkeypatch.delenv(store_mod.STORE_ENV, raising=False)
    monkeypatch.delenv(store_mod.STORE_PATH_ENV, raising=False)


@pytest.fixture
def blank_store(tmp_path):
    """Стор клиентской инстанции, созданный режимом blank."""
    target = tmp_path / "client" / "nodes.json"
    path, created = resolve_store_path(f"blank:{target}", env={})
    assert created, "режим blank обязан создать файл графа"
    return path


# --- (а) свежая инстанция с blank: ноль узлов --------------------------------

def test_a_blank_instance_is_empty(blank_store):
    """Клиент получает инструменты и пустой граф — ни одного нашего узла."""
    assert json.loads(blank_store.read_text(encoding="utf-8")) == {}

    core = MnemosCore(Store(blank_store), plugins=[])
    assert len(core.store) == 0

    for query in ("Алиса", "AcmeTrader", "бот", "trader", "Иван"):
        res = core.memory_search({"query": query})
        assert res["count"] == 0, f"чистый граф ответил на {query!r}: {res}"
        assert res["results"] == []

    stats = core.memory_stats({})
    assert stats["nodes"] == 0 and stats["edges"] == 0


def test_a_blank_via_env(tmp_path, monkeypatch):
    """env MNEMOS_STORE=blank + MNEMOS_STORE_PATH — путь запуска у клиента."""
    target = tmp_path / "env" / "nodes.json"
    monkeypatch.setenv(store_mod.STORE_ENV, "blank")
    monkeypatch.setenv(store_mod.STORE_PATH_ENV, str(target))
    path, created = resolve_store_path(None)
    assert (path, created) == (target, True)
    assert len(Store(path)) == 0


def test_a_blank_default_target_is_local_nodes_json(tmp_path, monkeypatch):
    """Без MNEMOS_STORE_PATH blank кладёт граф в ./nodes.json рабочего каталога."""
    monkeypatch.chdir(tmp_path)
    path, created = resolve_store_path("blank", env={})
    assert created is True
    assert str(path) == store_mod.DEFAULT_STORE_NAME
    assert (tmp_path / store_mod.DEFAULT_STORE_NAME).exists()
    assert len(Store(path)) == 0


def test_blank_never_overwrites_existing_graph(tmp_path):
    """blank не стирает чужой граф — он отказывается стартовать (Д1, баг-хант)."""
    target = tmp_path / "nodes.json"
    store = Store(target)
    node = store.add(make_node(claim="память клиента, накопленная за месяц",
                               source="клиент"))
    before = target.read_bytes()

    with pytest.raises(ValueError, match="уже 1 узлов"):
        resolve_store_path(f"blank:{target}", env={})
    # данные на месте: отказ стартовать — не то же самое, что затирание
    assert target.read_bytes() == before
    assert Store(target).get(node["id"]) is not None


def test_blank_on_empty_existing_file_is_fine(tmp_path):
    """Пустой файл на пути — не конфликт: он и так пуст, рестарт проходит."""
    target = tmp_path / "nodes.json"
    target.write_text("{}\n", encoding="utf-8")
    path, created = resolve_store_path(f"blank:{target}", env={})
    assert (path, created) == (target, False)
    assert len(Store(path)) == 0


def test_blank_refuses_a_directory(tmp_path):
    """blank:<каталог> — ошибка старта, а не сервер, который не умеет писать (Д2)."""
    target = tmp_path / "graphdir"
    target.mkdir()
    with pytest.raises(ValueError, match="каталог"):
        resolve_store_path(f"blank:{target}", env={})


def test_blank_refuses_a_dangling_symlink(tmp_path):
    """Битая ссылка не подменяется молча обычным файлом (Д4)."""
    target = tmp_path / "link.json"
    target.symlink_to(tmp_path / "nowhere.json")
    with pytest.raises(ValueError, match="символьная ссылка"):
        resolve_store_path(f"blank:{target}", env={})
    assert target.is_symlink() and not target.exists()


def test_blank_refuses_unreadable_graph(tmp_path):
    """На пути лежит не-граф — говорим об этом, а не стартуем пустыми."""
    target = tmp_path / "nodes.json"
    target.write_text("это вообще не json", encoding="utf-8")
    with pytest.raises(ValueError, match="не читается как граф"):
        resolve_store_path(f"blank:{target}", env={})


def test_blank_target_detects_mode(tmp_path):
    """Режим blank определяется разбором спецификации, а не префиксом строки (Д3)."""
    assert blank_target("blank", env={}) == Path("nodes.json")
    assert blank_target(f"blank:{tmp_path}/x.json", env={}) == tmp_path / "x.json"
    assert blank_target("BLANK", env={}) == Path("nodes.json")
    # обычные пути, которые лишь НАЧИНАЮТСЯ на "blank"
    for spec in ("blankgraph.json", "blank_2026.json", "blanks/nodes.json"):
        assert blank_target(spec, env={}) is None, spec
    assert blank_target("", env={}) is None
    assert blank_target(None, env={"MNEMOS_STORE": "blank"}) == Path("nodes.json")


def test_ordinary_store_path_is_untouched_by_blank_logic(tmp_path):
    """Обычный путь остаётся обычным путём: файл не создаётся заранее."""
    target = tmp_path / "sub" / "my.json"
    path, created = resolve_store_path(str(target), env={})
    assert (path, created) == (target, False)
    assert not target.exists()


def test_windows_path_is_not_mistaken_for_blank(tmp_path):
    """'C:\\mnemos\\nodes.json' содержит ':' — но это путь, а не режим blank."""
    path, created = resolve_store_path(r"C:\mnemos\nodes.json", env={})
    assert created is False
    assert str(path).endswith("nodes.json")


# --- дистрибутив: инструменты без наших данных -------------------------------

def test_blank_template_ships_empty(tmp_path):
    """Шаблон в пакете — пустой JSON-объект, иначе это утечка наших узлов."""
    template = blank_template_path()
    assert template.exists(), f"нет шаблона пустого графа: {template}"
    assert json.loads(template.read_text(encoding="utf-8")) == {}


def test_package_ships_no_node_store():
    """В пакете нет ни одного стора с узлами — только шаблон nodes.blank.json."""
    pkg = blank_template_path().parent.parent
    for path in pkg.rglob("*.json"):
        if path.name == store_mod.BLANK_TEMPLATE_NAME:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        nodes = [v for v in data.values()
                 if isinstance(v, dict) and "claim" in v and "id" in v]
        assert not nodes, f"в пакет уехали узлы памяти: {path} ({len(nodes)})"


def test_non_empty_template_is_a_startup_error(tmp_path, monkeypatch):
    """Если наш стор подменит шаблон — старт падает, а не раздаёт данные."""
    leaked = tmp_path / "nodes.blank.json"
    leaked.write_text(json.dumps({"mn_1": {"id": "mn_1", "claim": "наш узел"}}),
                      encoding="utf-8")
    monkeypatch.setattr(store_mod, "blank_template_path", lambda: leaked)
    with pytest.raises(ValueError, match="не пуст"):
        create_blank_store(tmp_path / "client.json")
    assert not (tmp_path / "client.json").exists()


# --- (б) grounded по умолчанию ------------------------------------------------

@pytest.fixture
def core(blank_store):
    """Ядро на чистом графе + один узел: политика по умолчанию, без env."""
    c = MnemosCore(Store(blank_store), plugins=[], ground_by_default=None)
    for spec in CORPUS:
        c.memory_add(dict(spec))
    return c


def test_b_default_is_grounded_on(core, monkeypatch):
    monkeypatch.delenv(GROUND_ENV, raising=False)
    assert core.ground_by_default is True


def test_b_answer_without_pre_pass_is_ungrounded(core):
    """Приказ: нет пред-прохода -> ответ не принимается (ungrounded)."""
    answer = CORPUS[0]["claim"] + "."
    out = core.memory_ground({"answer_text": answer, "session_id": "s-no-pre"})

    assert out["verdict"] == "ungrounded"
    assert out["passed_through_graph"] == "нет"
    assert out["ground_by_default"] is True
    assert out["require_pre_pass"] is True
    assert any("no_pre_pass" in n for n in out["notes"])
    # сверка утверждений при этом не теряется: текст-то из графа
    assert out["claims_verdict"] == "grounded"


def test_b_pre_pass_makes_the_same_answer_grounded(core):
    core.memory_ground_prepare({"query": "Где работает боевой бот?",
                                "session_id": "s-pre"})
    out = core.memory_ground({"answer_text": CORPUS[0]["claim"] + ".",
                              "session_id": "s-pre"})
    assert out["verdict"] == "grounded"
    assert out["passed_through_graph"] == "да"


def test_b_policy_off_accepts_answer_without_pre_pass(blank_store):
    """Выключено осознанно — сверка остаётся, требование пред-прохода уходит."""
    core = MnemosCore(Store(blank_store), plugins=[], ground_by_default=False)
    core.memory_add(dict(CORPUS[0]))
    out = core.memory_ground({"answer_text": CORPUS[0]["claim"] + ".",
                              "session_id": "s-off"})
    assert core.ground_by_default is False
    assert out["ground_by_default"] is False
    assert out["require_pre_pass"] is False
    assert out["verdict"] == "grounded"


def test_b_explicit_argument_beats_policy(core):
    """require_pre_pass=false у клиента без сессий работает и при политике on."""
    out = core.memory_ground({"answer_text": CORPUS[0]["claim"] + ".",
                              "require_pre_pass": False})
    assert out["require_pre_pass"] is False
    assert out["verdict"] == "grounded"


def test_b_null_require_pre_pass_does_not_weaken_policy(core):
    """require_pre_pass: null — это «не задано», а не «не требовать»."""
    out = core.memory_ground({"answer_text": CORPUS[0]["claim"] + ".",
                              "require_pre_pass": None})
    assert out["require_pre_pass"] is True
    assert out["verdict"] == "ungrounded"


def test_b_prepare_reports_policy(core):
    prep = core.memory_ground_prepare({"query": "Где работает бот?",
                                       "session_id": "s-rep"})
    assert prep["ground_by_default"] is True
    assert "memory_ground_prepare" in prep["policy"]


# --- политика: env, рукопожатие, /health -------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True), ("да", True),
    (" 1 ", True),
    ("0", False), ("false", False), ("no", False), ("off", False),
    ("нет", False), ("OFF", False),
])
def test_env_flag_values(monkeypatch, raw, expected):
    monkeypatch.setenv(GROUND_ENV, raw)
    assert _env_flag(GROUND_ENV, True) is expected


@pytest.mark.parametrize("raw", ["", "   "])
def test_empty_env_does_not_disable_policy(monkeypatch, raw):
    """`Environment=MNEMOS_GROUND_BY_DEFAULT=` — это «не задано», не «выключить»."""
    monkeypatch.setenv(GROUND_ENV, raw)
    assert _env_flag(GROUND_ENV, True) is True
    assert _env_flag(GROUND_ENV, False) is False


def test_env_flag_garbage_fails_loudly(monkeypatch):
    """Опечатка не должна молча оставлять политику не в том состоянии."""
    monkeypatch.setenv(GROUND_ENV, "flase")
    with pytest.raises(ValueError, match=GROUND_ENV):
        _env_flag(GROUND_ENV, True)


def test_env_disables_policy(blank_store, monkeypatch):
    monkeypatch.setenv(GROUND_ENV, "0")
    assert MnemosCore(Store(blank_store), plugins=[]).ground_by_default is False


def test_argument_beats_env(blank_store, monkeypatch):
    monkeypatch.setenv(GROUND_ENV, "0")
    core = MnemosCore(Store(blank_store), plugins=[], ground_by_default=True)
    assert core.ground_by_default is True


def test_initialize_advertises_policy(blank_store):
    """Агент узнаёт про обязательный проход на рукопожатии, а не из README.

    `instructions` — в КОРНЕ InitializeResult (D1): в serverInfo по спеке MCP
    живёт только Implementation{name, version}, и SDK молча вырезает оттуда
    лишние ключи — контракт не доезжал бы ни до одного клиента.
    """
    core = MnemosCore(Store(blank_store), plugins=[])
    result = core.handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"})["result"]
    assert set(result["serverInfo"]) == {"name", "version"}
    assert "memory_ground_prepare" in result["instructions"]
    assert result["_meta"]["ground_by_default"] is True

    off = MnemosCore(Store(blank_store), plugins=[], ground_by_default=False)
    result_off = off.handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"})["result"]
    assert "instructions" not in result_off
    assert result_off["_meta"]["ground_by_default"] is False


def test_http_server_passes_policy_through(blank_store):
    httpd = MCPHttpServer(("127.0.0.1", 0), Store(blank_store), plugins=[],
                          ground_by_default=False)
    try:
        assert httpd.mnemos.ground_by_default is False
    finally:
        httpd.server_close()


def test_grounded_tools_are_listed(blank_store):
    core = MnemosCore(Store(blank_store), plugins=[])
    names = {t["name"] for t in core.tools}
    assert {"memory_ground_prepare", "memory_ground", "memory_answer",
            "memory_ground_log"} <= names


# --- закрытые дыры политики (баг-хант 03.09) ---------------------------------

def test_empty_pre_pass_does_not_count(blank_store):
    """D2: пред-проход, не нашедший в графе НИЧЕГО, — это не проход."""
    core = MnemosCore(Store(blank_store), plugins=[])
    core.memory_add(dict(CORPUS[0]))

    prep = core.memory_ground_prepare({"query": "квантовая криптография в Перу",
                                       "session_id": "s-empty"})
    assert prep["node_ids"] == [], "корпус не должен отвечать на этот вопрос"

    out = core.memory_ground({"answer_text": CORPUS[0]["claim"] + ".",
                              "session_id": "s-empty"})
    assert out["verdict"] == "ungrounded"
    assert any("no_pre_pass" in n for n in out["notes"])
    # событие в журнале при этом есть — пустой проход виден аудиту
    log = core.memory_ground_log({"limit": 50, "session_id": "s-empty",
                                  "event": "prepare"})
    assert log["count"] == 1 and log["records"][0]["nodes"] == 0


def test_empty_pre_pass_from_log_does_not_count(blank_store):
    """То же правило на пути «после рестарта»: пред-проход читается из журнала."""
    core = MnemosCore(Store(blank_store), plugins=[])
    core.memory_add(dict(CORPUS[0]))
    core.memory_ground_prepare({"query": "Где работает боевой бот?",
                                "session_id": "s-full"})
    core.memory_ground_prepare({"query": "квантовая криптография в Перу",
                                "session_id": "s-log"})
    # рестарт: трекер в памяти пуст, источник правды — журнал
    core.sessions = grounding.SessionTracker()
    assert core.ground_log.find_pre_pass("s-log") is None
    found = core.ground_log.find_pre_pass("s-full")
    assert found is not None and found["source"] == "log"


@pytest.mark.parametrize("bad", [0, 1, "", "false", "true", [], {}, 0.0])
def test_require_pre_pass_rejects_non_boolean(core, bad):
    """D4: схема объявляет boolean — «почти булево» не должно снимать политику."""
    with pytest.raises(ValueError, match="require_pre_pass"):
        core.memory_ground({"answer_text": CORPUS[0]["claim"] + ".",
                            "require_pre_pass": bad})


def test_reinforce_rejects_non_boolean(core):
    with pytest.raises(ValueError, match="reinforce"):
        core.memory_ground({"answer_text": CORPUS[0]["claim"] + ".",
                            "reinforce": "yes"})


def test_ungrounded_answer_does_not_reinforce(core):
    """D5: отвергнутый ответ не качает веса графа и не переписывает стор."""
    nid = core.store.search(CORPUS[0]["claim"])[0]["id"]
    before = core.store.get(nid)["weight"]

    out = core.memory_ground({"answer_text": CORPUS[0]["claim"] + ".",
                              "session_id": "s-nopre"})
    assert out["verdict"] == "ungrounded"
    assert out["reinforced"] == []
    assert "reinforce_skipped" in out
    assert core.store.get(nid)["weight"] == before


def test_grounded_answer_still_reinforces(core):
    core.memory_ground_prepare({"query": "Где работает боевой бот?",
                                "session_id": "s-yes"})
    out = core.memory_ground({"answer_text": CORPUS[0]["claim"] + ".",
                              "session_id": "s-yes"})
    assert out["verdict"] == "grounded"
    assert out["reinforced"], "подтверждённый ответ обязан подкрепить опору"


def test_ground_log_records_policy_bypass(core):
    """D4-аудит: обход политики через require_pre_pass=false виден в журнале."""
    core.memory_ground({"answer_text": CORPUS[0]["claim"] + ".",
                        "session_id": "s-bypass", "require_pre_pass": False})
    rec = core.memory_ground_log({"limit": 10, "session_id": "s-bypass",
                                  "event": "ground"})["records"][0]
    assert rec["require_pre_pass"] is False
    assert rec["ground_by_default"] is True


@pytest.mark.parametrize("key", ["session_id", "agent"])
def test_oversized_ids_are_rejected(core, key):
    """D6: мегабайтный session_id не должен раздувать журнал."""
    with pytest.raises(ValueError, match=key):
        core.memory_ground({"answer_text": "текст.", key: "x" * 5000})


def test_ground_log_clips_long_query(blank_store):
    """D6: длинный вопрос обрезается в записи, sha и session_id — нет."""
    core = MnemosCore(Store(blank_store), plugins=[])
    core.memory_add(dict(CORPUS[0]))
    core.memory_ground_prepare({"query": "ю" * 5000, "session_id": "s-long"})
    rec = core.memory_ground_log({"limit": 5, "session_id": "s-long"})["records"][0]
    assert len(rec["query"]) < 500
    assert rec["query"].endswith(grounding.GROUND_LOG_TRUNC_MARK)
    assert rec["session_id"] == "s-long"


def test_ground_log_rotates(tmp_path, monkeypatch):
    """D6: журнал не растёт бесконечно — переполненный уезжает в .1."""
    monkeypatch.setattr(grounding, "GROUND_LOG_MAX_BYTES", 2048)
    log = grounding.GroundLog(tmp_path / "ground_log.jsonl")
    for i in range(200):
        log.append("prepare", session_id=f"s{i}", query="ы" * 100, nodes=1)
    rotated = tmp_path / "ground_log.jsonl.1"
    assert rotated.exists(), "ротация не сработала"
    assert log.path.stat().st_size < 2048 * 2
    # append-only не нарушен: записи не переписаны, они в .1
    assert rotated.read_text(encoding="utf-8").count("\n") > 1


def test_lock_timeout_is_real(tmp_path):
    """D7: у locked_file был недостижимый timeout — flock без LOCK_NB не сдаётся."""
    from mnemos.bus import locked_file
    target = tmp_path / "ground_log.jsonl"
    with locked_file(target):
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            with locked_file(target, timeout=0.3):
                pass
        assert time.monotonic() - started < 5.0


def test_plugins_see_a_finished_core(blank_store, monkeypatch):
    """D8: плагину передаётся `self` — политика и журнал к этому моменту есть."""
    from mnemos.plugins import PluginManager

    seen = {}
    original = PluginManager.handlers

    def spy(self, core):
        seen.update({name: hasattr(core, name) for name in
                     ("ground_by_default", "ground_log", "sessions")})
        return original(self, core)

    monkeypatch.setattr(PluginManager, "handlers", spy)
    MnemosCore(Store(blank_store), plugins=[])
    assert seen == {"ground_by_default": True, "ground_log": True,
                    "sessions": True}, seen


# --- (в) боевой стор не изменён ----------------------------------------------

@pytest.mark.skipif(not os.path.exists(PROD_STORE),
                    reason=f"нет боевого стора: {PROD_STORE}")
def test_c_prod_store_untouched_by_blank_flow(tmp_path):
    """Полный прогон blank + grounded не трогает боевой nodes.json."""
    def sha():
        with open(PROD_STORE, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    before = sha()
    target = tmp_path / "client" / "nodes.json"
    path, created = resolve_store_path(f"blank:{target}", env={})
    assert created
    core = MnemosCore(Store(path), plugins=[])
    core.memory_add(dict(CORPUS[0]))
    core.memory_ground_prepare({"query": "Где работает бот?", "session_id": "s-c"})
    core.memory_ground({"answer_text": CORPUS[0]["claim"] + ".",
                        "session_id": "s-c"})
    core.memory_answer({"query": "Где работает бот?"})
    core.memory_ground_log({"limit": 10, "stats": True})

    assert sha() == before, "боевой стор изменился во время прогона blank/grounded"
    # журнал проходов лёг рядом с КЛИЕНТСКИМ стором, не с боевым
    assert core.ground_log.path.parent == path.parent
