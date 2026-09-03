# -*- coding: utf-8 -*-
"""Тесты поиска по code-слою: ранжирование, реестр репозиториев, плагин, watcher.

Плагин code во всех тестах включается ЯВНО (plugins=[..., "code"]) — по
умолчанию он выключен, и это проверяется отдельным тестом. Корни репозиториев
переопределяются через MNEMOS_CODE_ROOTS на временный каталог, фоновый поток
выключается через MNEMOS_CODE_WATCH=0: тесты не должны трогать боевые деревья.
"""

import os

import pytest

from mnemos import code_index as ci
from mnemos import plugins as pl
from mnemos.code_search import CodeIndex, CodeRegistry, file_class_weight
from mnemos.code_watcher import CodeWatcher, watch_enabled, watch_interval
from mnemos.server import MnemosCore
from mnemos.store import Store

pytestmark = pytest.mark.skipif(not ci.TS_OK, reason=f"нет py-tree-sitter: {ci.TS_ERR}")


RISK = '''# -*- coding: utf-8 -*-
"""Риск-модуль бота."""


def position_size_usd(equity, price):
    """Размер позиции в долларах на сделку."""
    return equity * price


def stop_price_from_entry(entry, pct):
    """Цена стоп-лосса от цены входа."""
    return entry * (1 - pct)


class RiskEngine:
    """Движок риска."""

    def apply(self, signal):
        return position_size_usd(1, 2)
'''

TESTS_FILE = '''# -*- coding: utf-8 -*-
"""Тесты риска — не должны вытеснять исходник из топа."""


def test_position_size_usd():
    assert position_size_usd(1, 1) == 1
'''

DOCS = "# Алиса\nПравило Акме: halt после серии убытков.\n"


@pytest.fixture
def repo_root(tmp_path):
    root = tmp_path / "acmetrader"
    (root / "acmetrader").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "acmetrader" / "risk.py").write_text(RISK, encoding="utf-8")
    (root / "tests" / "test_risk.py").write_text(TESTS_FILE, encoding="utf-8")
    (root / "АЛИСА.md").write_text(DOCS, encoding="utf-8")
    return root


@pytest.fixture
def graph_path(tmp_path, repo_root):
    repo = ci.Repo("acmetrader", str(repo_root),
                   str(tmp_path / "code" / "acmetrader.code_graph.json"))
    ci.refresh_repo(repo)
    return repo.graph_path


@pytest.fixture
def index(graph_path):
    return CodeIndex(graph_path, repo="acmetrader")


@pytest.fixture
def registry(tmp_path, repo_root, monkeypatch):
    monkeypatch.setenv(ci.ENV_ROOTS, f"acmetrader={repo_root}")
    monkeypatch.setenv(ci.ENV_DATA_DIR, str(tmp_path / "code"))
    monkeypatch.setenv("MNEMOS_CODE_WATCH", "0")
    reg = CodeRegistry()
    reg.refresh(force=True)
    return reg


# --- ранжирование ------------------------------------------------------------


def test_search_finds_symbol_by_name(index):
    rows = index.search("position_size_usd", top_k=5)
    assert rows and rows[0]["qname"] == "acmetrader.risk.position_size_usd"
    assert rows[0]["file"] == "acmetrader/risk.py"
    assert rows[0]["symbol_kind"] == "function"
    assert rows[0]["line"] > 0 and rows[0]["signature"].startswith("def position_size_usd(")
    assert rows[0]["repo"] == "acmetrader"


def test_search_case_insensitive(index):
    a = index.search("POSITION_SIZE_USD", top_k=3)
    b = index.search("position_size_usd", top_k=3)
    assert [r["qname"] for r in a] == [r["qname"] for r in b]


def test_search_natural_language_hits_body(index):
    rows = index.search("цена стоп-лосса от цены входа", top_k=5)
    assert any(r["qname"] == "acmetrader.risk.stop_price_from_entry" for r in rows)


def test_search_cyrillic_query_finds_markdown(index):
    rows = index.search("Алиса halt убытков", top_k=5)
    assert any(r["file"] == "АЛИСА.md" for r in rows)


def test_search_empty_query_returns_empty(index):
    assert index.search("", top_k=5) == []
    assert index.search("   ", top_k=5) == []
    assert index.search("!!!", top_k=5) == []


def test_search_top_k_respected(index):
    assert len(index.search("риск", top_k=1)) <= 1
    assert len(index.search("position", top_k=2)) <= 2


def test_search_granularity_file(index):
    rows = index.search("position_size_usd", top_k=3, granularity="file")
    assert rows and rows[0]["file"] == "acmetrader/risk.py"
    assert set(rows[0]) == {"file", "repo", "score"}


def test_v2_downranks_tests(index):
    rows = index.search("position_size_usd", top_k=5, granularity="file")
    order = [r["file"] for r in rows]
    assert order.index("acmetrader/risk.py") < order.index("tests/test_risk.py")


def test_file_class_weights():
    assert file_class_weight("acmetrader/risk.py") == 1.0
    assert file_class_weight("tests/test_risk.py") < 1.0
    assert file_class_weight("pkg/test_x.py") < 1.0
    assert file_class_weight("docs/guide.md") == 0.5


def test_search_symbol_kind_filter(index):
    rows = index.search("risk", top_k=10, symbol_kind="class")
    assert rows and all(r["symbol_kind"] == "class" for r in rows)


def test_rank_v1_available(index):
    rows = index.search("position_size_usd", top_k=5, rank="v1")
    assert rows and rows[0]["qname"] == "acmetrader.risk.position_size_usd"


# --- symbols / callers -------------------------------------------------------


def test_symbols_all_and_filtered(index):
    all_syms = index.symbols(limit=100)
    assert len(all_syms) >= 5
    funcs = index.symbols(symbol_kind="function", limit=100)
    assert {s["qname"] for s in funcs} >= {
        "acmetrader.risk.position_size_usd", "acmetrader.risk.stop_price_from_entry"}
    in_file = index.symbols(file="acmetrader/risk.py", limit=100)
    assert all(s["file"] == "acmetrader/risk.py" for s in in_file)
    assert len(index.symbols(limit=2)) == 2


def test_callers_inbound_outbound(index):
    inbound = index.callers("acmetrader.risk.position_size_usd", direction="inbound")
    assert any(e["from"] == "acmetrader.risk.RiskEngine.apply" for e in inbound)
    outbound = index.callers("acmetrader.risk.RiskEngine.apply", direction="outbound")
    assert any(e["to"] == "acmetrader.risk.position_size_usd" for e in outbound)
    assert index.callers("нет.такого.символа") == []


def test_stale_seconds_present(index):
    assert index.stale_seconds >= 0.0
    assert index.age_seconds >= 0.0


# --- реестр ------------------------------------------------------------------


def test_registry_search_and_status(registry):
    rows = registry.search("position_size_usd", top_k=5)
    assert rows and rows[0]["repo"] == "acmetrader"
    status = registry.status()
    assert status[0]["indexed"] is True and status[0]["nodes"] > 0
    assert status[0]["index_bytes"] > 0
    assert registry.stale_seconds() is not None


def test_registry_refresh_keeps_index_in_memory(registry):
    """Тик без изменений не перечитывает граф с диска (тот же объект в кэше)."""
    first = registry.indexes()[0]
    stats = registry.refresh()
    assert stats[0]["mode"] == "unchanged"
    assert registry.indexes()[0] is first
    assert first.stale_seconds < 5.0


def test_registry_refresh_reports_error_without_dropping_others(registry, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("диск отвалился")

    monkeypatch.setattr("mnemos.code_search.refresh_repo_graph", boom)
    stats = registry.refresh()
    assert "error" in stats[0]


def test_registry_unknown_repo_raises(registry):
    with pytest.raises(ValueError):
        registry.search("x", repo="нет-такого")


def test_registry_reloads_changed_graph(registry, repo_root):
    assert not registry.search("уникальное_имя_функции", top_k=5)
    with open(os.path.join(str(repo_root), "acmetrader", "risk.py"), "a",
              encoding="utf-8") as f:
        f.write("\n\ndef уникальное_имя_функции():\n    pass\n")
    os.utime(os.path.join(str(repo_root), "acmetrader", "risk.py"), (0, 0))
    stats = registry.refresh()
    assert stats[0]["mode"] == "incremental"
    rows = registry.search("уникальное_имя_функции", top_k=5)
    assert rows and rows[0]["qname"].endswith("уникальное_имя_функции")


def test_registry_without_graph_returns_nothing(tmp_path, repo_root, monkeypatch):
    monkeypatch.setenv(ci.ENV_ROOTS, f"acmetrader={repo_root}")
    monkeypatch.setenv(ci.ENV_DATA_DIR, str(tmp_path / "пусто"))
    reg = CodeRegistry()
    assert reg.search("position_size_usd") == []
    assert reg.status()[0]["indexed"] is False
    assert reg.stale_seconds() is None


def test_registry_multi_repo(tmp_path, repo_root, monkeypatch):
    other = tmp_path / "panel"
    other.mkdir()
    (other / "panel.py").write_text("def отдельный_символ_панели():\n    pass\n",
                                    encoding="utf-8")
    monkeypatch.setenv(ci.ENV_ROOTS, f"acmetrader={repo_root},panel={other}")
    monkeypatch.setenv(ci.ENV_DATA_DIR, str(tmp_path / "code"))
    reg = CodeRegistry()
    reg.refresh(force=True)
    assert set(reg.repo_names()) == {"acmetrader", "panel"}
    rows = reg.search("отдельный_символ_панели", top_k=5)
    assert rows and rows[0]["repo"] == "panel"
    only = reg.search("position_size_usd", top_k=5, repo="panel")
    assert not any(r["repo"] == "acmetrader" for r in only)


# --- плагин ------------------------------------------------------------------


def call(core, name, args=None):
    """Вызов MCP-инструмента через таблицу обработчиков ядра (как tools/call)."""
    return core._handlers[name](args or {})


@pytest.fixture
def code_core(tmp_path, repo_root, monkeypatch):
    monkeypatch.setenv(ci.ENV_ROOTS, f"acmetrader={repo_root}")
    monkeypatch.setenv(ci.ENV_DATA_DIR, str(tmp_path / "code"))
    monkeypatch.setenv("MNEMOS_CODE_WATCH", "0")
    store = Store(tmp_path / "nodes.json")
    core = MnemosCore(store, plugins=["code"])
    call(core, "code_refresh", {"force": True})
    return core


def test_plugin_disabled_by_default():
    assert "code" not in pl.DEFAULT_ENABLED
    assert pl.CodePlugin.default_enabled is False
    assert "code" in pl.known_plugin_names()
    names = pl.resolve_enabled_plugins(env=None, plugins=None,
                                       config_path=None, search_dirs=["/nonexistent"])
    assert "code" not in names or os.environ.get("MNEMOS_PLUGINS", "").find("code") >= 0


def test_plugin_tools_absent_without_flag(tmp_path):
    core = MnemosCore(Store(tmp_path / "nodes.json"), plugins=["context_engine", "gates"])
    names = {t["name"] for t in core.tools}
    assert not (names & {"code_search", "code_symbols", "code_callers", "code_refresh"})


def test_plugin_tools_present_when_enabled(code_core):
    names = {t["name"] for t in code_core.tools}
    assert {"code_search", "code_symbols", "code_callers", "code_refresh"} <= names
    # ядро памяти не потеряно
    assert {"memory_add", "memory_search"} <= names


def test_plugin_code_search(code_core):
    out = call(code_core, "code_search", {"query": "position_size_usd", "top_k": 3})
    assert out["count"] >= 1
    assert out["results"][0]["qname"] == "acmetrader.risk.position_size_usd"
    assert out["repos"] == ["acmetrader"]
    assert out["stale_seconds"] is not None
    assert "signature" in out["results"][0] and "score" in out["results"][0]


def test_plugin_code_search_validation(code_core):
    with pytest.raises(ValueError):
        call(code_core, "code_search", {"query": "   "})
    with pytest.raises(ValueError):
        call(code_core, "code_search", {"query": "x", "granularity": "мусор"})
    with pytest.raises(ValueError):
        call(code_core, "code_search", {"query": "x", "top_k": "много"})


def test_plugin_code_symbols_and_callers(code_core):
    syms = call(code_core, "code_symbols", {"file": "acmetrader/risk.py", "limit": 50})
    assert syms["count"] >= 4
    assert all(s["file"] == "acmetrader/risk.py" for s in syms["results"])
    callers = call(code_core, 
        "code_callers", {"qname": "acmetrader.risk.position_size_usd"})
    assert callers["direction"] == "inbound" and callers["count"] >= 1
    with pytest.raises(ValueError):
        call(code_core, "code_callers", {"qname": ""})
    with pytest.raises(ValueError):
        call(code_core, "code_callers", {"qname": "x", "direction": "боком"})


def test_plugin_code_refresh_reports_measurements(code_core):
    out = call(code_core, "code_refresh", {})
    row = out["repos"][0]
    assert row["repo"] == "acmetrader"
    assert row["mode"] in ("unchanged", "incremental", "full")
    assert row["files_scanned"] >= 3 and row["nodes"] > 0 and row["index_bytes"] > 0
    assert row["reindexed_ms"] >= 0.0
    assert "stale_seconds" in row
    assert out["watcher"] is None  # MNEMOS_CODE_WATCH=0


def test_plugin_does_not_touch_store(code_core):
    """Гарантия R3/R4: инструменты кода не пишут в стор решений."""
    before = code_core.store.path.read_bytes() if code_core.store.path.exists() else b""
    call(code_core, "code_search", {"query": "position_size_usd"})
    call(code_core, "code_symbols", {})
    call(code_core, "code_refresh", {})
    after = code_core.store.path.read_bytes() if code_core.store.path.exists() else b""
    assert before == after
    assert len(code_core.store) == 0
    assert not any(n.get("kind") == "code_symbol" for n in code_core.store.all())


# --- watcher -----------------------------------------------------------------


def test_watch_env_switches(monkeypatch):
    assert watch_enabled(None) is True
    for off in ("0", "off", "false", "NO"):
        assert watch_enabled(off) is False
    assert watch_enabled("1") is True
    assert watch_interval(None) == 60.0
    assert watch_interval("120") == 120.0
    assert watch_interval("0.1") == 5.0      # ниже минимума не опускаемся
    assert watch_interval("мусор") == 60.0


def test_watcher_tick_reindexes_change(registry, repo_root):
    watcher = CodeWatcher(registry, interval=5.0)
    stats = watcher.tick()
    assert stats and stats[0]["mode"] == "unchanged"
    path = os.path.join(str(repo_root), "acmetrader", "risk.py")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n\ndef символ_от_вотчера():\n    pass\n")
    os.utime(path, (0, 0))
    stats = watcher.tick()
    assert stats[0]["mode"] == "incremental"
    assert registry.search("символ_от_вотчера", top_k=3)
    st = watcher.status()
    assert st["ticks"] == 2 and st["repos_reindexed"] >= 1 and st["errors"] == []
    assert st["running"] is False


def test_watcher_thread_start_stop(registry):
    watcher = CodeWatcher(registry, interval=5.0)
    watcher.start()
    assert watcher.running is True
    watcher.stop(timeout=10)
    assert watcher.running is False
    assert watcher.ticks >= 1


def test_watcher_survives_broken_registry():
    class Boom:
        def refresh(self, force=False):
            raise RuntimeError("диск отвалился")

    watcher = CodeWatcher(Boom(), interval=5.0)
    assert watcher.tick() == []
    assert watcher.ticks == 1 and watcher.last_errors
