# -*- coding: utf-8 -*-
"""Тесты индексатора code-слоя (mnemos/code_index.py).

Покрывают: парс tree-sitter (module/class/function/method/route), рёбра
DEFINES/IMPORTS/CALLS/INHERITS/USAGE, токенизацию (кириллица, camelCase,
регистр), профиль расширений, лимит размера файла, исключения (.bak,
__pycache__, .venv, node_modules), манифест и инкрементальную
переиндексацию (изменение / добавление / удаление файла).
"""

import json
import os

import pytest

from mnemos import code_index as ci

pytestmark = pytest.mark.skipif(not ci.TS_OK, reason=f"нет py-tree-sitter: {ci.TS_ERR}")


SAMPLE = '''# -*- coding: utf-8 -*-
"""Модуль риска."""
import os
from decimal import Decimal


class RiskBase:
    """База риска."""

    def limit(self):
        return 1


class RiskEngine(RiskBase):
    """Движок риска — считает размер позиции."""

    def position_size_usd(self, equity, price):
        """Размер позиции в долларах."""
        return helper_calc(equity) * price


def helper_calc(equity):
    """Вспомогательный расчёт."""
    return Decimal(equity)


@app.route("/api/state")
def api_state():
    """Состояние — отдаёт стакан."""
    return helper_calc(1)
'''

CYR = '''# -*- coding: utf-8 -*-
"""Алиса: правило Акме про halt после серии убытков."""


def проверка_халта(серия):
    """Отключение торговли."""
    return серия > 3
'''


def make_repo(tmp_path, with_extras=True):
    """Небольшое дерево-репозиторий для индексации."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "risk.py").write_text(SAMPLE, encoding="utf-8")
    (root / "pkg" / "алиса.py").write_text(CYR, encoding="utf-8")
    (root / "README.md").write_text("# Проект\nПравило Акме: halt\n", encoding="utf-8")
    (root / "conf.yaml").write_text("halt: false\n", encoding="utf-8")
    if with_extras:
        # НЕ должны попасть в индекс
        (root / "data.json").write_text('{"a": 1}', encoding="utf-8")
        (root / "notes.txt").write_text("текст", encoding="utf-8")
        (root / "old.bak-0109.py").write_text("def bak_symbol():\n    pass\n", encoding="utf-8")
        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "x.py").write_text("def cached():\n    pass\n", encoding="utf-8")
        (root / ".venv").mkdir()
        (root / ".venv" / "lib.py").write_text("def venv_symbol():\n    pass\n", encoding="utf-8")
        (root / "node_modules").mkdir()
        (root / "node_modules" / "m.py").write_text("def node_symbol():\n    pass\n",
                                                    encoding="utf-8")
        (root / "pkg" / "big.py").write_text("x = 1\n" + "# " + "a" * 2_000_000 + "\n",
                                             encoding="utf-8")
    return root


# --- токенизация -------------------------------------------------------------


def test_tokenize_cyrillic_preserved():
    toks = ci.tokenize("Алиса halt после серии")
    assert "алиса" in toks and "halt" in toks and "серии" in toks


def test_tokenize_camel_case_and_lowercase():
    toks = ci.tokenize("PositionSizeUSD")
    assert "positionsizeusd" in toks
    assert "position" in toks and "size" in toks and "usd" in toks


def test_tokenize_snake_and_empty():
    assert set(ci.tokenize("position_size_usd")) == {"position", "size", "usd"}
    assert ci.tokenize("") == []
    assert ci.tokenize(None) == []


# --- обход дерева ------------------------------------------------------------


def test_iter_files_profile_and_exclusions(tmp_path):
    root = make_repo(tmp_path)
    files, skipped_big, skipped_bak = ci.iter_files(str(root))
    rels = {os.path.relpath(p, root).replace("\\", "/") for p in files}
    assert rels == {"pkg/risk.py", "pkg/алиса.py", "README.md", "conf.yaml"}
    assert skipped_big == 1          # pkg/big.py > 1 МиБ
    assert skipped_bak >= 1          # old.bak-0109.py
    assert not any("data.json" in r or "notes.txt" in r for r in rels)
    assert not any(".venv" in r or "__pycache__" in r or "node_modules" in r for r in rels)


def test_iter_files_skips_any_venv_flavour(tmp_path):
    root = make_repo(tmp_path)
    for name in (".venv-test", "venv3.12", "venv"):
        d = root / name
        d.mkdir()
        (d / "lib.py").write_text("def чужой_символ():\n    pass\n", encoding="utf-8")
    files, _, _ = ci.iter_files(str(root))
    rels = [os.path.relpath(p, root) for p in files]
    assert not any("venv" in r for r in rels)


def test_profile_all_includes_json(tmp_path):
    root = make_repo(tmp_path)
    files, _, _ = ci.iter_files(str(root), exts=ci.EXT_PROFILES["all"])
    rels = {os.path.relpath(p, root) for p in files}
    assert "data.json" in rels and "notes.txt" in rels


def test_max_file_bytes_off_includes_big(tmp_path):
    root = make_repo(tmp_path)
    files, skipped_big, _ = ci.iter_files(str(root), max_file_bytes=0)
    assert skipped_big == 0
    assert any(p.endswith("big.py") for p in files)


# --- парс --------------------------------------------------------------------


def test_parse_symbols_kinds(tmp_path):
    root = make_repo(tmp_path)
    g = ci.build_graph(str(root))
    by_qname = {n["qname"]: n for n in g["nodes"]}
    assert by_qname["pkg.risk"]["symbol_kind"] == "module"
    assert by_qname["pkg.risk.RiskEngine"]["symbol_kind"] == "class"
    assert by_qname["pkg.risk.helper_calc"]["symbol_kind"] == "function"
    m = by_qname["pkg.risk.RiskEngine.position_size_usd"]
    assert m["symbol_kind"] == "method"
    assert m["signature"].startswith("def position_size_usd(")
    assert "Размер позиции" in m["doc"]
    assert m["line"] < m["end_line"]
    r = by_qname["pkg.risk.api_state"]
    assert r["symbol_kind"] == "route" and r["route"] == "/api/state"


def test_parse_edges(tmp_path):
    root = make_repo(tmp_path)
    g = ci.build_graph(str(root))
    edges = {(e["s"], e["d"], e["t"]) for e in g["edges"]}
    assert ("pkg.risk", "pkg.risk.RiskEngine", "DEFINES") in edges
    assert ("pkg.risk.RiskEngine", "pkg.risk.RiskEngine.position_size_usd", "DEFINES") in edges
    assert ("pkg.risk", "os", "IMPORTS") in edges
    assert ("pkg.risk.RiskEngine.position_size_usd", "pkg.risk.helper_calc", "CALLS") in edges
    assert ("pkg.risk.RiskEngine", "pkg.risk.RiskBase", "INHERITS") in edges
    # неразрешимое имя -> USAGE, а не выдуманное ребро
    assert any(e["t"] == "USAGE" and e["d"] == "Decimal" for e in g["edges"])
    # каждое ребро знает свой файл (нужно для инкремента)
    assert all(e.get("f") for e in g["edges"])


def test_cyrillic_symbols_indexed(tmp_path):
    root = make_repo(tmp_path)
    g = ci.build_graph(str(root))
    assert "pkg.алиса.проверка_халта" in {n["qname"] for n in g["nodes"]}
    assert "алиса" in g["sym_index"]
    assert "халта" in g["sym_index"]


def test_file_index_covers_bodies(tmp_path):
    root = make_repo(tmp_path)
    g = ci.build_graph(str(root))
    assert "README.md" in g["files"] and "conf.yaml" in g["files"]
    fidx = {f: i for i, f in enumerate(g["files"])}
    assert fidx["conf.yaml"] in g["file_index"]["halt"]
    assert fidx["README.md"] in g["file_index"]["акме"]


def test_meta_and_manifest(tmp_path):
    root = make_repo(tmp_path)
    g = ci.build_graph(str(root), repo="repo")
    meta = g["meta"]
    assert meta["schema"] == ci.SCHEMA_VERSION
    assert meta["files_total"] == 4 and meta["py_files"] == 2
    assert meta["py_parsed"] == 2 and meta["py_failed"] == 0
    assert meta["skipped_too_big"] == 1
    assert set(g["manifest"]) == {"pkg/risk.py", "pkg/алиса.py", "README.md", "conf.yaml"}
    entry = g["manifest"]["pkg/risk.py"]
    assert len(entry["sha1"]) == 40 and entry["size"] > 0


def test_build_graph_missing_root(tmp_path):
    with pytest.raises(ci.CodeIndexError):
        ci.build_graph(str(tmp_path / "нет-такого"))


# --- запись/чтение -----------------------------------------------------------


def test_write_and_load_graph(tmp_path):
    root = make_repo(tmp_path)
    g = ci.build_graph(str(root))
    out = tmp_path / "graphs" / "repo.code_graph.json"
    size = ci.write_graph(g, str(out))
    assert out.exists() and size == out.stat().st_size
    assert not (tmp_path / "graphs" / "repo.code_graph.json.tmp").exists()
    back = ci.load_graph(str(out))
    assert back["meta"]["nodes"] == len(g["nodes"])


# --- репозитории -------------------------------------------------------------


def test_repo_name_and_roots_spec():
    assert ci.repo_name_for("/opt/acmetrader/") == "acmetrader"
    assert ci.parse_roots_spec("a=/opt/a, /opt/b") == [("a", "/opt/a"), ("b", "/opt/b")]
    assert ci.parse_roots_spec("") == []


def test_resolve_repos_filters_missing(tmp_path):
    root = make_repo(tmp_path)
    repos = ci.resolve_repos(data_dir=str(tmp_path / "code"),
                             roots=f"repo={root},ghost={tmp_path / 'ghost'}")
    assert [r.name for r in repos] == ["repo"]
    assert repos[0].graph_path.endswith("repo" + ci.GRAPH_SUFFIX)
    assert repos[0].exts == ci.EXT_PROFILES["code"]
    assert repos[0].max_file_bytes == ci.MAX_FILE_BYTES


def test_resolve_repos_env(tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    monkeypatch.setenv(ci.ENV_ROOTS, str(root))
    monkeypatch.setenv(ci.ENV_DATA_DIR, str(tmp_path / "envcode"))
    repos = ci.resolve_repos()
    assert [r.name for r in repos] == ["repo"]
    assert repos[0].graph_path.startswith(str(tmp_path / "envcode"))


# --- инкремент ---------------------------------------------------------------


def _repo(tmp_path):
    root = make_repo(tmp_path)
    return ci.Repo("repo", str(root), str(tmp_path / "code" / "repo.code_graph.json"))


def test_refresh_full_then_unchanged(tmp_path):
    repo = _repo(tmp_path)
    first = ci.refresh_repo(repo)
    assert first["mode"] == "full" and first["nodes"] > 0
    assert first["index_bytes"] > 0 and first["files_scanned"] == 4
    second = ci.refresh_repo(repo)
    assert second["mode"] == "unchanged" and second["changed"] == 0
    assert second["nodes"] == first["nodes"] and second["edges"] == first["edges"]


def test_refresh_unchanged_does_not_rewrite_graph(tmp_path):
    """Тик без изменений не переписывает граф (на /opt/acmetrader это 12 МБ)."""
    repo = _repo(tmp_path)
    ci.refresh_repo(repo)
    mtime = os.path.getmtime(repo.graph_path)
    size = os.path.getsize(repo.graph_path)
    stat = ci.refresh_repo(repo)
    assert stat["mode"] == "unchanged"
    assert os.path.getmtime(repo.graph_path) == mtime
    assert stat["index_bytes"] == size
    # свежесть отмечена в крошечном сайдкар-файле
    checked = ci.read_checked(repo.graph_path)
    assert checked is not None and os.path.getsize(ci.checked_path(repo.graph_path)) < 200


def test_refresh_reuses_in_memory_graph(tmp_path):
    repo = _repo(tmp_path)
    _, graph = ci.refresh_repo_graph(repo)
    os.remove(repo.graph_path)  # с диска читать нечего — должен взять переданный граф
    stat, graph2 = ci.refresh_repo_graph(repo, graph=graph)
    assert stat["mode"] == "unchanged" and graph2 is graph


def test_refresh_force_rebuilds(tmp_path):
    repo = _repo(tmp_path)
    ci.refresh_repo(repo)
    forced = ci.refresh_repo(repo, force=True)
    assert forced["mode"] == "full"


def test_refresh_incremental_on_change(tmp_path):
    repo = _repo(tmp_path)
    base = ci.refresh_repo(repo)
    path = os.path.join(repo.root, "pkg", "risk.py")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n\ndef новая_функция():\n    return helper_calc(2)\n")
    os.utime(path, (0, 0))  # mtime заведомо расходится с манифестом
    stat = ci.refresh_repo(repo)
    assert stat["mode"] == "incremental" and stat["changed"] == 1
    assert stat["nodes"] == base["nodes"] + 1
    g = ci.load_graph(repo.graph_path)
    qnames = {n["qname"] for n in g["nodes"]}
    assert "pkg.risk.новая_функция" in qnames
    # ребро CALLS нового символа резолвнуто по полной таблице имён
    assert any(e["s"] == "pkg.risk.новая_функция" and e["d"] == "pkg.risk.helper_calc"
               and e["t"] == "CALLS" for e in g["edges"])
    # старые файлы не потеряны
    assert "pkg.алиса.проверка_халта" in qnames
    assert set(g["manifest"]) == {"pkg/risk.py", "pkg/алиса.py", "README.md", "conf.yaml"}


def test_refresh_incremental_add_and_delete(tmp_path):
    repo = _repo(tmp_path)
    ci.refresh_repo(repo)
    new_file = os.path.join(repo.root, "pkg", "extra.py")
    with open(new_file, "w", encoding="utf-8") as f:
        f.write("def уникальный_символ():\n    pass\n")
    stat = ci.refresh_repo(repo)
    assert stat["mode"] == "incremental" and stat["changed"] == 1
    g = ci.load_graph(repo.graph_path)
    assert "pkg.extra.уникальный_символ" in {n["qname"] for n in g["nodes"]}
    fidx = {f: i for i, f in enumerate(g["files"])}
    assert fidx["pkg/extra.py"] in g["file_index"]["уникальный"]

    os.remove(new_file)
    stat2 = ci.refresh_repo(repo)
    assert stat2["deleted"] == 1
    g2 = ci.load_graph(repo.graph_path)
    assert "pkg.extra.уникальный_символ" not in {n["qname"] for n in g2["nodes"]}
    assert "pkg/extra.py" not in g2["files"]
    assert not any(e.get("f") == "pkg/extra.py" for e in g2["edges"])
    # постинги перенумерованы корректно: ни один индекс не вышел за границы
    for postings in g2["file_index"].values():
        assert all(0 <= i < len(g2["files"]) for i in postings)


def test_refresh_same_content_touched_is_not_reparsed(tmp_path):
    """mtime изменился, содержимое — нет: sha1 решает, перепарса быть не должно."""
    repo = _repo(tmp_path)
    ci.refresh_repo(repo)
    path = os.path.join(repo.root, "pkg", "risk.py")
    os.utime(path, (12345, 12345))
    stat = ci.refresh_repo(repo)
    assert stat["mode"] == "unchanged" and stat["changed"] == 0


def test_refresh_broken_graph_falls_back_to_full(tmp_path):
    repo = _repo(tmp_path)
    ci.refresh_repo(repo)
    with open(repo.graph_path, "w", encoding="utf-8") as f:
        f.write("{битый json")
    stat = ci.refresh_repo(repo)
    assert stat["mode"] == "full" and stat["nodes"] > 0
    assert json.loads(open(repo.graph_path, encoding="utf-8").read())["meta"]["nodes"] > 0


def test_refresh_all_reports_error_per_repo(tmp_path):
    ok = _repo(tmp_path)
    bad = ci.Repo("ghost", str(tmp_path / "ghost"), str(tmp_path / "code" / "g.json"))
    stats = ci.refresh_all([ok, bad])
    assert stats[0]["mode"] == "full"
    assert "error" in stats[1]
