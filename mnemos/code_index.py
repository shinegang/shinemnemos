# -*- coding: utf-8 -*-
"""ShineMnemos: индексатор кода (code-слой) — py-tree-sitter -> code_graph.json.

Перенос PoC (/opt/bench-memory/poc/index_code.py, 373 строки, замерен на
боевых деревьях) в состав Mnemos с добавлением мультирепо и инкремента.

ПРАВИЛО 1 (архитектурная гарантия): этот модуль НЕ трогает боевой стор
решений (data/nodes.json). Код живёт в отдельных файлах
  <data_dir>/code/<repo>.code_graph.json
— один файл на репозиторий, чтобы переиндексация одного корня не переписывала
граф другого. Обоснование раздельного хранения — §2.2 отчёта
ФЭЙБЛ-ИССЛЕДОВАНИЕ-ИНДЕКСАЦИЯ-01.09 (recall решений 0.944 против 0.767 при
смешивании; латентность memory_search 0.164 мс против 5.538 мс).

Что делает:
  * обходит корень репозитория, берёт файлы профиля "code"
    (.py .sh .yaml .yml .toml .cfg .ini .md — БЕЗ .json/.txt: на боевом
    /opt/acmetrader .json это 88.2% байт и 0 символов, §4.5 отчёта);
  * лимит 1 МиБ на файл; исключения: .git, .venv, node_modules, __pycache__,
    .pytest_cache, .mypy_cache и всё, в имени чего есть ".bak";
  * .py -> узлы kind=code_symbol: module / class / function / method / route;
  * рёбра: DEFINES, IMPORTS, CALLS (резолв по имени), INHERITS, USAGE;
  * два инвертированных индекса: по символам и по ТЕЛАМ файлов;
  * манифест mtime/size/sha1 -> инкрементальная переиндексация.

Только stdlib + tree-sitter (MIT) + tree-sitter-python (MIT).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # честный БЛОКЕР, а не тихий фолбэк на регулярки
    from tree_sitter import Language, Parser
    import tree_sitter_python as tsp

    TS_OK = True
    TS_ERR = ""
except Exception as exc:  # pragma: no cover - зависит от окружения
    TS_OK = False
    TS_ERR = repr(exc)

PY_LANG = Language(tsp.language()) if TS_OK else None


# ============================================================================
# Конфигурация индексации
# ============================================================================

# Профиль "code" — ДЕФОЛТ и требование внедрения (§4.5): без .json/.txt.
EXT_PROFILES: Dict[str, frozenset] = {
    "code": frozenset({".py", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".md"}),
    "all": frozenset({".py", ".md", ".yaml", ".yml", ".sh", ".json", ".toml",
                      ".cfg", ".ini", ".txt"}),
}
DEFAULT_PROFILE = "code"
TEXT_EXT = EXT_PROFILES[DEFAULT_PROFILE]

SKIP_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__",
                       ".pytest_cache", ".mypy_cache", ".ruff_cache"})
# ".bak" в любом виде: acmetrader.bak/, .bak-0109/, x.bak-moltbook-0109.py
BAK_RE = re.compile(r"\.bak", re.IGNORECASE)
# любые окружения: venv, .venv, .venv-test, venv3.12 — там чужой код, не наш
VENV_RE = re.compile(r"^\.?venv", re.IGNORECASE)

MAX_FILE_BYTES = 1 << 20  # 1 МиБ: крупнее — это данные, а не исходник

# Корни по умолчанию (VPS). Несуществующие молча пропускаются — на 5090
# набор каталогов другой, падать из-за этого сервер не должен.
DEFAULT_ROOTS: Tuple[str, ...] = (
    "/opt/acmetrader",
    "/opt/panel",
    "/opt/acme-hub",
    "/opt/acme-site",
    "/opt/mnemos",
)

ENV_ROOTS = "MNEMOS_CODE_ROOTS"        # "имя=/путь,/другой/путь"
ENV_DATA_DIR = "MNEMOS_CODE_DATA"      # каталог для *.code_graph.json
ENV_PROFILE = "MNEMOS_CODE_PROFILE"    # code | all
ENV_MAX_BYTES = "MNEMOS_CODE_MAX_FILE_BYTES"

GRAPH_SUFFIX = ".code_graph.json"
SCHEMA_VERSION = 1

# токенизация: латиница+цифры+КИРИЛЛИЦА (кириллицу терять нельзя)
_SPLIT_RE = re.compile(r"[^0-9A-Za-zЀ-ӿ]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+|[Ѐ-ӿ]+")

ROUTE_RE = re.compile(
    r"@[\w\.]*(?:route|get|post|put|delete|patch)\s*\(\s*[\"']([^\"']+)[\"']", re.I
)


class CodeIndexError(RuntimeError):
    """Индексация невозможна (нет tree-sitter, недоступен корень и т.п.)."""


def require_tree_sitter() -> None:
    if not TS_OK:
        raise CodeIndexError(
            "py-tree-sitter недоступен: " + TS_ERR
            + " (нужны tree-sitter>=0.25 и tree-sitter-python>=0.25)"
        )


def tokenize(text: Any) -> List[str]:
    """Токены: части по разделителям + camelCase-разбиение. Всё в нижний регистр."""
    out: List[str] = []
    for part in _SPLIT_RE.split(text or ""):
        if not part:
            continue
        out.append(part.lower())
        subs = _CAMEL_RE.findall(part)
        if len(subs) > 1:
            out.extend(s.lower() for s in subs)
    return out


def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# Разбор Python через tree-sitter
# ============================================================================


def _txt(src: bytes, node: Any) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _field(node: Any, name: str) -> Any:
    return node.child_by_field_name(name)


def _docstring(src: bytes, body: Any) -> str:
    if body is None or body.named_child_count == 0:
        return ""
    first = body.named_child(0)
    if first.type == "expression_statement" and first.named_child_count:
        s = first.named_child(0)
        if s.type == "string":
            return _txt(src, s)[:400]
    return ""


def _decorator_names(src: bytes, decorated: Any) -> List[str]:
    return [_txt(src, ch).strip() for ch in decorated.children if ch.type == "decorator"]


def _import_names(src: bytes, node: Any) -> List[str]:
    out: List[str] = []
    mod = _field(node, "module_name")
    if mod is not None:
        out.append(_txt(src, mod))
    for ch in node.children:
        if ch.type in ("dotted_name", "aliased_import") and ch != mod:
            out.append(_txt(src, ch).split(" as ")[0].strip())
    return [o for o in out if o]


def _calls_in(src: bytes, body: Any) -> List[str]:
    """Имена вызываемых функций внутри поддерева (обход без рекурсии)."""
    out: List[str] = []
    stack = [body]
    while stack:
        n = stack.pop()
        if n.type == "call":
            fn = _field(n, "function")
            if fn is not None:
                t = _txt(src, fn)
                out.append(t.split(".")[-1] if "." in t else t)
        stack.extend(n.children)
    return out


def module_qname(rel: str) -> str:
    return rel[:-3].replace("/", ".").replace("\\", ".")


DESCEND_TYPES = frozenset({
    "block", "if_statement", "try_statement", "with_statement", "for_statement",
    "while_statement", "else_clause", "elif_clause", "except_clause",
    "finally_clause", "module",
})


def parse_python(rel: str, src_bytes: bytes, parser: Any) -> Tuple[List[Dict], List[Dict]]:
    """-> (nodes, edges). Узлы module/class/function/method(+route), рёбра сырые.

    ОТЛИЧИЕ ОТ PoC: `decorated_definition` разбирается на месте, а не кладётся
    в стек как контейнер. В PoC (index_code.py:120) декорированное определение
    клалось в стек и затем обходились его ДЕТИ — сам символ узлом не становился,
    из-за чего терялись все декорированные функции, включая @route. Поэтому
    число узлов у порта выше, чем в замерах PoC (§4.1-4.2 отчёта).
    """
    tree = parser.parse(src_bytes)
    root = tree.root_node
    mod_q = module_qname(rel)
    nodes: List[Dict[str, Any]] = [{
        "id": mod_q, "kind": "code_symbol", "symbol_kind": "module",
        "name": mod_q.split(".")[-1], "qname": mod_q, "file": rel,
        "line": 1, "end_line": root.end_point[0] + 1, "signature": "", "doc": "",
    }]
    edges: List[Dict[str, Any]] = []
    stack: List[Tuple[Any, Optional[str], List[str]]] = [(root, None, [])]

    def emit_function(ch: Any, cls: Optional[str], decos: List[str]) -> None:
        nm = _field(ch, "name")
        if nm is None:
            return
        name = _txt(src_bytes, nm)
        qn = f"{cls}.{name}" if cls else f"{mod_q}.{name}"
        params = _field(ch, "parameters")
        sig = f"def {name}{_txt(src_bytes, params) if params is not None else '()'}"
        rec: Dict[str, Any] = {
            "id": qn, "kind": "code_symbol",
            "symbol_kind": "method" if cls else "function", "name": name,
            "qname": qn, "file": rel, "line": ch.start_point[0] + 1,
            "end_line": ch.end_point[0] + 1, "signature": sig,
            "doc": _docstring(src_bytes, _field(ch, "body")),
        }
        if decos:
            rec["decorators"] = decos
            m = ROUTE_RE.search(" ".join(decos))
            if m:
                rec["symbol_kind"] = "route"
                rec["route"] = m.group(1)
        nodes.append(rec)
        edges.append({"s": cls or mod_q, "d": qn, "t": "DEFINES", "f": rel})
        body = _field(ch, "body")
        if body is not None:
            for callee in _calls_in(src_bytes, body):
                edges.append({"s": qn, "d": callee, "t": "CALLS_RAW", "f": rel})

    def emit_class(ch: Any, decos: List[str]) -> None:
        nm = _field(ch, "name")
        if nm is None:
            return
        name = _txt(src_bytes, nm)
        qn = f"{mod_q}.{name}"
        rec: Dict[str, Any] = {
            "id": qn, "kind": "code_symbol", "symbol_kind": "class",
            "name": name, "qname": qn, "file": rel,
            "line": ch.start_point[0] + 1, "end_line": ch.end_point[0] + 1,
            "signature": f"class {name}",
            "doc": _docstring(src_bytes, _field(ch, "body")),
        }
        if decos:
            rec["decorators"] = decos
        nodes.append(rec)
        edges.append({"s": mod_q, "d": qn, "t": "DEFINES", "f": rel})
        sup = _field(ch, "superclasses")
        if sup is not None:
            for base in _SPLIT_RE.split(_txt(src_bytes, sup)):
                if base and base != "object":
                    edges.append({"s": qn, "d": base, "t": "INHERITS_RAW", "f": rel})
        body = _field(ch, "body")
        if body is not None:
            stack.append((body, qn, []))

    def handle(ch: Any, cls: Optional[str], decos: List[str]) -> None:
        if ch.type == "function_definition":
            emit_function(ch, cls, decos)
        elif ch.type == "class_definition":
            emit_class(ch, decos)
        elif ch.type in ("import_statement", "import_from_statement"):
            for name in _import_names(src_bytes, ch):
                edges.append({"s": mod_q, "d": name, "t": "IMPORTS", "f": rel})
        elif ch.type in DESCEND_TYPES:
            stack.append((ch, cls, []))

    while stack:
        node, cls, _decos = stack.pop()
        for ch in node.children:
            if ch.type == "decorated_definition":
                inner = _field(ch, "definition")
                if inner is not None:
                    handle(inner, cls, _decorator_names(src_bytes, ch))
                continue
            handle(ch, cls, [])
    return nodes, edges


def resolve_calls(nodes: Sequence[Dict], edges: Sequence[Dict]) -> Tuple[List[Dict], int, int]:
    """CALLS_RAW/INHERITS_RAW -> CALLS/INHERITS при однозначном имени, иначе USAGE."""
    by_name: Dict[str, List[str]] = {}
    for n in nodes:
        by_name.setdefault(n["name"], []).append(n["qname"])
    out: List[Dict[str, Any]] = []
    resolved = usage = 0
    for e in edges:
        if e["t"] not in ("CALLS_RAW", "INHERITS_RAW"):
            out.append(e)
            continue
        real = e["t"][:-4] if e["t"].endswith("_RAW") else e["t"]
        cands = by_name.get(e["d"], [])
        target = None
        if len(cands) == 1:
            target = cands[0]
        elif len(cands) > 1:
            mod = e["s"].rsplit(".", 1)[0]
            same = [c for c in cands if c.startswith(mod + ".")]
            if len(same) == 1:
                target = same[0]
        if target is not None:
            out.append({"s": e["s"], "d": target, "t": real, "f": e["f"]})
            resolved += 1
        else:
            out.append({"s": e["s"], "d": e["d"], "t": "USAGE", "f": e["f"]})
            usage += 1
    return out, resolved, usage


# ============================================================================
# Обход дерева и индексы
# ============================================================================


def iter_files(root: str, exts: Iterable[str] = TEXT_EXT,
               max_file_bytes: int = MAX_FILE_BYTES,
               skip_dirs: Iterable[str] = SKIP_DIRS) -> Tuple[List[str], int, int]:
    """-> (абсолютные пути, пропущено крупных, пропущено .bak)."""
    exts = set(exts)
    skip = set(skip_dirs)
    files: List[str] = []
    skipped_big = skipped_bak = 0
    for dirpath, dirnames, filenames in os.walk(root):
        kept = []
        for d in dirnames:
            if d in skip or VENV_RE.match(d):
                continue
            if BAK_RE.search(d):
                skipped_bak += 1
                continue
            kept.append(d)
        dirnames[:] = kept
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() not in exts:
                continue
            if BAK_RE.search(fn):
                skipped_bak += 1
                continue
            p = os.path.join(dirpath, fn)
            try:
                if max_file_bytes and os.path.getsize(p) > max_file_bytes:
                    skipped_big += 1
                    continue
            except OSError:
                continue
            files.append(p)
    return sorted(files), skipped_big, skipped_bak


def build_sym_index(nodes: Sequence[Dict]) -> Dict[str, List[int]]:
    inv: Dict[str, List[int]] = {}
    for i, n in enumerate(nodes):
        toks = set(tokenize(n["qname"])) | set(tokenize(n["name"])) \
            | set(tokenize(n["file"])) | set(tokenize(n.get("doc", ""))) \
            | set(tokenize(n.get("signature", ""))) | set(tokenize(n.get("route", "")))
        for t in toks:
            inv.setdefault(t, []).append(i)
    return inv


def file_body_tokens(path: str, rel: str) -> List[str]:
    with open(path, "rb") as f:
        raw = f.read()
    return sorted(set(tokenize(rel) + tokenize(raw.decode("utf-8", "replace"))))


def _now() -> float:
    return time.time()


def _utc(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


# ============================================================================
# Полная сборка графа
# ============================================================================


def build_graph(root: str, exts: Iterable[str] = TEXT_EXT,
                max_file_bytes: int = MAX_FILE_BYTES,
                skip_dirs: Iterable[str] = SKIP_DIRS,
                repo: str = "") -> Dict[str, Any]:
    """Полная индексация корня -> граф (в памяти). Ничего не пишет на диск."""
    require_tree_sitter()
    if not os.path.isdir(root):
        raise CodeIndexError(f"каталог не существует: {root}")
    parser = Parser(PY_LANG)
    exts = set(exts)

    t0 = time.perf_counter()
    files, skipped_big, skipped_bak = iter_files(root, exts, max_file_bytes, skip_dirs)
    t_walk = time.perf_counter() - t0

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    manifest: Dict[str, Dict[str, Any]] = {}
    file_tokens: Dict[str, List[str]] = {}
    py_files = parsed_py = failed_py = 0
    parse_errors: List[str] = []
    src_bytes_total = 0

    t1 = time.perf_counter()
    for path in files:
        rel = os.path.relpath(path, root).replace("\\", "/")
        try:
            st = os.stat(path)
            digest = sha1_file(path)
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            continue
        manifest[rel] = {"mtime": st.st_mtime, "size": st.st_size, "sha1": digest}
        src_bytes_total += st.st_size
        text = raw.decode("utf-8", "replace")
        file_tokens[rel] = sorted(set(tokenize(rel) + tokenize(text)))
        if rel.endswith(".py"):
            py_files += 1
            try:
                n, e = parse_python(rel, raw, parser)
                nodes.extend(n)
                edges.extend(e)
                parsed_py += 1
            except Exception as exc:  # pragma: no cover - на боевых 892/892 ок
                failed_py += 1
                if len(parse_errors) < 20:
                    parse_errors.append(f"{rel}: {exc!r}")
    t_parse = time.perf_counter() - t1

    t2 = time.perf_counter()
    edges, n_resolved, n_usage = resolve_calls(nodes, edges)
    sym_index = build_sym_index(nodes)
    rel_files = sorted(file_tokens)
    fidx = {r: i for i, r in enumerate(rel_files)}
    file_index: Dict[str, List[int]] = {}
    for r in rel_files:
        for t in file_tokens[r]:
            file_index.setdefault(t, []).append(fidx[r])
    t_index = time.perf_counter() - t2

    now = _now()
    return {
        "meta": {
            "schema": SCHEMA_VERSION,
            "repo": repo or os.path.basename(os.path.abspath(root)),
            "root": os.path.abspath(root),
            "built_ts": now, "built_utc": _utc(now),
            "checked_ts": now, "checked_utc": _utc(now),
            "mode": "full",
            "files_total": len(manifest), "py_files": py_files,
            "py_parsed": parsed_py, "py_failed": failed_py,
            "parse_errors": parse_errors,
            "skipped_too_big": skipped_big, "skipped_bak": skipped_bak,
            "ext_profile": sorted(exts), "max_file_bytes": max_file_bytes,
            "src_bytes": src_bytes_total,
            "nodes": len(nodes), "edges": len(edges),
            "calls_resolved": n_resolved, "usage_unresolved": n_usage,
            "timings_ms": {
                "walk": round(t_walk * 1000, 1),
                "parse": round(t_parse * 1000, 1),
                "index": round(t_index * 1000, 1),
                "total": round((time.perf_counter() - t0) * 1000, 1),
            },
        },
        "nodes": nodes, "edges": edges,
        "sym_index": sym_index, "files": rel_files, "file_index": file_index,
        "manifest": manifest,
    }


def write_graph(graph: Dict[str, Any], path: str) -> int:
    """Атомарная запись графа. -> размер файла в байтах."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False)
    os.replace(tmp, p)
    size = p.stat().st_size
    graph["meta"]["index_bytes"] = size
    return size


def load_graph(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Отметка "индекс сверялся с диском" — отдельный крошечный файл рядом с графом.
# Писать ради неё весь граф (12 МБ на /opt/acmetrader) нельзя: тик вотчера
# раз в 60 с превратился бы в 12 МБ записи и ~800 мс вместо ~55 мс скана.
CHECKED_SUFFIX = ".checked"


def checked_path(graph_path: str) -> str:
    return graph_path + CHECKED_SUFFIX


def write_checked(graph_path: str, ts: float) -> None:
    p = Path(checked_path(graph_path))
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"checked_ts": ts, "checked_utc": _utc(ts)}, f)
        os.replace(tmp, p)
    except OSError:  # отметка свежести — не повод падать
        pass


def read_checked(graph_path: str) -> Optional[float]:
    try:
        with open(checked_path(graph_path), encoding="utf-8") as f:
            return float(json.load(f)["checked_ts"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


# ============================================================================
# Инкрементальная переиндексация
# ============================================================================


def scan_changes(root: str, manifest: Dict[str, Dict[str, Any]],
                 exts: Iterable[str] = TEXT_EXT,
                 max_file_bytes: int = MAX_FILE_BYTES,
                 skip_dirs: Iterable[str] = SKIP_DIRS) -> Dict[str, Any]:
    """Скан mtime+size, при расхождении — sha1.

    -> {"files": {rel: abs}, "changed": [rel], "deleted": [rel], "scanned": n}
    """
    files, _, _ = iter_files(root, exts, max_file_bytes, skip_dirs)
    present: Dict[str, str] = {}
    changed: List[str] = []
    for path in files:
        rel = os.path.relpath(path, root).replace("\\", "/")
        present[rel] = path
        old = manifest.get(rel)
        try:
            st = os.stat(path)
        except OSError:
            continue
        if old is None:
            changed.append(rel)
            continue
        if old.get("mtime") == st.st_mtime and old.get("size") == st.st_size:
            continue
        # mtime/size разошлись — решает содержимое
        try:
            if sha1_file(path) != old.get("sha1"):
                changed.append(rel)
            else:
                # тот же контент: подтянуть mtime, чтобы не хэшировать снова
                old["mtime"] = st.st_mtime
                old["size"] = st.st_size
        except OSError:
            continue
    deleted = [rel for rel in manifest if rel not in present]
    return {"files": present, "changed": sorted(changed),
            "deleted": sorted(deleted), "scanned": len(present)}


def apply_changes(graph: Dict[str, Any], root: str, changed: Sequence[str],
                  deleted: Sequence[str], files: Dict[str, str]) -> Dict[str, Any]:
    """Перепарсить только изменённые файлы и перестроить индексы. Мутирует graph."""
    require_tree_sitter()
    parser = Parser(PY_LANG)
    affected = set(changed) | set(deleted)
    if not affected:
        return graph

    t0 = time.perf_counter()
    # 1. выкинуть узлы и рёбра затронутых файлов
    graph["nodes"] = [n for n in graph["nodes"] if n["file"] not in affected]
    graph["edges"] = [e for e in graph["edges"] if e.get("f") not in affected]
    manifest = graph["manifest"]
    for rel in deleted:
        manifest.pop(rel, None)

    # 2. перепарсить изменённые
    new_nodes: List[Dict[str, Any]] = []
    new_raw: List[Dict[str, Any]] = []
    new_tokens: Dict[str, List[str]] = {}
    py_failed = 0
    for rel in changed:
        path = files.get(rel)
        if path is None:
            continue
        try:
            st = os.stat(path)
            digest = sha1_file(path)
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            continue
        manifest[rel] = {"mtime": st.st_mtime, "size": st.st_size, "sha1": digest}
        new_tokens[rel] = sorted(set(tokenize(rel) + tokenize(raw.decode("utf-8", "replace"))))
        if rel.endswith(".py"):
            try:
                n, e = parse_python(rel, raw, parser)
                new_nodes.extend(n)
                new_raw.extend(e)
            except Exception:  # pragma: no cover
                py_failed += 1

    # 3. резолв новых рёбер по ПОЛНОЙ таблице имён (старые + новые узлы)
    all_nodes = graph["nodes"] + new_nodes
    resolved, n_res, n_use = resolve_calls(all_nodes, new_raw)
    graph["nodes"] = all_nodes
    graph["edges"].extend(resolved)

    # 4. индекс символов — полная пересборка (только память, без диска)
    graph["sym_index"] = build_sym_index(graph["nodes"])

    # 5. индекс тел файлов — точечное обновление постингов
    graph["files"], graph["file_index"] = _reindex_files(
        graph["files"], graph["file_index"], affected, new_tokens
    )

    now = _now()
    meta = graph["meta"]
    meta.update({
        "built_ts": now, "built_utc": _utc(now),
        "checked_ts": now, "checked_utc": _utc(now),
        "mode": "incremental",
        "files_total": len(manifest),
        "py_files": sum(1 for r in manifest if r.endswith(".py")),
        "py_failed": py_failed,
        "nodes": len(graph["nodes"]), "edges": len(graph["edges"]),
        "src_bytes": sum(int(m.get("size") or 0) for m in manifest.values()),
        "timings_ms": {"total": round((time.perf_counter() - t0) * 1000, 1),
                       "resolved": n_res, "usage": n_use},
    })
    return graph


def _reindex_files(files: Sequence[str], file_index: Dict[str, List[int]],
                   removed: Iterable[str],
                   new_tokens: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, List[int]]]:
    """Убрать из постингов затронутые файлы и добавить свежие токены."""
    removed = set(removed)
    keep = [r for r in files if r not in removed]
    added = sorted(r for r in new_tokens if r not in set(keep))
    new_files = keep + added
    pos = {r: i for i, r in enumerate(new_files)}
    old_pos = {r: i for i, r in enumerate(files)}
    remap = {old_pos[r]: pos[r] for r in keep}

    out: Dict[str, List[int]] = {}
    for t, lst in file_index.items():
        nl = [remap[i] for i in lst if i in remap]
        if nl:
            out[t] = nl
    touched = set()
    for r, toks in new_tokens.items():
        i = pos[r]
        for t in toks:
            out.setdefault(t, []).append(i)
            touched.add(t)
    for t in touched:
        out[t].sort()
    return new_files, out


# ============================================================================
# Репозитории (мультирепо)
# ============================================================================


class Repo:
    """Один индексируемый корень: имя, путь, файл графа."""

    __slots__ = ("name", "root", "graph_path", "exts", "max_file_bytes", "skip_dirs")

    def __init__(self, name: str, root: str, graph_path: str,
                 exts: Iterable[str] = TEXT_EXT,
                 max_file_bytes: int = MAX_FILE_BYTES,
                 skip_dirs: Iterable[str] = SKIP_DIRS) -> None:
        self.name = name
        self.root = os.path.abspath(root)
        self.graph_path = graph_path
        self.exts = frozenset(exts)
        self.max_file_bytes = max_file_bytes
        self.skip_dirs = frozenset(skip_dirs)

    def __repr__(self) -> str:  # pragma: no cover - диагностика
        return f"Repo({self.name!r}, {self.root!r})"

    def exists(self) -> bool:
        return os.path.isdir(self.root)


_NAME_RE = re.compile(r"[^0-9A-Za-z_.-]+")


def repo_name_for(root: str) -> str:
    base = os.path.basename(os.path.abspath(root)) or "root"
    return _NAME_RE.sub("_", base).strip("_.") or "root"


def parse_roots_spec(spec: str) -> List[Tuple[str, str]]:
    """"имя=/путь,/другой" -> [(имя, путь)]. Разделители — запятая и ';'."""
    out: List[Tuple[str, str]] = []
    for chunk in re.split(r"[,;]", spec or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            name, _, path = chunk.partition("=")
            name, path = name.strip(), path.strip()
        else:
            name, path = "", chunk
        if not path:
            continue
        out.append((name or repo_name_for(path), path))
    return out


def default_data_dir(store_path: Optional[str] = None) -> str:
    """Каталог графов: env -> <каталог стора>/code -> ./data/code."""
    env = os.environ.get(ENV_DATA_DIR)
    if env:
        return env
    if store_path:
        return str(Path(store_path).resolve().parent / "code")
    return str(Path("data").resolve() / "code")


def resolve_repos(data_dir: Optional[str] = None, roots: Any = None,
                  store_path: Optional[str] = None,
                  require_existing: bool = True) -> List[Repo]:
    """Список репозиториев: аргумент -> env MNEMOS_CODE_ROOTS -> DEFAULT_ROOTS.

    Несуществующие каталоги отсеиваются (require_existing=True) — набор
    каталогов на VPS и на 5090 разный, сервер из-за этого падать не должен.
    """
    data_dir = data_dir or default_data_dir(store_path)
    if roots is None:
        env = os.environ.get(ENV_ROOTS)
        pairs = parse_roots_spec(env) if env is not None else \
            [(repo_name_for(r), r) for r in DEFAULT_ROOTS]
    elif isinstance(roots, str):
        pairs = parse_roots_spec(roots)
    else:
        pairs = []
        for item in roots:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                pairs.append((str(item[0]), str(item[1])))
            else:
                pairs.append((repo_name_for(str(item)), str(item)))

    profile = os.environ.get(ENV_PROFILE, DEFAULT_PROFILE)
    exts = EXT_PROFILES.get(profile, TEXT_EXT)
    try:
        max_bytes = int(os.environ.get(ENV_MAX_BYTES, MAX_FILE_BYTES))
    except (TypeError, ValueError):
        max_bytes = MAX_FILE_BYTES

    out: List[Repo] = []
    seen = set()
    for name, root in pairs:
        if name in seen:
            continue
        seen.add(name)
        repo = Repo(name, root, os.path.join(data_dir, name + GRAPH_SUFFIX),
                    exts=exts, max_file_bytes=max_bytes)
        if require_existing and not repo.exists():
            continue
        out.append(repo)
    return out


def refresh_repo_graph(repo: Repo, force: bool = False,
                       graph: Optional[Dict[str, Any]] = None
                       ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Обновить граф репозитория. -> (статистика, граф).

    force или отсутствующий/битый граф -> полная сборка;
    иначе скан mtime+size(+sha1) и перепарс только изменённых файлов.

    graph — уже загруженный граф (передаёт CodeRegistry): экономит чтение
    и разбор 12 МБ JSON на каждом тике вотчера.
    Ничего не изменилось -> граф НЕ перезаписывается, обновляется только
    отметка свежести <graph>.checked.
    """
    t0 = time.perf_counter()
    mode = "full"
    scanned = 0
    changed: List[str] = []
    deleted: List[str] = []

    if force:
        graph = None
    if graph is not None and (graph.get("meta", {}).get("schema") != SCHEMA_VERSION
                              or graph.get("meta", {}).get("root") != repo.root):
        graph = None
    if graph is None and not force and os.path.exists(repo.graph_path):
        try:
            graph = load_graph(repo.graph_path)
            meta = graph.get("meta", {})
            if meta.get("schema") != SCHEMA_VERSION or meta.get("root") != repo.root:
                graph = None  # другая схема/переехавший корень — честнее пересобрать
        except (OSError, json.JSONDecodeError, ValueError):
            graph = None

    if graph is None:
        graph = build_graph(repo.root, repo.exts, repo.max_file_bytes,
                            repo.skip_dirs, repo=repo.name)
        scanned = graph["meta"]["files_total"]
        changed = sorted(graph["manifest"])
        index_bytes = write_graph(graph, repo.graph_path)
    else:
        scan = scan_changes(repo.root, graph["manifest"], repo.exts,
                            repo.max_file_bytes, repo.skip_dirs)
        scanned, changed, deleted = scan["scanned"], scan["changed"], scan["deleted"]
        if changed or deleted:
            mode = "incremental"
            apply_changes(graph, repo.root, changed, deleted, scan["files"])
            index_bytes = write_graph(graph, repo.graph_path)
        else:
            mode = "unchanged"
            index_bytes = graph["meta"].get("index_bytes") or (
                os.path.getsize(repo.graph_path) if os.path.exists(repo.graph_path) else 0)

    now = _now()
    graph["meta"]["checked_ts"] = now
    graph["meta"]["checked_utc"] = _utc(now)
    graph["meta"]["index_bytes"] = index_bytes
    write_checked(repo.graph_path, now)

    meta = graph["meta"]
    stats = {
        "repo": repo.name, "root": repo.root, "mode": mode,
        "files_scanned": scanned, "changed": len(changed), "deleted": len(deleted),
        "reindexed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "nodes": meta["nodes"], "edges": meta["edges"],
        "index_bytes": index_bytes,
        "built_utc": meta["built_utc"], "checked_utc": meta["checked_utc"],
        "stale_seconds": 0.0,
        "graph_path": repo.graph_path,
    }
    return stats, graph


def refresh_repo(repo: Repo, force: bool = False,
                 graph: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """То же, но возвращает только статистику (JSON-совместимый ответ)."""
    return refresh_repo_graph(repo, force=force, graph=graph)[0]


def refresh_all(repos: Sequence[Repo], force: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for repo in repos:
        try:
            out.append(refresh_repo(repo, force=force))
        except Exception as exc:
            out.append({"repo": repo.name, "root": repo.root, "error": repr(exc)})
    return out


# ============================================================================
# CLI: python3 -m mnemos.code_index --root <dir> --out <graph.json>
# ============================================================================


def _main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description="Индексация кода для Mnemos")
    ap.add_argument("--root", help="корень репозитория (одиночный режим)")
    ap.add_argument("--out", help="файл графа (одиночный режим)")
    ap.add_argument("--data-dir", help="каталог графов (мультирепо)")
    ap.add_argument("--repos", help="'имя=/путь,/путь' (по умолчанию — DEFAULT_ROOTS)")
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(EXT_PROFILES))
    ap.add_argument("--max-file-bytes", type=int, default=MAX_FILE_BYTES)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    if not TS_OK:
        print("БЛОКЕР: py-tree-sitter недоступен: " + TS_ERR, file=sys.stderr)
        return 3

    if a.root:
        out = a.out or os.path.join(a.data_dir or ".", repo_name_for(a.root) + GRAPH_SUFFIX)
        repo = Repo(repo_name_for(a.root), a.root, out,
                    exts=EXT_PROFILES[a.profile], max_file_bytes=a.max_file_bytes)
        print(json.dumps(refresh_repo(repo, force=a.force), ensure_ascii=False, indent=1))
        return 0

    repos = resolve_repos(data_dir=a.data_dir, roots=a.repos)
    print(json.dumps(refresh_all(repos, force=a.force), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
