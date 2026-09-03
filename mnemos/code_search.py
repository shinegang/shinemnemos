# -*- coding: utf-8 -*-
"""ShineMnemos: поиск по code-слою (keyword, без эмбеддингов).

Перенос PoC (/opt/bench-memory/poc/code_search.py, 215 строк) в состав
Mnemos + мультирепо-реестр с ленивой загрузкой и авто-перечитыванием графа.

Ранжирование (v2 по умолчанию, §2.1 отчёта):
  score(symbol) = Σ_t idf(t)·вес_поля(t) · покрытие,
     поля: name(4.0) > qname(2.0) > signature(1.5) = route(1.5) > file(1.0) > doc(0.8)
  + 6.0 за точное совпадение имени символа с запросом,
  + 3.0 за подстроку запроса в qname,
  × вес класса файла: source 1.0 / vendor 0.5 / test 0.35  (v2)
Файловый слой: те же idf по токенам ТЕЛА файла (вес 0.5) — попадания
в тело кода не теряются. Агрегация в файл — по МАКСИМУМУ символа
(сумма награждает файл с десятком слабых упоминаний) + 0.25·тело.

Регистр не важен. Кириллица в токенах сохраняется.

Честные числа PoC на общем ground truth (§4.3):
  v1 (без единой подгонки): recall@5 kw 0.678 / nl 0.444
  v2 (настроен на тех же промахах): recall@5 kw 0.689 / nl 0.422, MRR kw 0.717
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from .code_index import (Repo, load_graph, read_checked, refresh_repo_graph,
                         resolve_repos, tokenize)

FIELD_WEIGHTS = (("name", 4.0), ("qname", 2.0), ("signature", 1.5),
                 ("route", 1.5), ("file", 1.0), ("doc", 0.8))
FILE_BODY_WEIGHT = 0.5
EXACT_NAME_BONUS = 6.0
QNAME_SUBSTR_BONUS = 3.0

# приоритет "определение > упоминание > тест" явными весами класса файла
TEST_RE = re.compile(r"(^|/)tests?/|(^|/)test_[^/]*$|_test\.py$|conftest\.py$")
VENDOR_RE = re.compile(r"(^|/)(docs|examples?|samples?|vendor|third_party)/")
CLASS_WEIGHT_SOURCE = 1.0
CLASS_WEIGHT_TEST = 0.35
CLASS_WEIGHT_VENDOR = 0.5
BODY_MIX_V2 = 0.25

DEFAULT_RANK = "v2"
MAX_TOP_K = 200


def file_class_weight(path: str) -> float:
    if TEST_RE.search(path):
        return CLASS_WEIGHT_TEST
    if VENDOR_RE.search(path):
        return CLASS_WEIGHT_VENDOR
    return CLASS_WEIGHT_SOURCE


class CodeIndex:
    """Загруженный граф одного репозитория + поиск по нему."""

    def __init__(self, path_or_graph: Any, repo: str = "",
                 path: Optional[str] = None) -> None:
        if isinstance(path_or_graph, dict):
            g = path_or_graph
            self.path = path or ""
        else:
            self.path = str(path_or_graph)
            g = load_graph(self.path)
        self.graph = g  # исходный словарь: переиспользуется при инкременте
        self.meta = g.get("meta", {})
        self.repo = repo or self.meta.get("repo") or ""
        self.nodes = g.get("nodes", [])
        self.edges = g.get("edges", [])
        self.sym_index = g.get("sym_index", {})
        self.files = g.get("files", [])
        self.file_index = g.get("file_index", {})
        self.manifest = g.get("manifest", {})
        self.loaded_mtime = os.path.getmtime(self.path) if self.path and \
            os.path.exists(self.path) else 0.0

        n_sym = max(1, len(self.nodes))
        n_file = max(1, len(self.files))
        self._idf_sym = {t: math.log(1.0 + n_sym / len(p))
                         for t, p in self.sym_index.items() if p}
        self._idf_file = {t: math.log(1.0 + n_file / len(p))
                          for t, p in self.file_index.items() if p}
        self._fields = [
            {f: set(tokenize(n.get(f, ""))) for f, _ in FIELD_WEIGHTS} for n in self.nodes
        ]
        self._file_sets = {t: set(p) for t, p in self.file_index.items()}

    # -- свежесть ----------------------------------------------------------
    @property
    def stale_seconds(self) -> float:
        """Сколько секунд назад индекс в последний раз сверялся с диском.

        Берётся максимум из отметки в графе и отметки <graph>.checked —
        последнюю обновляет вотчер, не переписывая весь граф.
        """
        checked = float(self.meta.get("checked_ts") or self.meta.get("built_ts") or 0.0)
        if self.path:
            side = read_checked(self.path)
            if side and side > checked:
                checked = side
        return round(max(0.0, time.time() - checked), 1)

    @property
    def age_seconds(self) -> float:
        built = self.meta.get("built_ts") or 0.0
        return round(max(0.0, time.time() - float(built)), 1)

    # -- поиск -------------------------------------------------------------
    def search(self, query: str, top_k: int = 5, granularity: str = "symbol",
               rank: str = DEFAULT_RANK, symbol_kind: Optional[str] = None) -> List[Dict[str, Any]]:
        qtokens = list(dict.fromkeys(tokenize(query)))
        if not qtokens:
            return []
        qlow = (query or "").strip().lower()

        cand: set = set()
        for t in qtokens:
            cand.update(self.sym_index.get(t, ()))
        sym_scores: Dict[int, float] = {}
        for i in cand:
            n = self.nodes[i]
            if symbol_kind and n.get("symbol_kind") != symbol_kind:
                continue
            fields = self._fields[i]
            score = 0.0
            covered = 0
            for t in qtokens:
                idf = self._idf_sym.get(t, 0.0)
                best = 0.0
                for fname, w in FIELD_WEIGHTS:
                    if t in fields[fname]:
                        best = max(best, idf * w)
                if best > 0:
                    covered += 1
                    score += best
            if not score:
                continue
            score *= covered / len(qtokens)
            if n["name"].lower() == qlow:
                score += EXACT_NAME_BONUS
            elif qlow and qlow in n["qname"].lower():
                score += QNAME_SUBSTR_BONUS
            sym_scores[i] = score

        # файловый слой (тело кода); при фильтре по типу символа не нужен
        file_scores: Dict[str, float] = {}
        if not symbol_kind:
            fcand: set = set()
            for t in qtokens:
                fcand.update(self.file_index.get(t, ()))
            for fi in fcand:
                score, covered = 0.0, 0
                for t in qtokens:
                    if fi in self._file_sets.get(t, ()):
                        covered += 1
                        score += self._idf_file.get(t, 0.0) * FILE_BODY_WEIGHT
                if score:
                    file_scores[self.files[fi]] = score * (covered / len(qtokens))

        if rank == "v2":
            sym_scores = {i: s * file_class_weight(self.nodes[i]["file"])
                          for i, s in sym_scores.items()}
            file_scores = {f: s * file_class_weight(f) for f, s in file_scores.items()}

        if granularity == "file":
            if rank == "v2":
                best_sym: Dict[str, float] = {}
                for i, s in sym_scores.items():
                    f = self.nodes[i]["file"]
                    best_sym[f] = max(best_sym.get(f, 0.0), s)
                agg = {f: best_sym.get(f, 0.0) + BODY_MIX_V2 * file_scores.get(f, 0.0)
                       for f in set(best_sym) | set(file_scores)}
            else:
                agg = dict(file_scores)
                for i, s in sym_scores.items():
                    f = self.nodes[i]["file"]
                    agg[f] = agg.get(f, 0.0) + s
            rows = sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
            return [{"file": f, "repo": self.repo, "score": round(s, 4)} for f, s in rows]

        out: List[Dict[str, Any]] = []
        for i, s in sorted(sym_scores.items(),
                           key=lambda kv: (-kv[1], self.nodes[kv[0]]["qname"])):
            n = self.nodes[i]
            out.append({
                "qname": n["qname"], "symbol_kind": n["symbol_kind"],
                "file": n["file"], "line": n["line"], "end_line": n["end_line"],
                "signature": n.get("signature", ""), "repo": self.repo,
                "score": round(s, 4),
            })
        sym_files = {r["file"] for r in out}
        for f, s in sorted(file_scores.items(), key=lambda kv: (-kv[1], kv[0])):
            if f not in sym_files:
                out.append({"qname": f, "symbol_kind": "file", "file": f,
                            "line": 1, "end_line": 1, "signature": "",
                            "repo": self.repo, "score": round(s, 4)})
        out.sort(key=lambda r: (-r["score"], r["qname"]))
        return out[:top_k]

    # -- сопутствующие инструменты ----------------------------------------
    def symbols(self, file: Optional[str] = None, symbol_kind: Optional[str] = None,
                limit: int = 100) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for n in self.nodes:
            if file and n["file"] != file:
                continue
            if symbol_kind and n["symbol_kind"] != symbol_kind:
                continue
            out.append({"qname": n["qname"], "symbol_kind": n["symbol_kind"],
                        "file": n["file"], "line": n["line"],
                        "signature": n.get("signature", ""), "repo": self.repo})
            if len(out) >= limit:
                break
        return out

    def callers(self, qname: str, direction: str = "inbound",
                limit: int = 50) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        short = qname.rsplit(".", 1)[-1]
        for e in self.edges:
            if e["t"] not in ("CALLS", "USAGE"):
                continue
            if direction == "inbound":
                # USAGE-рёбра указывают на неразрешённое короткое имя
                if e["d"] != qname and not (e["t"] == "USAGE" and e["d"] == short):
                    continue
            else:
                if e["s"] != qname:
                    continue
            out.append({"from": e["s"], "to": e["d"], "edge": e["t"],
                        "file": e.get("f", ""), "repo": self.repo})
            if len(out) >= limit:
                break
        return out


class CodeRegistry:
    """Реестр репозиториев: ленивая загрузка графов + авто-перечитывание.

    Граф перечитывается, если файл на диске изменился (mtime) — так поиск
    видит результат работы code_watcher без рестарта сервера.
    """

    def __init__(self, repos: Optional[Sequence[Repo]] = None,
                 data_dir: Optional[str] = None, roots: Any = None,
                 store_path: Optional[str] = None) -> None:
        self.repos: List[Repo] = list(repos) if repos is not None else \
            resolve_repos(data_dir=data_dir, roots=roots, store_path=store_path)
        self._loaded: Dict[str, CodeIndex] = {}

    # -- доступ ------------------------------------------------------------
    def repo_names(self) -> List[str]:
        return [r.name for r in self.repos]

    def get_repo(self, name: str) -> Repo:
        for r in self.repos:
            if r.name == name:
                return r
        raise ValueError(
            f"неизвестный репозиторий {name!r}; доступные: {', '.join(self.repo_names()) or '—'}"
        )

    def select(self, repo: Optional[str] = None) -> List[Repo]:
        return [self.get_repo(repo)] if repo else list(self.repos)

    def index(self, repo: Repo) -> Optional[CodeIndex]:
        """Загруженный индекс репозитория или None, если графа ещё нет."""
        path = repo.graph_path
        if not os.path.exists(path):
            self._loaded.pop(repo.name, None)
            return None
        cached = self._loaded.get(repo.name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return cached
        if cached is not None and cached.loaded_mtime == mtime:
            return cached
        try:
            idx = CodeIndex(path, repo=repo.name)
        except (OSError, json.JSONDecodeError, ValueError):
            return cached
        self._loaded[repo.name] = idx
        return idx

    def indexes(self, repo: Optional[str] = None) -> List[CodeIndex]:
        out = []
        for r in self.select(repo):
            idx = self.index(r)
            if idx is not None:
                out.append(idx)
        return out

    def stale_seconds(self, repo: Optional[str] = None) -> Optional[float]:
        """Максимальная (худшая) несвежесть по выбранным репозиториям."""
        vals = [i.stale_seconds for i in self.indexes(repo)]
        return max(vals) if vals else None

    # -- инструменты -------------------------------------------------------
    def search(self, query: str, top_k: int = 5, repo: Optional[str] = None,
               granularity: str = "symbol", symbol_kind: Optional[str] = None,
               rank: str = DEFAULT_RANK) -> List[Dict[str, Any]]:
        top_k = max(1, min(int(top_k), MAX_TOP_K))
        rows: List[Dict[str, Any]] = []
        for idx in self.indexes(repo):
            rows.extend(idx.search(query, top_k=top_k, granularity=granularity,
                                   rank=rank, symbol_kind=symbol_kind))
        key = "file" if granularity == "file" else "qname"
        rows.sort(key=lambda r: (-r["score"], r.get("repo", ""), r.get(key, "")))
        return rows[:top_k]

    def symbols(self, repo: Optional[str] = None, file: Optional[str] = None,
                symbol_kind: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 5000))
        out: List[Dict[str, Any]] = []
        for idx in self.indexes(repo):
            out.extend(idx.symbols(file=file, symbol_kind=symbol_kind,
                                   limit=limit - len(out)))
            if len(out) >= limit:
                break
        return out[:limit]

    def callers(self, qname: str, direction: str = "inbound",
                repo: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 5000))
        out: List[Dict[str, Any]] = []
        for idx in self.indexes(repo):
            out.extend(idx.callers(qname, direction=direction, limit=limit - len(out)))
            if len(out) >= limit:
                break
        return out[:limit]

    def refresh(self, repo: Optional[str] = None, force: bool = False) -> List[Dict[str, Any]]:
        """Обновить индексы. Уже загруженный граф переиспользуется (без чтения с диска).

        Ошибка одного репозитория не роняет остальные — она возвращается
        в его строке как "error".
        """
        out: List[Dict[str, Any]] = []
        for r in self.select(repo):
            cached = self._loaded.get(r.name)
            in_memory = None
            if cached is not None and not force:
                try:
                    fresh_on_disk = os.path.getmtime(r.graph_path) == cached.loaded_mtime
                except OSError:
                    fresh_on_disk = False
                if fresh_on_disk:
                    in_memory = cached.graph
            try:
                stats, graph = refresh_repo_graph(r, force=force, graph=in_memory)
            except Exception as exc:
                self._loaded.pop(r.name, None)
                out.append({"repo": r.name, "root": r.root, "error": repr(exc)})
                continue
            out.append(stats)
            if in_memory is not None and stats["mode"] == "unchanged":
                continue  # тот же объект уже в кэше, пересобирать нечего
            try:
                idx = CodeIndex(graph, repo=r.name, path=r.graph_path)
                idx.loaded_mtime = os.path.getmtime(r.graph_path)
                self._loaded[r.name] = idx
            except OSError:
                self._loaded.pop(r.name, None)
        return out

    def status(self, repo: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for r in self.select(repo):
            idx = self.index(r)
            if idx is None:
                out.append({"repo": r.name, "root": r.root, "indexed": False})
                continue
            out.append({
                "repo": r.name, "root": r.root, "indexed": True,
                "nodes": len(idx.nodes), "edges": len(idx.edges),
                "files": len(idx.files),
                "built_utc": idx.meta.get("built_utc", ""),
                "stale_seconds": idx.stale_seconds,
                "index_bytes": os.path.getsize(r.graph_path)
                if os.path.exists(r.graph_path) else 0,
            })
        return out


def _main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description="Поиск по code-слою Mnemos")
    ap.add_argument("--graph", help="один файл графа")
    ap.add_argument("--data-dir", help="каталог графов (мультирепо)")
    ap.add_argument("--repos", help="'имя=/путь,/путь'")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--granularity", default="symbol", choices=["symbol", "file"])
    ap.add_argument("--rank", default=DEFAULT_RANK, choices=["v1", "v2"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.graph:
        rows = CodeIndex(a.graph).search(a.query, a.top_k, a.granularity, rank=a.rank)
    else:
        reg = CodeRegistry(data_dir=a.data_dir, roots=a.repos)
        rows = reg.search(a.query, a.top_k, granularity=a.granularity, rank=a.rank)
    if a.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
    else:
        for r in rows:
            if a.granularity == "file":
                print(f"{r['score']:8.3f}  {r.get('repo','')}: {r['file']}")
            else:
                print(f"{r['score']:8.3f}  {r['symbol_kind']:8s} {r['qname']}  "
                      f"({r.get('repo','')}: {r['file']}:{r['line']})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
