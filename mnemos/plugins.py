# -*- coding: utf-8 -*-
"""ShineMnemos: механизм плагинов — подключаемые/отключаемые модули.

Решение владельца (23.02): память (ядро memory_*) остаётся внутренней,
а контекст-модуль (context_engine) — ОТДЕЛЬНЫЙ плагин, который позже можно
продавать отдельно (x402). Этот модуль — фундамент для этого: список
включённых плагинов задаётся конфигом, выключенный плагин не экспортирует
инструменты в MCP и не используется сервером.

Плагины:
  context_engine  — инструменты контекста: context_compact (сжатие сессии
                    с выгрузкой в память), context_prefix (кэш-стабильный
                    префикс DeepSeek on-disk caching), context_defragment
                    (дефрагментация сессии). ВКЛЮЧЁН по умолчанию.
  gates           — слой гейтов качества Г1-Г5 (mnemos.gates) перед
                    выгрузкой сводок: выключен -> сводки пишутся без
                    фильтрации. Собственных инструментов нет (capability).
  code            — code-слой: индексация репозиториев (tree-sitter) и
                    инструменты code_search / code_symbols / code_callers /
                    code_refresh. ВЫКЛЮЧЕН по умолчанию: включается только
                    явным MNEMOS_PLUGINS="context_engine,gates,code".
                    Стор решений не трогает — граф кода лежит отдельными
                    файлами data/code/<repo>.code_graph.json.

Конфигурация включённых плагинов (приоритет сверху вниз):
  1. аргумент plugins API/встраивания ([] — явно всё выключено);
  2. env MNEMOS_PLUGINS — список через запятую: "context_engine,gates";
     пустая строка или токен "none" — плагинов нет (явный off);
  3. файл plugins.json — {"enabled": ["context_engine"]} рядом со store
     или в рабочем каталоге (--plugins-config задаёт явный путь);
  4. дефолты: context_engine, gates (оба включены).

Неизвестное имя плагина — ошибка конфигурации (ValueError), а не тихий
пропуск: опечатка в конфиге должна быть видна сразу.

Только stdlib. context_engine импортируется лениво, внутри обработчиков —
выключенный плагин не тянет код модуля в рантайм сервера.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# --- имя конфигурации --------------------------------------------------------

ENV_NAME = "MNEMOS_PLUGINS"
CONFIG_FILENAME = "plugins.json"

# Дефолты: контекст-модуль нужен (решение владельца), гейты качества — тоже
# (пример конфигурации: MNEMOS_PLUGINS="context_engine,gates").
DEFAULT_ENABLED: List[str] = ["context_engine", "gates"]


# ============================================================================
# Базовый класс плагина
# ============================================================================


class Plugin:
    """Базовый класс плагина.

    Подкласс задаёт:
      name         — уникальное имя (ключ конфигурации);
      description  — человекочитаемое описание;
      tools        — MCP-схемы инструментов (name/description/inputSchema);
      default_enabled — флаг по умолчанию (сейчас все — True).

    Обработчики — методы экземпляра с именем = имени инструмента и
    сигнатурой handler(core, args) -> dict. Для capability-плагинов
    (без инструментов) обработчиков нет.
    """

    name: str = ""
    description: str = ""
    tools: Tuple[Dict[str, Any], ...] = ()
    default_enabled: bool = True


# ============================================================================
# Плагин context_engine — инструменты контекста (MCP)
# ============================================================================

TOOL_CONTEXT_COMPACT: Dict[str, Any] = {
    "name": "context_compact",
    "description": (
        "Сжать старую часть сессии с выгрузкой в память (плагин context_engine): "
        "свежие сообщения остаются в окне (RAM), старое уходит в иерархию "
        "сводок kind='context_summary' в Store (Mnemos). Возвращает окно, "
        "ссылки на память и статистику экономии."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                                "claim": {"type": "string"},
                                "ts": {"type": "string"},
                            },
                        },
                    ]
                },
                "description": "Сообщения сессии (строки или dict'ы)",
            },
            "keep_recent": {
                "type": "integer",
                "description": "Сколько свежих сообщений оставить в окне (по умолчанию 8)",
            },
            "chunk_size": {
                "type": "integer",
                "description": "Размер чанка для детальных сводок (по умолчанию 3)",
            },
            "hierarchy_cap": {
                "type": "integer",
                "description": "Порог построения уровня 1+ сводок (по умолчанию 6)",
            },
        },
        "required": ["messages"],
    },
}

TOOL_CONTEXT_PREFIX: Dict[str, Any] = {
    "name": "context_prefix",
    "description": (
        "Собрать кэш-стабильный канонический префикс для DeepSeek on-disk "
        "caching (плагин context_engine): НЕИЗМЕННАЯ шапка (system + стабильные "
        "инструкции) + APPEND-ONLY хвост. tail_blocks добавляются в конец хвоста "
        "(единственная разрешённая мутация) — повторная сборка байт-в-байт "
        "идентична, кэш жив. Возвращает hash/text/tokens + проверку стабильности."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "system": {"type": "string", "description": "Системный промпт (первый блок шапки)"},
            "static_blocks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Стабильные инструкции шапки",
            },
            "tail_blocks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Блоки, добавляемые в append-only хвост перед сборкой",
            },
        },
    },
}

TOOL_CONTEXT_DEFRAGMENT: Dict[str, Any] = {
    "name": "context_defragment",
    "description": (
        "Дефрагментировать сессию (плагин context_engine): старую сессию сжать "
        "в память (сводки в Store), вернуть новый стартовый контекст "
        "[сводка-преамбула + окно]. Возвращает new_context, экономию токенов "
        "и ссылки на память."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "session": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                                "claim": {"type": "string"},
                                "ts": {"type": "string"},
                            },
                        },
                    ]
                },
                "description": "Сообщения старой сессии",
            },
            "keep_recent": {
                "type": "integer",
                "description": "Сколько свежих сообщений оставить в окне (по умолчанию 8)",
            },
            "system": {"type": "string", "description": "Системный промпт новой сессии"},
            "static_blocks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Стабильные инструкции новой сессии",
            },
            "preamble_limit": {
                "type": "integer",
                "description": "Лимит символов сводки-преамбулы (по умолчанию 700)",
            },
            "chunk_size": {
                "type": "integer",
                "description": "Размер чанка для сводок (по умолчанию 3)",
            },
            "hierarchy_cap": {
                "type": "integer",
                "description": "Порог уровня 1+ сводок (по умолчанию 6)",
            },
        },
        "required": ["session"],
    },
}


def _int_arg(args: Dict[str, Any], key: str, default: int) -> int:
    v = args.get(key, default)
    try:
        return int(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"context_engine: параметр {key} должен быть целым числом") from exc


def _str_list(v: Any, name: str) -> List[str]:
    """Принимает список строк (или одиночную строку) — иначе ошибка."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return list(v)
    raise ValueError(f"context_engine: параметр {name} должен быть списком строк")


class ContextEnginePlugin(Plugin):
    """Плагин context_engine: инструменты контекста поверх Store.

    Экземпляр создаётся на каждый MnemosCore — состояние префикса
    (append-only хвост) не течёт между серверами.
    """

    name = "context_engine"
    description = (
        "Контекст-модуль: сжатие сессий с выгрузкой в память, "
        "кэш-стабильный префикс DeepSeek, дефрагментация."
    )
    tools = (TOOL_CONTEXT_COMPACT, TOOL_CONTEXT_PREFIX, TOOL_CONTEXT_DEFRAGMENT)

    def _compactor(self, core: Any, args: Dict[str, Any]) -> Any:
        # ленивый импорт: выключенный плагин не грузит context_engine
        from .context_engine import HierarchicalCompactor

        return HierarchicalCompactor(
            store=core.store,
            chunk_size=_int_arg(args, "chunk_size", 3),
            hierarchy_cap=_int_arg(args, "hierarchy_cap", 6),
            gates_enabled=core.gates_enabled,
        )

    def context_compact(self, core: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        from .context_engine import estimate_tokens

        messages = args.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                "context_compact: параметр messages (непустой список) обязателен"
            )
        keep_recent = _int_arg(args, "keep_recent", 8)
        compactor = self._compactor(core, args)
        window, refs = compactor.compact(messages, keep_recent=keep_recent)
        summaries = [r for r in refs if r.get("type") == "summary"]
        raws = [r for r in refs if r.get("type") == "raw"]
        return {
            "window": window,
            "memory_refs": refs,
            "messages_in": len(messages),
            "window_count": len(window),
            "summary_count": len(summaries),
            "raw_count": len(raws),
            "window_tokens": sum(estimate_tokens(m["content"]) for m in window),
            "gates": "on" if core.gates_enabled else "off",
        }

    def context_prefix(self, core: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        from .context_engine import CanonicalPrefix

        prefix = getattr(core, "_ctx_prefix", None)
        if prefix is None:
            prefix = CanonicalPrefix()
            core._ctx_prefix = prefix  # хвост живёт между вызовами
        system = str(args.get("system") or "")
        static_blocks = _str_list(args.get("static_blocks"), "static_blocks")
        for block in _str_list(args.get("tail_blocks"), "tail_blocks"):
            prefix.append_tail(block)

        # стабильность — относительно ПРЕДЫДУЩЕЙ сборки (build_prefix перезаписывает
        # состояние шапки, поэтому сравниваем со снапшотом прошлого вызова)
        last = getattr(core, "_ctx_last", None)  # {"system", "static_blocks", "tail"}
        out = prefix.build_prefix(system, static_blocks)

        if last is None:
            head_stable, tail_append_only, reasons = True, True, []
        else:
            head_stable = system == last["system"] and static_blocks == last["static_blocks"]
            prev_tail = last["tail"]
            tail_append_only = list(prefix._tail)[: len(prev_tail)] == prev_tail
            reasons = []
            if not head_stable:
                reasons.append(
                    "ШАПКА ИЗМЕНЕНА: system/static_blocks другие — префиксный кэш сломан"
                )
            if not tail_append_only:
                reasons.append("ХВОСТ МУТИРОВАН: разрешено только добавление в конец")
        core._ctx_last = {
            "system": system,
            "static_blocks": static_blocks,
            "tail": list(prefix._tail),
        }
        out["stability"] = {
            "stable": head_stable and tail_append_only,
            "head_stable": head_stable,
            "tail_append_only": tail_append_only,
            "reasons": reasons,
        }
        return out

    def context_defragment(self, core: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        from .context_engine import ContextDefragmenter

        session = args.get("session")
        if not isinstance(session, list) or not session:
            raise ValueError(
                "context_defragment: параметр session (непустой список) обязателен"
            )
        keep_recent = _int_arg(args, "keep_recent", 8)
        preamble_limit = _int_arg(args, "preamble_limit", 700)
        system = str(args.get("system") or "")
        static_blocks = _str_list(args.get("static_blocks"), "static_blocks")
        defragmenter = ContextDefragmenter(compactor=self._compactor(core, args))
        return defragmenter.defragment(
            session,
            store=core.store,
            keep_recent=keep_recent,
            system=system,
            static_blocks=static_blocks,
            preamble_limit=preamble_limit,
        )


# ============================================================================
# Плагин gates — слой качества Г1-Г5 (capability, без инструментов)
# ============================================================================


class GatesPlugin(Plugin):
    """Плагин gates: гейты качества Г1-Г5 перед выгрузкой сводок.

    Собственных MCP-инструментов нет. Включён — сводки context_engine
    проходят run_write_gates (reject не пишется, flag помечается);
    выключен — сводки пишутся без фильтрации (core.gates_enabled=False).
    """

    name = "gates"
    description = "Слой гейтов качества Г1-Г5 при выгрузке сводок в память."
    tools = ()


# ============================================================================
# Плагин code — code-слой (индексация репозиториев + 4 MCP-инструмента)
# ============================================================================

TOOL_CODE_SEARCH: Dict[str, Any] = {
    "name": "code_search",
    "description": (
        "Найти символы кода (плагин code): keyword-поиск по индексу "
        "репозиториев — имена/qname/сигнатуры/докстринги/маршруты + тела "
        "файлов. Регистр не важен, кириллица учитывается. Возвращает "
        "qname/file/line/signature/score без тел кода."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Запрос (ключевые слова или фраза)"},
            "top_k": {"type": "integer", "description": "Сколько результатов (по умолчанию 5)"},
            "repo": {"type": "string", "description": "Имя репозитория (по умолчанию — все)"},
            "granularity": {
                "type": "string",
                "enum": ["symbol", "file"],
                "description": "Гранулярность результата (по умолчанию symbol)",
            },
            "symbol_kind": {
                "type": "string",
                "description": "Фильтр: module|class|function|method|route",
            },
        },
        "required": ["query"],
    },
}

TOOL_CODE_SYMBOLS: Dict[str, Any] = {
    "name": "code_symbols",
    "description": (
        "Перечислить символы кода (плагин code): все символы репозитория "
        "или конкретного файла, с фильтром по типу (module/class/function/"
        "method/route)."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Имя репозитория (по умолчанию — все)"},
            "file": {"type": "string", "description": "Путь файла относительно корня репозитория"},
            "symbol_kind": {
                "type": "string",
                "description": "Фильтр: module|class|function|method|route",
            },
            "limit": {"type": "integer", "description": "Максимум записей (по умолчанию 100)"},
        },
    },
}

TOOL_CODE_CALLERS: Dict[str, Any] = {
    "name": "code_callers",
    "description": (
        "Кто вызывает символ / кого вызывает символ (плагин code): рёбра "
        "CALLS (резолвнутые) и USAGE (неоднозначное имя) по qname."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "qname": {"type": "string", "description": "Полное имя символа (module.path.name)"},
            "direction": {
                "type": "string",
                "enum": ["inbound", "outbound"],
                "description": "inbound — кто вызывает (по умолчанию); outbound — кого вызывает",
            },
            "repo": {"type": "string", "description": "Имя репозитория (по умолчанию — все)"},
            "limit": {"type": "integer", "description": "Максимум рёбер (по умолчанию 50)"},
        },
        "required": ["qname"],
    },
}

TOOL_CODE_REFRESH: Dict[str, Any] = {
    "name": "code_refresh",
    "description": (
        "Обновить индекс кода (плагин code): скан mtime+size, при "
        "расхождении sha1, перепарс только изменённых файлов. force=true — "
        "полная переиндексация. Возвращает по каждому репозиторию число "
        "файлов, изменений, время, узлы/рёбра, размер индекса и stale_seconds."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Имя репозитория (по умолчанию — все)"},
            "force": {"type": "boolean", "description": "Полная переиндексация (по умолчанию false)"},
        },
    },
}


def _code_str(args: Dict[str, Any], key: str) -> Optional[str]:
    v = args.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValueError(f"code: параметр {key} должен быть строкой")
    v = v.strip()
    return v or None


def _code_int(args: Dict[str, Any], key: str, default: int) -> int:
    v = args.get(key, default)
    try:
        return int(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"code: параметр {key} должен быть целым числом") from exc


class CodePlugin(Plugin):
    """Плагин code: индекс кода репозиториев + 4 MCP-инструмента.

    ВЫКЛЮЧЕН по умолчанию (нет в DEFAULT_ENABLED): включается явным
    MNEMOS_PLUGINS="context_engine,gates,code". Откат — убрать "code".

    Стор решений (data/nodes.json) не читается и не пишется: граф кода
    живёт в отдельных файлах data/code/<repo>.code_graph.json (§2.2 отчёта
    ФЭЙБЛ-ИССЛЕДОВАНИЕ-ИНДЕКСАЦИЯ-01.09: recall@5 решений 0.944 сохраняется).

    mnemos.code_index / code_search / code_watcher импортируются ЛЕНИВО —
    выключенный плагин не тянет tree-sitter в рантайм сервера.
    """

    name = "code"
    description = (
        "Code-слой: индексация репозиториев (tree-sitter) и поиск по символам, "
        "файлам и графу вызовов. Отдельный индекс, стор решений не затрагивается."
    )
    tools = (TOOL_CODE_SEARCH, TOOL_CODE_SYMBOLS, TOOL_CODE_CALLERS, TOOL_CODE_REFRESH)
    default_enabled = False

    def __init__(self) -> None:
        self._registry: Any = None
        self._watcher: Any = None

    # -- ленивая инициализация --------------------------------------------
    def registry(self, core: Any) -> Any:
        """Реестр репозиториев (создаётся при первом обращении к инструменту)."""
        if self._registry is None:
            from .code_search import CodeRegistry

            store_path = getattr(getattr(core, "store", None), "path", None)
            self._registry = CodeRegistry(store_path=str(store_path) if store_path else None)
        return self._registry

    def watcher(self, core: Any) -> Any:
        """Фоновый обходчик (если не выключен env MNEMOS_CODE_WATCH=0)."""
        from .code_watcher import CodeWatcher, watch_enabled

        if not watch_enabled():
            return None
        if self._watcher is None:
            self._watcher = CodeWatcher(self.registry(core), initial_delay=1.0).start()
        return self._watcher

    def _envelope(self, core: Any, repo: Optional[str]) -> Dict[str, Any]:
        reg = self.registry(core)
        self.watcher(core)
        return {"repos": [r.name for r in reg.select(repo)],
                "stale_seconds": reg.stale_seconds(repo)}

    # -- инструменты -------------------------------------------------------
    def code_search(self, core: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("code_search: параметр query (непустая строка) обязателен")
        granularity = _code_str(args, "granularity") or "symbol"
        if granularity not in ("symbol", "file"):
            raise ValueError("code_search: granularity должен быть 'symbol' или 'file'")
        repo = _code_str(args, "repo")
        reg = self.registry(core)
        env = self._envelope(core, repo)
        results = reg.search(
            query, top_k=_code_int(args, "top_k", 5), repo=repo,
            granularity=granularity, symbol_kind=_code_str(args, "symbol_kind"),
        )
        return {"query": query, "granularity": granularity, "count": len(results),
                "results": results, **env}

    def code_symbols(self, core: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        repo = _code_str(args, "repo")
        reg = self.registry(core)
        env = self._envelope(core, repo)
        results = reg.symbols(repo=repo, file=_code_str(args, "file"),
                              symbol_kind=_code_str(args, "symbol_kind"),
                              limit=_code_int(args, "limit", 100))
        return {"count": len(results), "results": results, **env}

    def code_callers(self, core: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        qname = args.get("qname")
        if not isinstance(qname, str) or not qname.strip():
            raise ValueError("code_callers: параметр qname (непустая строка) обязателен")
        direction = _code_str(args, "direction") or "inbound"
        if direction not in ("inbound", "outbound"):
            raise ValueError("code_callers: direction должен быть 'inbound' или 'outbound'")
        repo = _code_str(args, "repo")
        reg = self.registry(core)
        env = self._envelope(core, repo)
        results = reg.callers(qname.strip(), direction=direction, repo=repo,
                              limit=_code_int(args, "limit", 50))
        return {"qname": qname.strip(), "direction": direction,
                "count": len(results), "results": results, **env}

    def code_refresh(self, core: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        repo = _code_str(args, "repo")
        force = bool(args.get("force", False))
        reg = self.registry(core)
        stats = reg.refresh(repo=repo, force=force)
        watcher = self.watcher(core)
        for row in stats:
            if "stale_seconds" not in row:
                row["stale_seconds"] = None
        return {"repos": stats, "count": len(stats),
                "watcher": watcher.status() if watcher is not None else None}


# --- реестр ----------------------------------------------------------------

PLUGIN_CLASSES: Tuple[type, ...] = (ContextEnginePlugin, GatesPlugin, CodePlugin)


def known_plugin_names() -> List[str]:
    return sorted(p.name for p in PLUGIN_CLASSES)


# ============================================================================
# Конфигурация
# ============================================================================


def _normalize_names(spec: Any) -> List[str]:
    """Список имён из строки через запятую или iterable (с dedupe, порядок хранится).

    Специальный токен "none" (без учёта регистра) — явно ноль плагинов:
    удобный off-переключатель для env/CLI на Windows, где пустая строка
    не доходит до дочернего процесса (py-лаунчер/PowerShell).
    """
    if isinstance(spec, str):
        if spec.strip().lower() == "none":
            return []
        items = [p.strip() for p in spec.split(",")]
    else:
        items = [str(p).strip() for p in spec]
    seen: set = set()
    out: List[str] = []
    for name in items:
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _validate(names: Sequence[str]) -> None:
    unknown = [n for n in names if n not in {p.name for p in PLUGIN_CLASSES}]
    if unknown:
        raise ValueError(
            "неизвестный(е) плагин(ы): "
            + ", ".join(repr(n) for n in unknown)
            + f"; доступные: {', '.join(known_plugin_names())}"
        )


def _load_config_file(path: Path) -> List[str]:
    # utf-8-sig: терпим UTF-8 BOM (PowerShell 5.1 по умолчанию пишет с BOM)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"plugins.json {path}: не удалось прочитать/распарсить: {exc!r}") from exc
    if not isinstance(data, dict) or "enabled" not in data:
        raise ValueError(f"plugins.json {path}: обязателен ключ \"enabled\" (список имён)")
    enabled = data["enabled"]
    if not isinstance(enabled, list) or not all(isinstance(x, str) for x in enabled):
        raise ValueError(f"plugins.json {path}: \"enabled\" должен быть списком строк")
    return enabled


def _find_config(
    config_path: Optional[str], search_dirs: Optional[Sequence[str]]
) -> Optional[List[str]]:
    if config_path:
        return _load_config_file(Path(config_path))
    dirs: List[Path] = [Path(d) for d in (search_dirs or [])] + [Path.cwd()]
    seen: set = set()
    for d in dirs:
        key = os.path.normcase(str(d))
        if key in seen:
            continue
        seen.add(key)
        candidate = d / CONFIG_FILENAME
        if candidate.is_file():
            return _load_config_file(candidate)
    return None


def resolve_enabled_plugins(
    plugins: Any = None,
    env: Optional[str] = None,
    config_path: Optional[str] = None,
    search_dirs: Optional[Sequence[str]] = None,
) -> List[str]:
    """Определяет включённые плагины по цепочке: arg -> env -> plugins.json -> дефолты.

    plugins      — явный список/строка ([] — всё выключено; None — дальше по цепочке);
    env          — значение MNEMOS_PLUGINS (None — читать os.environ);
                   заданная пустая строка — плагинов НЕТ (явный off);
    config_path  — явный путь к plugins.json (иначе поиск в search_dirs и cwd);
    search_dirs  — каталоги для поиска plugins.json (например, каталог store).

    Возвращает список имён включённых плагинов в порядке конфигурации.
    Неизвестное имя -> ValueError.
    """
    if plugins is not None:
        names = _normalize_names(plugins)
    else:
        raw = env if env is not None else os.environ.get(ENV_NAME)
        if raw is None:
            cfg = _find_config(config_path, search_dirs)
            names = _normalize_names(cfg) if cfg is not None else list(DEFAULT_ENABLED)
        else:
            names = _normalize_names(raw)  # пусто -> [] (плагины выключены)
    _validate(names)
    return names


# ============================================================================
# Менеджер: активные плагины одного ядра
# ============================================================================


class PluginManager:
    """Активные плагины одного MnemosCore (свежие экземпляры — без общего состояния)."""

    def __init__(self, enabled_names: Sequence[str]) -> None:
        self._enabled = list(enabled_names)
        classes = {p.name: p for p in PLUGIN_CLASSES}
        self.active = [classes[name]() for name in self._enabled if name in classes]

    @property
    def enabled(self) -> List[str]:
        return list(self._enabled)

    def tool_schemas(self) -> List[Dict[str, Any]]:
        return [tool for plugin in self.active for tool in plugin.tools]

    def handlers(self, core: Any) -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
        """Имя инструмента -> callable(args). Обработчики видят core (store/флаги)."""
        out: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        for plugin in self.active:
            for tool in plugin.tools:
                fn = getattr(plugin, tool["name"], None)
                if fn is not None:
                    # fn — связанный метод (self уже привязан): зовём fn(core, args)
                    out[tool["name"]] = (lambda f: (lambda args: f(core, args)))(fn)
        return out
