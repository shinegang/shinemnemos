# -*- coding: utf-8 -*-
"""ShineMnemos: MCP-сервер-скелет (JSON-RPC 2.0 поверх stdlib http.server).

fastmcp/mcp недоступны в системном python — интерфейс MCP реализован вручную
по спецификации Model Context Protocol (JSON-RPC 2.0 over HTTP):

  POST /  с телом {"jsonrpc":"2.0","id":1,"method":"initialize",...}
  методы: initialize, notifications/initialized, ping, tools/list, tools/call

Ядро (всегда):
  memory_add(claim, source, evidence, context, kind, links) -> узел
  memory_verify(node_id) -> вердикт П1-П6 (verdict, score, notes);
      проход «pass» ПОДКРЕПЛЯЕТ узел (вес растёт — проверенная память крепче)
  memory_search(query, top_k) -> топ-k узлов по подстроке
  memory_rewrite(node_id, new_claim, source, reason) -> переписывание узла
      новым фактом (старое утверждение остаётся в revisions — пластичность)
  memory_reinforce(node_id, delta) -> подкрепление веса вручную
  memory_link(parent_id, claim, ...) -> новый узел-ребёнок внутри родителя
      («граф в узле», структурная рекурсия; глубина ограничена MAX_DEPTH)

Плагины (контекст-модуль — отдельный подключаемый/отключаемый плагин,
решение владельца; см. mnemos/plugins.py):
  context_engine (ВКЛЮЧЁН по умолчанию) — context_compact, context_prefix,
      context_defragment; выключен — инструментов нет в tools/list и tools/call.
  gates — слой гейтов Г1-Г5 при выгрузке сводок (capability, без инструментов).

Конфигурация: env MNEMOS_PLUGINS="context_engine,gates" | plugins.json
{"enabled": [...]} | аргумент --plugins | параметр plugins при встраивании.

Запуск:  py -3.12 -m mnemos --port 8765 --store nodes.json
"""

from __future__ import annotations

import json
import os
import signal
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import re

from . import grounding
from .budget import ensure_router_tag
from .gates import run_read_gates, run_write_gates
from .model import DEFAULT_HALF_LIFE_HOURS, MemoryNode, make_node
from .plugins import PluginManager, resolve_enabled_plugins
from .store import Store, blank_target, resolve_store_path
from .truth_gate import check_and_update

# Г4 сверяет новый claim со всем стором (линейно): 43 мс на 3121 узле
# (замер 01.09). Потолок скана — чтобы запись не деградировала при росте
# графа; сверяются самые свежие узлы, факт усечения виден в ответе.
GATE_MAX_SCAN = 5000

_RE_NODE_ID = re.compile(r"\bmn_[0-9a-f]{6,}\b")


class GateRejected(ValueError):
    """Запись отклонена гейтами Г1-Г5 (не баг сервера, а решение памяти)."""


def _first_node_id(text: str) -> Optional[str]:
    """id узла, на который сослался гейт (для подкрепления оригинала)."""
    m = _RE_NODE_ID.search(str(text or ""))
    return m.group(0) if m else None


def _as_float(value: Any) -> Optional[float]:
    """float или None — без падения на мусорном вводе от клиента."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    """int или None — как _as_float, для целочисленных параметров."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

PROTOCOL_VERSION = "2025-03-26"
SERVER_NAME = "shinemnemos"
# Ф1 02.09: бюджет system-prompt из памяти по умолчанию (план Ильи: <= 1200 токенов)
PROMPT_DEFAULT_TOKENS = 1200
SERVER_VERSION = "0.4.0"  # 03.09: обязательный проход через граф + бланковый стор

# Grounded по умолчанию (приказ Ильи 03.09). Проход через граф — не опция, а
# режим работы: сервер стартует с ним ВКЛЮЧЁННЫМ у всех пользователей, и без
# пред-прохода (memory_ground_prepare) ответ агента помечается ungrounded.
# Выключается осознанно — env MNEMOS_GROUND_BY_DEFAULT=0 или
# `python -m mnemos --no-ground-by-default`.
GROUND_BY_DEFAULT = True
GROUND_ENV = "MNEMOS_GROUND_BY_DEFAULT"
# Потолок длины строк-идентификаторов протокола (session_id, agent): они едут
# в журнал как есть, поэтому ограничиваются на входе.
MAX_ID_LEN = 200

_TRUE_WORDS = ("1", "true", "yes", "on", "да")
_FALSE_WORDS = ("0", "false", "no", "off", "нет")


def _env_flag(name: str, default: bool, env: Optional[Dict[str, str]] = None) -> bool:
    """Булев флаг из окружения. Мусор — ошибка старта, а не тихий дефолт.

    Опечатка в MNEMOS_GROUND_BY_DEFAULT=flase не должна молча оставлять
    политику в состоянии, которого администратор не выбирал: он узнает об
    этом не из логов, а из ответов агента, которые «почему-то ungrounded».
    """
    env = os.environ if env is None else env
    raw = env.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value == "":
        # Пустое значение — это «не задано», а НЕ «выключить». Юнит со строкой
        # `Environment=MNEMOS_GROUND_BY_DEFAULT=` не должен молча снимать
        # обязательный проход через граф: выключение делается словом.
        return default
    if value in _TRUE_WORDS:
        return True
    if value in _FALSE_WORDS:
        return False
    raise ValueError(
        f"{name}={raw!r}: ожидается одно из {_TRUE_WORDS + _FALSE_WORDS}"
    )


# фикс аудита 26.08: лимит HTTP-тела — защита от DoS (B5): клиент с
# Content-Length: 10GB не должен заставить сервер читать 10 ГБ в память.
MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 МБ

# JSON-RPC коды ошибок
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
NODE_NOT_FOUND = -32002  # свой код: узел не найден

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "memory_add",
        "description": (
            "Добавить узел памяти. Создаёт узел, прогоняет через truth-gate "
            "П1-П6 и возвращает его с вердиктом truth_check."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "Утверждение (обязательно)"},
                "source": {"type": "string", "description": "Источник утверждения"},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Свидетельства для воспроизводимости (П5)",
                },
                "context": {"type": "string", "description": "Контекст (полнота, П6)"},
                "kind": {
                    "type": "string",
                    "enum": ["fact", "hypothesis", "refuted", "outdated", "rule"],
                    "description": (
                        "Тип узла. rule — правило команды: не тускнеет ниже 0.5 "
                        "и не удаляется уборкой. Хабы (kind=hub) через MCP не "
                        "создаются — только офлайн-кластеризатором"
                    ),
                },
                "author": {
                    "type": "string",
                    "description": (
                        "Кто создаёт узел/рёбра (например alice) — паспорт "
                        "рёбер link_meta"
                    ),
                },
                "links": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ссылки на другие узлы (непротиворечивость, П4)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Метки узла (правило, торговля, инфра, p0 ...)",
                },
                "confidence": {
                    "type": "number",
                    "description": "Уверенность 0..1 (по умолчанию — score truth-gate / 6)",
                },
                "ttl_hours": {
                    "type": "number",
                    "description": "Протухнет через N часов (volatile-факты: баланс, equity, статус)",
                },
                "valid_until": {
                    "type": "string",
                    "description": "Явный ISO-8601 срок годности (приоритетнее ttl_hours)",
                },
            },
            "required": ["claim"],
        },
    },
    {
        "name": "memory_verify",
        "description": "Проверить узел памяти по протоколу П1-П6 → verdict + score + notes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "id узла памяти"}
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "memory_search",
        "description": (
            "Найти узлы памяти, топ-k. mode=substring — по подстроке "
            "(claim/source/context/evidence; у узлов на уровне L1/L2 — по "
            "сжатому тексту уровня + ключам, буст L0/L1); если целая фраза не "
            "найдена — фоллбек по основам слов (mode в ответе token_fallback). "
            "mode=budget — token-budgeting (top_k auto по сложности, граф-"
            "расширение). mode=semantic — по смыслу (fastembed). "
            "mode=rrf — слияние Ф1+BM25 (+плотный, если есть fastembed) по "
            "Reciprocal Rank Fusion; выигрыш даёт на ключевых словах."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос"},
                "top_k": {
                    "type": ["integer", "string"],
                    "description": (
                        "Сколько результатов (1-50) или \"auto\" — по сложности "
                        "вопроса: простой 5 / средний 10 / сложный 20 (02.09)"
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["substring", "semantic", "budget", "rrf"],
                    "description": (
                        "Режим поиска (по умолчанию substring). budget — "
                        "token-budgeting: подстрока + основы слов (NL-вопросы), "
                        "граф-расширение по рёбрам, хабы отдельно, токенный бюджет. "
                        "rrf — слияние сигналов (ML-BOOST 03.09): для ключевых слов, "
                        "для NL-вопроса хуже budget"
                    ),
                },
                "budget": {"type": "boolean", "description": "То же, что mode=budget"},
                "token_budget": {"type": "integer", "description": "Бюджет токенов выдачи (budget)"},
                "expand": {"type": "boolean", "description": "Граф-расширение в budget (по умолчанию true)"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Оставить только узлы со всеми этими тегами",
                },
                "kind": {"type": "string", "description": "Оставить только узлы этого вида (например rule)"},
                "relevance_gate": {
                    "type": "boolean",
                    "description": "Прогнать выдачу через Г2+Г5 (свежесть и релевантность), по умолчанию false",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_rewrite",
        "description": (
            "Переписать узел новым фактом (пластичность): claim заменяется, "
            "старое утверждение остаётся в revisions. Узел снова проходит "
            "truth-gate П1-П6."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "id узла памяти"},
                "new_claim": {"type": "string", "description": "Новое утверждение"},
                "source": {"type": "string", "description": "Источник нового факта"},
                "reason": {"type": "string", "description": "Почему переписываем"},
            },
            "required": ["node_id", "new_claim"],
        },
    },
    {
        "name": "memory_reinforce",
        "description": (
            "Подкрепить узел: вес растёт (до 1.0), last_used обновляется. "
            "Так память запоминает, что узел пригодился."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "id узла памяти"},
                "delta": {"type": "number", "description": "Прибавка веса (по умолчанию 0.05)"},
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "memory_link",
        "description": (
            "Создать новый узел-ребёнок внутри родителя («граф в узле»). "
            "Глубина ограничена (MAX_DEPTH), циклы запрещены."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "parent_id": {"type": "string", "description": "id родительского узла"},
                "claim": {"type": "string", "description": "Утверждение дочернего узла"},
                "source": {"type": "string", "description": "Источник"},
                "context": {"type": "string", "description": "Контекст (П6)"},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Свидетельства (П5)",
                },
            },
            "required": ["parent_id", "claim"],
        },
    },
    # -- инструменты уборки и связывания (01.09, аудит §5.6) ------------------
    {
        "name": "memory_link_existing",
        "description": (
            "Связать два УЖЕ СУЩЕСТВУЮЩИХ узла ребром from_id -> to_id. "
            "memory_link умеет только создавать новый узел-ребёнка; связать "
            "два живых узла до 01.09 было нечем — отсюда 2 ребра на 5251 узел."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "from_id": {"type": "string", "description": "id узла-источника"},
                "to_id": {"type": "string", "description": "id узла-цели"},
                "bidirectional": {
                    "type": "boolean",
                    "description": "Связать в обе стороны (по умолчанию false)",
                },
                "author": {
                    "type": "string",
                    "description": (
                        "Кто провёл связь (например alice). Пишется в паспорт "
                        "ребра link_meta; без него автор — unknown"
                    ),
                },
                "rel": {
                    "type": "string",
                    "enum": ["related_to", "part_of", "has_part", "conflicts_with",
                             "supersedes", "duplicate_of", "refers_to"],
                    "description": (
                        "Тип связи (02.09, письмо Qwen): related_to по умолчанию; "
                        "part_of/has_part — член/хаб; conflicts_with — противоречие; "
                        "supersedes — новый приказ отменяет старый; duplicate_of; "
                        "refers_to — явная ссылка"
                    ),
                },
            },
            "required": ["from_id", "to_id"],
        },
    },
    # -- рефакторинг по письму Qwen (02.09): промпт из памяти и граф-запросы --
    {
        "name": "memory_prompt",
        "description": (
            "Собрать system-prompt из памяти для локальной LLM: бюджетный поиск "
            "по вопросу и/или «конституция» (все sys_cmd + persona_def узлы), "
            "секции по тегам-маршрутизаторам (правила первыми, никогда не режутся), "
            "противоречия и хабы. format: plain | chatml (Qwen) | llama3."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Вопрос агента (необязательно)"},
                "constitution": {
                    "type": "boolean",
                    "description": "Добавить все правила/приказы и роли (по умолчанию false)",
                },
                "max_tokens": {"type": "integer", "description": "Бюджет промпта в токенах (по умолчанию 1200)"},
                "format": {"type": "string", "enum": ["plain", "chatml", "llama3"]},
                "session_id": {
                    "type": "string",
                    "description": (
                        "id диалога: регистрирует пред-проход через граф, чтобы "
                        "последующий memory_ground не пометил ответ ungrounded"
                    ),
                },
                "agent": {"type": "string", "description": "Кто спрашивает (для журнала проходов)"},
            },
        },
    },
    {
        "name": "memory_graph",
        "description": (
            "Граф-запросы: op=neighbors (node_id[, rel]) | path (a, b) | "
            "hub (key: сущность или id хаба) | hubs | rules_for (situation: "
            "«какие правила применять в кризисе») | conflicts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": ["neighbors", "path", "hub", "hubs", "rules_for", "conflicts"],
                },
                "node_id": {"type": "string"},
                "a": {"type": "string"},
                "b": {"type": "string"},
                "key": {"type": "string"},
                "rel": {"type": "string"},
                "situation": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["op"],
        },
    },
    {
        "name": "memory_decay",
        "description": (
            "Затухание весов: узлы, которыми не пользовались, тускнеют "
            "(half-life по умолчанию 168 ч). kind=rule не опускается ниже 0.5. "
            "Один дамп на весь батч. Запускается по cron 0 4 * * *."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Один узел (иначе весь стор)"},
                "half_life_hours": {
                    "type": "number",
                    "description": "Период полураспада веса, часов (по умолчанию 168)",
                },
            },
        },
    },
    {
        "name": "memory_prune",
        "description": (
            "Уборка стора по формальному правилу. По умолчанию dry_run=true — "
            "только список кандидатов с причинами, ничего не меняется. "
            "kind=rule не трогается никогда; узлы с входящими ссылками не "
            "удаляются; удаление ограничено max_delete."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "rule": {
                    "type": "string",
                    "enum": ["expired_ttl", "exact_dupes", "source_prefix", "weak"],
                    "description": (
                        "expired_ttl — истёкший TTL в outdated (без удаления); "
                        "exact_dupes — полные дубли claim, выживает свежайший; "
                        "source_prefix — вынос чужого корпуса (нужен export_path); "
                        "weak — слабые, старые и никем не связанные"
                    ),
                },
                "dry_run": {"type": "boolean", "description": "По умолчанию true"},
                "max_delete": {"type": "integer", "description": "Потолок удаления (по умолчанию 100)"},
                "source_prefix": {"type": "string", "description": "Префикс source для правила source_prefix"},
                "older_than_days": {"type": "integer", "description": "Порог возраста для weak (30)"},
                "weak_weight": {"type": "number", "description": "Порог веса для weak (0.1)"},
                "export_path": {"type": "string", "description": "Куда выгрузить удаляемое (обязателен для source_prefix)"},
            },
            "required": ["rule"],
        },
    },
    {
        "name": "memory_summarize",
        "description": (
            "Свернуть кластер похожих узлов в один узел-концентрат: кластер "
            "отбирается по source_prefix/tags/подстроке, исходные узлы "
            "становятся детьми свёртки и переводятся в kind=outdated "
            "(не удаляются). dry_run=true по умолчанию."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "Утверждение узла-свёртки (обязательно при dry_run=false)"},
                "source_prefix": {"type": "string", "description": "Отбор кластера по префиксу source"},
                "contains": {"type": "string", "description": "Отбор кластера по подстроке в claim"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Отбор кластера по тегам"},
                "source": {"type": "string", "description": "Источник узла-свёртки"},
                "evidence": {"type": "array", "items": {"type": "string"}, "description": "Свидетельства свёртки (П5)"},
                "context": {"type": "string", "description": "Контекст свёртки (П6)"},
                "dry_run": {"type": "boolean", "description": "По умолчанию true"},
                "max_nodes": {"type": "integer", "description": "Потолок сворачиваемых узлов (по умолчанию 500)"},
            },
        },
    },
    {
        "name": "memory_stats",
        "description": (
            "Паспорт памяти: узлы, виды, теги, рёбра (и битые), сироты, дубли, "
            "распределение весов, протухшие по TTL, узлы без evidence, байты."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    # -- обязательный проход через граф (03.09, приказ Ильи) ------------------
    {
        "name": "memory_ground_prepare",
        "description": (
            "ШАГ 1 (ДО ответа, обязателен). Ищет по графу и собирает "
            "system-prompt из найденных узлов; регистрирует пред-проход "
            "сессии — без него memory_ground пометит ответ ungrounded. "
            "Сразу проверяет graph-first: если готовый ответ уже есть в "
            "памяти (graph_first.hit=true), отдай его и НЕ зови LLM — "
            "генерация сожжёт токены пользователя впустую."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Вопрос пользователя"},
                "session_id": {
                    "type": "string",
                    "description": (
                        "id диалога — связывает пред-проход с последующим "
                        "memory_ground. Без него сгенерируется свой (в ответе)"
                    ),
                },
                "agent": {"type": "string", "description": "Кто спрашивает (alice, fable, ...)"},
                "constitution": {
                    "type": "boolean",
                    "description": "Добавить правила/приказы и роли в промпт (по умолчанию false)",
                },
                "max_tokens": {"type": "integer", "description": "Бюджет промпта (по умолчанию 1200)"},
                "format": {"type": "string", "enum": ["plain", "chatml", "llama3"]},
                "graph_first": {
                    "type": "boolean",
                    "description": "Проверять готовый ответ из графа (по умолчанию true)",
                },
                "threshold": {
                    "type": "number",
                    "description": "Порог покрытия вопроса узлом для graph-first (по умолчанию 0.75)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_ground",
        "description": (
            "ШАГ 3 (ПОСЛЕ ответа, обязателен). Режет ответ на утверждения и "
            "сверяет каждое с графом -> вердикт grounded | partial | "
            "ungrounded, «прошёл через граф: да/нет/частично», список "
            "узлов-источников и unsupported_claims (что агент выдумал). "
            "Без пред-прохода (memory_ground_prepare) вердикт — ungrounded."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "answer_text": {"type": "string", "description": "Текст ответа агента"},
                "query": {"type": "string", "description": "Исходный вопрос (для журнала)"},
                "session_id": {"type": "string", "description": "Тот же id, что в memory_ground_prepare"},
                "agent": {"type": "string", "description": "Кто отвечал"},
                "require_pre_pass": {
                    "type": "boolean",
                    "description": (
                        "Требовать пред-проход. По умолчанию — политика сервера "
                        "ground_by_default (при старте включена: без пред-прохода "
                        "ответ ungrounded). false — только сверка утверждений, "
                        "для клиентов без сессий"
                    ),
                },
                "reinforce": {
                    "type": "boolean",
                    "description": "Подкрепить узлы-источники (по умолчанию true): память запоминает, что узел пригодился",
                },
                "max_claims": {"type": "integer", "description": "Потолок разбираемых утверждений (по умолчанию 24)"},
            },
            "required": ["answer_text"],
        },
    },
    {
        "name": "memory_answer",
        "description": (
            "Режим graph-first: если ответ на вопрос уже лежит в графе "
            "(покрытие >= порога, вес и уверенность выше порогов, лидер "
            "оторвался от второго) — вернуть его БЕЗ вызова LLM (ноль "
            "токенов генерации). Иначе llm_required=true и готовый "
            "grounded-промпт для модели."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Вопрос пользователя"},
                "session_id": {"type": "string"},
                "agent": {"type": "string"},
                "threshold": {"type": "number", "description": "Порог покрытия (по умолчанию 0.75)"},
                "min_weight": {"type": "number", "description": "Порог веса узла (по умолчанию 0.5)"},
                "min_confidence": {"type": "number", "description": "Порог уверенности (по умолчанию 0.5)"},
                "with_prompt": {
                    "type": "boolean",
                    "description": "Приложить промпт для LLM при промахе (по умолчанию true)",
                },
                "max_tokens": {"type": "integer", "description": "Бюджет промпта при промахе"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_ground_log",
        "description": (
            "Журнал проходов через граф (append-only ground_log.jsonl): "
            "агент, время, запрос, узлы, вердикт. Фильтры по сессии, "
            "событию (prepare|ground|answer) и агенту."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Сколько последних записей (по умолчанию 50)"},
                "session_id": {"type": "string"},
                "event": {"type": "string", "enum": ["prepare", "ground", "answer"]},
                "agent": {"type": "string"},
                "stats": {
                    "type": "boolean",
                    "description": "Добавить сводку по вердиктам и экономии токенов",
                },
            },
        },
    },
]


class MnemosCore:
    """Ядро MCP-сервера: логика инструментов, без HTTP.

    plugins — включённые плагины: None (по умолчанию) — цепочка конфигурации
    (env MNEMOS_PLUGINS -> plugins.json -> дефолты context_engine,gates);
    [] — без плагинов; список/строка — явный набор.
    plugins_config — явный путь к plugins.json (используется, если plugins=None).
    """

    def __init__(
        self, store: Store, plugins: Any = None, plugins_config: Optional[str] = None,
        ground_log_path: Optional[str] = None,
        ground_by_default: Optional[bool] = None,
    ) -> None:
        self.store = store
        search_dirs = [str(store.path.parent)] if getattr(store, "path", None) is not None else None
        enabled = resolve_enabled_plugins(
            plugins=plugins, config_path=plugins_config, search_dirs=search_dirs
        )
        self.plugins = PluginManager(enabled)
        # гейты качества (Г1-Г5) включаются плагином gates
        self.gates_enabled = "gates" in enabled
        self.tools: List[Dict[str, Any]] = list(TOOLS) + self.plugins.tool_schemas()
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "memory_add": self.memory_add,
            "memory_verify": self.memory_verify,
            "memory_search": self.memory_search,
            "memory_rewrite": self.memory_rewrite,
            "memory_reinforce": self.memory_reinforce,
            "memory_link": self.memory_link,
            "memory_link_existing": self.memory_link_existing,
            "memory_decay": self.memory_decay,
            "memory_prune": self.memory_prune,
            "memory_summarize": self.memory_summarize,
            "memory_stats": self.memory_stats,
            "memory_prompt": self.memory_prompt,
            "memory_graph": self.memory_graph,
            "memory_ground_prepare": self.memory_ground_prepare,
            "memory_ground": self.memory_ground,
            "memory_answer": self.memory_answer,
            "memory_ground_log": self.memory_ground_log,
        }
        # Обязательный проход через граф (03.09): журнал проходов лежит рядом
        # со стором, сессии пред-проходов — в памяти процесса (журнал остаётся
        # источником правды и переживает рестарт).
        #
        # Строго ДО plugins.handlers(self) (баг-хант 03.09, D8): плагинам
        # передаётся `self`, и плагин, которому при инициализации понадобится
        # политика или журнал, получал бы AttributeError на недостроенном ядре.
        log_path = ground_log_path or (
            str(store.path.parent / grounding.GROUND_LOG_NAME)
            if getattr(store, "path", None) is not None
            else grounding.GROUND_LOG_NAME
        )
        self.ground_log = grounding.GroundLog(log_path)
        self.sessions = grounding.SessionTracker()
        # Политика сервера: проход через граф обязателен у всех, пока
        # администратор явно не отключил его (аргумент -> env -> дефолт True).
        self.ground_by_default = (
            _env_flag(GROUND_ENV, GROUND_BY_DEFAULT)
            if ground_by_default is None else bool(ground_by_default)
        )
        self._handlers.update(self.plugins.handlers(self))

    # -- инструменты ----------------------------------------------------------
    def memory_add(self, args: Dict[str, Any]) -> Dict[str, Any]:
        claim = args.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("memory_add: параметр claim (строка) обязателен")
        evidence = args.get("evidence")
        if evidence is None:
            evidence = []
        if isinstance(evidence, str):
            evidence = [evidence]
        links = args.get("links") or []
        if isinstance(links, str):
            links = [links]
        tags = args.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        ttl_hours = _as_float(args.get("ttl_hours"))
        conf = _as_float(args.get("confidence"))
        if str(args.get("kind", "fact")) == "hub":
            # ревью 02.09 (п.2): подставной хаб с тегом entity:X подменял
            # навигацию memory_graph/rules_for; хабы строит только
            # офлайн-кластеризатор (mnemos_hubs.py), не клиент
            raise ValueError("memory_add: kind=hub через MCP запрещён — хабы строятся офлайн")
        node = make_node(
            claim=claim.strip(),
            source=str(args.get("source", "")),
            evidence=[str(e) for e in evidence],
            context=str(args.get("context", "")),
            kind=str(args.get("kind", "fact")),
            links=[str(l) for l in links],
            confidence=conf,
            ttl_hours=ttl_hours,
            valid_until=args.get("valid_until"),
            tags=[str(t) for t in tags],
        )
        check_and_update(node, registry=self.store.get)
        if conf is None:
            # уверенность по умолчанию — из truth-gate (score/6), а не 0.5:
            # узел 6/6 с evidence должен ранжироваться выше узла 3/6 без него
            tc = node.truth_check if isinstance(node.truth_check, dict) else {}
            score = _as_float(tc.get("score"))
            node.confidence = (
                max(0.05, min(1.0, score / 6.0)) if score is not None else 0.5
            )
        # Ф1 02.09 (письмо Qwen): каждому узлу — ровно один тег-маршрутизатор
        # sys_cmd / persona_def / world_state; явный тег клиента уважается,
        # без него — классификатор по эвристике (budget.classify_router)
        node.tags = ensure_router_tag(node.tags, {"claim": node.claim, "context": node.context, "kind": node.kind})
        author = str(args.get("author") or "").strip()
        if author and links:
            # рёбра, заданные при создании узла, тоже получают паспорт
            ts = node.ts
            node.link_meta = {
                str(l): {"author": author, "ts": ts} for l in links
            }
        self._apply_write_gates(node)
        return self.store.snapshot([self.store.add(node)])[0]

    def _apply_write_gates(self, node: MemoryNode) -> None:
        """Гейты записи Г1-Г5 ПЕРЕД store.add (фикс 01.09, аудит §5.1).

        Это та самая дыра, из которой вытекло всё остальное: run_write_gates
        существовал, был покрыт тестами и НЕ ВЫЗЫВАЛСЯ из memory_add — поэтому
        в сторе 5090 накопилось 2833 узла «алиса вердикт» без evidence и
        без context (замер 01.09), а в аудите 29.08 — 842 дубля из 5251.

        Поведение:
          * reject  -> запись запрещена (ValueError -> INVALID_PARAMS), и если
            гейт указал на конкретный узел-дубликат, тот узел ПОДКРЕПЛЯЕТСЯ
            (reinforce) — «подкрепите существующий, а не плодите копию»
            перестаёт быть просто советом в тексте ошибки;
          * flag    -> узел не отвергается, но понижается до kind=hypothesis
            (гипотеза, а не факт) и получает пометку в truth_check.gates.

        kind="rule" гейты проходит, но НЕ понижается по flag: правило команды
        задаёт Иван, а не порог уверенности.
        """
        if not self.gates_enabled:
            return
        existing = self.store.all()
        if len(existing) > GATE_MAX_SCAN:
            # Г4 линейна по стору: 43 мс на 3121 узел (замер 01.09). Чтобы
            # запись не деградировала на порядок при росте, сверяем с самыми
            # свежими GATE_MAX_SCAN узлами и честно пишем это в ответ.
            existing = sorted(
                existing, key=lambda d: str(d.get("ts") or ""), reverse=True
            )[:GATE_MAX_SCAN]
        registry = {d["id"]: d for d in existing}
        res = run_write_gates(node.to_dict(), registry=registry, existing=existing)
        gate_note = {
            "verdict": res.verdict, "reason": res.reason,
            "scanned": len(existing), "capped": len(self.store) > GATE_MAX_SCAN,
        }
        if isinstance(node.truth_check, dict):
            node.truth_check = {**node.truth_check, "gates": gate_note}
        if res.verdict == "reject":
            dup_id = _first_node_id(res.reason)
            if dup_id and self.store.get(dup_id) is not None:
                # подкрепляем оригинал вместо создания копии
                w = self.store.reinforce(dup_id, delta=0.05)
                raise GateRejected(
                    f"{res.reason} [узел {dup_id} подкреплён, вес {w:.2f}]"
                )
            raise GateRejected(res.reason)
        if res.verdict == "flag" and node.kind == "fact":
            node.kind = "hypothesis"

    def memory_verify(self, args: Dict[str, Any]) -> Dict[str, Any]:
        node_id = args.get("node_id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("memory_verify: параметр node_id (строка) обязателен")
        node = self.store.get(node_id)
        if node is None:
            raise KeyError(f"memory_verify: узел {node_id!r} не найден")
        result = check_and_update(node, registry=self.store.get)
        if result.verdict == "pass":
            # проверенная память крепче: проход truth-gate подкрепляет узел.
            # Фикс 01.09 (аудит §5.7): прибавка зависит от score, а не всегда
            # +0.05. Иначе живой узел с почасовым обслуживанием упирается в
            # потолок 1.0 за сутки и теряет различимость с болтовнёй.
            # 6/6 -> +0.04, 5/6 -> +0.02, 4/6 -> 0.
            score = _as_float(getattr(result, "score", None))
            delta = 0.05 if score is None else max(0.0, 0.02 * (score - 4))
            mnode = MemoryNode.from_dict(node)
            mnode.reinforce(delta)
            node = mnode.to_dict()
        self.store.update(node)
        out = result.as_dict()
        out["weight"] = node["weight"]
        return out

    def memory_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("memory_search: параметр query (строка) обязателен")
        top_k = args.get("top_k", 5)
        # 02.09 (письмо Qwen): top_k="auto" или budget=true -> бюджетный поиск,
        # где top_k выбирается по сложности запроса (5 / 10 / 20)
        auto_k = isinstance(top_k, str) and top_k.strip().lower() == "auto"
        use_budget = bool(args.get("budget", False)) or auto_k
        try:
            top_k = None if auto_k else int(top_k)
        except (TypeError, ValueError):
            top_k = 5
        mode = args.get("mode", "substring")
        if mode not in ("substring", "semantic", "budget", "rrf"):
            raise ValueError(
                f"memory_search: mode должен быть 'substring', 'semantic', "
                f"'budget' или 'rrf', получено {mode!r}"
            )
        if mode == "budget":
            use_budget = True
        if mode in ("semantic", "rrf"):
            # semantic/rrf + top_k="auto" — это свой режим с k=5, а не budget
            use_budget = False
            top_k = top_k or 5
        want_tags = args.get("tags") or []
        if isinstance(want_tags, str):
            want_tags = [want_tags]
        want_tags = [str(t) for t in want_tags]
        want_kind = args.get("kind")
        use_read_gates = bool(args.get("relevance_gate", False))

        def _filter(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Фильтры тегов/вида + Г5-релевантность (фикс 01.09, аудит §5.2/§5.4).

            До 01.09 отделить правило от болтовни можно было только регуляркой
            по claim: тегов не было ни в схеме, ни в данных (0 уникальных тегов
            на 5251 узел). Теперь memory_search(query="риск", tags=["правило"])
            возвращает правило, а не 685 узлов со словом «риск».
            """
            out = nodes
            if want_tags:
                out = [n for n in out if set(want_tags) <= set(n.get("tags") or [])]
            if want_kind:
                out = [n for n in out if n.get("kind") == want_kind]
            if use_read_gates and self.gates_enabled:
                out = run_read_gates(out, query)
            return out

        if use_budget:
            out = self.store.search_budget(
                query, top_k=top_k,
                token_budget=_as_int(args.get("token_budget")),
                expand=bool(args.get("expand", True)),
            )
            kept = {r["id"] for r in _filter(list(out.get("results") or []))}
            out["results"] = [r for r in out["results"] if r["id"] in kept]
            out["count"] = len(out["results"])
            out["mode"] = "budget"
            self.store.touch(out["results"])
            return out
        if mode == "rrf":
            # ML-BOOST 03.09: слияние Ф1 + BM25F (+ плотный, если есть
            # fastembed) по RRF. Стор не меняется: touch — ниже, по уже
            # отфильтрованной выдаче, как и в подстрочном пути.
            results = _filter(self.store.search_rrf(query, top_k=top_k))
            self.store.touch(results)
            return {"query": query, "mode": "rrf", "count": len(results),
                    "results": self.store.snapshot(results)}
        if mode == "semantic":
            # бенчмарк-фикс 26.08: семантический режим; если fastembed не
            # установлен — честная ошибка с текстом, а не «внутренняя ошибка».
            try:
                results = self.store.search_semantic(query, top_k=top_k)
            except ImportError as exc:
                raise ValueError(f"memory_search (semantic): {exc}") from exc
            kept = {id(n) for n in _filter([n for n, _ in results])}
            results = [(n, sc) for n, sc in results if id(n) in kept]
            self.store.touch([n for n, _ in results])
            snaps = self.store.snapshot([n for n, _ in results])
            return {
                "query": query,
                "mode": "semantic",
                "count": len(results),
                "results": [
                    {"node": node, "score": round(score, 6)}
                    for node, (_, score) in zip(snaps, results)
                ],
            }
        # Ф1: touch=False здесь — использование отмечаем ниже только для
        # узлов, прошедших фильтры tags/kind/гейты (то, что агент увидел)
        results = _filter(self.store.search(query, top_k=top_k, touch=False))
        mode_out = "substring"
        if not results and len(query.split()) >= 2:
            # Ф1 02.09: NL-фоллбек. Подстрока целой фразы («Кто принимает
            # решение пускать сигнал в ордер?») не встречается ни в одном узле
            # -> 0 результатов; по основам слов (budget-поиск) находится.
            # Ключевые слова этот путь не трогает: он включается только при
            # пустой подстрочной выдаче. Замер 02.09: NL recall@5 0.000 -> 0.6+.
            fb = self.store.search_budget(query, top_k=top_k)
            marks = {r["id"]: {"score": r.get("score"), "via": r.get("via"), "rel": r.get("rel"),
                               "matched": (r.get("why") or {}).get("matched")}
                     for r in fb.get("results") or []}
            got = [self.store.get(i) for i in marks]
            results = _filter([n for n in got if n])
            if results:
                mode_out = "token_fallback"
                self.store.touch(results)
                out = self.store.snapshot(results)
                for n in out:  # прозрачность: чем найден узел — словами или ребром (via/rel)
                    n["search"] = marks[n["id"]]
                return {"query": query, "mode": mode_out, "count": len(out), "results": out}
        self.store.touch(results)
        return {"query": query, "mode": mode_out, "count": len(results),
                "results": self.store.snapshot(results)}

    # -- уборка и связывание (01.09, аудит §5.6) ------------------------------
    def memory_link_existing(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from_id, to_id = args.get("from_id"), args.get("to_id")
        if not isinstance(from_id, str) or not from_id.strip():
            raise ValueError("memory_link_existing: параметр from_id (строка) обязателен")
        if not isinstance(to_id, str) or not to_id.strip():
            raise ValueError("memory_link_existing: параметр to_id (строка) обязателен")
        return self.store.link_existing(
            from_id.strip(),
            to_id.strip(),
            bidirectional=bool(args.get("bidirectional")),
            author=str(args.get("author") or ""),
            rel=str(args.get("rel") or ""),
        )

    # -- рефакторинг по письму Qwen (02.09): промпт из памяти, граф-запросы ----
    def memory_prompt(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from .budget import ROUTER_PERSONA, ROUTER_SYS_CMD, build_system_prompt, router_of, wrap_prompt

        query = args.get("query")
        query = query.strip() if isinstance(query, str) and query.strip() else ""
        constitution = bool(args.get("constitution", False))
        if not query and not constitution:
            raise ValueError("memory_prompt: нужен query и/или constitution=true")
        fmt = str(args.get("format") or "plain")
        if fmt not in ("plain", "chatml", "llama3"):
            raise ValueError("memory_prompt: format должен быть plain | chatml | llama3")
        max_tokens = _as_int(args.get("max_tokens"))
        if max_tokens is None:
            max_tokens = PROMPT_DEFAULT_TOKENS
        if max_tokens <= 0:
            raise ValueError("memory_prompt: max_tokens должен быть > 0")
        nodes: List[Dict[str, Any]] = []
        conflicts: List[Dict[str, Any]] = []
        hubs: List[Dict[str, Any]] = []
        if constitution:
            nodes += [
                n for n in self.store.all()
                if n.get("kind") not in ("refuted", "outdated", "hub")
                and router_of(n) in (ROUTER_SYS_CMD, ROUTER_PERSONA)
            ]
        search = None
        if query:
            search = self.store.search_budget(query)
            nodes += [self.store.get(r["id"]) for r in search["results"]]
            conflicts = search["conflicts"]
            hubs = [self.store.get(h["id"]) for h in search["hubs"]]
        # отменённые приказы (входящее supersedes) — помечаем и режем первыми
        g = self.store.graph()
        superseded = {b: a for a, b, rel in ((a, b, r) for a, edges in g.out.items() for b, r in edges)
                      if rel == "supersedes"
                      and (self.store.get(a) or {}).get("kind") not in ("refuted", "outdated")}
        res = build_system_prompt(nodes, max_tokens=max_tokens, conflicts=conflicts,
                                  hubs=[h for h in hubs if h], query=query or None,
                                  superseded=superseded)
        res["format"] = fmt
        res["prompt"] = wrap_prompt(res["text"], fmt, query)
        res["node_ids"] = [n["id"] for n in nodes if n]
        if search is not None:
            res["classification"] = search["classification"]
        # Пред-проход (03.09): memory_prompt по вопросу — это и есть «сначала
        # спросили граф». Клиент, который уже ходит через memory_prompt, чинит
        # свой grounding одним параметром session_id, не меняя вызов.
        sid = self._session_id(args)
        if sid and query:
            rec = self._register_pre_pass(sid, "memory_prompt", query,
                                          res["node_ids"],
                                          self._short_text(args, "agent"))
            res["session_id"] = sid
            res["logged_at"] = rec["ts"]
        return res

    def memory_graph(self, args: Dict[str, Any]) -> Dict[str, Any]:
        op = str(args.get("op") or "")
        g = self.store.graph()
        if op == "neighbors":
            nid = args.get("node_id")
            if not isinstance(nid, str) or not nid.strip():
                raise ValueError("memory_graph neighbors: параметр node_id обязателен")
            return g.neighbors(nid.strip(), rel=args.get("rel") or None)
        if op == "path":
            a, b = args.get("a"), args.get("b")
            if not (isinstance(a, str) and isinstance(b, str) and a.strip() and b.strip()):
                raise ValueError("memory_graph path: параметры a и b обязательны")
            return {"a": a, "b": b, "path": g.path(a.strip(), b.strip())}
        if op == "hub":
            key = args.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValueError("memory_graph hub: параметр key обязателен")
            return g.hub(key.strip())
        if op == "hubs":
            return {"hubs": g.list_hubs()}
        if op == "rules_for":
            sit = args.get("situation")
            if not isinstance(sit, str) or not sit.strip():
                raise ValueError("memory_graph rules_for: параметр situation обязателен")
            return g.rules_for(sit.strip(), limit=_as_int(args.get("limit")) or 10)
        if op == "conflicts":
            return {"conflicts": g.conflicts()}
        raise ValueError(f"memory_graph: неизвестная операция {op!r}; есть {g.OPS}")

    # -- обязательный проход через граф (03.09, приказ Ильи) ------------------
    def _new_session_id(self) -> str:
        return "gs_" + uuid.uuid4().hex[:12]

    def _register_pre_pass(self, session_id: str, tool: str, query: str,
                           node_ids: List[str], agent: str) -> Dict[str, Any]:
        """Отметить пред-проход: и в памяти процесса, и в append-only журнале.

        Журнал — источник правды (переживает рестарт сервера), трекер в памяти
        — быстрый путь: ответ приходит через секунды после подготовки, лезть
        за этим в файл на каждый memory_ground незачем.
        """
        self.sessions.register(session_id, tool=tool, query=query,
                               node_ids=list(node_ids), agent=agent or None)
        return self.ground_log.append(
            "prepare", tool=tool, session_id=session_id, agent=agent or None,
            query=query, node_ids=list(node_ids), nodes=len(node_ids),
        )

    def _find_pre_pass(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        return self.sessions.get(session_id) or self.ground_log.find_pre_pass(session_id)

    def _policy_flag(self, args: Dict[str, Any], key: str,
                     default: Optional[bool] = None) -> bool:
        """Булев параметр протокола: отсутствие -> политика, мусор -> ошибка.

        Голый `bool(args[key])` был дырой (баг-хант 03.09, D4): схема объявляет
        boolean, а на деле проходило что угодно — `0`, `""`, `[]`, `{}` снимали
        требование пред-прохода, а строка `"false"` (истинная в питоне!) его,
        наоборот, включала. Ослабление политики опечаткой должно быть громким.
        """
        value = args.get(key)
        if value is None:
            return self.ground_by_default if default is None else default
        if isinstance(value, bool):
            return value
        raise ValueError(
            f"{key}: ожидается true или false, получено {value!r} "
            f"({type(value).__name__})"
        )

    @staticmethod
    def _short_text(args: Dict[str, Any], key: str,
                    limit: int = MAX_ID_LEN) -> str:
        """Короткая строка-идентификатор (agent, session_id) с потолком длины.

        Потолок здесь, а не в журнале: session_id и agent пишутся в журнал
        как есть (обрезанный id перестал бы джойниться), поэтому мегабайтный
        id должен отлетать на входе, а не раздувать файл (D6).
        """
        value = args.get(key)
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError(f"{key}: ожидается строка, получено {type(value).__name__}")
        value = value.strip()
        if len(value) > limit:
            raise ValueError(f"{key}: длина {len(value)} — потолок {limit} символов")
        return value

    def _session_id(self, args: Dict[str, Any]) -> str:
        return self._short_text(args, "session_id")

    @staticmethod
    def _threshold(args: Dict[str, Any], key: str, default: float) -> float:
        """Порог из аргументов; отсутствие -> умолчание, мусор -> ошибка.

        Через `or` тут нельзя: порог 0.0 — валидное «пропускать всё», и `or`
        подменил бы его умолчанием, тихо ужесточив то, что клиент ослабил.
        """
        if args.get(key) is None:
            return default
        v = _as_float(args.get(key))
        if v is None or not (0.0 <= v <= 1.0):
            raise ValueError(f"{key}: ожидается число в [0, 1], получено {args.get(key)!r}")
        return v

    def memory_ground_prepare(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """ШАГ 1: поиск по графу + промпт + graph-first, с записью пред-прохода."""
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("memory_ground_prepare: параметр query (строка) обязателен")
        query = query.strip()
        session_id = self._session_id(args) or self._new_session_id()
        agent = self._short_text(args, "agent")

        gf: Dict[str, Any] = {"hit": False, "reason": "graph_first выключен параметром",
                              "answer": None}
        if bool(args.get("graph_first", True)):
            gf = grounding.graph_first(
                self.store, query,
                coverage_threshold=self._threshold(
                    args, "threshold", grounding.GRAPH_FIRST_COVERAGE),
            )
        prompt = self.memory_prompt({
            "query": query,
            "constitution": bool(args.get("constitution", False)),
            "max_tokens": args.get("max_tokens"),
            "format": args.get("format"),
        })
        node_ids = list(prompt.get("node_ids") or [])
        if gf.get("hit") and gf.get("node_id") and gf["node_id"] not in node_ids:
            node_ids.append(gf["node_id"])
        rec = self._register_pre_pass(session_id, "memory_ground_prepare",
                                      query, node_ids, agent)
        return {
            "session_id": session_id,
            "query": query,
            "agent": agent or None,
            "policy": grounding.SYSTEM_PROMPT_TEMPLATE,
            "ground_by_default": self.ground_by_default,
            "graph_first": gf,
            "prompt": prompt["prompt"],
            "text": prompt["text"],
            "tokens": prompt["tokens"],
            "over_budget": prompt.get("over_budget"),
            "sections": prompt.get("sections"),
            "classification": prompt.get("classification"),
            "node_ids": node_ids,
            "next_step": (
                "graph_first.hit=true -> отдай graph_first.answer без вызова LLM; "
                "иначе ответь по выдержке и вызови "
                f"memory_ground(answer_text=..., session_id={session_id!r})"
            ),
            "logged_at": rec["ts"],
        }

    def memory_ground(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """ШАГ 3: сверка готового ответа с графом -> вердикт + узлы-источники."""
        answer = args.get("answer_text")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("memory_ground: параметр answer_text (строка) обязателен")
        session_id = self._session_id(args)
        agent = self._short_text(args, "agent")
        query = str(args.get("query") or "").strip()
        # Умолчание берём из политики сервера (ground_by_default), а не из
        # константы: `args.get(key, True)` вернул бы None при явном
        # require_pre_pass: null и тихо ослабил бы политику до False.
        require = self._policy_flag(args, "require_pre_pass")
        # не `or MAX_CLAIMS`: 0 — это ошибка клиента, а `or` молча превратил бы
        # её в умолчание и разобрал бы все 24 утверждения вместо нуля
        max_claims = _as_int(args.get("max_claims"))
        if max_claims is None:
            max_claims = grounding.MAX_CLAIMS
        elif max_claims <= 0:
            raise ValueError("memory_ground: max_claims должен быть > 0")

        pre = self._find_pre_pass(session_id)
        if not query and pre:
            query = str(pre.get("query") or "")
        out = grounding.ground_answer(
            self.store, answer, query=query, pre_pass=pre,
            require_pre_pass=require, max_claims=max_claims,
        )
        out["session_id"] = session_id or None
        out["agent"] = agent or None
        out["ground_by_default"] = self.ground_by_default
        out["require_pre_pass"] = require
        # Узел, которым агент реально подкрепил ответ, пригодился — это ровно
        # тот сигнал, ради которого существует reinforce. Подкрепляем только
        # опоры подтверждённых утверждений, а не всё, что нашлось поиском.
        #
        # И только у ответа, который политику ПРОШЁЛ (баг-хант 03.09, D5):
        # раньше вес рос и при verdict=ungrounded, то есть клиент без
        # пред-прохода накачивал веса графа, а каждый такой вызов ещё и
        # переписывал nodes.json по разу на узел-опору. Отвергнутый ответ не
        # доказывает, что узел пригодился.
        reinforced: List[Dict[str, Any]] = []
        skipped_reinforce = None
        if self._policy_flag(args, "reinforce", default=True):
            if out["verdict"] == "ungrounded":
                skipped_reinforce = (
                    "вердикт ungrounded: веса узлов не трогаем — отвергнутый "
                    "ответ не подтверждает, что узел пригодился"
                )
            else:
                strong = {s["id"] for c in out["claims"] if c["verdict"] == "supported"
                          for s in c["support"][:1]}
                for nid in sorted(strong):
                    try:
                        reinforced.append({"id": nid,
                                           "weight": self.store.reinforce(nid)})
                    except KeyError:
                        continue  # узел удалили между ответом и сверкой — не беда
        out["reinforced"] = reinforced
        if skipped_reinforce:
            out["reinforce_skipped"] = skipped_reinforce
        rec = self.ground_log.append(
            "ground", session_id=session_id or None, agent=agent or None,
            query=query or None, verdict=out["verdict"],
            passed_through_graph=out["passed_through_graph"],
            claims_verdict=out["claims_verdict"],
            grounded_ratio=out["grounded_ratio"], counts=out["counts"],
            node_ids=out["source_node_ids"], pre_pass=bool(pre),
            # фиксируем и фактическое требование политики: обход через
            # require_pre_pass=false должен быть виден в журнале (D4)
            require_pre_pass=require, ground_by_default=self.ground_by_default,
            answer_sha256=grounding.answer_sha256(answer),
            answer_preview=answer.strip()[:200],
            answer_tokens=out["answer_tokens"],
        )
        out["logged_at"] = rec["ts"]
        return out

    def memory_answer(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """graph-first: готовый ответ из графа без LLM, иначе — промпт для LLM."""
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("memory_answer: параметр query (строка) обязателен")
        query = query.strip()
        session_id = self._session_id(args) or self._new_session_id()
        agent = self._short_text(args, "agent")
        search = self.store.search_budget(query)
        gf = grounding.graph_first(
            self.store, query, search=search,
            coverage_threshold=self._threshold(
                args, "threshold", grounding.GRAPH_FIRST_COVERAGE),
            min_weight=self._threshold(
                args, "min_weight", grounding.GRAPH_FIRST_WEIGHT),
            min_confidence=self._threshold(
                args, "min_confidence", grounding.GRAPH_FIRST_CONFIDENCE),
        )
        out: Dict[str, Any] = {
            "session_id": session_id, "query": query, "agent": agent or None,
            "mode": "graph_first",
            "hit": bool(gf.get("hit")),
            "llm_required": not gf.get("hit"),
            "answer": gf.get("answer"),
            "graph_first": gf,
        }
        node_ids = [gf["node_id"]] if gf.get("hit") else []
        if gf.get("hit"):
            # выдали узел агенту — отмечаем использование, как и обычный поиск
            node = self.store.get(gf["node_id"])
            if node is not None:
                self.store.touch([node])
        elif bool(args.get("with_prompt", True)):
            prompt = self.memory_prompt({"query": query,
                                         "max_tokens": args.get("max_tokens")})
            out["prompt"] = prompt["prompt"]
            out["tokens"] = prompt["tokens"]
            node_ids = list(prompt.get("node_ids") or [])
            out["policy"] = grounding.SYSTEM_PROMPT_TEMPLATE
        out["node_ids"] = node_ids
        rec = self._register_pre_pass(session_id, "memory_answer", query,
                                      node_ids, agent)
        self.ground_log.append(
            "answer", session_id=session_id, agent=agent or None, query=query,
            hit=bool(gf.get("hit")), node_ids=node_ids,
            tokens_saved_min=gf.get("tokens_saved_min"),
            llm_calls_saved=gf.get("llm_calls_saved"),
        )
        out["logged_at"] = rec["ts"]
        out["next_step"] = (
            "ответ из графа: LLM звать не нужно"
            if gf.get("hit") else
            "ответь по prompt и вызови memory_ground(answer_text=..., "
            f"session_id={session_id!r})"
        )
        return out

    def memory_ground_log(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Журнал проходов через граф (append-only) + сводка."""
        limit = _as_int(args.get("limit"))
        if limit is None:
            limit = 50
        elif limit <= 0:
            raise ValueError("memory_ground_log: limit должен быть > 0")
        event = args.get("event")
        if event is not None and event not in grounding.GroundLog.EVENTS:
            raise ValueError(
                f"memory_ground_log: event должен быть из "
                f"{grounding.GroundLog.EVENTS}, получено {event!r}"
            )
        records = self.ground_log.read(
            limit=limit,
            session_id=str(args["session_id"]) if args.get("session_id") else None,
            event=event,
            agent=str(args["agent"]) if args.get("agent") else None,
        )
        out: Dict[str, Any] = {
            "path": str(self.ground_log.path),
            "count": len(records),
            "records": records,
        }
        if bool(args.get("stats", False)):
            grounds = [r for r in records if r.get("event") == "ground"]
            answers = [r for r in records if r.get("event") == "answer"]
            verdicts: Dict[str, int] = {}
            for r in grounds:
                v = str(r.get("verdict") or "?")
                verdicts[v] = verdicts.get(v, 0) + 1
            out["stats"] = {
                "window": len(records),
                "prepare": sum(1 for r in records if r.get("event") == "prepare"),
                "ground": len(grounds),
                "answer": len(answers),
                "verdicts": verdicts,
                "graph_first_hits": sum(1 for r in answers if r.get("hit")),
                "llm_calls_saved": sum(int(r.get("llm_calls_saved") or 0) for r in answers),
                "tokens_saved_min": sum(int(r.get("tokens_saved_min") or 0) for r in answers),
                "note": "сводка по окну журнала (последние limit записей), не за всё время",
            }
        return out

    def memory_decay(self, args: Dict[str, Any]) -> Dict[str, Any]:
        node_id = args.get("node_id")
        half_life = _as_float(args.get("half_life_hours")) or DEFAULT_HALF_LIFE_HOURS
        out = self.store.decay(
            node_id=node_id if isinstance(node_id, str) and node_id.strip() else None,
            half_life_hours=half_life,
        )
        weights = sorted(out.values())
        n = len(weights) or 1
        st = self.store.stats()
        return {
            "decayed": len(out),
            "half_life_hours": half_life,
            "levels": st.get("levels"),
            "quantized": st.get("quantized"),
            "weight_min": round(weights[0], 4) if weights else None,
            "weight_max": round(weights[-1], 4) if weights else None,
            "weight_mean": round(sum(weights) / n, 4) if weights else None,
            "below_0_5": sum(1 for w in weights if w < 0.5),
        }

    def memory_prune(self, args: Dict[str, Any]) -> Dict[str, Any]:
        rule = args.get("rule")
        if not isinstance(rule, str) or not rule.strip():
            raise ValueError("memory_prune: параметр rule (строка) обязателен")
        return self.store.prune(
            rule=rule.strip(),
            dry_run=bool(args.get("dry_run", True)),   # безопасно по умолчанию
            max_delete=int(args.get("max_delete", 100)),
            source_prefix=str(args.get("source_prefix", "")),
            older_than_days=int(args.get("older_than_days", 30)),
            weak_weight=_as_float(args.get("weak_weight")) or 0.1,
            export_path=args.get("export_path"),
        )

    def memory_summarize(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Свёртка кластера похожих узлов в один узел-концентрат.

        Механизм для «узла сна» (Ж5) и свёрток N8/N9 из аудита: 2833 узла
        «алиса вердикт» — это не память, это лог; в память должна идти
        суточная свёртка. Исходные узлы НЕ удаляются: они становятся детьми
        свёртки и переводятся в kind=outdated — история сохраняется.
        """
        source_prefix = str(args.get("source_prefix", ""))
        contains = str(args.get("contains", ""))
        want_tags = args.get("tags") or []
        if isinstance(want_tags, str):
            want_tags = [want_tags]
        if not (source_prefix or contains or want_tags):
            raise ValueError(
                "memory_summarize: нужен хотя бы один отбор кластера — "
                "source_prefix, contains или tags"
            )
        max_nodes = max(1, int(args.get("max_nodes", 500)))
        cluster = []
        for d in self.store.all():
            if d.get("kind") in ("rule", "outdated", "refuted"):
                continue
            if source_prefix and not str(d.get("source") or "").startswith(source_prefix):
                continue
            if contains and contains.lower() not in str(d.get("claim") or "").lower():
                continue
            if want_tags and not set(map(str, want_tags)) <= set(d.get("tags") or []):
                continue
            cluster.append(d)
        cluster.sort(key=lambda d: str(d.get("ts") or ""))
        capped = len(cluster) > max_nodes
        cluster = cluster[:max_nodes]
        preview = {
            "cluster_size": len(cluster),
            "capped_by_max_nodes": capped,
            "ts_from": cluster[0].get("ts") if cluster else None,
            "ts_to": cluster[-1].get("ts") if cluster else None,
            "sample_claims": [str(d.get("claim"))[:80] for d in cluster[:5]],
            "ids": [d["id"] for d in cluster[:200]],
        }
        if bool(args.get("dry_run", True)):
            return {**preview, "dry_run": True, "applied": False}
        claim = args.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(
                "memory_summarize: при dry_run=false параметр claim "
                "(утверждение узла-свёртки) обязателен"
            )
        if not cluster:
            raise ValueError("memory_summarize: кластер пуст — сворачивать нечего")
        evidence = args.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        node = make_node(
            claim=claim.strip(),
            source=str(args.get("source", "memory_summarize")),
            evidence=[str(e) for e in evidence] or [
                f"свёртка {len(cluster)} узлов: {preview['ts_from']} .. {preview['ts_to']}"
            ],
            context=str(args.get("context", ""))
            or f"Свёртка кластера ({len(cluster)} узлов). Исходные узлы -> outdated.",
            kind="fact",
            tags=[str(t) for t in want_tags],
        )
        node.tags = ensure_router_tag(node.tags, {"claim": node.claim, "context": node.context, "kind": node.kind})
        check_and_update(node, registry=self.store.get)
        # свёртка по определению похожа на свои исходники — Г4 обязан её
        # отклонить; гейты здесь не применяем осознанно (это и есть лечение).
        summary = self.store.add(node)
        linked, marked = 0, 0
        for d in cluster:
            try:
                self.store.link_existing(summary["id"], d["id"])
                linked += 1
            except (KeyError, ValueError):
                continue
            cur = self.store.get(d["id"])
            if cur is not None and cur.get("kind") not in ("rule",):
                cur["kind"] = "outdated"
                marked += 1
        self.store._save()
        return {
            **preview,
            "dry_run": False,
            "applied": True,
            "summary_node": summary["id"],
            "linked": linked,
            "marked_outdated": marked,
        }

    def memory_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.store.stats()

    def memory_rewrite(self, args: Dict[str, Any]) -> Dict[str, Any]:
        node_id = args.get("node_id")
        new_claim = args.get("new_claim")
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("memory_rewrite: параметр node_id (строка) обязателен")
        if not isinstance(new_claim, str) or not new_claim.strip():
            raise ValueError("memory_rewrite: параметр new_claim (строка) обязателен")
        node = self.store.rewrite(
            node_id,
            new_claim.strip(),
            source=str(args.get("source", "")),
            reason=str(args.get("reason", "")),
        )
        check_and_update(node, registry=self.store.get)
        self.store.update(node)
        return self.store.snapshot([node])[0]

    def memory_reinforce(self, args: Dict[str, Any]) -> Dict[str, Any]:
        node_id = args.get("node_id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("memory_reinforce: параметр node_id (строка) обязателен")
        delta = args.get("delta", 0.05)
        try:
            delta = float(delta)
        except (TypeError, ValueError):
            delta = 0.05
        weight = self.store.reinforce(node_id, delta=delta)
        return {"node_id": node_id, "weight": weight}

    def memory_link(self, args: Dict[str, Any]) -> Dict[str, Any]:
        parent_id = args.get("parent_id")
        claim = args.get("claim")
        if not isinstance(parent_id, str) or not parent_id.strip():
            raise ValueError("memory_link: параметр parent_id (строка) обязателен")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("memory_link: параметр claim (строка) обязателен")
        evidence = args.get("evidence")
        if evidence is None:
            evidence = []
        if isinstance(evidence, str):
            evidence = [evidence]
        node = make_node(
            claim=claim.strip(),
            source=str(args.get("source", "")),
            evidence=[str(e) for e in evidence],
            context=str(args.get("context", "")),
        )
        node.tags = ensure_router_tag(node.tags, {"claim": node.claim, "context": node.context, "kind": node.kind})
        check_and_update(node, registry=self.store.get)
        child = self.store.add_child(parent_id, node)
        return {
            "node": self.store.snapshot([child])[0],
            "parent_id": parent_id,
            "depth": self.store.depth(child["id"]),
        }

    # -- диспетчер ------------------------------------------------------------
    def handle_request(self, req: Any) -> Optional[Dict[str, Any]]:
        """Обрабатывает JSON-RPC запрос; None для нотификаций (без id)."""
        if not isinstance(req, dict):
            return self._error(None, INVALID_REQUEST, "запрос должен быть объектом")
        method = req.get("method")
        req_id = req.get("id")
        if req_id is None:
            # нотификация — ответ не нужен
            return None
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    # Контракт едет клиенту на рукопожатии: агент узнаёт про
                    # обязательный проход через граф до первого ответа, а не из
                    # README, который он не читал.
                    #
                    # Именно здесь, в корне InitializeResult (баг-хант 03.09,
                    # D1). Лежало в serverInfo — а serverInfo по спеке MCP это
                    # Implementation{name, version}, и SDK разбирает его через
                    # z.object: лишние ключи не роняют клиента, их МОЛЧА
                    # вырезает. Контракт доезжал до нас, но не до клиента.
                    **({"instructions": grounding.SYSTEM_PROMPT_TEMPLATE}
                       if self.ground_by_default else {}),
                    "_meta": {"ground_by_default": self.ground_by_default},
                },
            }
        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": self.tools}}
        if method == "tools/call":
            params = req.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            handler = self._handlers.get(name)
            if handler is None:
                return self._error(req_id, INVALID_PARAMS, f"неизвестный инструмент: {name!r}")
            try:
                out = handler(args)
            except (ValueError, KeyError) as exc:
                code = NODE_NOT_FOUND if isinstance(exc, KeyError) else INVALID_PARAMS
                return self._error(req_id, code, str(exc))
            except Exception as exc:  # pragma: no cover
                return self._error(req_id, INTERNAL_ERROR, f"внутренняя ошибка: {exc}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(out, ensure_ascii=False)}
                    ],
                    "isError": False,
                },
            }
        return self._error(req_id, METHOD_NOT_FOUND, f"метод не найден: {method!r}")

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }


class _Handler(BaseHTTPRequestHandler):
    server: "MCPHttpServer"  # тип: наш сервер (устанавливается HTTPServer)

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                    "nodes": len(self.server.mnemos.store),
                    "ground_by_default": self.server.mnemos.ground_by_default,
                },
            )
        else:
            self._send_json(
                404,
                self.server.mnemos._error(None, METHOD_NOT_FOUND, f"нет пути {path!r}"),
            )

    def do_POST(self) -> None:
        try:
            try:
                length = int(self.headers.get("Content-Length", 0))
            except (TypeError, ValueError):  # кривой Content-Length — как 0
                length = 0
            length = max(0, length)  # отрицательный — тоже как 0 (B5)
            if length > MAX_BODY_BYTES:
                # фикс аудита 26.08: лимит тела — отвечаем 413 до чтения,
                # чтобы огромный Content-Length не съел память (B5).
                self._send_json(
                    413,
                    self.server.mnemos._error(
                        None,
                        INVALID_REQUEST,
                        f"тело запроса превышает лимит {MAX_BODY_BYTES} байт",
                    ),
                )
                return
            raw = self.rfile.read(length) if length else b""
            try:
                req = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(
                    400,
                    self.server.mnemos._error(None, PARSE_ERROR, "невалидный JSON"),
                )
                return
            resp = self.server.mnemos.handle_request(req)
            if resp is None:  # нотификация
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self._send_json(200, resp)
        except Exception as exc:  # pragma: no cover
            self._send_json(
                500, self.server.mnemos._error(None, INTERNAL_ERROR, f"HTTP: {exc}")
            )

    def log_message(self, *args: Any) -> None:  # тихо
        pass


class MCPHttpServer(ThreadingHTTPServer):
    """HTTP-сервер MCP (JSON-RPC 2.0), потоковый, с ядром MnemosCore."""

    daemon_threads = True

    def __init__(
        self, addr: tuple, store: Store, plugins: Any = None,
        plugins_config: Optional[str] = None,
        ground_by_default: Optional[bool] = None,
    ) -> None:
        self.mnemos = MnemosCore(store, plugins=plugins, plugins_config=plugins_config,
                                 ground_by_default=ground_by_default)
        super().__init__(addr, _Handler)


def run(
    host: str = "127.0.0.1",
    port: int = 8765,
    store_path: Optional[str] = None,
    plugins: Any = None,
    plugins_config: Optional[str] = None,
    ground_by_default: Optional[bool] = None,
) -> None:
    """Запускает MCP-сервер (блокирующий).

    store_path     — путь к nodes.json, либо "blank"/"blank:<путь>" для чистого
                     графа новой инстанции (None — env MNEMOS_STORE);
    plugins        — включённые плагины (None — env/plugins.json/дефолты);
    plugins_config — явный путь к plugins.json;
    ground_by_default — обязательный проход через граф (None — env/дефолт True).
    """
    path, created_blank = resolve_store_path(store_path)
    store = Store(path)
    httpd = MCPHttpServer((host, port), store, plugins=plugins,
                          plugins_config=plugins_config,
                          ground_by_default=ground_by_default)
    print(f"ShineMnemos MCP {SERVER_VERSION} на http://{host}:{httpd.server_address[1]}/")
    note = "  (создан пустым: blank)" if created_blank else ""
    print(f"Хранилище: {store.path}  (узлов: {len(store)}){note}")
    # режим определяем разбором спецификации, а не по префиксу строки: путь
    # blankgraph.json — это обычный путь, и врать про него нельзя (Д3)
    if not created_blank and blank_target(store_path) is not None:
        print("blank: файл уже был на месте и пуст — открыт как есть, не перезаписан")
    ground = httpd.mnemos.ground_by_default
    print(
        "Проход через граф обязателен (ground_by_default): "
        + ("да — ответ без memory_ground_prepare помечается ungrounded"
           if ground else f"НЕТ (выключен через {GROUND_ENV}/--no-ground-by-default)")
    )
    enabled = httpd.mnemos.plugins.enabled
    print(f"Плагины: {', '.join(enabled) if enabled else '(нет)'}")

    # Ф1 02.09: systemctl stop шлёт SIGTERM — переводим его в KeyboardInterrupt,
    # чтобы finally записал накопленные usage/level (иначе попадания в поиск
    # и подъёмы L1->L0 с момента последней записи терялись бы при рестарте)
    def _sigterm(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _sigterm)
    except (ValueError, OSError):  # не главный поток / Windows
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлен.")
    finally:
        httpd.server_close()
        if store.flush_usage():
            print("usage/level записаны при остановке")


def serve_in_thread(store: Store, host: str = "127.0.0.1", port: int = 0) -> MCPHttpServer:
    """Запускает сервер в фоновом потоке (для тестов и встраивания)."""
    httpd = MCPHttpServer((host, port), store)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd
