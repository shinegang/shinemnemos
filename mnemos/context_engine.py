# -*- coding: utf-8 -*-
"""ShineMnemos: context_engine — революционная работа с контекстом.

Окно = RAM, Mnemos = диск/индекс (research: context_window_research.md).
Модуль оркестрирует контекст агентских циклов без внешних API — только
stdlib + то, что уже есть в репо (MemoryNode/Store/truth-gate; fastembed —
опционально, как в векторах).

Компоненты:
  HierarchicalCompactor — иерархическое сжатие старой части сессии с
      выгрузкой в Store: свежее сжимается ПОДРОБНЕЕ (мелкие чанки), старое —
      в ВЫЖИМКИ (крупные чанки); над уровнем 0 при необходимости строится
      пирамида сводок. Узлы kind='context_summary' со ссылками на исходные
      сообщения. Перед записью каждая сводка прогоняется через гейты
      качества mnemos.gates (если модуль доступен) — reject не пишется,
      flag помечается в контексте узла.
  CanonicalPrefix      — кэш-стабильный канонический префикс для DeepSeek
      on-disk caching: НЕИЗМЕННАЯ шапка (system + стабильные инструкции) +
      APPEND-ONLY хвост. Любая мутация записанного ломает префиксный кэш,
      поэтому check_stability() запрещает изменение шапки и вставки в
      середину хвоста. Повторный build_prefix — байт-в-байт идентичен.
  ContextDefragmenter  — «дефрагментация» сессии: из старой сессии делает
      чистую новую с компактной сводкой-«памятью» (через Compactor) и
      возвращает новый стартовый контекст: [сводка-преамбула] + окно.

Как использовать в агентском цикле:
  1. Каждый ход: messages = session; если len > порога —
     window, refs = compactor.compact(messages, keep_recent=8)
     -> window идёт в промпт, refs -> ссылки на память в Mnemos.
  2. Префикс: prefix.build_prefix(system, static_blocks); память добавлять
     только через prefix.append_tail(block) — кэш DeepSeek жив.
  3. Между эпизодами: new_ctx = defrag.defragment(session, store) —
     свежая сессия без истории, но с памятью.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .model import MemoryNode, make_node, now_iso

# --- токены/факты (эвристики, stdlib) ------------------------------------------

_NUM_RE = re.compile(r"\d[\d\s.,%$€₽¥+-]*")
_ENTITY_RE = re.compile(r"(?<![A-Za-zА-ЯЁа-яё])([A-ZА-ЯЁ][A-Za-zА-ЯЁа-яё0-9_.-]{2,})")
_VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\w*\b", re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
# Частые слова в начале русских предложений — не сущности.
_ENTITY_STOP = {
    "мы", "вчера", "сегодня", "завтра", "пользователь", "пользователи",
    "агент", "когда", "что", "это", "там", "тут", "вот", "тогда", "далее",
    "итак", "наш", "наша", "наши", "моя", "мой", "мои", "теперь", "нужно",
    "можно", "надо", "поэтому", "потому", "конечно", "да", "нет", "так",
    "the", "it", "this", "that", "we", "they", "he", "she", "i", "you",
}


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Приблизительный подсчёт токенов (RU/EN смесь ~4 симв./токен).

    Точный счётчик недоступен без токенизатора модели; для оценок
    экономии (сравнение «до/после») этой точности достаточно.
    """
    text = str(text or "")
    if not text.strip():
        return 0
    return max(1, int(math.ceil(len(text) / max(0.1, chars_per_token))))


def _extract_facts(texts: Sequence[str]) -> List[str]:
    """Вытаскивает «ключевые факты» из текстов: числа, версии, сущности.

    Чистая эвристика на stdlib (аналог П3-культуры truth-gate: цифры и
    имена собственные — проверяемые якоря). Сводки кладут их в claim,
    чтобы подстрочный поиск Store находил сводку по факту.
    """
    facts: List[str] = []
    seen: set = set()
    for t in texts:
        t = str(t or "")
        # числа/проценты: "97 000", "12%", "2.4.1", "2025"
        for m in _NUM_RE.findall(t):
            raw = m.strip(" ,.")
            tokens = raw.split()
            if tokens and all(tok.isdigit() for tok in tokens) and len(tokens) > 1:
                # группы цифр: "97 000" -> "97000" (тысячный разделитель);
                # "3 2025" -> "3" (не склеиваем части разных чисел)
                tok = "".join(tokens) if all(len(tok) <= 3 for tok in tokens) else tokens[0]
            else:
                tok = raw  # "12%", "2.4.1", "2025" — как есть
            if len(tok) >= 2 and tok not in seen:
                seen.add(tok)
                facts.append(tok)
        # версии: v2.4.1, 2.4.1
        for m in _VERSION_RE.findall(t):
            if m not in seen:
                seen.add(m)
                facts.append(m)
        # сущности (заглавные слова)
        for m in _ENTITY_RE.findall(t):
            low = m.lower()
            if low in _ENTITY_STOP or len(m) < 3 or m in seen:
                continue
            seen.add(m)
            facts.append(m)
        if len(facts) >= 12:
            break
    return facts[:12]


def _first_sentence(text: str, limit: int = 160) -> str:
    """Первое предложение текста, обрезанное до limit символов."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t:
        return ""
    parts = _SENT_SPLIT_RE.split(t)
    head = parts[0] if parts else t
    if len(head) <= limit:
        return head
    cut = head[: limit - 1].rsplit(" ", 1)[0]
    return (cut or head[:limit]) + "…"


def _norm_messages(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    """Нормализует сообщения сессии в dict {id, role, content, ts}.

    Принимает: str, dict {content|claim, role, id, ts}, MemoryNode.
    """
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(messages):
        if isinstance(m, str):
            d = {"id": f"msg_{i}", "role": "user", "content": m, "ts": ""}
        elif isinstance(m, MemoryNode):
            d = {
                "id": str(m.id),
                "role": "memory",
                "content": m.claim,
                "ts": m.ts,
            }
        elif isinstance(m, dict):
            content = m.get("content")
            if content is None:
                content = m.get("claim", "")
            d = {
                "id": str(m.get("id") or f"msg_{i}"),
                "role": str(m.get("role") or "user"),
                "content": str(content or ""),
                "ts": str(m.get("ts") or ""),
            }
        else:  # pragma: no cover — защита от неожиданных типов
            d = {"id": f"msg_{i}", "role": "user", "content": str(m), "ts": ""}
        out.append(d)
    return out


# ============================================================================
# HierarchicalCompactor — иерархическое сжатие с выгрузкой в Store
# ============================================================================


class HierarchicalCompactor:
    """Окно = RAM (последние N сообщений), старое — иерархия сводок в Store.

    compact(messages, keep_recent=N) -> (window, memory_refs)

    Иерархия: свежая половина старых сообщений сжимается ПОДРОБНО (мелкие
    чанки, факты в claim), древняя — в ВЫЖИМКИ (крупные чанки, короче);
    если сводок уровня 0 больше hierarchy_cap, над ними строится уровень 1
    (сводки сводок со ссылками links на детей) — и так до успокоения.

    Каждый узел: kind='context_summary', links -> исходные сообщения
    (уровень 0) или дочерние сводки (уровень >0), evidence -> источники,
    context -> JSON-метаданные {level, span, detailed, source_messages}.
    """

    def __init__(
        self,
        store: Optional[Any] = None,
        chunk_size: int = 3,
        hierarchy_cap: int = 6,
        gates_enabled: bool = True,
    ) -> None:
        self.store = store
        self.chunk_size = max(1, int(chunk_size))
        self.hierarchy_cap = max(2, int(hierarchy_cap))
        self.gates_enabled = bool(gates_enabled)
        # снапшоты на время одного compact() — для гейтов (детерминизм)
        self._existing: List[Dict[str, Any]] = []
        self._registry: Dict[str, Any] = {}
        self._now: datetime = datetime.now(timezone.utc)

    # -- публичный API ------------------------------------------------------

    def compact(
        self,
        messages: Sequence[Any],
        keep_recent: int = 8,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Сжимает старую часть сессии в иерархию сводок, возвращает окно.

        window      — последние keep_recent сообщений (RAM, идут в промпт);
        memory_refs — что выгружено в память: сырые сообщения (type='raw')
                      и узлы-сводки (type='summary', с node_id/level/гейтом).
        """
        msgs = _norm_messages(messages)
        n = len(msgs)
        k = max(0, int(keep_recent))
        if k >= n:
            return msgs, []
        window = msgs[n - k:] if k else []

        refs: List[Dict[str, Any]] = [
            {"type": "raw", "message_id": m["id"], "role": m["role"],
             "kept_in_window": False}
            for m in msgs[: n - k]
        ]
        if self.store is None:
            # память не подключена: отдаём только рефы на исходники
            return window, refs

        # снапшот памяти для гейтов (Г4 сверяет с существующими узлами)
        self._now = datetime.now(timezone.utc)
        self._existing = self.store.all()
        self._registry = {d["id"]: d for d in self._existing}

        old = msgs[: n - k]
        nodes = self._build_hierarchy(old)
        stored_refs = self._store_nodes(nodes)
        return window, refs + stored_refs

    # -- иерархия ------------------------------------------------------------

    def _build_hierarchy(self, old: List[Dict[str, Any]]) -> List[MemoryNode]:
        """Строит пирамиду сводок над сообщениями old.

        Уровень 0: свежая половина — детальные сводки (чанк = chunk_size),
        древняя половина — выжимки (чанк = chunk_size*2, короче текст).
        Уровни 1+: сводки сводок, пока число корневых > hierarchy_cap.
        """
        mid = len(old) // 2
        old_part, fresh_part = old[:mid], old[mid:]

        level0: List[MemoryNode] = []
        # свежее -> подробнее: мелкие чанки, детальные сводки
        for start in range(0, len(fresh_part), self.chunk_size):
            chunk = fresh_part[start: start + self.chunk_size]
            level0.append(
                self._summary_for_chunk(
                    chunk, level=0, span=(mid + start, mid + start + len(chunk)),
                    detailed=True,
                )
            )
        # старое -> выжимки: крупные чанки, короткие сводки
        cs2 = max(1, self.chunk_size * 2)
        for start in range(0, len(old_part), cs2):
            chunk = old_part[start: start + cs2]
            level0.append(
                self._summary_for_chunk(
                    chunk, level=0, span=(start, start + len(chunk)),
                    detailed=False,
                )
            )
        if not level0:
            return []

        all_nodes: List[MemoryNode] = list(level0)
        cur = level0
        level = 1
        while len(cur) > self.hierarchy_cap:
            parents: List[MemoryNode] = []
            for start in range(0, len(cur), self.hierarchy_cap):
                group = cur[start: start + self.hierarchy_cap]
                parents.append(self._summary_of_summaries(group, level=level))
            all_nodes.extend(parents)
            cur = parents
            level += 1
        return all_nodes

    def _summary_for_chunk(
        self,
        chunk: List[Dict[str, Any]],
        level: int,
        span: Tuple[int, int],
        detailed: bool,
    ) -> MemoryNode:
        """Сводка уровня 0 по чанку сообщений (детальная или выжимка)."""
        texts = [m["content"] for m in chunk]
        roles = sorted({m["role"] for m in chunk})
        facts = _extract_facts(texts)
        n = len(chunk)
        fact_line = "; ".join(facts) if facts else "явных числовых фактов нет"
        if detailed:
            headline = _first_sentence(" ".join(texts), limit=160)
            claim = (
                f"Сводка обсуждения ({n} сообщ., роли: {', '.join(roles)}): "
                f"ключевые факты: {fact_line}. {headline}"
            )
        else:
            # выжимка короче по многословию, но ФАКТЫ сохраняются все —
            # деталь из старой части восстанавливается поиском в Store
            claim = f"Выжимка ({n} сообщ.): ключевые факты: {fact_line}."
        meta = {
            "engine": "hierarchical_compactor",
            "level": level,
            "span": list(span),
            "detailed": detailed,
            "source_messages": [m["id"] for m in chunk],
        }
        return MemoryNode(
            claim=claim,
            kind="context_summary",
            source=f"context_engine.compactor/level{level}",
            evidence=[m["id"] for m in chunk],
            context=json.dumps(meta, ensure_ascii=False),
            links=[m["id"] for m in chunk],
            ts=self._now.isoformat(timespec="milliseconds"),
        )

    def _summary_of_summaries(
        self, group: List[MemoryNode], level: int
    ) -> MemoryNode:
        """Сводка уровня >0: сжимает дочерние сводки в одну."""
        child_facts = _extract_facts([nd.claim for nd in group])
        fact_line = "; ".join(child_facts[:6]) if child_facts else "—"
        claim = (
            f"Иерархическая сводка уровня {level}: объединяет {len(group)} "
            f"под-сводок сессии; ключевые факты: {fact_line}."
        )
        meta = {
            "engine": "hierarchical_compactor",
            "level": level,
            "span": [group[0].id, group[-1].id],
            "detailed": False,
            "child_summaries": [nd.id for nd in group],
        }
        return MemoryNode(
            claim=claim,
            kind="context_summary",
            source=f"context_engine.compactor/level{level}",
            evidence=[nd.id for nd in group],
            context=json.dumps(meta, ensure_ascii=False),
            links=[nd.id for nd in group],
            ts=self._now.isoformat(timespec="milliseconds"),
        )

    # -- гейты + выгрузка ------------------------------------------------------

    def _gate_node(self, node_dict: Dict[str, Any]) -> Dict[str, str]:
        """Хук гейтов перед выгрузкой: mnemos.gates.run_write_gates.

        Если модуль gates отсутствует/не импортируется — сводка пишется
        без фильтра (гейты опциональны, ничего не ломаем).
        """
        if not self.gates_enabled:
            return {"verdict": "pass", "reason": "гейты отключены"}
        try:
            from .gates import run_write_gates

            result = run_write_gates(
                node_dict,
                registry=self._registry,
                now=self._now,
                existing=self._existing,
            )
            return {"verdict": result.verdict, "reason": result.reason}
        except Exception as exc:  # gates сломаны/недоступны — не мешаем записи
            return {"verdict": "pass", "reason": f"гейты недоступны: {exc!r}"}

    def _store_nodes(self, nodes: List[MemoryNode]) -> List[Dict[str, Any]]:
        """Прогоняет сводки через гейты и складывает выжившие в Store.

        reject -> НЕ пишется (знание не теряется: сырые рефы уже в refs);
        flag   -> пишется, но с пометкой гейта в context;
        pass   -> пишется как есть.
        """
        stored: List[Dict[str, Any]] = []
        for node in nodes:
            d = node.to_dict()
            gate = self._gate_node(d)
            if gate["verdict"] == "reject":
                stored.append(
                    {
                        "type": "summary_rejected",
                        "claim": d["claim"],
                        "level": json.loads(d["context"]).get("level")
                        if isinstance(d.get("context"), str) else None,
                        "gate": "reject",
                        "reason": gate["reason"],
                    }
                )
                continue
            if gate["verdict"] == "flag":
                # философия «понизить, а не выбросить»: пометка в context
                try:
                    meta = json.loads(d["context"])
                except (TypeError, ValueError):
                    meta = {}
                meta["gate"] = {"verdict": "flag", "reason": gate["reason"]}
                d["context"] = json.dumps(meta, ensure_ascii=False)
                node = MemoryNode.from_dict(d)
            try:
                self.store.add(node)
            except Exception:  # прагматично: узел не сохранился — реф останется
                continue
            stored.append(
                {
                    "type": "summary",
                    "node_id": node.id,
                    "kind": "context_summary",
                    "level": self._node_level(node),
                    "gate": gate["verdict"],
                    "claim": node.claim,
                }
            )
        return stored

    @staticmethod
    def _node_level(node: MemoryNode) -> Optional[int]:
        try:
            return int(json.loads(node.context).get("level"))
        except (TypeError, ValueError, AttributeError):
            return None


# ============================================================================
# CanonicalPrefix — кэш-стабильный канонический префикс (DeepSeek on-disk)
# ============================================================================


class CanonicalPrefix:
    """Префикс = НЕИЗМЕННАЯ шапка + APPEND-ONLY хвост (кэш-стабильность).

    DeepSeek кэширует вход по префиксу (LRU, on disk): любой ход агента,
    который мутирует раннюю часть промпта, платит полную цену префилла.
    Канонический префикс превращает это в скидку −90…−95%:
      - шапка (system + стабильные инструкции) никогда не меняется;
      - память/ходы добавляются ТОЛЬКО в конец хвоста (append_tail);
      - check_stability() детектирует мутации, ломающие кэш.
    build_prefix() детерминирован: повторный вызов с теми же аргументами
    возвращает байт-в-байт тот же hash/text (условие кэш-хита).
    """

    _HEAD_MARK = "=== ГОЛОВА (неизменяемая) ==="
    _TAIL_MARK = "=== ХВОСТ (append-only) ==="

    def __init__(self) -> None:
        self._system: str = ""
        self._static: List[str] = []
        self._tail: List[str] = []

    # -- сборка ----------------------------------------------------------------

    def build_prefix(
        self,
        system: str = "",
        static_blocks: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Собирает канонический префикс: шапка + текущий append-only хвост.

        system        — системный промпт (первый блок шапки);
        static_blocks — стабильные инструкции (дальше в шапке);
        хвост         — берётся из состояния экземпляра (append_tail).

        Возвращает dict: hash, head_hash, tail_hash, system, static_blocks,
        tail_blocks, text, tokens. Идентичен при повторном вызове.
        """
        self._system = str(system or "")
        self._static = [str(b) for b in (static_blocks or [])]
        head = [self._system] + self._static
        tail = list(self._tail)

        head_hash = self._hash_of(head)
        tail_hash = self._hash_of(tail)
        text = self._render(head, tail)
        return {
            "hash": self._hash_of([head_hash, tail_hash]),
            "head_hash": head_hash,
            "tail_hash": tail_hash,
            "system": self._system,
            "static_blocks": self._static,
            "tail_blocks": tail,
            "text": text,
            "tokens": estimate_tokens(text),
        }

    def append_tail(self, block: str) -> None:
        """Единственная разрешённая мутация: добавить блок в конец хвоста."""
        self._tail.append(str(block))

    # -- стабильность ------------------------------------------------------------

    def check_stability(
        self,
        system: str = "",
        static_blocks: Optional[Sequence[str]] = None,
        tail_blocks: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Проверяет кандидата на кэш-стабильность относительно состояния.

        tail_blocks=None — проверить только шапку с текущим хвостом.
        Возвращает {stable, head_stable, tail_append_only, reasons}.
        """
        reasons: List[str] = []
        head_stable = True
        if str(system or "") != self._system:
            head_stable = False
            reasons.append(
                "ШАПКА ИЗМЕНЕНА: system промпт другой — префиксный кэш сломан"
            )
        if list(static_blocks or []) != self._static:
            head_stable = False
            reasons.append(
                "ШАПКА ИЗМЕНЕНА: static_blocks другие — префиксный кэш сломан"
            )

        tail = list(tail_blocks) if tail_blocks is not None else list(self._tail)
        append_only = tail[: len(self._tail)] == self._tail
        if not append_only:
            reasons.append(
                "ХВОСТ МУТИРОВАН: разрешено только добавление в конец "
                "(вставки/перестановки/удаления ломают кэш)"
            )
        elif len(tail) < len(self._tail):
            reasons.append("ХВОСТ УКОРОЧЕН: удаление блоков ломает кэш")

        return {
            "stable": head_stable and append_only,
            "head_stable": head_stable,
            "tail_append_only": append_only,
            "reasons": reasons,
        }

    # -- внутреннее ----------------------------------------------------------------

    @staticmethod
    def _hash_of(blocks: Sequence[Any]) -> str:
        """sha256 канонического JSON: детерминизм = кэш-стабильность."""
        payload = json.dumps(
            list(blocks), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _render(head: List[str], tail: List[str]) -> str:
        parts: List[str] = []
        if head:
            parts.append(CanonicalPrefix._HEAD_MARK)
            parts.extend(head)
        if tail:
            parts.append(CanonicalPrefix._TAIL_MARK)
            parts.extend(tail)
        return "\n\n".join(parts)


# ============================================================================
# ContextDefragmenter — «дефрагментация» сессии
# ============================================================================


class ContextDefragmenter:
    """Из старой сессии делает чистую новую с компактной сводкой-«памятью».

    defragment(session, store, keep_recent) -> новый стартовый контекст:
      [сводка-преамбула] + окно. Старая история сжимается Compactor'ом в
      Store (память), а в новый контекст кладётся только сводка со ссылками
      на узлы — деталь восстанавливается поиском (store.search, ~1 мкс).
    """

    def __init__(self, compactor: Optional[HierarchicalCompactor] = None) -> None:
        self.compactor = compactor or HierarchicalCompactor()
        self.prefix = CanonicalPrefix()

    def defragment(
        self,
        session: Sequence[Any],
        store: Optional[Any] = None,
        keep_recent: int = 8,
        system: str = "",
        static_blocks: Optional[Sequence[str]] = None,
        preamble_limit: int = 700,
    ) -> Dict[str, Any]:
        """Дефрагментирует сессию. Возвращает:

        new_context  — стартовый контекст новой сессии (преамбула + окно);
        old_tokens   — размер исходной сессии (приблизительно);
        new_tokens   — размер нового стартового контекста;
        saved_ratio  — доля сокращения (0..1);
        memory_refs  — что выгружено в память (рефы Compactor'а);
        preamble     — текст сводки-памяти.
        """
        if store is not None:
            self.compactor.store = store
        msgs = _norm_messages(session)
        window, refs = self.compactor.compact(msgs, keep_recent=keep_recent)

        preamble_text = self._build_preamble(refs, limit=preamble_limit)
        preamble_msg: Dict[str, Any] = {
            "id": "memory_preamble",
            "role": "system" if system else "user",
            "content": preamble_text,
            "ts": now_iso(),
        }
        new_context = [preamble_msg] + list(window)

        old_tokens = sum(estimate_tokens(m["content"]) for m in msgs)
        new_tokens = sum(estimate_tokens(m["content"]) for m in new_context)
        saved_ratio = (
            1.0 - (new_tokens / old_tokens) if old_tokens else 0.0
        )
        return {
            "new_context": new_context,
            "old_tokens": old_tokens,
            "new_tokens": new_tokens,
            "saved_ratio": round(saved_ratio, 4),
            "memory_refs": refs,
            "preamble": preamble_text,
            "prefix": self.prefix.build_prefix(system, static_blocks),
        }

    @staticmethod
    def _build_preamble(refs: List[Dict[str, Any]], limit: int = 700) -> str:
        """Сводка-«память» из рефов Compactor'а (компактный текст со ссылками).

        Кладёт в начало нового контекста ключевые факты + node_id сводок —
        агент видит память, а деталь подтягивает поиском из Store.
        """
        lines = ["Память из предыдущей сессии (детали — в ShineMnemos, поиск по фактам):"]
        used = len(lines[0])
        for r in refs:
            if r.get("type") != "summary" or not r.get("claim"):
                continue
            line = f"- {r['claim']}  [id={r['node_id']}]"
            if used + len(line) > limit:
                break
            lines.append(line)
            used += len(line)
        if len(lines) == 1:
            lines.append("- (в памяти нет узлов-сводок)")
        return "\n".join(lines)
