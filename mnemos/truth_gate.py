# -*- coding: utf-8 -*-
"""ShineMnemos: truth-gate — протокол П1-П6.

Проверка утверждения перед тем, как память «поверит» ему (и пометит
verdict в truth_check узла). Шесть проверок-заглушек с внятной логикой:

  П1 Свежесть         — ts валидный ISO-8601, не из будущего (skew ≤ 5 мин),
                         не старше max_age_days (по умолчанию 365).
  П2 Источник         — поле source заполнено (непустая строка).
  П3 Цифры            — в claim есть числа (regex \\d).
  П4 Непротиворечивость — links не ссылаются на узел со статусом refuted
                         (нужен registry: id -> узел/kind; без registry
                         проверяется только валидность ссылок).
  П5 Воспроизводимость — evidence непустой (есть свидетельство, по которому
                         утверждение можно перепроверить).
  П6 Полнота          — поле context заполнено (утверждение дано в контексте).

Вердикт: pass, если прошло >= 4 из 6 (порог PASS_THRESHOLD), иначе fail.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from .model import MemoryNode

PASS_THRESHOLD = 4
DEFAULT_MAX_AGE_DAYS = 365
DEFAULT_FUTURE_SKEW_SECONDS = 300  # 5 минут допуска на рассинхрон часов

_DIGIT_RE = re.compile(r"\d")


class TruthResult:
    """Результат прохода truth-gate: вердикт, score, пояснения, детали по П1-П6."""

    __slots__ = ("verdict", "score", "notes", "checks")

    def __init__(
        self,
        verdict: str,
        score: int,
        notes: List[str],
        checks: Dict[str, Dict[str, Any]],
    ) -> None:
        self.verdict = verdict
        self.score = score
        self.notes = notes
        self.checks = checks

    def as_dict(self) -> Dict[str, Any]:
        return {
            "P1": self.checks["P1"],
            "P2": self.checks["P2"],
            "P3": self.checks["P3"],
            "P4": self.checks["P4"],
            "P5": self.checks["P5"],
            "P6": self.checks["P6"],
            "verdict": self.verdict,
            "score": self.score,
            "summary": f"{self.score}/6 проверок пройдено — вердикт: {self.verdict}",
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<TruthResult verdict={self.verdict!r} score={self.score}/6>"


# Registry: id узла -> узел (dict|MemoryNode) | kind (str) | None (не найден).
Registry = Optional[Union[Mapping[str, Any], Callable[[str], Optional[Any]]]]


def _parse_ts(ts: str) -> Optional[datetime]:
    """Парсит ISO-8601; naive-время трактуется как UTC. None — если не парсится."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _kind_of(registry: Registry, link_id: str) -> Optional[str]:
    """Статус (kind) узла-цели ссылки по registry."""
    if registry is None:
        return None
    try:
        target = registry(link_id) if callable(registry) else registry.get(link_id)
    except Exception:  # registry может кинуть — считаем цель неизвестной
        return None
    if target is None:
        return None
    if isinstance(target, MemoryNode):
        return target.kind
    if isinstance(target, dict):
        return target.get("kind")
    return str(target)


# --- отдельные проверки ------------------------------------------------------

def check_freshness(
    node: Dict[str, Any],
    now: Optional[datetime] = None,
    max_age_days: Optional[int] = DEFAULT_MAX_AGE_DAYS,
    future_skew_seconds: int = DEFAULT_FUTURE_SKEW_SECONDS,
) -> Dict[str, Any]:
    """П1 Свежесть: ts валиден, не из будущего, не старше max_age_days."""
    ts = node.get("ts")
    dt = _parse_ts(ts) if ts else None
    if dt is None:
        return {"pass": False, "note": "П1 свежесть: ts отсутствует или не ISO-8601"}
    now = now if now is not None else datetime.now(timezone.utc)
    age = now - dt
    if age < timedelta(seconds=-future_skew_seconds):
        return {
            "pass": False,
            "note": f"П1 свежесть: ts из будущего ({ts}) — рассинхрон часов?",
        }
    if max_age_days is not None and age > timedelta(days=max_age_days):
        return {
            "pass": False,
            "note": f"П1 свежесть: узел старше {max_age_days} дн (ts={ts}) — устарел",
        }
    return {"pass": True, "note": f"П1 свежесть: ts={ts} в допуске"}


def check_source(node: Dict[str, Any]) -> Dict[str, Any]:
    """П2 Источник: поле source заполнено."""
    src = node.get("source")
    if isinstance(src, str) and src.strip():
        return {"pass": True, "note": f"П2 источник: {src.strip()[:60]}"}
    return {"pass": False, "note": "П2 источник: поле source пустое — неизвестно, кто сказал"}


def check_numbers(node: Dict[str, Any]) -> Dict[str, Any]:
    """П3 Цифры: в claim есть числа (проверяемость количественных утверждений)."""
    claim = node.get("claim") or ""
    if _DIGIT_RE.search(claim):
        found = _DIGIT_RE.findall(claim)
        return {"pass": True, "note": f"П3 цифры: в claim найдены числа: {', '.join(found[:5])}"}
    return {"pass": False, "note": "П3 цифры: в claim нет чисел — утверждение не количественное"}


def check_consistency(node: Dict[str, Any], registry: Registry = None) -> Dict[str, Any]:
    """П4 Непротиворечивость: links не ссылаются на refuted-узлы."""
    links = node.get("links") or []
    if not links:
        return {"pass": True, "note": "П4 непротиворечивость: ссылок нет — конфликтов нет"}
    for link in links:
        if not isinstance(link, str) or not link.strip():
            return {"pass": False, "note": f"П4 непротиворечивость: битая ссылка {link!r}"}
        kind = _kind_of(registry, link)
        if kind == "refuted":
            return {
                "pass": False,
                "note": f"П4 непротиворечивость: ссылка {link} ведёт на опровергнутый узел (refuted)",
            }
    return {"pass": True, "note": f"П4 непротиворечивость: {len(links)} ссылок, refuted-целей нет"}


def check_reproducibility(node: Dict[str, Any]) -> Dict[str, Any]:
    """П5 Воспроизводимость: evidence непустой."""
    ev = node.get("evidence") or []
    non_empty = [e for e in ev if isinstance(e, str) and e.strip()]
    if non_empty:
        return {
            "pass": True,
            "note": f"П5 воспроизводимость: {len(non_empty)} свидетельств(а), можно перепроверить",
        }
    return {"pass": False, "note": "П5 воспроизводимость: evidence пуст — нечем подтвердить"}


def check_completeness(node: Dict[str, Any]) -> Dict[str, Any]:
    """П6 Полнота: поле context заполнено."""
    ctx = node.get("context")
    if isinstance(ctx, str) and ctx.strip():
        return {"pass": True, "note": f"П6 полнота: контекст задан ({len(ctx.strip())} симв.)"}
    return {"pass": False, "note": "П6 полнота: поле context пустое — утверждение без контекста"}


# --- главная функция ----------------------------------------------------------

def check_claim(
    node: Union[MemoryNode, Dict[str, Any]],
    registry: Registry = None,
    now: Optional[datetime] = None,
    max_age_days: Optional[int] = DEFAULT_MAX_AGE_DAYS,
    future_skew_seconds: int = DEFAULT_FUTURE_SKEW_SECONDS,
) -> TruthResult:
    """Прогоняет узел через П1-П6.

    registry — карта id -> узел (или callable id -> узел), нужен для П4.
    Возвращает TruthResult: verdict ('pass' при score >= 4), score, notes,
    детали по каждой проверке.
    """
    if node is None:
        raise ValueError("check_claim: узел не может быть None")
    d = node.to_dict() if isinstance(node, MemoryNode) else dict(node)

    checks: Dict[str, Dict[str, Any]] = {
        "P1": check_freshness(d, now=now, max_age_days=max_age_days,
                              future_skew_seconds=future_skew_seconds),
        "P2": check_source(d),
        "P3": check_numbers(d),
        "P4": check_consistency(d, registry=registry),
        "P5": check_reproducibility(d),
        "P6": check_completeness(d),
    }
    score = sum(1 for c in checks.values() if c["pass"])
    verdict = "pass" if score >= PASS_THRESHOLD else "fail"
    notes = [c["note"] for c in checks.values()]
    return TruthResult(verdict=verdict, score=score, notes=notes, checks=checks)


def check_and_update(
    node: Union[MemoryNode, Dict[str, Any]],
    registry: Registry = None,
    now: Optional[datetime] = None,
    max_age_days: Optional[int] = DEFAULT_MAX_AGE_DAYS,
    future_skew_seconds: int = DEFAULT_FUTURE_SKEW_SECONDS,
) -> TruthResult:
    """check_claim + запись результата в truth_check узла (мутирует узел)."""
    result = check_claim(
        node, registry=registry, now=now,
        max_age_days=max_age_days, future_skew_seconds=future_skew_seconds,
    )
    payload = result.as_dict()
    if isinstance(node, MemoryNode):
        node.truth_check = payload
    else:
        node["truth_check"] = payload
    return result
