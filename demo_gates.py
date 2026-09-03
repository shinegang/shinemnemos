# -*- coding: utf-8 -*-
"""ShineMnemos: demo 5 новых гейтов качества памяти (исследование).

Запуск:  python demo_gates.py          (только stdlib, без внешних API)

Что показывает: на маленьком наборе примеров каждый из 5 гейтов
пропускает хорошую запись (✅ ПРОПУСК) и отсекает плохую (❌ ОТКЛОН),
либо пропускает с предупреждением (⚠️ ДОПУСК — понижение/пометка).

Пять гейтов (кандидаты поверх truth-gate П1-П6):

  Г1 CONFIDENCE-GATE   — «уверенность автора»: не пускать в память
                         зыбкие формулировки как факты.
  Г2 STALENESS-GATE    — «свежесть/устаревание»: volatile-темы
                         (курс/статус/версия…) стареют быстро; на чтении
                         — recency-конфликты и вес узла.
  Г3 SOURCE-TRUST-GATE — «доверие к источнику»: градация A/B/C +
                         наследование недоверия по ссылкам (усиление П2).
  Г4 CONSISTENCY-GATE  — «противоречия и дубликаты»: на записи сравнение
                         с памятью: дубликат → отсечь, противоречие с
                         подтверждённым фактом → отсечь (или явный refuted).
  Г5 RELEVANCE-GATE    — «релевантность чтения»: не засорять контекст
                         агента почти-не-по-теме узлами (порог сходства).

Каждая проверка — честная эвристика на stdlib (regex/лексиконы/
Jaccard-сходство токенов); в продакшене те же контракты, но с LLM/эмбеддингами
(см. gates_research.md). В конце — сводка и схема встраивания в pipeline.

Контракт гейта: gate(node: dict, ctx: dict) -> GateResult
  ctx: {"nodes": [...], "registry": {...}, "query": "...",
        "subjects": {...}, "threshold": float, "now": datetime}
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# --- консоль: UTF-8, чтобы эмодзи/русский не падали на Windows ---------------
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --- общие помощники -----------------------------------------------------------

STOPWORDS = {
    "что", "это", "как", "так", "вот", "ещё", "уже", "был", "была", "было",
    "были", "будет", "будут", "очень", "просто", "который", "которая",
    "которые", "такой", "такая", "такие", "сам", "сама", "сами", "всего",
    "только", "тоже", "даже", "здесь", "там", "тут", "потом", "потому",
    "поэтому", "например", "какой", "какая", "какие", "свой", "своя", "свои",
}


def _tokens(text: str) -> List[str]:
    """Нормализация: lower, пунктуация вон, стоп-слова вон.

    Слова >=3 симв.; числовые токены >=2 цифр сохраняются — для памяти
    фактов цифры (курс, версии, суммы) важнее слов."""
    t = re.sub(r"[^\w\s$%]", " ", text.lower())
    return [
        w for w in t.split()
        if (len(w) >= 3 or (len(w) >= 2 and w.isdigit())) and w not in STOPWORDS
    ]


def _jaccard(a: List[str], b: List[str]) -> float:
    """Jaccard-сходство наборов токенов (0..1)."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def _parse_ts(ts: Any) -> Optional[datetime]:
    """ISO-8601 -> datetime (UTC). None — не парсится."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _now(ctx: Optional[Dict[str, Any]]) -> datetime:
    return (ctx or {}).get("now") or datetime.now(timezone.utc)


@dataclass
class GateResult:
    """Результат гейта: pass | reject | flag + причина на человеческом языке."""

    verdict: str            # "pass" | "reject" | "flag"
    reason: str

    @property
    def ok(self) -> bool:
        return self.verdict == "pass"


def ok(reason: str) -> GateResult:
    return GateResult("pass", reason)


def reject(reason: str) -> GateResult:
    return GateResult("reject", reason)


def flag(reason: str) -> GateResult:
    return GateResult("flag", reason)


# ============================================================================
# Г1 CONFIDENCE-GATE — «уверенность автора записи»
# ============================================================================

HEDGE_MARKERS = (
    "наверное", "наверно", "кажется", "возможно", "вероятно", "похоже",
    "вроде", "вроде бы", "думаю", "полагаю", "не уверен", "не уверена",
    "вряд ли", "как будто", "якобы", "предположительно", "примерно", "может",
    "probably", "maybe", "i think", "i guess", "not sure", "possibly",
    "perhaps", "seems", "apparently", "supposedly",
)
CERTAINTY_MARKERS = (
    "точно", "факт", "подтверждено", "подтверждён", "гарантированно",
    "измерено", "проверено", "100%", "certainly", "definitely", "confirmed",
    "verified", "measured", "checked",
)

CONF_PASS = 0.50   # факт допустим
CONF_MIN = 0.30    # ниже — вон из фактов


def check_confidence(node: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> GateResult:
    """Г1: уверенность автора. Что фильтрует: «наверное-факты».

    Эвристика: явное поле confidence (0..1) — берём как есть; иначе база
    0.75, минус маркеры неуверенности (-> 0.35), минус отсутствие источника
    (-> 0.25), плюс маркеры подтверждения (-> 0.85).
    Факт с conf < 0.30 -> reject; 0.30..0.50 -> flag (понизить до hypothesis);
    hypothesis пропускается всегда (низкая уверенность — его природа).
    Вход: узел (claim, kind, source, confidence?). Выход: pass/reject/flag + причина.
    """
    claim = str(node.get("claim") or "")
    kind = str(node.get("kind") or "fact")
    src = str(node.get("source") or "")

    conf = node.get("confidence")
    if conf is None:
        conf = 0.75
        hedges = [h for h in HEDGE_MARKERS if h in claim.lower()]
        if hedges:
            conf = min(conf, 0.35)
        if any(m in claim.lower() for m in CERTAINTY_MARKERS):
            conf = max(conf, 0.85)
        if not src.strip():
            conf = min(conf, 0.25)
    else:
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0

    if kind == "hypothesis":
        return ok(
            f"kind=hypothesis — низкая уверенность допустима (conf={conf:.2f})"
        )
    if conf < CONF_MIN:
        detail = ""
        hedges = [h for h in HEDGE_MARKERS if h in claim.lower()]
        if hedges:
            detail = f"; маркеры неуверенности: {', '.join(hedges[:3])}"
        if not src.strip():
            detail += "; нет источника"
        return reject(
            f"уверенность {conf:.2f} < {CONF_MIN:.2f} — слишком зыбкое для факта{detail}"
        )
    if conf < CONF_PASS:
        return flag(
            f"уверенность {conf:.2f} в [{CONF_MIN:.2f}, {CONF_PASS:.2f}) — "
            f"сохранить как hypothesis, а не факт"
        )
    return ok(f"уверенность {conf:.2f} >= {CONF_PASS:.2f} — факт допустим")


# ============================================================================
# Г2 STALENESS-GATE — «свежесть / устаревание»
# ============================================================================

VOLATILE_LEXICON = (
    "курс", "цена", "ценник", "статус", "баланс", "версия", "погода",
    "онлайн", "offline", "доступ", "занят", "свободен", "аптайм", "uptime",
    "price", "status", "balance", "version", "online", "капитализация",
)
VOLATILE_MAX_AGE_DAYS = 7
STABLE_MAX_AGE_DAYS = 365


def check_staleness(node: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> GateResult:
    """Г2: свежесть/устаревание. Что фильтрует: устаревшие данные.

    На записи: у volatile-тем (курс/статус/версия…) лимит возраста 7 дней,
    у стабильных — 365 (расширение П1). На чтении: recency-конфликт —
    если по той же теме есть более свежая запись с другим содержанием,
    старый узел помечается outdated (flag); вес ниже порога тоже отсекается.
    Вход: узел (claim, ts), ctx.subjects {тема: {ts, claim}}. Выход: verdict+причина.
    """
    ts = node.get("ts")
    dt = _parse_ts(ts)
    if dt is None:
        return reject("ts отсутствует или не ISO-8601 — свежесть не проверить")
    now = _now(ctx)
    claim_l = str(node.get("claim") or "").lower()

    volatile = any(w in claim_l for w in VOLATILE_LEXICON)
    max_age = timedelta(days=VOLATILE_MAX_AGE_DAYS if volatile else STABLE_MAX_AGE_DAYS)
    age = now - dt
    if age > max_age:
        tag = "volatile-тема (курс/статус/версия…)" if volatile else "обычная тема"
        return reject(
            f"возраст {age.days} дн > лимита {max_age.days} дн для {tag} — "
            f"данные устарели"
        )

    # recency-конфликт: та же тема, более свежая запись с другим содержанием
    subject = " ".join(_tokens(str(node.get("claim") or ""))[:2])
    newer = (ctx or {}).get("subjects") or {}
    if subject and subject in newer:
        nv = newer[subject]
        nv_dt = _parse_ts(nv.get("ts"))
        if nv_dt and nv_dt > dt and str(nv.get("claim") or "") != str(node.get("claim") or ""):
            return flag(
                f"есть более свежая запись по теме «{subject}» "
                f"({str(nv.get('claim'))[:36]}…) — старый узел помечать как outdated"
            )

    kind_tag = "volatile-тема" if volatile else "стабильная тема"
    return ok(f"возраст {age.days} дн <= {max_age.days} дн ({kind_tag}) — свежо")


# ============================================================================
# Г3 SOURCE-TRUST-GATE — «доверие к источнику»
# ============================================================================

TIER_A_MARKERS = (
    "отчёт", "отчет", "офиц", "документ", "док.", "таблиц", "лог", "измер",
    "протокол", "спецификац", "данные сенсора", "метрики", "report",
    "document", "official", "log", "measured", "spec", "metrics", "api ",
)
TIER_C_MARKERS = (
    "слух", "кто-то", "говорят", "интернет", "chatgpt", "не помню", "в чате",
    "слышал", "сплетни", "не уверен", "rumor", "someone", "heard",
    "internet", "i don't remember", "где-то прочитал", "кажется из разговора",
)


def _source_tier(source: str) -> tuple:
    s = (source or "").lower().strip()
    if not s:
        return "C", "пустой источник"
    if any(m in s for m in TIER_A_MARKERS):
        return "A", f"доверенный класс: «{s[:44]}»"
    if any(m in s for m in TIER_C_MARKERS):
        return "C", f"непроверяемый класс: «{s[:44]}»"
    return "B", f"обычный источник: «{s[:44]}»"


def check_source_trust(node: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> GateResult:
    """Г3: доверие к источнику. Что фильтрует: факты «из слухов».

    Градация источника A (0.9) / B (0.6) / C (0.15). Факт из C -> reject
    (предложение — сохранить как hypothesis); hypothesis из C -> flag.
    Наследование недоверия: ссылка на факт из ненадёжного источника -> reject.
    Вход: узел (source, kind, links), ctx.registry {id: {kind, source}}.
    Выход: pass/reject/flag + причина. Усиливает П2 (там только «непустой»).
    """
    tier, why = _source_tier(str(node.get("source") or ""))
    kind = str(node.get("kind") or "fact")

    for link in node.get("links") or []:
        target = ((ctx or {}).get("registry") or {}).get(link)
        if isinstance(target, dict):
            ttier, _ = _source_tier(str(target.get("source") or ""))
            if ttier == "C" and str(target.get("kind") or "") == "fact":
                return reject(
                    f"ссылка {link} ведёт на факт из ненадёжного источника (C) — "
                    f"доверие по ссылкам не наследуется"
                )

    if tier == "C":
        if kind == "fact":
            return reject(
                f"факт из ненадёжного источника ({why}) — сохраните как hypothesis"
            )
        return flag(f"hypothesis из ненадёжного источника — допустимо ({why})")
    return ok(f"источник класса {tier}: {why}")


# ============================================================================
# Г4 CONSISTENCY-GATE — «противоречия и дубликаты»
# ============================================================================

DUP_THRESHOLD = 0.55          # Jaccard >= — это дубликат, а не новая память
CONTRADICTION_THRESHOLD = 0.30  # Jaccard >= и разная полярность — конфликт
NEG_PATTERNS = (
    "не ", "нет", "никогда", "отсутствует", "недоступн", "не работает",
    "не было", "не вырос", "упал", "сломан", "потерян", "no ", "not ",
    "never", "unavailable", "down", "offline", "failed", "без ",
)


def _has_negation(claim: str) -> bool:
    c = " " + (claim or "").lower() + " "
    return any(p in c for p in NEG_PATTERNS)


def check_consistency(node: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> GateResult:
    """Г4: противоречия и дубликаты. Что фильтрует: копии и конфликты с памятью.

    На записи сверяем новый claim со всеми узлами (лексический прокси
    семантики — Jaccard по токенам; в проде — эмбеддинги):
      * сходство >= 0.55          -> дубликат: reject (предложить reinforce);
      * сходство >= 0.30 и разная полярность (негация) -> конфликт:
          - существующий узел — подтверждённый факт (вес >= 0.5):
            reject, если новое не оформлено как kind=refuted + evidence;
            иначе pass — явное опровержение;
          - существующий узел слабый -> flag (снизить вес нового).
    Вход: узел (claim, kind, evidence), ctx.nodes [{id, claim, kind, weight}].
    Выход: pass/reject/flag + причина. Расширяет П4 (там только links->refuted).
    """
    claim = str(node.get("claim") or "")
    tn = _tokens(claim)
    existing = (ctx or {}).get("nodes") or []

    conflicts: List[tuple] = []
    for other in existing:
        if other.get("id") == node.get("id"):
            continue
        to = _tokens(str(other.get("claim") or ""))
        j = _jaccard(tn, to)
        neg_diff = _has_negation(claim) != _has_negation(str(other.get("claim") or ""))
        # дубликат — то же содержание, та же полярность
        if j >= DUP_THRESHOLD and not neg_diff:
            return reject(
                f"дубликат узла {other.get('id')} (сходство {j:.0%}, "
                f"«{str(other.get('claim'))[:30]}…») — подкрепите существующий узел, "
                f"а не плодите копию"
            )
        # противоречие — общая тема, но противоположная полярность
        if j >= CONTRADICTION_THRESHOLD and neg_diff:
            conflicts.append((other, j))

    for other, j in conflicts:
        weight = float(other.get("weight") or 1.0)
        if other.get("kind") == "fact" and weight >= 0.5:
            if node.get("kind") != "refuted" or not (node.get("evidence") or []):
                return reject(
                    f"противоречит подтверждённому факту {other.get('id')} "
                    f"(вес {weight:.2f}, сходство {j:.0%}) — для записи укажите "
                    f"kind=refuted + evidence"
                )
            return ok(
                f"оформлено как явное опровержение (kind=refuted + evidence) — "
                f"конфликт разрешён, узел {other.get('id')} станет refuted"
            )
        return flag(
            f"противоречие с {other.get('id')} (сходство {j:.0%}), но тот узел "
            f"слабый (вес {weight:.2f}) — вес нового узла снижается"
        )
    return ok(f"противоречий и дубликатов с {len(existing)} узлом(ами) не найдено")


# ============================================================================
# Г5 RELEVANCE-GATE — «релевантность чтения»
# ============================================================================

RELEVANCE_THRESHOLD = 2.0
CLAIM_HIT_WEIGHT = 3.0
CONTEXT_HIT_WEIGHT = 1.5
SOURCE_HIT_WEIGHT = 1.0


def check_relevance(node: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> GateResult:
    """Г5: релевантность на чтении. Что фильтрует: шум в контексте агента.

    Кандидат из поиска проходит порог: score = 3.0*попаданий токенов запроса
    в claim + 1.5*в context + 1.0*в source; ниже threshold — не отдавать
    агенту (память, которая не по теме, хуже отсутствия памяти).
    Лексический прокси косинуса: в проде с эмбеддингами порог по косинусу
    (~0.5), сюда же вешаются recency и вес узла (см. Г2).
    Вход: узел (claim, source, context), ctx.query, ctx.threshold.
    Выход: pass/reject + причина.
    """
    query = str((ctx or {}).get("query") or "")
    qt = _tokens(query)
    if not qt:
        return reject("пустой запрос — релевантность не определить")
    threshold = float((ctx or {}).get("threshold", RELEVANCE_THRESHOLD))

    claim_l = str(node.get("claim") or "").lower()
    ctx_l = str(node.get("context") or "").lower()
    src_l = str(node.get("source") or "").lower()

    score = (
        CLAIM_HIT_WEIGHT * sum(1 for t in qt if t in claim_l)
        + CONTEXT_HIT_WEIGHT * sum(1 for t in qt if t in ctx_l)
        + SOURCE_HIT_WEIGHT * sum(1 for t in qt if t in src_l)
    )
    if score < threshold:
        return reject(
            f"релевантность {score:.1f} < {threshold} по запросу «{query}» — "
            f"узел почти не по теме, не засорять контекст агента"
        )
    return ok(f"релевантность {score:.1f} >= {threshold} — узел идёт в контекст агента")


# ============================================================================
# Демо-набор
# ============================================================================

NOW = datetime(2026, 2, 10, 12, 0, 0, tzinfo=timezone.utc)


def _iso(**kw) -> str:
    """ISO из NOW со смещением: days/hours/minutes."""
    delta = timedelta(**kw)
    return (NOW + delta).isoformat(timespec="seconds")


# Общий контекст: «память» из 3 узлов + реестр источников по ссылкам.
EXISTING_NODES = [
    {"id": "mn_btc", "claim": "курс BTC 97 000", "kind": "fact", "weight": 0.9},
    {"id": "mn_srv", "claim": "Сервер работает стабильно", "kind": "fact", "weight": 0.9},
    {"id": "mn_arch", "claim": "Вчера обсудили архитектуру микросервисов",
     "kind": "hypothesis", "weight": 0.3},
]
REGISTRY = {
    "mn_hearsay": {"kind": "fact", "source": "слух с форума"},
    "mn_btc": {"kind": "fact", "source": "финотчёт, таблица 2"},
    "mn_srv": {"kind": "fact", "source": "мониторинг, лог-файл"},
}

CASES: List[Dict[str, Any]] = [
    # --- Г1 Confidence -------------------------------------------------------
    dict(gate=check_confidence, label="явная уверенность 0.90, kind=fact",
         node={"claim": "Релиз v2.4.1 выйдет 12 мая в 18:00 (подтверждено)",
               "kind": "fact", "confidence": 0.9,
               "source": "роадмап команды"},
         ctx={}, expected="pass"),
    dict(gate=check_confidence, label="явная уверенность 0.15, kind=fact",
         node={"claim": "Завтра будет обвал рынка", "kind": "fact",
               "confidence": 0.15, "source": "прогноз коллеги"},
         ctx={}, expected="reject"),
    dict(gate=check_confidence, label="«Кажется…» и нет источника, kind=fact",
         node={"claim": "Кажется, вчера деплой прошёл успешно",
               "kind": "fact", "source": ""},
         ctx={}, expected="reject"),
    dict(gate=check_confidence, label="«Возможно…» но kind=hypothesis",
         node={"claim": "Возможно, перейдём на PostgreSQL 17", "kind": "hypothesis",
               "source": "обсуждение в команде"},
         ctx={}, expected="pass"),
    # --- Г2 Staleness --------------------------------------------------------
    dict(gate=check_staleness, label="volatile-тема, возраст 1 мин",
         node={"claim": "Статус сервера: online (проверка 2 мин назад)",
               "ts": _iso(minutes=-1)},
         ctx={}, expected="pass"),
    dict(gate=check_staleness, label="стабильная тема, возраст 100 дн",
         node={"claim": "Архитектура: PostgreSQL 16 как основная БД",
               "ts": _iso(days=-100)},
         ctx={}, expected="pass"),
    dict(gate=check_staleness, label="volatile-тема, возраст 400 дн",
         node={"claim": "Статус сервера: online", "ts": _iso(days=-400)},
         ctx={}, expected="reject"),
    dict(gate=check_staleness, label="recency-конфликт: есть более свежая запись",
         node={"claim": "Статус сервера: online", "ts": _iso(days=-6)},
         ctx={"subjects": {"статус сервера": {"ts": _iso(days=-1),
                                              "claim": "Статус сервера: degraded после сбоя"}}},
         expected="flag"),
    # --- Г3 Source/Trust -----------------------------------------------------
    dict(gate=check_source_trust, label="доверенный источник, kind=fact",
         node={"claim": "Выручка выросла на 12% в Q3 2025",
               "source": "финотчёт за Q3, стр. 12, таблица 2", "kind": "fact"},
         ctx={}, expected="pass"),
    dict(gate=check_source_trust, label="«кто-то сказал в чате», kind=fact",
         node={"claim": "Конкурент подал на банкротство",
               "source": "кто-то сказал в чате", "kind": "fact"},
         ctx={}, expected="reject"),
    dict(gate=check_source_trust, label="тот же слух, но kind=hypothesis",
         node={"claim": "Конкурент может подать на банкротство",
               "source": "кто-то сказал в чате", "kind": "hypothesis"},
         ctx={}, expected="flag"),
    dict(gate=check_source_trust, label="ссылка на факт из ненадёжного источника",
         node={"claim": "Сумма долга конкурента — 2 млрд", "kind": "fact",
               "source": "официальный отчёт", "links": ["mn_hearsay"]},
         ctx={"registry": REGISTRY}, expected="reject"),
    # --- Г4 Consistency ------------------------------------------------------
    dict(gate=check_consistency, label="новый факт, конфликтов нет",
         node={"claim": "Бэкенд мигрирован на Python 3.12", "kind": "fact"},
         ctx={"nodes": EXISTING_NODES}, expected="pass"),
    dict(gate=check_consistency, label="дубликат существующего узла",
         node={"claim": "BTC стоит 97 000", "kind": "fact"},
         ctx={"nodes": EXISTING_NODES}, expected="reject"),
    dict(gate=check_consistency, label="противоречие подтверждённому факту",
         node={"claim": "Сервер не работает", "kind": "fact"},
         ctx={"nodes": EXISTING_NODES}, expected="reject"),
    dict(gate=check_consistency, label="противоречие, но явный refuted + evidence",
         node={"claim": "Сервер не работает", "kind": "refuted",
               "evidence": ["логи: 503 с 12:40"]},
         ctx={"nodes": EXISTING_NODES}, expected="pass"),
    # --- Г5 Relevance --------------------------------------------------------
    dict(gate=check_relevance, label="запрос «курс BTC»: попадание в claim",
         node={"claim": "Курс BTC: 97 000$ на 12:00", "source": "биржа",
               "context": "утренний обзор"},
         ctx={"query": "курс BTC"}, expected="pass"),
    dict(gate=check_relevance, label="запрос «курс BTC»: токен только в context",
         node={"claim": "Вчера гуляли по Москве", "source": "личное",
               "context": "обсуждали отпуск, курс валют ни к чему"},
         ctx={"query": "курс BTC"}, expected="reject"),
    dict(gate=check_relevance, label="запрос «курс BTC»: вообще не по теме",
         node={"claim": "Погода в Москве: +15°C", "source": "метео",
               "context": "прогноз на неделю"},
         ctx={"query": "курс BTC"}, expected="reject"),
]

GATE_META = {
    "check_confidence": ("Г1 CONFIDENCE-GATE", "уверенность автора записи"),
    "check_staleness": ("Г2 STALENESS-GATE", "свежесть / устаревание (чтение+запись)"),
    "check_source_trust": ("Г3 SOURCE-TRUST-GATE", "доверие к источнику (усиление П2)"),
    "check_consistency": ("Г4 CONSISTENCY-GATE", "противоречия и дубликаты (усиление П4)"),
    "check_relevance": ("Г5 RELEVANCE-GATE", "релевантность на чтении"),
}

ICON = {"pass": "✅", "reject": "❌", "flag": "⚠️"}
LABEL = {"pass": "ПРОПУСК", "reject": "ОТКЛОН", "flag": "ДОПУСК"}


def run_case(case: Dict[str, Any]) -> bool:
    """Прогоняет один кейс, печатает строку. Возвращает True — ожидание совпало."""
    gate_fn = case["gate"]
    # детерминизм: фиксированное «сейчас» для всех гейтов (часы среды не важны)
    ctx = dict(case.get("ctx") or {})
    ctx.setdefault("now", NOW)
    res = gate_fn(case["node"], ctx)
    ok_expected = res.verdict == case["expected"]
    mark = "  OK" if ok_expected else "  !! НЕ СОВПАЛО С ОЖИДАНИЕМ"
    print(f"  [{case['label']}]")
    print(f"     {ICON[res.verdict]} {LABEL[res.verdict]} — {res.reason}{mark}")
    return ok_expected


def main() -> int:
    print("=" * 78)
    print(" ShineMnemos · demo 5 новых гейтов качества памяти (поверх truth-gate П1-П6)")
    print("=" * 78)

    totals = {"pass": 0, "reject": 0, "flag": 0}
    mismatches = 0
    total = 0
    current_gate = None

    for case in CASES:
        gate_name = case["gate"].__name__
        if gate_name != current_gate:
            current_gate = gate_name
            title, desc = GATE_META[gate_name]
            print()
            print("-" * 78)
            print(f" {title} — {desc}")
            print("-" * 78)
        totals[case["expected"]] += 1
        total += 1
        if not run_case(case):
            mismatches += 1

    print()
    print("=" * 78)
    print(f" ИТОГ: кейсов {total} | ожидаемых ПРОПУСК: {totals['pass']}, "
          f"ОТКЛОН: {totals['reject']}, ДОПУСК(флаг): {totals['flag']}")
    print(f" Совпало с ожиданием: {total - mismatches}/{total} "
          + ("✅ демо зелёное" if mismatches == 0 else "❌ есть расхождения"))
    print()
    print(" Схема встраивания в pipeline Mnemos:")
    print("   ЗАПИСЬ  memory_add(claim,…)")
    print("        -> Г1 Уверенность -> Г3 Доверие к источнику")
    print("        -> Г4 Противоречия/дубликаты -> Г2 Свежесть")
    print("        -> truth-gate П1-П6 -> store.add(nodes.json)")
    print("   ЧТЕНИЕ  memory_search(query)")
    print("        -> store.search/search_semantic")
    print("        -> Г2 Свежесть (вес/outdated) -> Г5 Релевантность (top-k)")
    print("        -> контекст агента")
    print("=" * 78)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
