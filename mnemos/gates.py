# -*- coding: utf-8 -*-
"""ShineMnemos: гейты качества памяти (Г1-Г5), импортируемый модуль.

Контракт — из gates_research.md, §7. Пять гейтов поверх truth-gate П1-П6:

  Г1 CONFIDENCE-GATE   — «уверенность автора»: зыбкие формулировки не
                         попадают в память как факты (verimem: abstention).
  Г2 STALENESS-GATE    — «свежесть»: volatile-темы (курс/статус/версия…)
                         стареют за 7 дней vs 365 для стабильных; на чтении
                         — recency-конфликты и вес узла.
  Г3 SOURCE-TRUST-GATE — «доверие к источнику»: градация A/B/C + отказ в
                         наследовании доверия по ссылкам (усиление П2).
  Г4 CONSISTENCY-GATE  — «противоречия и дубликаты»: Jaccard-сходство с
                         памятью: дубликат → reject; конфликт с фактом
                         веса >= 0.5 → reject (кроме явного kind=refuted).
  Г5 RELEVANCE-GATE    — «релевантность на чтении»: шум не идёт в контекст
                         агента (порог по покрытию токенов запроса).

Пайплайн (как в research):
  ЗАПИСЬ  run_write_gates(node, registry, now)  — Г1 -> Г3 -> Г4 -> Г2
  ЧТЕНИЕ  run_read_gates(candidates, query)     — Г2 -> Г5

Философия: жёсткий reject только когда запись вредит памяти; в спорных
случаях — flag («понизить статус»), а не выбросить знание.

Только stdlib. Логика идентична demo_gates.py (19/19 кейсов зелёные).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# --- общие помощники -----------------------------------------------------------

STOPWORDS = {
    "что", "это", "как", "так", "вот", "ещё", "уже", "был", "была", "было",
    "были", "будет", "будут", "очень", "просто", "который", "которая",
    "которые", "такой", "такая", "такие", "сам", "сама", "сами", "всего",
    "только", "тоже", "даже", "здесь", "там", "тут", "потом", "потому",
    "поэтому", "например", "какой", "какая", "какие", "свой", "своя", "свои",
}

_RE_NONWORD = re.compile(r"[^\w\s$%]")
_RE_DIGIT = re.compile(r"\d")


def _tokens(text: str) -> List[str]:
    """Нормализация: lower, пунктуация вон, стоп-слова вон.

    Слова >=3 симв.; числовые токены >=2 цифр сохраняются — для памяти
    фактов цифры (курс, версии, суммы) важнее слов."""
    t = _RE_NONWORD.sub(" ", str(text or "").lower())
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

    На записи: у volatile-тем лимит возраста 7 дней, у стабильных — 365
    (расширение П1). На чтении: recency-конфликт — если по той же теме есть
    более свежая запись с другим содержанием, старый узел помечается
    outdated (flag); вес ниже порога тоже отсекается.
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

    Градация источника A (0.9) / B (0.6) / C (0.15). Факт из C -> reject;
    hypothesis из C -> flag. Наследование недоверия: ссылка на факт из
    ненадёжного источника -> reject. Усиливает П2 (там только «непустой»).
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

DUP_THRESHOLD = 0.55            # Jaccard >= — это дубликат, а не новая память
CONTRADICTION_THRESHOLD = 0.30  # Jaccard >= и разная полярность — конфликт
NEG_PATTERNS = (
    "не ", "нет", "никогда", "отсутствует", "недоступн", "не работает",
    "не было", "не вырос", "упал", "сломан", "потерян", "no ", "not ",
    "never", "unavailable", "down", "offline", "failed", "без ",
)


def _has_negation(claim: str) -> bool:
    c = " " + str(claim or "").lower() + " "
    return any(p in c for p in NEG_PATTERNS)


# -- значимые различия (фикс 01.09) -----------------------------------------
# Jaccard по словам не отличает «премию я бы хотел…» ×208 (настоящий дубль)
# от «CRV long -> reject» / «SKR long -> approve» (разные решения, общий
# шаблон). На коротких claim'ах шаблон даёт сходство 0.67-0.90, и голый порог
# 0.55 зарубил бы ВСЕ различающиеся только числом или тикером факты —
# проверено на tests/test_server.py::test_memory_search_topk_limit, где
# 5 разных замеров латентности схлопнулись в 1.
# Поэтому дубликат = высокое сходство И совпадающий «значимый набор»:
# числа и тикеры/аббревиатуры. Даты и время из набора исключены осознанно:
# один и тот же вердикт по той же монете, повторённый через час, — это
# подкрепление, а не новая память (ровно то, что просит Г4 в тексте reject).
_RE_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b")
_RE_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_RE_TICKER = re.compile(r"\b[A-ZА-Я][A-ZА-Я0-9]{1,11}\b")


# Слова-решения: смена вердикта по тому же предмету — это НОВАЯ память,
# а не копия. Без них «CRV long -> approve» схлопывалось с «CRV long ->
# reject» (одинаковые числа и тикеры, высокое сходство), и агент читал бы
# устаревшее решение как актуальное.
DECISION_MARKERS = (
    "approve", "reject", "одобр", "отклон", "да", "нет", "long", "short",
    "buy", "sell", "лонг", "шорт", "покуп", "прода", "pass", "fail",
    "вход", "выход", "халт", "halt", "стоп", "hold",
)


def _salient(claim: str) -> set:
    """«О чём именно» утверждение: числа, тикеры и принятые решения.

    Даты и время исключены: повтор того же вердикта через час — это
    подкрепление, а не новая память.
    """
    text = _RE_DATE.sub(" ", str(claim or ""))
    nums = {n.replace(",", ".").rstrip(".0") or "0" for n in _RE_NUMBER.findall(text)}
    low = text.lower()
    decisions = {m for m in DECISION_MARKERS if m in low}
    return nums | set(_RE_TICKER.findall(text)) | decisions


def _content_tokens(claim: str) -> List[str]:
    """Токены claim'а БЕЗ дат и времени — для сверки содержания (Г4).

    Алиса зашивает timestamp в сам текст («Вердикт Алиса 2026-09-01
    02:05:23: CRV long -> reject»). Дата — метаданные узла (поле ts), а не
    содержание, но в Jaccard она давала до 6 различающихся токенов и роняла
    сходство одинаковых вердиктов с 0.9 до 0.33 — дубликаты проскакивали.
    """
    return _tokens(_RE_DATE.sub(" ", str(claim or "")))


def check_consistency(node: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> GateResult:
    """Г4: противоречия и дубликаты. Что фильтрует: копии и конфликты с памятью.

    Сверка нового claim со всеми узлами (Jaccard — лексический прокси
    семантики; в проде — эмбеддинги): сходство >= 0.55 и та же полярность
    -> дубликат: reject; сходство >= 0.30 и разная полярность -> конфликт
    (против факта веса >= 0.5 — reject, кроме явного kind=refuted+evidence;
    против слабого узла — flag). Расширяет П4 (там только links->refuted).
    """
    claim = str(node.get("claim") or "")
    tn = _content_tokens(claim)
    sn = _salient(claim)
    existing = (ctx or {}).get("nodes") or []

    conflicts: List[tuple] = []
    for other in existing:
        if other.get("id") == node.get("id"):
            continue
        other_claim = str(other.get("claim") or "")
        to = _content_tokens(other_claim)
        j = _jaccard(tn, to)
        neg_diff = _has_negation(claim) != _has_negation(other_claim)
        # дубликат — то же содержание, та же полярность И те же числа/тикеры
        if j >= DUP_THRESHOLD and not neg_diff and sn == _salient(other_claim):
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

    score = 3.0*попаданий токенов запроса в claim + 1.5*в context +
    1.0*в source; ниже порога — не отдавать агенту. Лексический прокси
    косинуса (в проде с эмбеддингами — прямой порог по косинусу ~0.5).
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
# Пайплайны: запись и чтение (gates_research.md, §7)
# ============================================================================


def run_write_gates(
    node: Dict[str, Any],
    registry: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
    existing: Optional[List[Dict[str, Any]]] = None,
) -> GateResult:
    """Полный проход гейтов записи: Г1 -> Г3 -> Г4 -> Г2 (порядок из research).

    - registry: id -> узел (для наследования доверия по ссылкам, Г3);
    - existing: список узлов памяти для сверки на дубликаты/противоречия (Г4);
      по умолчанию — значения registry;
    - now: фиксированное «сейчас» (детерминизм тестов/гейтов).

    Возвращает первый непройденный результат (reject/flag) или pass.
    """
    ctx: Dict[str, Any] = {}
    if now is not None:
        ctx["now"] = now
    res = check_confidence(node, ctx)
    if res.verdict != "pass":
        return res
    res = check_source_trust(node, {"registry": registry or {}, **ctx})
    if res.verdict != "pass":
        return res
    res = check_consistency(
        node,
        {"nodes": existing if existing is not None else list((registry or {}).values())},
    )
    if res.verdict != "pass":
        return res
    return check_staleness(node, ctx)


def run_read_gates(
    candidates: List[Dict[str, Any]],
    query: str,
    threshold: Optional[float] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Полный проход гейтов чтения: Г2 (свежесть) -> Г5 (релевантность).

    Возвращает кандидатов, достойных контекста агента (reject — вон).
    """
    out: List[Dict[str, Any]] = []
    for node in candidates:
        ctx: Dict[str, Any] = {"query": query}
        if now is not None:
            ctx["now"] = now
        res = check_staleness(node, ctx)
        if res.verdict == "reject":
            continue
        rctx: Dict[str, Any] = {"query": query}
        if threshold is not None:
            rctx["threshold"] = threshold
        if now is not None:
            rctx["now"] = now
        if check_relevance(node, rctx).verdict == "reject":
            continue
        out.append(node)
    return out
