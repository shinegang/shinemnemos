# -*- coding: utf-8 -*-
"""ShineMnemos: рефакторинг по письму Qwen (02.09.2026) — библиотечная часть.

Что здесь (stdlib only, без внешних зависимостей):
  * онтология сущностей ENTITIES (та же, что в optimize/scripts/mnemos_autolink.py)
    и теги-маршрутизаторы ROUTERS: sys_cmd / persona_def / world_state;
  * типы рёбер RELS: related_to / part_of / has_part / conflicts_with /
    supersedes / duplicate_of / refers_to (паспорт ребра link_meta[to].rel);
  * BudgetSearch — token-budgeting поиск: top_k по сложности запроса
    (простой 5 / средний 10 / сложный 20), токенный бюджет, граф-расширение
    по рёбрам, хабы (kind=hub) отдельно от ответов;
  * build_system_prompt — сборка system-prompt из найденных узлов по секциям
    роутеров (правила первыми и никогда не режутся), обёртки chatml/llama3;
  * Graph — граф-запросы: neighbors / path / hub / rules_for / conflicts.

Замер 02.09 на копии VPS-стора (49 узлов, GT 15 запросов режима decisions):
  Store.search по ключевым словам   recall@5 0.928 (бенч 01.09 на 36 узлах: 0.944)
  BudgetSearch по ключевым словам   recall@5 0.944
  Store.search по NL-вопросам       recall@5 0.000
  BudgetSearch по NL-вопросам       recall@5 0.767 (0.661 без автолинковки/хабов)
Прототип и отчёт: /opt/bench-memory/mnemos_refactor/.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .model import DEFAULT_HALF_LIFE_HOURS, WEIGHT_MIN, _parse_iso, weight_floor

# --- онтология сущностей (маркеры — подстроки по нормализованному тексту) -------
ENTITIES: Dict[str, Tuple[str, ...]] = {
    # ревью 02.09 (Ф1, п.7): сняты шумные маркеры — «лимит»/«защит»/«стоп»
    # (ловили «лимиты обновлены» и «защита loopback»), «usdc» (CRV/USDC:USDC),
    # «$»/«usd» (usd внутри usdc), «узел/узла/узлов» (в сторе памяти — везде),
    # «5090» (любое упоминание машины), «тест»/«прогон» (подстроки).
    # Та же онтология, что в /opt/bench-memory/optimize/scripts/mnemos_autolink.py — править оба.
    "алиса":     ("алиса", "alice", "robin", "llama-70b", "llama-3.3-70b"),
    "шорты":      ("шорт", "short_enabled", "llm_allow_shorts", "short "),
    "hl":         ("hyperliquid", "hl_", "usdc:usdc", " hl ", "hl-"),
    "mnemos":     ("mnemos", "мнемос", "shinemnemos", "memory_add", "memory_search",
                   "memory_stats", "memory_link"),
    "иван":       ("иван", "ивана", "иваном", "приказ владельца", "владелец"),
    "переезд":    ("переезд", "миграц", "катовер", "переехать", "vast", "3090"),
    "x402":       ("x402", "eip-3009", "eip-712", "payto", " 402 ", "402 payment", "http 402"),
    "memory_api": ("memory-api", "memory_api", "/memory/search", "api402x"),
    "правило1":   ("правило 1", "чек-5", "обходные пути", "в1 ", "в2 ", "в5 "),
    "защиты":     ("halt", "equity-floor", "daily_halt", "защиты сняты", "защиты действуют",
                   "стоп-конституц", "лимиты сняты", "серии убытков", "стоп после"),
    "акме1":      ("акме-1", "акме-1", "оркестратор"),
    "узел-2":      ("узел-2", "vps1", "10.0.0.2"),
    "акме3":      ("акме-3", "акме-3", "мак ", "мака"),
    "граф":       ("граф", "ребр", "сирот", "edges", "orphan", "хаб"),
    "evidence":   ("evidence", "пруф", "truth-gate", "truth_check", "п1-п6",
                   "доказательств"),
    "торговля":   ("сделк", "ордер", "позици", "trades_pnl", "winrate", "pnl",
                   "депозит", "экзекьютор", "торг"),
    "деньги":     ("mrr", "клиент", "подписк", "платящ", "цена", "микроплатеж",
                   "микро-платеж", "выручк", "доход"),
    "тесты":      ("pytest", "бэктест", "backtest", "a/b", "тесты зелён", "тестов"),
    "панель":     ("панель", "дашборд", "dashboard", "панел"),
}

STOP_WORDS = {
    "который", "которая", "которое", "которые", "этот", "эта", "это", "эти",
    "было", "были", "будет", "может", "можно", "нужно", "только", "после",
    "перед", "через", "если", "чтобы", "также", "тоже", "ещё", "еще", "как",
    "что", "при", "для", "или", "над", "под", "уже", "все", "всех", "него",
    "нет", "без", "there", "with", "from", "that", "this", "have",
    "контекст", "основание", "вывод", "правило", "приказ",
}

# --- теги-маршрутизаторы ------------------------------------------------------
ROUTER_SYS_CMD = "sys_cmd"
ROUTER_PERSONA = "persona_def"
ROUTER_WORLD = "world_state"
ROUTERS = (ROUTER_SYS_CMD, ROUTER_PERSONA, ROUTER_WORLD)
TAG_HUB = "hub"
KIND_HUB = "hub"

# --- типы рёбер -----------------------------------------------------------------
REL_RELATED = "related_to"
REL_PART_OF = "part_of"
REL_HAS_PART = "has_part"
REL_CONFLICTS = "conflicts_with"
REL_SUPERSEDES = "supersedes"
REL_DUPLICATE = "duplicate_of"
REL_REFERS = "refers_to"
RELS = (REL_RELATED, REL_PART_OF, REL_HAS_PART, REL_CONFLICTS,
        REL_SUPERSEDES, REL_DUPLICATE, REL_REFERS)

# --- бюджеты по сложности запроса ---------------------------------------------
BUDGETS = {
    "simple": {"top_k": 5, "tokens": 1200},
    "medium": {"top_k": 10, "tokens": 2500},
    "complex": {"top_k": 20, "tokens": 5000},
}
_SIMPLE_Q = re.compile(r"\b(кто|где|когда|сколько|какой|какая|какое|на каком|на какой|есть ли|что такое|who|where|when|how many|which)\b", re.I)
_COMPLEX_Q = re.compile(r"\b(почему|зачем|как (устроен|работает|связан|поступ)|сравни|объясни|что если|что делать|в кризисе|план|стратег|итог|история|все правила|какие правила|why|how does|compare|explain)\b", re.I)
_INTENT = (
    (ROUTER_SYS_CMD, re.compile(r"\b(правил|приказ|можно ли|разреш|запрещ|обязан|должен|нельзя|что делать|как поступ|кризис)", re.I)),
    (ROUTER_PERSONA, re.compile(r"\b(кто такой|кто в команде|роль|роли|личност|кто (принимает|решает|отвечает)|права)", re.I)),
    (ROUTER_WORLD, re.compile(r"\b(сколько|где|порт|адрес|состояни|на какой|на каком|стоит|значение|цена|кошел|модел|желез)", re.I)),
)
_EXPAND = {REL_SUPERSEDES: 0.6, REL_CONFLICTS: 0.6, REL_REFERS: 0.5,
           REL_DUPLICATE: 0.3, REL_RELATED: 0.45}
_FIELD_WEIGHTS = (("claim", 3.0), ("source", 2.0), ("context", 1.0))
_NODE_ID_RE = re.compile(r"\bmn_[0-9a-f]{12}\b")
_TOKEN_RE = re.compile(r"[a-zа-яё0-9_./-]{2,}", re.IGNORECASE)


# --- текст ------------------------------------------------------------------------
def norm(s: Any) -> str:
    return str(s or "").lower().replace("ё", "е")


def stem(w: str) -> str:
    return w[:6]


def node_text(n: Dict[str, Any]) -> str:
    parts = [str(n.get("claim") or ""), str(n.get("context") or ""), str(n.get("source") or "")]
    ev = n.get("evidence") or []
    if isinstance(ev, (list, tuple)):
        parts.extend(str(e) for e in ev)
    return "\n".join(parts)


_STOP_STEMS = {stem(s) for s in STOP_WORDS}


def tokens(text: str, min_len: int = 3) -> List[str]:
    """Основы слов без стоп-слов и id узлов (кириллица/латиница — 6 символов)."""
    out: List[str] = []
    for w in _TOKEN_RE.findall(norm(text)):
        w = w.strip("./-")
        if len(w) < min_len or _NODE_ID_RE.match(w):
            continue
        s = stem(w) if re.match(r"[а-яa-z]+$", w) else w
        if s in _STOP_STEMS:
            continue
        out.append(s)
    return out


def entities_of(text: str) -> Set[str]:
    t = norm(text)
    return {name for name, markers in ENTITIES.items() if any(m in t for m in markers)}


_CYR_RE = re.compile(r"[а-яёА-ЯЁ]")


def estimate_tokens(text: str, chars_per_token: float = 4.0, cyr_chars_per_token: float = 2.6) -> int:
    """Оценка токенов без токенайзера: латиница/цифры ~4 симв./токен,
    кириллица ~2.6 (Qwen/Llama режут русский мельче; ревью 02.09, п.7 —
    иначе бюджет 1200 «по оценке» = ~1800 у модели)."""
    text = str(text or "")
    if not text.strip():
        return 0
    cyr = len(_CYR_RE.findall(text))
    return max(1, int(math.ceil(cyr / cyr_chars_per_token + (len(text) - cyr) / chars_per_token)))


def hub_id(entity: str) -> str:
    """Детерминированный id хаба сущности (формат mn_ + 12 hex)."""
    return "mn_" + hashlib.sha1(f"hub:{entity}".encode("utf-8")).hexdigest()[:12]


def edge_rel(node: Dict[str, Any], to_id: str) -> str:
    meta = (node.get("link_meta") or {}).get(to_id) or {}
    return str(meta.get("rel") or REL_RELATED)


def router_of(n: Dict[str, Any]) -> Optional[str]:
    return next((t for t in (n.get("tags") or []) if t in ROUTERS), None)


# --- классификатор роутера для НОВОГО узла (Ф1 02.09; из mnemos_routers.py) ----
# Баллы по регуляркам по claim + context[:300]; kind=rule даёт +1.5 к sys_cmd;
# при равенстве приоритет sys_cmd > persona_def > world_state (правило дороже
# описания). На боевом сторе 02.09: sys_cmd 17 / persona_def 4 / world_state 28.
_ROUTER_PATTERNS: Dict[str, List[Tuple[str, float]]] = {
    ROUTER_SYS_CMD: [
        (r"^\s*(правило|приказ)\b", 3.0),
        (r"\b(приказ|правило)\s+(ивана|памяти|\d)", 2.0),
        (r"\b(обязан|обязана|запрещ|нельзя|не (переделывать|откатывать|ограничиваем|возвращать|трогаем|тащить))", 2.0),
        (r"\b(напомнить|задача после|после переезда|план ивана)", 2.5),
        (r"\b(режим торговли|защиты сняты|торгует только)", 1.5),
        (r"\b(чек-5|обходные пути)", 2.0),
        (r"\b(дефолт|по умолчанию)\b", 0.4),  # < 0.5 дефолта world_state: не даёт ничьей
    ],
    ROUTER_PERSONA: [
        (r"\bроли команды\b", 3.0),
        (r"\b(личност|характер|семь[иея]\b|слияни)", 2.5),
        (r"\b(трейдер-агент|агент команды|оркестратор)\b", 2.0),
        (r"^\s*(алиса|акме-[123]|k-2so|иван)\b[^.]{0,20}(—|-)\s", 2.0),
        (r"\bтебе дан доступ\b", 3.0),
        (r"\b(права|доступ)\s+(алиса|агента)", 1.5),
        (r"\bпакт\b", 1.5),
    ],
    ROUTER_WORLD: [
        (r"\b\d{1,3}([.,]\d+)?\s?(usdc|usd|\$|%|узл|байт|сдел|мс|порт|гб|gb)", 1.5),
        (r"\b(порт|адрес|кошел|эндпоинт|лендинг|http|127\.0\.0\.1|0x[0-9a-f]{4})", 1.5),
        (r"\b(настроен|настроено|живёт|живет|слушает)\b", 1.0),
        (r"\b(постмортем|отчёт|отчет|ledger|стор .* на \d|на \d\d\.\d\d)", 1.0),
        (r"\b(протокол|api|сервер|модель|фасилитатор|маркет|конкурент)", 0.7),
        (r"\b(это|—)\s+(движок|платн|поиск|прото|кошел)", 0.7),
    ],
}
_ROUTER_ORDER = {ROUTER_SYS_CMD: 0, ROUTER_PERSONA: 1, ROUTER_WORLD: 2}
_ROUTER_RE = {r: [(re.compile(p, re.IGNORECASE), w) for p, w in pats]
              for r, pats in _ROUTER_PATTERNS.items()}


def classify_router(n: Dict[str, Any]) -> Tuple[str, float]:
    """Тег-маршрутизатор узла по эвристике: (router, уверенность = разрыв
    между лучшим и вторым баллом). Хаб — по большинству членов, здесь не решается."""
    text = norm(str(n.get("claim") or "") + " " + str(n.get("context") or "")[:300])
    scores = {r: 0.0 for r in ROUTERS}
    for r, pats in _ROUTER_RE.items():
        for rx, w in pats:
            if rx.search(text):
                scores[r] += w
    if n.get("kind") == "rule":
        scores[ROUTER_SYS_CMD] += 1.5
    elif scores[ROUTER_WORLD] == 0:
        scores[ROUTER_WORLD] += 0.5  # дефолт для всего, кроме правил: факт о мире
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], _ROUTER_ORDER[kv[0]]))
    if ranked[0][1] <= 0:
        return ROUTER_WORLD, 0.0  # ревью 02.09 (п.4): ничья без сигналов — не sys_cmd
    return ranked[0][0], round(ranked[0][1] - ranked[1][1], 2)


def ensure_router_tag(tags: List[str], n: Dict[str, Any]) -> List[str]:
    """Ровно один тег-роутер: явный тег клиента (первый, без учёта регистра)
    уважается, остальные роутер-теги снимаются; без явного — классификатор."""
    explicit = [t for t in tags if str(t).lower() in ROUTERS]
    rest = [t for t in tags if str(t).lower() not in ROUTERS]
    router = str(explicit[0]).lower() if explicit else classify_router(n)[0]
    return rest + [router]


def classify_query(q: str) -> Dict[str, Any]:
    """Сложность запроса -> top_k/бюджет; намерение -> тег-роутер."""
    words = re.findall(r"[\wа-яё$%.-]+", q, re.I)
    n = len(words)
    ents = entities_of(q)
    signals: List[str] = []
    score = 0
    if _COMPLEX_Q.search(q):
        score += 1
        signals.append("сложный вопрос-маркер")
    conj = len(re.findall(r"\b(и|или|а также|плюс)\b", q, re.I)) + q.count(",") + q.count(";")
    if conj >= 2:
        score += 1
        signals.append(f"союзов/запятых {conj}")
    if n >= 14:
        score += 2
        signals.append(f"{n} слов (>=14)")
    elif n >= 8:
        score += 1
        signals.append(f"{n} слов (>=8)")
    if len(ents) >= 3:
        score += 1
        signals.append(f"сущностей {len(ents)}")
    if _SIMPLE_Q.search(q) and n <= 8:
        score -= 1
        signals.append("простой вопрос-маркер")
    if q.strip().endswith("?") and n <= 3:
        score -= 1
    level = "simple" if score <= 0 else ("medium" if score <= 2 else "complex")
    intent = next((r for r, pat in _INTENT if pat.search(q)), None)
    return {"complexity": level, "score": score, "words": n, "entities": sorted(ents),
            "signals": signals, "intent": intent, **BUDGETS[level]}


def _decayed_weight(n: Dict[str, Any], now: datetime) -> float:
    w = float(n.get("weight", WEIGHT_MIN) or WEIGHT_MIN)
    last = _parse_iso(str(n.get("last_used") or ""))
    done = _parse_iso(str(n.get("decayed_at") or "")) if n.get("decayed_at") else None
    ref = max([t for t in (last, done) if t is not None], default=None)
    if ref is None or now <= ref or DEFAULT_HALF_LIFE_HOURS <= 0:
        return w
    dt_h = (now - ref).total_seconds() / 3600.0
    return max(weight_floor(str(n.get("kind") or "fact")), w * (0.5 ** (dt_h / DEFAULT_HALF_LIFE_HOURS)))


def _active(n: Dict[str, Any], now: datetime) -> bool:
    if n.get("kind") in ("refuted", "outdated"):
        return False
    vu = _parse_iso(str(n.get("valid_until") or "")) if n.get("valid_until") else None
    return vu is None or now < vu


# ============================================================================
class BudgetSearch:
    """Индекс поверх снимка узлов {id: dict}. Пересобирать при изменении стора."""

    def __init__(self, nodes: Dict[str, Dict[str, Any]]):
        self.nodes = nodes
        self.toks: Dict[str, Set[str]] = {}
        self.hub_by_entity: Dict[str, str] = {}
        self.adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        df: Dict[str, int] = defaultdict(int)
        n_real = 0
        for nid, n in nodes.items():
            self.toks[nid] = set(tokens(node_text(n)))
            if n.get("kind") == KIND_HUB:
                for t in n.get("tags") or []:
                    if t.startswith("entity:"):
                        self.hub_by_entity[t[7:]] = nid
            else:
                n_real += 1
                for s in self.toks[nid]:
                    df[s] += 1
            for to in n.get("links") or []:
                rel = edge_rel(n, to)
                self.adj[nid].append((to, rel))
                self.adj[to].append((nid, rel))
        n_real = max(1, n_real)
        self.idf = {s: math.log(1.0 + n_real / d) for s, d in df.items()}

    def score_node(self, nid: str, q: str, qtoks: List[str], now: datetime) -> Tuple[float, Dict[str, Any]]:
        n = self.nodes[nid]
        ql = q.strip().lower()
        phrase = 0.0
        for field, w in _FIELD_WEIGHTS:
            if ql in str(n.get(field, "")).lower():
                phrase += w
        for ev in n.get("evidence") or []:
            if isinstance(ev, str) and ql in ev.lower():
                phrase += 1.0
        qmass = sum(self.idf.get(t, 1.0) for t in qtoks) or 1.0
        matched = [t for t in qtoks if t in self.toks[nid]]
        token = sum(self.idf.get(t, 1.0) for t in matched) / qmass
        conf = max(0.0, min(1.0, float(n.get("confidence", 0.5) or 0.5)))
        boost = conf * _decayed_weight(n, now)
        return phrase + 2.0 * token + 2.0 * boost, {
            "phrase": phrase, "token": round(token, 3), "matched": matched, "boost": round(boost, 3)}

    def search(self, q: str, top_k: Optional[int] = None, token_budget: Optional[int] = None,
               expand: bool = True, min_token: float = 0.25) -> Dict[str, Any]:
        if not q or not str(q).strip():
            return {"query": q, "count": 0, "results": [], "hubs": [], "conflicts": []}
        cls = classify_query(q)
        k = max(1, min(int(top_k), 50)) if top_k else cls["top_k"]
        budget = int(token_budget) if token_budget else cls["tokens"]
        now = datetime.now(timezone.utc)
        qtoks = list(dict.fromkeys(tokens(q)))
        intent = cls["intent"]

        scored: List[Tuple[float, str, str, Dict[str, Any]]] = []
        for nid, n in self.nodes.items():
            if n.get("kind") == KIND_HUB or not _active(n, now):
                continue
            s, why = self.score_node(nid, q, qtoks, now)
            if why["phrase"] <= 0 and why["token"] < min_token:
                continue
            if intent and intent in (n.get("tags") or []):
                s *= 1.1
                why["intent_boost"] = intent
            scored.append((s, str(n.get("ts") or ""), nid, why))
        scored.sort(key=lambda t: t[1], reverse=True)
        scored.sort(key=lambda t: -t[0])

        picked: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for s, _, nid, why in scored:
            picked[nid] = {"score": round(s, 3), "why": why, "via": None, "rel": None}
            order.append(nid)

        if expand:
            extra: List[Tuple[float, str, str, str]] = []
            for seed in order[: max(3, k // 3)]:
                ps = picked[seed]["score"]
                for nb, rel in self.adj.get(seed, []):
                    nbn = self.nodes.get(nb)
                    if nbn is None or nbn.get("kind") == KIND_HUB or nb in picked or not _active(nbn, now):
                        continue
                    f = _EXPAND.get(rel)
                    if f is not None:
                        extra.append((ps * f, nb, seed, rel))
            extra.sort(key=lambda t: -t[0])
            for s, nb, seed, rel in extra:
                if nb not in picked:
                    picked[nb] = {"score": round(s, 3), "why": {"expanded": True}, "via": seed, "rel": rel}
                    order.append(nb)
            order.sort(key=lambda i: -picked[i]["score"])

        results: List[Dict[str, Any]] = []
        used = 0
        for nid in order[:k]:
            n = self.nodes[nid]
            cost = estimate_tokens(str(n.get("claim") or "") + " " + str(n.get("context") or "")[:240])
            trimmed = False
            if used + cost > budget:
                cost = estimate_tokens(str(n.get("claim") or ""))
                trimmed = True
                if used + cost > budget * 1.3:
                    break
            used += cost
            results.append({
                "id": nid, "kind": n.get("kind"), "router": router_of(n),
                "claim": n.get("claim"),
                "context": None if trimmed else str(n.get("context") or "")[:240],
                "source": n.get("source"), "ts": n.get("ts"), "tags": list(n.get("tags") or []),
                "confidence": n.get("confidence", 0.5), "weight": n.get("weight", 1.0),
                "score": picked[nid]["score"], "via": picked[nid]["via"], "rel": picked[nid]["rel"],
                "why": picked[nid]["why"], "trimmed": trimmed, "tokens": cost,
            })
        conflicts = [
            {"a": r["id"], "b": nb, "b_claim": str(self.nodes[nb].get("claim"))[:120]}
            for r in results for nb, rel in self.adj.get(r["id"], [])
            if rel == REL_CONFLICTS and nb in self.nodes
        ]
        hubs = [{"entity": e, "id": self.hub_by_entity[e], "claim": self.nodes[self.hub_by_entity[e]].get("claim")}
                for e in cls["entities"] if e in self.hub_by_entity]
        return {"query": q, "classification": cls, "top_k": k, "token_budget": budget,
                "tokens_used": used, "count": len(results), "candidates": len(order),
                "results": results, "hubs": hubs, "conflicts": conflicts}


# ============================================================================
_HEADERS = {
    ROUTER_SYS_CMD: "## ПРАВИЛА И ПРИКАЗЫ (обязательны, sys_cmd)",
    ROUTER_PERSONA: "## КТО МЫ (persona_def)",
    ROUTER_WORLD: "## СОСТОЯНИЕ МИРА (world_state, проверяй дату)",
}
PREAMBLE = ("Ниже — выдержка из памяти команды Акме (ShineMnemos). Правила обязательны. "
            "Факты помечены датой: если факт старше твоих данных — перепроверь. "
            "Ссылайся на узлы по id (mn_...).")


def _date(ts: Any) -> str:
    dt = _parse_iso(str(ts or ""))
    return dt.strftime("%d.%m.%Y") if dt else "без даты"


def _node_line(n: Dict[str, Any], with_context: bool) -> str:
    conf = float(n.get("confidence", 0.5) or 0.5)
    src = str(n.get("source") or "").strip()
    line = f"[{n.get('id')} · {_date(n.get('ts'))} · {n.get('kind')} · conf {conf:.2f}] {str(n.get('claim') or '').strip()}"
    if src:
        line += f" (источник: {src[:60]})"
    ctx = str(n.get("context") or "").strip()
    if with_context and ctx and ctx not in ("c", "-"):
        line += f"\n    контекст: {ctx[:240]}"
    return line


def build_system_prompt(nodes: Iterable[Dict[str, Any]], max_tokens: int = 3000,
                        conflicts: Optional[List[Dict[str, Any]]] = None,
                        hubs: Optional[List[Dict[str, Any]]] = None,
                        query: Optional[str] = None,
                        superseded: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Собирает system-prompt из узлов по секциям роутеров.

    Бюджет max_tokens (оценка estimate_tokens): сначала жертвуем context у
    всех узлов, потом хвостом world_state. sys_cmd и persona_def не режутся.
    superseded {id: id-отменяющего} (ревью 02.09, п.12): отменённые приказы
    помечаются «[ОТМЕНЁН ← mn_...]», идут последними в правилах и при
    нехватке бюджета выбрасываются первыми — история, не действующее правило.
    Возвращает {text, tokens, sections, dropped, world_kept}.
    """
    superseded = superseded or {}
    groups: Dict[str, List[Dict[str, Any]]] = {r: [] for r in ROUTERS}
    seen: Set[str] = set()
    for n in nodes:
        if not n or n.get("id") in seen or n.get("kind") == KIND_HUB:
            continue
        seen.add(n["id"])
        groups[router_of(n) or ROUTER_WORLD].append(n)
    for r in groups:
        groups[r].sort(key=lambda n: str(n.get("ts") or ""), reverse=True)
        if r == ROUTER_SYS_CMD:
            groups[r].sort(key=lambda n: (n["id"] in superseded, n.get("kind") != "rule"))

    def render(with_context: bool, world_limit: Optional[int]) -> str:
        parts = [PREAMBLE]
        if query:
            parts.append(f"Вопрос агента: {query}")
        for r in ROUTERS:
            items = groups[r]
            if r == ROUTER_WORLD and world_limit is not None:
                items = items[:world_limit]
            if not items:
                continue
            parts.append(_HEADERS[r])
            for i, n in enumerate(items, 1):
                ln = _node_line(n, with_context)
                if n["id"] in superseded:
                    ln = f"[ОТМЕНЁН ← {superseded[n['id']]}] {ln}"
                parts.append(f"{i}. {ln}" if r == ROUTER_SYS_CMD else f"- {ln}")
        if conflicts:
            parts.append("## ПРОТИВОРЕЧИЯ / ОТМЕНЫ (реши по дате и по приказу)")
            for c in conflicts:
                parts.append(f"- {c.get('a')} <-> {c.get('b')}: {str(c.get('b_claim') or '')[:100]}")
        if hubs:
            parts.append("## НАВИГАЦИЯ (хабы памяти, memory_graph op=hub)")
            for h in hubs:
                parts.append(f"- {h.get('id')}: {str(h.get('claim') or '')[:100]}")
        return "\n".join(parts)

    text = render(True, None)
    dropped: List[str] = []
    if estimate_tokens(text) > max_tokens:
        text = render(False, None)
        dropped.append("context у всех узлов")
    wl = len(groups[ROUTER_WORLD])
    while estimate_tokens(text) > max_tokens and wl > 0:
        wl -= 1
        text = render(False, wl)
    if wl < len(groups[ROUTER_WORLD]):
        dropped.append(f"world_state усечён до {wl} из {len(groups[ROUTER_WORLD])}")
    # отменённые приказы — первые кандидаты на вылет, когда правил больше бюджета
    while estimate_tokens(text) > max_tokens and any(n["id"] in superseded for n in groups[ROUTER_SYS_CMD]):
        gone = groups[ROUTER_SYS_CMD].pop()
        dropped.append(f"отменённый {gone['id']} выброшен")
        text = render(False, wl)
    total = estimate_tokens(text)
    # правила и роли не режутся принципиально: если их одних больше бюджета —
    # честно говорим over_budget, а не выкидываем приказ Ильи молча
    return {"text": text, "tokens": total, "max_tokens": int(max_tokens),
            "over_budget": total > int(max_tokens),
            "sections": {r: len(groups[r]) for r in ROUTERS}, "dropped": dropped, "world_kept": wl}


def wrap_prompt(text: str, fmt: str = "plain", user: str = "") -> str:
    """Обёртка под локальную модель: plain | chatml (Qwen) | llama3."""
    if fmt == "chatml":
        out = f"<|im_start|>system\n{text}<|im_end|>\n"
        if user:
            out += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        return out
    if fmt == "llama3":
        out = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{text}<|eot_id|>"
        if user:
            out += (f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n")
        return out
    if fmt != "plain":
        raise ValueError("format должен быть plain | chatml | llama3")
    return text


# ============================================================================
class Graph:
    """Граф-запросы по снимку узлов (только чтение)."""

    OPS = ("neighbors", "path", "hub", "hubs", "rules_for", "conflicts")

    def __init__(self, nodes: Dict[str, Dict[str, Any]]):
        self.nodes = nodes
        self.out: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.inc: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for nid, n in nodes.items():
            for to in n.get("links") or []:
                if to in nodes:
                    rel = edge_rel(n, to)
                    self.out[nid].append((to, rel))
                    self.inc[to].append((nid, rel))
        self.hubs = {nid: n for nid, n in nodes.items() if n.get("kind") == KIND_HUB}
        self.hub_by_entity: Dict[str, str] = {}
        for hid, h in self.hubs.items():
            for t in h.get("tags") or []:
                if t.startswith("entity:"):
                    self.hub_by_entity[t[7:]] = hid

    def _brief(self, nid: str, **extra: Any) -> Dict[str, Any]:
        n = self.nodes[nid]
        d = {"id": nid, "kind": n.get("kind"), "claim": str(n.get("claim") or "")[:100]}
        d.update(extra)
        return d

    def neighbors(self, nid: str, rel: Optional[str] = None) -> Dict[str, Any]:
        if nid not in self.nodes:
            raise KeyError(f"узел {nid} не найден")
        if rel is not None and rel not in RELS:
            raise ValueError(f"rel должно быть одним из {RELS}, получено {rel!r}")
        return {
            "id": nid,
            "out": [self._brief(b, rel=r) for b, r in self.out.get(nid, []) if rel is None or r == rel],
            "in": [self._brief(a, rel=r) for a, r in self.inc.get(nid, []) if rel is None or r == rel],
        }

    def path(self, a: str, b: str, skip_hubs: bool = False) -> List[Dict[str, Any]]:
        if a not in self.nodes or b not in self.nodes:
            raise KeyError("path: оба конца должны существовать")
        prev: Dict[str, Tuple[str, str]] = {}
        seen = {a}
        q = deque([a])
        while q:
            cur = q.popleft()
            if cur == b:
                break
            for nb, rel in self.out.get(cur, []) + self.inc.get(cur, []):
                if nb in seen or (skip_hubs and nb in self.hubs):
                    continue
                seen.add(nb)
                prev[nb] = (cur, rel)
                q.append(nb)
        if a != b and b not in prev:
            return []
        out = [self._brief(b)]
        cur = b
        while cur != a:
            p, rel = prev[cur]
            out.append(self._brief(p, rel=rel))
            cur = p
        return list(reversed(out))

    def hub(self, key: str) -> Dict[str, Any]:
        hid = self.hub_by_entity.get(key, key)
        h = self.hubs.get(hid)
        if h is None:
            raise KeyError(f"хаб {key!r} не найден; есть: {sorted(self.hub_by_entity)}")
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for m, rel in self.out.get(hid, []):
            if rel == REL_HAS_PART:
                groups[router_of(self.nodes[m]) or "?"].append(self._brief(m))
        related = [self._brief(b) for b, r in self.out.get(hid, []) if r == REL_RELATED]
        return {"id": hid, "claim": h.get("claim"), "members": dict(groups), "related_hubs": related}

    def list_hubs(self) -> List[Dict[str, Any]]:
        return [{"entity": e, "id": h,
                 "members": sum(1 for _, r in self.out.get(h, []) if r == REL_HAS_PART),
                 "related": [b for b, r in self.out.get(h, []) if r == REL_RELATED]}
                for e, h in sorted(self.hub_by_entity.items())]

    def rules_for(self, situation: str, limit: int = 10) -> Dict[str, Any]:
        """Какие правила (sys_cmd) применять в ситуации: сущности -> хабы ->
        члены; плюс совпадения слов; плюс 1 хоп supersedes/conflicts."""
        limit = max(1, min(int(limit or 10), 100))
        now = datetime.now(timezone.utc)
        ents = entities_of(situation)
        qtoks = set(tokens(situation))
        pool: Dict[str, float] = defaultdict(float)
        why: Dict[str, List[str]] = defaultdict(list)
        for e in ents:
            hid = self.hub_by_entity.get(e)
            for m, rel in self.out.get(hid, []) if hid else []:
                if rel == REL_HAS_PART:
                    pool[m] += 1.0
                    why[m].append(f"хаб {e}")
        for nid, n in self.nodes.items():
            if n.get("kind") == KIND_HUB:
                continue
            common = qtoks & set(tokens(node_text(n)))
            if common:
                pool[nid] += 0.5 * len(common)
                why[nid].append("слова: " + ", ".join(sorted(common)[:4]))
        for nid in list(pool):
            for nb, rel in self.out.get(nid, []) + self.inc.get(nid, []):
                if rel in (REL_SUPERSEDES, REL_CONFLICTS) and nb not in pool:
                    pool[nb] += 0.8
                    why[nb].append(f"{rel} от {nid}")
        ranked: List[Tuple[float, str]] = []
        for nid, base in pool.items():
            n = self.nodes[nid]
            if n.get("kind") in ("refuted", "outdated", KIND_HUB) or ROUTER_SYS_CMD not in (n.get("tags") or []):
                continue
            ts = _parse_iso(str(n.get("ts") or "")) or now
            fresh = max(0.0, 1.0 - (now - ts).days / 60.0)
            # отменённое правило (входящее supersedes) уходит вниз, отменяющее — вверх:
            # «какие правила применять» — значит действующие, старое показываем как историю
            superseded = any(r == REL_SUPERSEDES for _, r in self.inc.get(nid, []))
            supersedes = any(r == REL_SUPERSEDES for _, r in self.out.get(nid, []))
            adj = (-1.5 if superseded else 0.0) + (0.5 if supersedes else 0.0)
            score = base + _decayed_weight(n, now) + fresh + (0.5 if n.get("kind") == "rule" else 0.0) + adj
            ranked.append((round(score, 3), nid))
        ranked.sort(reverse=True)
        rules = []
        for score, nid in ranked[:limit]:
            flags = [f"{rel}->{b}" for b, rel in self.out.get(nid, []) if rel in (REL_SUPERSEDES, REL_CONFLICTS)]
            flags += [f"{rel}<-{a}" for a, rel in self.inc.get(nid, []) if rel in (REL_SUPERSEDES, REL_CONFLICTS)]
            rules.append(self._brief(nid, score=score, ts=self.nodes[nid].get("ts"), why=why[nid], flags=flags))
        return {"situation": situation, "entities": sorted(ents),
                "hubs": [self.hub_by_entity[e] for e in ents if e in self.hub_by_entity], "rules": rules}

    def conflicts(self) -> List[Dict[str, Any]]:
        out = []
        for a, edges in self.out.items():
            for b, rel in edges:
                if rel in (REL_CONFLICTS, REL_SUPERSEDES, REL_DUPLICATE):
                    out.append({"from": a, "to": b, "rel": rel,
                                "from_claim": str(self.nodes[a].get("claim") or "")[:80],
                                "to_claim": str(self.nodes[b].get("claim") or "")[:80]})
        return out
