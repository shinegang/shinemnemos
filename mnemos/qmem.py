# -*- coding: utf-8 -*-
"""ShineMnemos: уровни детализации узла (квантование памяти, Ф1 02.09.2026).

Идея Ильи (аналогия с квантованием LLM): узел памяти живёт не «в fp32»
целиком, а на уровне детализации по свежести, весу и востребованности:

    L0  полный   claim + context + evidence + source           (fp16)
    L1  сжатый   экстрактивная выжимка ~30 % L0 + хвост ключей  (int8)
    L2  тезис    1–2 предложения claim + короткий хвост ключей  (int4)

Что лежит в узле (все уровни хранятся, ничего не выбрасывается):

    node["level"]  = 0 | 1 | 2        текущий уровень по политике
    node["levels"] = {"src": <hash L0>, "1": {claim, context, evidence, source, keys},
                                        "2": {claim, context, evidence, source, keys}}
    node["usage"]  = {"count": N, "last_hit": ISO, "hits": [ISO, ... <= 20]}

Поиск (store.search) сканирует ТЕКСТ ТЕКУЩЕГО УРОВНЯ каждого узла и даёт
буст совпадению на L0/L1 (LEVEL_BOOST): полному тексту доверия больше,
чем тезису. Попадание в выдачу — «использование»: usage обновляется и узел
поднимается на L0 (дековантование — полный текст уже в узле). Пересчёт
уровней по политике — в ночном decay (store.decay без node_id).

Политика перехода (decide_level) — мягче прототипа /opt/bench-memory/qmem
(рекомендация §6 отчёта ФЭЙБЛ-КВАНТОВАНИЕ-ПАМЯТИ-02.09): любое из условий
уровня удерживает узел на нём.

    L0: возраст ts <= 30 д  или простой <= 14 д  или вес >= 0.85  или >= 3 попаданий за 7 д
    L1: возраст ts <= 90 д  или простой <= 45 д  или вес >= 0.50
    L2: иначе
    kind=rule и kind=hub — всегда L0 (конституция нужна агенту целиком);
    refuted/outdated/протухшие по TTL — L2 (в выдачу они всё равно не идут).

Простой = now − max(last_used, usage.last_hit): usage не трогает last_used,
поэтому затухание весов (decay) от квантования не зависит.

Только stdlib. Прототип и замеры: /opt/bench-memory/qmem/ (REPORT.md).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .model import DEFAULT_HALF_LIFE_HOURS, WEIGHT_MIN, _parse_iso, weight_floor

# ---------------------------------------------------------------------------
# константы уровней и политики
# ---------------------------------------------------------------------------
L0, L1, L2 = 0, 1, 2
MAX_LEVEL = L2
LEVEL_NAMES = {L0: "L0 полный", L1: "L1 сжатый", L2: "L2 тезис"}

# буст совпадения по уровню (аддитивно к рангу)
LEVEL_BOOST = {L0: 1.0, L1: 0.6, L2: 0.2}

POLICY: Dict[int, Dict[str, float]] = {
    L0: {"age_days": 30, "idle_days": 14, "weight": 0.85, "hits_7d": 3},
    L1: {"age_days": 90, "idle_days": 45, "weight": 0.50},
}
ALWAYS_L0_KINDS = ("rule", "hub")
INACTIVE_KINDS = ("refuted", "outdated")

L1_RATIO = 0.30          # целевая доля текста L0 для L1
L1_MIN_CHARS = 120       # короткий узел ниже этого не режем
L2_RATIO = 0.20
L2_MIN_CHARS = 100
L2_MAX_SENTENCES = 2
L2_KEYS = 5
L2_SECOND_SENTENCE_IF_FIRST_SHORTER = 60
KEY_MAX_WORDS, KEY_MIN_WORDS = 7, 3
KEY_WORD_SLOTS = 3       # имена/термины
KEY_NUM_SLOTS = 4        # числа/идентификаторы
KEY_MIN_SALIENCE = 1.2
KEY_TAIL_WORD_SALIENCE = 2.4   # обычное (не outlier) слово попадает в хвост, только если редкое
KEY_SUBJECT_BONUS = 1.0
KEY_PER_CHARS = 150            # +1 слот хвоста на каждые 150 симв. L0 (ревью 02.09, п.1)
KEY_TAIL_MAX = 14
CLAUSE_MIN_CHARS = 60
USAGE_HITS_KEEP = 20

_STOP = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только её мне было
вот от меня ещё нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас
нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их
чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой
совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при
наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три
эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно
всю между это этого также
the a an of to in on for and or is are be by with as at it this that from not
""".split())

_TOKEN_RE = re.compile(r"/?\w[\w\-\./:=+%@]*")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[«\"(\[A-ZА-ЯЁ0-9])|;\s+|\n+")
_CLAUSE_RE = re.compile(r"[,;:—–(\[]\s|\s[—–-]\s")


# ---------------------------------------------------------------------------
# утилиты
# ---------------------------------------------------------------------------
def tokens(text: str) -> List[str]:
    return [t.strip(".,:;=") for t in _TOKEN_RE.findall(text or "") if t.strip(".,:;=")]


def sentences(text: str) -> List[str]:
    return [p.strip() for p in _SENT_SPLIT_RE.split(text or "") if p and p.strip()]


def l0_fields(d: Dict[str, Any]) -> Dict[str, Any]:
    """Поля L0 узла в том виде, в котором их видит поиск."""
    return {
        "claim": str(d.get("claim") or ""),
        "context": str(d.get("context") or ""),
        "evidence": [e for e in (d.get("evidence") or []) if isinstance(e, str)],
        "source": str(d.get("source") or ""),
    }


def fields_chars(f: Dict[str, Any]) -> int:
    return len(f.get("claim", "")) + len(f.get("context", "")) + \
        sum(len(e) for e in f.get("evidence", [])) + len(f.get("source", ""))


def source_hash(d: Dict[str, Any]) -> str:
    """Отпечаток L0-текста: если он изменился (rewrite/update) — уровни устарели."""
    f = l0_fields(d)
    raw = "\x1f".join([f["claim"], f["context"], f["source"]] + f["evidence"])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# различительность слов (idf по корпусу + бонусы за «идентификаторность»)
# ---------------------------------------------------------------------------
class Salience:
    """idf по узлам корпуса + бонусы: цифры, латиница, пути/идентификаторы."""

    def __init__(self, nodes: Iterable[Dict[str, Any]]):
        df: Counter = Counter()
        n = 0
        for d in nodes:
            n += 1
            f = l0_fields(d)
            seen = {t.lower() for t in tokens(" ".join([f["claim"], f["context"], f["source"]] + f["evidence"]))}
            df.update(seen)
        self.n = max(1, n)
        self.df = df

    def token(self, tok: str) -> float:
        low = tok.lower()
        if low in _STOP:
            return 0.0
        has_digit = any(c.isdigit() for c in tok)
        if len(low) < 3 and not has_digit:
            return 0.0
        idf = math.log((self.n + 1) / (self.df.get(low, 0) + 1)) + 0.1
        bonus = 0.0
        if has_digit:
            bonus += 0.8
        if re.search(r"[_/:@]|\.\w", tok):
            bonus += 0.6
        if re.search(r"[A-Za-z]", tok):
            bonus += 0.5
        if tok[:1].isupper() and re.search(r"[А-ЯЁ]", tok[:1]):
            bonus += 0.4
        if len(tok) > 40:
            bonus -= 1.0
        return idf + bonus

    @staticmethod
    def is_outlier(tok: str, sentence_initial: bool = False) -> bool:
        if any(c.isdigit() for c in tok) or re.search(r"[_/:@A-Za-z]", tok):
            return True
        return bool(not sentence_initial and tok[:1].isupper() and re.search(r"[А-ЯЁ]", tok[:1]))

    def sentence(self, sent: str) -> float:
        toks = tokens(sent)
        if not toks:
            return 0.0
        vals = sorted((self.token(t) for t in toks), reverse=True)[:8]
        return sum(vals) / math.sqrt(max(20, len(sent)))


# ---------------------------------------------------------------------------
# построение уровней
# ---------------------------------------------------------------------------
def _is_numeric_like(tok: str) -> bool:
    return any(c.isdigit() for c in tok) or bool(re.search(r"[_/:@]", tok))


def keys(d: Dict[str, Any], sal: Salience, n_max: int = KEY_MAX_WORDS,
         n_min: int = KEY_MIN_WORDS, exclude_text: str = "") -> str:
    """Хвост ключей: 3–7 самых различительных слов узла (аналог outlier-features
    в LLM.int8 — то, что спрашивают точно: порты, суммы, имена, пути).

    Два пула слотов: имена/термины и числа/идентификаторы — иначе числа с
    высоким idf вытесняют субъект узла. exclude_text — текст, который на этом
    уровне и так хранится (в хвост идут только outlier-токены, которых там нет).
    Дедупликация по подстроке: «2833» уже покрывает «2833/2833».
    """
    f = l0_fields(d)
    order: Dict[str, int] = {}
    best: Dict[str, float] = {}
    weak: Dict[str, float] = {}
    orig: Dict[str, str] = {}
    pos = 0
    excl = exclude_text.strip().lower()
    tail_mode = bool(excl)
    for field, weight in (("claim", 2.0), ("context", 0.0), ("evidence", -0.3), ("source", -0.8)):
        text = " ".join(f[field]) if field == "evidence" else f[field]
        initials = {tokens(snt)[0].lower() for snt in sentences(text) if tokens(snt)}
        for j, t in enumerate(tokens(text)):
            low = t.lower()
            base = sal.token(t)
            if base <= 0.0:
                continue
            s = base + weight + (KEY_SUBJECT_BONUS if field == "claim" and j < 3 else 0.0)
            if low not in order:
                order[low] = pos
                orig[low] = t
                pos += 1
            weak[low] = max(weak.get(low, -9.0), s)
            if base < KEY_MIN_SALIENCE:
                continue
            if tail_mode and not sal.is_outlier(t, sentence_initial=(low in initials)) \
                    and base < KEY_TAIL_WORD_SALIENCE:
                # ревью 02.09 (п.1): обычное слово из claim («шорт», «Алиса» во
                # втором предложении) иначе не попадало ни в выжимку, ни в хвост,
                # и через 45 дней без обращений узел терялся по своему же слову
                continue
            best[low] = max(best.get(low, -9.0), s)
    ranked = [t for t, _ in sorted(best.items(), key=lambda kv: -kv[1])]
    if excl:
        ranked = [t for t in ranked if t not in excl]
    words = [t for t in ranked if not _is_numeric_like(t)]
    nums = [t for t in ranked if _is_numeric_like(t)]
    picked: List[str] = []

    def take(pool: List[str], n: int) -> None:
        got = 0
        for t in pool:
            if got >= n:
                break
            if any(t in p or p in t for p in picked):
                continue
            picked.append(t)
            got += 1

    word_slots = max(1, round(n_max * KEY_WORD_SLOTS / (KEY_WORD_SLOTS + KEY_NUM_SLOTS)))
    take(words, word_slots)
    take(nums, n_max - len(picked))
    if len(picked) < n_max:
        take([t for t in ranked if t not in picked], n_max - len(picked))
    if len(picked) < n_min and not tail_mode:
        fallback = [t for t, _ in sorted(weak.items(), key=lambda kv: -kv[1])
                    if t not in picked and t not in excl]
        take(fallback, n_min - len(picked))
    picked = picked[:n_max]
    picked.sort(key=lambda t: order[t])
    return " ".join(orig[t] for t in picked)


def clause_cut(text: str, budget: int) -> str:
    """Режет предложение по границе оборота (запятая, двоеточие, тире, скобка),
    чтобы уложиться в budget; хвост помечается «…». Короче CLAUSE_MIN_CHARS
    не режем — обрывок без смысла хуже длинного оборота."""
    if len(text) <= budget:
        return text
    cut = 0
    for m in _CLAUSE_RE.finditer(text):
        if m.start() > budget:
            break
        cut = m.start()
    if cut < max(CLAUSE_MIN_CHARS, budget // 2):
        return text
    return text[:cut].rstrip(" ,;:—–(-") + "…"


def build_l1(d: Dict[str, Any], sal: Salience, ratio: float = L1_RATIO) -> Dict[str, Any]:
    """Экстрактивная выжимка ~ratio текста L0 с сохранением структуры полей.
    Первое предложение claim держится всегда; дальше — жадно по плотности
    различительности. Короткие узлы (<= L1_MIN_CHARS) не режутся."""
    f = l0_fields(d)
    total = fields_chars(f)
    if total <= L1_MIN_CHARS:
        return {**f, "keys": ""}
    k = keys(d, sal)
    budget = max(L1_MIN_CHARS, int(total * ratio)) - len(k)

    cands: List[Tuple[float, str, int, str]] = []
    claim_s = sentences(f["claim"])
    for i, s in enumerate(claim_s):
        cands.append((sal.sentence(s) + (0.5 if i == 0 else 0.0), "claim", i, s))
    for i, s in enumerate(sentences(f["context"])):
        cands.append((sal.sentence(s), "context", i, s))
    for i, e in enumerate(f["evidence"]):
        cands.append((sal.sentence(e) * 0.9, "evidence", i, e))

    chosen: Dict[Tuple[str, int], str] = {}
    used = 0
    if claim_s:
        first = clause_cut(claim_s[0], int(budget * 1.15))
        chosen[("claim", 0)] = first
        used += len(first)
    for _score, field, i, text in sorted(cands, key=lambda c: -c[0]):
        key = (field, i)
        if key in chosen:
            continue
        if used + len(text) <= budget * 1.15:
            chosen[key] = text
            used += len(text)
    out = {
        "claim": " ".join(chosen[k] for k in sorted(chosen) if k[0] == "claim"),
        "context": " ".join(chosen[k] for k in sorted(chosen) if k[0] == "context"),
        "evidence": [chosen[k] for k in sorted(chosen) if k[0] == "evidence"],
        "source": f["source"] if used + len(f["source"]) <= budget * 1.3 else "",
    }
    out["keys"] = keys(d, sal, n_max=tail_slots(d, KEY_MAX_WORDS), exclude_text=" ".join(
        [out["claim"], out["context"], out["source"]] + out["evidence"]))
    return out


def build_l2(d: Dict[str, Any], sal: Salience) -> Dict[str, Any]:
    """Тезис: первое предложение claim (+ самое различительное второе, если
    первое короткое) + хвост ключей (<= L2_KEYS)."""
    f = l0_fields(d)
    total = fields_chars(f)
    sents = sentences(f["claim"]) or [f["claim"]]
    budget = max(L2_MIN_CHARS, int(total * L2_RATIO))
    thesis = [clause_cut(sents[0], budget)]
    if len(sents) > 1 and len(thesis[0]) < L2_SECOND_SENTENCE_IF_FIRST_SHORTER:
        rest = max(sents[1:], key=sal.sentence)
        thesis.append(clause_cut(rest, max(CLAUSE_MIN_CHARS, budget - len(thesis[0]))))
    claim = " ".join(thesis[:L2_MAX_SENTENCES])
    return {"claim": claim, "context": "", "evidence": [], "source": "",
            "keys": keys(d, sal, n_max=tail_slots(d, L2_KEYS), exclude_text=claim)}


def tail_slots(d: Dict[str, Any], base: int) -> int:
    """Число слотов хвоста ключей пропорционально длине L0 (base + 1 на 150 симв.)."""
    return min(KEY_TAIL_MAX, base + fields_chars(l0_fields(d)) // KEY_PER_CHARS)


def build_levels(d: Dict[str, Any], sal: Salience) -> Dict[str, Any]:
    """Все сжатые уровни узла + отпечаток исходного текста."""
    return {"src": source_hash(d), "1": build_l1(d, sal), "2": build_l2(d, sal)}


def levels_stale(d: Dict[str, Any]) -> bool:
    lv = d.get("levels")
    return not isinstance(lv, dict) or lv.get("src") != source_hash(d) \
        or not isinstance(lv.get("1"), dict) or not isinstance(lv.get("2"), dict)


def effective_level(d: Dict[str, Any]) -> int:
    """Уровень, по которому реально идёт поиск: level>0 без сжатого текста
    (старый узел до Ф1, правка файла) — это L0 и по тексту, и по бусту."""
    lv = current_level(d)
    if lv > L0 and not isinstance((d.get("levels") or {}).get(str(lv)), dict):
        return L0
    return lv


def level_text(d: Dict[str, Any]) -> Dict[str, Any]:
    """Текст, по которому идёт поиск: поля текущего уровня. L0 — сам узел."""
    lv = effective_level(d)
    if lv > L0:
        return (d.get("levels") or {})[str(lv)]
    return {**l0_fields(d), "keys": ""}


def normalize(d: Dict[str, Any]) -> None:
    """Привести служебные поля к схеме при загрузке файла: мусор в usage/level
    не должен ронять search/stats/decay; ключи level/usage существуют всегда
    (touch не меняет набор ключей узла — важно для копирования вне лока)."""
    d["level"] = current_level(d)
    u = d.get("usage")
    if not isinstance(u, dict):
        u = {}
    try:
        u["count"] = int(u.get("count", 0) or 0)
    except (TypeError, ValueError):
        u["count"] = 0
    u["hits"] = [h for h in (u.get("hits") or []) if isinstance(h, str)][-USAGE_HITS_KEEP:] \
        if isinstance(u.get("hits"), list) else []
    if not isinstance(u.get("last_hit"), str):
        u.pop("last_hit", None)
    d["usage"] = u
    if "levels" in d and not isinstance(d.get("levels"), dict):
        d.pop("levels")


def current_level(d: Dict[str, Any]) -> int:
    try:
        lv = int(d.get("level", L0) or L0)
    except (TypeError, ValueError):
        return L0
    return min(max(lv, L0), MAX_LEVEL)


# ---------------------------------------------------------------------------
# политика перехода
# ---------------------------------------------------------------------------
def _decay_ref(d: Dict[str, Any]) -> Optional[datetime]:
    refs = [_parse_iso(str(d.get("last_used") or "")),
            _parse_iso(str(d.get("decayed_at") or "")) if d.get("decayed_at") else None]
    refs = [r for r in refs if r is not None]
    return max(refs) if refs else None


def effective_weight(d: Dict[str, Any], now: datetime) -> float:
    """Вес «как будто сейчас» — 1-в-1 store._node_decayed_weight."""
    w = float(d.get("weight", WEIGHT_MIN) or WEIGHT_MIN)
    ref = _decay_ref(d)
    if ref is None or now <= ref or DEFAULT_HALF_LIFE_HOURS <= 0:
        return w
    dt_h = (now - ref).total_seconds() / 3600.0
    return max(weight_floor(str(d.get("kind") or "fact")), w * (0.5 ** (dt_h / DEFAULT_HALF_LIFE_HOURS)))


def _usage(d: Dict[str, Any]) -> Dict[str, Any]:
    u = d.get("usage")
    return u if isinstance(u, dict) else {}


def hits_7d(d: Dict[str, Any], now: datetime) -> int:
    cutoff = now - timedelta(days=7)
    hits = _usage(d).get("hits")
    if not isinstance(hits, list):
        return 0
    return sum(1 for h in hits if isinstance(h, str) and (_parse_iso(h) or cutoff) > cutoff)


def last_activity(d: Dict[str, Any]) -> Optional[datetime]:
    """max(last_used, usage.last_hit) — точка отсчёта простоя."""
    cands = [_parse_iso(str(d.get("last_used") or "")),
             _parse_iso(str(_usage(d).get("last_hit") or ""))]
    cands = [c for c in cands if c is not None]
    return max(cands) if cands else None


def decide_level(d: Dict[str, Any], now: datetime,
                 policy: Dict[int, Dict[str, float]] = POLICY) -> int:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    kind = str(d.get("kind") or "fact")
    if kind in ALWAYS_L0_KINDS:
        return L0
    if kind in INACTIVE_KINDS:
        return MAX_LEVEL
    vu = _parse_iso(str(d.get("valid_until") or "")) if d.get("valid_until") else None
    if vu is not None and now >= vu:
        return MAX_LEVEL
    ts = _parse_iso(str(d.get("ts") or "")) or now
    act = last_activity(d) or ts
    age = (now - ts).total_seconds() / 86400.0
    idle = (now - act).total_seconds() / 86400.0
    w = effective_weight(d, now)
    h7 = hits_7d(d, now)
    for lv in (L0, L1):
        p = policy[lv]
        if age <= p["age_days"] or idle <= p["idle_days"] or w >= p["weight"] \
                or (lv == L0 and h7 >= p.get("hits_7d", 10 ** 9)):
            return lv
    return MAX_LEVEL


def touch(d: Dict[str, Any], now: datetime) -> bool:
    """Попадание в выдачу: usage += 1, узел поднимается на L0.
    Возвращает True, если узел изменился (уровень или usage)."""
    iso = now.isoformat(timespec="milliseconds")
    u = dict(_usage(d))
    try:
        cnt = int(u.get("count", 0) or 0)
    except (TypeError, ValueError):
        cnt = 0
    u["count"] = cnt + 1
    prev = _parse_iso(str(u.get("last_hit") or ""))
    if prev is None or now > prev:
        u["last_hit"] = iso
    hits = u.get("hits") if isinstance(u.get("hits"), list) else []
    u["hits"] = [h for h in hits if isinstance(h, str)][-(USAGE_HITS_KEEP - 1):] + [iso]
    d["usage"] = u
    if current_level(d) != L0:
        d["level"] = L0
    return True


def level_histogram(nodes: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    h = {str(lv): 0 for lv in (L0, L1, L2)}
    for d in nodes:
        h[str(current_level(d))] += 1
    return h
