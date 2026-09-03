# -*- coding: utf-8 -*-
"""Лексический слой поиска Mnemos: стеммер + BM25F + RRF. ТОЛЬКО stdlib.

Перенос из исследования ML-BOOST (03.09.2026, /opt/bench-memory/mlboost) в
боевой Mnemos. Отличие от исследовательской версии — ноль зависимостей:
`snowballstemmer` и `numpy` заменены на собственный порт русского Snowball и
обычные списки. Это требование requirements.txt («рантайм — ноль
зависимостей, только stdlib»): боевой демон запускается системным python3, в
котором нет ни snowballstemmer, ни fastembed.

Что здесь есть и зачем:
  * stem()   — снежный ком (рус. Snowball + лёгкий англ.) вместо обрезки слова:
               «шорт» достаёт «шорты». Подстрочный поиск этого не умеет.
  * tokens() — основы + разбиение составных идентификаторов
               («127.0.0.1:8765» -> 8765, «llama-70B» -> llama, 70b).
  * BM25F    — классический BM25 (k1=1.2, b=0.75) по полям узла с весами
               Mnemos (claim 3 / keys 3 / source 2 / context 1 / evidence 1).
  * rrf()    — Reciprocal Rank Fusion, k=60 (канон Cormack 2009).

RRF выбран потому, что он БЕЗПАРАМЕТРИЧЕН: он работает с рангами, а не со
шкалами, и его не надо подбирать. Подбор весов слияния на 15 запросах GT —
подгонка: замер leave-one-query-out (ML-BOOST §5.2) показал, что подобранное
слияние даёт на НОВОМ запросе 0.8500 против 0.8833 у неподобранного.

ВАЖНО про уровни. В боевом Mnemos фильтр Блума из ML-BOOST НЕ применяется и
здесь его нет — см. REPORT.md, раздел ML-BOOST. Причина: в проде полный текст
узла всегда лежит в самом узле, а Store.search ранжирует max(полный, уровень),
поэтому сжатие L1/L2 не делает слова недостижимыми (замер: kw L0 0.9444,
L1 0.9278, L2 0.9444). Фильтр лечил свойство прототипа qmem с холодным стором,
которого в проде нет.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# стеммер: русский Snowball (порт) + лёгкий английский
# ---------------------------------------------------------------------------
_RU_VOWELS = "аеиоуыэюя"
_CYR_RE = re.compile(r"[а-яё]", re.IGNORECASE)


def _ru_regions(w: str) -> Tuple[int, int, int]:
    """RV / R1 / R2 по определению Snowball для русского."""
    rv = len(w)
    for i, ch in enumerate(w):
        if ch in _RU_VOWELS:
            rv = i + 1
            break
    r1 = len(w)
    for i in range(1, len(w)):
        if w[i] not in _RU_VOWELS and w[i - 1] in _RU_VOWELS:
            r1 = i + 1
            break
    r2 = len(w)
    for i in range(r1 + 1, len(w)):
        if w[i] not in _RU_VOWELS and w[i - 1] in _RU_VOWELS:
            r2 = i + 1
            break
    return rv, r1, r2


def _match(w: str, start: int, endings: Sequence[str]) -> str:
    """Самое длинное окончание из endings, целиком лежащее правее start."""
    best = ""
    for e in endings:
        if len(e) > len(best) and w.endswith(e) and len(w) - len(e) >= start:
            best = e
    return best


def _match_g1(w: str, rv: int, endings: Sequence[str]) -> str:
    """Окончание «группы 1»: ему обязана предшествовать «а» или «я», причём
    сама эта буква тоже должна лежать в RV.

    Без проверки «а/я в RV» стеммер срезает последнюю букву у слов, где гласная
    стоит до начала RV: «план» -> «пла», «дал» -> «да», «брать» -> «бра»
    (расхождение с эталонным snowballstemmer, найдено сверкой на словаре
    корпуса — 7 слов из 878)."""
    e = _match(w, rv, endings)
    if not e:
        return ""
    i = len(w) - len(e)
    return e if i - 1 >= rv and w[i - 1] in "ая" else ""


# группа 1 требует предшествующей «а» или «я» (Snowball: PERFECTIVE GERUND)
_PERF_1 = ("в", "вши", "вшись")
_PERF_2 = ("ив", "ивши", "ившись", "ыв", "ывши", "ывшись")
_ADJECTIVE = ("ее", "ие", "ые", "ое", "ими", "ыми", "ей", "ий", "ый", "ой", "ем",
              "им", "ым", "ом", "его", "ого", "ему", "ому", "их", "ых", "ую", "юю",
              "ая", "яя", "ою", "ею")
_PARTICIPLE_1 = ("ем", "нн", "вш", "ющ", "щ")
_PARTICIPLE_2 = ("ивш", "ывш", "ующ")
_VERB_1 = ("ла", "на", "ете", "йте", "ли", "й", "л", "ем", "н", "ло", "но", "ет",
           "ют", "ны", "ть", "ешь", "нно")
_VERB_2 = ("ила", "ыла", "ена", "ейте", "уйте", "ите", "или", "ыли", "ей", "уй",
           "ил", "ыл", "им", "ым", "ен", "ило", "ыло", "ено", "ят", "ует", "уют",
           "ит", "ыт", "ены", "ить", "ыть", "ишь", "ую", "ю")
_NOUN = ("а", "ев", "ов", "ие", "ье", "е", "иями", "ями", "ами", "еи", "ии", "и",
         "ией", "ей", "ой", "ий", "й", "иям", "ям", "ием", "ем", "ам", "ом", "о",
         "у", "ах", "иях", "ях", "ы", "ь", "ию", "ью", "ю", "ия", "ья", "я")
_DERIVATIONAL = ("ост", "ость")
_SUPERLATIVE = ("ейш", "ейше")


def stem_ru(w: str) -> str:
    """Русский стеммер Snowball (шаги 1–4)."""
    w = w.replace("ё", "е")
    rv, _r1, r2 = _ru_regions(w)

    # --- шаг 1 -------------------------------------------------------------
    out = ""
    e = _match(w, rv, _PERF_2) or _match_g1(w, rv, _PERF_1)
    if e:
        out = w[: len(w) - len(e)]
    if not out:
        # рефлексив
        e = _match(w, rv, ("ся", "сь"))
        if e:
            w = w[: len(w) - len(e)]
            rv, _r1, r2 = _ru_regions(w)
        # адъективное
        a = _match(w, rv, _ADJECTIVE)
        if a:
            w2 = w[: len(w) - len(a)]
            p = _match(w2, rv, _PARTICIPLE_2) or _match_g1(w2, rv, _PARTICIPLE_1)
            w = w2[: len(w2) - len(p)] if p else w2
        else:
            v = _match(w, rv, _VERB_2) or _match_g1(w, rv, _VERB_1)
            if v:
                w = w[: len(w) - len(v)]
            else:
                n = _match(w, rv, _NOUN)
                w = w[: len(w) - len(n)] if n else w
    else:
        w = out
    rv, _r1, r2 = _ru_regions(w)

    # --- шаг 2: снять «и» --------------------------------------------------
    if w.endswith("и") and len(w) - 1 >= rv:
        w = w[:-1]
    # --- шаг 3: словообразовательное в R2 ---------------------------------
    rv, _r1, r2 = _ru_regions(w)
    d = _match(w, r2, _DERIVATIONAL)
    if d:
        w = w[: len(w) - len(d)]
    # --- шаг 4 -------------------------------------------------------------
    if w.endswith("нн"):
        w = w[:-1]
    else:
        s = _match(w, 0, _SUPERLATIVE)
        if s:
            w = w[: len(w) - len(s)]
            if w.endswith("нн"):
                w = w[:-1]
        elif w.endswith("ь"):
            w = w[:-1]
    return w


def stem_en(w: str) -> str:
    """Английский стеммер: снимаются ТОЛЬКО словоизменительные окончания
    (мн. число и глагольные -s/-es/-ies/-ed/-ing), остаток не короче 3.

    Словообразовательные суффиксы (-ation, -ness, -ment, -able) намеренно не
    трогаются. Полный Porter2 в корпусе Mnemos не окупается: латиница здесь —
    почти целиком идентификаторы и имена (mnemos, llama, x402, coinbase), а
    агрессивное снятие даёт вредные склейки («location» -> «loc» смешивается с
    «local»). Для BM25 важна не парность с эталоном, а ОДИНАКОВОСТЬ основы у
    запроса и документа — она обеспечена тем, что стеммер один и тот же.
    """
    if w.endswith("ies") and len(w) >= 6:
        return w[:-3] + "y"
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: len(w) - len(suf)]
    # «es» снимается только после шипящих/свистящих (buses, boxes), иначе это
    # обычное «s»: «gates» -> «gate», а не «gat»
    if w.endswith("es") and len(w) >= 5 and w[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) >= 4:
        return w[:-1]
    return w


def stem(w: str) -> str:
    """Кириллица -> русский Snowball, чистая латиница -> английский, иначе как есть."""
    if _CYR_RE.search(w):
        return stem_ru(w)
    if w.isalpha():
        return stem_en(w)
    return w


# ---------------------------------------------------------------------------
# токенизация
# ---------------------------------------------------------------------------
STOP = set("""
и в во не что он на я с со как а то все всё она так его но да ты к у же вы за бы по только её мне
было вот от меня ещё нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до
вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя
их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого
какой какая какое совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех
никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них
много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более
всегда конечно всю между это также при чем сколько
the a an of to in on for and or is are be by with as at it this that from not what which who where
when how does do
""".split())
_STOP_STEMS = {stem(w) for w in STOP}

_TOKEN_RE = re.compile(r"[a-zа-яё0-9][a-zа-яё0-9_./:+%@-]*", re.IGNORECASE)
_NODE_ID_RE = re.compile(r"^mn_[0-9a-f]{12}$")
_SPLIT_RE = re.compile(r"[./:+%@_-]+")


def tokens(text: str, min_len: int = 2, subtokens: bool = True) -> List[str]:
    """Основы слов текста. subtokens=True дополнительно разбивает составные
    идентификаторы: без этого точечные запросы «8765», «70B» ловятся только
    подстрокой, а после стемминга подстроки уже нет."""
    out: List[str] = []
    for w in _TOKEN_RE.findall(str(text or "").lower().replace("ё", "е")):
        w = w.strip("./:-+%@_")
        if len(w) < min_len or _NODE_ID_RE.match(w):
            continue
        s = stem(w)
        if not (s in _STOP_STEMS or s in STOP):
            out.append(s)
        if subtokens and _SPLIT_RE.search(w):
            for p in _SPLIT_RE.split(w):
                if len(p) >= min_len and p != w and p not in _STOP_STEMS and p not in STOP:
                    out.append(stem(p))
    return out


# ---------------------------------------------------------------------------
# BM25F
# ---------------------------------------------------------------------------
FIELDS: Tuple[Tuple[str, float], ...] = (
    ("claim", 3.0), ("keys", 3.0), ("source", 2.0), ("context", 1.0), ("evidence", 1.0))


def node_fields(n: Dict[str, Any]) -> Dict[str, str]:
    """Поля узла в плоском виде. Понимает и «сырой» узел Mnemos, и горячий
    узел qmem-прототипа (текст в n['text'])."""
    src = n.get("text") if isinstance(n.get("text"), dict) else n
    ev = src.get("evidence") or []
    return {
        "claim": str(src.get("claim") or ""),
        "context": str(src.get("context") or ""),
        "source": str(src.get("source") or ""),
        "keys": str(src.get("keys") or ""),
        "evidence": " ".join(str(e) for e in ev if isinstance(e, str)),
    }


def node_blob_of(n: Dict[str, Any]) -> str:
    """Весь текст узла одной строкой — то, что эмбеддится плотным сигналом."""
    f = node_fields(n)
    return "\n".join(f[name] for name, _ in FIELDS if f[name])


class BM25F:
    """BM25 по «взвешенно-склеенному» документу: токены поля повторяются
    round(weight) раз — стандартный приём BM25F. k1/b — классические 1.2/0.75,
    idf — сглаженный Lucene-вариант (никогда не отрицательный)."""

    K1 = 1.2
    B = 0.75

    def __init__(self, nodes: Dict[str, Dict[str, Any]],
                 field_weights: Sequence[Tuple[str, float]] = FIELDS):
        self.ids = list(nodes)
        self.fw = list(field_weights)
        self.tf: Dict[str, Dict[str, int]] = {}
        self.dl: Dict[str, int] = {}
        df: Dict[str, int] = defaultdict(int)
        for nid in self.ids:
            f = node_fields(nodes[nid])
            counts: Dict[str, int] = defaultdict(int)
            n_tok = 0
            for name, w in self.fw:
                rep = max(1, int(round(w)))
                for t in tokens(f.get(name, "")):
                    counts[t] += rep
                    n_tok += rep
            self.tf[nid] = dict(counts)
            self.dl[nid] = n_tok
            for t in counts:
                df[t] += 1
        n = max(1, len(self.ids))
        self.avgdl = (sum(self.dl.values()) / n) or 1.0
        self.idf = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.df = dict(df)
        self.n_docs = n

    def score(self, nid: str, qtoks: Sequence[str]) -> float:
        tf = self.tf.get(nid) or {}
        dl = self.dl.get(nid, 0)
        s = 0.0
        for t in qtoks:
            f = tf.get(t)
            if not f:
                continue
            s += self.idf.get(t, 0.0) * (f * (self.K1 + 1)) / (
                f + self.K1 * (1 - self.B + self.B * dl / self.avgdl))
        return s

    def scores(self, qtoks: Sequence[str]) -> Dict[str, float]:
        return {nid: self.score(nid, qtoks) for nid in self.ids}

    def rank(self, query: str, depth: int = 20,
             skip: Iterable[str] = ()) -> List[str]:
        """Топ-depth id по BM25, только узлы с ненулевым скором."""
        skip = set(skip)
        s = self.scores(list(dict.fromkeys(tokens(query))))
        hits = [nid for nid in self.ids if nid not in skip and s.get(nid, 0.0) > 0.0]
        hits.sort(key=lambda n: -s[n])
        return hits[:depth]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
RRF_K = 60


def rrf(lists: Sequence[Sequence[str]], k: int = RRF_K,
        weights: Sequence[float] = ()) -> List[str]:
    """RRF по канону Cormack 2009: score(d) = sum_i w_i / (k + rank_i(d)).

    Списки, в которых документа нет, за него просто не голосуют. Веса по
    умолчанию равные — подбирать их на 15 запросах нельзя (см. модульный
    докстринг)."""
    ws = list(weights) or [1.0] * len(lists)
    score: Dict[str, float] = {}
    for w, lst in zip(ws, lists):
        for i, nid in enumerate(lst, 1):
            score[nid] = score.get(nid, 0.0) + w / (k + i)
    return sorted(score, key=lambda n: -score[n])
