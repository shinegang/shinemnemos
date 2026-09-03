# -*- coding: utf-8 -*-
"""ShineMnemos: обязательный проход ответа через граф памяти (Grounded Answer).

Приказ Ильи 03.09: перед КАЖДЫМ ответом агент обязан пройти через граф —
чтобы пользователь не жёг токены вникуда, а ответ опирался на свою память,
а не на выдумку модели. Здесь — механика этого прохода.

Протокол (три шага, все три пишутся в append-only журнал ground_log.jsonl):

  1. ДО ответа  — memory_ground_prepare(query): бюджетный поиск по графу +
     сборка system-prompt из найденных узлов. Регистрирует «пред-проход»
     (pre-pass) сессии: время, запрос, id выданных узлов. Без этого шага
     ответ на шаге 3 помечается ungrounded (reason=no_pre_pass) — сколько бы
     утверждений он ни подтвердил.
  2. Ответ      — генерирует клиент (LLM) поверх полученного промпта. Либо
     не генерирует вовсе: см. graph_first() — режим «ноль токенов».
  3. ПОСЛЕ      — memory_ground(answer_text): ответ режется на утверждения,
     каждое сверяется с графом (покрытие по idf-взвешенным основам слов +
     жёсткая сверка чисел), выдаётся вердикт по утверждению и по ответу
     целиком: grounded / partial / ungrounded, плюс список узлов-источников.

Почему покрытие, а не «семантическая похожесть»: рантайм Mnemos — stdlib,
без моделей (fastembed опционален и на проде не стоит). Покрытие по основам
слов с весами idf — тот же сигнал, на котором работает budget-поиск, то есть
grounding меряет ровно ту память, которую агент реально мог прочитать.

Числа — отдельный жёсткий гейт. Выдумка модели чаще всего выглядит как
правильные слова с неправильной цифрой («вес 0.9», «14 сделок»), и покрытие
по словам такую подмену пропускает: слова-то из графа. Поэтому число из
утверждения, которого нет в узле-опоре, роняет вердикт до unsupported
(reason=number_mismatch) даже при покрытии 1.0.

Только stdlib.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from .budget import KIND_HUB, estimate_tokens, node_text, norm, tokens
from .bus import locked_file

# --- пороги (все вынесены сюда: приказ требует «порог», а не магию в коде) ---

SUPPORT_STRONG = 0.60   # покрытие утверждения графом -> supported
SUPPORT_WEAK = 0.30     # покрытие -> partial (ниже — unsupported)
GROUNDED_RATIO = 0.80   # доля подтверждённых утверждений -> grounded
PARTIAL_RATIO = 0.40    # доля -> partial (ниже — ungrounded)

# «Пред-проход» протухает: поиск, сделанный вчера, не заземляет сегодняшний
# ответ — граф между ними мог измениться (decay в 04:00, новые узлы).
PRE_PASS_TTL_SECONDS = 3600.0

MAX_CLAIMS = 24         # потолок разбора ответа (защита от простыни на 10 КБ)
MIN_CLAIM_TOKENS = 2    # короче — не утверждение, а связка («Итак, вот:»)
TOP_SUPPORT = 3         # сколько узлов-опор показывать на утверждение
CANDIDATE_TOKENS = 8    # по скольким самым редким основам собирать кандидатов
MAX_CANDIDATES = 400    # потолок кандидатов на утверждение

# --- graph-first: ответ прямо из графа, без вызова LLM (ноль токенов) --------
# Порог откалиброван на 15 запросах ground_truth.json (замер 03.09,
# eval_grounding.py, таблица «КАЛИБРОВКА ПОРОГА»): 0.70 и выше не срабатывает
# ни разу, 0.60-0.65 срабатывают на 2-3 запросах и НИ РАЗУ не отдают узел не
# из эталона. Взят верхний край этого коридора: выборка мала (15 запросов), а
# цена ошибки тут — молча отданный пользователю неверный ответ вместо честного
# «не знаю». Клиент может опустить порог параметром threshold.
GRAPH_FIRST_COVERAGE = 0.65    # покрытие вопроса узлом (recall вопроса узлом)
GRAPH_FIRST_WEIGHT = 0.50      # вес узла с учётом затухания
GRAPH_FIRST_CONFIDENCE = 0.50  # уверенность узла (truth-gate score/6)
GRAPH_FIRST_MARGIN = 1.15      # во сколько раз лидер должен обойти второго

GROUND_LOG_NAME = "ground_log.jsonl"
GROUND_LOG_MAX_TAIL = 512 * 1024  # сколько байт хвоста журнала читать
# Потолки записи (баг-хант 03.09, D6). Ответ обрезался и раньше
# (answer_preview), а вопрос — нет: один вызов с мегабайтным query давал
# мегабайтную строку журнала. Журнал лежит рядом со стором, то есть у нас — в
# git-репозитории с автопушем каждые 5 минут; расти без границ ему нельзя.
GROUND_LOG_MAX_TEXT = 400          # вопрос и прочие длинные строковые поля
GROUND_LOG_MAX_BYTES = 16 * 1024 * 1024  # ротация: .jsonl -> .jsonl.1
GROUND_LOG_TRUNC_MARK = "…[обрезано]"
# Поля, которые обрезать нельзя: это идентификаторы и хеши — обрезанные, они
# врут (sha перестаёт сходиться, session_id перестаёт джойниться). Их длину
# ограничивает сервер на входе (MAX_ID_LEN), а не журнал на выходе.
GROUND_LOG_KEEP_WHOLE = ("answer_sha256", "session_id", "event", "ts")

# Шаблон для клиентов (приказ, п.2). Отдаётся в memory_ground_prepare, чтобы
# контракт ехал вместе с промптом, а не только жил в README.
SYSTEM_PROMPT_TEMPLATE = (
    "У тебя есть память команды (MCP-сервер ShineMnemos). Порядок работы "
    "обязателен и нарушать его нельзя:\n"
    "1. ДО ответа вызови memory_ground_prepare(query=<вопрос пользователя>, "
    "session_id=<id диалога>). Он вернёт выдержку из графа памяти и "
    "graph_first — готовый ответ, если он в памяти уже есть.\n"
    "2. Если graph_first.hit = true — отдай graph_first.answer как есть и "
    "НЕ генерируй свой текст: ответ уже в памяти, генерация сожжёт токены "
    "пользователя впустую.\n"
    "3. Иначе отвечай ТОЛЬКО на фактах из выданной выдержки. Чего нет в "
    "выдержке — того ты не знаешь: так и напиши, не додумывай.\n"
    "4. ПОСЛЕ ответа вызови memory_ground(answer_text=<твой ответ>, "
    "session_id=<тот же id>). Если вердикт не grounded — покажи "
    "пользователю неподтверждённые утверждения из unsupported_claims "
    "и не выдавай их за факты."
)

# Разметка, которую надо снять перед разбором. Подчёркивание сюда НЕ входит
# специально: в этом графе оно живёт внутри путей и имён (proxy_pool.txt,
# alice_signal_decision, /opt/migration_backup/...), то есть ровно в самых
# различающих токенах. Стирая «_» как markdown-выделение, мы разбивали такой
# токен надвое и теряли опору: узел с точной цитатой пути переставал
# подтверждать ответ с этим путём.
_MD_NOISE = re.compile(
    r"\*+|`+|~~|^[ \t]*#{1,6}[ \t]+|^[ \t]*>[ \t]?", re.MULTILINE
)
_BULLET = re.compile(r"^\s*(?:[-*•—]|\d+[.)])\s+")
_SENT_SPLIT = re.compile(
    r"(?<=[.!?…])\s+(?=[«\"'(\[]?[A-ZА-ЯЁ0-9])"   # конец предложения
    r"|[\n\r]+"                                     # перевод строки
    r"|\s*;\s+"                                     # точка с запятой
)
# Число: 12, 0.75, 1,5, 2x3090, 12345678, 13:10, 90%. Проценты и разделители
# нормализуются в _numbers, чтобы «0,75» и «0.75» считались одним числом.
_NUM = re.compile(r"\d+(?:[.,:]\d+)*")
# Служебные зачины — это не утверждения о мире, проверять их нечего.
_META = re.compile(
    r"^\s*(вот|итак|ниже|коротко|кратко|резюме|итого|проверил|проверила|"
    r"смотри|смотрите|отвечаю|поясню|например|то есть|здесь|тут|"
    r"here|so|ok|okay|summary|in short|note)\b[\s,:—-]*$",
    re.IGNORECASE,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime] = None) -> str:
    return (dt or _now()).isoformat(timespec="milliseconds")


def _parse_iso(ts: Any) -> Optional[datetime]:
    if not isinstance(ts, str) or not ts.strip():
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


# ============================================================================
# Разбор ответа на утверждения
# ============================================================================

def split_claims(text: str, max_claims: int = MAX_CLAIMS) -> List[str]:
    """Режет ответ агента на проверяемые утверждения.

    Выбрасывает вопросы (агент спрашивает, а не утверждает), служебные зачины
    и обрывки короче MIN_CLAIM_TOKENS значимых основ: «Итак:» проверять нечем,
    и попытка это сделать только размывает итоговую долю.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for piece in _SENT_SPLIT.split(text):
        if piece is None:
            continue
        s = _BULLET.sub("", _MD_NOISE.sub(" ", piece)).strip()
        if s.endswith("?"):
            continue  # вопрос — не утверждение, проверять нечего
        s = s.strip(" \t\r\n-—:;.!…")
        if not s or _META.match(s):
            continue
        if len(tokens(s)) < MIN_CLAIM_TOKENS:
            continue
        key = norm(s)
        if key in seen:      # повтор одного и того же тезиса не должен
            continue         # ни улучшать, ни ухудшать долю подтверждённых
        seen.add(key)
        out.append(s)
        if len(out) >= max_claims:
            break
    return out


def _norm_num(raw: str) -> str:
    """Число в нормальной форме: 0,75 -> 0.75, 12.50 -> 12.5, «90%» -> 90."""
    v = raw.replace(",", ".")
    if "." in v and ":" not in v:
        v = v.rstrip("0").rstrip(".") or "0"
    return v


def _numbers(text: str) -> Set[str]:
    """Множество чисел текста (без контекста)."""
    return {_norm_num(m) for m in _NUM.findall(str(text or ""))}


_WORD = re.compile(r"[^\s]+")


def number_contexts(text: str) -> Dict[str, Set[str]]:
    """Числа текста с соседями: {число: {"<основа-слева", ">основа-справа"}}.

    Голой сверки множеств чисел мало, и это видно на замере 03.09: ответ
    «ПРАВИЛО 2 Акмеа» подтверждался узлом «ПРАВИЛО 1 Акмеа ... В2 свежее
    доказательство» — двойка в узле есть, но совсем не та. Соседнее слово
    привязывает число к тому, что оно измеряет.

    Соседи берутся по СЛОВАМ (не по токенам поиска), потому что число часто
    сидит внутри слова: «В2», «2x3090», «ЧЕК-5».

    Ключ «=основа» — само слово, внутри которого стоит число. Оно сильнее
    любых соседей: если и в ответе, и в узле написано «0xE606...», число
    подтверждено, что бы вокруг ни стояло. Без этого ключа честный пересказ,
    выбросивший соседнее слово, ловил ложную тревогу (замер 03.09, D12).
    """
    words = _WORD.findall(str(text or ""))
    stems = [stem_of(w) for w in words]
    out: Dict[str, Set[str]] = defaultdict(set)
    for i, w in enumerate(words):
        for m in _NUM.findall(w):
            v = _norm_num(m)
            if stems[i]:
                out[v].add("=" + stems[i])
            if i > 0 and stems[i - 1]:
                out[v].add("<" + stems[i - 1])
            if i + 1 < len(words) and stems[i + 1]:
                out[v].add(">" + stems[i + 1])
    return out


def stem_of(word: str) -> str:
    """Основа слова-соседа: та же нормализация, что у поиска (6 символов)."""
    toks = tokens(word, min_len=2)
    return toks[0] if toks else ""


def numbers_missing(claim_ctx: Dict[str, Set[str]],
                    node_ctx: Dict[str, Set[str]]) -> List[str]:
    """Числа утверждения, которых узел-опора не подтверждает.

    Число считается подтверждённым, если оно есть в узле И совпал хотя бы
    один сосед (слева или справа). Достаточно одного соседа: пересказ
    своими словами переставляет и выбрасывает слова, и требование обоих
    давало бы ложные тревоги на честном пересказе. Если соседей нет ни у
    одной из сторон (число стоит особняком) — засчитываем голое совпадение.
    """
    missing: List[str] = []
    for num, ctx in claim_ctx.items():
        if num not in node_ctx:
            missing.append(num)
            continue
        node_side = node_ctx[num]
        if ctx and node_side and not (ctx & node_side):
            missing.append(num)
    return sorted(missing)


# ============================================================================
# Индекс опор поверх BudgetSearch
# ============================================================================

class SupportIndex:
    """Обратный индекс основа -> узлы поверх готового BudgetSearch.

    Строится один раз на версию стора и кэшируется на самом движке: движок
    Store._budget_engine() пересобирается при каждой записи (_version), значит
    и индекс опор не переживёт изменение графа.
    """

    __slots__ = ("engine", "inv", "nums", "unseen_idf")

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.inv: Dict[str, List[str]] = defaultdict(list)
        self.nums: Dict[str, Dict[str, Set[str]]] = {}
        for nid, n in engine.nodes.items():
            if n.get("kind") == KIND_HUB:
                continue  # хаб — навигация, а не факт: опорой быть не может
            for s in engine.toks.get(nid, ()):
                self.inv[s].append(nid)
            self.nums[nid] = number_contexts(node_text(n))
        # Вес основы, которой в графе НЕТ ВООБЩЕ. BudgetSearch подставляет
        # таким основам 1.0 — для поиска это неважно (их всё равно никто не
        # matched), а для grounding это переворачивает смысл: слово, которого
        # память не знает, — самый сильный признак выдумки, а получает вес
        # меньше, чем любая известная редкая основа (на 5000 узлов idf редкой
        # основы ~8.5 против 1.0). Тогда ответ, где половина слов выдумана,
        # набирал высокое покрытие за счёт второй половины. Незнакомая основа
        # весит как самая редкая известная.
        self.unseen_idf = max(engine.idf.values(), default=1.0)

    @classmethod
    def of(cls, engine: Any) -> "SupportIndex":
        idx = getattr(engine, "_ground_index", None)
        if idx is None:
            idx = cls(engine)
            try:
                engine._ground_index = idx
            except AttributeError:  # движок со __slots__ — просто не кэшируем
                pass
        return idx

    def idf(self, token: str) -> float:
        """Вес основы: известной — её idf, незнакомой — вес самой редкой."""
        return self.engine.idf.get(token, self.unseen_idf)

    def qmass(self, qtoks: Iterable[str]) -> float:
        """Полная «масса» утверждения — знаменатель покрытия."""
        return sum(self.idf(t) for t in qtoks) or 1.0

    def coverage(self, nid: str, qtoks: List[str], qmass: float
                 ) -> Tuple[float, List[str]]:
        """Доля массы утверждения, покрытая узлом (0..1), и что совпало."""
        ntoks = self.engine.toks.get(nid) or set()
        matched = [t for t in qtoks if t in ntoks]
        got = sum(self.idf(t) for t in matched)
        return (got / qmass if qmass else 0.0), matched

    def candidates(self, qtoks: List[str]) -> List[str]:
        """Кандидаты в опоры: узлы, содержащие самые редкие основы запроса.

        Полный скан стора на каждое утверждение — это O(утверждений × узлов);
        на 5000 узлов и 24 утверждениях это заметно. Редкие основы дают тот же
        топ дешевле: узел без единой редкой основы утверждения всё равно не
        наберёт покрытия выше порога. Основы, которых в графе нет, из отбора
        выброшены: постинг-лист у них пустой, а место в лимите они занимают.
        """
        known = [t for t in set(qtoks) if t in self.inv]
        rare = sorted(known, key=lambda t: -self.idf(t))[:CANDIDATE_TOKENS]
        seen: Dict[str, None] = OrderedDict()
        for t in rare:
            for nid in self.inv.get(t, ()):
                seen.setdefault(nid, None)
                if len(seen) >= MAX_CANDIDATES:
                    return list(seen)
        return list(seen)


def _brief(n: Dict[str, Any], coverage: float, matched: List[str],
           nums_ok: Optional[bool] = None) -> Dict[str, Any]:
    """Краткая карточка узла-опоры. nums_ok=None — сверять было нечего
    (graph-first сверяет вопрос с узлом, а не утверждение ответа)."""
    out = {
        "id": n.get("id"),
        "kind": n.get("kind"),
        "claim": str(n.get("claim") or "")[:200],
        "source": n.get("source"),
        "ts": n.get("ts"),
        "weight": round(float(n.get("weight", 1.0) or 1.0), 4),
        "confidence": round(float(n.get("confidence", 0.5) or 0.5), 4),
        "coverage": round(coverage, 3),
        "matched": matched,
    }
    if nums_ok is not None:
        out["numbers_ok"] = nums_ok
    return out


def verify_claim(engine: Any, claim: str) -> Dict[str, Any]:
    """Сверяет одно утверждение с графом -> вердикт + узлы-опоры.

    Вердикты:
      supported   — покрытие >= SUPPORT_STRONG и все числа утверждения нашлись
                    в узле-опоре;
      partial     — покрытие >= SUPPORT_WEAK (или сильное покрытие, но число
                    не сошлось: слова из памяти, цифра выдумана);
      unsupported — в графе нет опоры;
      refuted     — лучшая опора найдена, но это узел kind=refuted/outdated:
                    ответ опирается на то, что память уже отменила.
    """
    idx = SupportIndex.of(engine)
    qtoks = list(dict.fromkeys(tokens(claim)))
    qnums_ctx = number_contexts(claim)
    qnums = set(qnums_ctx)
    empty = {
        "claim": claim, "verdict": "unsupported", "coverage": 0.0,
        "reason": "в утверждении нет значимых слов для сверки",
        "numbers": {"in_claim": sorted(qnums), "matched": [], "missing": sorted(qnums)},
        "support": [],
    }
    if not qtoks:
        return empty
    qmass = idx.qmass(qtoks)

    live: List[Tuple[float, Dict[str, Any]]] = []
    stale: List[Tuple[float, Dict[str, Any]]] = []
    for nid in idx.candidates(qtoks):
        n = engine.nodes.get(nid)
        if n is None:
            continue
        cov, matched = idx.coverage(nid, qtoks, qmass)
        if cov <= 0.0:
            continue
        missing = numbers_missing(qnums_ctx, idx.nums.get(nid, {}))
        brief = _brief(n, cov, matched, not missing)
        brief["numbers_missing"] = missing
        (stale if n.get("kind") in ("refuted", "outdated") else live).append((cov, brief))
    live.sort(key=lambda t: -t[0])
    stale.sort(key=lambda t: -t[0])

    if not live and not stale:
        return {**empty, "reason": "ни один узел графа не содержит слов утверждения"}

    best_cov, best = (live[0] if live else (0.0, None))
    stale_cov, stale_best = (stale[0] if stale else (0.0, None))

    # Опора на отменённую память — отдельный, самый громкий вердикт: ответ
    # звучит подтверждённым, а подтверждает его узел, который память отменила.
    if stale_best is not None and stale_cov >= SUPPORT_STRONG and stale_cov >= best_cov:
        return {
            "claim": claim, "verdict": "refuted", "coverage": round(stale_cov, 3),
            "reason": (
                f"лучшая опора — узел {stale_best['id']} вида "
                f"{stale_best['kind']}: память это утверждение отменила"
            ),
            "numbers": {"in_claim": sorted(qnums),
                        "matched": sorted(qnums - set(stale_best["numbers_missing"])),
                        "missing": stale_best["numbers_missing"]},
            "support": [s for _, s in stale[:TOP_SUPPORT]],
        }

    if best is None:
        best_cov, best = stale_cov, stale_best

    # Числа сверяем по узлу-опоре, а не по всему графу: «0.944» из другого
    # исследования не подтверждает цифру в этом утверждении.
    missing = best["numbers_missing"]
    matched_nums = sorted(qnums - set(missing))
    if best_cov >= SUPPORT_STRONG and not missing:
        verdict, reason = "supported", f"покрытие {best_cov:.2f} узлом {best['id']}"
    elif best_cov >= SUPPORT_STRONG and missing:
        verdict = "partial"
        reason = (
            f"слова из памяти (покрытие {best_cov:.2f}), но чисел "
            f"{', '.join(missing)} нет в опоре {best['id']} — number_mismatch"
        )
    elif best_cov >= SUPPORT_WEAK:
        verdict = "partial"
        reason = f"частичное покрытие {best_cov:.2f} узлом {best['id']}"
    else:
        verdict = "unsupported"
        reason = f"покрытие {best_cov:.2f} ниже порога {SUPPORT_WEAK}"

    return {
        "claim": claim, "verdict": verdict, "coverage": round(best_cov, 3),
        "reason": reason,
        "numbers": {"in_claim": sorted(qnums), "matched": matched_nums,
                    "missing": missing},
        "support": [s for _, s in (live or stale)[:TOP_SUPPORT]],
    }


# ============================================================================
# Итоговый вердикт по ответу
# ============================================================================

def _aggregate(claims: List[Dict[str, Any]]) -> Tuple[str, float, Dict[str, int]]:
    counts = {"total": len(claims), "supported": 0, "partial": 0,
              "unsupported": 0, "refuted": 0}
    for c in claims:
        counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
    if not claims:
        return "ungrounded", 0.0, counts
    # частичное подтверждение считаем половиной: ответ, где всё «похоже на
    # правду, но без опоры», не должен получать тот же вердикт, что ответ,
    # каждое утверждение которого лежит в графе.
    ratio = (counts["supported"] + 0.5 * counts["partial"]) / counts["total"]
    if not counts["supported"]:
        # «частично прошёл через граф» обязано означать «часть ответа граф
        # подтвердил». Без единого подтверждённого утверждения подтверждать
        # нечего: замер 03.09 показал 5 ответов, выдуманных целиком, которым
        # одно только лексическое сходство («кошелёк», «ключи», «трейдера»)
        # давало ratio 0.5 и успокаивающий вердикт partial.
        return "ungrounded", round(ratio, 3), counts
    if counts["refuted"]:
        verdict = "ungrounded" if ratio < GROUNDED_RATIO else "partial"
    elif ratio >= GROUNDED_RATIO and not counts["unsupported"]:
        # grounded — это обещание «в ответе нет ничего мимо памяти», поэтому
        # одного неподтверждённого утверждения достаточно, чтобы его снять.
        # Иначе длина ответа разбавляет выдумку: замер 03.09, D03 — пять
        # подтверждённых предложений и одно ложное давали ratio 0.83 и
        # вердикт grounded, хотя ложное утверждение было честно помечено.
        verdict = "grounded"
    elif ratio >= PARTIAL_RATIO:
        verdict = "partial"
    else:
        verdict = "ungrounded"
    return verdict, round(ratio, 3), counts


_HUMAN = {"grounded": "да", "partial": "частично", "ungrounded": "нет"}


def ground_answer(
    store: Any,
    answer_text: str,
    query: str = "",
    pre_pass: Optional[Dict[str, Any]] = None,
    require_pre_pass: bool = True,
    max_claims: int = MAX_CLAIMS,
) -> Dict[str, Any]:
    """Прогоняет готовый ответ агента через граф.

    pre_pass — запись пред-прохода (см. GroundLog.find_pre_pass) или None.
    require_pre_pass=True (умолчание, приказ Ильи): без пред-прохода итоговый
    вердикт — ungrounded, даже если все утверждения подтвердились. Вердикт по
    самим утверждениям при этом не теряется: он лежит в claims_verdict.
    """
    with store._lock:
        engine = store._budget_engine()
    claim_texts = split_claims(answer_text, max_claims=max_claims)
    claims = [verify_claim(engine, c) for c in claim_texts]
    claims_verdict, ratio, counts = _aggregate(claims)

    pre_ok = bool(pre_pass)
    verdict = claims_verdict
    notes: List[str] = []
    if require_pre_pass and not pre_ok:
        verdict = "ungrounded"
        notes.append(
            "no_pre_pass: перед ответом не было memory_ground_prepare/"
            "memory_prompt по этой сессии — либо пред-проход был, но не нашёл "
            "в графе ни одного узла, что то же самое: опереться было не на что"
        )
    if not claim_texts:
        notes.append("в ответе не нашлось проверяемых утверждений")

    # узлы-источники: уникальные опоры подтверждённых и частичных утверждений
    sources: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for c in claims:
        if c["verdict"] in ("unsupported",):
            continue
        for s in c["support"]:
            prev = sources.get(s["id"])
            if prev is None or s["coverage"] > prev["coverage"]:
                sources[s["id"]] = s
    ordered = sorted(sources.values(), key=lambda s: -s["coverage"])

    return {
        "verdict": verdict,
        "passed_through_graph": _HUMAN[verdict],
        "claims_verdict": claims_verdict,
        "grounded_ratio": ratio,
        "counts": counts,
        "query": query,
        "pre_pass": pre_pass or {"present": False},
        "require_pre_pass": bool(require_pre_pass),
        "notes": notes,
        "claims": claims,
        "unsupported_claims": [
            {"claim": c["claim"], "verdict": c["verdict"], "reason": c["reason"]}
            for c in claims if c["verdict"] in ("unsupported", "refuted")
        ],
        "source_nodes": ordered,
        "source_node_ids": [s["id"] for s in ordered],
        "thresholds": {
            "support_strong": SUPPORT_STRONG, "support_weak": SUPPORT_WEAK,
            "grounded_ratio": GROUNDED_RATIO, "partial_ratio": PARTIAL_RATIO,
        },
        "answer_tokens": estimate_tokens(answer_text),
    }


# ============================================================================
# graph-first: ответ из графа без вызова LLM (ноль токенов генерации)
# ============================================================================

def graph_first(
    store: Any,
    query: str,
    coverage_threshold: float = GRAPH_FIRST_COVERAGE,
    min_weight: float = GRAPH_FIRST_WEIGHT,
    min_confidence: float = GRAPH_FIRST_CONFIDENCE,
    margin: float = GRAPH_FIRST_MARGIN,
    search: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Есть ли в графе готовый ответ — так, чтобы LLM звать не пришлось.

    Условия попадания (все сразу, иначе hit=false):
      * узел покрывает вопрос на coverage_threshold и выше (recall вопроса);
      * узел живой (не refuted/outdated/hub, TTL не истёк) и это факт/правило;
      * вес с учётом затухания >= min_weight, уверенность >= min_confidence;
      * лидер обходит второго кандидата не меньше чем в margin раз — иначе
        в графе два разных ответа, и выбирать между ними должна модель,
        а не порог.

    search — уже посчитанный store.search_budget(query) (чтобы не искать дважды).
    """
    q = str(query or "").strip()
    if not q:
        return {"hit": False, "reason": "пустой запрос", "answer": None}
    if search is None:
        search = store.search_budget(q)
    results = list(search.get("results") or [])
    if not results:
        return {"hit": False, "reason": "граф ничего не нашёл по запросу",
                "answer": None, "candidates": 0}

    with store._lock:
        engine = store._budget_engine()
    idx = SupportIndex.of(engine)
    qtoks = list(dict.fromkeys(tokens(q)))
    qmass = idx.qmass(qtoks)
    now = _now()

    scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for r in results:
        n = engine.nodes.get(r["id"])
        if n is None:
            continue
        cov, matched = idx.coverage(r["id"], qtoks, qmass)
        scored.append((cov, r, {"matched": matched}))
    scored.sort(key=lambda t: -t[0])
    if not scored:
        return {"hit": False, "reason": "кандидаты исчезли из стора",
                "answer": None, "candidates": 0}

    cov, top, why = scored[0]
    node = engine.nodes[top["id"]]
    from .budget import _active, _decayed_weight  # локально: только тут нужны

    weight = _decayed_weight(node, now)
    conf = float(node.get("confidence", 0.5) or 0.5)
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    checks = {
        "coverage": {"value": round(cov, 3), "need": coverage_threshold,
                     "pass": cov >= coverage_threshold},
        "weight": {"value": round(weight, 3), "need": min_weight,
                   "pass": weight >= min_weight},
        "confidence": {"value": round(conf, 3), "need": min_confidence,
                       "pass": conf >= min_confidence},
        "kind": {"value": node.get("kind"), "need": "fact|rule",
                 "pass": node.get("kind") in ("fact", "rule")},
        "active": {"value": True, "need": True, "pass": _active(node, now)},
        "margin": {"value": round(cov / runner_up, 3) if runner_up > 0 else None,
                   "need": margin,
                   "pass": runner_up <= 0.0 or cov >= runner_up * margin},
    }
    failed = [k for k, v in checks.items() if not v["pass"]]
    if failed:
        return {
            "hit": False, "answer": None,
            "reason": "не прошли пороги graph-first: " + ", ".join(failed),
            "checks": checks, "candidates": len(scored),
            "best_node": top["id"], "best_coverage": round(cov, 3),
        }
    return {
        "hit": True,
        "answer": str(node.get("claim") or ""),
        "node_id": top["id"],
        "node": _brief(node, cov, why["matched"]),
        "checks": checks,
        "candidates": len(scored),
        "llm_calls_saved": 1,
        # Честная бухгалтерия: сэкономлено ровно столько, сколько стоило бы
        # уехать в модель — контекст из памяти (мы его измерили) плюс сам
        # ответ. Размер генерации модели заранее неизвестен, поэтому здесь
        # только нижняя граница, и она так и подписана.
        "tokens_saved_min": int(search.get("tokens_used") or 0)
        + estimate_tokens(str(node.get("claim") or "")),
        "tokens_saved_note": (
            "нижняя граница: контекст памяти "
            f"{int(search.get('tokens_used') or 0)} токенов + ответ "
            f"{estimate_tokens(str(node.get('claim') or ''))}; генерация модели "
            "не учтена (её размер заранее неизвестен)"
        ),
    }


# ============================================================================
# Append-only журнал проходов
# ============================================================================

class GroundLog:
    """Append-only JSONL: каждый проход через граф — одна строка.

    Формат строки: {ts, event, agent, session_id, query, verdict, node_ids,
    counts, answer_sha256, answer_preview}. Файл только дописывается: разбор
    «почему агент так ответил» должен опираться на запись, сделанную в тот
    момент, а не на пересобранную задним числом.
    """

    EVENTS = ("prepare", "ground", "answer")

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)

    @staticmethod
    def _clip(key: str, value: Any) -> Any:
        """Длинную строку — под потолок; идентификаторы и хеши — как есть."""
        if not isinstance(value, str) or key in GROUND_LOG_KEEP_WHOLE:
            return value
        if len(value) <= GROUND_LOG_MAX_TEXT:
            return value
        return value[:GROUND_LOG_MAX_TEXT] + GROUND_LOG_TRUNC_MARK

    def _rotate_if_big(self) -> Optional[Path]:
        """Отвести разросшийся журнал в .1 и начать новый. Зовётся ПОД замком.

        Append-only это не нарушает: ни одна запись не переписывается, файл
        целиком уезжает в сторону. Храним одно поколение — журнал живёт рядом
        со стором (у нас — в git-репозитории с автопушем), и расти бесконечно
        ему нельзя. Читатель и так смотрит только хвост GROUND_LOG_MAX_TAIL.
        """
        try:
            if self.path.stat().st_size < GROUND_LOG_MAX_BYTES:
                return None
        except OSError:
            return None
        rotated = self.path.with_name(self.path.name + ".1")
        try:
            os.replace(self.path, rotated)
        except OSError:
            return None
        return rotated

    def append(self, event: str, **fields: Any) -> Dict[str, Any]:
        if event not in self.EVENTS:
            raise ValueError(f"ground_log: event должен быть из {self.EVENTS}, "
                             f"получено {event!r}")
        rec = {"ts": _iso(), "event": event}
        rec.update({k: self._clip(k, v) for k, v in fields.items() if v is not None})
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with locked_file(self.path):
            self._rotate_if_big()
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line)
                f.flush()
        return rec

    def _tail_lines(self, max_bytes: int = GROUND_LOG_MAX_TAIL) -> List[str]:
        """Хвост журнала: журнал растёт вечно, читать его целиком нельзя."""
        if not self.path.exists():
            return []
        try:
            size = self.path.stat().st_size
            with open(self.path, "rb") as f:
                if size > max_bytes:
                    f.seek(size - max_bytes)
                    f.readline()  # первая строка обрезана посередине — выкинуть
                raw = f.read()
        except OSError:
            return []
        return raw.decode("utf-8", errors="replace").splitlines()

    def read(self, limit: int = 50, session_id: Optional[str] = None,
             event: Optional[str] = None, agent: Optional[str] = None,
             ) -> List[Dict[str, Any]]:
        """Последние записи журнала (новые в конце), с фильтрами."""
        out: List[Dict[str, Any]] = []
        for raw in self._tail_lines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue  # оборванная строка параллельного писателя — пропуск
            if not isinstance(rec, dict):
                continue
            if session_id is not None and rec.get("session_id") != session_id:
                continue
            if event is not None and rec.get("event") != event:
                continue
            if agent is not None and rec.get("agent") != agent:
                continue
            out.append(rec)
        return out[-max(1, int(limit)):] if out else []

    def find_pre_pass(self, session_id: str, ttl_seconds: Optional[float] = None,
                      now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """Свежий пред-проход этой сессии или None (протух/не было).

        ttl_seconds=None читает PRE_PASS_TTL_SECONDS в момент вызова, а не
        в момент объявления функции: иначе порог нельзя было бы поменять
        в рантайме, а значение из значения по умолчанию застывало навсегда.
        """
        if not session_id:
            return None
        ttl = PRE_PASS_TTL_SECONDS if ttl_seconds is None else float(ttl_seconds)
        now = now or _now()
        for rec in reversed(self.read(limit=2000, session_id=session_id)):
            if rec.get("event") not in ("prepare", "answer"):
                continue
            ts = _parse_iso(rec.get("ts"))
            if ts is None:
                continue
            age = (now - ts).total_seconds()
            if age > ttl:
                return None  # записи идут по времени: дальше только старее
            if not (rec.get("node_ids") or []):
                # то же правило, что в SessionTracker.get (D2): последний
                # пред-проход этой сессии ничего не нашёл — значит прохода нет
                return None
            return {
                "present": True, "session_id": session_id, "ts": rec.get("ts"),
                "age_seconds": round(age, 3), "tool": rec.get("tool"),
                "query": rec.get("query"), "node_ids": rec.get("node_ids") or [],
                "source": "log",
            }
        return None


class SessionTracker:
    """Пред-проходы в памяти процесса — быстрый путь для find_pre_pass.

    Журнал остаётся источником правды (переживает рестарт), но ходить в файл
    на каждый memory_ground не нужно: ответ приходит через секунды после
    подготовки, в том же процессе.
    """

    def __init__(self, capacity: int = 512) -> None:
        self.capacity = max(1, int(capacity))
        self._data: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def register(self, session_id: str, **fields: Any) -> None:
        if not session_id:
            return
        self._data[session_id] = {"ts": _iso(), **fields}
        self._data.move_to_end(session_id)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def get(self, session_id: str, ttl_seconds: Optional[float] = None,
            now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        rec = self._data.get(session_id)
        if rec is None:
            return None
        if not rec.get("node_ids"):
            # Пустой пред-проход — не проход (баг-хант 03.09, D2). Поиск,
            # который не нашёл в графе НИЧЕГО, не мог заземлить ответ: засчитав
            # его, мы выдавали бы «прошёл через граф: да» за один лишь факт
            # вызова инструмента с этим session_id.
            return None
        ts = _parse_iso(rec.get("ts"))
        if ts is None:
            return None
        ttl = PRE_PASS_TTL_SECONDS if ttl_seconds is None else float(ttl_seconds)
        age = ((now or _now()) - ts).total_seconds()
        if age > ttl:
            return None
        return {"present": True, "session_id": session_id, "ts": rec.get("ts"),
                "age_seconds": round(age, 3), "tool": rec.get("tool"),
                "query": rec.get("query"), "node_ids": rec.get("node_ids") or [],
                "source": "memory"}


def answer_sha256(text: str) -> str:
    """Отпечаток ответа для журнала: сам ответ в журнал целиком не кладём."""
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()
