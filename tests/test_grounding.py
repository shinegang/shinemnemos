# -*- coding: utf-8 -*-
"""Обязательный проход через граф (Grounded Answer, приказ Ильи 03.09).

Три сценария из приказа:
  (а) ответ, подкреплённый графом            -> grounded;
  (б) ответ с выдуманным фактом              -> ungrounded-пометка;
  (в) graph-first отдаёт ответ из графа       -> LLM не вызывается.
Плюс: журнал проходов append-only, пред-проход и его протухание, числа.
"""

import json
import time

import pytest

from mnemos import grounding
from mnemos.server import MnemosCore
from mnemos.store import Store


# --- корпус ------------------------------------------------------------------

CORPUS = [
    {
        "claim": "Боевой бот AcmeTrader работает на арендованной машине TW с двумя картами 3090, инстанс vast 12345678",
        "source": "инвентаризация инфраструктуры 02.09",
        "evidence": ["vast.ai list instances 02.09"],
        "context": "Где физически исполняется боевой бот",
        "tags": ["инфра"],
    },
    {
        "claim": "Судья Алиса восстановлен 02.09 по спецификации, вердикты пишутся в журнал и в граф",
        "source": "журнал восстановления 02.09",
        "evidence": ["tools/alice_signal_decision.py"],
        "context": "Восстановление судьи после потери 5090",
        "tags": ["торговля"],
    },
    {
        "claim": "Генератор идей выдаёт цикл раз в 150 секунд, первая сделка после восстановления — STX short в 13:10 UTC",
        "source": "signals_loop.py и журнал бота",
        "evidence": ["tools/signals_loop.py", "журнал 02.09 13:10Z"],
        "context": "Параметры генератора идей после восстановления",
        "tags": ["торговля"],
    },
    {
        "claim": "Recall@5 боевого стора Mnemos после выкатки Ф1 равен 0.9444 на приватном корпусе",
        "source": "eval_recall.py прогон 02.09",
        "evidence": ["eval_recall.py, 36 запросов"],
        "context": "Качество поиска после рефакторинга по письму Qwen",
        "tags": ["память"],
    },
    {
        "claim": "Ручные позиции не размещаем: торгует только бот",
        "source": "Иван, правило 5",
        "evidence": ["правила Акме 23.08"],
        "context": "Правила работы с боевым ботом",
        "kind": "rule",
        "tags": ["правило"],
    },
]

# Узел, который память уже отменила: ответ, опирающийся на него, должен
# получить вердикт refuted, а не supported.
REFUTED = {
    "claim": "Боевой бот AcmeTrader работает на машине с картой 5090 в Гонконге",
    "source": "старая инвентаризация 28.08",
    "evidence": ["устаревший инвентарь"],
    "context": "5090 умерла 02.09 — узел отменён",
    "kind": "refuted",
}


@pytest.fixture
def core(tmp_path):
    """Ядро MCP с наполненным стором. Гейты выключены: корпус тестовый,
    Г4 отклонил бы соседние формулировки как дубли."""
    store = Store(tmp_path / "nodes.json")
    c = MnemosCore(store, plugins=[])
    for spec in CORPUS:
        c.memory_add(dict(spec))
    node = c.memory_add({k: v for k, v in REFUTED.items() if k != "kind"})
    got = store.get(node["id"])
    got["kind"] = "refuted"
    store.update(got)
    return c


# --- (а) ответ, подкреплённый графом -> grounded ------------------------------

def test_a_grounded_answer(core):
    prep = core.memory_ground_prepare({
        "query": "Где работает боевой бот и что с судьёй Алиса?",
        "session_id": "s-a", "agent": "fable",
    })
    assert prep["session_id"] == "s-a"
    assert prep["node_ids"], "пред-проход обязан вернуть узлы графа"

    answer = (
        "Боевой бот AcmeTrader работает на арендованной машине TW с двумя "
        "картами 3090, инстанс vast 12345678. "
        "Судья Алиса восстановлен 02.09 по спецификации, вердикты пишутся "
        "в журнал и в граф."
    )
    out = core.memory_ground({"answer_text": answer, "session_id": "s-a",
                              "agent": "fable"})
    assert out["verdict"] == "grounded"
    assert out["passed_through_graph"] == "да"
    assert out["pre_pass"]["present"] is True
    assert out["counts"]["supported"] == out["counts"]["total"] == 2
    assert out["unsupported_claims"] == []
    assert len(out["source_node_ids"]) >= 2
    # узлы-источники подкреплены: память запомнила, что они пригодились
    assert {r["id"] for r in out["reinforced"]} <= set(out["source_node_ids"])


def test_a_without_pre_pass_is_ungrounded(core):
    """Приказ: без пред-прохода ответ ungrounded, даже если он весь из графа."""
    answer = CORPUS[0]["claim"] + "."
    out = core.memory_ground({"answer_text": answer, "session_id": "s-none"})
    assert out["verdict"] == "ungrounded"
    assert out["passed_through_graph"] == "нет"
    assert out["claims_verdict"] == "grounded"  # сверка утверждений не потеряна
    assert any("no_pre_pass" in n for n in out["notes"])

    # require_pre_pass=false — путь для клиентов без сессий
    lax = core.memory_ground({"answer_text": answer, "session_id": "s-none",
                              "require_pre_pass": False})
    assert lax["verdict"] == "grounded"


# --- (б) ответ с выдуманным фактом -> ungrounded-пометка ----------------------

def test_b_fabricated_fact_is_flagged(core):
    core.memory_ground_prepare({"query": "Что с ботом?", "session_id": "s-b"})
    answer = (
        "Боевой бот AcmeTrader работает на арендованной машине TW с двумя "
        "картами 3090. "
        "Бот подключён к бирже Binance и торгует фьючерсы с плечом 50x. "
        "Партнёрское соглашение с фондом Sequoia подписано в июле."
    )
    out = core.memory_ground({"answer_text": answer, "session_id": "s-b"})
    assert out["verdict"] in ("partial", "ungrounded")
    verdicts = [c["verdict"] for c in out["claims"]]
    assert verdicts[0] == "supported"
    assert verdicts[1] == "unsupported"
    assert verdicts[2] == "unsupported"
    bad = {c["claim"] for c in out["unsupported_claims"]}
    assert any("Binance" in c for c in bad)
    assert any("Sequoia" in c for c in bad)


def test_b_number_swap_is_not_supported(core):
    """Слова из графа, цифра выдумана — самый частый вид галлюцинации."""
    core.memory_ground_prepare({"query": "Какой recall у Mnemos?", "session_id": "s-num"})
    out = core.memory_ground({
        "answer_text": "Recall@5 боевого стора Mnemos после выкатки Ф1 равен 0.9871 на приватном корпусе.",
        "session_id": "s-num",
    })
    claim = out["claims"][0]
    assert claim["verdict"] != "supported"
    assert "0.9871" in claim["numbers"]["missing"]
    assert "number_mismatch" in claim["reason"]

    # та же фраза с настоящей цифрой проходит
    ok = core.memory_ground({
        "answer_text": "Recall@5 боевого стора Mnemos после выкатки Ф1 равен 0.9444 на приватном корпусе.",
        "session_id": "s-num",
    })
    assert ok["claims"][0]["verdict"] == "supported"
    assert ok["verdict"] == "grounded"


def test_b_number_needs_the_right_neighbour(core):
    """Голой сверки множеств чисел мало: «ПРАВИЛО 2» не подтверждается узлом,
    где двойка стоит в «В2 свежее доказательство» (замер 03.09, D05)."""
    node = {"claim": "ПРАВИЛО 1 Акмеа: обходные пути запрещены. В2 свежее доказательство в этом ходу",
            "source": "Иван", "evidence": ["правила 23.08"], "context": "Чек-лист"}
    core.memory_add(node)
    eng = core.store._budget_engine()
    bad = grounding.verify_claim(eng, "ПРАВИЛО 2 Акмеа: обходные пути запрещены")
    assert bad["verdict"] != "supported"
    assert "2" in bad["numbers"]["missing"]
    good = grounding.verify_claim(eng, "ПРАВИЛО 1 Акмеа: обходные пути запрещены")
    assert good["verdict"] == "supported"


def test_b_number_survives_dropped_neighbour(core):
    """И наоборот: пересказ, выбросивший соседнее слово, не должен ловить
    ложную тревогу, если само слово с числом совпало (замер 03.09, D12)."""
    core.memory_add({
        "claim": "Продакшн payTo-кошелёк платёжки: 0xE606 настроен в .env",
        "source": "apix402-repo/.env", "evidence": [".env"], "context": "Куда идут платежи",
    })
    out = grounding.verify_claim(core.store._budget_engine(),
                                 "Продакшн payTo-кошелёк 0xE606")
    assert out["verdict"] == "supported", out["reason"]
    assert out["numbers"]["missing"] == []


def test_b_all_claims_partial_is_not_partial_verdict(core):
    """Ни одного подтверждённого утверждения — значит «частично» нельзя:
    подтверждать нечего (замер 03.09: 5 выдумок получали успокаивающий partial)."""
    core.memory_ground_prepare({"query": "Чей кошелёк?", "session_id": "s-z"})
    core.memory_add({
        "claim": "Продакшн payTo-кошелёк платёжки настроен в .env, боевые ключи трейдера не трогаем",
        "source": "apix402", "evidence": [".env"], "context": "Платежи",
    })
    out = core.memory_ground({
        "answer_text": "Платежи получает личный кошелёк трейдера, ключи от него лежат в репозитории.",
        "session_id": "s-z"})
    assert out["counts"]["supported"] == 0
    assert out["verdict"] == "ungrounded"


def test_b_one_fabrication_blocks_grounded(core):
    """Длинный правдивый ответ не должен разбавлять одну выдумку до grounded.

    Вопрос пред-прохода намеренно такой, чтобы граф на него ЧТО-ТО отдал:
    после фикса D2 (03.09) пустой пред-проход перестал считаться проходом, и
    вердикт роняется до ungrounded ещё до подсчёта долей — а тест здесь про
    доли, а не про политику. Пустой случай проверяет
    test_ground_default.py::test_empty_pre_pass_does_not_count.
    """
    core.memory_ground_prepare({"query": "Что известно про бота, Алиса и "
                                         "генератор идей?", "session_id": "s-mix"})
    answer = " ".join(c["claim"] + "." for c in CORPUS[:4]) + \
        " Партнёрское соглашение с фондом Sequoia подписано в июле."
    out = core.memory_ground({"answer_text": answer, "session_id": "s-mix"})
    assert out["counts"]["supported"] == 4
    assert out["counts"]["unsupported"] == 1
    assert out["grounded_ratio"] >= 0.8      # доля сама по себе высокая
    assert out["verdict"] == "partial"        # но одно утверждение мимо памяти
    assert out["passed_through_graph"] == "частично"


def test_b_answer_leaning_on_refuted_node(core):
    core.memory_ground_prepare({"query": "Где бот?", "session_id": "s-ref"})
    out = core.memory_ground({
        "answer_text": "Боевой бот AcmeTrader работает на машине с картой 5090 в Гонконге.",
        "session_id": "s-ref",
    })
    assert out["claims"][0]["verdict"] == "refuted"
    assert out["verdict"] == "ungrounded"
    assert out["counts"]["refuted"] == 1


# --- (в) graph-first: ответ из графа без LLM ----------------------------------

def test_c_graph_first_answers_without_llm(core):
    out = core.memory_answer({
        "query": "Ручные позиции не размещаем: торгует только бот",
        "session_id": "s-c",
    })
    assert out["hit"] is True
    assert out["llm_required"] is False
    assert out["answer"] == "Ручные позиции не размещаем: торгует только бот"
    assert "prompt" not in out, "при попадании промпт для LLM не собирается"
    assert out["graph_first"]["llm_calls_saved"] == 1
    assert out["graph_first"]["tokens_saved_min"] > 0
    for name, chk in out["graph_first"]["checks"].items():
        assert chk["pass"], f"порог {name} должен быть пройден"


def test_c_graph_first_miss_falls_back_to_prompt(core):
    out = core.memory_answer({"query": "Какая погода в Бангкоке сегодня?",
                              "session_id": "s-c2"})
    assert out["hit"] is False
    assert out["llm_required"] is True
    assert out["answer"] is None
    assert "policy" in out and "memory_ground" in out["policy"]


def test_c_graph_first_refuses_ambiguous_leader(core):
    """Два одинаково подходящих узла — выбирать должна модель, не порог."""
    twin = dict(CORPUS[1])
    twin["claim"] = twin["claim"].replace("восстановлен", "поднят заново")
    core.memory_add(twin)
    out = core.memory_answer({"query": CORPUS[1]["claim"], "session_id": "s-c3"})
    assert out["hit"] is False
    assert "margin" in out["graph_first"]["reason"]


def test_c_graph_first_hit_after_answer_is_grounded(core):
    """memory_answer тоже регистрирует пред-проход: ответ из графа, отданный
    пользователю, обязан проходить memory_ground как grounded."""
    a = core.memory_answer({"query": "Ручные позиции не размещаем: торгует только бот",
                            "session_id": "s-c4"})
    out = core.memory_ground({"answer_text": a["answer"], "session_id": "s-c4"})
    assert out["verdict"] == "grounded"


# --- журнал проходов ----------------------------------------------------------

def test_log_is_append_only_and_complete(core, tmp_path):
    core.memory_ground_prepare({"query": "Где бот?", "session_id": "s-log",
                                "agent": "alice"})
    core.memory_ground({"answer_text": CORPUS[0]["claim"] + ".",
                        "session_id": "s-log", "agent": "alice"})
    path = tmp_path / grounding.GROUND_LOG_NAME
    assert path.exists()
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    events = [r["event"] for r in lines]
    assert events == ["prepare", "ground"]
    prep, ground = lines
    assert prep["agent"] == "alice" and prep["query"] == "Где бот?"
    assert prep["node_ids"]
    assert ground["verdict"] == "grounded"
    assert ground["pre_pass"] is True
    assert ground["answer_sha256"] == grounding.answer_sha256(CORPUS[0]["claim"] + ".")
    assert "claim" in ground["answer_preview"] or ground["answer_preview"]

    # дозапись не перетирает историю
    core.memory_ground_prepare({"query": "И ещё раз", "session_id": "s-log"})
    after = path.read_text(encoding="utf-8").splitlines()
    assert len(after) == 3

    out = core.memory_ground_log({"session_id": "s-log", "stats": True})
    assert out["count"] == 3
    assert out["stats"]["verdicts"] == {"grounded": 1}
    assert out["stats"]["prepare"] == 2


def test_log_read_filters_and_survives_restart(core, tmp_path):
    core.memory_ground_prepare({"query": "Где бот?", "session_id": "s-r",
                                "agent": "fable"})
    # новое ядро на том же каталоге: трекер сессий в памяти пуст, пред-проход
    # должен найтись в журнале
    fresh = MnemosCore(Store(tmp_path / "nodes.json"), plugins=[])
    assert fresh.sessions.get("s-r") is None
    pre = fresh._find_pre_pass("s-r")
    assert pre is not None and pre["source"] == "log"
    out = fresh.memory_ground({"answer_text": CORPUS[0]["claim"] + ".",
                               "session_id": "s-r"})
    assert out["verdict"] == "grounded"


def test_pre_pass_expires(core, monkeypatch):
    core.memory_ground_prepare({"query": "Где бот?", "session_id": "s-ttl"})
    monkeypatch.setattr(grounding, "PRE_PASS_TTL_SECONDS", 0.0)
    time.sleep(0.01)
    out = core.memory_ground({"answer_text": CORPUS[0]["claim"] + ".",
                              "session_id": "s-ttl"})
    assert out["verdict"] == "ungrounded"
    assert any("no_pre_pass" in n for n in out["notes"])


# --- разбор ответа на утверждения ---------------------------------------------

def test_split_claims():
    text = (
        "# Итог\n"
        "Вот:\n"
        "- Бот работает на TW.\n"
        "- Recall равен 0.9444 по замеру.\n"
        "А что с 5090? \n"
        "Она умерла 02.09."
    )
    claims = grounding.split_claims(text)
    assert "Бот работает на TW" in claims[0]
    assert any("0.9444" in c for c in claims), "десятичная точка не должна резать фразу"
    assert not any(c.endswith("?") for c in claims), "вопрос — не утверждение"
    assert not any(c.strip() in ("Вот", "Итог") for c in claims)


def test_split_claims_dedupes_and_caps():
    assert grounding.split_claims("Бот работает на TW. Бот работает на TW.") == \
        ["Бот работает на TW"]
    many = " ".join(f"Утверждение номер {i} про память." for i in range(50))
    assert len(grounding.split_claims(many, max_claims=5)) == 5


def test_unknown_words_weigh_as_much_as_rarest(core):
    """Основа, которой в графе нет вообще, — сильнейший признак выдумки и
    обязана весить как самая редкая известная, а не 1.0 (иначе выдумка
    набирает покрытие за счёт знакомой половины ответа)."""
    idx = grounding.SupportIndex.of(core.store._budget_engine())
    assert idx.unseen_idf == max(core.store._budget_engine().idf.values())
    assert idx.idf("совершенно-незнакомая-основа") == idx.unseen_idf
    assert idx.idf("совершенно-незнакомая-основа") >= max(
        idx.idf(t) for t in ("алиса", "граф") if t in core.store._budget_engine().idf)


def test_underscores_are_not_stripped_as_markdown():
    """Подчёркивание — часть путей и имён, а не markdown-выделение: стирая
    его, мы теряли самый различающий токен ответа."""
    claims = grounding.split_claims(
        "**Жирный** заголовок: файл /opt/mnemos/tools/mnemos_decay.sh лежит на VPS.")
    assert "mnemos_decay.sh" in claims[0]
    assert "*" not in claims[0]


def test_empty_answer_rejected(core):
    with pytest.raises(ValueError):
        core.memory_ground({"answer_text": "   "})


def test_answer_without_claims(core):
    core.memory_ground_prepare({"query": "Где бот?", "session_id": "s-e"})
    out = core.memory_ground({"answer_text": "Итак:", "session_id": "s-e"})
    assert out["counts"]["total"] == 0
    assert out["verdict"] == "ungrounded"
    assert any("проверяемых утверждений" in n for n in out["notes"])


# --- MCP-интерфейс ------------------------------------------------------------

def test_tools_are_exposed_over_mcp(rpc):
    names = {t["name"] for t in rpc("tools/list")["result"]["tools"]}
    assert {"memory_ground", "memory_ground_prepare", "memory_answer",
            "memory_ground_log"} <= names


def test_memory_ground_over_mcp(rpc):
    rpc("tools/call", {"name": "memory_add", "arguments": {
        "claim": "Прокси-пул на 61 адрес лежит в /opt/migration_backup/proxy_pool.txt",
        "source": "инвентарь прокси 02.09",
        "evidence": ["proxy_pool.txt"],
        "context": "Где взять список прокси",
    }})
    prep = json.loads(rpc("tools/call", {"name": "memory_ground_prepare", "arguments": {
        "query": "Где лежит прокси-пул?", "session_id": "m1"}})["result"]["content"][0]["text"])
    assert "memory_ground" in prep["policy"]
    out = json.loads(rpc("tools/call", {"name": "memory_ground", "arguments": {
        "answer_text": "Прокси-пул на 61 адрес лежит в /opt/migration_backup/proxy_pool.txt.",
        "session_id": "m1"}})["result"]["content"][0]["text"])
    assert out["verdict"] == "grounded"
    assert out["source_node_ids"]


def test_memory_prompt_registers_pre_pass(core):
    """Клиент, уже ходящий через memory_prompt, чинит grounding параметром."""
    res = core.memory_prompt({"query": "Где работает боевой бот?",
                              "session_id": "s-mp", "agent": "alice"})
    assert res["session_id"] == "s-mp"
    out = core.memory_ground({"answer_text": CORPUS[0]["claim"] + ".",
                              "session_id": "s-mp"})
    assert out["pre_pass"]["tool"] == "memory_prompt"
    assert out["verdict"] == "grounded"


def test_bad_params(core):
    with pytest.raises(ValueError):
        core.memory_ground_prepare({"query": ""})
    with pytest.raises(ValueError):
        core.memory_answer({"query": "  "})
    with pytest.raises(ValueError):
        core.memory_ground_log({"event": "nope"})
    with pytest.raises(ValueError):
        core.memory_ground({"answer_text": "текст", "max_claims": 0})
