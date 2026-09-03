# -*- coding: utf-8 -*-
"""Бенчмарк-фикс 26.08: семантический поиск Store.search_semantic (fastembed).

Если fastembed недоступен в системном python — весь модуль пропускается
(pytest.skip с пометкой); если доступен — реальный прогон: 20 RU-фактов,
перефраз находит нужный узел в топ-5 (закрывает дыру «семантика 0/25»).
"""

import json
import sys
import types

import pytest

fastembed = pytest.importorskip(
    "fastembed",
    reason="fastembed не установлен в системном python — семантические тесты пропущены",
)

from mnemos import Store, now_iso  # noqa: E402

# 20 RU-фактов: (claim, перефраз-запрос). Факты попарно далеки по смыслу,
# чтобы top-5 при 20 узлах был честным (нужный узел обязан попасть).
RU_FACTS = [
    ("Команда переехала в новый офис на Ленинском проспекте",
     "люди теперь работают в другом здании на Ленинском"),
    ("Релиз версии 2.4 выйдет в пятницу вечером",
     "новая версия продукта появится в конце недели"),
    ("Бюджет маркетинга сократили на треть",
     "денег на рекламу стало заметно меньше"),
    ("Серверы перенесли в облако AWS",
     "инфраструктура теперь работает в амазоновском облаке"),
    ("Клиент из Берлина оплатил счёт за февраль",
     "немецкий заказчик закрыл февральскую оплату"),
    ("Компания наняла двух новых разработчиков",
     "в штат добавились пара программистов"),
    ("Собрание команды перенесли на вторник",
     "встреча состоится в другой день — во вторник"),
    ("Продажи выросли после обновления сайта",
     "обновлённый сайт привёл к росту продаж"),
    ("Отключили старый API без предупреждения",
     "старый интерфейс интеграции перестал работать внезапно"),
    ("Дизайнер предложил тёмную тему для приложения",
     "в приложении предложили сделать тёмный режим"),
    ("База данных увеличена до двух терабайт",
     "хранилище данных расширили до 2 ТБ"),
    ("Тесты проходят быстрее после оптимизации",
     "прогон проверок ускорился благодаря оптимизации"),
    ("Зарплата инженеров вырастет с первого апреля",
     "сотрудникам поднимут оплату труда с апреля"),
    ("Проект закрыли из-за нехватки финансирования",
     "деньги на проект закончились, его прикрыли"),
    ("Новый клиент использует только английский язык",
     "заказчик общается исключительно на английском"),
    ("Документацию переписали на вики",
     "инструкции перенесли в систему вики"),
    ("Сервис мониторинга падал трижды за ночь",
     "система наблюдения отключалась несколько раз ночью"),
    ("Команда поддержки отвечает клиентам за час",
     "служба помощи решает обращения пользователей в течение часа"),
    ("Закупили новое железо для тестовой среды",
     "купили оборудование для стенда разработки"),
    ("Стратегия компании изменилась после квартала",
     "планы фирмы пересмотрели по итогам квартала"),
]


def test_search_semantic_finds_paraphrase_top5(tmp_path):
    """Бенчмарк-фикс 26.08: перефраз находит нужный узел в топ-5 из 20."""
    store = Store(tmp_path / "nodes.json")
    store.add_many([
        {"id": f"mn_f{i}", "kind": "fact", "claim": claim,
         "source": "тест", "context": "бенчмарк", "ts": now_iso()}
        for i, (claim, _) in enumerate(RU_FACTS)
    ])
    miss = []
    for i, (claim, para) in enumerate(RU_FACTS):
        top = store.search_semantic(para, top_k=5)
        ids = [node["id"] for node, _score in top]
        if f"mn_f{i}" not in ids:
            miss.append((i, para[:60], ids))
    assert not miss, f"перефразы вне топ-5: {miss}"


def test_search_semantic_returns_sorted_scores(tmp_path):
    store = Store(tmp_path / "nodes.json")
    store.add_many([
        {"id": f"mn_f{i}", "kind": "fact", "claim": claim,
         "source": "тест", "context": "бенчмарк", "ts": now_iso()}
        for i, (claim, _) in enumerate(RU_FACTS[:5])
    ])
    top = store.search_semantic("денег на рекламу стало заметно меньше", top_k=5)
    assert len(top) == 5
    scores = [score for _node, score in top]
    assert scores == sorted(scores, reverse=True)  # по убыванию близости
    assert all(isinstance(score, float) for score in scores)


def test_search_semantic_cache_invalidated_on_rewrite(tmp_path):
    """Кэш эмбеддингов пересчитывается при изменении claim (бенчмарк-фикс 26.08)."""
    store = Store(tmp_path / "nodes.json")
    store.add_many([
        {"id": "mn_a", "kind": "fact",
         "claim": "Команда переехала в новый офис на Ленинском проспекте",
         "source": "т", "context": "офис", "ts": now_iso()},
        {"id": "mn_b", "kind": "fact",
         "claim": "Серверы перенесли в облако AWS",
         "source": "т", "context": "инфра", "ts": now_iso()},
    ])
    # тёплый кэш
    top = store.search_semantic("люди теперь работают в другом здании", top_k=2)
    assert top[0][0]["id"] == "mn_a"
    # rewrite меняет claim — вектор mn_a обязан пересчитаться
    store.rewrite("mn_a", "Бюджет маркетинга сократили на треть", source="новое")
    top = store.search_semantic("денег на рекламу стало заметно меньше", top_k=2)
    assert top[0][0]["id"] == "mn_a"          # новый claim найден по смыслу
    top = store.search_semantic("люди теперь работают в другом здании", top_k=2)
    assert top[0][0]["id"] == "mn_b"          # старый claim больше не релевантен


def test_search_semantic_empty_query(tmp_path):
    store = Store(tmp_path / "nodes.json")
    store.add({"id": "mn_e", "kind": "fact", "claim": "что-то важное",
               "source": "т", "ts": now_iso()})
    assert store.search_semantic("   ") == []
    assert store.search_semantic("") == []


def test_search_semantic_import_error_message(monkeypatch, tmp_path):
    """Без fastembed — честный ImportError с подсказкой, а не голый сбой."""
    fake = types.ModuleType("fastembed")  # модуль есть, TextEmbedding — нет
    monkeypatch.setitem(sys.modules, "fastembed", fake)
    store = Store(tmp_path / "nodes.json")
    store.add({"id": "mn_x", "kind": "fact", "claim": "что-то",
               "source": "т", "ts": now_iso()})
    with pytest.raises(ImportError, match="pip install fastembed"):
        store.search_semantic("что-то")


# -- MCP-инструмент memory_search mode="semantic" ---------------------------
def test_memory_search_semantic_mode_via_mcp(rpc):
    rpc("tools/call", {"name": "memory_add", "arguments": {
        "claim": "Команда переехала в новый офис на Ленинском проспекте",
        "source": "внутренний чат", "context": "офис"}})
    resp = rpc("tools/call", {"name": "memory_search", "arguments": {
        "query": "люди теперь работают в другом здании",
        "mode": "semantic", "top_k": 3}})
    assert resp.get("error") is None, resp
    out = json.loads(resp["result"]["content"][0]["text"])
    assert out["mode"] == "semantic"
    assert out["count"] >= 1
    assert out["results"][0]["node"]["claim"].startswith("Команда переехала")
    assert isinstance(out["results"][0]["score"], float)


def test_memory_search_semantic_missing_fastembed_honest_error(rpc, monkeypatch, server):
    """mode=semantic без fastembed — честная ошибка с текстом, не INTERNAL_ERROR."""

    def boom(query, top_k=5):
        raise ImportError(
            "семантический поиск требует пакет fastembed — установите его: "
            "pip install fastembed"
        )

    monkeypatch.setattr(server.mnemos.store, "search_semantic", boom)
    resp = rpc("tools/call", {"name": "memory_search", "arguments": {
        "query": "x", "mode": "semantic"}})
    assert resp["error"]["code"] == -32602
    assert "fastembed" in resp["error"]["message"]
    assert "pip install fastembed" in resp["error"]["message"]


def test_memory_search_substring_default_mode_unchanged(rpc):
    """mode по умолчанию — substring; API-форма ответа не изменился."""
    rpc("tools/call", {"name": "memory_add", "arguments": {
        "claim": "Курс BTC вырос на 3% за день", "source": "биржа"}})
    resp = rpc("tools/call", {"name": "memory_search", "arguments": {"query": "BTC"}})
    out = json.loads(resp["result"]["content"][0]["text"])
    assert out["mode"] == "substring"
    assert out["count"] == 1
    assert out["results"][0]["claim"].startswith("Курс BTC")
