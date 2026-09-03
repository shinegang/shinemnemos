# -*- coding: utf-8 -*-
"""Тесты лексического слоя ML-BOOST: стеммер, токены, BM25F, RRF, search_rrf."""

import hashlib
import json

import pytest

from mnemos import make_node, mlsearch
from mnemos.server import MnemosCore


# ---------------------------------------------------------------------------
# стеммер
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("word,expected", [
    # дефект, найденный сверкой с эталонным snowballstemmer: «а/я» перед
    # окончанием группы 1 обязана сама лежать в RV, иначе стеммер срезал
    # последнюю букву у слова целиком («план» -> «пла»)
    ("план", "план"),
    ("дал", "дал"),
    ("дан", "дан"),
    ("брать", "брат"),
    ("снять", "снят"),
    ("данных", "дан"),
    ("данные", "дан"),
])
def test_stem_ru_rv_guard(word, expected):
    assert mlsearch.stem_ru(word) == expected


def test_stem_ru_unifies_word_forms():
    """Смысл стемминга: «шорт» и «шорты» должны сойтись в одну основу —
    подстрочный поиск этого не умеет (дефект ML-BOOST §2)."""
    assert mlsearch.stem("шорт") == mlsearch.stem("шорты")
    assert mlsearch.stem("торговля") == mlsearch.stem("торговли")


def test_stem_en_is_inflectional_only():
    assert mlsearch.stem("gates") == "gate"      # не «gat»
    assert mlsearch.stem("pass") == "pass"       # «ss» не режем
    assert mlsearch.stem("location") == "location"  # словообразование не трогаем
    assert mlsearch.stem("policies") == "policy"


def test_stem_keeps_identifiers():
    for tok in ("x402", "70b", "8765"):
        assert mlsearch.stem(tok) == tok


# ---------------------------------------------------------------------------
# токенизация
# ---------------------------------------------------------------------------
def test_tokens_split_compound_identifiers():
    got = mlsearch.tokens("127.0.0.1:8765")
    assert "8765" in got
    got2 = mlsearch.tokens("llama-70B")
    assert "70b" in got2 and "llama" in got2


def test_tokens_drop_stopwords_and_node_ids():
    assert mlsearch.tokens("и в на для") == []
    assert mlsearch.tokens("mn_c1423a58e02a") == []


# ---------------------------------------------------------------------------
# BM25F
# ---------------------------------------------------------------------------
def _corpus():
    return {
        "a": {"claim": "Алиса торгует шортами", "context": "", "source": "", "evidence": []},
        "b": {"claim": "Отчёт по выручке", "context": "выручка выросла", "source": "", "evidence": []},
        "c": {"claim": "Порт MCP 127.0.0.1:8765", "context": "", "source": "", "evidence": []},
    }


def test_bm25_finds_by_word_form_not_substring():
    bm = mlsearch.BM25F(_corpus())
    # «шорт» не является подстрокой «шортами», но основа общая
    assert bm.rank("шорт", depth=5)[0] == "a"


def test_bm25_finds_subtoken():
    bm = mlsearch.BM25F(_corpus())
    assert bm.rank("8765", depth=5)[0] == "c"


def test_bm25_idf_never_negative():
    bm = mlsearch.BM25F(_corpus())
    assert all(v >= 0.0 for v in bm.idf.values())


def test_bm25_rank_skips_and_returns_only_hits():
    bm = mlsearch.BM25F(_corpus())
    assert bm.rank("выручка", depth=5, skip=["b"]) == []
    assert bm.rank("такогословаНЕТ", depth=5) == []


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------
def test_rrf_canonical_formula():
    # один список: порядок сохраняется
    assert mlsearch.rrf([["x", "y", "z"]]) == ["x", "y", "z"]
    # документ, стоящий вторым в обоих списках, обгоняет первого из одного
    fused = mlsearch.rrf([["a", "c"], ["b", "c"]])
    assert fused[0] == "c"


def test_rrf_missing_document_does_not_vote():
    fused = mlsearch.rrf([["a"], []])
    assert fused == ["a"]


# ---------------------------------------------------------------------------
# Store.search_rrf / MCP
# ---------------------------------------------------------------------------
def _fill(store):
    store.add(make_node("Алиса торгует шортами по сигналу", kind="fact"))
    store.add(make_node("Дневной стоп daily_halt 16:45", kind="rule"))
    store.add(make_node("Порт MCP 127.0.0.1:8765", kind="fact"))
    return store


def test_search_rrf_returns_nodes(store):
    _fill(store)
    got = store.search_rrf("шорт", top_k=5)
    assert got and any("Алиса" in n["claim"] for n in got)


def test_search_rrf_empty_query(store):
    _fill(store)
    assert store.search_rrf("", top_k=5) == []


def test_search_rrf_respects_top_k(store):
    _fill(store)
    assert len(store.search_rrf("Алиса daily_halt порт", top_k=1)) == 1


def test_search_rrf_excludes_hubs(store):
    _fill(store)
    store.add(make_node("ХАБ «Алиса»: 3 узла", kind="hub"))
    assert all(n.get("kind") != "hub" for n in store.search_rrf("Алиса", top_k=5))


def test_search_rrf_does_not_write_store(store, tmp_path):
    """ПРАВИЛО 1: поиск не трогает стор — sha256 файла не меняется."""
    _fill(store)
    path = store.path
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    store.search_rrf("Алиса шорт порт", top_k=5)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_search_rrf_survives_without_fastembed(store, monkeypatch):
    """Без fastembed плотный сигнал молча не голосует, поиск работает."""
    _fill(store)

    def boom(self):
        raise ImportError("fastembed не установлен")
    monkeypatch.setattr(type(store), "_get_embedder", boom)
    assert store.search_rrf("шорт", top_k=5)


def test_memory_search_mode_rrf(store):
    _fill(store)
    core = MnemosCore(store)
    out = core.memory_search({"query": "шорт", "mode": "rrf", "top_k": 5})
    assert out["mode"] == "rrf" and out["count"] >= 1


def test_memory_search_mode_rrf_auto_top_k(store):
    """top_k='auto' в режиме rrf не должен молча уводить в budget."""
    _fill(store)
    core = MnemosCore(store)
    out = core.memory_search({"query": "шорт", "mode": "rrf", "top_k": "auto"})
    assert out["mode"] == "rrf"


def test_memory_search_rejects_unknown_mode(store):
    core = MnemosCore(store)
    with pytest.raises(ValueError, match="rrf"):
        core.memory_search({"query": "x", "mode": "нетакого"})


def test_memory_search_default_mode_unchanged(store):
    """Совместимость MCP: режим по умолчанию остаётся substring."""
    _fill(store)
    core = MnemosCore(store)
    assert core.memory_search({"query": "Алиса"})["mode"] == "substring"
