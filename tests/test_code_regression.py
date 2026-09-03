# -*- coding: utf-8 -*-
"""ГЕЙТ R4: внедрение code-слоя не роняет recall@5 поиска решений.

Порог: recall@5 >= 0.944 на /opt/bench-memory/ground_truth.json (режим
decisions, 15 запросов, форма kw) — то же число и та же разметка, что в
PoC (§4.3 отчёта ФЭЙБЛ-ИССЛЕДОВАНИЕ-ИНДЕКСАЦИЯ-01.09). Формулы метрик
1-в-1 из eval_decisions_regression.py.

Проверяется три вещи:
  A. база — копия боевого стора даёт recall@5 >= порога;
  C. раздельный дизайн — при живом рядом индексе кода метрика та же
     (побайтово тот же стор, те же топ-5);
  + боевой стор /opt/mnemos/mnemos/data/nodes.json не изменился байт-в-байт.

Тесты пропускаются (skip), если на хосте нет боевого стора или файла
разметки: на 5090 пути другие — гейт там прогоняется со своими путями
через MNEMOS_GATE_STORE / MNEMOS_GATE_GT.
"""

import json
import os
import shutil

import pytest

from mnemos import code_index as ci
from mnemos.code_search import CodeRegistry
from mnemos.store import Store

PROD_STORE = os.environ.get("MNEMOS_GATE_STORE", "/opt/mnemos/mnemos/data/nodes.json")
GROUND_TRUTH = os.environ.get("MNEMOS_GATE_GT", "/opt/bench-memory/ground_truth.json")
RECALL_GATE = 0.944
TOPK = 5

pytestmark = [
    pytest.mark.skipif(not os.path.exists(PROD_STORE),
                       reason=f"нет боевого стора: {PROD_STORE}"),
    pytest.mark.skipif(not os.path.exists(GROUND_TRUTH),
                       reason=f"нет разметки: {GROUND_TRUTH}"),
]


def load_gt(mode="decisions"):
    with open(GROUND_TRUTH, encoding="utf-8") as f:
        return json.load(f)[mode]


def measure(store, gt, form="kw"):
    """recall@5 / P@5 / MRR / hit@1 — формулы 1-в-1 из PoC (metrics.py)."""
    rec = prec = mrr = hit1 = 0.0
    per_query = {}
    for q in gt:
        gts = set(q["gt"])
        top5 = [n.get("id") for n in store.search(q[form], top_k=TOPK)]
        inter = gts & set(top5)
        rr = 0.0
        for i, nid in enumerate(top5, 1):
            if nid in gts:
                rr = 1.0 / i
                break
        rec += len(inter) / len(gts)
        prec += len(inter) / TOPK
        mrr += rr
        hit1 += 1.0 if (top5 and top5[0] in gts) else 0.0
        per_query[q["id"]] = top5
    n = len(gt)
    return {"recall@5": rec / n, "precision@5": prec / n, "MRR": mrr / n,
            "hit@1": hit1 / n, "queries": n, "per_query": per_query}


@pytest.fixture
def store_copy(tmp_path):
    """Копия боевого стора: боевой файл ТОЛЬКО читается (правило 1)."""
    dst = tmp_path / "nodes.json"
    shutil.copy2(PROD_STORE, dst)
    return dst


def test_gate_recall_decisions_not_degraded(store_copy):
    """A: recall@5 решений на базовой конфигурации >= 0.944."""
    res = measure(Store(store_copy), load_gt())
    assert res["queries"] == 15
    assert res["recall@5"] >= RECALL_GATE, (
        f"ГЕЙТ ПРОВАЛЕН: recall@5={res['recall@5']:.4f} < {RECALL_GATE}; "
        f"MRR={res['MRR']:.4f} hit@1={res['hit@1']:.4f}"
    )


def test_gate_recall_with_code_layer_alive(tmp_path, store_copy):
    """C: индекс кода рядом (раздельный дизайн) не меняет ни одного топ-5."""
    if not ci.TS_OK:
        pytest.skip(f"нет py-tree-sitter: {ci.TS_ERR}")
    gt = load_gt()
    before = measure(Store(store_copy), gt)

    # рядом со стором поднимается полноценный code-слой
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "risk.py").write_text(
        'def position_size_usd(equity):\n    """Алиса: halt после серии убытков."""\n'
        "    return equity\n", encoding="utf-8")
    reg = CodeRegistry(repos=[ci.Repo("repo", str(repo_root),
                                      str(tmp_path / "code" / "repo.code_graph.json"))])
    stats = reg.refresh(force=True)
    assert stats[0]["nodes"] > 0
    assert reg.search("position_size_usd", top_k=5), "code_search должен что-то находить"

    after = measure(Store(store_copy), gt)
    assert after["recall@5"] >= RECALL_GATE
    assert after["recall@5"] == before["recall@5"]
    assert after["per_query"] == before["per_query"]


def test_code_graph_written_outside_store(tmp_path, store_copy):
    """Файл графа кода лежит отдельно и не трогает файл стора."""
    if not ci.TS_OK:
        pytest.skip(f"нет py-tree-sitter: {ci.TS_ERR}")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "a.py").write_text("def f():\n    pass\n", encoding="utf-8")
    before = store_copy.read_bytes()

    repos = ci.resolve_repos(store_path=str(store_copy),
                             roots=f"repo={repo_root}")
    assert repos[0].graph_path == str(tmp_path / "code" / "repo.code_graph.json")
    ci.refresh_repo(repos[0])

    assert os.path.exists(repos[0].graph_path)
    assert store_copy.read_bytes() == before
    graph = ci.load_graph(repos[0].graph_path)
    assert graph["meta"]["nodes"] > 0
    # в сторе нет и не может быть узлов code_symbol (KINDS — закрытый перечень)
    nodes = json.loads(store_copy.read_text(encoding="utf-8"))
    dumped = json.dumps(nodes, ensure_ascii=False)
    assert "code_symbol" not in dumped


def test_prod_store_untouched(store_copy):
    """Боевой стор побайтово равен копии — ни один тест его не менял."""
    with open(PROD_STORE, "rb") as f:
        prod = f.read()
    assert prod == store_copy.read_bytes()
