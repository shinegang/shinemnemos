# -*- coding: utf-8 -*-
"""Замер recall боевого Mnemos по уровням квантования и режимам поиска.

Запуск:  python3 eval_recall.py [--gt /opt/bench-memory/ground_truth.json]

ПРАВИЛО 1: боевой стор /opt/mnemos/mnemos/data/nodes.json только КОПИРУЕТСЯ
(shutil.copy2) во временный каталог; Store всегда открывается на копии. sha256
боевого файла печатается до и после прогона и обязан совпасть — иначе выход 1.

Что меряется (recall@5 / P@5 / MRR / hit@1, top_k=5, метрики 1-в-1 из
bench_qmem/bench_ml):
  * уровни L0/L1/L2 — все узлы копии принудительно опускаются на уровень;
  * пути: Store.search (подстрока), search_budget (Ф1), search_rrf (ML-BOOST),
    и сквозной MCP-путь memory_search — то, что реально видит агент;
  * две формы запроса: kw (ключевые слова) и nl (вопрос на естественном языке).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mnemos import qmem  # noqa: E402
from mnemos.server import MnemosCore  # noqa: E402
from mnemos.store import Store  # noqa: E402

PROD = HERE / "data" / "nodes.json"
DEFAULT_GT = "/opt/bench-memory/ground_truth.json"
TOPK = 5


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(fn: Callable[[str, int], Sequence[str]], gt: List[Dict[str, Any]],
            form: str) -> Dict[str, float]:
    rec = prec = mrr = hit1 = 0.0
    for q in gt:
        gts = set(q["gt"])
        top5 = list(fn(q[form], TOPK))[:TOPK]
        inter = gts & set(top5)
        rr = next((1.0 / i for i, nid in enumerate(top5, 1) if nid in gts), 0.0)
        rec += len(inter) / len(gts)
        prec += len(inter) / TOPK
        mrr += rr
        hit1 += 1.0 if (top5 and top5[0] in gts) else 0.0
    n = max(1, len(gt))
    return {"recall@5": round(rec / n, 4), "P@5": round(prec / n, 4),
            "MRR": round(mrr / n, 4), "hit@1": round(hit1 / n, 4)}


def copy_at_level(tmp: Path, level: int, tag: str) -> Path:
    """Копия боевого стора, все узлы принудительно опущены на уровень level."""
    path = tmp / f"{tag}_L{level}.json"
    src = json.loads(PROD.read_text(encoding="utf-8"))
    sal = qmem.Salience(src.values())
    for d in src.values():
        d["levels"] = qmem.build_levels(d, sal)
        d["level"] = level
    path.write_text(json.dumps(src, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=DEFAULT_GT)
    args = ap.parse_args()
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))["decisions"]

    before = sha(PROD)
    print(f"боевой стор: {PROD}")
    print(f"sha256 ДО:    {before}")
    print(f"GT: {len(gt)} запросов, {sum(len(q['gt']) for q in gt)} пар, top_k={TOPK}\n")

    rows: List[tuple] = []
    with tempfile.TemporaryDirectory(prefix="mnemos_eval_") as td:
        tmp = Path(td)
        for level in (0, 1, 2):
            for form in ("kw", "nl"):
                # каждый путь — на своей свежей копии: touch поднимает узлы на
                # L0 и исказил бы замер следующего пути
                st = Store(str(copy_at_level(tmp, level, f"sub_{form}")))
                sub = measure(lambda q, k: [n["id"] for n in st.search(q, top_k=k, touch=False)], gt, form)

                st2 = Store(str(copy_at_level(tmp, level, f"bud_{form}")))
                bud = measure(lambda q, k: [r["id"] for r in st2.search_budget(q, top_k=k)["results"]], gt, form)

                st3 = Store(str(copy_at_level(tmp, level, f"rrf_{form}")))
                rrf = measure(lambda q, k: [n["id"] for n in st3.search_rrf(q, top_k=k)], gt, form)

                st4 = Store(str(copy_at_level(tmp, level, f"mcp_{form}")))
                core = MnemosCore(st4)
                mcp = measure(
                    lambda q, k: [n["id"] for n in core.memory_search({"query": q, "top_k": k})["results"]],
                    gt, form)

                for name, m in (("Store.search (подстрока)", sub), ("search_budget (Ф1)", bud),
                                ("search_rrf (ML-BOOST)", rrf), ("memory_search (MCP)", mcp)):
                    rows.append((f"L{level}", form, name, m))

    hdr = f"{'уровень':<8} {'форма':<5} {'путь':<26} {'recall@5':>9} {'P@5':>6} {'MRR':>6} {'hit@1':>6}"
    print(hdr)
    print("-" * len(hdr))
    for level, form, name, m in rows:
        print(f"{level:<8} {form:<5} {name:<26} {m['recall@5']:>9.4f} {m['P@5']:>6.3f} "
              f"{m['MRR']:>6.3f} {m['hit@1']:>6.3f}")

    after = sha(PROD)
    ok = before == after
    print(f"\nsha256 ПОСЛЕ: {after}")
    print(f"боевой стор не тронут: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
