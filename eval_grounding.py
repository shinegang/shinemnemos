# -*- coding: utf-8 -*-
"""Замер обязательного прохода через граф (Grounded Answer, приказ Ильи 03.09).

Запуск:  python3 eval_grounding.py [--gt /opt/bench-memory/ground_truth.json]
         [--json out.json]

ПРАВИЛО 1 (как в eval_recall.py): боевой стор только КОПИРУЕТСЯ во временный
каталог, Store открывается на копии, sha256 боевого файла печатается ДО и
ПОСЛЕ и обязан совпасть — иначе выход 1.

Что меряется. На тех же 15 запросах ground_truth.json, на которых мерился
recall (замер 02.09), для каждого запроса строятся ПЯТЬ форм ответа агента:

  цитата          — дословный claim эталонного узла. Контроль: если такой
                    ответ не grounded, гейт сломан (ложная тревога).
  пересказ        — тот же claim, из которого выброшено каждое третье
                    значимое слово. Так выглядит ответ модели своими словами;
                    проверяет, что порог не требует дословности.
  цитата+выдумка  — к цитате дописано ОДНО ложное утверждение. Проверяет, что
                    выдумку видно даже когда 50% ответа — правда (самый
                    частый и самый опасный случай).
  подмена числа   — цитата, в которой каждое число заменено другим. Слова из
                    памяти, цифра выдумана.
  чистая выдумка  — ответ целиком из ложных утверждений по теме запроса.

Плюс graph-first: сработал ли ответ из графа без LLM, и попал ли он в
эталон запроса (hit по НЕ тому узлу опаснее промаха — он молча отдаёт
пользователю неверный ответ, поэтому считается отдельной колонкой).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mnemos import grounding  # noqa: E402
from mnemos.budget import tokens as budget_tokens  # noqa: E402
from mnemos.server import MnemosCore  # noqa: E402
from mnemos.store import Store  # noqa: E402

PROD = HERE / "data" / "nodes.json"
DEFAULT_GT = "/opt/bench-memory/ground_truth.json"
TOPK = 5

# Ложные утверждения по теме каждого запроса: правдоподобные по форме, но в
# графе их нет. Пишутся руками — сгенерировать «правдоподобную ложь»
# перестановкой слов нельзя, получится бессмыслица, которую поймает любой порог.
FAKES: Dict[str, str] = {
    "D01": "Решение пускать сигнал в ордер принимает внешний риск-менеджер Сергей по телефону.",
    "D02": "Halt после серии убытков включён и стоит на трёх убыточных сделках подряд.",
    "D03": "За неделю без присмотра бот заработал 812 долларов при винрейте 71 процент.",
    "D04": "Шорты были запрещены в конфиге, поэтому генератор физически не мог их предложить.",
    "D05": "Перед каждым ответом Иван требует сверяться с гороскопом и ждать одобрения юриста.",
    "D06": "Узел памяти считается фактом, если его подтвердили три независимых агента голосованием.",
    "D07": "Один запрос к Memory-API стоит четыре доллара по подписке Stripe.",
    "D08": "MCP-сервер Mnemos слушает публичный адрес на порту 443 через Cloudflare Tunnel.",
    "D09": "Цена в x402 берётся из смарт-контракта оракула Chainlink на сети Solana.",
    "D10": "Пустой запрос к /memory/search возвращает случайный узел памяти и списывает оплату.",
    "D11": "Через тридцать дней проект закрывают, если у него меньше тысячи пользователей в Telegram.",
    "D12": "Платежи получает личный кошелёк трейдера, ключи от него лежат в репозитории.",
    "D13": "Алиса работает на облачном кластере Google TPU v5 через официальный API OpenAI.",
    "D14": "В сторе памяти на первое сентября лежало четыре тысячи узлов и двести мегабайт.",
    "D15": "Алиса может удалять любые узлы графа и переписывать приказы Ильи без подтверждения.",
}

_NUM = re.compile(r"\d")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_sentence(claim: str, limit: int = 220) -> str:
    """Первое предложение claim — ответ агента, а не весь узел целиком."""
    s = str(claim or "").strip()
    m = re.search(r"[.!?](?:\s|$)", s[:limit])
    out = s[: m.end()].strip() if m else s[:limit].strip()
    return out if out.endswith((".", "!", "?")) else out + "."


def retell(text: str) -> str:
    """Пересказ: выброшено каждое третье значимое слово (>=4 симв.)."""
    words = text.split()
    out, drop = [], 0
    for w in words:
        if len(re.sub(r"\W", "", w)) >= 4 and not _NUM.search(w):
            drop += 1
            if drop % 3 == 0:
                continue
        out.append(w)
    return " ".join(out)


def swap_numbers(text: str) -> str:
    """Каждое число заменено другим (цифра +1 по модулю 10)."""
    return re.sub(r"\d", lambda m: str((int(m.group(0)) + 1) % 10), text)


def variants(claim: str, fake: str) -> List[Tuple[str, str]]:
    cite = first_sentence(claim)
    out = [
        ("цитата", cite),
        ("пересказ", retell(cite)),
        ("цитата+выдумка", cite + " " + fake),
    ]
    swapped = swap_numbers(cite)
    if swapped != cite:
        # в цитате без цифр подменять нечего: такая строка совпала бы с
        # «цитатой» и посчиталась бы провалом гейта, которого не было
        out.append(("подмена числа", swapped))
    out.append(("чистая выдумка", fake))
    return out


# Чего мы ждём от гейта на каждой форме ответа. «Ожидание» здесь — это не
# подгонка: это то, ради чего гейт написан, и расхождение с ним видно в таблице.
EXPECTED = {
    "цитата": {"grounded"},
    "пересказ": {"grounded", "partial"},
    "цитата+выдумка": {"partial", "ungrounded"},
    "подмена числа": {"partial", "ungrounded"},
    "чистая выдумка": {"ungrounded"},
}


def run(gt: List[Dict[str, Any]], core: MnemosCore, nodes: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    gf_rows: List[Dict[str, Any]] = []
    repeat_rows: List[Dict[str, Any]] = []
    for q in gt:
        qid, nl = q["id"], q["nl"]
        gt_ids = set(q["gt"])
        claim = nodes[q["gt"][0]]["claim"]
        fake = FAKES[qid]

        # контекст: recall@5 бюджетного поиска по этому запросу — то, что агент
        # реально получает на шаге 1 (метрика 1-в-1 из eval_recall.py)
        top5 = [r["id"] for r in core.store.search_budget(nl, top_k=TOPK)["results"]][:TOPK]
        recall = len(gt_ids & set(top5)) / len(gt_ids)

        # graph-first по вопросу пользователя
        ans = core.memory_answer({"query": nl, "session_id": f"{qid}-gf"})
        gf = ans["graph_first"]
        best = gf.get("node_id") or gf.get("best_node")
        gf_rows.append({
            "id": qid, "hit": bool(ans["hit"]),
            "node": gf.get("node_id"),
            "correct": bool(ans["hit"]) and gf.get("node_id") in gt_ids,
            "coverage": gf.get("checks", {}).get("coverage", {}).get("value")
            if gf.get("checks") else gf.get("best_coverage"),
            "reason": None if ans["hit"] else gf.get("reason"),
            "tokens_saved_min": gf.get("tokens_saved_min") or 0,
            # калибровка: лучший кандидат и то, эталонный ли он. Без этого
            # порог можно только угадать — а угаданный порог либо не срабатывает
            # никогда, либо молча отдаёт пользователю не тот узел.
            "best_node": best,
            "best_is_gt": best in gt_ids if best else None,
            "other_checks_pass": [k for k, v in (gf.get("checks") or {}).items()
                                  if k != "coverage" and not v["pass"]],
        })

        # Сценарий, ради которого graph-first и написан: тот же вопрос
        # спрашивают повторно, уже в формулировке, которая легла в память
        # (кэш FAQ, дежурный вопрос, повтор в диалоге). Вопрос «своими
        # словами» и повтор формулировки — это два разных режима, и мерить
        # их одной цифрой нечестно.
        rep = core.memory_answer({"query": first_sentence(claim),
                                  "session_id": f"{qid}-rep"})
        rgf = rep["graph_first"]
        repeat_rows.append({
            "id": qid, "hit": bool(rep["hit"]),
            "correct": bool(rep["hit"]) and rgf.get("node_id") in gt_ids,
            "coverage": (rgf.get("checks", {}).get("coverage", {}).get("value")
                         if rgf.get("checks") else rgf.get("best_coverage")) or 0.0,
            "tokens_saved_min": rgf.get("tokens_saved_min") or 0,
            "reason": None if rep["hit"] else rgf.get("reason"),
        })

        for form, answer in variants(claim, fake):
            sid = f"{qid}-{form}"
            core.memory_ground_prepare({"query": nl, "session_id": sid, "agent": "eval"})
            out = core.memory_ground({"answer_text": answer, "session_id": sid,
                                      "agent": "eval", "reinforce": False})
            # поймали ли выдумку: хотя бы одно ложное утверждение помечено
            caught = None
            if form in ("цитата+выдумка", "чистая выдумка"):
                fake_key = set(budget_tokens(fake))
                caught = any(
                    fake_key & set(budget_tokens(c["claim"]))
                    and c["verdict"] in ("unsupported", "refuted", "partial")
                    for c in out["claims"]
                )
            elif form == "подмена числа":
                caught = any(c["numbers"]["missing"] for c in out["claims"])
            rows.append({
                "id": qid, "form": form, "recall@5": round(recall, 3),
                "verdict": out["verdict"],
                "ratio": out["grounded_ratio"],
                "claims": out["counts"]["total"],
                "supported": out["counts"]["supported"],
                "unsupported": out["counts"]["unsupported"] + out["counts"]["refuted"],
                "sources": len(out["source_node_ids"]),
                "expected_ok": out["verdict"] in EXPECTED[form],
                "caught": caught,
            })
    return {"rows": rows, "graph_first": gf_rows, "repeat": repeat_rows}


def report(res: Dict[str, Any], gt_n: int) -> Tuple[str, Dict[str, Any]]:
    rows, gf, rep = res["rows"], res["graph_first"], res["repeat"]
    lines: List[str] = []
    hdr = (f"{'запрос':<6} {'форма ответа':<16} {'recall@5':>8} {'вердикт':<12} "
           f"{'доля':>5} {'утв.':>4} {'подтв.':>6} {'не подтв.':>9} "
           f"{'узлы':>4} {'ждали':>6} {'поймал':>7}")
    lines += [hdr, "-" * len(hdr)]
    for r in rows:
        caught = "-" if r["caught"] is None else ("да" if r["caught"] else "НЕТ")
        lines.append(
            f"{r['id']:<6} {r['form']:<16} {r['recall@5']:>8.2f} {r['verdict']:<12} "
            f"{r['ratio']:>5.2f} {r['claims']:>4} {r['supported']:>6} "
            f"{r['unsupported']:>9} {r['sources']:>4} "
            f"{('да' if r['expected_ok'] else 'НЕТ'):>6} {caught:>7}"
        )

    by_form: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        f = by_form.setdefault(r["form"], {"n": 0, "ok": 0, "verdicts": {},
                                           "caught": 0, "catchable": 0})
        f["n"] += 1
        f["ok"] += int(r["expected_ok"])
        f["verdicts"][r["verdict"]] = f["verdicts"].get(r["verdict"], 0) + 1
        if r["caught"] is not None:
            f["catchable"] += 1
            f["caught"] += int(r["caught"])

    lines += ["", "ИТОГ ПО ФОРМАМ ОТВЕТА", ""]
    h2 = f"{'форма ответа':<16} {'n':>3} {'как ждали':>10} {'выдумка поймана':>16}  вердикты"
    lines += [h2, "-" * len(h2)]
    for form in EXPECTED:
        f = by_form.get(form)
        if not f:
            continue
        cat = f"{f['caught']}/{f['catchable']}" if f["catchable"] else "-"
        vs = ", ".join(f"{k}:{v}" for k, v in sorted(f["verdicts"].items()))
        lines.append(f"{form:<16} {f['n']:>3} {f['ok']:>4}/{f['n']:<5} {cat:>16}  {vs}")

    hits = [g for g in gf if g["hit"]]
    lines += ["", "GRAPH-FIRST (ответ из графа, без вызова LLM)", ""]
    h3 = f"{'запрос':<6} {'сработал':<9} {'узел':<16} {'из эталона':<11} {'покрытие':>8}  причина промаха"
    lines += [h3, "-" * len(h3)]
    for g in gf:
        lines.append(
            f"{g['id']:<6} {('да' if g['hit'] else 'нет'):<9} "
            f"{str(g['node'] or '-'):<16} "
            f"{('да' if g['correct'] else ('НЕТ' if g['hit'] else '-')):<11} "
            f"{(g['coverage'] if g['coverage'] is not None else 0):>8.2f}  "
            f"{(g['reason'] or '')[:60]}"
        )

    rhits = [r for r in rep if r["hit"]]
    lines += ["", "GRAPH-FIRST НА ПОВТОРЕ (запрос = формулировка из памяти)", ""]
    h5 = (f"{'запрос':<6} {'сработал':<9} {'из эталона':<11} {'покрытие':>8} "
          f"{'токенов':>8}  причина промаха")
    lines += [h5, "-" * len(h5)]
    for r in rep:
        why = (r["reason"] or "").replace("не прошли пороги graph-first: ", "")
        lines.append(
            f"{r['id']:<6} {('да' if r['hit'] else 'нет'):<9} "
            f"{('да' if r['correct'] else ('НЕТ' if r['hit'] else '-')):<11} "
            f"{r['coverage']:>8.2f} {r['tokens_saved_min']:>8}  {why}"
        )

    # Калибровка порога покрытия: при каком пороге graph-first начинает
    # срабатывать и не начинает ли он при этом отдавать НЕ тот узел.
    lines += ["", "КАЛИБРОВКА ПОРОГА GRAPH-FIRST (по лучшему кандидату)", ""]
    h4 = f"{'порог':>6} {'сработал бы':>12} {'из них верных':>14} {'ошибочных ответов':>18}"
    lines += [h4, "-" * len(h4)]
    sweep = []
    for th in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        fire = [g for g in gf
                if (g["coverage"] or 0) >= th and not g["other_checks_pass"]]
        right = sum(1 for g in fire if g["best_is_gt"])
        sweep.append({"threshold": th, "fired": len(fire), "correct": right,
                      "wrong": len(fire) - right})
        lines.append(f"{th:>6.2f} {len(fire):>12} {right:>14} {len(fire) - right:>18}")

    summary = {
        "запросов": gt_n,
        "ответов проверено": len(rows),
        "вердикт как ждали": f"{sum(r['expected_ok'] for r in rows)}/{len(rows)}",
        "ложных тревог (цитата не grounded)": sum(
            1 for r in rows if r["form"] == "цитата" and r["verdict"] != "grounded"),
        "пропущенных выдумок": sum(
            1 for r in rows if r["caught"] is False),
        "graph-first на вопросе своими словами": f"{len(hits)}/{len(gf)}",
        "graph-first на повторе формулировки": f"{len(rhits)}/{len(rep)}",
        "из них отдали НЕ эталонный узел": sum(
            1 for r in hits + rhits if not r["correct"]),
        "токенов сэкономлено (нижняя граница)": sum(
            r["tokens_saved_min"] for r in hits + rhits),
        "вызовов LLM сэкономлено": len(hits) + len(rhits),
    }
    lines += ["", "СВОДКА", ""]
    for k, v in summary.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines), {**summary, "sweep": sweep}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=DEFAULT_GT)
    ap.add_argument("--json", default=None, help="куда сложить сырые строки")
    args = ap.parse_args()
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))["decisions"]
    nodes = json.loads(PROD.read_text(encoding="utf-8"))

    before = sha(PROD)
    print(f"боевой стор: {PROD}\nsha256 ДО:    {before}")
    print(f"GT: {len(gt)} запросов, форм ответа {len(EXPECTED)}, top_k={TOPK}\n")

    with tempfile.TemporaryDirectory(prefix="mnemos_ground_") as td:
        tmp = Path(td)
        shutil.copy2(PROD, tmp / "nodes.json")
        core = MnemosCore(Store(tmp / "nodes.json"), plugins=[])
        res = run(gt, core, nodes)

    text, summary = report(res, len(gt))
    print(text)
    if args.json:
        Path(args.json).write_text(
            json.dumps({**res, "summary": summary}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"\nсырые данные: {args.json}")

    after = sha(PROD)
    ok = before == after
    print(f"\nsha256 ПОСЛЕ: {after}\nбоевой стор не тронут: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
