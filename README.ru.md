# Шина/память ShineMnemos — первый кирпич

**Инструменты — вам, граф стартует пустым. Grounded-ответы включены по умолчанию.**

ShineMnemos — локальный MCP-сервер памяти, который делает память агента
принудительной: каждый ответ проходит через ваш граф знаний и получает вердикт
`grounded / partial / ungrounded` плюс список выдуманных предложений. Если
ответ уже есть в графе — он отдаётся как есть, LLM не вызывается. Свежая
инстанция стартует с чистым графом: продаём инструменты, а не данные.
Python 3.12, в рантайме только stdlib, без облака и без API моделей.

**[Сайт](https://shinegang.click)** ·
**[Тарифы и покупка](https://shinegang.click/console.html#buy)** ·
[Интеграции](integrations/README-integrations.md) ·
[English](README.md)

## Быстрый старт

```bash
git clone https://github.com/shinegang/shinemnemos.git && cd shinemnemos
python3 -m mnemos --host 127.0.0.1 --port 8765 --store blank   # 1. сервер
python3 bridge/mnemos_bridge.py --selftest                     # 2. мост (без модели)
bash integrations/claude_code_add.sh                           # 3. клиент (Claude Code)
```

Конфиги Claude Desktop / Cursor / llama.cpp / LangChain — в
[integrations/](integrations/README-integrations.md).

## Почему ShineMnemos

Каждый тезис проверяется по файлу и строке этого репозитория.

- **Grounded по умолчанию** — `memory_ground` без предшествующего
  `memory_ground_prepare` даёт вердикт `ungrounded` (`no_pre_pass`):
  [mnemos/grounding.py:527](mnemos/grounding.py#L527); политика включена по
  умолчанию: [mnemos/server.py:102](mnemos/server.py#L102).
- **Graph-first** — `memory_answer` отвечает прямо из графа (ноль токенов
  генерации) либо `llm_required: true` с готовым grounded-промптом:
  [mnemos/server.py:1267](mnemos/server.py#L1267).
- **Инструменты, а не данные** — шаблон
  [mnemos/data/nodes.blank.json](mnemos/data/nodes.blank.json) = `{}`; `blank`
  не перезаписывает существующий граф
  ([mnemos/store.py:240](mnemos/store.py#L240)), непустой шаблон = отказ старта
  ([mnemos/store.py:170](mnemos/store.py#L170)).
- **Аудит** — append-only журнал `ground_log.jsonl` рядом со стором
  ([mnemos/grounding.py:736](mnemos/grounding.py#L736)), чтение —
  `memory_ground_log` ([mnemos/server.py:1302](mnemos/server.py#L1302)).
- **Truth-check** — протокол П1–П6, `pass` при ≥ 4 из 6
  ([mnemos/truth_gate.py:29](mnemos/truth_gate.py#L29),
  [mnemos/truth_gate.py:215](mnemos/truth_gate.py#L215)).
- **Локально и без зависимостей** — JSON-RPC на stdlib `http.server`
  ([mnemos/server.py:40](mnemos/server.py#L40)), в рантайме ноль сторонних
  пакетов ([requirements.txt](requirements.txt)); опциональные `fastembed` и
  tree-sitter под гардами, не обязательны.
- **Проверяемость** — `python -m pytest tests -q` (наш прогон: 352 passed,
  62 skipped; скипы — отсутствующие опциональные зависимости, гейты приватного
  корпуса тоже скипаются чисто), `python demo_gates.py` — пять гейтов качества
  целиком на stdlib. Харнесы замеров тоже в репо
  ([eval_grounding.py](eval_grounding.py), [eval_recall.py](eval_recall.py)) —
  они читают граф из `data/nodes.json` (в репо он пуст) и ground truth
  (`--gt`): положите свои.

## Купить план

Движок в этом репо — Apache-2.0, работает полностью локально и бесплатно.
Планы на сайте добавляют лимиты повыше (агенты, узлы, операции, срок журнала)
и дополнительные гарды: шесть тарифов (Free, Solo, Pro, Team, Scale,
Enterprise) от $0 до $499/мес, Enterprise — по договорённости. Оплата —
x402-челлендж, USDC в сети Base. Сайт: **[shinegang.click](https://shinegang.click)**,
покупка: **[shinegang.click/console.html#buy](https://shinegang.click/console.html#buy)**.

---

> Клиентская инстанция стартует с ЧИСТЫМ графом (`--store blank`), проход через
> граф обязателен по умолчанию (`ground_by_default`). Наш боевой стор — только
> внутренний, в дистрибутив не едет. Клиентам — [README.md](README.md) (EN).

Проект «память, дающая агенту путь к сознанию» (см. `D:\deepseek harness\memory_project.md`).
Этот репозиторий — первый кирпич по роадмапу RECON_PRODUCTS: **локальный MCP memory server
с узлами правды П1-П6 + шина агентов**. Работает на системном Python 3.12, только stdlib,
сервер (VPS) и бот не затрагиваются.

## Что внутри

```
D:\mnemos\
  mnemos\
    model.py       — модель узла памяти (JSON): id, kind, claim, source,
                     evidence[], context, ts, links[], truth_check
    truth_gate.py  — check_claim(node): протокол П1-П6, вердикт pass при >=4/6,
                     score + notes по каждой проверке
    bus.py         — шина агентов (mnemos_bus.jsonl, append-only): msg/heartbeat/duty,
                     блокировка msvcrt.locking (Windows) / fcntl (POSIX),
                     heartbeat-writer beat(from, interval_check)
    store.py       — хранилище узлов nodes.json (атомарная запись) + подстрочный поиск
    server.py      — MCP-сервер-скелет: JSON-RPC 2.0 на stdlib http.server
                     (initialize, tools/list, tools/call: memory_add/verify/search)
    __main__.py    — запуск: py -3.12 -m mnemos
  tests\           — pytest-тесты (наш прогон в публичном репо: 352 passed, 62 skipped)
```

## Модель узла (JSON)

```json
{
  "id": "mn_1a2b3c4d5e6f",
  "kind": "fact",                      // fact | hypothesis | refuted | outdated
  "claim": "Выручка выросла на 12% в Q3 2025",
  "source": "финотчёт компании за Q3",
  "evidence": ["стр. 4 отчёта, таблица 2"],
  "context": "Квартальный обзор, сравниваем с Q2",
  "ts": "2025-06-01T12:00:00.000+00:00",
  "links": ["mn_..."],
  "truth_check": { "P1": {"pass": true, "note": "..."}, ..., "verdict": "pass", "score": 6 }
}
```

## Truth-gate: протокол П1-П6

| Проверка | Логика (честная заглушка) |
|---|---|
| П1 Свежесть | ts валиден ISO-8601, не из будущего (>5 мин), не старше 365 дней |
| П2 Источник | поле `source` непустое |
| П3 Цифры | в `claim` есть числа (regex `\d`) — утверждение количественное |
| П4 Непротиворечивость | `links` не ссылаются на узлы со статусом `refuted` (по registry) |
| П5 Воспроизводимость | `evidence` непустой — есть чем перепроверить |
| П6 Полнота | поле `context` заполнено |

**Вердикт: `pass`, если прошло ≥4 из 6**, иначе `fail`. Возвращает verdict + score + notes
(пояснение по каждой проверке). `check_and_update(node, registry)` записывает результат в `truth_check`.

## Шина агентов

`bus.py` — append-only JSONL (`mnemos_bus.jsonl`), формат строки:

```json
{"ts": "...", "from": "agent-a", "to": "*", "kind": "msg", "text": "привет"}
```

- `append(from, to, kind, text)` — запись с блокировкой (Windows: `msvcrt.locking` по lock-файлу
  с retry-циклом; POSIX: `fcntl.flock`; сама запись — append/аналог O_APPEND).
- `read(sender=, receiver=, kinds=, after_ts=, before_ts=, limit=)` — чтение с фильтрами.
- `beat(from, interval_check)` — heartbeat-writer: пишет heartbeat только если с последнего
  прошло ≥ интервала (число секунд) или если callable `(bus, from_id, last_msg) -> bool`
  вернул True.

## MCP-сервер

fastmcp/mcp в системном python нет — интерфейс MCP сделан вручную (JSON-RPC 2.0 на stdlib):

```
POST /  {"jsonrpc":"2.0","id":1,"method":"tools/call",
         "params":{"name":"memory_add","arguments":{"claim":"..."}}}
```

Методы: `initialize`, `notifications/initialized`, `ping`, `tools/list`, `tools/call`.
Инструменты:

| Инструмент | Аргументы | Возвращает |
|---|---|---|
| `memory_add` | claim (обяз.), source, evidence[], context, kind, links[] | узел с truth_check (авто-прогон П1-П6) |
| `memory_verify` | node_id | verdict + score + notes по П1-П6 |
| `memory_search` | query (обяз.), top_k | топ-k узлов по подстроке (rank: claim>source>context>evidence) |
| `memory_prompt` | query и/или constitution, max_tokens, format, **session_id**, agent | system-prompt из памяти; с `session_id` регистрирует пред-проход |
| `memory_ground_prepare` | query (обяз.), session_id, agent, constitution, max_tokens, format, graph_first, threshold | ШАГ 1: промпт из графа + `graph_first` + `policy` + запись пред-прохода |
| `memory_ground` | answer_text (обяз.), query, session_id, agent, require_pre_pass, reinforce, max_claims | ШАГ 3: `grounded/partial/ungrounded`, «прошёл через граф: да/нет/частично», `source_nodes`, `unsupported_claims` |
| `memory_answer` | query (обяз.), session_id, threshold, min_weight, min_confidence, with_prompt | graph-first: ответ из графа без LLM либо `llm_required: true` + промпт |
| `memory_ground_log` | limit, session_id, event, agent, stats | append-only журнал проходов через граф + сводка |

Подробности по обязательному проходу — в разделе
[Grounded Answers](#grounded-answers--dont-burn-tokens-on-nothing).

## Точный поиск (хеш-индекс, фикс перф 27.08)

`Store.search_exact(query)` — точный поиск узла по claim (claim == query по
каноническому ключу: strip + casefold) за **O(1)** через in-memory hash-индекс
(канонический claim → id). Индекс лениво синхронизирован с хранилищем
(полная загрузка при старте, обновление при каждой записи); при промахе —
fallback-скан с самолечением, точность всегда 100% (без substring-ложных
срабатываний: `"BTC"` не найдёт `"BTC вырос"`).

- В `search(query)` быстрый путь включён по умолчанию: если query — точный
  claim, ответ приходит из индекса; иначе — прежний substring-скан.
- Отключение: `Store(path, use_hash_index=False)` или
  `search(q, use_hash_index=False)` — старое поведение.
- Полная пересборка индекса после ручной правки: `store.reindex()`.
- Бенчмарк и цифры — во внутреннем репо, в публичный дистрибутив не входят.

## Режимы поиска (`memory_search`)

| mode | что делает | recall@5 на GT (kw / nl) |
|---|---|---|
| `substring` (по умолчанию) | подстрока по claim/source/context/evidence; если целая фраза не найдена — фоллбек по основам слов (`token_fallback`) | 0.9444 / 0.8667 |
| `budget` | token-budgeting: основы слов, граф-расширение, хабы отдельно, бюджет токенов | 0.9444 / 0.8667 |
| `rrf` | слияние Ф1 + BM25F (+плотный, если есть `fastembed`) по Reciprocal Rank Fusion | 0.9444 / 0.8167 |
| `semantic` | по смыслу, косинус (требует `fastembed`) | — |

**Когда что брать.** Ключевые слова — любой режим; `rrf` с установленным `fastembed`
поднимает kw до **0.9833**. Вопрос на естественном языке — `substring` (он сам уйдёт в
`token_fallback`) или `budget`; `rrf` на NL хуже (0.8167 против 0.8667), поэтому туда он
не роутится.

Замер: `python3 eval_recall.py` на нашем боевом сторе и ground truth — они приватны и в
публичное репо не входят (запускайте на своём графе; стор только копируется, sha256
сверяется до и после).

## Лексический слой (`mnemos/mlsearch.py`, ML-BOOST 03.09)

Стеммер (русский Snowball + английские словоизменительные окончания), BM25F и RRF —
**на чистом stdlib**, без `snowballstemmer` и `numpy`: рантайм Mnemos остаётся
беззависимым. Русский стеммер сверен с эталонным `snowballstemmer` на словаре боевого
корпуса — расхождений 0 из 878 словоформ.

Что это даёт поверх подстроки: «шорт» находит «шорты» (общая основа), а составные
идентификаторы разбиваются на части (`127.0.0.1:8765` → `8765`, `llama-70B` → `70b`).

## Grounded Answers — don't burn tokens on nothing

*(English by design: this is the section clients read. Order of 03.09: every
answer an agent gives must go through the memory graph first.)*

An LLM agent with a memory server still hallucinates, because nothing forces it
to *use* the memory — or to check what it just said against it. ShineMnemos
makes that mandatory and, more importantly, **auditable**: every answer carries
a verdict saying whether the graph backs it, which nodes back it, and which
sentences the graph has never heard of.

### The three-step contract

| step | tool | what it does |
|---|---|---|
| 1. before answering | `memory_ground_prepare(query, session_id)` | searches the graph, builds the system prompt from the nodes it found, registers a *pre-pass* for the session, and checks graph-first |
| 2. answering | *(your LLM)* | answers **only** from the excerpt it was handed |
| 3. after answering | `memory_ground(answer_text, session_id)` | splits the answer into claims, verifies each against the graph, returns `grounded / partial / ungrounded` + source nodes + `unsupported_claims` |

Skip step 1 and step 3 returns `ungrounded` with `no_pre_pass`, no matter how
good the answer is. That is the point: an answer that never consulted memory is
not grounded, it is lucky. Clients that cannot keep a session id can opt out
with `require_pre_pass: false` — the per-claim verification still runs.

Drop-in system prompt (also returned verbatim as `policy` by
`memory_ground_prepare`, so the contract travels with the prompt):

```
1. Before answering, call memory_ground_prepare(query, session_id).
2. If graph_first.hit is true — return graph_first.answer as is and do NOT
   generate: the answer is already in memory, generating burns the user's
   tokens for nothing.
3. Otherwise answer ONLY from the excerpt. What is not in it, you do not know.
4. After answering, call memory_ground(answer_text, session_id). If the verdict
   is not `grounded`, show the user unsupported_claims instead of passing them
   off as facts.
```

### How a claim is verified

Each sentence of the answer is scored against the graph by **idf-weighted stem
coverage** — the same signal the budget search runs on, so grounding measures
exactly the memory the agent could actually have read. No embedding model is
required; the runtime stays stdlib-only.

Two details do most of the work:

- **Unknown words weigh the most.** A stem that appears nowhere in the graph is
  the strongest possible sign of invention, so it is weighted like the rarest
  known stem — not like a common one. Without this an answer that is half real
  and half invented scores as mostly covered.
- **Numbers are a hard gate, and they are checked in context.** Hallucination
  usually looks like the right words around a wrong number. A number counts as
  confirmed only if the supporting node has it *and* shares the word it sits in
  or one of its neighbours — so "RULE 2" is not confirmed by a node that says
  "RULE 1 ... C2 fresh evidence".

Per-claim verdicts: `supported`, `partial`, `unsupported`, `refuted` (the best
support is a node the memory has already retracted — the loudest signal of all).

Answer-level verdict: `grounded` only if every claim is supported (one
unsupported sentence is enough to lose it — length must not dilute a lie);
`partial` if at least one claim is confirmed and the rest are close;
`ungrounded` otherwise, including when nothing at all was confirmed.

### Graph-first: zero tokens when the answer is already in memory

`memory_answer(query)` returns the answer straight from the graph — no LLM call
— when a single node covers the question above threshold, is alive, weighs
enough, is confident enough, **and** clearly beats the runner-up. If any check
fails it returns `llm_required: true` plus a grounded prompt, and says which
threshold blocked it. Two answers that fit equally well is not a hit: choosing
between them is the model's job, not a threshold's.

Measured on the 15-query ground-truth set (`python3 eval_grounding.py`, prod
store copied, sha256 verified before/after; the prod store and ground-truth
file are private and not shipped in this repo — run the harness on your own
graph):

| scenario | fires | wrong node returned | tokens saved (lower bound) |
|---|---|---|---|
| question asked in the user's own words | 2/15 | 0 | measured per hit |
| question repeated as it is phrased in memory (FAQ / repeat) | 8/15 | 0 | 9053 total |

The 7 repeat-misses are blocked on purpose: 6 nodes had decayed below the weight
floor, 1 had a duplicate twin (`margin`). Faded and ambiguous memory does not
get to answer on its own.

### Verification quality (same run, 74 answers)

| answer form | verdicts | invention caught |
|---|---|---|
| verbatim quote of a node | `grounded` 15/15 | — (control: 0 false alarms) |
| retelling in other words | `grounded` 15/15 | — |
| quote + one invented sentence | `partial` 15/15 | 15/15 |
| quote with numbers swapped | `ungrounded` 12, `partial` 2 | 14/14 |
| fully invented answer | `ungrounded` 15/15 | 15/15 |

0 false alarms, 0 missed inventions. **Limit worth stating:** this is lexical
grounding. An answer that reuses the graph's vocabulary to state the opposite
("the trader's personal wallet, keys in the repo" against a node about the
production payTo wallet) lands in `partial`/`ungrounded`, not because the gate
understood the negation but because coverage and the number gate fell short.
Contradiction detection is `conflicts_with` edges in the graph, not this module.

### Audit trail

Every pass appends one line to `ground_log.jsonl` next to the store — agent,
time, query, node ids, verdict, counts, answer sha256 and preview. Append-only:
the record of *why the agent answered that way* has to be the one written at the
time. Read it with `memory_ground_log(limit, session_id?, event?, stats?)`.

Cost on a 5000-node store: `memory_ground` ~11 ms for a 20-claim answer;
`memory_answer` ~130 ms; `memory_ground_prepare` ~195 ms (two budget searches —
one for graph-first, one inside the prompt builder).

### Blank store and the default policy (03.09)

Grounded is the **start-up policy**, not an opt-in: `initialize` and `/health`
both report `ground_by_default: true`, and `memory_ground` takes its default for
`require_pre_pass` from it. Turn it off deliberately with
`--no-ground-by-default` or `MNEMOS_GROUND_BY_DEFAULT=0`.

A client instance starts with an **empty graph**: `--store blank` (or
`blank:<path>`, or `MNEMOS_STORE=blank` + `MNEMOS_STORE_PATH`) creates a graph
from `mnemos/data/nodes.blank.json` — an empty JSON object.

`blank` is a **provisioning** mode with two hard promises: it never overwrites an
existing file, *and* the graph it hands you is really empty. Point it at a path
that already holds nodes and the server refuses to start, naming the path and
telling you to run with `--store <path>` instead — quietly serving someone else's
graph under the word "blank" is the worse failure. A directory, a dangling
symlink or a non-graph file are refused for the same reason. So: provision once
with `blank`, run with the plain path afterwards.

If the template is ever found non-empty, start-up fails loudly: that is our leak
detector for someone else's nodes riding along in a release.

Client-facing page: [README.md](README.md).

## Запуск

```powershell
# тесты (нужен pytest: py -3.12 -m pip install pytest)
cd D:\mnemos
py -3.12 -m pytest -q            # публичное репо, наш прогон: 352 passed, 62 skipped (03.09)

# замер обязательного прохода через граф (боевой стор только копируется)
py -3.12 eval_grounding.py

# MCP-сервер (хранилище nodes.json рядом)
py -3.12 -m mnemos --host 127.0.0.1 --port 8765 --store nodes.json

# новая клиентская инстанция: чистый граф + обязательный проход через граф
py -3.12 -m mnemos --store blank

# здоровье
Invoke-RestMethod http://127.0.0.1:8765/health

# из Python
from mnemos import Bus, Store, check_claim, make_node
```

## Следующие шаги (по роадмапу)

1. **Граф-индекс**: рёбра links в отдельный граф, обход по связям (основа П4).
2. **Векторный слой** поверх подстрочного поиска (эмбеддинги локально, без облака — white spot №4).
3. **Обсидиан-мост**: экспорт узлов в markdown-граф (человеко-читаемость, git-совместимость).
4. **Публикация в шину**: «нет вердикта — нет публикации» (white spot №8: гейт П1-П6 в append).
5. **Чекпоинты-конденсаты** сессий + восстановление.
6. **Бенчмарк П1-П6**: 100 утверждений (50/50), accuracy и $/узел (white spot №3).
