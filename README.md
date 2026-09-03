# ShineMnemos — grounded memory for agents

**You get the tools — your graph starts empty. Grounded answers are ON by default.**

ShineMnemos is a local MCP memory server that makes an agent's memory
*enforceable*: every answer passes through your knowledge graph and comes back
with a verdict — `grounded`, `partial` or `ungrounded` — plus the exact
sentences the model made up. If the answer is already in the graph, it is
returned as-is and the LLM is never called. A fresh instance starts with a
blank graph: we sell the tools, not the data. Python 3.12, standard library
only at runtime. No cloud, no model API.

**[Website](https://shinegang.click)** ·
**[Plans & pricing](https://shinegang.click/console.html#buy)** ·
[Integrations](integrations/README-integrations.md) ·
[Русский](README.ru.md)

## Quick start

```bash
git clone https://github.com/shinegang/shinemnemos.git && cd shinemnemos

# 1. start the memory server — stdlib only, nothing to install
python3 -m mnemos --host 127.0.0.1 --port 8765 --store blank

# 2. in a second terminal: check the stdio bridge against the live server
python3 bridge/mnemos_bridge.py --selftest

# 3. register it in your MCP client (Claude Code shown)
bash integrations/claude_code_add.sh
```

`curl -s http://127.0.0.1:8765/health` →
`{"ok": true, ..., "nodes": 0, "ground_by_default": true}`

Configs for Claude Desktop, Cursor, llama.cpp, LangChain — and any client that
takes the standard `mcpServers` JSON (OpenAI Agents SDK, Gemini CLI, Windsurf):
[integrations/](integrations/README-integrations.md). When you outgrow the free
engine: [plans on the site](https://shinegang.click/console.html#buy).

## Why ShineMnemos

Every claim below points at the file and line in this repo where it lives.

- **Grounded by default.** Call `memory_ground` without a prior
  `memory_ground_prepare` and the verdict is `ungrounded` (`no_pre_pass`), no
  matter how good the text reads — memory cannot be silently skipped
  ([mnemos/grounding.py:527](mnemos/grounding.py#L527); policy ON by default:
  [mnemos/server.py:102](mnemos/server.py#L102)).
- **Graph-first answers.** `memory_answer` returns the answer straight from the
  graph with zero generation tokens, or `llm_required: true` plus a ready
  grounded prompt ([mnemos/server.py:1267](mnemos/server.py#L1267)).
- **Tools, not data.** A fresh instance is provisioned from an empty template —
  [mnemos/data/nodes.blank.json](mnemos/data/nodes.blank.json) is `{}` — and
  `blank` refuses to overwrite an existing graph
  ([mnemos/store.py:240](mnemos/store.py#L240)). A template that ever ships
  non-empty is a start-up error
  ([mnemos/store.py:170](mnemos/store.py#L170)).
- **Every pass is audited.** `ground_log.jsonl` is an append-only journal next
  to your store: agent, time, query, node ids, verdict — written at
  [mnemos/grounding.py:736](mnemos/grounding.py#L736), read with
  `memory_ground_log` ([mnemos/server.py:1302](mnemos/server.py#L1302)).
- **Claims are truth-checked.** Every node runs the P1–P6 protocol — freshness,
  source, numbers, consistency, reproducibility, completeness — and passes only
  at ≥ 4 of 6 ([mnemos/truth_gate.py:29](mnemos/truth_gate.py#L29),
  [mnemos/truth_gate.py:215](mnemos/truth_gate.py#L215)).
- **Local & lean.** JSON-RPC on stdlib `http.server`
  ([mnemos/server.py:40](mnemos/server.py#L40)); zero third-party runtime
  dependencies ([requirements.txt](requirements.txt) — pytest is the only dev
  dependency). Optional extras (`fastembed` semantic search, tree-sitter code
  index) are guarded, never required.
- **Check, don't trust.** `python -m pytest tests -q` runs the whole suite on
  bare Python 3.12 + pytest (our run: 352 passed, 62 skipped — every skip is a
  missing optional dependency; tests gated on our private corpus also skip
  cleanly when it is absent), and
  [demo_gates.py](demo_gates.py) (`python demo_gates.py`) exercises the five
  memory-quality gates end to end on stdlib alone. The
  measurement harnesses ship too ([eval_grounding.py](eval_grounding.py),
  [eval_recall.py](eval_recall.py)) — they read the graph at `data/nodes.json`
  (empty in this repo) and a ground-truth file (`--gt`), so bring your own.

---

## 1. Grounded answers are ON by default

Every answer goes through the graph. This is the server's start-up policy, not an opt-in
flag you have to discover:

| Step | Tool | What it does |
|---|---|---|
| 1 — **before** the answer | `memory_ground_prepare(query, session_id)` | Searches the graph, builds a system prompt from the nodes it found, and **registers the pre-pass** for this session. Also returns `graph_first`: if the answer is already in memory, take it and skip the LLM entirely. |
| 2 — the answer | *(your model)* | Generates from the excerpt — or does not generate at all, if step 1 already had the answer. |
| 3 — **after** the answer | `memory_ground(answer_text, session_id)` | Splits the answer into claims, checks each one against the graph, and returns a verdict: `grounded` / `partial` / `ungrounded`, plus source nodes and `unsupported_claims` — what the model made up. |
| any time | `memory_answer(query)` | Graph-first on its own: returns the answer straight from the graph (zero generation tokens) or `llm_required: true` with a ready grounded prompt. |
| audit | `memory_ground_log` | Append-only journal of every pass through the graph: who asked, when, which nodes, which verdict. |

**No pre-pass, no credit.** If `memory_ground` is called without a matching
`memory_ground_prepare`, the verdict is `ungrounded` (`notes: no_pre_pass`) no matter how
many claims the text supports. That is the point: an answer that never consulted the graph
did not come from your memory, and the log says so.

The policy is advertised on the MCP handshake — `initialize` returns
`ground_by_default` in the result's `_meta` and, while the policy is on, the
policy text as `instructions` — so an agent learns the contract before its
first answer instead of from a README it never read.

Turning it off is a deliberate act:

```bash
python -m mnemos --no-ground-by-default        # this run only
MNEMOS_GROUND_BY_DEFAULT=0 python -m mnemos    # via environment
```

A client with no session concept can also pass `require_pre_pass: false` on a single
`memory_ground` call — claim checking still runs, only the pre-pass requirement is waived.

## 2. Your graph starts empty

```bash
python -m mnemos --store blank                       # ./nodes.json, empty
python -m mnemos --store blank:/var/lib/mnemos.json  # explicit path
MNEMOS_STORE=blank MNEMOS_STORE_PATH=/var/lib/mnemos.json python -m mnemos
```

`blank` is the **provisioning** mode, and it keeps two promises at once:

- it **never overwrites an existing file** — that file is someone's data;
- the graph you get **really is empty** — that is the promise on this page.

When those two collide — you pointed `blank` at a path that already holds a graph with
nodes — the server **refuses to start** and tells you what to do: run with
`--store <that path>` if it is your graph, or pick another path. Silently handing you
someone else's nodes under the word "blank" would be the worse failure. A path that exists
but is empty is not a collision; it starts normally.

So: provision once with `blank`, then run the service with the plain path.

```bash
python -m mnemos --store blank:/var/lib/mnemos/nodes.json   # first run
python -m mnemos --store /var/lib/mnemos/nodes.json         # every run after
```

`blank` also refuses a directory, a dangling symlink, or a file that is not a graph,
instead of starting a server whose writes would fail later.

The template it copies (`mnemos/data/nodes.blank.json`) is an empty JSON object, and the
server refuses to start if that template is ever found non-empty — that check exists so no
one else's nodes can ride along in a release.

Nothing in this distribution contains our operational store. What you get is the engine:
nodes, edges, search, gates, the truth-gate, plugins, and the grounding protocol above.

## 3. Put your own facts in

```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
 "params": {"name": "memory_add",
            "arguments": {"claim": "Deploys go out on Tuesdays",
                          "source": "team handbook",
                          "evidence": ["handbook.md#release"]}}}
```

Then run the three steps: `memory_ground_prepare` → your answer → `memory_ground`.

## 4. Configuration

| Option | Default | Meaning |
|---|---|---|
| `--store <path>` / `MNEMOS_STORE` | `./nodes.json` | Graph file. `blank` or `blank:<path>` creates an empty one. |
| `MNEMOS_STORE_PATH` | `./nodes.json` | Where `blank` puts the graph when no path is given. |
| `--ground-by-default` / `--no-ground-by-default` | **on** | Mandatory pass through the graph. |
| `MNEMOS_GROUND_BY_DEFAULT` | `1` | Same, via environment. `0` turns it off. |
| `--plugins`, `--plugins-config` / `MNEMOS_PLUGINS` | `context_engine,gates` | Enabled plugins. |

The pass journal (`ground_log.jsonl`) is written next to your graph file, never anywhere
else.

## 5. Tests

```bash
python -m pytest tests -q     # our run: 352 passed, 62 skipped
```

The skips are missing optional dependencies (tree-sitter, `fastembed`); tests
gated on our private corpus also skip cleanly when it is absent — nothing fails.

`tests/test_ground_default.py` covers exactly what this page promises: a blank
instance answers `memory_search` with zero nodes
([tests/test_ground_default.py:74](tests/test_ground_default.py#L74)), an answer
without a pre-pass comes back `ungrounded`
([tests/test_ground_default.py:236](tests/test_ground_default.py#L236)),
`blank` does not overwrite an existing graph
([tests/test_ground_default.py:110](tests/test_ground_default.py#L110)), and no
node store ships inside the package
([tests/test_ground_default.py:195](tests/test_ground_default.py#L195)).

## Get a plan

The engine in this repo is Apache-2.0 and runs entirely on your machine — free.
Plans on the site add higher limits (agents, nodes, operations, journal
retention) and extra guards on top of it:

- **[shinegang.click](https://shinegang.click)** — what ShineMnemos is, product tour;
- **[Plans & pricing](https://shinegang.click/console.html#buy)** — six plans
  (Free, Solo, Pro, Team, Scale, Enterprise) from $0 to $499/month, Enterprise
  custom. Checkout is an x402 payment challenge — USDC on Base.
