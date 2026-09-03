# ShineMnemos — grounded memory for agents

**You get the tools — your graph starts empty. Grounded answers are ON by default.**

ShineMnemos is a local MCP memory server. We sell the tools, not the data: a fresh
instance starts with a **blank graph** — not one node of anyone else's memory — and the
agent running on top of it **cannot answer without going through that graph first**.

Python 3.12, standard library only at runtime. No cloud, no model API.

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
`serverInfo.ground_by_default` and the instruction text — so an agent learns the contract
before its first answer instead of from a README it never read.

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

## 3. Quickstart

```bash
python -m mnemos --host 127.0.0.1 --port 8765 --store blank
curl -s http://127.0.0.1:8765/health
# {"ok": true, "nodes": 0, "ground_by_default": true, ...}
```

Fill it with your own facts:

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
python -m pytest tests -q
```

`tests/test_ground_default.py` covers exactly what this page promises: a blank instance
answers `memory_search` with zero nodes, an answer without a pre-pass comes back
`ungrounded`, `blank` does not overwrite an existing graph, and no node store ships inside
the package.
