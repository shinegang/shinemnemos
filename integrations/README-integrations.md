# ShineMnemos — integrations

ShineMnemos runs as a local HTTP JSON-RPC server plus a stdio MCP bridge.
Any MCP client connects to the **bridge**, not to the HTTP port directly.

## 1. Start the memory server

```bash
cd /path/to/shinemnemos
python3.12 -m mnemos --host 127.0.0.1 --port 8765 --store data/nodes.json
```

Grounding is **on by default**: every agent is expected to pass through the
memory graph before answering (memory_ground_prepare → answer → memory_ground).

## 2. Point your client at the bridge

Replace `/ABSOLUTE/PATH/TO/shinemnemos` with the real checkout path, then copy
the matching config:

| Client | Config |
| ------ | ------ |
| Claude Desktop | `claude_desktop.json` → `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows) |
| Claude Code | `bash integrations/claude_code_add.sh` |
| Cursor | `cursor_mcp.json` → `.cursor/mcp.json` in the project |
| llama.cpp | `llama_cpp_mcp_servers.json` via `--mcp-servers-json` (or `--ui-mcp-proxy`) |
| LangChain | `pip install langchain-mcp-adapters`, see `langchain_example.py` |
| OpenAI Agents SDK / Gemini CLI / Windsurf / VS Code | any of the `mcpServers` blocks above — they all accept the standard MCP JSON shape |

## 3. Bridge tuning (optional)

All knobs are environment variables: `MNEMOS_FORMAT=full` for uncompressed
results, `MNEMOS_MAX_RESULT_CHARS`, `MNEMOS_MAX_ITEMS`,
`MNEMOS_TOOLS` (whitelist), `MNEMOS_BRIDGE_LOG` (jsonl call log).

Self-check without a model:

```bash
python3.12 bridge/mnemos_bridge.py --selftest   # against a live server
python3.12 bridge/mnemos_bridge.py --tools      # schemas the model will see
```
