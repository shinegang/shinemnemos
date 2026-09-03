#!/bin/bash
# Claude Code CLI: register the ShineMnemos bridge as an MCP server.
# Run from the repo root.
set -u
claude mcp add shinemnemos \
  --env MNEMOS_URL=http://127.0.0.1:8765/ \
  -- python3.12 "$(pwd)/bridge/mnemos_bridge.py"
