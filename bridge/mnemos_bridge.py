#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ShineMnemos stdio MCP bridge.

Turns the ShineMnemos HTTP JSON-RPC endpoint into a stdio MCP server, so any
MCP client (Claude Desktop / Claude Code / Cursor / llama.cpp / LangChain /
OpenAI Agents SDK / Gemini CLI) can use it with a two-line config.

Why a bridge exists: a raw memory_search answer can carry thousands of tokens
of full node JSON (truth_check, evidence, link_meta, usage). The bridge keeps
one connection to Mnemos, exposes compact tool schemas, compresses results to
"id | claim [score]" lines, caps them honestly (the cap is reported in the
text), and logs every call for measurement.

Run (as a stdio MCP server):
    MNEMOS_URL=http://127.0.0.1:8765/ python3 bridge/mnemos_bridge.py

Checks without a model:
    python3 bridge/mnemos_bridge.py --selftest   # against a live Mnemos
    python3 bridge/mnemos_bridge.py --tools      # schemas the model will see

Environment:
    MNEMOS_URL              http://127.0.0.1:8765/   Mnemos HTTP address
    MNEMOS_TOOLS            whitelist (default: all six grounded tools)
    MNEMOS_FORMAT           compact | full          (default compact)
    MNEMOS_MAX_RESULT_CHARS 1500                    result cap in characters
    MNEMOS_MAX_ITEMS        5                       max nodes per result
    MNEMOS_CLAIM_CHARS      400                     max claim length
    MNEMOS_TIMEOUT          20                      request timeout, sec
    MNEMOS_CONNECT_TIMEOUT  3                       connect timeout, sec
    MNEMOS_BRIDGE_LOG       (empty)                 jsonl call log file
    MNEMOS_FULL_SCHEMAS     0                       1 = pass Mnemos schemas as-is
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# ---------------------------------------------------------------- configuration

URL = os.environ.get("MNEMOS_URL", "http://127.0.0.1:8765/")
DEFAULT_TOOLS = ("memory_prompt,memory_search,memory_add,"
                 "memory_ground_prepare,memory_ground,memory_answer")
TOOLS_WHITELIST = [t.strip() for t in os.environ.get(
    "MNEMOS_TOOLS", DEFAULT_TOOLS).split(",") if t.strip()]
FORMAT = os.environ.get("MNEMOS_FORMAT", "compact").strip().lower()
MAX_RESULT_CHARS = int(os.environ.get("MNEMOS_MAX_RESULT_CHARS", "1500"))
MAX_ITEMS = int(os.environ.get("MNEMOS_MAX_ITEMS", "5"))
CLAIM_CHARS = int(os.environ.get("MNEMOS_CLAIM_CHARS", "400"))
TIMEOUT = float(os.environ.get("MNEMOS_TIMEOUT", "20"))
CONNECT_TIMEOUT = float(os.environ.get("MNEMOS_CONNECT_TIMEOUT", "3"))
LOG_PATH = os.environ.get("MNEMOS_BRIDGE_LOG", "")
FULL_SCHEMAS = os.environ.get("MNEMOS_FULL_SCHEMAS", "0") == "1"

PROTOCOL_VERSION = "2024-11-05"
BRIDGE_NAME = "shinemnemos-bridge"
BRIDGE_VERSION = "1.0.0"

# Compact schemas: the model pays for these tokens on EVERY request.
SHORT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "memory_prompt": {
        "name": "memory_prompt",
        "description": "Memory: assemble ready context for a question in ONE call. Call this instead of a series of searches.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_tokens": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    "memory_search": {
        "name": "memory_search",
        "description": "Memory: find facts by query. Returns 'id | fact' lines.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    "memory_add": {
        "name": "memory_add",
        "description": "Memory: store a new fact (claim) with its source.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "source": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["claim"],
        },
    },
    "memory_ground_prepare": {
        "name": "memory_ground_prepare",
        "description": "Memory: mandatory graph pass BEFORE answering. Returns context and session_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "session_id": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    "memory_ground": {
        "name": "memory_ground",
        "description": "Memory: check a ready answer against the graph (returns verdict).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "query": {"type": "string"},
                "answer_text": {"type": "string"},
            },
            "required": ["answer_text"],
        },
    },
    "memory_answer": {
        "name": "memory_answer",
        "description": "Memory: final grounded answer with mode/hit info.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
}


# ------------------------------------------------------------------- transport

class MnemosClient:
    """One keep-alive connection to Mnemos with one retry.

    The connection is reused only while the server answers HTTP/1.1 without
    Connection: close; otherwise the bridge reconnects honestly.
    """

    def __init__(self, url: str = URL) -> None:
        u = urlparse(url)
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port or 80
        self.path = u.path or "/"
        self._conn: Optional[http.client.HTTPConnection] = None
        self.reused = 0
        self.reconnects = 0

    def _connect(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(
                self.host, self.port, timeout=CONNECT_TIMEOUT)
            self._conn.connect()
            self._conn.sock.settimeout(TIMEOUT)
            self.reconnects += 1
        return self._conn

    def _drop(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except OSError:
            pass
        self._conn = None

    def rpc(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8",
                   "Connection": "keep-alive",
                   "Content-Length": str(len(body))}
        last: Optional[Exception] = None
        for attempt in (1, 2):  # second attempt: stale keep-alive
            try:
                conn = self._connect()
                had = conn.sock is not None
                conn.request("POST", self.path, body=body, headers=headers)
                resp = conn.getresponse()
                raw = resp.read()
                if resp.will_close:
                    self._drop()
                elif had:
                    self.reused += 1
                if not raw:
                    return {"jsonrpc": "2.0", "id": payload.get("id"),
                            "result": {"content": [{"type": "text", "text": "ok"}]}}
                return json.loads(raw.decode("utf-8"))
            except (http.client.HTTPException, OSError, ValueError) as exc:
                last = exc
                self._drop()
                if attempt == 2:
                    break
        return {"jsonrpc": "2.0", "id": payload.get("id"),
                "error": {"code": -32001, "message": f"Mnemos unavailable: {last}"}}


# --------------------------------------------------------------- result shaping

def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _cap(lines: List[str], total: int, shown: int) -> str:
    """Join lines under MAX_RESULT_CHARS and honestly report the truncation."""
    out: List[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > MAX_RESULT_CHARS:
            break
        out.append(line)
        used += len(line) + 1
    text = "\n".join(out)
    if len(out) < total:
        text += (f"\n(showing {len(out)} of {total}; full JSON — "
                 f"same call with format=\"full\")")
    return text or "(empty)"


def compact_search(data: Dict[str, Any]) -> str:
    results = data.get("results") or []
    total = len(results)
    lines: List[str] = []
    for r in results[:MAX_ITEMS]:
        node = r.get("node") if isinstance(r.get("node"), dict) else r
        nid = node.get("id", "?")
        claim = _clip(node.get("claim") or node.get("text") or "", CLAIM_CHARS)
        score = r.get("score", node.get("score"))
        tail = f" [{float(score):.2f}]" if isinstance(score, (int, float)) else ""
        lines.append(f"{nid} | {claim}{tail}")
    head = f"found {total} (mode {data.get('mode', '?')})"
    return head + "\n" + _cap(lines, min(total, MAX_ITEMS), MAX_ITEMS)


def compact_prompt(data: Dict[str, Any]) -> str:
    text = data.get("text") or data.get("prompt") or ""
    ids = data.get("node_ids") or []
    body = _clip(text, MAX_RESULT_CHARS) if len(text) > MAX_RESULT_CHARS else text
    if ids:
        body += "\nIDS: " + ",".join(ids[:MAX_ITEMS * 2])
    return body


def compact_ground_prepare(data: Dict[str, Any]) -> str:
    parts: List[str] = []
    sid = data.get("session_id")
    if sid:
        parts.append(f"session_id: {sid}")
    gf = data.get("graph_first") or {}
    if isinstance(gf, dict) and gf.get("hit") and gf.get("answer"):
        parts.append("READY ANSWER FROM GRAPH: " + _clip(gf["answer"], CLAIM_CHARS * 2))
    text = data.get("text") or data.get("prompt") or ""
    if text:
        parts.append(_clip(text, MAX_RESULT_CHARS))
    ids = data.get("node_ids") or []
    if ids:
        parts.append("IDS: " + ",".join(ids[:MAX_ITEMS * 2]))
    return "\n".join(parts) or "(empty)"


def compact_ground(data: Dict[str, Any]) -> str:
    counts = data.get("counts") or {}
    return (f"verdict: {data.get('verdict')} | ratio: {data.get('grounded_ratio')} | "
            f"supported {counts.get('supported', '?')}/{counts.get('total', '?')}")


def compact_answer(data: Dict[str, Any]) -> str:
    ans = data.get("answer")
    if ans:
        return f"[{data.get('mode', '?')}] " + _clip(ans, MAX_RESULT_CHARS)
    return f"[{data.get('mode', '?')}] hit={data.get('hit')} llm_required={data.get('llm_required')}"


def compact_add(data: Dict[str, Any]) -> str:
    node = data.get("node") or data
    return f"stored: {node.get('id', data.get('id', 'ok'))}"


COMPACTORS = {
    "memory_search": compact_search,
    "memory_prompt": compact_prompt,
    "memory_ground_prepare": compact_ground_prepare,
    "memory_ground": compact_ground,
    "memory_answer": compact_answer,
    "memory_add": compact_add,
}


def compact(tool: str, text: str) -> str:
    """Compress a tool result. Any failure — pass through under the char cap:
    silently dropping data is never allowed."""
    fn = COMPACTORS.get(tool)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return _clip(text, MAX_RESULT_CHARS)
    if fn is None or not isinstance(data, dict):
        return _clip(text, MAX_RESULT_CHARS)
    try:
        return fn(data)
    except (KeyError, TypeError, ValueError):
        return _clip(text, MAX_RESULT_CHARS)


# ------------------------------------------------------------------- MCP logic

class Bridge:
    def __init__(self) -> None:
        self.client = MnemosClient()
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self.log_fh = open(LOG_PATH, "a", encoding="utf-8") if LOG_PATH else None

    def _log(self, **kw: Any) -> None:
        if self.log_fh is None:
            return
        kw["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.log_fh.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self.log_fh.flush()

    def tools(self) -> List[Dict[str, Any]]:
        if self._tools_cache is not None:
            return self._tools_cache
        upstream = self.client.rpc({"jsonrpc": "2.0", "id": "t", "method": "tools/list",
                                    "params": {}})
        got = (upstream.get("result") or {}).get("tools") or []
        names = {t.get("name") for t in got}
        out: List[Dict[str, Any]] = []
        for name in TOOLS_WHITELIST:
            if names and name not in names:
                continue  # tool absent upstream — never promise it to the model
            if FULL_SCHEMAS or name not in SHORT_SCHEMAS:
                found = next((t for t in got if t.get("name") == name), None)
                if found:
                    out.append(found)
            else:
                out.append(SHORT_SCHEMAS[name])
        if not out:  # server silent — offer the whitelist's short schemas
            out = [SHORT_SCHEMAS[n] for n in TOOLS_WHITELIST if n in SHORT_SCHEMAS]
        self._tools_cache = out
        return out

    def call(self, req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        args = dict(params.get("arguments") or {})
        want_full = str(args.pop("format", "")).lower() == "full" or FORMAT == "full"
        if TOOLS_WHITELIST and name not in TOOLS_WHITELIST:
            return {"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32601, "message": f"tool not enabled: {name}"}}
        t0 = time.perf_counter()
        resp = self.client.rpc({"jsonrpc": "2.0", "id": req_id, "method": "tools/call",
                                "params": {"name": name, "arguments": args}})
        ms = (time.perf_counter() - t0) * 1000
        if "error" in resp:
            self._log(tool=name, ms=round(ms, 2), error=resp["error"].get("message"))
            return {"jsonrpc": "2.0", "id": req_id, "error": resp["error"]}
        content = (resp.get("result") or {}).get("content") or [{}]
        text = content[0].get("text", "")
        out = text if want_full else compact(name, text)
        self._log(tool=name, ms=round(ms, 2), chars_in=len(text), chars_out=len(out),
                  reused=self.client.reused, reconnects=self.client.reconnects)
        return {"jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": out}], "isError": False}}

    def handle(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = req.get("method")
        req_id = req.get("id")
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": req_id, "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": BRIDGE_NAME, "version": BRIDGE_VERSION}}}
        if method in ("notifications/initialized", "initialized", "notifications/cancelled"):
            return None
        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": self.tools()}}
        if method == "tools/call":
            return self.call(req_id, req.get("params") or {})
        if method in ("resources/list", "prompts/list"):
            key = method.split("/")[0]
            return {"jsonrpc": "2.0", "id": req_id, "result": {key: []}}
        if req_id is None:
            return None
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"method not supported: {method}"}}

    def serve(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {
                    "code": -32700, "message": "invalid JSON"}}, ensure_ascii=False) + "\n")
                sys.stdout.flush()
                continue
            reqs = req if isinstance(req, list) else [req]
            for r in reqs:
                resp = self.handle(r) if isinstance(r, dict) else None
                if resp is not None:
                    sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                    sys.stdout.flush()


# ------------------------------------------------------------------- selftest

def _selftest() -> int:
    """Run against a live Mnemos: chars/tokens before and after compression."""
    try:
        from tokenizers import Tokenizer  # optional
        tok_path = os.environ.get("QWEN_TOKENIZER", "/tmp/qwen_tok.json")
        tok = Tokenizer.from_file(tok_path)
        ntok = lambda s: len(tok.encode(s).ids)  # noqa: E731
        unit = "tok"
    except Exception:
        ntok = len
        unit = "chars"
    b = Bridge()
    cases = [
        ("memory_prompt", {"query": "setup", "max_tokens": 1200}),
        ("memory_search", {"query": "setup", "top_k": 5}),
        ("memory_search", {"query": "rule", "top_k": 5}),
        ("memory_ground_prepare", {"query": "What is stored about the project?",
                                    "session_id": "bridge-selftest"}),
    ]
    schemas = json.dumps(b.tools(), ensure_ascii=False)
    print(f"schemas for the model: {len(b.tools())} tools, {ntok(schemas)} {unit}")
    tot_in = tot_out = 0
    for name, args in cases:
        raw = b.client.rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                            "params": {"name": name, "arguments": args}})
        if "error" in raw:
            print(f"  {name:22s} ERROR: {raw['error']['message']}")
            continue
        text = raw["result"]["content"][0]["text"]
        out = compact(name, text)
        tot_in += ntok(text)
        tot_out += ntok(out)
        print(f"  {name:22s} {ntok(text):6d} -> {ntok(out):5d} {unit}  "
              f"(x{ntok(text) / max(1, ntok(out)):.1f})")
    print(f"results total: {tot_in} -> {tot_out} {unit} "
          f"(x{tot_in / max(1, tot_out):.1f}); "
          f"connection reused {b.client.reused} times, reconnects {b.client.reconnects}")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return _selftest()
    if "--tools" in sys.argv:
        print(json.dumps(Bridge().tools(), ensure_ascii=False, indent=2))
        return 0
    Bridge().serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
