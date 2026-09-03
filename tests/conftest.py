# -*- coding: utf-8 -*-
"""Общие фикстуры для тестов ShineMnemos."""

import json
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

# D:\mnemos должен быть в sys.path, чтобы `import mnemos` работал из tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos import Bus, Store  # noqa: E402
from mnemos.server import MCPHttpServer  # noqa: E402


@pytest.fixture
def bus(tmp_path):
    return Bus(tmp_path / "mnemos_bus.jsonl")


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "nodes.json")


@pytest.fixture
def server(tmp_path):
    """MCP-сервер на эфемерном порту, поднятый в фоновом потоке."""
    store = Store(tmp_path / "nodes.json")
    httpd = MCPHttpServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def rpc(server):
    """Клиент JSON-RPC: rpc(method, params=None, _id=1) -> dict ответа."""
    port = server.server_address[1]

    def call(method, params=None, _id=1):
        body = {"jsonrpc": "2.0", "id": _id, "method": method}
        if params is not None:
            body["params"] = params
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    return call


@pytest.fixture
def good_claim():
    """Узел, проходящий все 6 проверок П1-П6 (при свежем ts)."""
    return {
        "claim": "Выручка выросла на 12% в Q3 2025",
        "source": "финотчёт компании за Q3",
        "evidence": ["стр. 4 отчёта, таблица 2"],
        "context": "Квартальный обзор, сравниваем с Q2",
        "links": [],
    }
