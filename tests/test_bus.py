# -*- coding: utf-8 -*-
"""Тесты шины: append/read, фильтры, heartbeat, конкурентные записи."""

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from mnemos.bus import BROADCAST, Bus


def test_append_creates_file(bus, tmp_path):
    bus.append(from_id="agent-a", text="привет")
    assert (tmp_path / "mnemos_bus.jsonl").exists()


def test_append_returns_message_with_ts(bus):
    msg = bus.append(from_id="agent-a", to="agent-b", kind="msg", text="привет")
    assert msg["from"] == "agent-a"
    assert msg["to"] == "agent-b"
    assert msg["kind"] == "msg"
    assert msg["text"] == "привет"
    assert "T" in msg["ts"] and msg["ts"].endswith("+00:00")


def test_append_line_is_valid_jsonl(bus):
    bus.append(from_id="agent-a", text="один")
    bus.append(from_id="agent-b", text="два")
    with open(bus.path, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert [m["text"] for m in lines] == ["один", "два"]


def test_read_all_in_order(bus):
    bus.append(from_id="a", text="1")
    bus.append(from_id="b", text="2")
    bus.append(from_id="c", text="3")
    msgs = bus.read()
    assert [m["text"] for m in msgs] == ["1", "2", "3"]


def test_read_filter_by_sender(bus):
    bus.append(from_id="a", text="1")
    bus.append(from_id="b", text="2")
    bus.append(from_id="a", text="3")
    msgs = bus.read(sender="a")
    assert [m["text"] for m in msgs] == ["1", "3"]


def test_read_filter_by_receiver_broadcast(bus):
    bus.append(from_id="a", to="b", text="1")
    bus.append(from_id="a", to="*", text="2")
    bus.append(from_id="a", to="c", text="3")
    msgs = bus.read(receiver="b")
    assert [m["text"] for m in msgs] == ["1", "2"]  # broadcast доходит до всех


def test_read_filter_by_kind(bus):
    bus.append(from_id="a", kind="msg", text="m")
    bus.append(from_id="a", kind="heartbeat", text="h")
    bus.append(from_id="a", kind="duty", text="d")
    assert [m["kind"] for m in bus.read(kinds=("heartbeat", "duty"))] == [
        "heartbeat", "duty",
    ]


def test_read_limit(bus):
    for i in range(5):
        bus.append(from_id="a", text=str(i))
    assert len(bus.read(limit=2)) == 2


def test_read_empty_bus_returns_empty(bus):
    assert bus.read() == []
    assert bus.count() == 0


def test_append_invalid_kind_raises(bus):
    with pytest.raises(ValueError):
        bus.append(from_id="a", kind="spam", text="x")


def test_append_empty_sender_raises(bus):
    with pytest.raises(ValueError):
        bus.append(from_id="", text="x")


def test_count(bus):
    bus.append(from_id="a", text="1")
    bus.append(from_id="b", text="2")
    assert bus.count() == 2


def test_broadcast_constant():
    assert BROADCAST == "*"


# --- heartbeat ----------------------------------------------------------------

def test_beat_first_time_writes_heartbeat(bus):
    msg = bus.beat("agent-a", interval_check=60)
    assert msg is not None
    assert msg["kind"] == "heartbeat"
    assert msg["from"] == "agent-a"


def test_beat_within_interval_suppressed(bus):
    bus.beat("agent-a", interval_check=60)
    second = bus.beat("agent-a", interval_check=60)
    assert second is None
    assert bus.count() == 1


def test_beat_after_interval_writes_again(bus):
    bus.beat("agent-a", interval_check=60)
    # старый heartbeat: проставляем ts 2 минуты назад и снова бьём
    old = datetime.now(timezone.utc) - timedelta(minutes=2)
    bus.append(from_id="agent-a", kind="heartbeat",
               text="old", ts=old.isoformat(timespec="microseconds"))
    msg = bus.beat("agent-a", interval_check=60)
    assert msg is not None
    assert bus.count() == 3  # 1 + подмена + новый


def test_beat_with_callable_interval_check(bus):
    # callable решает: бьём, только если last равен None
    first = bus.beat("agent-a", interval_check=lambda _b, _f, last: last is None)
    assert first is not None
    second = bus.beat("agent-a", interval_check=lambda _b, _f, last: last is None)
    assert second is None


def test_beat_distinct_agents_independent(bus):
    bus.beat("agent-a", interval_check=60)
    assert bus.beat("agent-b", interval_check=60) is not None
    assert bus.count() == 2


def test_heartbeat_readable_via_read(bus):
    bus.beat("agent-a", interval_check=60, text="alive@x")
    hb = bus.read(sender="agent-a", kinds=("heartbeat",))
    assert len(hb) == 1
    assert hb[0]["text"] == "alive@x"


# --- конкурентность -------------------------------------------------------------

def test_concurrent_appends_are_safe(bus):
    n_threads, per_thread = 6, 10

    def worker(wid):
        for i in range(per_thread):
            bus.append(from_id=f"w{wid}", text=f"{wid}-{i}")

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert bus.count() == n_threads * per_thread
    msgs = bus.read()
    pairs = [(m["from"], m["text"]) for m in msgs]
    assert len(set(pairs)) == n_threads * per_thread  # ни одно сообщение не потеряно/не задвоено
    assert all(m["kind"] == "msg" for m in msgs)
