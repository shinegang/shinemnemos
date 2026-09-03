# -*- coding: utf-8 -*-
"""ShineMnemos: шина агентов (mnemos_bus.jsonl).

Append-only JSONL-шина для общения агентов: сообщения (msg), хартбиты
(heartbeat) и дежурства (duty). Формат строки:

    {"ts": "2025-01-01T12:00:00.000000+00:00",
     "from": "agent-a", "to": "*", "kind": "msg",
     "text": "привет"}

Блокировка: Windows — msvcrt.locking по lock-файлу (region lock, байт 0)
с retry-циклом; POSIX — fcntl.flock. Сама запись идёт в режиме append
(аналог O_APPEND), так что параллельные писатели не затирают друг друга.

Heartbeat-writer: beat(from_id, interval_check) — пишет heartbeat только
если последний heartbeat этого агента старше интервала (или interval_check
вернул True).
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

KINDS = ("msg", "heartbeat", "duty")
BROADCAST = "*"  # значение `to` для сообщений всем

if os.name == "nt":  # Windows
    import msvcrt

    def _lock(fh) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(fh) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)

else:  # POSIX
    import fcntl

    def _lock(fh) -> None:
        # LOCK_NB обязателен (баг-хант 03.09, D7): без него flock ждёт вечно,
        # никогда не бросает OSError — и цикл ожидания с deadline в locked_file
        # оказывался мёртвым кодом, а объявленный timeout не срабатывал. Ждём
        # мы там сами, короткими попытками; блокирующий flock отнимал бы у нас
        # право сдаться. Раньше это висело только на шине, теперь на этом замке
        # сидит журнал проходов — то есть поток HTTP-запроса.
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


@contextlib.contextmanager
def locked_file(path: Union[str, Path], timeout: float = 10.0):
    """Эксклюзивная межпроцессная блокировка по lock-файлу рядом с path.

    Вынесено из Bus._locked (03.09), чтобы тем же замком пользовался
    append-only журнал проходов через граф (grounding.GroundLog): два
    независимых писателя одного формата не должны иметь двух разных
    реализаций блокировки.
    """
    path = Path(path)
    lock_path = path.with_name(path.name + ".lock")
    fh = open(lock_path, "a+b")  # создаёт lock-файл при необходимости
    try:
        if os.fstat(fh.fileno()).st_size == 0:
            fh.write(b"\0")
            fh.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                _lock(fh)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"не удалось взять блокировку {lock_path} за {timeout}с"
                    )
                time.sleep(0.02)
        try:
            yield
        finally:
            _unlock(fh)
    finally:
        fh.close()


class Bus:
    """Append-only JSONL-шина с межпроцессной блокировкой записи."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)

    # -- запись -------------------------------------------------------------
    def _locked(self, timeout: float = 10.0):
        """Эксклюзивная блокировка на время записи (по lock-файлу)."""
        return locked_file(self.path, timeout=timeout)

    def append(
        self,
        from_id: str,
        to: str = BROADCAST,
        kind: str = "msg",
        text: str = "",
        ts: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Добавляет сообщение в шину и возвращает его dict."""
        if not isinstance(from_id, str) or not from_id.strip():
            raise ValueError("from_id: отправитель не может быть пустым")
        if kind not in KINDS:
            raise ValueError(f"kind: ожидается одно из {KINDS}, получено {kind!r}")
        msg = {
            "ts": ts if ts is not None else _now_iso(),
            "from": from_id,
            "to": to,
            "kind": kind,
            "text": text,
        }
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        with self._locked(timeout=timeout):
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line)
                f.flush()
        return msg

    # -- чтение --------------------------------------------------------------
    def read(
        self,
        *,
        sender: Optional[str] = None,
        receiver: Optional[str] = None,
        kinds: Optional[Iterable[str]] = None,
        after_ts: Optional[str] = None,
        before_ts: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Читает сообщения в порядке записи с фильтрами."""
        if not self.path.exists():
            return []
        kinds_set = set(kinds) if kinds is not None else None
        out: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue  # оборванная строка (идут параллельные записи) — пропуск
                if not isinstance(msg, dict):
                    continue
                if sender is not None and msg.get("from") != sender:
                    continue
                if receiver is not None:
                    to = msg.get("to")
                    if to not in (receiver, BROADCAST):
                        continue
                if kinds_set is not None and msg.get("kind") not in kinds_set:
                    continue
                if after_ts is not None and (msg.get("ts") or "") < after_ts:
                    continue
                if before_ts is not None and (msg.get("ts") or "") > before_ts:
                    continue
                out.append(msg)
                if limit is not None and len(out) >= limit:
                    break
        return out

    def count(self) -> int:
        return len(self.read())

    # -- heartbeat ------------------------------------------------------------
    def _last_heartbeat(self, from_id: str) -> Optional[Dict[str, Any]]:
        hb = self.read(sender=from_id, kinds=("heartbeat",))
        return hb[-1] if hb else None

    def beat(
        self,
        from_id: str,
        interval_check: Union[float, int, Callable[[Any, str, Optional[Dict[str, Any]]], bool]] = 60.0,
        to: str = BROADCAST,
        text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Heartbeat-writer: пишет heartbeat, только если пора.

        interval_check — число (секунды): бьём, если прошло >= интервала с
        последнего heartbeat этого агента; или callable(bus, from_id, last_msg)
        -> bool: бьём, если вернул True. Возвращает сообщение или None.
        """
        last = self._last_heartbeat(from_id)
        if callable(interval_check):
            should = bool(interval_check(self, from_id, last))
        else:
            interval = float(interval_check)
            should = True
            if last is not None:
                last_dt = _parse_ts(last.get("ts"))
                if last_dt is not None:
                    age = (datetime.now(timezone.utc) - last_dt).total_seconds()
                    if age < interval:
                        should = False
        if not should:
            return None
        if text is None:
            text = f"alive@{_now_iso()}"
        return self.append(from_id, to=to, kind="heartbeat", text=text)
