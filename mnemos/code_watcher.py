# -*- coding: utf-8 -*-
"""ShineMnemos: фоновое обновление code-слоя (R1 — устаревание индекса).

Поток раз в N секунд (по умолчанию 60) сканирует корни по mtime+size, при
расхождении считает sha1 и перепарсит ТОЛЬКО изменившиеся файлы.

Замеры PoC на боевом /opt/acmetrader (§2.4 отчёта): скан 2013 файлов —
53.4 мс, перепарс одного файла ~13 мс. То есть 53 мс работы на 60 000 мс
интервала = 0.09% одного ядра.

Поток демонический: не мешает завершению процесса. Ошибка одного репозитория
не роняет ни поток, ни сервер — она копится в last_errors и видна в
code_refresh/status.

Конфигурация:
  MNEMOS_CODE_WATCH      — "0"/"off"/"false" выключает поток совсем;
  MNEMOS_CODE_WATCH_SEC  — интервал в секундах (по умолчанию 60, минимум 5).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

ENV_WATCH = "MNEMOS_CODE_WATCH"
ENV_INTERVAL = "MNEMOS_CODE_WATCH_SEC"
DEFAULT_INTERVAL = 60.0
MIN_INTERVAL = 5.0

_OFF = {"0", "off", "false", "no", "none"}


def watch_enabled(env: Optional[str] = None) -> bool:
    raw = env if env is not None else os.environ.get(ENV_WATCH)
    if raw is None:
        return True  # плагин code сам по себе выключен по умолчанию
    return raw.strip().lower() not in _OFF


def watch_interval(env: Optional[str] = None) -> float:
    raw = env if env is not None else os.environ.get(ENV_INTERVAL)
    if raw is None:
        return DEFAULT_INTERVAL
    try:
        return max(MIN_INTERVAL, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL


class CodeWatcher:
    """Фоновый поток: периодический инкрементальный code_refresh реестра."""

    def __init__(self, registry: Any, interval: Optional[float] = None,
                 initial_delay: float = 0.0) -> None:
        self.registry = registry
        self.interval = watch_interval() if interval is None else max(MIN_INTERVAL, float(interval))
        self.initial_delay = max(0.0, float(initial_delay))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.ticks = 0
        self.refreshed = 0          # сколько репо реально переиндексировано
        self.last_tick_ts: float = 0.0
        self.last_stats: List[Dict[str, Any]] = []
        self.last_errors: List[str] = []

    # -- жизненный цикл ----------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> "CodeWatcher":
        if self.running:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="mnemos-code-watcher",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None

    def _loop(self) -> None:
        if self.initial_delay and self._stop.wait(self.initial_delay):
            return
        while not self._stop.is_set():
            self.tick()
            if self._stop.wait(self.interval):
                return

    # -- одна итерация (вызывается и из тестов напрямую) -------------------
    def tick(self) -> List[Dict[str, Any]]:
        with self._lock:
            stats: List[Dict[str, Any]] = []
            errors: List[str] = []
            try:
                stats = self.registry.refresh(force=False)
            except Exception as exc:  # поток не должен умирать никогда
                errors.append(repr(exc))
            for s in stats:
                if s.get("error"):
                    errors.append(f"{s.get('repo', '?')}: {s['error']}")
                elif s.get("mode") in ("full", "incremental"):
                    self.refreshed += 1
            self.ticks += 1
            self.last_tick_ts = time.time()
            self.last_stats = stats
            self.last_errors = errors[-20:]
            return stats

    # -- диагностика -------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "interval_sec": self.interval,
            "ticks": self.ticks,
            "repos_reindexed": self.refreshed,
            "last_tick_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime(self.last_tick_ts))
            if self.last_tick_ts else "",
            "seconds_since_tick": round(time.time() - self.last_tick_ts, 1)
            if self.last_tick_ts else None,
            "errors": list(self.last_errors),
        }
