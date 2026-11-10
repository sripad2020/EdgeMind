"""Background telemetry sampler with bounded memory.

* Samples are kept in a ring buffer (``deque(maxlen=capacity)``): O(capacity) memory,
  O(1) append, regardless of how long the runtime lives.
* Per-window aggregates (peaks, means) are updated incrementally on each sample, so ending
  a window is O(1) - no scan over the buffer.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import psutil

from .._validate import require_float, require_int
from .profiler import TelemetrySample, take_sample


@dataclass(frozen=True, slots=True)
class WindowStats:
    samples: int
    peak_rss: int
    peak_vram_used: int | None
    min_ram_available: int
    mean_cpu_percent: float
    max_temperature_c: int | None


class _WindowAccumulator:
    __slots__ = ("count", "cpu_sum", "max_temp", "min_avail", "peak_rss", "peak_vram")

    def __init__(self) -> None:
        self.count = 0
        self.peak_rss = 0
        self.peak_vram: int | None = None
        self.min_avail: int | None = None
        self.cpu_sum = 0.0
        self.max_temp: int | None = None

    def add(self, s: TelemetrySample) -> None:
        self.count += 1
        self.peak_rss = max(self.peak_rss, s.process_rss)
        if s.vram_used is not None:
            self.peak_vram = s.vram_used if self.peak_vram is None else max(self.peak_vram,
                                                                             s.vram_used)
        self.min_avail = (
            s.ram_available if self.min_avail is None else min(self.min_avail, s.ram_available)
        )
        self.cpu_sum += s.cpu_percent
        if s.temperature_c is not None:
            self.max_temp = (
                s.temperature_c if self.max_temp is None else max(self.max_temp, s.temperature_c)
            )

    def finish(self) -> WindowStats:
        return WindowStats(
            samples=self.count,
            peak_rss=self.peak_rss,
            peak_vram_used=self.peak_vram,
            min_ram_available=self.min_avail or 0,
            mean_cpu_percent=self.cpu_sum / self.count if self.count else 0.0,
            max_temperature_c=self.max_temp,
        )


class TelemetryMonitor:
    """Thread-safe background sampler. Use as a context manager or call start()/stop()."""

    def __init__(
        self, *, gpu_index: int | None = None, interval_s: float = 0.25, capacity: int = 1024
    ) -> None:
        self._interval = require_float("interval_s", interval_s, min_value=0.01, max_value=60.0)
        cap = require_int("capacity", capacity, min_value=1, max_value=1_000_000)
        self._gpu_index = gpu_index
        self._process = psutil.Process()
        self._buffer: deque[TelemetrySample] = deque(maxlen=cap)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._window: _WindowAccumulator | None = None
        psutil.cpu_percent(interval=None)  # prime the non-blocking CPU counter

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="air-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 4 + 1.0)
            self._thread = None

    def __enter__(self) -> TelemetryMonitor:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_now()
            self._stop.wait(self._interval)

    def sample_now(self) -> TelemetrySample:
        s = take_sample(self._process, self._gpu_index)
        with self._lock:
            self._buffer.append(s)
            if self._window is not None:
                self._window.add(s)
        return s

    def latest(self) -> TelemetrySample:
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
        return self.sample_now()

    def begin_window(self) -> None:
        with self._lock:
            self._window = _WindowAccumulator()

    def end_window(self) -> WindowStats:
        self.sample_now()  # guarantee at least one sample per window
        with self._lock:
            acc, self._window = self._window, None
        return (acc or _WindowAccumulator()).finish()
