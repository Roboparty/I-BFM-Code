from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _jsonable(value: Any) -> Any:
    """Convert numpy-rich runtime snapshots without doing JSON work in the control loop."""
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return _jsonable(value.item())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class CarryRunRecorder:
    """Non-blocking JSONL recorder for real-time carry deployments.

    The control loop only copies a small snapshot and calls ``put_nowait``.
    JSON encoding and disk writes happen on a dedicated daemon thread.
    """

    _STOP = object()

    def __init__(
        self,
        log_root: str | os.PathLike[str],
        metadata: Mapping[str, Any],
        *,
        queue_size: int = 4096,
    ) -> None:
        if int(queue_size) <= 0:
            raise ValueError("queue_size must be positive")

        self.log_root = Path(log_root).expanduser().resolve(strict=False)
        self.log_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = self._create_unique_run_dir(timestamp)
        self.samples_path = self.run_dir / "samples.jsonl"
        self.metadata_path = self.run_dir / "metadata.json"
        self.summary_path = self.run_dir / "summary.json"

        full_metadata = {
            "schema_version": 1,
            "created_at_local": datetime.now().astimezone().isoformat(),
            "run_name": self.run_dir.name,
            **dict(metadata),
        }
        with self.metadata_path.open("x", encoding="utf-8") as file:
            json.dump(_jsonable(full_metadata), file, ensure_ascii=False, indent=2)
            file.write("\n")

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=int(queue_size))
        self._ready = threading.Event()
        self._closed = False
        self._writer_error: BaseException | None = None
        self.sample_count = 0
        self.dropped_count = 0
        self._thread = threading.Thread(
            target=self._writer_main,
            name=f"carry-recorder-{self.run_dir.name}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("Timed out while starting carry data recorder")
        if self._writer_error is not None:
            raise RuntimeError(f"Failed to start carry data recorder: {self._writer_error}")

    def _create_unique_run_dir(self, timestamp: str) -> Path:
        for suffix in range(1000):
            name = timestamp if suffix == 0 else f"{timestamp}_{suffix:03d}"
            candidate = self.log_root / name
            try:
                candidate.mkdir(exist_ok=False)
                return candidate
            except FileExistsError:
                continue
        raise RuntimeError(f"Could not allocate a unique recording directory under {self.log_root}")

    @property
    def writer_error(self) -> str | None:
        return None if self._writer_error is None else str(self._writer_error)

    def record(self, sample: Mapping[str, Any]) -> bool:
        """Queue one sample without ever waiting on the filesystem."""
        if self._closed or self._writer_error is not None:
            return False
        try:
            self._queue.put_nowait(dict(sample))
            return True
        except queue.Full:
            self.dropped_count += 1
            return False

    def close(self, timeout_s: float = 10.0) -> None:
        if self._closed:
            return
        self._closed = True

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        stop_queued = False
        while self._thread.is_alive() and time.monotonic() < deadline:
            try:
                self._queue.put(self._STOP, timeout=0.1)
                stop_queued = True
                break
            except queue.Full:
                continue
        if not stop_queued and self._thread.is_alive() and self._writer_error is None:
            self._writer_error = RuntimeError("Recorder queue did not drain before timeout")
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive() and self._writer_error is None:
            self._writer_error = RuntimeError("Recorder writer did not stop before timeout")

        summary = {
            "schema_version": 1,
            "closed_at_local": datetime.now().astimezone().isoformat(),
            "samples_written": int(self.sample_count),
            "samples_dropped": int(self.dropped_count),
            "writer_error": self.writer_error,
        }
        try:
            with self.summary_path.open("x", encoding="utf-8") as file:
                json.dump(summary, file, ensure_ascii=False, indent=2)
                file.write("\n")
        except Exception as exc:
            if self._writer_error is None:
                self._writer_error = exc

    def _writer_main(self) -> None:
        try:
            with self.samples_path.open("x", encoding="utf-8", buffering=1024 * 1024) as file:
                self._ready.set()
                while True:
                    item = self._queue.get()
                    try:
                        if item is self._STOP:
                            break
                        file.write(
                            json.dumps(
                                _jsonable(item),
                                ensure_ascii=False,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                        )
                        file.write("\n")
                        self.sample_count += 1
                        if self.sample_count % 50 == 0:
                            file.flush()
                    finally:
                        self._queue.task_done()
                file.flush()
        except BaseException as exc:
            self._writer_error = exc
            self._ready.set()
