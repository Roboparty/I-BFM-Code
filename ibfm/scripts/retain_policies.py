#!/usr/bin/env python3
"""Keep the latest policy and the lowest-EMD evaluated policy from a rolling run.

The rolling ``checkpoint/`` remains the resumable checkpoint. These copies
contain policy weights and configuration only; optimizer and replay are omitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path


def read_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (FileNotFoundError, OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def evaluation_score(path: Path, step: int, expected_rows: int) -> float | None:
    values = []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    row_step = int(float(row.get("timestep", "")))
                except (TypeError, ValueError):
                    continue
                if row_step != step:
                    continue
                value = float(row["emd"])
                if not math.isfinite(value):
                    return None
                values.append(value)
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError):
        return None
    if len(values) != expected_rows:
        return None
    return sum(values) / len(values)


def link_step(root: Path, name: str, step_dir: Path) -> None:
    pending = root / f".{name}.{os.getpid()}"
    pending.unlink(missing_ok=True)
    pending.symlink_to(step_dir.name, target_is_directory=True)
    pending.replace(root / name)


def current_step(root: Path, name: str) -> int | None:
    target = root / name
    if not target.is_symlink():
        return None
    data = read_json(target / "policy.json")
    return int(data["global_time"]) if data and "global_time" in data else None


def scan(run_dir: Path, expected_rows: int) -> None:
    checkpoint = run_dir / "checkpoint"
    status_path = checkpoint / "train_status.json"
    status = read_json(status_path)
    if status is None or "global_time" not in status:
        return
    step = int(status["global_time"])
    model_file = checkpoint / "model" / "model.safetensors"
    required = (
        model_file,
        checkpoint / "model" / "config.json",
        checkpoint / "model" / "init_kwargs.json",
        checkpoint / "config.json",
        checkpoint / "init_kwargs.json",
    )
    if any(not item.is_file() for item in required):
        return

    root = run_dir / "policies"
    root.mkdir(exist_ok=True)
    step_dir = root / f"step_{step:012d}"
    if current_step(root, "latest") != step and not step_dir.is_dir():
        pending = root / f".step_{step:012d}.{os.getpid()}"
        shutil.rmtree(pending, ignore_errors=True)
        pending.mkdir()
        before = model_file.stat()
        try:
            (pending / "model").mkdir()
            for source in required:
                dest = pending / source.relative_to(checkpoint)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
            after = model_file.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                return
            if read_json(status_path) != status:
                return
            (pending / "train_status.json").write_text(json.dumps(status, indent=2) + "\n")
            (pending / "policy.json").write_text(json.dumps({"global_time": step}, indent=2) + "\n")
            pending.rename(step_dir)
        finally:
            shutil.rmtree(pending, ignore_errors=True)
    if not step_dir.is_dir():
        return
    if current_step(root, "latest") != step:
        link_step(root, "latest", step_dir)
        print(f"[policy] latest={step}", flush=True)

    score = evaluation_score(run_dir / "humanoidverse_tracking_eval.csv", step, expected_rows)
    best = read_json(root / "best.json")
    if score is not None and (best is None or score < float(best["emd"])):
        link_step(root, "best", step_dir)
        metadata = {"global_time": step, "emd": score, "metric": "mean EMD over all evaluation motions"}
        pending = root / f".best.json.{os.getpid()}"
        pending.write_text(json.dumps(metadata, indent=2) + "\n")
        pending.replace(root / "best.json")
        print(f"[policy] best={step} emd={score:.6f}", flush=True)

    keep = {current_step(root, name) for name in ("latest", "best")}
    for old in root.glob("step_*"):
        if old.is_dir() and int(old.name.removeprefix("step_")) not in keep:
            shutil.rmtree(old)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-eval-rows", type=int, default=2719)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.expected_eval_rows < 1 or args.poll_seconds <= 0:
        parser.error("expected rows and poll seconds must be positive")
    while True:
        scan(args.run_dir, args.expected_eval_rows)
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
