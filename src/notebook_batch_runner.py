"""Execute an existing notebook for several forecast targets in memory."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Iterable

import nbformat
from nbclient import NotebookClient


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def run_notebook_targets(
    notebook_path: str | Path,
    targets: Iterable[str],
    *,
    timeout: int | None = None,
) -> None:
    """Run ``notebook_path`` once per target without saving notebook copies."""
    path = Path(notebook_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    targets = list(targets)
    if not targets:
        return

    old_target = os.environ.get("FORECAST_TARGET")
    old_child = os.environ.get("FORECAST_BATCH_CHILD")

    try:
        os.environ["FORECAST_BATCH_CHILD"] = "1"
        for index, target in enumerate(targets, start=1):
            started = perf_counter()
            print(f"[Batch {index}/{len(targets)}] Bat dau target: {target}")
            os.environ["FORECAST_TARGET"] = target

            notebook = nbformat.read(path, as_version=4)
            kernel_name = notebook.metadata.get("kernelspec", {}).get("name", "python3")
            client = NotebookClient(
                notebook,
                timeout=timeout,
                kernel_name=kernel_name,
                resources={"metadata": {"path": str(path.parent)}},
                allow_errors=False,
                record_timing=False,
            )
            client.execute()
            elapsed = perf_counter() - started
            print(f"[Batch {index}/{len(targets)}] Hoan tat {target} sau {elapsed:.1f}s")
    finally:
        _restore_env("FORECAST_TARGET", old_target)
        _restore_env("FORECAST_BATCH_CHILD", old_child)
