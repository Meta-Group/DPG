#!/usr/bin/env python3
"""Profile hardware and recommend parallel settings for journal experiments."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _run_text(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def _memory_info() -> Dict[str, int | None]:
    info: Dict[str, int | None] = {"total_kib": None, "available_kib": None, "swap_total_kib": None, "swap_free_kib": None}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw_value = line.split(":", 1)
            value = int(raw_value.strip().split()[0])
            if key == "MemTotal":
                info["total_kib"] = value
            elif key == "MemAvailable":
                info["available_kib"] = value
            elif key == "SwapTotal":
                info["swap_total_kib"] = value
            elif key == "SwapFree":
                info["swap_free_kib"] = value
    except Exception:
        pass
    return info


def _threadpool_info() -> List[Dict[str, Any]]:
    try:
        import numpy as np
        import sklearn.ensemble  # noqa: F401
        import threadpoolctl

        a = np.random.default_rng(0).random((64, 64))
        _ = a @ a.T
        return threadpoolctl.threadpool_info()
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def _versions() -> Dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ["numpy", "pandas", "sklearn", "scipy"]:
        try:
            module = __import__(package)
            versions[package] = str(module.__version__)
        except Exception as exc:
            versions[package] = f"UNAVAILABLE: {type(exc).__name__}"
    return versions


def _recommendations(logical_cpus: int, available_kib: int | None, load1: float | None) -> Dict[str, Any]:
    available_gib = (available_kib or 0) / (1024 * 1024)
    busy = load1 is not None and load1 > logical_cpus * 0.35

    if logical_cpus >= 32:
        dpg_workers = 12 if busy else 16
        baseline_workers = 6 if busy else 8
    elif logical_cpus >= 16:
        dpg_workers = 6 if busy else 8
        baseline_workers = 3 if busy else 4
    else:
        dpg_workers = max(logical_cpus // 2, 1)
        baseline_workers = max(logical_cpus // 3, 1)

    if available_gib and available_gib < 24:
        dpg_workers = min(dpg_workers, 6)
        baseline_workers = min(baseline_workers, 3)

    return {
        "dpg_parallel_workers": int(max(dpg_workers, 1)),
        "baseline_parallel_workers": int(max(baseline_workers, 1)),
        "rf_n_jobs_per_worker": 1,
        "native_thread_env": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
        "reasoning": (
            "Use process-level parallelism across dataset/config jobs and keep RandomForest, OpenMP, "
            "and BLAS threads at 1 per worker to avoid oversubscription."
        ),
        "load_sensitive": bool(busy),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile hardware for journal experiment scheduling.")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("experiments_local_explanation/results_journal_v1/hardware"),
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    logical_cpus = int(affinity or os.cpu_count() or 1)
    load = os.getloadavg() if hasattr(os, "getloadavg") else (None, None, None)
    memory = _memory_info()
    disk = shutil.disk_usage(Path.cwd())

    profile = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "versions": _versions(),
        "cpu": {
            "os_cpu_count": os.cpu_count(),
            "affinity_cpu_count": affinity,
            "nproc": _run_text(["nproc"]),
            "nproc_all": _run_text(["nproc", "--all"]),
            "lscpu": _run_text(["lscpu"]),
        },
        "load_average": {"1m": load[0], "5m": load[1], "15m": load[2]},
        "memory": memory,
        "disk_cwd": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "threadpools": _threadpool_info(),
    }
    profile["recommendations"] = _recommendations(
        logical_cpus=logical_cpus,
        available_kib=memory.get("available_kib"),
        load1=load[0] if load[0] is not None else None,
    )

    json_path = out_dir / "hardware_profile.json"
    md_path = out_dir / "hardware_recommendation.md"
    json_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rec = profile["recommendations"]
    available_gib = (memory.get("available_kib") or 0) / (1024 * 1024)
    md_path.write_text(
        "\n".join(
            [
                "# Journal Hardware Recommendation",
                "",
                f"- Logical CPUs available to Python: {logical_cpus}",
                f"- Load average: {load[0]:.2f}, {load[1]:.2f}, {load[2]:.2f}",
                f"- Available memory: {available_gib:.1f} GiB",
                f"- Recommended DPG workers: {rec['dpg_parallel_workers']}",
                f"- Recommended baseline workers: {rec['baseline_parallel_workers']}",
                "- Recommended `--rf_n_jobs`: 1",
                "- Set native thread environment variables before launching multi-process runs:",
                "",
                "```bash",
                "export OMP_NUM_THREADS=1",
                "export OPENBLAS_NUM_THREADS=1",
                "export MKL_NUM_THREADS=1",
                "export NUMEXPR_NUM_THREADS=1",
                "export VECLIB_MAXIMUM_THREADS=1",
                "```",
                "",
                "Use fewer workers for SHAP/LIME/Anchors than for DPG-only runs because those baselines have higher per-process memory and CPU variability.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved hardware profile: {json_path}")
    print(f"Saved hardware recommendation: {md_path}")
    print(json.dumps(profile["recommendations"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
