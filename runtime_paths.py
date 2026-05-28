#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runtime path helpers for portable HMSpectralGun execution.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return None


def _to_abs_path(path_value: str) -> Path:
    return Path(path_value).expanduser().resolve()


def _to_abs_dir(path_value: str) -> str:
    return str(_to_abs_path(path_value)) + os.sep


def resolve_runtime_paths(project_root: str | Path | None = None) -> dict:
    """
    Resolve all runtime paths needed by main/main_parallel.

    Environment variables (highest priority):
      - HMSPECTRALGUN_DATASET_MODEL_PATH
      - HMSPECTRALGUN_LAUNCH_PATH
      - HMSPECTRALGUN_BABSMA
      - HMSPECTRALGUN_BSYN
      - HMSPECTRALGUN_CONTOPAC_PATH
      - HMSPECTRALGUN_INTERPOLATOR_EXE
      - HMSPECTRALGUN_EXEC_PATH (directory containing babsma_lu and bsyn_lu)

    Optional fallback env aliases:
      - TURBOSPECTRUM_EXEC_PATH
      - TURBOSPECTRUM_ROOT
    """
    root = Path(project_root).expanduser().resolve() if project_root else Path(__file__).resolve().parent
    default_dataset = root / "marcs_generator" / "dataset"
    default_interpolator = root / "marcs_generator" / "interpol_modeles"

    dataset_model_path = _to_abs_dir(_env("HMSPECTRALGUN_DATASET_MODEL_PATH") or str(default_dataset))
    launch_path = _to_abs_dir(_env("HMSPECTRALGUN_LAUNCH_PATH") or os.getcwd())

    exec_dir_raw = _env("HMSPECTRALGUN_EXEC_PATH", "TURBOSPECTRUM_EXEC_PATH")
    if exec_dir_raw is None:
        ts_root = _env("TURBOSPECTRUM_ROOT")
        if ts_root:
            exec_dir_raw = str(_to_abs_path(ts_root) / "exec-gf")
    exec_dir = _to_abs_path(exec_dir_raw) if exec_dir_raw else None

    babsma_exec = _env("HMSPECTRALGUN_BABSMA")
    if babsma_exec is None and exec_dir is not None:
        babsma_exec = str(exec_dir / "babsma_lu")
    babsma_exec = str(_to_abs_path(babsma_exec)) if babsma_exec else "babsma_lu"

    bsyn_exec = _env("HMSPECTRALGUN_BSYN")
    if bsyn_exec is None and exec_dir is not None:
        bsyn_exec = str(exec_dir / "bsyn_lu")
    bsyn_exec = str(_to_abs_path(bsyn_exec)) if bsyn_exec else "bsyn_lu"

    contopac_path = _to_abs_dir(_env("HMSPECTRALGUN_CONTOPAC_PATH") or str(_to_abs_path(launch_path) / "contopac"))
    interpolator_exe = str(
        _to_abs_path(_env("HMSPECTRALGUN_INTERPOLATOR_EXE") or str(default_interpolator))
    )

    return {
        "project_root": str(root),
        "dataset_model_path": dataset_model_path,
        "launch_path": launch_path,
        "babsma_exec": babsma_exec,
        "bsyn_exec": bsyn_exec,
        "contopac_path": contopac_path,
        "interpolator_exe": interpolator_exe,
    }


def validate_turbospectrum_paths(runtime: dict) -> None:
    """
    Validate key runtime paths and provide actionable errors.
    """
    missing = []
    for exe_key in ("babsma_exec", "bsyn_exec"):
        exe = Path(runtime[exe_key]).expanduser()
        if exe.is_absolute() and not exe.exists():
            missing.append(f"{exe_key}={exe}")
    if missing:
        details = "; ".join(missing)
        raise FileNotFoundError(
            "Missing Turbospectrum executable(s). "
            "Set HMSPECTRALGUN_EXEC_PATH or HMSPECTRALGUN_BABSMA/HMSPECTRALGUN_BSYN. "
            f"Details: {details}"
        )
