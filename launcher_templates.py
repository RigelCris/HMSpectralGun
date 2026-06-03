#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helpers to bootstrap starter launcher files when input.ts is missing.
"""

from __future__ import annotations

import os
from pathlib import Path


def _with_trailing_sep(path: Path) -> str:
    return str(path.resolve()) + os.sep


def _write_if_missing(path: Path, content: str, created: list[Path]) -> None:
    if path.exists():
        return
    path.write_text(content, encoding="utf-8")
    created.append(path)


def create_launcher_templates(input_file: str) -> tuple[list[Path], list[Path]]:
    """
    Create a starter input.ts plus companion linelist.ts and abu.ts files.

    Existing files are preserved and never overwritten.
    """
    input_path = Path(input_file).expanduser().resolve()
    base_dir = input_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    output_dir = base_dir / "output_spectra"
    linelists_dir = base_dir / "linelists"
    models_dir = base_dir / "models"

    created_dirs: list[Path] = []
    for directory in (output_dir, linelists_dir, models_dir):
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory)

    input_template = (
        "# HMSpectralGun starter template\n"
        "# Fill the paths and the production row below.\n"
        f"{_with_trailing_sep(output_dir)}\n"
        f"{_with_trailing_sep(linelists_dir)}\n"
        f"{_with_trailing_sep(models_dir)}\n"
        "ExplicitModel=False\n"
        "interp=True\n"
        "NLTE=False\n"
        "# Model [Fe/H] [a/Fe] lam_i lam_f xi chemistry sampl RES resnum monoelem linelist_file abu_file snr extension\n"
        "3600,1.0  -1.00  0.20  15000  15500  2.0  st  *  28000  0.02  *  linelist.ts  abu.ts  *  txt\n"
    )
    linelist_template = (
        "# One line-list filename per row.\n"
        "# Names are resolved inside the linelist path declared in input.ts.\n"
        "vald_1.lin\n"
        "vald_2.lin\n"
    )
    abu_template = (
        "# AtomicNumber  Delta[X/Fe]\n"
        "8 0.20\n"
        "12 0.20\n"
        "26 0.00\n"
        "612613 15.0\n"
    )

    created_files: list[Path] = []
    _write_if_missing(input_path, input_template, created_files)
    _write_if_missing(base_dir / "linelist.ts", linelist_template, created_files)
    _write_if_missing(base_dir / "abu.ts", abu_template, created_files)

    return created_files, created_dirs
