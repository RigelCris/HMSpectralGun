"""Geometry selection and detection for MARCS atmospheres."""
import re
from pathlib import Path


def normalize_geometry(value):
    value = str(value).strip().lower()
    if value not in {"auto", "spherical", "plane-parallel"}:
        raise ValueError(f"Invalid geometry {value!r}: use auto, spherical or plane-parallel")
    return value


def resolve_geometry(value, logg):
    value = normalize_geometry(value)
    return ("plane-parallel" if float(logg) >= 4 else "spherical") if value == "auto" else value


def atmosphere_geometry(path, requested="auto"):
    with Path(path).open() as stream:
        header = stream.readline().strip().strip("'\"")
    if header.startswith("sphINTERPOL") or re.match(r"s\d", header):
        actual = "spherical"
    elif header.startswith("ppINTERPOL") or re.match(r"p\d", header):
        actual = "plane-parallel"
    else:
        raise ValueError(f"Cannot determine atmosphere geometry from {path}: {header}")
    requested = normalize_geometry(requested)
    if requested != "auto" and actual != requested:
        raise ValueError(f"Atmosphere {path} is {actual}, but geometry={requested} was requested")
    return actual
