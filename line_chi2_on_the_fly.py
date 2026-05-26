#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Line-by-line abundance analysis with on-the-fly HMSpectralGun synthesis.

Config JSON structure:
{
  "paths": {
    "obs_spectra_path": "...",
    "analysis_synth_path": "...",
    "results_path": "..."
  },
  "keywords": {
    "auto": true,
    "mcmc": false,
    "n_mcmc": 100,
    "synth_half_window": 3.0,
    "chi2_half_window": 0.35,
    "default_err": 0.01,
    "xfe_min": -0.8,
    "xfe_max": 0.8,
    "xfe_step": 0.1,
    "interp": "True",
    "nlte": "False",
    "chemistry": "st",
    "resnum": 0.02,
    "extension": "txt",
    "monoelem": "*",
    "dataset_model_path": ".../marcs_generator/dataset",
    "launch_path": ".../COM/",
    "linelist_library_path": ".../linelists/",
    "keep_synthetic": true,
    "cleanup_com_files": true,
    "optimize_shift": true,
    "optimize_norm": true
  },
  "stars": [
    {
      "name_obs_spec": "301_4MCMC.H",
      "teff": 3681,
      "logg": 1.25,
      "[Fe/H]": -1.0,
      "[a/Fe]": 0.2,
      "xi": 2.0,
      "RES": 28000,
      "abu_file": "abu.ts",
      "linelist_file": "linelistH.ts",
      "line_file": "lines_C13"
    }
  ]
}
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from mendeleev import element as chemical_element

try:
    from scipy.optimize import minimize, minimize_scalar
except ModuleNotFoundError:
    minimize = None
    minimize_scalar = None
try:
    from scipy.ndimage import gaussian_filter1d
except ModuleNotFoundError:
    gaussian_filter1d = None
try:
    from PyAstronomy import pyasl
except ModuleNotFoundError:
    pyasl = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ModuleNotFoundError:
    plt = None
    PdfPages = None

try:
    from HMSpectralGun.ModelMaker import ModelMaker, SkipModelError
    from HMSpectralGun.TurboSpecWriter import TurboSpecWriter
    from HMSpectralGun.HeaderCreator import HeaderCreator
    from HMSpectralGun.SpectrumConvolver import SpectrumConvolver
except ModuleNotFoundError:
    from ModelMaker import ModelMaker, SkipModelError
    from TurboSpecWriter import TurboSpecWriter
    from HeaderCreator import HeaderCreator
    from SpectrumConvolver import SpectrumConvolver


ALPHA_ELEMENTS = [8, 12, 14, 16, 20, 22]

SOLAR_ATOMIC_NUMBERS = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 90, 92
], dtype=int)

SOLAR_REFERENCES = np.array([
    0, 12.00, 10.93, 1.05, 1.38, 2.70, 8.56, 7.98, 8.77, 4.67, 8.15, 6.33, 7.58,
    6.48, 7.567, 5.48, 7.21, 5.29, 6.50, 5.12, 6.32, 3.09, 4.96, 4.01, 5.69,
    5.53, 7.51, 4.92, 6.25, 4.21, 4.60, 2.88, 3.58, 2.29, 3.33, 2.56, 3.25,
    2.60, 2.92, 2.21, 2.58, 1.42, 1.92, 1.84, 1.12, 1.66, 0.94, 1.77, 1.60,
    2.00, 1.00, 2.19, 1.51, 2.24, 1.07, 2.17, 1.13, 1.70, 0.58, 1.45, 1.00,
    0.52, 1.11, 0.28, 1.14, 0.51, 0.93, 0.00, 1.08, 0.06, 0.88, -0.17, 1.11,
    0.23, 1.25, 1.38, 1.64, 1.01, 1.13, 0.90, 2.00, 0.65, 0.06, -0.52
], dtype=float)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_dataset = script_dir / "marcs_generator" / "dataset"
    default_launch = script_dir.parents[2]

    parser = argparse.ArgumentParser(
        description="Line-by-line chi2 abundance analysis with on-the-fly HMSpectralGun synthesis."
    )
    parser.add_argument("--config", required=True, help="JSON config file")
    parser.add_argument("--auto", choices=["yes", "no"], help="Override keywords.auto")
    parser.add_argument("--star", help="Run only one star by exact name_obs_spec")
    parser.add_argument("--xfe-min", type=float, help="Override keywords.xfe_min")
    parser.add_argument("--xfe-max", type=float, help="Override keywords.xfe_max")
    parser.add_argument("--xfe-step", type=float, help="Override keywords.xfe_step")
    parser.add_argument("--n-mcmc", type=int, help="Override keywords.n_mcmc")
    parser.add_argument("--keep-synthetic", action="store_true", help="Override keywords.keep_synthetic=True")
    parser.add_argument("--cleanup-com-files", action="store_true", help="Override keywords.cleanup_com_files=True")
    parser.set_defaults(
        _default_dataset_model_path=str(default_dataset),
        _default_launch_path=str(default_launch) + os.sep,
    )
    return parser.parse_args()


def normalize_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def resolve_path(maybe_path, base_dirs):
    if maybe_path is None:
        return None
    path = Path(str(maybe_path)).expanduser()
    if path.is_absolute() and path.exists():
        return str(path)
    for base_dir in base_dirs:
        candidate = Path(base_dir) / path
        if candidate.exists():
            return str(candidate.resolve())
    return str(path.resolve()) if path.is_absolute() else str((Path(base_dirs[0]) / path).resolve())


def parse_optional_float(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null", "nan"}:
            return None
    return float(value)


def load_config(args):
    config_path = Path(args.config).expanduser().resolve()
    with open(config_path, "r") as handle:
        data = json.load(handle)

    paths = data.get("paths", {})
    keywords = data.get("keywords", {})
    stars = data.get("stars", [])
    if not isinstance(paths, dict) or not isinstance(keywords, dict) or not isinstance(stars, list):
        raise ValueError("Config must contain 'paths' dict, 'keywords' dict, and 'stars' list.")

    required_paths = ["obs_spectra_path", "analysis_synth_path", "results_path"]
    missing_paths = [key for key in required_paths if key not in paths]
    if missing_paths:
        raise ValueError("Missing required path keys: " + ", ".join(missing_paths))

    cfg_base_dirs = [str(config_path.parent), os.getcwd()]
    xfe_min_cfg = parse_optional_float(keywords.get("xfe_min", None))
    xfe_max_cfg = parse_optional_float(keywords.get("xfe_max", None))
    if args.xfe_min is not None:
        xfe_min_cfg = float(args.xfe_min)
    if args.xfe_max is not None:
        xfe_max_cfg = float(args.xfe_max)

    # Default automatic X/Fe span around 0 if limits are omitted/null.
    if xfe_min_cfg is None:
        xfe_min_cfg = -0.5
    if xfe_max_cfg is None:
        xfe_max_cfg = 0.5
    if xfe_min_cfg > xfe_max_cfg:
        raise ValueError(f"xfe_min ({xfe_min_cfg}) cannot be greater than xfe_max ({xfe_max_cfg}).")

    runtime = {
        "config_path": str(config_path),
        "config_dir": str(config_path.parent),
        "obs_spectra_path": resolve_path(paths["obs_spectra_path"], cfg_base_dirs),
        "analysis_synth_path": resolve_path(paths["analysis_synth_path"], cfg_base_dirs),
        "results_path": resolve_path(paths["results_path"], cfg_base_dirs),
        "dataset_model_path": resolve_path(
            keywords.get("dataset_model_path", args._default_dataset_model_path), cfg_base_dirs
        ),
        "launch_path": resolve_path(
            keywords.get("launch_path", args._default_launch_path), cfg_base_dirs
        ),
        "linelist_library_path": resolve_path(
            keywords.get("linelist_library_path", paths["obs_spectra_path"]), cfg_base_dirs
        ),
        "aux_file_path": resolve_path(
            keywords.get("aux_file_path", paths["obs_spectra_path"]), cfg_base_dirs
        ),
        "auto": normalize_bool(
            args.auto == "yes" if args.auto else keywords.get("auto", False), default=False
        ),
        "mcmc": normalize_bool(keywords.get("mcmc", False), default=False),
        "n_mcmc": int(args.n_mcmc if args.n_mcmc is not None else keywords.get("n_mcmc", 100)),
        "synth_half_window": float(keywords.get("synth_half_window", 3.0)),
        "chi2_half_window": float(keywords.get("chi2_half_window", 0.35)),
        "smooth_sigma": float(keywords.get("smooth_sigma", 0.0)),
        "default_err": float(keywords.get("default_err", 0.01)),
        "xfe_min": float(xfe_min_cfg),
        "xfe_max": float(xfe_max_cfg),
        "xfe_step": float(args.xfe_step if args.xfe_step is not None else keywords.get("xfe_step", 0.1)),
        "xfe_values": keywords.get("xfe_values"),
        "interp": str(keywords.get("interp", "True")),
        "nlte": str(keywords.get("nlte", "False")),
        "chemistry": str(keywords.get("chemistry", "st")),
        "resnum": float(keywords.get("resnum", 0.02)),
        "extension": str(keywords.get("extension", "txt")),
        "monoelem": str(keywords.get("monoelem", "*")),
        "sampl": str(keywords.get("sampl", "*")),
        "keep_synthetic": normalize_bool(
            True if args.keep_synthetic else keywords.get("keep_synthetic", True), default=True
        ),
        "cleanup_com_files": normalize_bool(
            True if args.cleanup_com_files else keywords.get("cleanup_com_files", True), default=True
        ),
        "optimize_shift": normalize_bool(keywords.get("optimize_shift", True), default=True),
        "optimize_norm": normalize_bool(keywords.get("optimize_norm", True), default=True),
        "optimize_resolution": normalize_bool(keywords.get("optimize_resolution", True), default=True),
    }
    runtime["synth_half_window"] = max(
        float(runtime["synth_half_window"]),
        8.0 * float(runtime["chi2_half_window"]),
    )

    selected_stars = stars
    if args.star:
        selected_stars = [row for row in stars if str(row.get("name_obs_spec", "")) == args.star]
        if not selected_stars:
            raise ValueError(f"No star entry found with name_obs_spec={args.star!r}")

    runtime["stars"] = selected_stars
    return runtime


def make_xfe_grid(runtime):
    if runtime["xfe_values"]:
        return np.array(sorted(set(float(val) for val in runtime["xfe_values"])), dtype=float)
    grid = np.arange(
        runtime["xfe_min"],
        runtime["xfe_max"] + 0.5 * runtime["xfe_step"],
        runtime["xfe_step"],
        dtype=float,
    )
    return np.round(grid, 6)


def parse_element_token(token):
    token = str(token).strip()
    if not token:
        raise ValueError("Empty element token")

    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", token):
        atomic_number = int(float(token))
        return atomic_number, chemical_element(atomic_number).symbol

    cleaned = token.replace("_", "").replace("-", "")
    match = re.match(r"^([A-Za-z]{1,2})(?:[IVX]+)?$", cleaned)
    if match is None:
        raise ValueError(f"Cannot parse element token {token!r}")

    symbol = match.group(1).capitalize()
    return chemical_element(symbol).atomic_number, symbol


def read_line_list(path):
    lines = []
    with open(path, "r") as handle:
        for raw in handle:
            row = raw.strip()
            if not row or row.startswith("#"):
                continue
            parts = row.split()
            if len(parts) < 2:
                continue
            try:
                l0 = float(parts[0])
                atomic_number, symbol = parse_element_token(parts[1])
            except Exception:
                continue
            ion = None
            ep = np.nan
            gflog = np.nan
            if len(parts) >= 3:
                try:
                    ion = int(float(parts[2]))
                except Exception:
                    ion = None
            if len(parts) >= 4:
                try:
                    ep = float(parts[3])
                except Exception:
                    ep = np.nan
            if len(parts) >= 5:
                try:
                    gflog = float(parts[4])
                except Exception:
                    gflog = np.nan
            lines.append({
                "l0": l0,
                "atomic_number": atomic_number,
                "symbol": symbol,
                "ion": ion,
                "ep": ep,
                "gflog": gflog,
                "raw": row,
            })
    if not lines:
        raise ValueError(f"No usable lines found in {path}")
    return lines


def read_observed_spectrum(path, default_err):
    data = np.genfromtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError("Observed spectrum must have at least 2 columns: wavelength flux")

    wavelength = data[:, 0].astype(float)
    flux = data[:, 1].astype(float)
    if data.shape[1] >= 3:
        error = np.abs(data[:, 2].astype(float))
    else:
        error = np.full_like(flux, float(default_err), dtype=float)
    if data.shape[1] >= 4:
        tell = data[:, 3].astype(float)
    else:
        tell = np.full_like(flux, np.nan, dtype=float)

    mask = np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
    if not np.any(mask):
        raise ValueError(f"No valid data points in observed spectrum: {path}")
    return wavelength[mask], flux[mask], error[mask], tell[mask]


def read_abu_file(path):
    data = np.genfromtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Abundance file must have 2 columns: {path}")

    atomic_number = data[:, 0].astype(float)
    abundance_diff = data[:, 1].astype(float)
    carbon_ratio = abundance_diff[atomic_number == 612613]
    keep = atomic_number != 612613
    return atomic_number[keep], abundance_diff[keep], carbon_ratio


def read_linelist_file(path):
    with open(path, "r") as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


def solar_reference(elem, abun, metallicity):
    elem = np.array(elem, dtype=int)
    abun = np.array(abun, dtype=float)
    if len(elem) != len(abun):
        raise ValueError("elem and abun must have the same length")

    output = []
    for atomic_number, delta in zip(elem, abun):
        match = np.where(SOLAR_ATOMIC_NUMBERS == atomic_number)[0]
        if len(match) == 0:
            continue
        output.append(delta + metallicity + SOLAR_REFERENCES[match[0]])
    return np.squeeze(np.array(output, dtype=float))


def infer_chemistry(star_row, runtime):
    if "chemistry" in star_row:
        return str(star_row["chemistry"])
    return str(runtime["chemistry"])


def get_star_float(star_row, *keys):
    for key in keys:
        if key in star_row:
            return float(star_row[key])
    raise KeyError(f"Missing required star key among: {', '.join(keys)}")


def get_star_str(star_row, *keys):
    for key in keys:
        if key in star_row:
            return str(star_row[key])
    raise KeyError(f"Missing required star key among: {', '.join(keys)}")


def build_star_context(star_row, runtime):
    name_obs_spec = get_star_str(star_row, "name_obs_spec", "obs_spec")
    chemistry = infer_chemistry(star_row, runtime)
    context = {
        "name_obs_spec": name_obs_spec,
        "teff": get_star_float(star_row, "teff", "Teff"),
        "logg": get_star_float(star_row, "logg"),
        "feh": get_star_float(star_row, "[Fe/H]", "feh", "metallicity"),
        "alpha": get_star_float(star_row, "[a/Fe]", "afe", "alpha"),
        "xi": get_star_float(star_row, "xi"),
        "res": get_star_float(star_row, "RES", "res"),
        "abu_file": get_star_str(star_row, "abu_file", "abu.ts"),
        "linelist_file": get_star_str(star_row, "linelist_file", "linelist.ts"),
        "line_file": get_star_str(star_row, "line_file", "line.ts"),
        "chemistry": chemistry,
        "monoelem": str(star_row.get("monoelem", runtime["monoelem"])),
        "resnum": float(star_row.get("resnum", runtime["resnum"])),
        "extension": str(star_row.get("extension", runtime["extension"])),
        "sampl": str(star_row.get("sampl", runtime["sampl"])),
        "star_slug": Path(name_obs_spec).stem,
    }
    base_dirs = [
        runtime["config_dir"],
        runtime["linelist_library_path"],
        runtime["aux_file_path"],
        runtime["obs_spectra_path"],
        os.getcwd(),
    ]
    context["obs_path"] = resolve_path(name_obs_spec, [runtime["obs_spectra_path"], runtime["config_dir"], os.getcwd()])
    context["abu_path"] = resolve_path(context["abu_file"], base_dirs)
    context["linelist_path"] = resolve_path(context["linelist_file"], base_dirs)
    context["line_path"] = resolve_path(context["line_file"], base_dirs)
    return context


def prepare_model(star_ctx, runtime, model_maker, model_dir):
    interp = str(runtime["interp"])
    teff = int(round(star_ctx["teff"]))
    logg = float(star_ctx["logg"])
    feh = float(star_ctx["feh"])
    xi = float(star_ctx["xi"])
    chemistry = str(star_ctx["chemistry"])

    if interp == "True":
        models = model_maker.select_models_for_interpolation(teff, logg, feh, xi, chemistry)
        return model_maker.write_interpolator(teff, logg, feh, xi, chemistry, model_dir, models)

    if interp.lower() == "nearest":
        return model_maker.select_nearest_model(teff, logg, feh, xi, chemistry, model_dir)

    raise ValueError(f"Unsupported interp keyword: {interp}")


def apply_abundance_pattern(star_ctx, target_atomic_number, target_xfe):
    elem, delta, carbon_ratio = read_abu_file(star_ctx["abu_path"])

    if star_ctx["alpha"] != 0.0:
        for atomic_number in ALPHA_ELEMENTS:
            delta[elem == float(atomic_number)] = float(star_ctx["alpha"])

    mask = elem == float(target_atomic_number)
    if np.any(mask):
        delta[mask] = float(target_xfe)
    else:
        elem = np.append(elem, float(target_atomic_number))
        delta = np.append(delta, float(target_xfe))

    isotopic_number = 0 if len(carbon_ratio) == 0 else 612613
    isotopic_value = 0.0 if len(carbon_ratio) == 0 else float(carbon_ratio[0])
    absolute_abundance = solar_reference(elem, delta, star_ctx["feh"])
    return elem, delta, absolute_abundance, isotopic_number, isotopic_value


def format_xfe_tag(symbol, xfe):
    sign = "p" if xfe >= 0 else "m"
    amp = int(round(abs(float(xfe)) * 100))
    return f"_{symbol}{sign}{amp:03d}"


def _format_window_token(value):
    return f"{float(value):.3f}".replace("-", "m").replace(".", "p")


def _find_existing_synthetic(save_path, extension, abundance_tag, lam_i, lam_f):
    """
    Find reusable synthetic files.
    Order:
    1) New precise window token format.
    2) Legacy integer window token format.
    3) Legacy files with extra suffixes after abundance tag (e.g. old _Lxxxx tags).
    """
    save_dir = Path(save_path)
    lam_i_prec = _format_window_token(lam_i)
    lam_f_prec = _format_window_token(lam_f)
    lam_i_int = int(float(lam_i))
    lam_f_int = int(float(lam_f))

    patterns = [
        f"*_{lam_i_prec}_{lam_f_prec}_*{abundance_tag}.{extension}",
        f"*_{lam_i_int}_{lam_f_int}_*{abundance_tag}.{extension}",
        f"*_{lam_i_int}_{lam_f_int}_*{abundance_tag}*.{extension}",
    ]
    for pattern in patterns:
        matches = sorted(save_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _build_synth_cache_key(runtime, star_ctx, model_name, line_info, trial_xfe, lam_i, lam_f):
    return "|".join([
        str(model_name),
        f"teff={float(star_ctx['teff']):.1f}",
        f"logg={float(star_ctx['logg']):.2f}",
        f"feh={float(star_ctx['feh']):+.3f}",
        f"alpha={float(star_ctx['alpha']):+.3f}",
        f"xi={float(star_ctx['xi']):.2f}",
        f"resnum={float(star_ctx['resnum']):.4f}",
        f"interp={runtime['interp']}",
        f"nlte={runtime['nlte']}",
        f"chem={runtime['chemistry']}",
        f"linelist={Path(star_ctx['linelist_path']).name}",
        f"line={float(line_info['l0']):.3f}",
        f"anum={int(line_info['atomic_number'])}",
        f"xfe={float(trial_xfe):+.3f}",
        f"lam_i={float(lam_i):.3f}",
        f"lam_f={float(lam_f):.3f}",
        f"ext={star_ctx['extension']}",
    ])


def _compact_grid_row(row):
    return {
        "line": row["line"],
        "element": row["element"],
        "atomic_number": row["atomic_number"],
        "ion": row.get("ion"),
        "ep": row.get("ep"),
        "gflog": row.get("gflog"),
        "trial_xfe": row["trial_xfe"],
        "chi2": row["chi2"],
        "n_pts": row["n_pts"],
        "norm_factor": row["norm_factor"],
        "wvl_shift": row["wvl_shift"],
        "res_value": row["res_value"],
        "chi2_half_window": row.get("chi2_half_window", np.nan),
        "status": row["status"],
        "spectrum_file": row["spectrum_file"],
        "message": row["message"],
    }


def _checkpoint_save(path, current_idx, accepted_rows_by_idx, grid_rows_compact):
    payload = {
        "current_idx": int(current_idx),
        "accepted_rows_by_idx": {str(int(k)): v for k, v in accepted_rows_by_idx.items()},
        "grid_rows_compact": list(grid_rows_compact),
    }
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2)
    tmp.replace(path)


def _checkpoint_load(path):
    if not Path(path).exists():
        return 0, {}, []
    with open(path, "r") as handle:
        payload = json.load(handle)
    current_idx = int(payload.get("current_idx", 0))
    accepted_raw = payload.get("accepted_rows_by_idx", {})
    accepted_rows_by_idx = {int(k): v for k, v in accepted_raw.items()}
    grid_rows_compact = payload.get("grid_rows_compact", [])
    return current_idx, accepted_rows_by_idx, grid_rows_compact


def synthesize_trial(
    runtime,
    star_ctx,
    model_name,
    turbo_writer,
    header_creator,
    convolver,
    line_info,
    trial_xfe,
    synth_cache_index=None,
):
    lam_i = float(line_info["l0"]) - float(runtime["synth_half_window"])
    lam_f = float(line_info["l0"]) + float(runtime["synth_half_window"])

    linespec = read_linelist_file(star_ctx["linelist_path"])
    elem, delta, abundance, isotopic_number, isotopic_value = apply_abundance_pattern(
        star_ctx, line_info["atomic_number"], trial_xfe
    )

    abundance_tag = f"{format_xfe_tag(line_info['symbol'], trial_xfe)}"
    cache_key = _build_synth_cache_key(runtime, star_ctx, model_name, line_info, trial_xfe, lam_i, lam_f)
    if synth_cache_index is not None:
        cached_name = synth_cache_index.get(cache_key)
        if cached_name:
            cached_path = Path(turbo_writer.save_path) / str(cached_name)
            if cached_path.exists():
                return cached_path.name, False

    existing_match = _find_existing_synthetic(
        turbo_writer.save_path,
        star_ctx["extension"],
        abundance_tag,
        lam_i,
        lam_f,
    )
    if existing_match is not None:
        if synth_cache_index is not None:
            synth_cache_index[cache_key] = existing_match.name
        return existing_match.name, False

    spectrum_name = turbo_writer.writer(
        model_name,
        star_ctx["feh"],
        star_ctx["alpha"],
        lam_i,
        lam_f,
        star_ctx["xi"],
        linespec,
        "*",
        elem,
        abundance,
        isotopic_number,
        isotopic_value,
        keyw="No",
        el="*",
        ext=star_ctx["extension"],
        deltalam=star_ctx["resnum"],
        interp=runtime["interp"],
        NLTE=runtime["nlte"],
        abundance_tag=abundance_tag,
    )

    if spectrum_name == "STOP":
        existing_match = _find_existing_synthetic(
            turbo_writer.save_path,
            star_ctx["extension"],
            abundance_tag,
            lam_i,
            lam_f,
        )
        if existing_match is not None:
            if synth_cache_index is not None:
                synth_cache_index[cache_key] = existing_match.name
            return existing_match.name, False
        raise RuntimeError(f"Spectrum marked as existing, but no file matched tag {abundance_tag}")

    com_path = os.path.join(runtime["launch_path"], f"{spectrum_name}.com")
    log_filename = f"log_fit_{Path(spectrum_name).stem}_{os.getpid()}.txt"
    log_path = os.path.join(runtime["launch_path"], log_filename)
    with open(log_path, "w") as log_handle:
        subprocess.run([com_path], stdout=log_handle, stderr=subprocess.STDOUT, check=False)

    header_creator.create_combined_header(
        spectrum_name,
        model=model_name,
        met=star_ctx["feh"],
        alpha=star_ctx["alpha"],
        elem=elem,
        deltaabu=delta,
        lam_min=lam_i,
        lam_max=lam_f,
        xi=star_ctx["xi"],
        isotopic_n=isotopic_number,
        isotopic_val=isotopic_value,
        keyw=star_ctx["chemistry"],
        deltalam=star_ctx["resnum"],
        log_filename=log_filename,
    )

    # Keep raw synthetic profile here; resolution is optimized/applied during chi2 fitting.
    final_name = spectrum_name

    if runtime["cleanup_com_files"]:
        safe_remove(os.path.join(runtime["launch_path"], f"{spectrum_name}.com"))
        safe_remove(log_path)

    if synth_cache_index is not None:
        synth_cache_index[cache_key] = final_name
    return final_name, True


def read_synthetic_normflux(path):
    data = np.genfromtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Synthetic file has fewer than 2 columns: {path}")
    return data[:, 0].astype(float), data[:, 1].astype(float)


def apply_resolution_to_model(wavelength, model_flux, res_value):
    if (pyasl is None) or (res_value is None):
        return model_flux
    try:
        res_float = float(res_value)
    except Exception:
        return model_flux
    if res_float <= 0:
        return model_flux
    try:
        broadened, _ = pyasl.instrBroadGaussFast(
            np.asarray(wavelength, dtype=float),
            np.asarray(model_flux, dtype=float),
            res_float,
            edgeHandling=None,
            fullout=True,
        )
        return broadened
    except Exception:
        return model_flux


def compute_chi2_for_params(
    obs_w, obs_f, obs_e, syn_w, syn_f, l0, chi2_half_window, norm_factor, shift, res_value, smooth_sigma=0.0
):
    shifted_w = obs_w + shift
    shifted_f = obs_f * norm_factor
    shifted_e = obs_e * abs(norm_factor)
    if gaussian_filter1d is not None and float(smooth_sigma) > 0.0:
        shifted_f = gaussian_filter1d(shifted_f, sigma=float(smooth_sigma))
        shifted_e = gaussian_filter1d(shifted_e, sigma=float(smooth_sigma))

    syn_broad = apply_resolution_to_model(syn_w, syn_f, res_value)
    syn_on_obs = np.interp(shifted_w, syn_w, syn_broad, left=np.nan, right=np.nan)
    mask = (
        (shifted_w >= (l0 - chi2_half_window))
        & (shifted_w <= (l0 + chi2_half_window))
        & np.isfinite(shifted_f)
        & np.isfinite(shifted_e)
        & np.isfinite(syn_on_obs)
        & (shifted_e > 0)
    )
    n_pts = int(np.sum(mask))
    if n_pts == 0:
        return np.inf, 0

    chi2 = float(np.sum(((shifted_f[mask] - syn_on_obs[mask]) / shifted_e[mask]) ** 2))
    return chi2, n_pts


def fit_trial_alignment(runtime, obs_w, obs_f, obs_e, syn_w, syn_f, l0):
    if minimize is None or (not runtime["optimize_shift"] and not runtime["optimize_norm"] and not runtime["optimize_resolution"]):
        chi2, n_pts = compute_chi2_for_params(
            obs_w, obs_f, obs_e, syn_w, syn_f, l0, runtime["chi2_half_window"], 1.0, 0.0,
            runtime.get("res_init"), runtime.get("smooth_sigma", 0.0)
        )
        return {
            "chi2": chi2,
            "n_pts": n_pts,
            "norm_factor": 1.0,
            "wvl_shift": 0.0,
            "res_value": float(runtime.get("res_init", 0.0)),
        }

    def objective(values):
        norm_factor = values[0] if runtime["optimize_norm"] else 1.0
        index = 1 if runtime["optimize_norm"] else 0
        shift = values[index] if runtime["optimize_shift"] else 0.0
        if runtime["optimize_shift"]:
            index += 1
        res_value = values[index] if runtime["optimize_resolution"] else runtime.get("res_init")
        chi2, _ = compute_chi2_for_params(
            obs_w, obs_f, obs_e, syn_w, syn_f, l0, runtime["chi2_half_window"], norm_factor, shift,
            res_value, runtime.get("smooth_sigma", 0.0)
        )
        if not np.isfinite(chi2):
            return 1.0e30
        return chi2

    x0 = []
    bounds = []
    if runtime["optimize_norm"]:
        x0.append(1.0)
        bounds.append((0.85, 1.15))
    if runtime["optimize_shift"]:
        x0.append(0.0)
        bounds.append((-0.20, 0.20))
    if runtime["optimize_resolution"]:
        r0 = float(runtime.get("res_init", 28000.0))
        rmin = float(runtime.get("res_min", max(1000.0, 0.6 * r0)))
        rmax = float(runtime.get("res_max", 1.4 * r0))
        x0.append(r0)
        bounds.append((rmin, rmax))

    result = minimize(objective, x0=np.array(x0, dtype=float), method="L-BFGS-B", bounds=bounds)
    best = np.array(result.x, dtype=float)
    index = 0
    norm_factor = 1.0
    shift = 0.0
    res_value = float(runtime.get("res_init", 0.0))
    if runtime["optimize_norm"]:
        norm_factor = float(best[index])
        index += 1
    if runtime["optimize_shift"]:
        shift = float(best[index])
        index += 1
    if runtime["optimize_resolution"]:
        res_value = float(best[index])

    chi2, n_pts = compute_chi2_for_params(
        obs_w, obs_f, obs_e, syn_w, syn_f, l0, runtime["chi2_half_window"], norm_factor, shift,
        res_value, runtime.get("smooth_sigma", 0.0)
    )
    return {
        "chi2": chi2,
        "n_pts": n_pts,
        "norm_factor": norm_factor,
        "wvl_shift": shift,
        "res_value": res_value,
    }


def estimate_mcmc_error(runtime, line_rows, obs_w, obs_f, obs_e, line_center):
    if not runtime["mcmc"] or runtime["n_mcmc"] <= 0:
        return {"best_xfe_err": np.nan, "n_mcmc_ok": 0}

    valid = [row for row in line_rows if row["status"] == "ok" and row["syn_w"] is not None]
    if len(valid) == 0:
        return {"best_xfe_err": np.nan, "n_mcmc_ok": 0}

    rng = np.random.default_rng()
    winners = []
    for _ in range(runtime["n_mcmc"]):
        perturbed_flux = obs_f + rng.normal(0.0, obs_e)
        best = None
        for row in valid:
            chi2, _ = compute_chi2_for_params(
                obs_w, perturbed_flux, obs_e,
                row["syn_w"], row["syn_f"],
                line_center,
                runtime["chi2_half_window"],
                row["norm_factor"],
                row["wvl_shift"],
                row.get("res_value", runtime.get("res_init")),
                row.get("smooth_sigma", runtime.get("smooth_sigma", 0.0)),
            )
            if np.isfinite(chi2) and (best is None or chi2 < best[0]):
                best = (chi2, row["trial_xfe"])
        if best is not None:
            winners.append(best[1])

    if len(winners) < 3:
        return {"best_xfe_err": np.nan, "n_mcmc_ok": len(winners)}

    return {
        "best_xfe_err": float(np.std(winners)),
        "n_mcmc_ok": len(winners),
    }


def estimate_local_xfe_error(line_rows, best_row, obs_w, obs_f, obs_e, line_center):
    """
    Estimate local [X/Fe] uncertainty from chi2 curvature around best solution:
    chi2(x) ~= chi2_min + ((x-x0)^2 / sigma^2), so sigma = sqrt(1/a) for parabola a*x^2+b*x+c.
    """
    valid = [row for row in line_rows if row.get("status") == "ok" and row.get("syn_w") is not None]
    if len(valid) < 3:
        return np.nan

    xfe_best = float(best_row["trial_xfe"])
    norm = float(best_row["norm_factor"])
    shift = float(best_row["wvl_shift"])
    resv = float(best_row.get("res_value", 28000.0))
    sm = float(best_row.get("smooth_sigma", 0.0))
    win = float(best_row.get("chi2_half_window", 0.35))

    points = []
    for row in valid:
        x = float(row["trial_xfe"])
        chi2, _ = compute_chi2_for_params(
            obs_w, obs_f, obs_e,
            row["syn_w"], row["syn_f"],
            line_center, win, norm, shift, resv, sm
        )
        if np.isfinite(chi2):
            points.append((x, float(chi2)))

    # Also include exact best point if it's interpolated and not in trial grid.
    if best_row.get("syn_w") is not None and best_row.get("syn_f") is not None:
        chi2_best, _ = compute_chi2_for_params(
            obs_w, obs_f, obs_e,
            best_row["syn_w"], best_row["syn_f"],
            line_center, win, norm, shift, resv, sm
        )
        if np.isfinite(chi2_best):
            points.append((xfe_best, float(chi2_best)))

    if len(points) < 3:
        return np.nan

    # Merge duplicates in x by keeping minimal chi2.
    merged = {}
    for x, c in points:
        if (x not in merged) or (c < merged[x]):
            merged[x] = c
    xs = np.array(sorted(merged.keys()), dtype=float)
    cs = np.array([merged[x] for x in xs], dtype=float)
    if xs.size < 3:
        return np.nan

    order = np.argsort(np.abs(xs - xfe_best))
    nfit = min(7, xs.size)
    sel = np.sort(order[:nfit])
    xfit = xs[sel]
    cfit = cs[sel]
    if np.unique(xfit).size < 3:
        return np.nan

    try:
        a, b, c0 = np.polyfit(xfit, cfit, 2)
    except Exception:
        return np.nan
    if not np.isfinite(a) or a <= 0:
        return np.nan
    sigma = float(np.sqrt(1.0 / a))
    if (not np.isfinite(sigma)) or sigma <= 0:
        return np.nan
    return sigma


def interpolate_synthetic_profile(line_rows, target_xfe):
    valid = [row for row in line_rows if row.get("status") == "ok" and row.get("syn_w") is not None and row.get("syn_f") is not None]
    if not valid:
        return None, None
    valid = sorted(valid, key=lambda row: float(row["trial_xfe"]))
    xs = np.array([float(row["trial_xfe"]) for row in valid], dtype=float)
    w_ref = np.asarray(valid[0]["syn_w"], dtype=float)
    if w_ref.size == 0:
        return None, None
    grid = []
    for row in valid:
        w = np.asarray(row["syn_w"], dtype=float)
        f = np.asarray(row["syn_f"], dtype=float)
        if w.shape != w_ref.shape or not np.allclose(w, w_ref):
            f = np.interp(w_ref, w, f, left=np.nan, right=np.nan)
        grid.append(f)
    grid = np.asarray(grid, dtype=float)

    x = float(target_xfe)
    if xs.size == 1:
        return w_ref, np.asarray(grid[0], dtype=float)
    if x <= xs[0]:
        return w_ref, np.asarray(grid[0], dtype=float)
    if x >= xs[-1]:
        return w_ref, np.asarray(grid[-1], dtype=float)
    i_hi = int(np.searchsorted(xs, x, side="right"))
    i_hi = min(max(i_hi, 1), xs.size - 1)
    i_lo = i_hi - 1
    x_lo = float(xs[i_lo])
    x_hi = float(xs[i_hi])
    if abs(x_hi - x_lo) < 1e-12:
        return w_ref, np.asarray(grid[i_lo], dtype=float)
    t = (x - x_lo) / (x_hi - x_lo)
    f = (1.0 - t) * grid[i_lo] + t * grid[i_hi]
    return w_ref, np.asarray(f, dtype=float)


def compute_equivalent_width_mA(syn_w, syn_f, line_center, half_window, res_value):
    if syn_w is None or syn_f is None:
        return np.nan
    w = np.asarray(syn_w, dtype=float)
    f = np.asarray(syn_f, dtype=float)
    if w.size == 0 or f.size == 0 or w.size != f.size:
        return np.nan
    fb = apply_resolution_to_model(w, f, res_value)
    mask = (w >= (float(line_center) - float(half_window))) & (w <= (float(line_center) + float(half_window)))
    if int(np.sum(mask)) < 3:
        return np.nan
    ew = np.trapz(1.0 - fb[mask], w[mask]) * 1.0e3
    return float(ew)


def infer_star_label(star_ctx):
    m = re.match(r"^(\d+)", str(star_ctx.get("star_slug", "")))
    if m:
        return m.group(1)
    return str(star_ctx.get("star_slug", "star"))


def plot_line_panel(fig, star_ctx, line_info, obs_w, obs_f, obs_e, obs_tell, line_rows, selected_xfe, xfe_err=np.nan, chi2_half_window=None):
    valid = [row for row in line_rows if row.get("status") == "ok" and row.get("syn_w") is not None and row.get("syn_f") is not None]
    if not valid:
        return
    best_row = min(valid, key=lambda row: abs(float(row["trial_xfe"]) - float(selected_xfe)))
    l0 = float(line_info["l0"])
    win_half = float(best_row.get("chi2_half_window", chi2_half_window if chi2_half_window is not None else 0.35))
    xfe = float(selected_xfe)
    resv = float(best_row.get("res_value", 28000.0))
    norm = float(best_row.get("norm_factor", 1.0))
    shift = float(best_row.get("wvl_shift", 0.0))
    sm = float(best_row.get("smooth_sigma", 0.0))
    chi2 = float(best_row.get("chi2", np.nan))
    npts = int(best_row.get("n_pts", 0))

    syn_w = np.asarray(best_row["syn_w"], dtype=float)
    syn_f = np.asarray(best_row["syn_f"], dtype=float)
    model_plot = apply_resolution_to_model(syn_w, syn_f, resv)

    obs_w = np.asarray(obs_w, dtype=float)
    obs_f = np.asarray(obs_f, dtype=float)
    obs_e = np.asarray(obs_e, dtype=float)
    obs_tell = np.asarray(obs_tell, dtype=float) if obs_tell is not None else np.full_like(obs_f, np.nan)

    obs_w_shift = obs_w + shift
    obs_f_shift = obs_f * norm
    obs_e_shift = obs_e * abs(norm)
    if gaussian_filter1d is not None and sm > 0.0:
        obs_f_shift = gaussian_filter1d(obs_f_shift, sigma=sm)
        obs_e_shift = gaussian_filter1d(obs_e_shift, sigma=sm)

    xlim_main = (l0 - 5.0, l0 + 5.0)
    mask_main = (obs_w_shift >= xlim_main[0]) & (obs_w_shift <= xlim_main[1])
    if int(np.sum(mask_main)) < 5:
        mask_main = np.full(obs_w_shift.shape, True, dtype=bool)

    fig.clf()
    fig.patch.set_facecolor("#f2f2f2")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.1], hspace=0.03)
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_ratio = fig.add_subplot(gs[1, 0], sharex=ax_spec)
    ax_spec.set_facecolor("#f2f2f2")
    ax_ratio.set_facecolor("#f2f2f2")

    ax_tell = ax_spec.twinx()
    if np.any(np.isfinite(obs_tell[mask_main])):
        ax_tell.plot(obs_w_shift[mask_main], obs_tell[mask_main], color="#6fbf73", lw=1.0, alpha=0.35)
    ax_tell.set_ylim(0.0, 1.1)
    ax_tell.set_ylabel("Telluric", color="#58a85f", fontsize=8)
    ax_tell.tick_params(axis="y", labelsize=7, colors="#58a85f")

    ax_spec.plot(obs_w_shift[mask_main], obs_f[mask_main], color="grey", lw=0.8, alpha=0.35, label="Obs (raw)")
    ax_spec.plot(obs_w_shift[mask_main], obs_f_shift[mask_main], color="black", lw=1.0, alpha=0.95, label="Obs (shift+norm)")
    ax_spec.fill_between(
        obs_w_shift[mask_main],
        (obs_f_shift - obs_e_shift)[mask_main],
        (obs_f_shift + obs_e_shift)[mask_main],
        color="grey",
        alpha=0.15,
    )
    ax_spec.plot(syn_w, model_plot, color="#ff2d2d", lw=1.5, label="Model")
    ax_spec.axhline(1.0, color="#222", ls="--", lw=0.8)
    ax_spec.axvline(l0, color="#444", lw=0.8, ls="--")
    ax_spec.axvspan(l0 - win_half, l0 + win_half, color="#f1dc8a", alpha=0.35, zorder=0)
    ax_spec.set_xlim(*xlim_main)
    ax_spec.set_ylim(0.5, 1.12)
    ax_spec.set_ylabel("Normalized flux")
    ax_spec.legend(loc="lower left", fontsize=7, framealpha=0.9)
    ax_spec.tick_params(labelbottom=False)

    obs_loc_w = obs_w_shift[mask_main]
    obs_loc_f = obs_f_shift[mask_main]
    with np.errstate(divide="ignore", invalid="ignore"):
        model_on_obs = np.interp(obs_loc_w, syn_w, model_plot, left=np.nan, right=np.nan)
        ratio = np.where((obs_loc_f > 0) & np.isfinite(model_on_obs), model_on_obs / obs_loc_f, np.nan)
    ax_ratio.plot(obs_loc_w, ratio, color="#ff2d2d", lw=1.1)
    ax_ratio.axhline(1.0, color="#222", ls="--", lw=0.8)
    ax_ratio.axvspan(l0 - win_half, l0 + win_half, color="#f1dc8a", alpha=0.35, zorder=0)
    ax_ratio.set_ylabel("Mod / Obs")
    ax_ratio.set_xlabel("λ [Å]")
    ax_ratio.set_ylim(0.8, 1.2)
    ax_ratio.set_xlim(*xlim_main)

    ax_inset = ax_spec.inset_axes([0.58, 0.10, 0.32, 0.42])
    ax_inset.set_facecolor("white")
    zoom_half = max(1.4, 2.2 * win_half)
    zoom_mask = (obs_w_shift >= (l0 - zoom_half)) & (obs_w_shift <= (l0 + zoom_half))
    ax_inset.plot(obs_w_shift[zoom_mask], obs_f_shift[zoom_mask], color="black", lw=1.0)
    ax_inset.plot(syn_w, model_plot, color="#ff2d2d", lw=1.4)
    ax_inset.axhline(1.0, color="#222", ls="--", lw=0.8)
    ax_inset.axvline(l0, color="#444", lw=0.8, ls="--")
    ax_inset.axvspan(l0 - win_half, l0 + win_half, color="#f1dc8a", alpha=0.35, zorder=0)
    ax_inset.set_xlim(l0 - zoom_half, l0 + zoom_half)
    obs_at_l0 = np.interp(l0, obs_w_shift, obs_f_shift, left=np.nan, right=np.nan)
    mod_at_l0 = np.interp(l0, syn_w, model_plot, left=np.nan, right=np.nan)
    y0 = np.nanmin(np.array([obs_at_l0, mod_at_l0], dtype=float))
    if not np.isfinite(y0):
        y0 = 0.75
    ax_inset.set_ylim(max(0.01, y0 - 0.15), 1.09)
    ax_inset.tick_params(labelsize=7)

    from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
    for _ax in (ax_spec, ax_ratio, ax_inset):
        _ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        _ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        _ax.minorticks_on()
        _ax.tick_params(axis="x", which="major", length=5)
        _ax.tick_params(axis="x", which="minor", length=3)

    star_label = infer_star_label(star_ctx)
    abs_met = float(star_ctx["feh"]) + xfe
    xfe_err_txt = f"{float(xfe_err):.3f}" if np.isfinite(xfe_err) else "nan"
    head = (
        f"Star {star_label}    λ₀={l0:.3f} Å   {line_info['symbol']}   "
        f"[M/H]={abs_met:+.3f}±{xfe_err_txt}   R={resv:.0f}   "
        f"norm={norm:.4f}   Δλ={shift:+.4f}   χ²={chi2:.2f} (n={npts})   ξ≈{float(star_ctx['xi']):.2f}"
    )
    fig.suptitle(head, x=0.03, y=0.99, ha="left", va="top", fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.10, hspace=0.03)


def choose_manual_xfe(line_info, line_rows, best_xfe, chi2_half_window=0.35, smooth_sigma_default=0.0):
    """
    Interactive dashboard (one line at a time), inspired by run_star_interactive.py.
    Keys:
      - left/right (or a/d): cycle [X/Fe]
      - enter/space/y: accept current [X/Fe]
      - q/escape: quit panel
    """
    fallback = {
        "result": float(best_xfe),
        "norm_factor": 1.0,
        "wvl_shift": 0.0,
        "res_value": 28000.0,
        "smooth_sigma": float(np.clip(smooth_sigma_default, 0.0, 2.0)),
        "chi2_half_window": float(chi2_half_window),
    }
    valid = [row for row in line_rows if row["status"] == "ok" and row["syn_w"] is not None]
    if not valid:
        return fallback
    if plt is None:
        fallback.update({
            "norm_factor": float(valid[0].get("norm_factor", 1.0)),
            "wvl_shift": float(valid[0].get("wvl_shift", 0.0)),
            "res_value": float(valid[0].get("res_value", 28000.0)),
        })
        return fallback

    sorted_rows = sorted(valid, key=lambda row: row["trial_xfe"])
    trial_values = np.array([row["trial_xfe"] for row in sorted_rows], dtype=float)
    idx0 = int(np.argmin(np.abs(trial_values - float(best_xfe))))
    best_seed = sorted_rows[idx0]
    fallback.update({
        "result": float(trial_values[idx0]),
        "norm_factor": float(best_seed.get("norm_factor", 1.0)),
        "wvl_shift": float(best_seed.get("wvl_shift", 0.0)),
        "res_value": float(best_seed.get("res_value", 28000.0)),
    })

    obs_w = np.asarray(sorted_rows[0]["obs_w"], dtype=float)
    obs_f = np.asarray(sorted_rows[0]["obs_f"], dtype=float)
    obs_e = np.asarray(sorted_rows[0]["obs_e"], dtype=float)
    if obs_w.size == 0:
        return fallback

    syn_w_base = np.asarray(sorted_rows[0]["syn_w"], dtype=float)
    syn_grid = []
    for row in sorted_rows:
        syn_w = np.asarray(row["syn_w"], dtype=float)
        syn_f = np.asarray(row["syn_f"], dtype=float)
        if syn_w.shape != syn_w_base.shape or not np.allclose(syn_w, syn_w_base):
            syn_f = np.interp(syn_w_base, syn_w, syn_f, left=np.nan, right=np.nan)
        syn_grid.append(syn_f)
    syn_grid = np.array(syn_grid, dtype=float)

    xfe_min = float(np.min(trial_values))
    xfe_max = float(np.max(trial_values))
    if abs(xfe_max - xfe_min) < 1e-10:
        xfe_min -= 0.001
        xfe_max += 0.001
    if trial_values.size > 1:
        min_step = float(np.min(np.diff(np.unique(np.sort(trial_values)))))
    else:
        min_step = 0.1
    xfe_step = max(0.002, min(0.02, min_step / 10.0))

    state = {
        "accepted": False,
        "result": float(trial_values[idx0]),
        "updating": False,
        "action": "none",
    }

    from matplotlib.widgets import Slider, Button
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import AutoMinorLocator, ScalarFormatter

    fig = plt.figure(figsize=(13.2, 10))
    fig.patch.set_facecolor("#f4f4f2")
    gs = GridSpec(17, 14, figure=fig, hspace=0.75, wspace=0.7)
    ax_spec = fig.add_subplot(gs[0:5, 0:10])
    ax_zoom = fig.add_subplot(gs[0:8, 10:14])
    ax_ratio = fig.add_subplot(gs[5:8, 0:10], sharex=ax_spec)
    ax_sl_xfe = fig.add_subplot(gs[9, 1:10])
    ax_sl_norm = fig.add_subplot(gs[10, 1:10])
    ax_sl_shift = fig.add_subplot(gs[11, 1:10])
    ax_sl_res = fig.add_subplot(gs[12, 1:10])
    ax_sl_sm = fig.add_subplot(gs[13, 1:10])
    ax_sl_dchi2 = fig.add_subplot(gs[14, 1:10])
    ax_btn_keep = fig.add_subplot(gs[9, 10:14])
    ax_btn_quit = fig.add_subplot(gs[10, 10:14])
    ax_btn_refit = fig.add_subplot(gs[11, 10:14])
    ax_btn_fitmet = fig.add_subplot(gs[12, 10:14])
    ax_btn_fitall = fig.add_subplot(gs[13, 10:14])
    ax_btn_prev = fig.add_subplot(gs[14, 10:12])
    ax_btn_next = fig.add_subplot(gs[14, 12:14])
    ax_btn_extend_minus = fig.add_subplot(gs[15, 10:12])
    ax_btn_extend_plus = fig.add_subplot(gs[15, 12:14])

    btn_keep = Button(ax_btn_keep, "Accept (y/enter)", color="#84d889", hovercolor="#6fcb75")
    btn_quit = Button(ax_btn_quit, "Quit (q/esc)", color="#e6c9c9", hovercolor="#dcaeae")
    btn_refit = Button(ax_btn_refit, "Refit nuis. (f)", color="#c7dcf0", hovercolor="#b2cfe8")
    btn_fitmet = Button(ax_btn_fitmet, "Fit met (m)", color="#cde5c9", hovercolor="#bcdab6")
    btn_fitall = Button(ax_btn_fitall, "Fit all (a)", color="#cfdce8", hovercolor="#bccfdf")
    btn_prev = Button(ax_btn_prev, "Prev (p)", color="#ececec", hovercolor="#e0e0e0")
    btn_next = Button(ax_btn_next, "Next (n)", color="#ececec", hovercolor="#e0e0e0")
    btn_extend_minus = Button(ax_btn_extend_minus, "Extend -0.25", color="#f2dfbf", hovercolor="#ead2a8")
    btn_extend_plus = Button(ax_btn_extend_plus, "Extend +0.25", color="#f2dfbf", hovercolor="#ead2a8")

    slider_xfe = Slider(
        ax=ax_sl_xfe,
        label="[X/Fe]",
        valmin=xfe_min,
        valmax=xfe_max,
        valinit=float(trial_values[idx0]),
        valstep=xfe_step,
        color="steelblue",
    )
    slider_norm = Slider(ax=ax_sl_norm, label="Norm", valmin=0.85, valmax=1.15, valinit=float(best_seed["norm_factor"]), valstep=0.001, color="grey")
    slider_shift = Slider(ax=ax_sl_shift, label="Delta lambda [A]", valmin=-0.20, valmax=0.20, valinit=float(best_seed["wvl_shift"]), valstep=0.0005, color="grey")
    slider_res = Slider(ax=ax_sl_res, label="R", valmin=max(1000.0, 0.6 * float(best_seed.get("res_value", 28000.0))),
                        valmax=1.4 * float(best_seed.get("res_value", 28000.0)),
                        valinit=float(best_seed.get("res_value", 28000.0)), valstep=100.0, color="grey")
    slider_sm = Slider(ax=ax_sl_sm, label="Sm [px]", valmin=0.0, valmax=2.0,
                       valinit=float(np.clip(smooth_sigma_default, 0.0, 2.0)), valstep=0.05, color="sandybrown")
    slider_dchi2 = Slider(ax=ax_sl_dchi2, label="chi2 width x", valmin=0.5, valmax=3.0,
                          valinit=1.0, valstep=0.05, color="#d4b16a")

    for _ax in [ax_sl_xfe, ax_sl_norm, ax_sl_shift, ax_sl_res, ax_sl_sm, ax_sl_dchi2]:
        _ax.set_facecolor("#f5f5f3")
    for _ax in [ax_btn_keep, ax_btn_quit, ax_btn_refit, ax_btn_fitmet, ax_btn_fitall, ax_btn_prev, ax_btn_next, ax_btn_extend_minus, ax_btn_extend_plus]:
        _ax.set_facecolor("#f5f5f3")

    info_text = fig.text(
        0.02, 0.965, "",
        fontsize=9, fontfamily="monospace", va="top",
        bbox={"facecolor": "#fff7d6", "alpha": 0.95, "boxstyle": "round"},
    )
    fig.text(
        0.02, 0.015,
        "left/right: X/Fe | -/+ : extend | p/n: prev/next line | f: refit nuis | m: fit met | a: fit all | y: accept | q: quit",
        fontsize=9, color="#5f5f5f",
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.06)

    def get_current_window_half():
        return float(chi2_half_window) * float(slider_dchi2.val)

    def syn_raw_for_xfe(xfe):
        xfe = float(np.clip(xfe, trial_values[0], trial_values[-1]))
        if trial_values.size == 1:
            raw = syn_grid[0]
            idx_near = 0
        else:
            i_hi = int(np.searchsorted(trial_values, xfe, side="right"))
            i_hi = min(max(i_hi, 1), len(trial_values) - 1)
            i_lo = i_hi - 1
            x_lo = float(trial_values[i_lo])
            x_hi = float(trial_values[i_hi])
            if abs(x_hi - x_lo) < 1e-12:
                t = 0.0
            else:
                t = (xfe - x_lo) / (x_hi - x_lo)
            raw = (1.0 - t) * syn_grid[i_lo] + t * syn_grid[i_hi]
            idx_near = int(np.argmin(np.abs(trial_values - xfe)))
        return raw, idx_near

    def evaluate_chi2(xfe, norm_factor, shift, res_value, sm_sigma, win_half):
        syn_raw, _ = syn_raw_for_xfe(xfe)
        chi2, npts = compute_chi2_for_params(
            obs_w, obs_f, obs_e,
            syn_w_base, syn_raw,
            float(line_info["l0"]), float(win_half),
            float(norm_factor), float(shift), float(res_value), float(sm_sigma),
        )
        return float(chi2), int(npts), syn_raw

    def redraw(_=None):
        if state["updating"]:
            return
        xfe = float(slider_xfe.val)
        norm_factor = float(slider_norm.val)
        shift = float(slider_shift.val)
        res_value = float(slider_res.val)
        sm_sigma = float(slider_sm.val)
        win_half = get_current_window_half()
        l0 = float(line_info["l0"])
        plot_half = max(1.5, win_half * 6.0)
        zoom_half = max(0.20, win_half * 1.8)

        chi2, npts, syn_raw = evaluate_chi2(xfe, norm_factor, shift, res_value, sm_sigma, win_half)
        model_plot = apply_resolution_to_model(syn_w_base, syn_raw, res_value)

        obs_plot_w = obs_w + shift
        obs_plot_f = obs_f * norm_factor
        obs_plot_e = obs_e * abs(norm_factor)
        if gaussian_filter1d is not None and sm_sigma > 0.0:
            obs_plot_f = gaussian_filter1d(obs_plot_f, sigma=sm_sigma)
            obs_plot_e = gaussian_filter1d(obs_plot_e, sigma=sm_sigma)

        mask_main = (obs_plot_w >= (l0 - plot_half)) & (obs_plot_w <= (l0 + plot_half))
        if not np.any(mask_main):
            mask_main = np.full(obs_plot_w.shape, True, dtype=bool)
        xlim_main = (l0 - plot_half, l0 + plot_half)

        ax_spec.clear()
        ax_spec.set_facecolor("#fbfbfa")
        ax_spec.set_title(f"{line_info['symbol']}  {line_info['l0']:.3f} A", fontsize=13, fontweight="semibold")
        ax_spec.plot(obs_plot_w[mask_main], obs_plot_f[mask_main], color="black", lw=1.0, label="Observed")
        ax_spec.fill_between(
            obs_plot_w[mask_main],
            (obs_plot_f - obs_plot_e)[mask_main],
            (obs_plot_f + obs_plot_e)[mask_main],
            color="#737373", alpha=0.18
        )
        ax_spec.plot(syn_w_base, model_plot, color="#cf123f", lw=1.8, label="Model")
        ax_spec.axhline(1.0, color="#222", ls="--", lw=0.9)
        ax_spec.axvline(l0, color="#333", lw=0.9, ls="--")
        ax_spec.axvspan(l0 - win_half, l0 + win_half, color="#f1dc8a", alpha=0.42, zorder=0)
        ax_spec.set_xlim(*xlim_main)
        ax_spec.set_ylim(0.01, 1.09)
        ax_spec.set_ylabel("Norm flux")
        ax_spec.grid(axis="y", color="#d9d9d9", alpha=0.35, lw=0.6)
        ax_spec.legend(loc="lower left", fontsize=9, framealpha=0.9, facecolor="#f8f8f8")

        obs_loc_mask = (obs_plot_w >= xlim_main[0]) & (obs_plot_w <= xlim_main[1])
        obs_loc_w = obs_plot_w[obs_loc_mask]
        obs_loc_f = obs_plot_f[obs_loc_mask]
        with np.errstate(divide="ignore", invalid="ignore"):
            syn_on_obs = np.interp(obs_loc_w, syn_w_base, model_plot, left=np.nan, right=np.nan)
            denom = obs_loc_f
            ratio = np.where(np.isfinite(syn_on_obs) & (denom > 0), syn_on_obs / denom, np.nan)

        ax_ratio.clear()
        ax_ratio.set_facecolor("#fbfbfa")
        ax_ratio.plot(obs_loc_w, ratio, color="#cf123f", lw=1.35)
        ax_ratio.axhline(1.0, color="#222", ls="--", lw=0.9)
        ax_ratio.axvspan(l0 - win_half, l0 + win_half, color="#f1dc8a", alpha=0.42, zorder=0)
        ax_ratio.set_ylabel("Mod / Obs")
        ax_ratio.set_xlabel("Wavelength [\u00c5]")
        ax_ratio.set_ylim(0.75, 1.25)
        ax_ratio.set_xlim(*ax_spec.get_xlim())
        ax_ratio.grid(axis="y", color="#d9d9d9", alpha=0.35, lw=0.6)

        ax_zoom.clear()
        ax_zoom.set_facecolor("#fbfbfa")
        zoom_mask = (obs_plot_w >= (l0 - zoom_half)) & (obs_plot_w <= (l0 + zoom_half))
        ax_zoom.plot(obs_plot_w[zoom_mask], obs_plot_f[zoom_mask], color="black", lw=1.0)
        ax_zoom.fill_between(
            obs_plot_w[zoom_mask],
            (obs_plot_f - obs_plot_e)[zoom_mask],
            (obs_plot_f + obs_plot_e)[zoom_mask],
            color="#737373", alpha=0.18
        )
        ax_zoom.plot(syn_w_base, model_plot, color="#cf123f", lw=1.6)
        ax_zoom.axhline(1.0, color="#222", ls="--", lw=0.9)
        ax_zoom.axvline(l0, color="#333", lw=0.9, ls="--")
        ax_zoom.axvspan(l0 - win_half, l0 + win_half, color="#f1dc8a", alpha=0.42, zorder=0)
        ax_zoom.set_xlim(l0 - zoom_half, l0 + zoom_half)
        ax_zoom.set_title("Local Zoom", fontsize=11, fontweight="semibold")
        ax_zoom.tick_params(labelsize=9)
        ax_zoom.grid(axis="y", color="#d9d9d9", alpha=0.35, lw=0.6)
        obs_at_l0 = np.interp(l0, obs_plot_w, obs_plot_f, left=np.nan, right=np.nan)
        mod_at_l0 = np.interp(l0, syn_w_base, model_plot, left=np.nan, right=np.nan)
        center_vals = np.array([obs_at_l0, mod_at_l0], dtype=float)
        center_vals = center_vals[np.isfinite(center_vals)]
        if center_vals.size == 0:
            y0 = 0.75
        else:
            y0 = float(np.min(center_vals))
        ax_zoom.set_ylim(max(0.01, y0 - 0.15), 1.09)

        for _ax in (ax_spec, ax_ratio, ax_zoom):
            fmt = ScalarFormatter(useOffset=False)
            fmt.set_scientific(False)
            _ax.xaxis.set_major_formatter(fmt)
            _ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            _ax.minorticks_on()
            _ax.tick_params(axis="x", which="major", length=6)
            _ax.tick_params(axis="x", which="minor", length=3)

        info_text.set_text(
            f"line={line_info['l0']:.3f}  element={line_info['symbol']}  trial=[X/Fe]={xfe:+.3f}\n"
            f"chi2={chi2:.3f}  npts={npts}  norm={norm_factor:.4f}  shift={shift:+.4f}  "
            f"R={res_value:.0f}  sm={sm_sigma:.1f}  dchi2={slider_dchi2.val:.2f} (±{win_half:.3f} A)"
        )
        fig.canvas.draw_idle()

    def set_controls(xfe=None, norm=None, shift=None, res=None, sm=None, dchi2=None):
        state["updating"] = True
        try:
            if xfe is not None:
                slider_xfe.set_val(float(np.clip(xfe, trial_values[0], trial_values[-1])))
            if norm is not None:
                slider_norm.set_val(float(norm))
            if shift is not None:
                slider_shift.set_val(float(shift))
            if res is not None:
                slider_res.set_val(float(res))
            if sm is not None:
                slider_sm.set_val(float(np.clip(sm, 0.0, 2.0)))
            if dchi2 is not None:
                slider_dchi2.set_val(float(dchi2))
        finally:
            state["updating"] = False
        redraw()

    def fit_metallicity_only():
        x0 = float(slider_xfe.val)
        norm = float(slider_norm.val)
        shift = float(slider_shift.val)
        resv = float(slider_res.val)
        sm = float(slider_sm.val)
        win_half = get_current_window_half()

        def objective(x):
            chi2, _, _ = evaluate_chi2(x, norm, shift, resv, sm, win_half)
            return chi2 if np.isfinite(chi2) else 1e30

        if minimize_scalar is not None:
            result = minimize_scalar(
                objective,
                bounds=(float(np.min(trial_values)), float(np.max(trial_values))),
                method="bounded",
                options={"xatol": max(0.001, xfe_step / 2.0)},
            )
            best_x = float(result.x) if result.success else x0
        else:
            test = np.linspace(float(np.min(trial_values)), float(np.max(trial_values)), 120)
            vals = np.array([objective(x) for x in test], dtype=float)
            best_x = float(test[int(np.nanargmin(vals))])
        set_controls(xfe=best_x)

    def fit_all():
        if minimize is None:
            fit_metallicity_only()
            return
        x0 = np.array([
            float(slider_xfe.val),
            float(slider_norm.val),
            float(slider_shift.val),
            float(slider_res.val),
            float(slider_dchi2.val),
        ], dtype=float)
        sm = float(slider_sm.val)
        win0 = float(chi2_half_window)
        bounds = [
            (float(np.min(trial_values)), float(np.max(trial_values))),
            (0.85, 1.15),
            (-0.20, 0.20),
            (float(slider_res.val) * 0.6, float(slider_res.val) * 1.4),
            (0.5, 3.0),
        ]

        def objective(p):
            dscale = float(p[4])
            win_half = win0 * dscale
            chi2, _, _ = evaluate_chi2(p[0], p[1], p[2], p[3], sm, win_half)
            if not np.isfinite(chi2):
                return 1e30
            # Light regularization: avoid over-wide windows unless really needed.
            penalty = 3.0 * (dscale - 1.0) ** 2
            return chi2 + penalty

        result = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 250})
        best = result.x if result.success else x0
        set_controls(xfe=best[0], norm=best[1], shift=best[2], res=best[3], dchi2=best[4])

    def on_accept(_):
        xfe = float(slider_xfe.val)
        norm = float(slider_norm.val)
        shift = float(slider_shift.val)
        resv = float(slider_res.val)
        sm = float(slider_sm.val)
        win_half = get_current_window_half()
        _, _, syn_raw = evaluate_chi2(xfe, norm, shift, resv, sm, win_half)
        state["result"] = xfe
        state["norm_factor"] = norm
        state["wvl_shift"] = shift
        state["res_value"] = resv
        state["smooth_sigma"] = sm
        state["chi2_half_window"] = win_half
        state["syn_w"] = np.array(syn_w_base, dtype=float)
        state["syn_f"] = np.array(syn_raw, dtype=float)
        state["action"] = "accept"
        state["accepted"] = True
        plt.close(fig)

    def on_quit(_):
        state["action"] = "quit"
        state["accepted"] = True
        plt.close(fig)

    def on_prev(_):
        state["action"] = "prev"
        state["accepted"] = True
        plt.close(fig)

    def on_next(_):
        state["action"] = "next"
        state["accepted"] = True
        plt.close(fig)

    def on_extend_minus(_):
        state["action"] = "extend_minus"
        state["extend_delta"] = 0.25
        state["accepted"] = True
        plt.close(fig)

    def on_extend_plus(_):
        state["action"] = "extend_plus"
        state["extend_delta"] = 0.25
        state["accepted"] = True
        plt.close(fig)

    def on_refit(_):
        if minimize is None:
            redraw()
            return
        xfe = float(slider_xfe.val)
        x0 = np.array([float(slider_norm.val), float(slider_shift.val), float(slider_res.val)], dtype=float)
        sm = float(slider_sm.val)
        win_half = get_current_window_half()
        bounds = [(0.85, 1.15), (-0.20, 0.20), (float(slider_res.val) * 0.6, float(slider_res.val) * 1.4)]

        def objective(p):
            chi2, _, _ = evaluate_chi2(xfe, p[0], p[1], p[2], sm, win_half)
            return chi2 if np.isfinite(chi2) else 1e30

        result = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
        best = result.x if result.success else x0
        set_controls(norm=best[0], shift=best[1], res=best[2])

    def on_key(event):
        key = (event.key or "").lower()
        if key in {"left"}:
            set_controls(xfe=float(slider_xfe.val) - xfe_step)
        elif key in {"right"}:
            set_controls(xfe=float(slider_xfe.val) + xfe_step)
        elif key in {"enter", " ", "y"}:
            on_accept(None)
        elif key in {"q", "escape"}:
            on_quit(None)
        elif key == "p":
            on_prev(None)
        elif key == "n":
            on_next(None)
        elif key in {"-", "_"}:
            on_extend_minus(None)
        elif key in {"+", "="}:
            on_extend_plus(None)
        elif key == "f":
            on_refit(None)
        elif key == "m":
            fit_metallicity_only()
        elif key == "a":
            fit_all()

    slider_xfe.on_changed(redraw)
    slider_norm.on_changed(redraw)
    slider_shift.on_changed(redraw)
    slider_res.on_changed(redraw)
    slider_sm.on_changed(redraw)
    slider_dchi2.on_changed(redraw)
    btn_keep.on_clicked(on_accept)
    btn_quit.on_clicked(on_quit)
    btn_refit.on_clicked(on_refit)
    btn_fitmet.on_clicked(lambda _evt: fit_metallicity_only())
    btn_fitall.on_clicked(lambda _evt: fit_all())
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_extend_minus.on_clicked(on_extend_minus)
    btn_extend_plus.on_clicked(on_extend_plus)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show(block=True)
    plt.close(fig)

    if state["accepted"]:
        return state
    fallback["action"] = "quit"
    return fallback


def save_star_outputs(runtime, star_ctx, best_rows, grid_rows, pdf_pages):
    results_dir = Path(runtime["results_path"])
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = star_ctx["star_slug"]
    star_label = infer_star_label(star_ctx)

    best_path = results_dir / f"{stem}_best.csv"
    grid_path = results_dir / f"{stem}_grid.csv"
    pdf_path = results_dir / f"{stem}_summary.pdf"
    legacy_txt_path = results_dir / f"results_{star_label}"
    legacy_pdf_path = results_dir / f"results_plots_{star_label}.pdf"

    with open(best_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "line", "element", "atomic_number", "ion", "ep", "gflog",
                "best_xfe", "best_xfe_err", "best_xfe_err_local", "best_xfe_err_mc",
                "best_chi2", "best_n_pts", "norm_factor", "wvl_shift", "res_value", "chi2_half_window",
                "area_mA", "area_mA_err",
                "status", "n_mcmc_ok"
            ],
        )
        writer.writeheader()
        writer.writerows(best_rows)

    with open(grid_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "line", "element", "atomic_number", "ion", "ep", "gflog",
                "trial_xfe", "chi2", "n_pts",
                "norm_factor", "wvl_shift", "res_value", "chi2_half_window",
                "status", "spectrum_file", "message"
            ],
        )
        writer.writeheader()
        rows = []
        for row in grid_rows:
            rows.append({
                "line": row["line"],
                "element": row["element"],
                "atomic_number": row["atomic_number"],
                "ion": row.get("ion"),
                "ep": row.get("ep"),
                "gflog": row.get("gflog"),
                "trial_xfe": row["trial_xfe"],
                "chi2": row["chi2"],
                "n_pts": row["n_pts"],
                "norm_factor": row["norm_factor"],
                "wvl_shift": row["wvl_shift"],
                "res_value": row["res_value"],
                "chi2_half_window": row.get("chi2_half_window", np.nan),
                "status": row["status"],
                "spectrum_file": row["spectrum_file"],
                "message": row["message"],
            })
        writer.writerows(rows)

    # Legacy fixed-width table, compatible with previous specMCMC-style outputs.
    headers = [
        "# line (Å)", "Element", "Ionization", "Energy Potential", "log(gf)",
        "Norm Factor", "Error Norm Factor", "Res", "Error Res",
        "Wvl Shift", "Error Wvl Shift", "Metallicity", "Error Metallicity",
        "Microturb", "Error Microturb", "Area (mÅ)", "Error Area (mÅ)", "Error Message",
    ]
    widths = [16, 16, 16, 16, 16, 14, 20, 12, 14, 14, 20, 14, 20, 14, 20, 16, 18, 29]
    with open(legacy_txt_path, "w") as handle:
        handle.write("".join(f"{h:<{w}}" for h, w in zip(headers, widths)).rstrip() + "\n")
        handle.write("".join(f"{'-' * (w - 1)} " for w in widths).rstrip() + "\n")
        for row in sorted(best_rows, key=lambda item: float(item.get("line", np.nan))):
            status = str(row.get("status", "ok"))
            is_ok = status == "ok"
            xfe = float(row.get("best_xfe", np.nan))
            xfe_err = float(row.get("best_xfe_err", np.nan))
            met_abs = float(star_ctx["feh"]) + xfe if np.isfinite(xfe) else np.nan
            vals = [
                f"{float(row.get('line', np.nan)):<16.3f}" if np.isfinite(float(row.get("line", np.nan))) else f"{'-999':<16}",
                f"{str(row.get('element', '-')):<16}",
                f"{int(row.get('ion')):<16d}" if row.get("ion") is not None else f"{'-':<16}",
                f"{float(row.get('ep', np.nan)):<16.3f}" if np.isfinite(float(row.get("ep", np.nan))) else f"{'-999':<16}",
                f"{float(row.get('gflog', np.nan)):<16.3f}" if np.isfinite(float(row.get("gflog", np.nan))) else f"{'-999':<16}",
                f"{float(row.get('norm_factor', np.nan)):<14.4f}" if np.isfinite(float(row.get("norm_factor", np.nan))) else f"{'-999':<14}",
                f"{float(row.get('norm_err', np.nan)):<20.4f}" if np.isfinite(float(row.get("norm_err", np.nan))) else f"{'0.0000':<20}",
                f"{float(row.get('res_value', np.nan)):<12.2f}" if np.isfinite(float(row.get("res_value", np.nan))) else f"{'-999':<12}",
                f"{float(row.get('res_err', np.nan)):<14.2f}" if np.isfinite(float(row.get("res_err", np.nan))) else f"{'0.00':<14}",
                f"{float(row.get('wvl_shift', np.nan)):<14.4f}" if np.isfinite(float(row.get("wvl_shift", np.nan))) else f"{'-999':<14}",
                f"{float(row.get('shift_err', np.nan)):<20.4f}" if np.isfinite(float(row.get("shift_err", np.nan))) else f"{'0.0000':<20}",
                f"{met_abs:<14.4f}" if np.isfinite(met_abs) else f"{'-999':<14}",
                f"{xfe_err:<20.4f}" if np.isfinite(xfe_err) else f"{'-999':<20}",
                f"{float(star_ctx['xi']):<14.4f}",
                f"{'0.0000':<20}",
                f"{float(row.get('area_mA', np.nan)):<16.4f}" if np.isfinite(float(row.get("area_mA", np.nan))) else f"{'-999':<16}",
                f"{float(row.get('area_mA_err', np.nan)):<18.4f}" if np.isfinite(float(row.get("area_mA_err", np.nan))) else f"{'-999':<18}",
                f"#{('#' if is_ok else status):<28}",
            ]
            handle.write("".join(vals).rstrip() + "\n")

    if pdf_pages is not None:
        pdf_pages.close()
        pdf_saved = str(pdf_path)
        try:
            shutil.copyfile(pdf_path, legacy_pdf_path)
        except Exception:
            pass
    else:
        pdf_saved = None

    return str(best_path), str(grid_path), pdf_saved


def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def run_star(runtime, star_row):
    star_ctx = build_star_context(star_row, runtime)
    print(f"Analysing star: {star_ctx['name_obs_spec']}")

    synth_dir = Path(runtime["analysis_synth_path"]) / star_ctx["star_slug"]
    model_dir = synth_dir / "_models"
    synth_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(runtime["results_path"]) / f"{star_ctx['star_slug']}_checkpoint.json"
    synth_cache_index_path = synth_dir / "synth_cache_index.json"

    turbo_writer = TurboSpecWriter(str(synth_dir) + os.sep, runtime["linelist_library_path"], str(model_dir) + os.sep, runtime["launch_path"])
    header_creator = HeaderCreator(runtime["launch_path"], str(synth_dir) + os.sep)
    convolver = SpectrumConvolver(str(synth_dir) + os.sep)
    model_maker = ModelMaker(runtime["dataset_model_path"])

    turbo_writer.check_and_create_contopacdir(str(model_dir) + os.sep)

    try:
        model_name = prepare_model(star_ctx, runtime, model_maker, str(model_dir) + os.sep)
    except SkipModelError as exc:
        raise RuntimeError(f"Cannot prepare model for {star_ctx['name_obs_spec']}: {exc}") from exc

    runtime["res_init"] = float(star_ctx["res"])
    runtime["res_min"] = max(1000.0, 0.6 * float(star_ctx["res"]))
    runtime["res_max"] = 1.4 * float(star_ctx["res"])

    obs_w, obs_f, obs_e, obs_tell = read_observed_spectrum(star_ctx["obs_path"], runtime["default_err"])
    line_list = read_line_list(star_ctx["line_path"])
    xfe_grid = make_xfe_grid(runtime)
    obs_min = float(np.min(obs_w))
    obs_max = float(np.max(obs_w))
    min_pts_line = 5

    pdf_path = Path(runtime["results_path"]) / f"{star_ctx['star_slug']}_summary.pdf"
    pdf_pages = PdfPages(pdf_path) if PdfPages is not None else None

    grid_rows = []
    accepted_rows_by_idx = {}
    line_cache = {}
    current_idx = 0
    total_lines = len(line_list)
    run_completed = False

    if synth_cache_index_path.exists():
        try:
            with open(synth_cache_index_path, "r") as handle:
                synth_cache_index = json.load(handle)
            if not isinstance(synth_cache_index, dict):
                synth_cache_index = {}
        except Exception:
            synth_cache_index = {}
    else:
        synth_cache_index = {}

    if checkpoint_path.exists():
        try:
            current_idx, accepted_rows_by_idx, grid_rows = _checkpoint_load(checkpoint_path)
            print(f"  resume checkpoint: idx={current_idx} accepted={len(accepted_rows_by_idx)}")
        except Exception as exc:
            print(f"  warning: cannot read checkpoint ({exc}), starting from scratch.")
            current_idx, accepted_rows_by_idx, grid_rows = 0, {}, []

    while 0 <= current_idx < total_lines:
        if current_idx in accepted_rows_by_idx:
            current_idx += 1
            continue
        line_info = line_list[current_idx]
        print(f"  [{current_idx + 1}/{total_lines}] {line_info['symbol']} {line_info['l0']:.3f} A")

        if current_idx not in line_cache:
            per_line_rows = []
            best_row = None
            line_center = float(line_info["l0"])
            win = float(runtime["chi2_half_window"])

            if (line_center + win) < obs_min or (line_center - win) > obs_max:
                line_cache[current_idx] = {
                    "valid": False,
                    "line_center": line_center,
                    "per_line_rows": [],
                    "best_row": {
                        "line": line_center,
                        "element": line_info["symbol"],
                        "atomic_number": int(line_info["atomic_number"]),
                        "ion": line_info.get("ion"),
                        "ep": line_info.get("ep"),
                        "gflog": line_info.get("gflog"),
                        "best_xfe": np.nan,
                        "best_xfe_err": np.nan,
                        "best_chi2": np.nan,
                        "best_n_pts": 0,
                        "norm_factor": np.nan,
                        "wvl_shift": np.nan,
                        "res_value": np.nan,
                        "chi2_half_window": np.nan,
                        "status": "line_outside_obs_range",
                        "n_mcmc_ok": 0,
                    },
                }
            else:
                obs_local_mask = (obs_w >= (line_center - win)) & (obs_w <= (line_center + win))
                if int(np.sum(obs_local_mask)) < min_pts_line:
                    line_cache[current_idx] = {
                        "valid": False,
                        "line_center": line_center,
                        "per_line_rows": [],
                        "best_row": {
                            "line": line_center,
                            "element": line_info["symbol"],
                            "atomic_number": int(line_info["atomic_number"]),
                            "ion": line_info.get("ion"),
                            "ep": line_info.get("ep"),
                            "gflog": line_info.get("gflog"),
                            "best_xfe": np.nan,
                            "best_xfe_err": np.nan,
                            "best_chi2": np.nan,
                            "best_n_pts": int(np.sum(obs_local_mask)),
                            "norm_factor": np.nan,
                            "wvl_shift": np.nan,
                            "res_value": np.nan,
                            "chi2_half_window": np.nan,
                            "status": "too_few_obs_points",
                            "n_mcmc_ok": 0,
                        },
                    }
                else:
                    for trial_xfe in xfe_grid:
                        spectrum_file = ""
                        generated_now = False
                        syn_w = None
                        syn_f = None
                        message = ""
                        status = "ok"
                        norm_factor = 1.0
                        wvl_shift = 0.0
                        res_value = float(star_ctx["res"])
                        smooth_sigma = float(runtime.get("smooth_sigma", 0.0))
                        chi2 = np.inf
                        n_pts = 0

                        try:
                            spectrum_file, generated_now = synthesize_trial(
                                runtime, star_ctx, model_name, turbo_writer, header_creator, convolver,
                                line_info, float(trial_xfe), synth_cache_index=synth_cache_index
                            )
                            syn_path = synth_dir / spectrum_file
                            syn_w, syn_f = read_synthetic_normflux(str(syn_path))
                            fit = fit_trial_alignment(runtime, obs_w, obs_f, obs_e, syn_w, syn_f, line_center)
                            chi2 = fit["chi2"]
                            n_pts = fit["n_pts"]
                            norm_factor = fit["norm_factor"]
                            wvl_shift = fit["wvl_shift"]
                            res_value = fit["res_value"]
                        except Exception as exc:
                            status = "error"
                            message = str(exc)
                            res_value = float(star_ctx["res"])

                        row = {
                            "line": line_center,
                            "element": line_info["symbol"],
                            "atomic_number": int(line_info["atomic_number"]),
                            "ion": line_info.get("ion"),
                            "ep": line_info.get("ep"),
                            "gflog": line_info.get("gflog"),
                            "trial_xfe": float(trial_xfe),
                            "chi2": chi2,
                            "n_pts": n_pts,
                            "norm_factor": norm_factor,
                            "wvl_shift": wvl_shift,
                            "status": status,
                            "spectrum_file": spectrum_file,
                            "message": message,
                            "res_value": res_value,
                            "chi2_half_window": float(runtime["chi2_half_window"]),
                            "smooth_sigma": smooth_sigma,
                            "syn_w": syn_w,
                            "syn_f": syn_f,
                            "obs_w": obs_w,
                            "obs_f": obs_f,
                            "obs_e": obs_e,
                            "obs_tell": obs_tell,
                        }
                        per_line_rows.append(row)
                        grid_rows.append(_compact_grid_row(row))

                        if status == "ok" and np.isfinite(chi2):
                            if best_row is None or chi2 < best_row["chi2"]:
                                best_row = row

                        # Never remove cached/reused synthetic files; only optionally remove fresh ones.
                        if (not runtime["keep_synthetic"]) and generated_now and spectrum_file:
                            safe_remove(str(synth_dir / spectrum_file))

                    if best_row is None:
                        line_cache[current_idx] = {
                            "valid": False,
                            "line_center": line_center,
                            "per_line_rows": per_line_rows,
                            "best_row": {
                                "line": line_center,
                                "element": line_info["symbol"],
                                "atomic_number": int(line_info["atomic_number"]),
                                "ion": line_info.get("ion"),
                                "ep": line_info.get("ep"),
                                "gflog": line_info.get("gflog"),
                                "best_xfe": np.nan,
                                "best_xfe_err": np.nan,
                                "best_chi2": np.nan,
                                "best_n_pts": 0,
                                "norm_factor": np.nan,
                                "wvl_shift": np.nan,
                                "res_value": np.nan,
                                "chi2_half_window": np.nan,
                                "status": "no_valid_trial",
                                "n_mcmc_ok": 0,
                            },
                        }
                    else:
                        line_cache[current_idx] = {
                            "valid": True,
                            "line_center": line_center,
                            "per_line_rows": per_line_rows,
                            "best_row": best_row,
                        }

        cached = line_cache[current_idx]
        if not cached["valid"]:
            accepted_rows_by_idx[current_idx] = cached["best_row"]
            current_idx += 1
            _checkpoint_save(checkpoint_path, current_idx, accepted_rows_by_idx, grid_rows)
            continue

        per_line_rows = cached["per_line_rows"]
        best_row = cached["best_row"]
        line_center = float(cached["line_center"])

        if not runtime["auto"]:
            try:
                selected = choose_manual_xfe(
                    line_info,
                    per_line_rows,
                    best_row["trial_xfe"],
                    chi2_half_window=runtime["chi2_half_window"],
                    smooth_sigma_default=runtime.get("smooth_sigma", 0.0),
                )
            except Exception as exc:
                print(
                    f"    [interactive fallback] line {line_center:.3f}: "
                    f"{type(exc).__name__}: {exc}"
                )
                selected = {
                    "action": "accept",
                    "result": float(best_row["trial_xfe"]),
                    "norm_factor": float(best_row["norm_factor"]),
                    "wvl_shift": float(best_row["wvl_shift"]),
                    "res_value": float(best_row.get("res_value", runtime.get("res_init", 28000.0))),
                    "smooth_sigma": float(best_row.get("smooth_sigma", runtime.get("smooth_sigma", 0.0))),
                    "chi2_half_window": float(runtime["chi2_half_window"]),
                }

            action = str(selected.get("action", "accept")).lower()
            if action == "quit":
                _checkpoint_save(checkpoint_path, current_idx, accepted_rows_by_idx, grid_rows)
                break
            if action == "prev":
                current_idx = max(0, current_idx - 1)
                _checkpoint_save(checkpoint_path, current_idx, accepted_rows_by_idx, grid_rows)
                continue
            if action == "next":
                current_idx += 1
                _checkpoint_save(checkpoint_path, current_idx, accepted_rows_by_idx, grid_rows)
                continue
            if action in {"extend", "extend_minus", "extend_plus"}:
                delta = float(selected.get("extend_delta", 0.25))
                existing_trials = np.array([float(row["trial_xfe"]) for row in per_line_rows], dtype=float)
                if action == "extend_minus":
                    targets = [float(np.min(existing_trials) - delta)]
                elif action == "extend_plus":
                    targets = [float(np.max(existing_trials) + delta)]
                else:
                    targets = [float(np.min(existing_trials) - delta), float(np.max(existing_trials) + delta)]
                added = 0
                for trial_xfe in targets:
                    if np.any(np.isclose(existing_trials, trial_xfe, atol=1e-8)):
                        continue
                    spectrum_file = ""
                    generated_now = False
                    syn_w = None
                    syn_f = None
                    message = ""
                    status = "ok"
                    norm_factor = 1.0
                    wvl_shift = 0.0
                    res_value = float(star_ctx["res"])
                    smooth_sigma = float(runtime.get("smooth_sigma", 0.0))
                    chi2 = np.inf
                    n_pts = 0
                    try:
                        spectrum_file, generated_now = synthesize_trial(
                            runtime, star_ctx, model_name, turbo_writer, header_creator, convolver,
                            line_info, float(trial_xfe), synth_cache_index=synth_cache_index
                        )
                        syn_path = synth_dir / spectrum_file
                        syn_w, syn_f = read_synthetic_normflux(str(syn_path))
                        fit = fit_trial_alignment(runtime, obs_w, obs_f, obs_e, syn_w, syn_f, line_center)
                        chi2 = fit["chi2"]
                        n_pts = fit["n_pts"]
                        norm_factor = fit["norm_factor"]
                        wvl_shift = fit["wvl_shift"]
                        res_value = fit["res_value"]
                    except Exception as exc:
                        status = "error"
                        message = str(exc)
                        res_value = float(star_ctx["res"])

                    row = {
                        "line": line_center,
                        "element": line_info["symbol"],
                        "atomic_number": int(line_info["atomic_number"]),
                        "ion": line_info.get("ion"),
                        "ep": line_info.get("ep"),
                        "gflog": line_info.get("gflog"),
                        "trial_xfe": float(trial_xfe),
                        "chi2": chi2,
                        "n_pts": n_pts,
                        "norm_factor": norm_factor,
                        "wvl_shift": wvl_shift,
                        "status": status,
                        "spectrum_file": spectrum_file,
                        "message": message,
                        "res_value": res_value,
                        "chi2_half_window": float(runtime["chi2_half_window"]),
                        "smooth_sigma": smooth_sigma,
                        "syn_w": syn_w,
                        "syn_f": syn_f,
                        "obs_w": obs_w,
                        "obs_f": obs_f,
                        "obs_e": obs_e,
                        "obs_tell": obs_tell,
                    }
                    per_line_rows.append(row)
                    grid_rows.append(_compact_grid_row(row))
                    existing_trials = np.append(existing_trials, trial_xfe)
                    if status == "ok" and np.isfinite(chi2):
                        if best_row is None or chi2 < best_row["chi2"]:
                            best_row = row
                    if (not runtime["keep_synthetic"]) and generated_now and spectrum_file:
                        safe_remove(str(synth_dir / spectrum_file))
                    added += 1

                if added > 0:
                    per_line_rows.sort(key=lambda row: float(row["trial_xfe"]))
                    cached["per_line_rows"] = per_line_rows
                    if best_row is not None:
                        cached["best_row"] = best_row
                    print(f"    extended X/Fe grid by ±{delta:.2f}: added {added} trial(s)")
                else:
                    print(f"    extend requested (±{delta:.2f}) but no new trial was needed.")

                _checkpoint_save(checkpoint_path, current_idx, accepted_rows_by_idx, grid_rows)
                continue

            selected_xfe = float(selected["result"])
            manual_match = [row for row in per_line_rows if np.isclose(row["trial_xfe"], selected_xfe)]
            if manual_match:
                best_row_ref = manual_match[0]
            else:
                best_row_ref = min(per_line_rows, key=lambda row: abs(float(row["trial_xfe"]) - selected_xfe))

            # Important: keep original trial rows untouched for grid.csv integrity.
            best_row = dict(best_row_ref)
            best_row["trial_xfe"] = float(selected_xfe)
            best_row["norm_factor"] = float(selected["norm_factor"])
            best_row["wvl_shift"] = float(selected["wvl_shift"])
            best_row["res_value"] = float(selected["res_value"])
            best_row["smooth_sigma"] = float(selected.get("smooth_sigma", runtime.get("smooth_sigma", 0.0)))
            best_row["chi2_half_window"] = float(selected.get("chi2_half_window", runtime["chi2_half_window"]))
            if selected.get("syn_w") is not None and selected.get("syn_f") is not None:
                best_row["syn_w"] = np.asarray(selected["syn_w"], dtype=float)
                best_row["syn_f"] = np.asarray(selected["syn_f"], dtype=float)
            best_row["chi2"], best_row["n_pts"] = compute_chi2_for_params(
                obs_w, obs_f, obs_e,
                best_row["syn_w"], best_row["syn_f"],
                line_center, best_row["chi2_half_window"],
                best_row["norm_factor"], best_row["wvl_shift"], best_row["res_value"],
                best_row["smooth_sigma"]
            )
            cached["best_row"] = best_row

        mcmc_info = estimate_mcmc_error(runtime, per_line_rows, obs_w, obs_f, obs_e, line_center)
        local_err = estimate_local_xfe_error(per_line_rows, best_row, obs_w, obs_f, obs_e, line_center)
        combined_err = local_err if np.isfinite(local_err) else mcmc_info["best_xfe_err"]
        ew_best = compute_equivalent_width_mA(
            best_row.get("syn_w"),
            best_row.get("syn_f"),
            line_center,
            best_row.get("chi2_half_window", runtime["chi2_half_window"]),
            best_row.get("res_value", runtime.get("res_init", 28000.0)),
        )
        ew_err = np.nan
        if np.isfinite(combined_err) and combined_err > 0:
            w_lo, f_lo = interpolate_synthetic_profile(per_line_rows, float(best_row["trial_xfe"]) - float(combined_err))
            w_hi, f_hi = interpolate_synthetic_profile(per_line_rows, float(best_row["trial_xfe"]) + float(combined_err))
            ew_lo = compute_equivalent_width_mA(
                w_lo, f_lo, line_center,
                best_row.get("chi2_half_window", runtime["chi2_half_window"]),
                best_row.get("res_value", runtime.get("res_init", 28000.0)),
            )
            ew_hi = compute_equivalent_width_mA(
                w_hi, f_hi, line_center,
                best_row.get("chi2_half_window", runtime["chi2_half_window"]),
                best_row.get("res_value", runtime.get("res_init", 28000.0)),
            )
            if np.isfinite(ew_lo) and np.isfinite(ew_hi):
                ew_err = 0.5 * abs(float(ew_hi) - float(ew_lo))

        accepted_rows_by_idx[current_idx] = {
            "line": line_center,
            "element": line_info["symbol"],
            "atomic_number": int(line_info["atomic_number"]),
            "ion": line_info.get("ion"),
            "ep": line_info.get("ep"),
            "gflog": line_info.get("gflog"),
            "best_xfe": best_row["trial_xfe"],
            "best_xfe_err": combined_err,
            "best_xfe_err_local": local_err,
            "best_xfe_err_mc": mcmc_info["best_xfe_err"],
            "best_chi2": best_row["chi2"],
            "best_n_pts": best_row["n_pts"],
            "norm_factor": best_row["norm_factor"],
            "wvl_shift": best_row["wvl_shift"],
            "res_value": best_row.get("res_value", np.nan),
            "chi2_half_window": best_row.get("chi2_half_window", runtime["chi2_half_window"]),
            "area_mA": ew_best,
            "area_mA_err": ew_err,
            "status": "ok",
            "n_mcmc_ok": mcmc_info["n_mcmc_ok"],
        }

        if pdf_pages is not None and plt is not None:
            fig = plt.figure(figsize=(10, 7), dpi=150)
            plot_line_panel(
                fig, star_ctx, line_info, obs_w, obs_f, obs_e, obs_tell,
                per_line_rows, best_row["trial_xfe"],
                xfe_err=combined_err,
                chi2_half_window=best_row.get("chi2_half_window", runtime["chi2_half_window"]),
            )
            pdf_pages.savefig(fig)
            plt.close(fig)

        current_idx += 1
        _checkpoint_save(checkpoint_path, current_idx, accepted_rows_by_idx, grid_rows)

    run_completed = current_idx >= total_lines
    with open(synth_cache_index_path, "w") as handle:
        json.dump(synth_cache_index, handle, indent=2)

    best_rows = [accepted_rows_by_idx[i] for i in sorted(accepted_rows_by_idx.keys())]
    results = save_star_outputs(runtime, star_ctx, best_rows, grid_rows, pdf_pages)
    if run_completed and checkpoint_path.exists():
        checkpoint_path.unlink()
    return results


if __name__ == "__main__":
    args = parse_args()
    runtime = load_config(args)

    os.makedirs(runtime["analysis_synth_path"], exist_ok=True)
    os.makedirs(runtime["results_path"], exist_ok=True)

    for star_row in runtime["stars"]:
        best_csv, grid_csv, pdf_file = run_star(runtime, star_row)
        print(f"  saved best results: {best_csv}")
        print(f"  saved trial grid:   {grid_csv}")
        if pdf_file:
            print(f"  saved summary PDF:  {pdf_file}")
