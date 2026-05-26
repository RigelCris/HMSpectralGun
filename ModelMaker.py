#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust MARCS model selector/interpolator helper.

Fix principali rispetto alla versione originale:
- parsing robusto dei nomi modello, non basato su indici fissi;
- uso effettivo della microturbulenza xi nella selezione;
- ricerca di un cubo di interpolazione realmente esistente;
- vincoli conservativi sui passi di interpolazione:
  Delta Teff <= 250 K, Delta logg <= 0.25 dex, Delta [Fe/H] <= 0.25 dex;
- nessuna extrapolazione: il target deve cadere dentro la coppia low/high;
- errore esplicito/SkipModelError invece di IndexError;
- selezione nearest model coerente in tutte le coordinate.
"""

from __future__ import annotations

import itertools
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ModelRecord:
    filename: str
    teff: float
    logg: float
    xi: float
    chem: str
    met: float


class ModelSelectionError(RuntimeError):
    """Base class for model-selection failures."""


class SkipModelError(ModelSelectionError):
    """
    Raised when no safe local interpolation cube exists.

    Catch this in the calling loop if you want to skip the current
    synthetic spectrum and continue with the next input row.
    """


class ModelMaker:
    MAX_TEFF_DELTA = 250.0
    MAX_LOGG_DELTA = 0.50
    MAX_MET_DELTA = 0.50

    MODEL_RE = re.compile(
        r"^s(?P<teff>\d+(?:\.\d+)?)_"
        r"g(?P<logg>[+-]?\d+(?:\.\d+)?)_"
        r"m(?P<mass>\d+(?:\.\d+)?)_"
        r"t(?P<xi>\d+)_"
        r"(?P<chem>[A-Za-z]{2})_"
        r"z(?P<met>[+-]?\d+(?:\.\d+)?)"
        r".*\.mod$"
    )

    def __init__(
        self,
        dataset_model_path: str,
        interpolator_exe: str = "/Users/cfanelli/astro/softw/TS-NLTE/COM/santerre/HMSpectralGun/marcs_generator/interpol_modeles",
    ):
        self.dataset_model_path = Path(dataset_model_path).expanduser()
        self.interpolator_exe = str(Path(interpolator_exe).expanduser())

    @staticmethod
    def _as_float(value) -> float:
        return float(np.asarray(value).item())

    @staticmethod
    def _same(a: float, b: float, atol: float = 1.0e-8) -> bool:
        return abs(float(a) - float(b)) <= atol

    @staticmethod
    def _sign_char(value: float) -> str:
        return "p" if float(value) >= 0.0 else "m"

    @staticmethod
    def _unique_sorted(values: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0:
            return arr
        return np.unique(np.round(arr, 8))

    # Function to find the nearest value in an array
    def find_nearest(self, array, value):
        array = np.asarray(array, dtype=float)
        if array.size == 0:
            raise ValueError("Cannot find nearest value: empty array.")
        idx = np.abs(array - float(value)).argmin()
        return array[idx]

    # Function to find the two nearest values in an array
    def find_2nearest(self, array, value):
        values = self._unique_sorted(array)
        if values.size == 0:
            raise ValueError("Cannot find two nearest values: empty array.")
        if values.size == 1:
            return values[0], values[0]

        value = float(value)
        exact = values[np.isclose(values, value, atol=1.0e-8)]
        if exact.size > 0:
            return exact[0], exact[0]

        order = np.argsort(np.abs(values - value))[:2]
        pair = np.sort(values[order])
        return pair[0], pair[1]

    def _candidate_pairs(
        self,
        values: Iterable[float],
        target: float,
        max_delta: float,
        coord_name: str,
    ) -> List[Tuple[float, float]]:
        """
        Return allowed low/high pairs for interpolation.

        Conservative rules:
        - no extrapolation: low <= target <= high;
        - no wide interpolation: high - low <= max_delta;
        - exact pairs (x, x) are allowed only when x == target.

        Candidate order prefers the exact match first, then the narrowest
        local interval. This prevents pathological choices such as
        Teff=(3600, 5000) just because the midpoint is close to the target.
        """
        values = self._unique_sorted(values)
        if values.size == 0:
            raise SkipModelError(f"No available values for {coord_name}.")

        target = float(target)
        max_delta = float(max_delta)
        pairs: List[Tuple[float, float]] = []

        for i, lo in enumerate(values):
            for hi in values[i:]:
                lo = float(lo)
                hi = float(hi)
                width = hi - lo

                if width > max_delta + 1.0e-8:
                    continue

                # No extrapolation: the requested value must be bracketed.
                if lo - 1.0e-8 <= target <= hi + 1.0e-8:
                    pairs.append((lo, hi))

        def score(pair: Tuple[float, float]) -> Tuple[float, float, float]:
            lo, hi = pair
            width = hi - lo
            midpoint = 0.5 * (lo + hi)
            exact_penalty = 0.0 if self._same(lo, target) and self._same(hi, target) else 1.0
            return (exact_penalty, width, abs(midpoint - target))

        seen = set()
        out: List[Tuple[float, float]] = []
        for pair in sorted(pairs, key=score):
            key = (round(pair[0], 8), round(pair[1], 8))
            if key not in seen:
                seen.add(key)
                out.append(pair)

        if not out:
            available = ", ".join(f"{v:g}" for v in values[:25])
            more = " ..." if values.size > 25 else ""
            raise SkipModelError(
                f"No safe interpolation pair for {coord_name}={target:g}. "
                f"Required: no extrapolation and Delta {coord_name} <= {max_delta:g}. "
                f"Available {coord_name}: {available}{more}"
            )

        return out

    def _parse_model_name(self, filename: str) -> ModelRecord:
        name = Path(str(filename)).name.strip()
        match = self.MODEL_RE.match(name)
        if match is None:
            raise ValueError(f"Cannot parse MARCS model filename: {name}")

        xi_token = match.group("xi")
        # MARCS filenames encode xi as t01, t02, t05, ...
        xi_value = float(int(xi_token))

        return ModelRecord(
            filename=name,
            teff=float(match.group("teff")),
            logg=float(match.group("logg")),
            xi=xi_value,
            chem=match.group("chem"),
            met=float(match.group("met")),
        )

    def _discover_model_files(self, keyw_chem: str | None = None) -> List[str]:
        pattern = "s*_m1.0_t*_*.mod" if keyw_chem is None else f"s*_m1.0_t*_{keyw_chem}_z*.mod"
        files = sorted(path.name for path in self.dataset_model_path.glob(pattern))

        # Fallback useful for debugging from an existing list_models file.
        if not files:
            list_file = self.dataset_model_path / "list_models"
            if list_file.exists():
                files = [line.strip() for line in list_file.read_text().splitlines() if line.strip()]
                if keyw_chem is not None:
                    files = [name for name in files if f"_{keyw_chem}_z" in name]

        if not files:
            raise FileNotFoundError(
                f"No MARCS .mod files found in {self.dataset_model_path} for chemistry={keyw_chem!r}."
            )

        # Keep list_models for compatibility with your previous workflow.
        list_file = self.dataset_model_path / "list_models"
        try:
            list_file.write_text("\n".join(files) + "\n")
        except OSError:
            # Non-fatal: model selection can continue without rewriting list_models.
            pass

        return files

    def _load_models(self, keyw_chem: str | None = None) -> List[ModelRecord]:
        records: List[ModelRecord] = []
        bad_names: List[str] = []

        for filename in self._discover_model_files(keyw_chem):
            try:
                record = self._parse_model_name(filename)
            except ValueError:
                bad_names.append(filename)
                continue
            if keyw_chem is None or record.chem == keyw_chem:
                records.append(record)

        if not records:
            details = ""
            if bad_names:
                details = f" Unparseable filenames: {bad_names[:5]}"
            raise ValueError(f"No parseable models found for chemistry={keyw_chem!r}.{details}")

        return sorted(records, key=lambda r: (r.teff, r.logg, r.met, r.xi, r.chem, r.filename))

    def _nearest_xi(self, records: Sequence[ModelRecord], xi: float) -> float:
        xis = self._unique_sorted(record.xi for record in records)
        return float(self.find_nearest(xis, xi))

    def _find_record(
        self,
        records: Sequence[ModelRecord],
        teff: float,
        logg: float,
        met: float,
        xi: float,
    ) -> ModelRecord | None:
        matches = [
            record
            for record in records
            if self._same(record.teff, teff)
            and self._same(record.logg, logg)
            and self._same(record.met, met)
            and self._same(record.xi, xi)
        ]
        if not matches:
            return None
        return sorted(matches, key=lambda r: r.filename)[0]

    def _cube_from_pairs(
        self,
        records: Sequence[ModelRecord],
        teff_pair: Tuple[float, float],
        logg_pair: Tuple[float, float],
        met_pair: Tuple[float, float],
        xi: float,
    ) -> Tuple[List[str] | None, List[Tuple[float, float, float, float]]]:
        # Required MARCS interpolator order:
        # 1 Tefflow logglow zlow; 2 Tefflow logglow zup;
        # 3 Tefflow loggup zlow; 4 Tefflow loggup zup;
        # 5 Teffup  logglow zlow; 6 Teffup  logglow zup;
        # 7 Teffup  loggup zlow; 8 Teffup  loggup zup.
        coords = [
            (teff_pair[0], logg_pair[0], met_pair[0], xi),
            (teff_pair[0], logg_pair[0], met_pair[1], xi),
            (teff_pair[0], logg_pair[1], met_pair[0], xi),
            (teff_pair[0], logg_pair[1], met_pair[1], xi),
            (teff_pair[1], logg_pair[0], met_pair[0], xi),
            (teff_pair[1], logg_pair[0], met_pair[1], xi),
            (teff_pair[1], logg_pair[1], met_pair[0], xi),
            (teff_pair[1], logg_pair[1], met_pair[1], xi),
        ]

        models: List[str] = []
        missing: List[Tuple[float, float, float, float]] = []
        missing_keys = set()

        for coord in coords:
            record = self._find_record(records, *coord)
            if record is None:
                key = tuple(round(float(x), 8) for x in coord)
                if key not in missing_keys:
                    missing_keys.add(key)
                    missing.append(coord)
            else:
                models.append(record.filename)

        if missing:
            return None, missing
        return models, []

    @staticmethod
    def _pair_score(pair: Tuple[float, float], target: float) -> Tuple[float, float]:
        lo, hi = pair
        target = float(target)
        midpoint = 0.5 * (lo + hi)
        width = abs(hi - lo)
        return (width, abs(midpoint - target))

    def _find_valid_cube(
        self,
        records: Sequence[ModelRecord],
        Teff: float,
        logg: float,
        met: float,
        xi: float,
    ) -> Tuple[List[str], Tuple[float, float], Tuple[float, float], Tuple[float, float], float]:
        xi_selected = self._nearest_xi(records, xi)
        same_xi = [record for record in records if self._same(record.xi, xi_selected)]

        teff_pairs = self._candidate_pairs(
            (record.teff for record in same_xi), Teff, self.MAX_TEFF_DELTA, "Teff"
        )
        logg_pairs = self._candidate_pairs(
            (record.logg for record in same_xi), logg, self.MAX_LOGG_DELTA, "logg"
        )
        met_pairs = self._candidate_pairs(
            (record.met for record in same_xi), met, self.MAX_MET_DELTA, "[Fe/H]"
        )

        tried = []
        for teff_pair, logg_pair, met_pair in itertools.product(teff_pairs, logg_pairs, met_pairs):
            models, missing = self._cube_from_pairs(same_xi, teff_pair, logg_pair, met_pair, xi_selected)
            score = (
                self._pair_score(teff_pair, Teff),
                self._pair_score(logg_pair, logg),
                self._pair_score(met_pair, met),
            )
            tried.append((score, teff_pair, logg_pair, met_pair, missing))
            if models is not None:
                print(
                    "Interpolation cube:",
                    f"Teff={teff_pair}",
                    f"DeltaTeff={teff_pair[1] - teff_pair[0]:.0f}K",
                    f"logg={logg_pair}",
                    f"Deltalogg={logg_pair[1] - logg_pair[0]:.2f}",
                    f"[Fe/H]={met_pair}",
                    f"DeltaFe={met_pair[1] - met_pair[0]:.2f}",
                    f"xi={xi_selected}",
                )
                return models, teff_pair, logg_pair, met_pair, xi_selected

        tried.sort(key=lambda item: item[0])
        _, best_teff_pair, best_logg_pair, best_met_pair, best_missing = tried[0]
        missing_text = "\n".join(
            f"  Teff={t:.0f}, logg={g:.2f}, [Fe/H]={z:.2f}, xi={x:.0f}"
            for t, g, z, x in best_missing[:20]
        )
        raise SkipModelError(
            "No complete safe MARCS interpolation cube found.\n"
            f"Target: Teff={Teff}, logg={logg}, [Fe/H]={met}, xi={xi}; "
            f"nearest available xi={xi_selected}.\n"
            f"Safety limits: DeltaTeff <= {self.MAX_TEFF_DELTA:.0f} K, "
            f"Deltalogg <= {self.MAX_LOGG_DELTA:.2f}, "
            f"Delta[Fe/H] <= {self.MAX_MET_DELTA:.2f}; no extrapolation allowed.\n"
            f"Nearest attempted safe cube: Teff={best_teff_pair}, "
            f"logg={best_logg_pair}, [Fe/H]={best_met_pair}.\n"
            f"Missing vertices:\n{missing_text}\n"
            "Skip this synthetic spectrum, add the missing MARCS models, or set Interp=False."
        )

    # Function to select models for interpolation
    def select_models_for_interpolation(self, Teff, logg, met, xi, keyw_chem):
        records = self._load_models(keyw_chem)
        models, _, _, _, _ = self._find_valid_cube(
            records=records,
            Teff=float(Teff),
            logg=float(logg),
            met=float(met),
            xi=float(xi),
        )
        return tuple(models)

    # Function to write the interpolator
    def write_interpolator(self, Teff, logg, met, xi, chem, model_path, models):
        if len(models) != 8:
            raise ValueError(f"The MARCS interpolator needs exactly 8 models; got {len(models)}.")

        model_path = Path(model_path).expanduser()
        model_path.mkdir(parents=True, exist_ok=True)

        Teff = float(Teff)
        logg = float(logg)
        met = float(met)

        sgn_logg = self._sign_char(logg)
        sgn_met = self._sign_char(met)
        logg_abs = f"{abs(logg):.2f}"
        met_abs = f"{abs(met):.2f}"
        teff_label = f"{int(round(Teff))}"

        name_model = f"T{teff_label}_G{sgn_logg}{logg_abs}_{chem}_Z{sgn_met}{met_abs}"
        interpol_file = model_path / f"{name_model}.interpol"
        alt_file = model_path / f"{name_model}.alt"

        if interpol_file.exists():
            return interpol_file.name

        print("Interpolating Model Atmosphere...")

        run_dir = Path.cwd()
        script_path = (run_dir / "interp_models.com").resolve()
        log_path = (run_dir / "logmarcs").resolve()

        model1, model2, model3, model4, model5, model6, model7, model8 = models
        dataset = str(self.dataset_model_path)
        out_dir = str(model_path)

        script = f"""#!/bin/csh -f

##################################################################################################
# Output turbospectrum/babsma format compatible
# Extrapolation is not advised, even if allowed by this program.
# Requires a cubic set of 8 MARCS models.
# Input order:
# model1: Tefflow logglow zlow
# model2: Tefflow logglow zup
# model3: Tefflow loggup zlow
# model4: Tefflow loggup zup
# model5: Teffup  logglow zlow
# model6: Teffup  logglow zup
# model7: Teffup  loggup zlow
# model8: Teffup  loggup zup
##################################################################################################

set model_path = '{out_dir}'
set dataset_model_path = '{dataset}'
set marcs_binary = '.false.'
set name_model = '{name_model}'

set model1 = '{model1}'
set model2 = '{model2}'
set model3 = '{model3}'
set model4 = '{model4}'
set model5 = '{model5}'
set model6 = '{model6}'
set model7 = '{model7}'
set model8 = '{model8}'

set Tref = '{Teff:.0f}'
set loggref = '{logg:.2f}'
set zref = '{met:.2f}'
set model_out = '{interpol_file}'
set model_out2 = '{alt_file}'

set test = '.false.'
set model_test = '${{dataset_model_path}}/${{model1}}'

{self.interpolator_exe} <<EOF
'${{dataset_model_path}}/${{model1}}'
'${{dataset_model_path}}/${{model2}}'
'${{dataset_model_path}}/${{model3}}'
'${{dataset_model_path}}/${{model4}}'
'${{dataset_model_path}}/${{model5}}'
'${{dataset_model_path}}/${{model6}}'
'${{dataset_model_path}}/${{model7}}'
'${{dataset_model_path}}/${{model8}}'
'${{model_out}}'
'${{model_out2}}'
${{Tref}}
${{loggref}}
${{zref}}
${{test}}
${{marcs_binary}}
'${{model_test}}'
EOF
"""

        script_path.write_text(script)
        script_path.chmod(0o755)

        with log_path.open("w") as log_file:
            result = subprocess.run([str(script_path)], cwd=str(run_dir), stdout=log_file, stderr=subprocess.STDOUT, text=True)

        try:
            script_path.unlink()
        except FileNotFoundError:
            pass

        try:
            (run_dir / "modele.sm").unlink()
        except FileNotFoundError:
            pass

        if result.returncode != 0:
            raise RuntimeError(
                f"MARCS interpolator failed with return code {result.returncode}. See {log_path.resolve()}"
            )

        if not interpol_file.exists():
            raise RuntimeError(
                f"MARCS interpolator finished without creating {interpol_file}. See {log_path.resolve()}"
            )

        return interpol_file.name

    # Function to select the nearest model
    def select_nearest_model(self, Teff, logg, met, xi, keyw_chem, model_path):
        records = self._load_models(keyw_chem)
        if not records:
            raise ValueError(f"No models available for chemistry={keyw_chem!r}.")

        Teff = float(Teff)
        logg = float(logg)
        met = float(met)
        xi = float(xi)

        # Scale the metric so that no single coordinate dominates only because
        # it has a larger numerical range.
        teff_values = self._unique_sorted(record.teff for record in records)
        logg_values = self._unique_sorted(record.logg for record in records)
        met_values = self._unique_sorted(record.met for record in records)
        xi_values = self._unique_sorted(record.xi for record in records)

        def scale(values: np.ndarray) -> float:
            span = float(np.max(values) - np.min(values)) if values.size > 1 else 1.0
            return span if span > 0.0 else 1.0

        s_teff = scale(teff_values)
        s_logg = scale(logg_values)
        s_met = scale(met_values)
        s_xi = scale(xi_values)

        def distance(record: ModelRecord) -> float:
            return (
                ((record.teff - Teff) / s_teff) ** 2
                + ((record.logg - logg) / s_logg) ** 2
                + ((record.met - met) / s_met) ** 2
                + ((record.xi - xi) / s_xi) ** 2
            )

        selected = min(records, key=distance)

        sgn_logg = self._sign_char(selected.logg)
        logg_abs = f"{abs(selected.logg):.2f}"
        name_model = f"T{int(round(selected.teff))}_G{sgn_logg}{logg_abs}"

        print(" Teff    : ", f"{selected.teff:.0f}", "K")
        print(" logg    : ", f"{selected.logg:.2f}", "dex")
        print(" [Fe/H]  : ", f"{selected.met:.2f}", "dex")
        print(" Mixture : ", selected.chem)
        print(" xi      : ", f"{selected.xi:.0f}", "km/s")
        print(" Model   : ", selected.filename)

        model_path = Path(model_path).expanduser()
        model_path.mkdir(parents=True, exist_ok=True)
        destination = model_path / name_model
        source = self.dataset_model_path / selected.filename

        if not destination.exists():
            if not source.exists():
                raise FileNotFoundError(f"Selected source model does not exist: {source}")
            shutil.copy2(source, destination)

        return name_model
