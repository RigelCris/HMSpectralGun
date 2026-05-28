# HMSpectralGun

HMSpectralGun generates synthetic stellar spectra using Turbospectrum.

## Quick start (clone-and-run)

1. Clone the repository.
2. Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Export runtime paths for your Turbospectrum installation:

```bash
export HMSPECTRALGUN_EXEC_PATH="/path/to/turbospectrum/exec-gf"
export HMSPECTRALGUN_LAUNCH_PATH="/path/to/turbospectrum/COM/santerre"
```

Optional overrides:

- `HMSPECTRALGUN_DATASET_MODEL_PATH` (default: `./marcs_generator/dataset`)
- `HMSPECTRALGUN_CONTOPAC_PATH` (default: `${HMSPECTRALGUN_LAUNCH_PATH}/contopac`)
- `HMSPECTRALGUN_INTERPOLATOR_EXE` (default: `./marcs_generator/interpol_modeles`)
- `HMSPECTRALGUN_BABSMA` and `HMSPECTRALGUN_BSYN` (if you prefer explicit executable paths)

4. Prepare your `input.ts` and run:

```bash
python main.py --input input.ts --progress
```

Parallel run:

```bash
python main_parallel.py --input input.ts
```

Optional: global commands from anywhere (`zsh`), without changing your working directory:

```bash
spectralgun() {
  /path/to/HMSpectralGun/.venv/bin/python /path/to/HMSpectralGun/main.py --input "${1:-input.ts}"
}

parallel_spectralgun() {
  /path/to/HMSpectralGun/.venv/bin/python /path/to/HMSpectralGun/main_parallel.py --input "${1:-input.ts}"
}
```

Add them to `~/.zshrc`, then run `source ~/.zshrc`.

Usage:

```bash
spectralgun /path/to/input.ts
parallel_spectralgun /path/to/input.ts
```

## Notes

- `main.py` and `main_parallel.py` no longer depend on machine-specific hardcoded paths.
- `line_fit_config.example.json` now uses relative placeholder paths and must be adapted to your dataset.
