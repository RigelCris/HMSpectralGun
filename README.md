# HMSpectralGun

HMSpectralGun generates synthetic stellar spectra using Turbospectrum.

## Quick start (clone-and-run)

1. Clone the repository.
2. Make sure `TurboSpectrum` is already installed on your machine.

You need a working TurboSpectrum installation before running HMSpectralGun. In particular, you must know the paths to:

- the `exec-gf` directory containing the TurboSpectrum executables
- the `COM` directory where HMSpectralGun writes and launches the generated `.com` scripts

3. Download the MARCS model grid separately and place the model files in `./marcs_generator/dataset/`.

HMSpectralGun does not ship the full MARCS model dataset. Download the MARCS models you need and copy or extract the `.mod` files into:

```bash
marcs_generator/dataset/
```

If you keep the MARCS dataset somewhere else, set `HMSPECTRALGUN_DATASET_MODEL_PATH` to that directory.

4. Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

5. Export runtime paths for your TurboSpectrum installation:

```bash
export HMSPECTRALGUN_EXEC_PATH="/path/to/turbospectrum/exec-gf"
export HMSPECTRALGUN_LAUNCH_PATH="/path/to/turbospectrum/COM"
```

Linux note: generated `.com` scripts use `csh` (`#!/bin/csh -f`), so install `csh`/`tcsh` if missing.

Optional overrides:

- `HMSPECTRALGUN_DATASET_MODEL_PATH` (default: `./marcs_generator/dataset`)
- `HMSPECTRALGUN_CONTOPAC_PATH` (default: `${HMSPECTRALGUN_LAUNCH_PATH}/contopac`)
- `HMSPECTRALGUN_INTERPOLATOR_EXE` (default: `./marcs_generator/interpol_modeles`)
- `HMSPECTRALGUN_BABSMA` and `HMSPECTRALGUN_BSYN` (if you prefer explicit executable paths)

6. Prepare your `input.ts` and run:

`savepath` is created automatically if it does not exist yet.

```bash
python main.py --input input.ts --progress
```

Parallel run:

```bash
python main_parallel.py --input input.ts
```

Optional: global commands from anywhere, without changing your working directory.

For `zsh` (`~/.zshrc`):

```bash
spectralgun() {
  /path/to/HMSpectralGun/.venv/bin/python /path/to/HMSpectralGun/main.py --input "${1:-input.ts}" --progress
}

parallel_spectralgun() {
  /path/to/HMSpectralGun/.venv/bin/python /path/to/HMSpectralGun/main_parallel.py --input "${1:-input.ts}"
}
```

For `bash` (`~/.bashrc`), use the same function block.

Reload your shell config:

- `zsh`: `source ~/.zshrc`
- `bash`: `source ~/.bashrc`

Usage:

```bash
spectralgun /path/to/input.ts
parallel_spectralgun /path/to/input.ts
```

## Notes

- `main.py` and `main_parallel.py` no longer depend on machine-specific hardcoded paths.
- `line_fit_config.example.json` now uses relative placeholder paths and must be adapted to your dataset.
