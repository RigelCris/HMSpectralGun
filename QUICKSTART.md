# HMSpectralGun Quickstart

1. Clone the repository and enter it:

```bash
git clone <repo-url>
cd HMSpectralGun
```

2. Make sure `TurboSpectrum` is already installed and built on your machine.

You need the paths to:

- `.../exec-gf`
- `.../COM/santerre`

3. Download the MARCS model grid separately.

This repository does not include the full MARCS dataset. Copy or extract the MARCS `.mod` files into:

```bash
marcs_generator/dataset/
```

If your MARCS models live elsewhere, export `HMSPECTRALGUN_DATASET_MODEL_PATH` to that directory instead.

4. Create the Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

5. Export the TurboSpectrum runtime paths:

```bash
export HMSPECTRALGUN_EXEC_PATH="/path/to/turbospectrum/exec-gf"
export HMSPECTRALGUN_LAUNCH_PATH="/path/to/turbospectrum/COM/santerre"
```

Optional:

```bash
export HMSPECTRALGUN_DATASET_MODEL_PATH="/path/to/marcs/models"
```

6. Run HMSpectralGun:

```bash
python main.py --input input.ts --progress
```

For parallel execution:

```bash
python main_parallel.py --input input.ts
```
