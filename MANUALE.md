# HMSpectralGun - Manuale Produzione Spettri Sintetici

## 1) Obiettivo
Questo manuale copre solo la produzione di spettri sintetici con:
- `main.py` (seriale)
- `main_parallel.py` (parallelo)

## 2) Requisiti
- Python 3.10+.
- Librerie: `numpy`, `pandas`, `tqdm`, `mendeleev`, `scipy`, `PyAstronomy`, `matplotlib`.
- Turbospectrum compilato e accessibile via variabili ambiente.

Variabili consigliate:
- `HMSPECTRALGUN_EXEC_PATH` (cartella con `babsma_lu` e `bsyn_lu`)
- `HMSPECTRALGUN_LAUNCH_PATH` (cartella dove scrivere/eseguire i `.com`)

Opzionali:
- `HMSPECTRALGUN_DATASET_MODEL_PATH`
- `HMSPECTRALGUN_CONTOPAC_PATH`
- `HMSPECTRALGUN_INTERPOLATOR_EXE`

## 3) File di input principale: `input.ts`
Struttura attesa:
1. `savepath`
2. `linelistpath`
3. `modelpath`
4. `ExplicitModel=True|False`
5. `interp=True|False|nearest`
6. terza keyword (`NLTE` nel parser)
7. tabella degli spettri (15 o 17 colonne)

Esempio:
```text
/path/output_spectra/
/path/linelists/
/path/models/
ExplicitModel=False
interp=True
NLTE=False
3600,1.0  -1.00  0.20  15000  15500  2.0  st  *  28000  0.02  *  linelistH.ts  abu.ts  *  txt
```

## 4) Colonne della tabella in `input.ts`
1. `Model` (nome modello esplicito oppure `Teff,logg`)
2. `[Fe/H]`
3. `[a/Fe]`
4. `lam_i`
5. `lam_f`
6. `xi`
7. `chemistry` (`st`, `ap`, `ae`, `mc`, `hc`)
8. `sampl` (`*` o passo di ricampionamento)
9. `RES` (risoluzione per convoluzione)
10. `resnum` (passo in sintesi)
11. `monoelem` (`*`, specie tipo `FeI`, o `CO/OH/CN`)
12. `linelist_file`
13. `abu_file`
14. `snr` (`*` o valore)
15. `extension` (es. `txt`)
16. `override_elem` (opzionale, numero atomico)
17. `override_xfe` (opzionale, valore [X/Fe])

Le colonne 16-17 (se presenti) sovrascrivono le abbondanze del file `abu_file`.

## 5) File ausiliari

### 5.1 `abu_file` (es. `abu.ts`)
Due colonne:
- numero atomico
- delta abbondanza [X/Fe]

Esempio:
```text
8 0.20
12 0.20
26 0.00
612613 15.0
```

`612613` viene interpretato come rapporto isotopico C12/C13.

### 5.2 `linelist_file` (indice)
È un file indice che contiene i nomi dei file di linelist, uno per riga:
```text
vald_1.lin
vald_2.lin
```

## 6) Produzione seriale (`main.py`)
Comando:
```bash
python main.py --input input.ts
```

Con barra avanzamento:
```bash
python main.py --input input.ts --progress
```

Flusso:
- selezione/preparazione modello (interpolato, nearest, esplicito),
- scrittura script `.com`,
- esecuzione Turbospectrum,
- creazione header nel file spettro,
- eventuale convoluzione (`RES`),
- eventuale ricampionamento (`sampl`) e rumore (`snr`).

## 7) Produzione parallela (`main_parallel.py`)
Comando:
```bash
python main_parallel.py --input input.ts
```

Caratteristiche:
- usa più processi (`cpu_count() - 1`),
- genera log principale `synth_main_YYYYMMDD_HHMMSS.log`,
- crea log dedicati per worker,
- rinomina output con suffisso `_k<indice>` per evitare collisioni.

## 7.1) Opzionale: comandi da ovunque (funzioni zsh)

Se vuoi lanciare senza entrare ogni volta nella cartella del progetto, aggiungi in `~/.zshrc`:

```bash
spectralgun() {
  /path/to/HMSpectralGun/.venv/bin/python /path/to/HMSpectralGun/main.py --input "${1:-input.ts}"
}

parallel_spectralgun() {
  /path/to/HMSpectralGun/.venv/bin/python /path/to/HMSpectralGun/main_parallel.py --input "${1:-input.ts}"
}
```

Poi ricarica la shell:

```bash
source ~/.zshrc
```

Esempio uso:

```bash
spectralgun /path/to/input.ts
parallel_spectralgun /path/to/input.ts
```

## 8) Output prodotti
- Script Turbospectrum: `<nome>.com` in `launchpath`.
- Spettri sintetici: in `savepath`.
- Se convoluti: tipicamente suffisso `_RXXk`.
- Se rumore aggiunto: suffisso `_SNR<valore>`.
- Se override [X/Fe]: tag nel nome (es. `_Fep010`).

## 9) Errori comuni
- `ModuleNotFoundError`: dipendenza Python mancante.
- `No safe interpolation pair` / `No complete safe MARCS interpolation cube`: griglia modelli insufficiente.
- `Missing/empty spectrum output`: Turbospectrum non ha prodotto output valido (controllare log).
- `Spectrum already exists`: file già presente non vuoto, run saltato volontariamente.

## 10) Checklist prima del batch
- path assoluti corretti;
- cartelle `savepath`, `modelpath`, `linelistpath` raggiungibili;
- `input.ts` validato su 1 spettro;
- linelist e `abu_file` risolti correttamente;
- test breve prima del run completo.
