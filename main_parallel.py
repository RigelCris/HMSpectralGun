#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import subprocess
import shutil
import logging
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from tqdm import tqdm
from mendeleev import element as chemical_element

try:
    from HMSpectralGun.Intro             import Intro
    from HMSpectralGun.InputParameters   import InputParameters
    from HMSpectralGun.LineListManager   import LineListManager
    from HMSpectralGun.ModelMaker        import ModelMaker, SkipModelError
    from HMSpectralGun.TurboSpecWriter   import TurboSpecWriter
    from HMSpectralGun.HeaderCreator     import HeaderCreator
    from HMSpectralGun.SpectrumConvolver import SpectrumConvolver
    from HMSpectralGun.SNR               import SpectrumNoiseAdder
    from HMSpectralGun.Resampling        import SpectrumResampler
except ModuleNotFoundError:
    # Fallback per esecuzione diretta dalla cartella HMSpectralGun.
    from Intro             import Intro
    from InputParameters   import InputParameters
    from LineListManager   import LineListManager
    from ModelMaker        import ModelMaker, SkipModelError
    from TurboSpecWriter   import TurboSpecWriter
    from HeaderCreator     import HeaderCreator
    from SpectrumConvolver import SpectrumConvolver
    from SNR               import SpectrumNoiseAdder
    from Resampling        import SpectrumResampler
def parse_arguments():
    p = argparse.ArgumentParser(description='Sintesi spettri in parallelo')
    p.add_argument('--input', type=str, default='input.ts',
                   help='File di input (default: input.ts)')
    return p.parse_args()

def setup_main_logger(log_path):
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(ch)
    return logger

def init_worker(input_file, launchpath_arg,
                savepath_arg, linelistpath_arg, modelpath_arg,
                model_map_arg):
    """
    Inizializza ogni processo: setta i global per i parametri e TurboSpecWriter.
    """
    global params, launchpath, savepath, linelistpath, modelpath, turbo_spec_writer, model_map
    params            = InputParameters(input_file)
    launchpath        = launchpath_arg
    savepath          = savepath_arg
    linelistpath      = linelistpath_arg
    modelpath         = modelpath_arg
    model_map         = model_map_arg
    turbo_spec_writer = TurboSpecWriter(savepath, linelistpath, modelpath, launchpath)

def _tag_output_name(filename, tag):
    path = Path(filename)
    return f"{path.stem}_{tag}{path.suffix}"

def _read_explicit_xfe_override(row):
    """
    Parse optional explicit abundance override from input row.
    Returns:
      (atomic_number:int|None, xfe:float|None, tag:str)
    """
    z_raw = row.get('override_elem', '*')
    xfe_raw = row.get('override_xfe', '*')

    if z_raw == '*' or xfe_raw == '*':
        return None, None, ''

    z = int(float(z_raw))
    xfe = float(xfe_raw)
    symbol = chemical_element(z).symbol
    sign = 'p' if xfe >= 0 else 'm'
    amp = int(round(abs(xfe) * 100))
    tag = f"_{symbol}{sign}{amp:03d}"
    return z, xfe, tag

def _log_indicates_failure(log_path):
    """
    Detect common Turbospectrum runtime failures in worker log files.
    """
    patterns = (
        "ERROR! Model file does not exist",
        "Fortran runtime error",
        "No such file or directory",
        "Segmentation fault",
        "Floating point exception",
        "Error termination"
    )
    try:
        with open(log_path, 'r', errors='replace') as fh:
            content = fh.read()
    except OSError:
        return "Unable to read worker log file."

    for pat in patterns:
        if pat in content:
            return pat
    return None

def worker(k):
    """
    Genera lo spettro k-esimo:
      0) controlla che il modello sia stato preparato per k
      1) prepara linelist e abbondanze
      2) writer + .com
      3) esegue il .com esterno (log dedicato per worker)
      4) crea sempre header base
      5) tenta la convoluzione (senza errori se manca output o RES invalido)
      6) risample & rumore
      7) pulizia linelist
      8) rinomina finale in *_k{k} per evitare collisioni
    """
    try:
        # 0) Se il modello non è stato preparato (SkipModelError nel main),
        #    esci pulito senza crashare con KeyError.
        if k not in model_map:
            return f"SKIPPED_k{k}_no_model"

        df   = params.get_dataframe()
        row  = df.iloc[k]
        model = model_map[k]

        llmgr  = LineListManager()
        header = HeaderCreator(launchpath, savepath)
        conv   = SpectrumConvolver(savepath)

        # 1) Preparazione linelist e abbondanze
        dref = params.process_solar_references(
            row['linelist_file'], row['abu_file'], row['[Fe/H]']
        )
        linespec, Crat = dref['linelist_content'], dref['Crat']
        isotopic_n     = 612613

        kw, keyvec = params.read_single_element(row['monoelem'])
        if kw == 'Yes':
            tmp_tag = f"k{k}_p{os.getpid()}"
            linespec = llmgr.create_linelist_single_element(
                linelistpath, linespec, keyvec, tmp_tag=tmp_tag)

        df_abu, _ = params.read_abu_file(row['abu_file'])
        elem   = df_abu['AtomicNumber'].to_numpy(dtype=float)
        delta  = df_abu['AbundanceDifference'].to_numpy(dtype=float)
        override_z, override_xfe, override_tag = _read_explicit_xfe_override(row)
        if Crat is None or len(Crat) == 0:
            isotopic_n = 0
            iso_val = 0.0
        else:
            iso_val = Crat.iloc[0]
        if row['[a/Fe]'] != 0.0:
            alpha = row['[a/Fe]']
            for el in [8,12,14,16,20,22]:
                delta[elem==el] = alpha

        # Priorità massima: override esplicito [X/Fe] da input.ts
        if override_z is not None:
            mask = (elem == float(override_z))
            if mask.any():
                delta[mask] = override_xfe
            else:
                elem = np.append(elem, float(override_z))
                delta = np.append(delta, float(override_xfe))

        abu = params.solar_reference(elem, delta, row['[Fe/H]'])

        # 2) Writer + .com
        raw_name = turbo_spec_writer.writer(
            model, row['[Fe/H]'], row['[a/Fe]'],
            row['lam_i'], row['lam_f'], row['xi'],
            linespec, row['snr'], elem, abu,
            isotopic_n, iso_val,
            keyw=kw, el=keyvec[2],
            ext=row['extension'], deltalam=row['resnum'],
            interp=params.get_keywords()[1],
            NLTE=params.get_keywords()[2],
            abundance_tag=override_tag
        )
        if raw_name == 'STOP':
            return None

        # 3) Esecuzione del .com esterno (MOOG, ecc.) con log dedicato per worker
        com_path = os.path.join(launchpath, f"{raw_name}.com")
        log_filename = f"log_{Path(raw_name).stem}_k{k}.txt"
        log_path = os.path.join(launchpath, log_filename)
        with open(log_path, "w") as worker_log:
            proc = subprocess.run(
                [com_path],
                stdout=worker_log,
                stderr=subprocess.STDOUT,
                check=False
            )

        fail_reason = _log_indicates_failure(log_path)
        if fail_reason is not None:
            raise RuntimeError(
                f"Turbospectrum failed ({fail_reason}). See log: {log_path}"
            )

        raw_output_path = os.path.join(savepath, raw_name)
        if (not os.path.exists(raw_output_path)) or os.path.getsize(raw_output_path) == 0:
            raise RuntimeError(
                f"Missing/empty spectrum output after .com execution (exit={proc.returncode}). "
                f"Expected file: {raw_output_path}. See log: {log_path}"
            )

        # 4) Creazione header base
        header.create_combined_header(
            raw_name,
            model=model,
            met=row['[Fe/H]'], alpha=row['[a/Fe]'],
            elem=elem, deltaabu=delta,
            lam_min=row['lam_i'], lam_max=row['lam_f'],
            xi=row['xi'],
            isotopic_n=isotopic_n, isotopic_val=iso_val,
            keyw=row['chemistry'], deltalam=row['resnum'],
            log_filename=log_filename
        )

        final_name = raw_name

        # 5) Convoluzione (silenziosa se manca il file output o RES non valido)
        try:
            conv_name = conv.instrbroad(raw_name, row['RES'], verbose='no')
            header.create_combined_header(
                conv_name,
                model=model,
                met=row['[Fe/H]'], alpha=row['[a/Fe]'],
                elem=elem, deltaabu=delta,
                lam_min=row['lam_i'], lam_max=row['lam_f'],
                xi=row['xi'],
                isotopic_n=isotopic_n, isotopic_val=iso_val,
                keyw=row['chemistry'], deltalam=row['resnum'], res=row['RES'],
                log_filename=log_filename
            )
            # Se la convoluzione va a buon fine, il file raw non serve più.
            if os.path.exists(raw_output_path):
                os.remove(raw_output_path)
            final_name = conv_name
        except (ValueError, FileNotFoundError):
            # Nessuna convoluzione: mantieni il file raw.
            pass

        # 6) Risample + rumore
        if row['sampl'] != '*':
            SpectrumResampler(savepath, final_name, row['sampl']).resample_and_save()
            if row['snr'] != '*':
                final_name = SpectrumNoiseAdder(
                    savepath, final_name, row['snr'], noise_type='GAUSS'
                ).add_noise_and_save()

        # 7) Pulizia linelist temporanei
        llmgr.delete_tmp_linelist(linelistpath, linespec, keyvec, kw)

        # 8) Rinomina finale per evitare collisioni fra worker
        final_output_path = os.path.join(savepath, final_name)
        if os.path.exists(final_output_path):
            unique_final_name = _tag_output_name(final_name, f"k{k}")
            os.replace(final_output_path, os.path.join(savepath, unique_final_name))

        old_com = os.path.join(launchpath, f"{raw_name}.com")
        if os.path.exists(old_com):
            unique_com = f"{_tag_output_name(raw_name, f'k{k}')}.com"
            os.replace(old_com, os.path.join(launchpath, unique_com))

        return None

    except Exception as e:
        # traceback completo per non vedere più messaggi criptici tipo "ERRORE inatteso: 2"
        import traceback
        print(f"[Worker {k}] ERRORE inatteso: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return f"ERROR_k{k}"

if __name__ == '__main__':
    args       = parse_arguments()
    cwd        = os.getcwd()
    input_file = os.path.join(cwd, args.input)

    # Percorsi di default
    dataset_dp = '/Users/cfanelli/astro/softw/TS-NLTE/COM/santerre/HMSpectralGun/marcs_generator/dataset/'
    launch_dp  = '/Users/cfanelli/astro/softw/TS-NLTE/COM/'

    # Logger principale
    stamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_log = os.path.join(cwd, f"synth_main_{stamp}.log")
    logger   = setup_main_logger(main_log)

    # Stampa header console
    params0 = InputParameters(input_file)
    save_dp, linelist_dp, model_dp = params0.get_paths()
    Intro.intro1()
    logger.info('*** Lista dei percorsi ***')
    logger.info(f"Save Path:     {save_dp}")
    logger.info(f"Linelist Path: {linelist_dp}")
    logger.info(f"Model Path:    {model_dp}")
    logger.info(f"Launch Path:   {launch_dp}")
    logger.info(f"Current Path:  {cwd}")

    # Prepara la directory contopac
    TurboSpecWriter(save_dp, linelist_dp, model_dp, launch_dp)\
        .check_and_create_contopacdir(model_dp)

    # Numero spettri e DataFrame
    n_spec = params0.write_output_spectra_count()
    df     = params0.get_dataframe()
    kws    = params0.get_keywords()

    # Mappatura modelli
    mm        = ModelMaker(dataset_dp)
    model_map = {}
    missing_explicit_models = []
    for k in range(n_spec):
        row = df.iloc[k]
        if kws[0]=='False' and kws[1]=='True':
            T, g     = map(float, row['Model'].split(',')[:2])
            try:
                models = mm.select_models_for_interpolation(
                    int(T), g, row['[Fe/H]'], row['xi'], row['chemistry']
                )

                fname = mm.write_interpolator(
                    int(T), g, row['[Fe/H]'], row['xi'], row['chemistry'], model_dp, models
                )

            except SkipModelError as err:
                print("")
                print("Skipping synthetic spectrum:")
                print(err)
                print("")
                continue
            
        elif kws[0]=='False' and kws[1].lower()=='nearest':
            T, g  = map(float, row['Model'].split(',')[:2])
            fname = mm.select_nearest_model(
                int(T), g, row['[Fe/H]'], row['xi'],
                row['chemistry'], model_dp
            )
        else:
            fname0 = row['Model'].split(',')[0]
            target_model = os.path.join(model_dp, fname0)
            if not os.path.exists(target_model):
                source_model = os.path.join(dataset_dp, fname0)
                if os.path.exists(source_model):
                    shutil.copy2(source_model, target_model)
                else:
                    missing_explicit_models.append((k, fname0))
                    continue
            fname = fname0

        model_map[k] = fname
        #logger.info(f"> Modello per riga {k}: {fname}")

    if missing_explicit_models:
        logger.warning(
            f"Spettri saltati: {len(missing_explicit_models)} modello/i esplicito/i non trovato/i "
            f"in model path ({model_dp}) e dataset ({dataset_dp})."
        )
        for k, name in missing_explicit_models:
            logger.warning(f"  - riga {k}: {name}")

    # Esecuzione parallela
    results = []
    if n_spec > 1:
        n_proc = max(1, cpu_count() - 1)
        logger.info(f"Schedulo {n_spec} spettri con {n_proc} worker…")
        with ProcessPoolExecutor(
            max_workers=n_proc,
            initializer=init_worker,
            initargs=(input_file, launch_dp,
                      save_dp, linelist_dp, model_dp,
                      model_map)
        ) as executor:
            futures = {executor.submit(worker, k): k for k in range(n_spec)}
            for _ in tqdm(
                as_completed(futures),
                total=n_spec,
                desc="Synthesizing spectra",
                dynamic_ncols=True,
                file=sys.stdout
            ):
                results.append(_.result())
    else:
        init_worker(input_file, launch_dp,
                    save_dp, linelist_dp, model_dp,
                    model_map)
        results = [worker(0)]


    # Riepilogo finale: distinguo skip (modello non preparato) da errori veri
    skipped = [r for r in results if r and r.startswith("SKIPPED_")]
    errs    = [r for r in results if r and r.startswith("ERROR_")]
    n_ok    = len(results) - len(skipped) - len(errs)

    logger.info(f"Riepilogo: {n_ok} OK, {len(skipped)} saltati, {len(errs)} errori "
                f"(totale {len(results)}).")

    if skipped:
        logger.warning(f"Spettri saltati perché il modello atmosferico "
                       f"non era disponibile/interpolabile ({len(skipped)}):")
        for fn in skipped:
            logger.warning(f"  - {fn}")

    if errs:
        logger.error(f"Spettri con errori inattesi ({len(errs)}):")
        for fn in errs:
            logger.error(f"  - {fn}")

    if not skipped and not errs:
        logger.info("Tutti gli spettri e i relativi header sono stati creati con successo.")

    logger.info(f"Log completo: {main_log}")
