#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import subprocess
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm

from HMSpectralGun.Intro             import Intro
from HMSpectralGun.InputParameters   import InputParameters
from HMSpectralGun.LineListManager   import LineListManager
from HMSpectralGun.ModelMaker        import ModelMaker
from HMSpectralGun.TurboSpecWriter   import TurboSpecWriter
from HMSpectralGun.HeaderCreator     import HeaderCreator
from HMSpectralGun.SpectrumConvolver import SpectrumConvolver
from HMSpectralGun.SNR               import SpectrumNoiseAdder
from HMSpectralGun.Resampling        import SpectrumResampler

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
                model_map_arg, log_path):
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

def worker(k):
    """
    Genera lo spettro k-esimo:
      1) prepara linelist e abbondanze
      2) writer + .com
      3) esegue il .com esterno
      4) crea sempre header base
      5) tenta la convoluzione (senza errori se manca il .spec)
      6) risample & rumore
      7) pulizia linelist
      8) rinomina finale in name_k{k}
    """
    try:
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
            linespec = llmgr.create_linelist_single_element(
                linelistpath, linespec, keyvec)

        df_abu, _ = params.read_abu_file(row['abu_file'])
        elem   = df_abu['AtomicNumber'].to_numpy(dtype=float)
        delta  = df_abu['AbundanceDifference'].to_numpy(dtype=float)
        iso_val = Crat.iloc[0]
        if row['[a/Fe]'] != 0.0:
            alpha = row['[a/Fe]']
            for el in [8,12,14,16,20,22]:
                delta[elem==el] = alpha
        abu = params.solar_reference(elem, delta, row['[Fe/H]'])

        # 2) Writer + .com
        start = time.time()
        raw_name = turbo_spec_writer.writer(
            model, row['[Fe/H]'], row['[a/Fe]'],
            row['lam_i'], row['lam_f'], row['xi'],
            linespec, row['snr'], elem, abu,
            isotopic_n, iso_val,
            keyw=kw, el=keyvec[2],
            ext=row['extension'], deltalam=row['resnum'],
            interp=params.get_keywords()[1],
            NLTE=params.get_keywords()[2]
        )
        if raw_name == 'STOP':
            return None

        # 3) Esecuzione del .com esterno (MOOG, ecc.)
        subprocess.run(
            f"{os.path.join(launchpath, raw_name)}.com > /dev/null 2>&1",
            shell=True, check=False
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
            keyw=row['chemistry'], deltalam=row['resnum']
        )
        #params.print_time(time.time() - start)

        final_name = raw_name

        # 5) Convoluzione (silenziosa se manca il .spec)
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
                keyw=row['chemistry'], deltalam=row['sampl'], res=row['RES']
            )
            # rimuovo i raw se la convoluzione ha successo
            os.remove(os.path.join(savepath, f"{raw_name}.spec"))
            os.remove(os.path.join(savepath, f"{raw_name}.hdr"))
            final_name = conv_name
        except (ValueError, FileNotFoundError):
            # niente output se il file .spec per la convoluzione non c'è
            pass

        # 6) Risample + rumore
        if row['sampl'] != '*':
            SpectrumResampler(savepath, final_name, row['sampl']).resample_and_save()
            if row['snr'] != '*':
                SpectrumNoiseAdder(
                    savepath, final_name, row['snr'], noise_type='GAUSS'
                ).add_noise_and_save()

        # 7) Pulizia linelist temporanei
        llmgr.delete_tmp_linelist(linelistpath, linespec, keyvec, kw)

        # 8) Rinomina finale per evitare collisioni
        unique = f"{final_name}_k{k}"
        # .spec
        old_spec = os.path.join(savepath, f"{final_name}.spec")
        if os.path.exists(old_spec):
            os.rename(old_spec, os.path.join(savepath, f"{unique}.spec"))
        # .hdr
        old_hdr = os.path.join(savepath, f"{final_name}.hdr")
        if os.path.exists(old_hdr):
            os.rename(old_hdr, os.path.join(savepath, f"{unique}.hdr"))
        # .com
        old_com = os.path.join(launchpath, f"{final_name}.com")
        if os.path.exists(old_com):
            os.rename(old_com, os.path.join(launchpath, f"{unique}.com"))

        return None

    except Exception as e:
        # qui restano solo errori davvero inattesi
        print(f"[Worker {k}] ERRORE inatteso: {e}", file=sys.stderr)
        return f"ERROR_{k}"

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
    for k in range(n_spec):
        row = df.iloc[k]
        if kws[0]=='False' and kws[1]=='True':
            T, g     = map(float, row['Model'].split(',')[:2])
            models   = mm.select_models_for_interpolation(
                int(T), g, row['[Fe/H]'], row['xi'], row['chemistry']
            )
            fname    = mm.write_interpolator(
                int(T), g, row['[Fe/H]'], row['chemistry'],
                model_dp, models
            )
        elif kws[0]=='False' and kws[1].lower()=='nearest':
            T, g  = map(float, row['Model'].split(',')[:2])
            fname = mm.select_nearest_model(
                int(T), g, row['[Fe/H]'], row['xi'],
                row['chemistry'], model_dp
            )
        else:
            fname0 = row['Model'].split(',')[0]
            subprocess.run(
                ['cp', f"{dataset_dp}{fname0}", model_dp],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            fname = fname0

        model_map[k] = fname
        #logger.info(f"> Modello per riga {k}: {fname}")

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
                      model_map, main_log)
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
                    model_map, main_log)
        results = [worker(0)]

    # Riepilogo finale
    errs = [r for r in results if r is not None]
    if errs:
        logger.error("Spettri non calcolati o senza header:")
        for fn in errs:
            logger.error(f"  - {fn}")
    else:
        logger.info("Tutti gli spettri e i relativi header sono stati creati con successo.")

    logger.info(f"Log completo: {main_log}")
