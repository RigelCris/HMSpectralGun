#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import numpy as np
from tqdm import tqdm
from mendeleev import element as chemical_element

# Importazione delle classi necessarie dal modulo HMSpectralGun
try:
    from HMSpectralGun.Intro             import Intro
    from HMSpectralGun.InputParameters   import InputParameters
    from HMSpectralGun.LineListManager   import LineListManager
    from HMSpectralGun.ModelMaker        import ModelMaker
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
    from ModelMaker        import ModelMaker
    from TurboSpecWriter   import TurboSpecWriter
    from HeaderCreator     import HeaderCreator
    from SpectrumConvolver import SpectrumConvolver
    from SNR               import SpectrumNoiseAdder
    from Resampling        import SpectrumResampler



def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Process some inputs')
    parser.add_argument('--input', type=str, default='input.ts', help='The input filename (default: input.ts)')
    parser.add_argument('--progress', action='store_true', help='Show a progress bar')
    return parser.parse_args()

def _read_explicit_xfe_override(row):
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

def main(k=0, show_progress=False, verbose=False, pbar=None):
    """Main function to process spectra based on input parameters."""
    
    # Inizializzazione delle classi con i rispettivi parametri
    linelist_manager       =   LineListManager()  # Assumendo che non richieda parametri
    model_maker            =   ModelMaker(dataset_model_path)
    header                 =   HeaderCreator(launchpath, savepath)
    spectrum_convolver     =   SpectrumConvolver(savepath)  # Assumendo che non richieda parametri

    explicit_model, interp, NLTE = params.get_keywords()
    
    df = params.get_dataframe()

    # Scelta del modello basata sui parametri espliciti e sull'interpolazione
    if (explicit_model == 'False') & (interp == 'True'):
        param = df.at[k, 'Model'].split(",")
        Teff, logg = int(param[0]), float(param[1])
        models = model_maker.select_models_for_interpolation(Teff, logg, df.at[k, '[Fe/H]'], df.at[k, 'xi'], df.at[k, 'chemistry'])
        model = model_maker.write_interpolator(Teff, logg, df.at[k, '[Fe/H]'], df.at[k, 'chemistry'], modelpath, models)
    elif (explicit_model == 'False') & (interp.lower() == "nearest"):
        param = df.at[k, 'Model'].split(",")
        Teff, logg = int(param[0]), float(param[1])
        model = model_maker.select_nearest_model(Teff, logg, df.at[k, '[Fe/H]'], df.at[k, 'xi'], df.at[k, 'chemistry'], modelpath)
    elif (explicit_model == 'True') & (interp == 'False'):
        model = df.at[k, 'Model'].split(",")[0]
        os.system('cp '+dataset_model_path+model+' '+modelpath)

    dict_linelist = params.process_solar_references(df.at[k, 'linelist_file'], df.at[k, 'abu_file'], df.at[k, '[Fe/H]'])
    linespec = dict_linelist['linelist_content']
    Crat = dict_linelist['Crat']
    isotopic_n = 612613

    # Gestione degli elementi singoli
    keyword, keyvec = params.read_single_element(df.at[k, 'monoelem'])
    if keyword == 'Yes':
        linespec = linelist_manager.create_linelist_single_element(linelistpath, linespec, keyvec)

    # Lettura dei file di abbondanza
    df_abu, _ = params.read_abu_file(df.at[k, 'abu_file'])
    elem, deltaabu = df_abu['AtomicNumber'].to_numpy(dtype=float), df_abu['AbundanceDifference'].to_numpy(dtype=float)
    override_z, override_xfe, override_tag = _read_explicit_xfe_override(df.iloc[k])
    if Crat is None or len(Crat) == 0:
        isotopic_n = 0
        isotopic_val = 0.0
    else:
        isotopic_val = Crat.iloc[0]

    # Aggiornamento delle abbondanze in base a [a/Fe]
    if df.at[k, '[a/Fe]'] != 0.:
        alpha = df.at[k, '[a/Fe]']
        for element in [8, 12, 14, 16, 20, 22]:
            deltaabu[elem == element] = alpha

    # Priorità massima: override esplicito [X/Fe] da input.ts
    if override_z is not None:
        mask = (elem == float(override_z))
        if mask.any():
            deltaabu[mask] = override_xfe
        else:
            elem = np.append(elem, float(override_z))
            deltaabu = np.append(deltaabu, float(override_xfe))

    abu = params.solar_reference(elem, deltaabu, df.at[k, '[Fe/H]'])

    # Computazione dello spettro
    start = time.time()
    namefile = turbo_spec_writer.writer(model, df.at[k, '[Fe/H]'], df.at[k, '[a/Fe]'], df.at[k, 'lam_i'], df.at[k, 'lam_f'],
                                        df.at[k, 'xi'], linespec, df.at[k, 'snr'], elem, abu, isotopic_n,
                                        isotopic_val, keyw=keyword, el=keyvec[2], ext=df.at[k, 'extension'], deltalam=df.at[k, 'resnum'],
                                        interp=interp, NLTE=NLTE, abundance_tag=override_tag)
    if namefile != 'STOP':
        if verbose:
            print("Computing spectrum:", namefile)
        os.system(launchpath + namefile+'.com > ' + launchpath + 'log')
        header.create_combined_header(namefile, model=model, met=df.at[k, '[Fe/H]'], alpha=df.at[k, '[a/Fe]'], elem=elem,
                                      deltaabu=deltaabu, lam_min=df.at[k, 'lam_i'], lam_max=df.at[k, 'lam_f'],
                                      xi=df.at[k, 'xi'], isotopic_n=isotopic_n, isotopic_val=isotopic_val,
                                      keyw=df.at[k, 'chemistry'], deltalam=df.at[k, 'resnum'])
        end = time.time()
        if verbose:
            params.print_time(end - start)

        # Convoluzione dello spettro
        try:
            namefile = spectrum_convolver.instrbroad(namefile, df.at[k, 'RES'], verbose='no')
            header.create_combined_header(namefile, model=model, met=df.at[k, '[Fe/H]'], alpha=df.at[k, '[a/Fe]'], elem=elem,
                                          deltaabu=deltaabu, lam_min=df.at[k, 'lam_i'], lam_max=df.at[k, 'lam_f'],
                                          xi=df.at[k, 'xi'], isotopic_n=isotopic_n, isotopic_val=isotopic_val,
                                          keyw=df.at[k, 'chemistry'], deltalam=df.at[k, 'sampl'], res=df.at[k, 'RES'])
        except (ValueError, FileNotFoundError):
            if verbose:
                print('ERROR IN COMPUTING SPECTRUM!!!!!!')
            problem_list.append(namefile)
            os.system('rm ' + savepath + namefile + '.spec')
            pass

        linelist_manager.delete_tmp_linelist(linelistpath, linespec, keyvec, keyword)
        
        if (df.at[k, 'sampl']!='*') & (df.at[k, 'snr']=='*'):
            resampler = SpectrumResampler(savepath, namefile, df.at[k, 'sampl'])
            resampler.resample_and_save()
            print('Spectrum resampled at ', df.at[k, 'sampl'], 'Å')

        elif (df.at[k, 'sampl']!='*') & (df.at[k, 'snr']!='*'):
            resampler = SpectrumResampler(savepath, namefile, df.at[k, 'sampl'])
            resampler.resample_and_save()
            spectrum_noise_adder   =   SpectrumNoiseAdder(savepath, namefile, df.at[k, 'snr'], noise_type='GAUSS')
            namefile = spectrum_noise_adder.add_noise_and_save()            
            print('Spectrum resampled at ', df.at[k, 'sampl'], 'Å')
            print('Added noise for SNR =', df.at[k, 'snr'])
        else:
            pass
    elif namefile == 'STOP':
        if verbose:
            print('STOP')

    if pbar:
        pbar.update(1)

# Parsing degli argomenti e inizializzazione delle variabili globali
args = parse_arguments()
current_path = os.getcwd()
input_file = os.path.join(current_path, args.input)
dataset_model_path = '/Users/cfanelli/astro/softw/TS-NLTE/COM/santerre/HMSpectralGun/marcs_generator/dataset/'
launchpath = '/Users/cfanelli/astro/softw/TS-NLTE/COM/'
params = InputParameters(input_file)
savepath, linelistpath, modelpath = params.get_paths()
turbo_spec_writer = TurboSpecWriter(savepath, linelistpath, modelpath, launchpath)

# Introduzione e stampa dei percorsi
Intro.intro1()
print('\n***  List of Paths  ***')
print(f"Save Path: {savepath}")
print(f"Linelist Path: {linelistpath}")
print(f"Model Path: {modelpath}")
print(f"Launch Path: {launchpath}")
print(f"Current Path: {current_path}")

if __name__ == '__main__':
    print("\n****************************************************************\n")
    number_of_spec = params.write_output_spectra_count()
    print("****************************************************************\n")

    problem_list = []
    turbo_spec_writer.check_and_create_contopacdir(modelpath)

    # Elaborazione degli spettri
    if number_of_spec == 1:
        main(show_progress=args.progress, verbose=not args.progress)
    elif number_of_spec > 1:
        if args.progress:
            with tqdm(total=number_of_spec, desc="Processing Spectra") as pbar:
                for nn in range(number_of_spec):
                    main(k=nn, show_progress=args.progress, verbose=not args.progress, pbar=pbar)
        else:
            for nn in range(number_of_spec):
                main(k=nn, show_progress=args.progress, verbose=not args.progress)
                print()

    # Stampa dei problemi riscontrati
    if problem_list and not args.progress:
        print("\n\nSpettri non calcolati per problemi vari:")
        for problem in problem_list:
            print(problem)
        print("\n\n")
    
