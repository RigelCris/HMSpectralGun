#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from os.path import exists
import re

class TurboSpecWriter:
    def __init__(self, save_path, linelist_path, model_path, launch_path ):
        self.model_path = model_path
        self.launch_path = launch_path
        self.save_path = save_path
        self.linelist_path = linelist_path

    def explicit_marcs(self, model_name):
        """
        Reads a model file and determines if the MARCS condition is true or false.

        Parameters:
        - model_name: str, name of the model file

        Returns:
        - str: '.true.' if the condition is met, '.false.' otherwise
        """
        model_file_path = os.path.join(self.model_path, model_name)
        model_content = np.genfromtxt(model_file_path, usecols=(0), delimiter=',', dtype='str', unpack=True)
        
        if model_content[0][1:12] == "sphINTERPOL":
            return '.false.'
        return '.true.'

    def check_and_create_contopacdir(self, path_to_dir):
        """
        Checks if a directory exists and creates it if it does not.

        Parameters:
        - path_to_dir: str, path to the directory
        """
        contopac_dir = os.path.join(path_to_dir, '../contopac')
        if not os.path.exists(contopac_dir):
            print("CONTOPAC directory does not exist. Creating it.")
            os.makedirs(contopac_dir)
        else:
            print("CONTOPAC directory already exists.")

    def writer(self, model_name, metallic, alpha, lam_min, lam_max, turbvel, linespec, lineseqw, elem, abu, isotopic_n, isotopic_val, keyw, el, ext, deltalam='0.01', interp='True', NLTE=False):
        """
        Writes a script for running the Turbospectrum software with specific parameters.

        Parameters:
        - model_name: str, name of the model file
        - metallic: float, metallicity value
        - alpha: float, alpha value
        - lam_min: float, minimum wavelength
        - lam_max: float, maximum wavelength
        - turbvel: float, turbulence velocity
        - linespec: list, line spectrum files
        - lineseqw: list, line equivalent width files
        - elem: list, element names
        - abu: list, abundances
        - isotopic_n: int, number of isotopic values
        - isotopic_val: float, isotopic value
        - keyw: str, keyword indicating special conditions
        - el: str, element symbol
        - ext: str, file extension
        - deltalam: str, delta lambda value
        - interp: str, interpolation method

        Returns:
        - str: Name of the script file created
        """
        if not isinstance(model_name, str):
            raise ValueError("model_name should be a string")

        # Determine the model name format using regex
        if interp.lower() in ('false'):
            match = re.match(r'([sp]\d+)_g([+-]?\d\.\d)_z([+-]?\d\.\d+)_a([+-]?\d\.\d+)\.mod$', model_name)
            if match:
                print('Spherical model') if match.group(1)[0] == 's' else print('Plane Parallel model')
                TT = match.group(1)[1:]
                gg = float(match.group(2))
                GG = f"{gg:.2f}"
                model_name_format = f't{TT}_g{GG}'
            else:
                print('Model not found')
                pass
        elif interp.lower() in ('true'):
            match = re.match(r'T(\d+)_G([pm]\d+\.\d+)_.*_Z([pm]\d+\.\d+)\.interpol', model_name)
            if match:
                TT = float(match.group(1))
                gg = float(match.group(2)[1:]) if match.group(2)[0] == 'p' else -float(match.group(2)[1:])
                GG = f"{gg:.2f}"
                model_name_format = model_name.split('Z')[0][:-4]
            else:
                raise ValueError(f"Invalid model name format: {model_name}")

        elif interp.lower() in ('nearest'):
#           print('Modello scritto co i piedi?')
#           model_name_format = model_name.split(".mod")[0]
            match = re.match(r'T(\d+)_G([pm]\d+\.\d+)_.*_Z([pm]\d+\.\d+)', model_name)
            if match:
                print('Spherical model') if match.group(1)[0] == 's' else print('Plane Parallel model')
                TT = match.group(1)[1:]
                gg = float(match.group(2)[1:]) if match.group(2)[0] == 'p' else -float(match.group(2)[1:])
                GG = f"{gg:.2f}"    
                model_name_format = model_name.split(".mod")[0] #f't{TT}_g{GG}'
            else:
                print('Model not found')
                pass


        # Determine spherical parameter
        spherical = 'T'
        ###########print('!!!OVERWRITTEN PP MODEL FOR TS BUG!!!')
        if (float(GG) >= 4):
             spherical = 'F'

        # Format metallicity and alpha values
        sgnm = 'm' if metallic < 0 else 'p'
        metallic_formatted = f"{abs(metallic):.2f}"
        sgna = 'm' if alpha < 0 else 'p'
        alpha_formatted = f"{abs(alpha):.2f}"

        # Format other parameters
        model_name, metallic, alpha = str(model_name), str(metallic), str(alpha)
        lam_minw, lam_maxw = str(int(float(lam_min))), str(int(float(lam_max)))
        lam_min, lam_max = str(int(float(lam_min)-1)), str(int(float(lam_max)+1))
        turbvel = f"{float(turbvel):.2f}"
        # Generate script name
        if (keyw.lower() == 'yes'):
            script_name = f"{model_name_format}_{lam_minw}_{lam_maxw}_xi{turbvel}_z{sgnm}{metallic_formatted}_a{sgna}{alpha_formatted}_only{el}.{ext}"
        else:
            script_name = f"{model_name_format}_{lam_minw}_{lam_maxw}_xi{turbvel}_z{sgnm}{metallic_formatted}_a{sgna}{alpha_formatted}.{ext}"

        # Check if the spectrum file already exists
        if exists(os.path.join(self.save_path, script_name)):
            print("Spectrum already exists")
            return 'STOP'

            
        # Write the script file
        with open(os.path.join(self.launch_path, script_name+'.com'), "w") as file:
            file.write("#!/bin/csh -f\n")
            file.write("\n")
            file.write("# Turbospectrum Script\n")
            file.write("\n")
            file.write("date\n")
            file.write(f"set spath = '{self.save_path}'\n")
            file.write(f"set mpath = '{self.model_path}'\n")
            file.write(f"set lpath = '{self.linelist_path}'\n")
            file.write(f"set dpath = '{self.launch_path}'\n")
            file.write("\n")
            file.write(f"foreach MODEL ({model_name})\n")
            file.write(f"set MODELn = '{model_name_format}'\n")
            file.write(f"set lam_min = '{lam_min}'\n")
            file.write(f"set lam_max = '{lam_max}'\n")
            file.write(f"set lam_minw = '{lam_minw}'\n")
            file.write(f"set lam_maxw = '{lam_maxw}'\n")
            file.write(f"set deltalam = '{deltalam}'\n")
            file.write(f"set METALLIC = '{metallic}'\n")
            file.write(f"set sgnm = '{sgnm}'\n")
            file.write(f"set METALLICn = '{metallic_formatted}'\n")
            file.write(f"set sgna = '{sgna}'\n")
            file.write(f"set alphan = '{alpha_formatted}'\n")
            file.write(f"set alpha = '{alpha}'\n")
            file.write(f"set TURBVEL = '{turbvel}'\n")
            file.write(f"set ext = '{ext}'\n")
            if keyw.lower() == 'yes':
                file.write(f"set xel = '{el}'\n")
                file.write(f"set SUFFIX = _${{lam_minw}}_${{lam_maxw}}_xi${{TURBVEL}}_z${{sgnm}}${{METALLICn}}_a${{sgna}}${{alphan}}_only${{xel}}\n")
            else:
                file.write(f"set SUFFIX = _${{lam_minw}}_${{lam_maxw}}_xi${{TURBVEL}}_z${{sgnm}}${{METALLICn}}_a${{sgna}}${{alphan}}\n")
            file.write(f"set result = ${{MODELn}}${{SUFFIX}}\n")
            file.write("\n")
            file.write("# Abundances from the model are not used\n")
            file.write("\n")
            file.write("time /Users/cfanelli/astro/softw/TS-NLTE/exec-gf/babsma_lu <<EOF\n")
            file.write(f"'LAMBDA_MIN:'  '${{lam_min}}'\n")
            file.write(f"'LAMBDA_MAX:'  '${{lam_max}}'\n")
            file.write(f"'LAMBDA_STEP:' '${{deltalam}}'\n")
            file.write(f"'MODELINPUT:' '$mpath/${{MODEL}}'\n")
            marcs_condition = '.false.' if ('interpol' in model_name or 'kur' in model_name) else '.true.'
            file.write(f"'MARCS-FILE:' '{marcs_condition}'\n")
            file.write("'MODELOPAC:' '/Users/cfanelli/astro/softw/TS-NLTE/COM/contopac/${MODEL}.opac'\n")
            file.write("'ABUND_SOURCE:' 'magg'\n")
            file.write(f"'METALLICITY:'    '${{METALLIC}}'\n")
            file.write(f"'ALPHA/Fe   :'    '{alpha}'\n")
            file.write("'HELIUM     :'    '0.00'\n")
            file.write("'R-PROCESS  :'    '0.00'\n")
            file.write("'S-PROCESS  :'    '0.00'\n")
            if elem[0] == 0:
                file.write(f"'INDIVIDUAL ABUNDANCES:'   '0'\n")
            else:
                file.write(f"'INDIVIDUAL ABUNDANCES:'   '{len(abu)}'\n")
                for element, abundance in zip(elem, abu):
                    file.write(f"{int(element)}  {abundance:4.2f}\n")
            file.write("'XIFIX:' 'T'\n")
            file.write("$TURBVEL\n")
            file.write("EOF\n")
            file.write("\n")
            file.write("########################################################################\n")
            file.write("\n")
            file.write("time /Users/cfanelli/astro/softw/TS-NLTE/exec-gf/bsyn_lu <<EOF\n")
            file.write("'NLTE :'          '.false.'\n")
            file.write("'NLTEINFOFILE:'  '../DATA/SPECIES_LTE_NLTE.dat'\n")
            file.write("#'SEGMENTSFILE:'     '${dpath}/segfile.txt'\n")
            file.write("#'RESOLUTION:'     '300000.'\n")
            file.write(f"'LAMBDA_MIN:'  '${{lam_min}}'\n")
            file.write(f"'LAMBDA_MAX:'  '${{lam_max}}'\n")
            file.write(f"'LAMBDA_STEP:' '${{deltalam}}'\n")
            file.write("'INTENSITY/FLUX:' 'Flux'\n")
            file.write("'ABFIND        :' '.false.'\n")
            file.write(f"'MODELOPAC:' '/Users/cfanelli/astro/softw/TS-NLTE/COM/contopac/${{MODEL}}.opac'\n")
            file.write(f"'RESULTFILE :' '$spath/${{result}}.${{ext}}'\n")
            file.write(f"'METALLICITY:'    '${{METALLIC}}'\n")
            file.write(f"'ALPHA/Fe   :'    '{alpha}'\n")
            file.write("'HELIUM     :'    '0.00'\n")
            file.write("'R-PROCESS  :'    '0.00'\n")
            file.write("'S-PROCESS  :'    '0.00'\n")
            if elem[0] == 0:
                file.write(f"'INDIVIDUAL ABUNDANCES:'   '0'\n")
            else:
                file.write(f"'INDIVIDUAL ABUNDANCES:'   '{len(abu)}'\n")
                for element, abundance in zip(elem, abu):
                    file.write(f"{int(element)}  {abundance:4.2f}\n")
            if isotopic_n != 0:
                C13 = 1 / (isotopic_val + 1)
                C12 = isotopic_val * C13
                file.write("'ISOTOPES : ' '2' \n")
                file.write(f"6.012  {C12:5.3f} \n")
                file.write(f"6.013  {C13:5.3f} \n")
            else:
                file.write("'ISOTOPES : ' '0'\n")
            file.write(f"'NFILES   :' '{len(linespec)}'\n")
            for line_file in linespec:
                file.write(f"$lpath/{line_file}\n")
            file.write(f"'SPHERICAL:'  '{spherical}' \n")
            file.write("  30\n")
            file.write("  300.00\n")
            file.write("  15\n")
            file.write("  1.30\n")
            file.write("EOF\n")
            file.write("########################################################################\n")
    

        os.system(f"chmod 777 {os.path.join(self.launch_path, script_name+'.com')}")
        
        return script_name

"""
filename = '/Users/cfanelli/astro/softw/Turbospectrum2019-master/COM-v19.1/santerre/input.ts'

params = InputParameters('input.ts')
params.decisor_model(params.get_keywords()[1])
save_path, linelist_path, model_path = params.get_paths()

launch_path = '/Users/cfanelli/astro/softw/Turbospectrum2019-master/COM-v19.1/santerre/'

turbo_spec = TurboSpec_writer(model_path, launch_path, save_path, linelist_path)

df = params.get_dataframe()

elem = [0]
abu = 0
isotopic_n = [0]
isotopic_val = 15


k = 0
#turbo_spec.check_and_create_contopacdir(path_to_dir)
x = turbo_spec.turbospec_writer(df.at[k, 'Model'],df.at[k, '[Fe/H]'], df.at[k, '[a/Fe]'],df.at[k, 'lam_i'], df.at[k, 'lam_f'], 
                            df.at[k, 'xi'], df.at[k, 'linelist_file'], df.at[k, 'eqw_file'], elem, abu, isotopic_n, isotopic_val, keyw='no', el='*', ext='txt', deltalam='0.01', interp='True', NLTE=True)
"""