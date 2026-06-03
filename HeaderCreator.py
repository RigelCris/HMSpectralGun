#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
from os.path import exists

class HeaderCreator:
    def __init__(self, launch_path, save_path):
        self.launch_path = launch_path
        self.save_path = save_path

    def _chemical_keyword(self, keyw):
        keywords = {
            'st': 'standard',
            'ap': 'alpha poor',
            'ae': 'alpha enhanced',
            'mc': 'moderately cycled',
            'hc': 'heavily cycled'
        }
        return keywords.get(keyw, keyw)

    def _prepare_header_lines(self, model, chem, met, alpha, xi, deltalam, lam_min, lam_max, isotopic_n, isotopic_val, res=None):
        header_lines = [
            f"# model         =  {model} \n",
            f"# mixture       =  {chem} \n",
            f"# [Fe/H]        =  {met} dex \n",
            f"# [a/Fe]        =  {alpha} dex \n",
            f"# xi            =  {xi} km/s \n",
            f"# sampl         =  {deltalam} \n",
            f"# lam_min       =  {int(lam_min-1)} Å\n",
            f"# lam_max       =  {int(lam_max+1)} Å\n",
            f"# C12/C13       =  {isotopic_val if isotopic_n != 89 else 'solar'} \n"
        ]
        if res is not None:
            header_lines.insert(6, f"# res           =  {res} \n")
        return header_lines

    def _modify_abundances(self, elem, deltaabu, alpha):
        #if alpha != 0:
        #    alpha_elements = [8, 12, 14, 16, 20, 22]
        #    for i, atomic_number in enumerate(elem):
        #        if atomic_number in alpha_elements:
        #            deltaabu[i-3] += alpha    #i-3 perchè il conteggio parte dalle 3 righe aggiunte
        abun = deltaabu.astype(object)
        abun = np.insert(abun, 0, '[X/Fe]')
        abun = np.insert(abun, 1, ' 0.0')
        abun = np.insert(abun, 2, '            INACTIVE (TODO)')
        return abun

    def _process_header_lines(self, lines, abun):
        start_line_content = " ELEMENT  ATOMIC NUMBER   LOG(ABUNDANCE)"
        try:
            start_line_index = next(i for i, line in enumerate(lines) if start_line_content in line)
        except StopIteration as exc:
            raise ValueError("Unable to find abundance block in Turbospectrum log output.") from exc
        end_line_index = start_line_index + 83  # Regola in base alla struttura del file

        header_lines = lines[start_line_index:end_line_index + 1]
        processed_header_lines = []
        for i, line in enumerate(header_lines):
            columns = line.split()
            element_name = '{:<6}'.format(columns[0])
            atomic_number = '{:>9}'.format(columns[1])
            logN_value = '{:>12}'.format(columns[2])
            value = '{:>15}'.format(abun[i])
            new_line = f"#    {element_name}{atomic_number}{logN_value}{value}\n"
            processed_header_lines.append(new_line)
        return processed_header_lines

    def create_combined_header(
        self,
        namefile,
        model,
        met,
        alpha,
        elem,
        deltaabu,
        lam_min,
        lam_max,
        xi,
        isotopic_n,
        isotopic_val,
        keyw,
        deltalam,
        res=None,
        log_filename='log'
    ):
        chem = self._chemical_keyword(keyw)
        header_lines = self._prepare_header_lines(model, chem, met, alpha, xi, deltalam, lam_min, lam_max, isotopic_n, isotopic_val, res)

        # Leggi il contenuto del file di log
        log_path = os.path.join(self.launch_path, log_filename)
        with open(log_path, 'r') as file:
            lines = file.readlines()

        abun = self._modify_abundances(elem, deltaabu, alpha)
        processed_header_lines = self._process_header_lines(lines, abun)

        combined_content = header_lines + processed_header_lines
        new_content = "".join(combined_content)

        # Leggi il contenuto originale del file
        with open(os.path.join(self.save_path, namefile), 'r') as file:
            original_content = file.read()

        # Scrivi il nuovo contenuto nel file
        with open(os.path.join(self.save_path, namefile), 'w') as file:
            file.write(new_content)
            file.write('# \n')
            file.write('# \n')
            file.write('# \n')
            #file.write('#wvl           normflux      flux   \n')
            file.write(original_content)



"""
# Esempio di utilizzo della classe
launch_path = '/Users/cristiano.fanelli/ASTRO/softw/TS-NLTE/COM/'
save_path = '/Users/cristiano.fanelli/ASTRO/softw/TS-NLTE/COM/syntspec/'
creator = HeaderCreator(launch_path, save_path)


# Utilizzo di create_combined_header
creator.create_combined_header(
    namefile='s5000_g+2.0_m1.0_t02_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00_15000_15500_xi2.00_zp0.00_ap0.00_R20k.txt',
    model='example_model',
    met=0.0,
    alpha=0.2,
    elem = np.arange(90),
    deltaabu = np.zeros(90),
    lam_min=4000,
    lam_max=7000,
    xi=1.5,
    isotopic_n=12,
    isotopic_val=90,
    keyw='st',
    deltalam=0.01,
    res=20000
)
"""
