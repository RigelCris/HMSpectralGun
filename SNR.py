#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from io import StringIO
import numpy as np

class SpectrumNoiseAdder:
    def __init__(self, savepath, input_file, snr, noise_type='GAUSS'):
        self.input_file = input_file
        self.savepath = savepath
        self.snr = snr
        self.noise_type = noise_type.upper()
        self.header = []
        self.data = None
        self._read_file()

    def _read_file(self):
        with open(self.savepath+self.input_file, 'r') as file:
            lines = file.readlines()
        
        header_lines = []
        data_lines = []
        for line in lines:
            if line.startswith('#'):
                header_lines.append(line)
            else:
                data_lines.append(line)
        
        self.header = header_lines
        self.data = pd.read_csv(StringIO(''.join(data_lines)), sep='\s+', header=None)

    def _add_noise(self):
        for col in [1, 2]:
            if self.noise_type == 'GAUSS':
                noise = np.random.normal(0, self.data[col].astype(float) / float(self.snr), self.data[col].shape)
            elif self.noise_type == 'POISSON':
                noise = np.random.poisson(self.data[col].astype(float) / float(self.snr), self.data[col].shape) - (self.data[col] / self.snr)
            self.data[col] += noise
        
    def _generate_output_filename(self):
        base, ext = self.input_file.rsplit('.', 1)
        return f"{base}_SNR{self.snr}.{ext}"

    def _write_file(self, output_file):
        with open(self.savepath+output_file, 'w') as file:
            k = 0
            for line in self.header:
                if k ==10:
                    file.write(f"# {self.noise_type} SNR     =  {self.snr}\n")
                file.write(line)
                k += 1
            self.data.to_csv(file, sep='\t', index=False, header=False, float_format='%.6f')

    def add_noise_and_save(self):
        self._add_noise()
        output_file = self._generate_output_filename()
        self._write_file(output_file)
        return output_file

"""
# Usage example
input_file = '/Users/cristiano.fanelli/ASTRO/softw/TS-NLTE/COM/syntspec//T4000_Gp1.00_st_Zp0.00_15000_15500_xi2.00_zp0.00_ap0.00_R20k.txt'
snr = 50
noise_type = 'GAUSS'

spectrum_noise_adder = SpectrumNoiseAdder(input_file, snr, noise_type)
output_file = spectrum_noise_adder.add_noise_and_save()
"""