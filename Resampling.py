#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from io import StringIO
import os

class SpectrumResampler:
    def __init__(self, savepath, input_file, resampl):
        self.savepath = savepath
        self.input_file = input_file
        self.resampl = resampl
        self.header = []
        self.data = None
        self._read_file()

    def _read_file(self):
        with open(os.path.join(self.savepath, self.input_file), 'r') as file:
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

    def _add_resampl_to_header(self):
        self.header.append(f"# resampl         = {self.resampl}\n")

    def _resample_data(self):
        wavelength = self.data[0].values
        normflux = self.data[1].values
        flux = self.data[2].values
        
        min_wavelength = wavelength.min()
        max_wavelength = wavelength.max()
        new_wavelength = np.arange(min_wavelength, max_wavelength+float(self.resampl), float(self.resampl))
        
        normflux_interpolator = interp1d(wavelength, normflux, kind='linear', fill_value="extrapolate")
        flux_interpolator = interp1d(wavelength, flux, kind='linear', fill_value="extrapolate")
        
        new_normflux = normflux_interpolator(new_wavelength)
        new_flux = flux_interpolator(new_wavelength)
        
        self.data = pd.DataFrame({
            0: new_wavelength,
            1: new_normflux,
            2: new_flux
        })

    #def _generate_output_filename(self):
    #    base, ext = self.input_file.rsplit('.', 1)
    #   return f"{base}_resampl{self.resampl}.{ext}"

    def _write_file(self, output_file):
        with open(output_file, 'w') as file:
            for line in self.header:
                file.write(line)
            self.data.to_csv(file, sep='\t', index=False, header=False, float_format='%.6f')

    def resample_and_save(self):
        #self._add_resampl_to_header()
        self._resample_data()
        #output_file = self._generate_output_filename()
        output_file = self.input_file
        self._write_file(os.path.join(self.savepath, output_file))
        return output_file

# Utilizzo della classe
#input_file = '/Users/cristiano.fanelli/ASTRO/softw/TS-NLTE/COM/syntspec/T7000_Gp4.00_st_Zp0.00_15500_15700_xi1.00_zp0.00_ap0.00_R3k.txt'
#resampl = 0.234  # Valore di esempio per il ricampionamento

#resampler = SpectrumResampler(input_file, resampl)
#output_file = resampler.resample_and_save()
#print(f"File salvato: {output_file}")
