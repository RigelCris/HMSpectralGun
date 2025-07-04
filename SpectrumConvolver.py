#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from PyAstronomy import pyasl
import time

class SpectrumConvolver:
    """
    Classe per gestire la convoluzione gaussiana degli spettri.

    Attributi:
        save_path (str): Il percorso in cui salvare i file convoluti.
    """

    def __init__(self, save_path):
        """
        Inizializza la classe con i percorsi forniti.

        Args:
            save_path (str): Il percorso in cui salvare i file convoluti.
        """
        self.save_path = save_path

    def instrbroad(self, input_file, res, verbose='yes'):
        """
        Esegue la convoluzione gaussiana utilizzando PyAstronomy.

        Args:
            input_file (str): Il nome del file di input.
            res (int): La risoluzione spettrale.
            verbose (str): Se 'yes', stampa i messaggi di log.

        Returns:
            str: Il nome del file di output.
        """
        res = int(res)
        output_file = '.'.join(input_file.split('.')[:-1]) + '_R' + str(int(res/1000)) + 'k.' + input_file.split('.')[-1]
        w, fn, f = np.genfromtxt(os.path.join(self.save_path, input_file), usecols=(0, 1, 2), unpack=True)
        start = time.time()
        fnd, fwhm = pyasl.instrBroadGaussFast(w, fn, res, edgeHandling=None, fullout=True)
        fd, fwhm = pyasl.instrBroadGaussFast(w, f, res, edgeHandling=None, fullout=True)
        end = time.time()

        if verbose != 'no':
            tt = end - start
            key = 'min' if tt > 60 else 'sec'
            tt = tt / 60 if tt > 60 else tt
            print('Time for gaussian convolution:', round(tt, 2), key)
            print("FWHM used for the Gaussian kernel:", round(fwhm, 3), "A")

        with open(os.path.join(self.save_path, output_file), "w") as file:
            file.write('#wvl           normflux      flux  \n')
            for val in zip(w, fnd, fd):
                file.write(' {:=9.3f}     {:=2.4f}        {:=5.2f}   \n'.format(val[0], val[1], val[2]))
            file.close()
        return output_file


    def rotbroad(self, input_file, vsini, epsilon=0.6, verbose='yes'):
        """
        Esegue la convoluzione rotazionale utilizzando pyasl.fastRotBroad.

        Args:
            input_file (str): Il nome del file di input.
            vsini (float): La velocità di rotazione proiettata (in km/s).
            epsilon (float): Coefficiente di oscuramento al bordo.
            verbose (str): Se 'yes', stampa i messaggi di log.

        Returns:
            str: Il nome del file di output.
        """
        output_file = '.'.join(input_file.split('.')[:-1]) + '_ROT' + str(int(vsini)) + 'k.' + input_file.split('.')[-1]   
        w, fn, f = np.genfromtxt(os.path.join(self.save_path, input_file), usecols=(0, 1, 2), unpack=True)
        start = time.time()
        fnd = pyasl.fastRotBroad(w, fn, epsilon, vsini)
        fd = pyasl.fastRotBroad(w, f, epsilon, vsini)
        end = time.time()

        if verbose != 'no':
            tt = end - start
            key = 'min' if tt > 60 else 'sec'
            tt = tt / 60 if tt > 60 else tt
            print('Time for rotational convolution:', round(tt, 2), key)


        with open(os.path.join(self.save_path, output_file), "w") as file:
            file.write('#wvl           normflux      flux  \n')
            for val in zip(w, fnd, fd):
                file.write(' {:=9.3f}     {:=2.4f}        {:=5.2f}   \n'.format(val[0], val[1], val[2]))

        return output_file


"""
# Esempio di utilizzo
save_path = '/Users/cristiano.fanelli/Downloads/'

convolver = SpectrumConvolver(save_path)

# Esempio di convoluzione con PyAstronomy
convolver.rotbroad('t4000_gp100.00_15500_16000_xi2.00_zm1.00_ap0.00.txt', 30)

"""