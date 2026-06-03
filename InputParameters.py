#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from pathlib import Path
from mendeleev import element
from collections import Counter
from os.path import exists
import pandas as pd


class InputParameters:
    """
    Classe per gestire i parametri di input per la sintesi degli spettri.

    Attributi:
        filename (str): Il nome del file di input.
        savepath (str): Il percorso di salvataggio.
        linelistpath (str): Il percorso del file della lista di righe.
        modelpath (str): Il percorso del modello.
        keyword1 (str): Prima parola chiave.
        keyword2 (str): Seconda parola chiave.
        keyword3 (str): Terza parola chiave.
        df (pd.DataFrame): DataFrame che contiene i dati di input.
        linelist_file_content (list): Contenuto del file della lista di righe.
        abu_file_content (pd.DataFrame): Contenuto del file delle abbondanze.

    Metodi:
        __init__(filename):
            Inizializza la classe con il file di input.
        decisor_model(interp):
            Stampa il tipo di modello di atmosfera da usare in base all'input.
        read_file():
            Legge il file di input e popola gli attributi della classe.
        read_linelist_file(linelist_file):
            Legge il contenuto del file della lista di righe.
        read_abu_file(abu_file):
            Legge il contenuto del file delle abbondanze e estrae Crat.
        get_paths():
            Restituisce i percorsi di salvataggio, del file della lista di righe e del modello.
        get_keywords():
            Restituisce le due parole chiave.
        get_dataframe():
            Restituisce il DataFrame dei dati di input.
        get_linelist_content():
            Restituisce il contenuto del file della lista di righe.
        solar_reference(elem, abun, met):
            Calcola le abbondanze scalate rispetto ai valori solari.
        process_solar_references():
            Processa le referenze solari per ogni riga del DataFrame.
        write_output_spectra_count():
            Stampa il numero di spettri da sintetizzare.
    """

    def __init__(self, filename):
        """
        Inizializza la classe con il file di input.

        Args:
            filename (str): Il nome del file di input.
        """
        self.filename = str(Path(filename).expanduser().resolve())
        self.base_dir = str(Path(self.filename).parent)
        self.savepath = None
        self.linelistpath = None
        self.modelpath = None
        self.keyword1 = None
        self.keyword2 = None
        self.keyword3 = None
        self.df = None
        self.linelist_file_content = None
        self.abu_file_content = None
        self.read_file()

    def _resolve_aux_path(self, aux_file):
        """
        Resolve auxiliary file paths in a robust way:
        1) absolute paths are used as-is
        2) relative paths are resolved against input.ts directory
        3) fallback to current working directory for backward compatibility
        """
        aux_path = Path(str(aux_file)).expanduser()
        if aux_path.is_absolute():
            return str(aux_path)

        candidate_from_input = Path(self.base_dir) / aux_path
        if candidate_from_input.exists():
            return str(candidate_from_input.resolve())

        candidate_from_cwd = Path(os.getcwd()) / aux_path
        return str(candidate_from_cwd.resolve())

    def decisor_model(self, interp):
        """
        Stampa il tipo di modello di atmosfera da usare in base all'input.

        Args:
            interp (str): Indicatore del tipo di modello (False, True, nearest).
        """
        if interp == "False":
            print()
            print("Use of a precompilated Model Atmosphere ")
            print()
        elif interp == "True":
            print()
            print("Model Atmosphere Interpolation activated ")
            print()
        elif interp.lower() == "nearest":
            print()
            print("Use the nearest Model Atmosphere ")
            print()

    def read_file(self):
        """
        Legge il file di input e popola gli attributi della classe.
        """
        with open(self.filename, 'r') as file:
            lines = file.readlines()

        # Rimuovi righe di commento
        lines = [line.strip() for line in lines if not line.startswith('#')]

        # Estrai i percorsi
        self.savepath = lines[0].split('\t')[0]
        self.linelistpath = lines[1].split('\t')[0]
        self.modelpath = lines[2].split('\t')[0]

        # Estrai le keyword
        self.keyword1 = lines[3].split('\t')[0].split('=')[1]     # Explicit model True/False
        self.keyword2 = lines[4].split('\t')[0].split('=')[1]     # interp model True/False/Nearest
        self.keyword3 = lines[5].split('\t')[0].split('=')[1]     # interp model True/False/Nearest

        # Leggi il resto delle righe come dati tabellari
        data_lines = lines[6:]

        data = []
        for line in data_lines:
            values = line.split()
            # Formato storico: 15 colonne
            # Nuovo formato: 17 colonne con override esplicito [X/Fe]
            # ... extension override_elem override_xfe
            if len(values) not in (15, 17):
                raise ValueError(
                    f"Expected 15 or 17 columns, but got {len(values)} columns in line: {line}"
                )

            def cast_value(value, col_type):
                value = value.strip()
                if value == '*':
                    return value
                return col_type(value)

            # Colonne base (15)
            parsed_values = [
                cast_value(values[0], str),    # Model
                cast_value(values[1], float),  # [Fe/H]
                cast_value(values[2], float),  # [a/Fe]
                cast_value(values[3], float),  # lam_i
                cast_value(values[4], float),  # lam_f
                cast_value(values[5], float),  # xi
                cast_value(values[6], str),    # chemistry
                cast_value(values[7], str),    # sampl
                cast_value(values[8], float),  # RES
                cast_value(values[9], float),  # resnum
                cast_value(values[10], str),   # monoelem
                cast_value(values[11], str),   # linelist_file
                cast_value(values[12], str),   # abu_file
                cast_value(values[13], str),   # snr
                cast_value(values[14], str),   # extension
            ]

            # Override opzionale [X/Fe]: atomic number + value
            if len(values) == 17:
                parsed_values.extend([
                    cast_value(values[15], float),  # override_elem
                    cast_value(values[16], float),  # override_xfe
                ])
            else:
                parsed_values.extend(['*', '*'])

            data.append(parsed_values)

        # Crea un DataFrame
        columns = [
            "Model", "[Fe/H]", "[a/Fe]", "lam_i", "lam_f", "xi", "chemistry",
            "sampl", "RES", "resnum", "monoelem", "linelist_file", "abu_file",
            "snr", "extension", "override_elem", "override_xfe"
        ]
        self.df = pd.DataFrame(data, columns=columns)

    def read_linelist_file(self, linelist_file):
        """
        Legge il contenuto del file della lista di righe.

        Args:
            linelist_file (str): Il nome del file della lista di righe.

        Returns:
            list: Contenuto del file della lista di righe.
        """
        linelist_file_path = self._resolve_aux_path(linelist_file)
        with open(linelist_file_path, 'r') as file:
            return file.readlines()

    def read_abu_file(self, abu_file):
        """
        Legge il contenuto del file delle abbondanze e estrae Crat.

        Args:
            abu_file (str): Il nome del file delle abbondanze.

        Returns:
            tuple: DataFrame delle abbondanze senza Crat, e Crat se esiste.
        """
        abu_file_path = self._resolve_aux_path(abu_file)
        abu_data = pd.read_csv(abu_file_path, sep=r'\s+', header=None, names=["AtomicNumber", "AbundanceDifference"])
        # Estrai Crat se esiste
        Crat = abu_data[abu_data["AtomicNumber"] == 612613]["AbundanceDifference"]
        # Rimuovi Crat dal DataFrame se esiste
        abu_data = abu_data[abu_data["AtomicNumber"] != 612613]
        return abu_data, Crat

    def read_single_element(self, single_element):

        keyword = 'No'
        keyvec = ['*','*','*']

        if single_element == '*':
            pass
        elif (single_element == 'CO') | (single_element == 'OH') | (single_element == 'CN'):
            keyword = 'Yes'
            keyvec = ['*','*',single_element]
        else:
            keyword = 'Yes'
            n_ion = Counter(single_element)
            ion = n_ion["I"]
            se0 = single_element[:len(single_element)-ion]
            #se1 = single_element[len(single_element)-ion:]
            el = element(se0)

            keyvec[0] = str(el.atomic_number)+'.000'
            keyvec[1] = str(ion)
            keyvec[2] = single_element

        return keyword, keyvec


    def get_paths(self):
        """
        Restituisce i percorsi di salvataggio, del file della lista di righe e del modello.

        Returns:
            tuple: Percorsi di salvataggio, del file della lista di righe e del modello.
        """
        return self.savepath, self.linelistpath, self.modelpath

    def get_keywords(self):
        """
        Restituisce le due parole chiave.

        Returns:
            tuple: Le due parole chiave.
        """
        return self.keyword1, self.keyword2, self.keyword3

    def get_dataframe(self):
        """
        Restituisce il DataFrame dei dati di input.

        Returns:
            pd.DataFrame: DataFrame dei dati di input.
        """
        return self.df

    def get_linelist_content(self):
        """
        Restituisce il contenuto del file della lista di righe.

        Returns:
            list: Contenuto del file della lista di righe.
        """
        return self.linelist_file_content

    def solar_reference(self, elem, abun, met):
        """
        Calcola le abbondanze scalate rispetto ai valori solari.

        Args:
            elem (list): Lista dei numeri atomici.
            abun (list): Lista delle differenze di abbondanza.
            met (float): Metallicità [Fe/H].

        Returns:
            np.array: Array delle abbondanze scalate.
        """
        # Le referenze sono Magg22
        atomic_n = np.array([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67,
            68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 90, 92
        ])

        sunref = np.array([
             0, 12.00, 10.93, 1.05, 1.38, 2.70, 8.56, 7.98, 8.77, 4.67, 8.15, 6.33, 7.58, 6.48,
            7.567, 5.48, 7.21, 5.29, 6.50, 5.12, 6.32, 3.09, 4.96, 4.01, 5.69, 5.53, 7.51, 4.92,
            6.25, 4.21, 4.60, 2.88, 3.58, 2.29, 3.33, 2.56, 3.25, 2.60, 2.92, 2.21, 2.58, 1.42,
            1.92, 1.84, 1.12, 1.66, 0.94, 1.77, 1.60, 2.00, 1.00, 2.19, 1.51, 2.24, 1.07, 2.17,
            1.13, 1.70, 0.58, 1.45, 1.00, 0.52, 1.11, 0.28, 1.14, 0.51, 0.93, 0.00, 1.08, 0.06,
            0.88, -0.17, 1.11, 0.23, 1.25, 1.38, 1.64, 1.01, 1.13, 0.90, 2.00, 0.65, 0.06, -0.52
        ])

        # Verifica che elem e abun siano array e abbiano la stessa lunghezza
        elem = np.array(elem, dtype=int)  
        abun = np.array(abun, dtype=float)

        if len(elem) != len(abun):
            raise ValueError("Gli array 'elem' e 'abun' devono avere la stessa lunghezza")

        # Calcolo delle abbondanze scalate
        logN_abuns = []
        for el, ab in zip(elem, abun):
            index = np.where(el - atomic_n == 0)[0]
            if len(index) == 0:
                print(f"Elemento {el} non trovato nell'array delle referenze, salto questo elemento.")
                continue
            logN_abun = ab + met + sunref[index[0]]
            logN_abuns.append(logN_abun)

        return np.squeeze(logN_abuns)

    def process_solar_references(self, linelist_file, abu_file, metallicity):
        """
        Processa le referenze solari per ogni riga del DataFrame.

        Returns:
            list: Lista di dizionari contenenti il contenuto della lista di righe, le abbondanze scalate e Crat.
        """
        linelist_file_content = [self.read_linelist_file(linelist_file)]
        linelist_file_content = [[item.strip() for item in sublist] for sublist in linelist_file_content][0]

        abu_data, Crat = self.read_abu_file(abu_file)
        atomic_numbers = abu_data['AtomicNumber'].values
        abundance_diff = abu_data['AbundanceDifference'].values

        met = metallicity
        logN_abuns = self.solar_reference(atomic_numbers, abundance_diff, met)
        results = ({
            'linelist_content': linelist_file_content,
            'logN_abuns': logN_abuns,
            'Crat': Crat if len(Crat) > 0 else None
        })
        return results

    def write_output_spectra_count(self):
        """
        Stampa il numero di spettri da sintetizzare.
        """
        num_spectra = len(self.df)
        print(f"Numero di spettri da sintetizzare: {num_spectra}")
        print()

        return num_spectra


    def print_time(self, timeprint):
        if timeprint > 60 :
            print("Time for computing spectrum: ",str(round(timeprint/60,2))," min")
        else:
            print("Time for computing spectrum: ",str(round(timeprint,2))," sec")

"""
# Esempio di utilizzo
filename = '/Users/cristiano.fanelli/ASTRO/softw/Turbospectrum2019-master/COM-v19.1/input.ts'
params = InputParameters(filename)
print(params.get_paths())
print(params.get_keywords())
params.decisor_model(params.get_keywords()[1])
# print(params.get_linelist_content())

# Ottieni il DataFrame
df = params.get_dataframe()
# Mostra l'intero DataFrame
print(df)

# Processa le referenze solari
abundances = params.process_solar_references()
#print(abundances[0]['Crat'])
#print(abundances[0]['linelist_content'])
#print(abundances[0]['logN_abuns'])

# Scrivi il numero di spettri da sintetizzare
params.write_output_spectra_count()
"""
