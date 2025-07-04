#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:02:47 2020

@author: cristiano
"""

import numpy as np
import os

class LineListManager:
    """
    Classe per gestire le operazioni sulla lista di righe.

    Metodi:
        use_only_molecular_data(linelist_path, linespec, keyv):
            Filtra i dati molecolari dalla lista di righe.
        create_linelist_single_element(linelist_path, linespec, keyv):
            Crea una lista di righe per un singolo elemento.
        delete_tmp_linelist(linelist_path, new_linespec, keyv, keyw='No'):
            Elimina i file temporanei della lista di righe.
    """

    @staticmethod
    def use_only_molecular_data(linelist_path, linespec, keyv):
        """
        Filtra i dati molecolari dalla lista di righe.

        Args:
            linelist_path (str): Il percorso della lista di righe.
            linespec (list): La lista delle specifiche delle righe.
            keyv (list): La chiave di ricerca.

        Returns:
            list: La nuova lista di righe contenente solo dati molecolari.
        """
        new_linespec = []
        for line in linespec:
            if keyv[2] in line:
                new_linespec.append(line)
        return new_linespec

    @staticmethod
    def create_linelist_single_element(linelist_path, linespec, keyv):
        """
        Crea una lista di righe per un singolo elemento.

        Args:
            linelist_path (str): Il percorso della lista di righe.
            linespec (list): La lista delle specifiche delle righe.
            keyv (list): La chiave di ricerca.

        Returns:
            list: La nuova lista di righe per un singolo elemento.
        """
        if keyv[2] in ['CO', 'OH', 'CN']:
            new_linespec = LineListManager.use_only_molecular_data(linelist_path, linespec, keyv)
        else:
            valdlist = [line for line in linespec if 'vald-' in line]
            atomic_n, atomic_ion, elem = keyv

            new_linespec = []
            for valdfile in valdlist:
                with open(os.path.join(linelist_path, valdfile), "r") as f:
                    datavald = f.readlines()

                n_start, n_lines = 0, 0
                for n in range(len(datavald)):
                    if datavald[n][0:2] == "' " and datavald[n][3:9] == atomic_n and datavald[n][26:27] == atomic_ion:
                        n_start = n
                        n_lines = int(datavald[n][32:43]) + 2

                newfile = f'tmp_{valdfile[:-5]}.{elem}'
                lines_w = datavald[n_start:n_start + n_lines]

                with open(os.path.join(linelist_path, newfile), "w") as file:
                    for line in lines_w:
                        file.write(f'{line} ')

                new_linespec.append(newfile)

        return new_linespec

    @staticmethod
    def delete_tmp_linelist(linelist_path, new_linespec, keyv, keyw='No'):
        """
        Elimina i file temporanei della lista di righe.

        Args:
            linelist_path (str): Il percorso della lista di righe.
            new_linespec (list): La nuova lista di righe da eliminare.
            keyv (list): La chiave di ricerca.
            keyw (str): Indicatore per eliminare o meno i file (default: 'No').
        """
        if keyw == 'Yes' and keyv[2] not in ['CO', 'OH', 'CN']:
            for newfile in new_linespec:
                os.remove(os.path.join(linelist_path, newfile))
        else:
            pass


"""

linelist_path = '/Users/cristiano.fanelli/ASTRO/softw/Turbospectrum2019-master/COM-v19.1/linelists/'
linespec = ['vald-14800-18100-hfs.list']
keyv = ['26.000', '1', 'FeI']

# Crea un oggetto LineListManager
manager = LineListManager()

# Usa solo dati molecolari
#molecular_data = manager.use_only_molecular_data(linelist_path, linespec, keyv)
#print("Molecular Data:", molecular_data)

# Crea una lista di righe per un singolo elemento
single_element_list = manager.create_linelist_single_element(linelist_path, linespec, keyv)
print("Single Element Line List:", single_element_list)

# Elimina i file temporanei della lista di righe
manager.delete_tmp_linelist(linelist_path, single_element_list, keyv, keyw='Yes')


"""