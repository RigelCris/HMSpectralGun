#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:02:47 2020

@author: cristiano
"""

import numpy as np
import os
import re

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
        # File names are not uniform across molecular databases.  In
        # particular the optical CN isotopologue files are named C12N14,
        # C12N15, C13N14 and C13N15 rather than containing the string "CN".
        aliases = {
            'CN': ('CN', 'C12N', 'C13N'),
            'CO': ('CO',),
            'OH': ('OH',),
        }
        patterns = aliases.get(keyv[2], (keyv[2],))
        new_linespec = []
        for line in linespec:
            if any(pattern in line for pattern in patterns):
                new_linespec.append(line)
        return new_linespec

    @staticmethod
    def create_linelist_single_element(linelist_path, linespec, keyv, tmp_tag=None):
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

            if tmp_tag is None:
                tmp_tag = f"p{os.getpid()}"

            header_re = re.compile(
                r"^'\s*(?P<atomic>\d+(?:\.\d+)?)\s*'\s+"
                r"(?P<ion>\d+)\s+(?P<count>\d+)"
            )
            target_atomic = float(atomic_n)
            target_ion = int(atomic_ion)
            new_linespec = []
            for valdfile in valdlist:
                with open(os.path.join(linelist_path, valdfile), "r") as f:
                    datavald = f.readlines()

                lines_w = []
                for n, line in enumerate(datavald):
                    match = header_re.match(line)
                    if match is None:
                        continue
                    if (
                        float(match.group('atomic')) == target_atomic
                        and int(match.group('ion')) == target_ion
                    ):
                        n_lines = int(match.group('count')) + 2
                        lines_w.extend(datavald[n:n + n_lines])

                # A species need not occur in every wavelength segment. Do
                # not pass empty temporary files to Turbospectrum.
                if not lines_w:
                    continue

                newfile = f'tmp_{tmp_tag}_{valdfile[:-5]}.{elem}'

                with open(os.path.join(linelist_path, newfile), "w") as file:
                    file.writelines(lines_w)

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
