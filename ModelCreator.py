#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:13:56 2023

@author: cristiano
"""
import numpy as np
import os
import input as input
import argparse
from mkmodels import ModelMaker


# Argument parser setup
parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument('--Teff', type=int, help='Temperature')
parser.add_argument('--logg', type=float, help='Surface gravity')
parser.add_argument('--xi', type=float, help='Microturbulence')
parser.add_argument('--met', type=float, help='Metallicity')
parser.add_argument('--mixture', type=str, help='Mixture : st (standard) ; ap (alpha poor) ; mc (midly cycled) ; hc (heavily cycled)')
parser.add_argument('--interpolate', type=str, help='no : take the nearest MARCS model ;  [yes] : interpolate the model')
parser.add_argument('--name_model', type=str, help='output name of the model')

args = parser.parse_args()

# Print arguments if provided
for arg, val in vars(args).items():
    if val is not None:
        print(f'{arg.capitalize()}: {val}')

# Unpack arguments
Teff, logg, xi, met, mixture, interpolate, name_model = args.Teff, args.logg, args.xi, args.met, args.mixture, args.interpolate, args.name_model

class ModelCreator:
    def __init__(self, model_path, dataset_model_path):
        self.model_path = model_path
        self.selector = ModelMaker(dataset_model_path)

    def create_model(self, Teff=4000, logg=1.5, xi=2.0, met=0.0, chem='st', interpolate='yes', name_model=None):
        """Create a model atmosphere based on the provided parameters."""
        models = self.selector.select_models_for_interpolation(Teff, logg, met, xi, chem)

        if interpolate == 'yes':
            model = self.selector.write_interpolator(Teff, logg, met, chem, self.model_path, models)
        else:
            model = self.selector.select_nearest_model(Teff, logg, met, xi, chem, self.model_path)

        if name_model:
            self.rename_model(model, name_model, Teff, logg, met, chem)
        return model

    def rename_model(self, model, name_model, Teff, logg, met, chem):
        """Rename the generated model and update its header."""
        prefix = model.split('.')[0]
        new_model_path = os.path.join(self.model_path, name_model + '.mod')
        os.rename(os.path.join(self.model_path, prefix + '.interpol'), new_model_path)
        header = f'# Teff={Teff} K   logg={logg} dex   [Fe/H]={met} dex   mixture={chem}'
        
        alt_file_path = os.path.join(self.model_path, prefix + '.alt')
        if os.path.exists(alt_file_path):
            with open(alt_file_path, 'r') as file:
                old_content = file.read()
            with open(alt_file_path, 'w') as file:
                file.write(header + '\n' + old_content)
            os.rename(alt_file_path, os.path.join(self.model_path, name_model + '.alt'))
        else:
            print(f"File '{prefix}.alt' not found. Ensure it is created before running the script.")


"""
# Instantiate ModelCreator
model_creator = ModelCreator(
    model_path='/Users/cristiano.fanelli/ASTRO/softw/Turbospectrum2019-master/COM-v19.1/models/',
    dataset_model_path='/Users/cristiano.fanelli/ASTRO/softw/Turbospectrum2019-master/COM-v19.1/marcs_generator/dataset/')

# Create model based on terminal arguments
model = model_creator.create_model(Teff=Teff, logg=logg, xi=xi, met=met, chem=mixture, interpolate=interpolate, name_model=name_model)

print()
print('Created model atmosphere: ' + model)
"""
