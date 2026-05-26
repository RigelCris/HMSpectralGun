#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 17:39:52 2022

@author: cristiano
"""
import numpy as np
import os
from os.path import exists

class ModelMaker:
    def __init__(self, dataset_model_path):
        self.dataset_model_path = dataset_model_path

    # Function to find the nearest value in an array
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    # Function to find the two nearest values in an array
    def find_2nearest(self, array, value):
        if value in array:
            return value, value
        else:
            array = np.asarray(array)
            idx1 = (np.abs(array - value)).argmin()
            value1 = array[idx1]
            idx2 = (np.abs(array[array != array[idx1]] - value)).argmin()
            value2 = array[array != array[idx1]][idx2]
        return value1, value2

    # Function to select models for interpolation
    def select_models_for_interpolation(self, Teff, logg, met, xi, keyw_chem):
        os.system("basename " + self.dataset_model_path + "s"+str(Teff)[0]+"*m1.0*" + keyw_chem + "*.mod > list_models && mv list_models " + self.dataset_model_path)
        all_models = np.genfromtxt(self.dataset_model_path + "list_models", dtype="str")
        target = np.array([Teff, logg, xi, met])
        teff, logg, xi, met = np.zeros((len(all_models))), np.zeros((len(all_models))), np.zeros((len(all_models))), np.zeros((len(all_models)))
        chem = np.asarray(['ab'] * len(all_models))
        
        # Parse the model parameters from the filenames
        for n in range(len(all_models)):
            teff[n] = int(all_models[n][1:5])
            logg[n] = float(all_models[n][7:11])
            xi[n] = int(all_models[n][19])
            chem[n] = all_models[n][21:23]
            met[n] = float(all_models[n][25:30])

        i = np.where(chem == keyw_chem)
        cube_models = all_models[i]
        cube_teff, cube_logg, cube_xi, cube_met = teff[i], logg[i], xi[i], met[i]

        xi_cond = self.find_nearest([1, 2, 5], target[2])
        teffs = np.sort(self.find_2nearest(cube_teff, target[0]))
        mets = np.sort(self.find_2nearest(cube_met[np.where((cube_teff == teffs[0]) | (cube_teff == teffs[1]))], target[3]))
        loggs = np.sort(self.find_2nearest(cube_logg[np.where((cube_teff == teffs[0]) | (cube_teff == teffs[1]))], target[1]))
        xis = self.find_nearest(cube_xi[np.where((cube_teff == teffs[0]) | (cube_teff == teffs[1]))], target[2])

        # Select the models based on the closest parameters
        print(teffs, mets, loggs, xis)
        model1 = cube_models[np.where((cube_teff == teffs[0]) & (cube_logg == loggs[0]) & (cube_met == mets[0]))][0]
        model2 = cube_models[np.where((cube_teff == teffs[0]) & (cube_logg == loggs[0]) & (cube_met == mets[1]))][0]
        model3 = cube_models[np.where((cube_teff == teffs[0]) & (cube_logg == loggs[1]) & (cube_met == mets[0]))][0]
        model4 = cube_models[np.where((cube_teff == teffs[0]) & (cube_logg == loggs[1]) & (cube_met == mets[1]))][0]
        model5 = cube_models[np.where((cube_teff == teffs[1]) & (cube_logg == loggs[0]) & (cube_met == mets[0]))][0]
        model6 = cube_models[np.where((cube_teff == teffs[1]) & (cube_logg == loggs[0]) & (cube_met == mets[1]))][0]
        model7 = cube_models[np.where((cube_teff == teffs[1]) & (cube_logg == loggs[1]) & (cube_met == mets[0]))][0]
        model8 = cube_models[np.where((cube_teff == teffs[1]) & (cube_logg == loggs[1]) & (cube_met == mets[1]))][0]

        return model1, model2, model3, model4, model5, model6, model7, model8

    # Function to write the interpolator
    def write_interpolator(self, Teff, logg, met, chem, model_path, models):
        model1, model2, model3, model4, model5, model6, model7, model8 = models
        sgn_logg = 'p' if logg >= 0 else 'm'
        sgn_met = 'p' if met >= 0 else 'm'
        met_w = str(abs(met))
        
        
        #if abs(met) < 1:
        #    met_w = '0' + str(int(abs(met * 100)))
        #else:
        #    met_w = str(int(abs(met * 100)))

        #if abs(logg) < 1:
        #    logg = '0' + str(int(abs(logg * 100)))
        #else:
        #    logg = str(int(abs(logg * 100)))
        met_w = str(f"{abs(met):.2f}")
        logg = str(f"{abs(logg):.2f}")

        name_model = f'T{Teff}_G{sgn_logg}{logg}_{chem}_Z{sgn_met}{met_w}'

        if exists(model_path + name_model + '.interpol'):
            #print("Model Atmosphere already exists")
            return name_model + '.interpol'
        else:
            print("Interpolating Model Atmosphere...")

            with open('interp_models.com', "w") as file:
                file.write("#!/bin/csh -f \n\n")
                file.write("################################################################################################## \n")
                file.write("# Output turbospectrum/babsma format compatible \n")
                file.write("# Extrapolation is not advised, even if allowed by this program. \n")
                file.write("# Requires an 'cubic' set of 8 MARCS binary format models, \n")
                file.write("# in other words \n")
                file.write("# !!!!!   MODELS MUST DIFFER 2 BY 2 BY ONLY ONE PARAMETER !!!!!! \n")
                file.write("# !!!!!!! ORDER OF THE INPUT MODELS MATTERS !!!!!!! \n")
                file.write("# here is the order of the files \n")
                file.write("# model1: Tefflow logglow zlow \n")
                file.write("# model2: Tefflow logglow zup \n")
                file.write("# model3: Tefflow loggup zlow \n")
                file.write("# model4: Tefflow loggup zup \n")
                file.write("# model5: Teffup logglow zlow \n")
                file.write("# model6: Teffup logglow zup \n")
                file.write("# model7: Teffup loggup zlow \n")
                file.write("# model8: Teffup loggup zup \n")
                file.write("###################################################################################################### \n\n\n")
                file.write(f"set model_path = '{model_path}' \n")
                file.write(f"set dataset_model_path = '{self.dataset_model_path}' \n")
                file.write("set marcs_binary = '.false.' \n")
                file.write(f"set name_model = '{name_model}' \n\n")
                file.write(f"set model1 = '{model1}' \n")
                file.write(f"set model2 = '{model2}' \n")
                file.write(f"set model3 = '{model3}' \n")
                file.write(f"set model4 = '{model4}' \n")
                file.write(f"set model5 = '{model5}' \n")
                file.write(f"set model6 = '{model6}' \n")
                file.write(f"set model7 = '{model7}' \n")
                file.write(f"set model8 = '{model8}' \n\n")
                file.write(f"set Tref = '{Teff}' \n")
                file.write(f"set loggref = '{logg}' \n")
                file.write(f"set zref = '{met}' \n")
                file.write("set model_out = ${model_path}${name_model}.interpol \n")
                file.write("set model_out2 = ${model_path}${name_model}.alt \n\n")
                file.write("set test = '.false.' \n")
                file.write(f"set model_test = '{model_path}/{model1}' \n\n")
                file.write("/Users/cfanelli/astro/softw/TS-NLTE/COM/santerre/HMSpectralGun/marcs_generator/interpol_modeles <<EOF \n")
                file.write("'${dataset_model_path}/${model1}' \n")
                file.write("'${dataset_model_path}/${model2}' \n")
                file.write("'${dataset_model_path}/${model3}' \n")
                file.write("'${dataset_model_path}/${model4}' \n")
                file.write("'${dataset_model_path}/${model5}' \n")
                file.write("'${dataset_model_path}/${model6}' \n")


                file.write("'${dataset_model_path}/${model7}' \n")
                file.write("'${dataset_model_path}/${model8}' \n")
                file.write("'${model_out}' \n")
                file.write("'${model_out2}' \n")
                file.write("${Tref} \n")
                file.write("${loggref} \n")
                file.write("${zref} \n")
                file.write("${test} \n")
                file.write("${marcs_binary} \n")
                file.write("'${model_test}' \n")
                file.write("EOF \n")

            os.system("chmod 777 interp_models.com")
            os.system("./interp_models.com > logmarcs")
            os.system("rm interp_models.com")
            os.system("rm modele.sm")

            return name_model + '.interpol'

    # Function to select the nearest model
    def select_nearest_model(self, Teff, logg, met, xi, keyw_chem, model_path):
        os.system("basename " + self.dataset_model_path + "*m1.0*" + keyw_chem + "*.mod > list_models && mv list_models " + self.dataset_model_path)
        all_models = np.genfromtxt(self.dataset_model_path + "list_models", dtype="str")
        target = np.array([Teff, logg, xi, met])
        teff, logg, xi, met = np.zeros((len(all_models))), np.zeros((len(all_models))), np.zeros((len(all_models))), np.zeros((len(all_models)))
        chem = np.asarray(['ab'] * len(all_models))
        
        # Parse the model parameters from the filenames
        for n in range(len(all_models)):
            teff[n] = int(all_models[n][1:5])
            logg[n] = float(all_models[n][7:11])
            xi[n] = int(all_models[n][19])
            chem[n] = all_models[n][21:23]
            met[n] = float(all_models[n][25:30])

        i = np.where(chem == keyw_chem)
        cube_models = all_models[i]
        cube_teff, cube_logg, cube_xi, cube_met = teff[i], logg[i], xi[i], met[i]

        xi_cond = self.find_nearest([1, 2, 5], target[2])
        teffs = self.find_nearest(cube_teff, target[0])
        mets = self.find_nearest(cube_met[np.where(cube_teff == teffs)], target[3])
        loggs = self.find_nearest(cube_logg[np.where(cube_teff == teffs)], target[1])
        xis = self.find_nearest(cube_xi[np.where(cube_teff == teffs)], target[2])

        selected_model = cube_models[np.where((cube_teff == teffs) & (cube_logg == loggs) & (cube_met == mets))][0]

        sgn_logg = 'p' if loggs >= 0 else 'm'
        sgn_met = 'p' if mets >= 0 else 'm'
        met_w = str(abs(mets))
        
        #if abs(mets) < 1:
        #    met_w = '0' + str(int(abs(mets * 100)))
        #else:
        #    met_w = str(int(abs(mets * 100)))
        met_w = str(f"{abs(mets):.2f}")
        logg = str(f"{abs(float(selected_model[7:11])):.2f}")


        #name_model = f'T{selected_model[1:5]}_G{sgn_logg}{logg}_{selected_model[21:23]}_Z{sgn_met}{met_w}'
        name_model = f'T{selected_model[1:5]}_G{sgn_logg}{logg}'

        print(' Teff    : ', selected_model[1:5], 'K')
        print(' logg    : ', selected_model[7:11], 'dex')
        print(' [Fe/H]  : ', selected_model[25:30], 'dex')
        print(' Mixture : ', selected_model[21:23])
        print(' xi      : ', selected_model[19], 'km/s')

        if exists(model_path + name_model):
            pass
        else:
            os.system("cp " + self.dataset_model_path + selected_model + " " + model_path + name_model)

        return name_model