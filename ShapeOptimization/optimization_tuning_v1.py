# General imports
import os

# Tudatpy imports
from tudatpy.data import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators

# Problem-specific imports
import CapsuleEntryUtilities as Util
import Optimizationutilities_tuning_v1 as OptUtil

# General python imports
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle

import pygmo as pg

spice_interface.load_standard_kernels()


#do optimization
#optimizer names are ihs, nsga2, moead, moead_gen, maco, nspso
#seeds needs to be equal size to num_repeats

optimizer_name = 'moead_gen'
nominal_seed = 42  
nominal_generations = 50
nominal_populations = 75
nominal_neighbours = 20
nominal_CR = 1
nominal_F = 0.5
nominal_eta_m = 20
nominal_realb = 0.9

seeds_to_test = [42, 84, 144, 169, 74, 29, 60, 1745, 1480025]
generations_to_test = [25, 50, 75, 100, 150]
pops_to_test = [30, 50, 66, 78, 100, 150]
neighbours_to_test = [10, 20, 30, 40, 50]
CR_to_test = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
F_to_test = [0.1, 0.3, 0.5, 0.7, 0.9]
eta_m_to_test = [10, 20, 30, 40, 50]
realb_to_test = [0.5, 0.6, 0.7, 0.8, 0.9]

test_for_seed = True
test_for_generations = True
test_for_pops = True
test_for_neighbours = True
test_for_CR = True
test_for_F = True
test_for_eta_m = True
test_for_realb = True

save_directory = 'ShapeOptimization/results/tuning_v1/'

range_per_parameter = [[0,5],
                       [2,5],
                       [0,3],
                        [-np.pi/2,0.],
                        [2.5,5.5],
                        [0.0,0.5],
                        [50,400]
                       ]

bounds = [[0,2,0,-np.pi/2,2.5,0.0,50],[5,5,3,0,5.5,0.5,400]]

if test_for_seed:
    for test_seed in seeds_to_test:
        print('Testing seed: ', test_seed)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = nominal_populations,
                    numgens = nominal_generations,
                    seed = test_seed, 
                    numneighbours = nominal_neighbours, 
                    test_CR = nominal_CR, 
                    test_F = nominal_F, 
                    test_eta_m = nominal_eta_m, 
                    test_realb = nominal_realb)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_seed_' + str(test_seed) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = save_directory + optimizer_name + '_seed_' + str(test_seed) +'_ancillary.dat'
        ancillary_data = [simulation_duration_tested]
        file = open(filename_ancillary,'wb')
        pickle.dump(ancillary_data,file)

if test_for_generations:
    for test_generations in generations_to_test:
        print('Testing generations: ', test_generations)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = nominal_populations,
                    numgens = test_generations,
                    seed = nominal_seed, 
                    numneighbours = nominal_neighbours, 
                    test_CR = nominal_CR, 
                    test_F = nominal_F, 
                    test_eta_m = nominal_eta_m, 
                    test_realb = nominal_realb)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_gens_' + str(test_generations) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = save_directory + optimizer_name + '_seed_' + str(test_seed) +'_ancillary.dat'
        ancillary_data = [simulation_duration_tested]
        file = open(filename_ancillary,'wb')
        pickle.dump(ancillary_data,file)

if test_for_pops:
    for test_pops in pops_to_test:
        print('Testing pops: ', test_pops)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = test_pops,
                    numgens = nominal_generations,
                    seed = nominal_seed, 
                    numneighbours = nominal_neighbours, 
                    test_CR = nominal_CR, 
                    test_F = nominal_F, 
                    test_eta_m = nominal_eta_m, 
                    test_realb = nominal_realb)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_pops_' + str(test_pops) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = save_directory + optimizer_name + '_seed_' + str(test_seed) +'_ancillary.dat'
        ancillary_data = [simulation_duration_tested]
        file = open(filename_ancillary,'wb')
        pickle.dump(ancillary_data,file)

if test_for_neighbours:
    for test_neighbours in neighbours_to_test:
        print('Testing neighbours: ', test_neighbours)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = nominal_populations,
                    numgens = nominal_generations,
                    seed = nominal_seed, 
                    numneighbours = test_neighbours, 
                    test_CR = nominal_CR, 
                    test_F = nominal_F, 
                    test_eta_m = nominal_eta_m, 
                    test_realb = nominal_realb)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_neighbours_' + str(test_neighbours) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = save_directory + optimizer_name + '_seed_' + str(test_seed) +'_ancillary.dat'
        ancillary_data = [simulation_duration_tested]
        file = open(filename_ancillary,'wb')
        pickle.dump(ancillary_data,file)

if test_for_CR:
    for test_CR in CR_to_test:
        print('Testing CR: ', test_CR)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = nominal_populations,
                    numgens = nominal_generations,
                    seed = nominal_seed, 
                    numneighbours = nominal_neighbours, 
                    test_CR = test_CR, 
                    test_F = nominal_F, 
                    test_eta_m = nominal_eta_m, 
                    test_realb = nominal_realb)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_CR_' + str(test_CR) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = save_directory + optimizer_name + '_seed_' + str(test_seed) +'_ancillary.dat'
        ancillary_data = [simulation_duration_tested]
        file = open(filename_ancillary,'wb')
        pickle.dump(ancillary_data,file)

if test_for_F:
    for test_F in F_to_test:
        print('Testing F: ', test_F)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = nominal_populations,
                    numgens = nominal_generations,
                    seed = nominal_seed, 
                    numneighbours = nominal_neighbours, 
                    test_CR = nominal_CR, 
                    test_F = test_F, 
                    test_eta_m = nominal_eta_m, 
                    test_realb = nominal_realb)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_F_' + str(test_F) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = save_directory + optimizer_name + '_seed_' + str(test_seed) +'_ancillary.dat'
        ancillary_data = [simulation_duration_tested]
        file = open(filename_ancillary,'wb')
        pickle.dump(ancillary_data,file)

if test_for_eta_m:
    for test_eta_m in eta_m_to_test:
        print('Testing eta_m: ', test_eta_m)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = nominal_populations,
                    numgens = nominal_generations,
                    seed = nominal_seed, 
                    numneighbours = nominal_neighbours, 
                    test_CR = nominal_CR, 
                    test_F = nominal_F, 
                    test_eta_m = test_eta_m, 
                    test_realb = nominal_realb)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_eta_m_' + str(test_eta_m) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = save_directory + optimizer_name + '_seed_' + str(test_seed) +'_ancillary.dat'
        ancillary_data = [simulation_duration_tested]
        file = open(filename_ancillary,'wb')
        pickle.dump(ancillary_data,file)

if test_for_realb:
    for test_realb in realb_to_test:
        print('Testing realb: ', test_realb)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = nominal_populations,
                    numgens = nominal_generations,
                    seed = nominal_seed, 
                    numneighbours = nominal_neighbours, 
                    test_CR = nominal_CR, 
                    test_F = nominal_F, 
                    test_eta_m = nominal_eta_m, 
                    test_realb = test_realb)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_realb_' + str(test_realb) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = save_directory + optimizer_name + '_seed_' + str(test_seed) +'_ancillary.dat'
        ancillary_data = [simulation_duration_tested]
        file = open(filename_ancillary,'wb')
        pickle.dump(ancillary_data,file)

print('Done')