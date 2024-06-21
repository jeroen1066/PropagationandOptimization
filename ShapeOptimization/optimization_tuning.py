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
import Optimizationutilities_tuning as OptUtil

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
nominal_generations = 25
nominal_populations = 45
nominal_neighbours = 20
nominal_CR = 1
nominal_F = 0.5
nominal_eta_m = 20
nominal_realb = 0.9

seeds_to_test = [42, 84, 144, 169]
generations_to_test = [25, 50, 75, 100, 150]
pops_to_test = [30, 45, 60, 75, 100, 150]
num_repeats = len(seeds_to_test)

range_per_parameter = [[0,5],
                       [2,5],
                       [0,3],
                        [-np.pi/2,0.],
                        [2.5,5.5],
                        [0.0,0.5],
                        [50,400]
                       ]

bounds = [[0,2,0,-np.pi/2,2.5,0.0,50],[5,5,3,0,5.5,0.5,400]]


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
    x = results.get_x()
    y = results.get_f()

    results_to_store.append([x,y])

    filename = 'results/tuning/' + optimizer_name + '_seed_' + str(test_seed) +'.dat' 
    file = open(filename,'wb')
    pickle.dump(results_to_store,file)