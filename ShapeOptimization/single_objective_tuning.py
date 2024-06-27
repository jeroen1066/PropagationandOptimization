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
import Optimisationutilities_single_tuning as OptUtil

# General python imports
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle

import pygmo as pg
import time as tm

spice_interface.load_standard_kernels()


#do optimization
#optimizer names: gaco (too many), de, sade, de1220, gwo (bad), ihs (not good enough), pso, pso_gen, sea (not good enough), sga, simulated_annealing (too long),
# bee_colony (too long), cmaes (too long), xnes (too long)
#seeds needs to be equal size to num_repeats

optimizer_name = 'sga'
num_repeats = 8
num_generations = 25
num_pops = 60
seeds = [2, 17, 31, 42, 66, 84, 144, 169]

range_per_parameter = [[0,5],
                       [2,5],
                       [0,3],
                        [-np.pi/2,0.],
                        [2.5,5.5],
                        [0.0,0.5],
                        [50,400]
                       ]

bounds = [[0,2,0,-np.pi/2,2.5,0.0,50],[5,5,3,0,5.5,0.5,400]]

start_time = tm.time()
opt = OptUtil.optimization(bounds, optimizer_name)
opt.optimize(num_pops,num_generations,num_repeats,seeds,start_time)
results = opt.results
results_per_generation = opt.results_per_generation

results_to_store = []

for i in range(num_repeats):
    x = results[i].get_x()
    y = results[i].get_f()
    y_per_gen = results_per_generation[i]

    results_to_store.append([x,y,y_per_gen])


filename = 'results/' + optimizer_name + '_single_tuning_seeds.dat'
file = open(filename,'wb')
pickle.dump(results_to_store,file)