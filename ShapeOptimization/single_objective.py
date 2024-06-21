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
import Optimisationutilities_single as OptUtil

# General python imports
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle

import pygmo as pg

spice_interface.load_standard_kernels()


#do optimization
#optimizer names: gaco, de, sade, de1220, gwo, ihs, pso, pso_gen, sea, sga, simulated_annealing, bee_colony, cmaes, xnes
#seeds needs to be equal size to num_repeats

optimizer_name = 'gaco'
num_repeats = 2
num_generations = 10
num_pops = 63
seeds = [42,84]

range_per_parameter = [[0,5],
                       [2,5],
                       [0,3],
                        [-np.pi/2,0.],
                        [2.5,5.5],
                        [0.0,0.5],
                        [50,400]
                       ]

bounds = [[0,2,0,-np.pi/2,2.5,0.0,50],[5,5,3,0,5.5,0.5,400]]

opt = OptUtil.optimization(bounds, optimizer_name)
opt.optimize(num_pops,num_generations,num_repeats,seeds)
results = opt.results
results_per_generation = opt.results_per_generation

results_to_store = []

for i in range(num_repeats):
    x = results[i].get_x()
    y = results[i].get_f()
    y_per_gen = results_per_generation[i]

    results_to_store.append([x,y,y_per_gen])


filename = 'results/' + optimizer_name + '_single.dat'
file = open(filename,'wb')
pickle.dump(results_to_store,file)