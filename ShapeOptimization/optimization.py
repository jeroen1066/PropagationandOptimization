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
import Optimizationutilities as OptUtil

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

optimizer_name = 'moead'
num_repeats = 5
num_generations = 5
num_pops = 10
seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]

range_per_parameter = [[0,5],
                       [2,5],
                       [0,3],
                        [-np.pi/2,0.],
                        [2.5,5.5],
                        [-0.25,0.5],
                        [50,400]
                       ]

bounds = [[0,2,0,-np.pi/2,2.5,-0.25,50],[5,5,3,0,5.5,0.5,400]]

opt = OptUtil.optimization(bounds, optimizer_name)
opt.optimize(num_pops,num_generations,num_repeats,seeds)
