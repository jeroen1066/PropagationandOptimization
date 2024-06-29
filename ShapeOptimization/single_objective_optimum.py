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

optimizer_name = 'sga'
data_file = 'results/' + optimizer_name + '_single_optimum.dat'
file = open(data_file,'rb')
data = pickle.load(file)
file.close()

for j in range(1):
    results = data[j]
    x = results[0]
    y = results[1]
    inputs = x[y.tolist().index(min(y))]
    print(inputs)

bounds = [[0,2,0,-np.pi/2,2.5,0.0,50],[5,5,3,0,5.5,0.5,400]]
optimizer_name = 'moead_gen'

opt = OptUtil.optimization(bounds, optimizer_name)
max_g, ld, mass = opt.getresults(inputs)

print(max_g, ld, mass)