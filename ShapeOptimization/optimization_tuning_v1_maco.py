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
import Optimizationutilities_tuning_v1_maco as OptUtil

# General python imports
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle

import pygmo as pg

spice_interface.load_standard_kernels()

optimizer_name = 'maco'
nominal_seed = 42  
nominal_generations = 50
nominal_populations = 100

nominal_kernel = 63
nominal_convergence_speed = 1
nominal_threshold = 1
nominal_std_convergence_speed = 7
nominal_eval_stop = 100000
nominal_focus = 0.0

seeds_to_test = [42, 84, 144, 169, 74, 29, 60, 1745, 1480025]
generations_to_test = [25, 50, 75, 100, 150]
pops_to_test = [30, 50, 75, 100, 150]

kernel_to_test = [10, 30, 50, 70, 90]
convergence_speed_to_test = [0.5, 1, 1.5, 2, 2.5]
threshold_to_test = [1, 2,  5, 10, 20]
std_convergence_speed_to_test = [3, 5, 7, 9, 11, 15, 21]
eval_stop_to_test = [50000, 75000, 100000, 125000, 150000, 200000, 300000]
focus_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]

test_for_seed = False
test_for_generations = False
test_for_pops = False

test_for_kernel = False
test_for_convergence_speed = False
test_for_threshold = False
test_for_std_convergence_speed = False
test_for_eval_stop = False
test_for_focus = False



save_directory = 'ShapeOptimization/results/tuning_v1_maco/'

range_per_parameter = [[0,5],
                       [2,5],
                       [0,3],
                        [-np.pi/2,0.],
                        [2.5,5.5],
                        [0.0,0.5],
                        [50,400]
                       ]

bounds = [[0,2,0,-np.pi/2,2.5,0.0,50],[5,5,3,0,5.5,0.5,400]]

#   opt.optimize(numpops = nominal_populations,
#                     numgens = nominal_generations,
#                     ker=nominal_kernel, 
#                     q=nominal_convergence_speed, 
#                     threshold=nominal_threshold, 
#                     n_gen_mark=nominal_std_convergence_speed, 
#                     evalstop=nominal_eval_stop, 
#                     focus=nominal_focus, 
#                     seed=nominal_seed)


if test_for_seed:
    for test_seed in seeds_to_test:
        print('Testing seed: ', test_seed)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = nominal_populations,
                    numgens = nominal_generations,
                    ker=nominal_kernel, 
                    q=nominal_convergence_speed, 
                    threshold=nominal_threshold, 
                    n_gen_mark=nominal_std_convergence_speed, 
                    evalstop=nominal_eval_stop, 
                    focus=nominal_focus, 
                    seed=test_seed)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_seed_' + str(test_seed) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = os.path.join(save_directory, optimizer_name + '_seed_' + str(test_seed) + '_ancillary.txt')
        with open(filename_ancillary, 'w') as file:
            file.write(str(simulation_duration_tested))

if test_for_generations:
    for test_generations in generations_to_test:
        print('Testing generations: ', test_generations)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = nominal_populations,
                    numgens = nominal_generations,
                    ker=nominal_kernel, 
                    q=nominal_convergence_speed, 
                    threshold=nominal_threshold, 
                    n_gen_mark=nominal_std_convergence_speed, 
                    evalstop=nominal_eval_stop, 
                    focus=nominal_focus, 
                    seed=nominal_seed)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_gens_' + str(test_generations) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = os.path.join(save_directory, optimizer_name + '_gens_' + str(test_generations) + '_ancillary.txt')
        with open(filename_ancillary, 'w') as file:
            file.write(str(simulation_duration_tested))

if test_for_pops:
    for test_pops in pops_to_test:
        print('Testing pops: ', test_pops)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = test_pops,
                    numgens = nominal_generations,
                    ker=nominal_kernel, 
                    q=nominal_convergence_speed, 
                    threshold=nominal_threshold, 
                    n_gen_mark=nominal_std_convergence_speed, 
                    evalstop=nominal_eval_stop, 
                    focus=nominal_focus, 
                    seed=nominal_seed)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_pops_' + str(test_pops) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = os.path.join(save_directory, optimizer_name + '_pops_' + str(test_pops) + '_ancillary.txt')
        with open(filename_ancillary, 'w') as file:
            file.write(str(simulation_duration_tested))

if test_for_kernel:
    for test_kernel in kernel_to_test:
        print('Testing kernel: ', test_kernel)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = nominal_populations,
                    numgens = nominal_generations,
                    ker=test_kernel, 
                    q=nominal_convergence_speed, 
                    threshold=nominal_threshold, 
                    n_gen_mark=nominal_std_convergence_speed, 
                    evalstop=nominal_eval_stop, 
                    focus=nominal_focus, 
                    seed=nominal_seed)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_kernel_' + str(test_kernel) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = os.path.join(save_directory, optimizer_name + '_kernel_' + str(test_kernel) + '_ancillary.txt')
        with open(filename_ancillary, 'w') as file:
            file.write(str(simulation_duration_tested))

if test_for_convergence_speed:
    for test_convergence_speed in convergence_speed_to_test:
        print('Testing convergence_speed: ', test_convergence_speed)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = nominal_populations,
                    numgens = nominal_generations,
                    ker=nominal_kernel, 
                    q=test_convergence_speed, 
                    threshold=nominal_threshold, 
                    n_gen_mark=nominal_std_convergence_speed, 
                    evalstop=nominal_eval_stop, 
                    focus=nominal_focus, 
                    seed=nominal_seed)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_q_' + str(test_convergence_speed) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = os.path.join(save_directory, optimizer_name + '_q_' + str(test_convergence_speed) + '_ancillary.txt')
        with open(filename_ancillary, 'w') as file:
            file.write(str(simulation_duration_tested))

if test_for_threshold:
    for test_threshold in threshold_to_test:
        print('Testing threshold: ', test_threshold)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = nominal_populations,
                    numgens = nominal_generations,
                    ker=nominal_kernel, 
                    q=nominal_convergence_speed, 
                    threshold=test_threshold, 
                    n_gen_mark=nominal_std_convergence_speed, 
                    evalstop=nominal_eval_stop, 
                    focus=nominal_focus, 
                    seed=nominal_seed)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_threshold_' + str(test_threshold) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = os.path.join(save_directory, optimizer_name + '_threshold_' + str(test_threshold) + '_ancillary.txt')
        with open(filename_ancillary, 'w') as file:
            file.write(str(simulation_duration_tested))

if test_for_std_convergence_speed:
    for test_std_convergence_speed in std_convergence_speed_to_test:
        print('Testing std_convergence_speed: ', test_std_convergence_speed)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = nominal_populations,
                    numgens = nominal_generations,
                    ker=nominal_kernel, 
                    q=nominal_convergence_speed, 
                    threshold=nominal_threshold, 
                    n_gen_mark=test_std_convergence_speed, 
                    evalstop=nominal_eval_stop, 
                    focus=nominal_focus, 
                    seed=nominal_seed)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_n_gen_mark_' + str(test_std_convergence_speed) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = os.path.join(save_directory, optimizer_name + '_n_gen_mark_' + str(test_std_convergence_speed) + '_ancillary.txt')
        with open(filename_ancillary, 'w') as file:
            file.write(str(simulation_duration_tested))

if test_for_eval_stop:
    for test_eval_stop in eval_stop_to_test:
        print('Testing eval_stop: ', test_eval_stop)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = nominal_populations,
                    numgens = nominal_generations,
                    ker=nominal_kernel, 
                    q=nominal_convergence_speed, 
                    threshold=nominal_threshold, 
                    n_gen_mark=nominal_std_convergence_speed, 
                    evalstop=test_eval_stop, 
                    focus=nominal_focus, 
                    seed=nominal_seed)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_evalstop_' + str(test_eval_stop) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = os.path.join(save_directory, optimizer_name + '_evalstop_' + str(test_eval_stop) + '_ancillary.txt')
        with open(filename_ancillary, 'w') as file:
            file.write(str(simulation_duration_tested))

if test_for_focus:
    for test_focus in focus_to_test:
        print('Testing focus: ', test_focus)
        opt = OptUtil.optimization(bounds, optimizer_name)
        opt.optimize(numpops = nominal_populations,
                    numgens = nominal_generations,
                    ker=nominal_kernel, 
                    q=nominal_convergence_speed, 
                    threshold=nominal_threshold, 
                    n_gen_mark=nominal_std_convergence_speed, 
                    evalstop=nominal_eval_stop, 
                    focus=test_focus, 
                    seed=nominal_seed)

        results = opt.results
        results_to_store = []
        x = results[0].get_x()
        y = results[0].get_f()

        results_to_store.append([x,y])

        filename = save_directory + optimizer_name + '_focus_' + str(test_focus) +'.dat' 
        file = open(filename,'wb')
        pickle.dump(results_to_store,file)

        simulation_duration_tested = opt.simulation_duration
        filename_ancillary = os.path.join(save_directory, optimizer_name + '_focus_' + str(test_focus) + '_ancillary.txt')
        with open(filename_ancillary, 'w') as file:
            file.write(str(simulation_duration_tested))

print('All tests completed')