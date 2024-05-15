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

# General python imports
import numpy as np
import matplotlib.pyplot as plt

###########################################################################
# PARAMETERS ##############################################################
###########################################################################

benchmark_step_size = 0.5

regular_step_size = 8.

integrators = [propagation_setup.integrator.CoefficientSets.rkf_45,
                propagation_setup.integrator.CoefficientSets.rkf_56,
                propagation_setup.integrator.CoefficientSets.rkf_78,
                propagation_setup.integrator.CoefficientSets.rkf_1210
                ]

timesteps = [1,2,4,8,16,32,64]

tolerances = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10,1e-11,1e-12]

spice_interface.load_standard_kernels()
# NOTE TO STUDENTS: INPUT YOUR PARAMETER SET HERE, FROM THE INPUT FILES
# ON BRIGHTSPACE, FOR YOUR SPECIFIC STUDENT NUMBER
shape_parameters = [7.6983117481,
                    2.0923385955,
                    1.7186406196,
                    -0.255984141,
                    0.1158838553,
                    0.3203083369]


###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Set simulation start epoch
simulation_start_epoch = 0.0  # s
# Set termination conditions
maximum_duration = constants.JULIAN_DAY  # s
termination_altitude = 25.0E3  # m
# Set vehicle properties
capsule_density = 250.0  # kg m-3

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Define settings for celestial bodies
bodies_to_create = ['Earth']
# Define coordinate system
global_frame_origin = 'Earth'
global_frame_orientation = 'J2000'

# Create body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)
# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Create and add capsule to body system
# NOTE TO STUDENTS: When making any modifications to the capsule vehicle, do NOT make them in this code, but in the
# add_capsule_to_body_system function
Util.add_capsule_to_body_system(bodies,
                                shape_parameters,
                                capsule_density)


###########################################################################
# CREATE (CONSTANT) PROPAGATION SETTINGS ##################################
###########################################################################

# Retrieve termination settings
termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                     maximum_duration,
                                                     termination_altitude)
# Retrieve dependent variables to save
dependent_variables_to_save = Util.get_dependent_variable_save_settings()
# Check whether there is any
are_dependent_variables_to_save = False if not dependent_variables_to_save else True

propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                 bodies,
                                                                 simulation_start_epoch,
                                                                 termination_settings,
                                                                 dependent_variables_to_save )

###########################################################################
# BENCHMARK CREATION#######################################################
###########################################################################

benchmark_step_size = 0.5
benchmark_propagator = propagation_setup.propagator.cowell

integrator_index = 1
state_histories = []
differences = []


current_integrator_settings = Util.get_integrator_settings(0,
                                                                        integrator_index,
                                                                        benchmark_step_size,
                                                                        simulation_start_epoch)

coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_56
integrator = propagation_setup.integrator
integrator_settings = integrator.runge_kutta_fixed_step(
        benchmark_step_size,
        coefficient_set,
        integrator.OrderToIntegrate.lower
)

propagator_settings.integrator_settings = integrator_settings
#propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
#    benchmark_step_size,
#    propagation_setup.integrator.CoefficientSets.rkf_56)
#propagator_settings.print_settings.print_dependent_variable_indices = True

benchmark_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies,
    propagator_settings )

benchmark_state_history = benchmark_dynamics_simulator.state_history
interpolator_settings = interpolators.lagrange_interpolation(8)
benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(
    benchmark_state_history,
    interpolator_settings
)

###########################################################################
# REGULAR SIMULATION ######################################################
###########################################################################



fixed_step_integrators_error = []
fixed_step_integrators_evaluations = []
variable_step_integrators_error = []
variable_step_integrators_evaluations = []


variable_step_tests = []
variable_step_tests_names = []

for integrator_index in range(4):
    fixed_step_errors = []
    fixed_step_evaluations = []
    for timestep in timesteps:
        settingsdescription = [integrators[integrator_index], timestep]
        #settings_fixed_steps.append(settingsdescription)

        integrator_settings = integrator.runge_kutta_fixed_step(
        timestep,
        integrators[integrator_index],
        integrator.OrderToIntegrate.lower
        )
        propagator_settings.integrator_settings = integrator_settings

        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies,
            propagator_settings )
        if not dynamics_simulator.integration_completed_successfully:
            continue
        
        state_history = dynamics_simulator.state_history

        errors = []
        lastepoch = list(state_history.keys())[-1]
        lastbenchmarkepoch = list(benchmark_state_history.keys())[-1]
        if lastepoch > lastbenchmarkepoch+30:
            continue
        for epoch in state_history.keys():
            if epoch > lastbenchmarkepoch - 10*benchmark_step_size:
                continue
            interpolated_state = benchmark_interpolator.interpolate(epoch)
            errors.append(np.linalg.norm(state_history[epoch] - interpolated_state))
        maxerror = max(errors)
        fixed_step_errors.append(maxerror)
        func_evals = dynamics_simulator.cumulative_number_of_function_evaluations
        fixed_step_evaluations.append(func_evals[lastepoch])
        print('timestep:',timestep,'evaluations:',func_evals[lastepoch])

    fixed_step_integrators_error.append(fixed_step_errors)
    fixed_step_integrators_evaluations.append(fixed_step_evaluations)


    variable_step_errors = []
    variable_step_evaluations = []
    lowest_error = np.inf

    for tolerance in tolerances:
        settingsdescription = [integrators[integrator_index], tolerance]
        #settings_variable_steps.append(settingsdescription)

        step_size_control_settings = propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(
            tolerance,
            tolerance,
            minimum_factor_increase = 0.01,
            maximum_factor_increase = 100
        )

        step_size_validation_settings = propagation_setup.integrator.step_size_validation(
            0.0001,
            1000
            )

        integrator_settings = integrator.runge_kutta_variable_step(
        benchmark_step_size,
        integrators[integrator_index],
        step_size_control_settings,
        step_size_validation_settings
        )
        propagator_settings.integrator_settings = integrator_settings

        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies,
            propagator_settings )
        if not dynamics_simulator.integration_completed_successfully:
            continue
        
        state_history = dynamics_simulator.state_history

        errors = []
        lastepoch = list(state_history.keys())[-1]
        lastbenchmarkepoch = list(benchmark_state_history.keys())[-1]
        if lastepoch > lastbenchmarkepoch+30:
            continue
        for epoch in state_history.keys():
            if epoch > lastbenchmarkepoch - 10:
                continue
            interpolated_state = benchmark_interpolator.interpolate(epoch)
            errors.append(np.linalg.norm(state_history[epoch] - interpolated_state))
        maxerror = max(errors)
        variable_step_errors.append(maxerror)
        func_evals = dynamics_simulator.cumulative_number_of_function_evaluations
        variable_step_evaluations.append(func_evals[lastepoch])
        print('tolerance:',tolerance,'evaluations:',func_evals[lastepoch])

        if maxerror < lowest_error:
            lowest_error = maxerror
            beststatehistory = state_history
            bestsettings = settingsdescription

    variable_step_tests.append(beststatehistory)
    name = 'integrator: ' + str(bestsettings[0]) + ', tolerance: ' + str(bestsettings[1])
    variable_step_tests_names.append(name)

    variable_step_integrators_error.append(variable_step_errors)
    variable_step_integrators_evaluations.append(variable_step_evaluations)

###########################################################################
# PLOTTING ###############################################################
###########################################################################

fixed_step_integrator_names = ['RK4', 'RK5', 'RK7', 'RK10']
variable_step_integrator_names = ['RKF45', 'RKF56', 'RKF78', 'RKF1012']

#plot error against number of function evaluations
for i in range(len(fixed_step_integrators_error)):
    plt.plot(fixed_step_integrators_evaluations[i], fixed_step_integrators_error[i], label = fixed_step_integrator_names[i], linestyle = '-',marker = 'o')
    plt.plot(variable_step_integrators_evaluations[i], variable_step_integrators_error[i], label = variable_step_integrator_names[i], linestyle = '--', marker = 'x')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Number of function evaluations')
plt.ylabel('Error[m]')
plt.legend()
plt.grid()
plt.show()


#look at variable timestep errors
for i in range(len(variable_step_tests)):
    state_history = variable_step_tests[i]
    errors = []
    epochs = []
    lastepoch = list(state_history.keys())[-1]
    for epoch in state_history.keys():
        if epoch > lastbenchmarkepoch - 10:
            continue
        interpolated_state = benchmark_interpolator.interpolate(epoch)
        errors.append(np.linalg.norm(state_history[epoch] - interpolated_state))
        epochs.append(epoch)
    plt.plot(epochs, errors, label=variable_step_tests_names[i], linestyle = '--', marker = 'o')
plt.yscale('log')
plt.xlabel('Time [s]')
plt.ylabel('Error[m]')
plt.legend()
plt.grid()
plt.show()