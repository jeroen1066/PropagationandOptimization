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
propagator_settings.print_settings.print_dependent_variable_indices = True

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

available_propagators = [propagation_setup.propagator.cowell,
                             propagation_setup.propagator.encke,
                             propagation_setup.propagator.gauss_keplerian,
                             propagation_setup.propagator.gauss_modified_equinoctial,
                             propagation_setup.propagator.unified_state_model_quaternions,
                             propagation_setup.propagator.unified_state_model_modified_rodrigues_parameters,
                             propagation_setup.propagator.unified_state_model_exponential_map]

available_propagators_names = ['cowell',
                               'encke',
                                 'gauss_keplerian',
                                    'gauss equinoctial',
                                    'USM-7',
                                    'USM-6',
                                    'USM-EM']

cowell_variable_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
encke_variable_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
gauss_keplerian_variable_names = ['a', 'e', 'i', 'RAAN', 'AOP', 'TA']
gauss_modified_equinoctial_variable_names = ['semiParameter', 'h1', 'h2', 'k1', 'k2', 'trueLongitude']
unified_state_model_quaternions_variable_names = ['x', 'y', 'z', 'q1', 'q2', 'q3', 'q4']
unified_state_model_modified_rodrigues_parameters_variable_names = ['x', 'y', 'z', 'q1', 'q2', 'q3', 'q4']
unified_state_model_exponential_map_variable_names = ['x', 'y', 'z', 'q1', 'q2', 'q3', 'q4']

variable_names = [cowell_variable_names,
                     encke_variable_names,
                     gauss_keplerian_variable_names,
                     gauss_modified_equinoctial_variable_names,
                     unified_state_model_quaternions_variable_names,
                     unified_state_model_modified_rodrigues_parameters_variable_names,
                     unified_state_model_exponential_map_variable_names]


errors_for_propagators = []
propagated_states = []

for propagator in available_propagators:
    propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                   bodies,
                                                                   simulation_start_epoch,
                                                                   termination_settings,
                                                                   dependent_variables_to_save,
                                                                   propagator )
    
    coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_56
    integrator = propagation_setup.integrator
    integrator_settings = integrator.runge_kutta_fixed_step(
            regular_step_size,
            coefficient_set,
            integrator.OrderToIntegrate.lower
    )

    propagator_settings.integrator_settings = integrator_settings
    #propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
    #    benchmark_step_size,
    #    propagation_setup.integrator.CoefficientSets.rkf_56)
    
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        propagator_settings )
    
    state_history = dynamics_simulator.state_history
    states_as_propagated = dynamics_simulator.unprocessed_state_history
    propagated_states.append(states_as_propagated)
    errors = {}
    end = list(state_history.keys())[-1]
    for epoch in state_history.keys():
        if epoch > end - 10 * benchmark_step_size:
            continue
        benchmark_state = benchmark_interpolator.interpolate(epoch)
        state = state_history[epoch]
        difference = np.linalg.norm(benchmark_state - state)
        errors[epoch] = difference

    errors_for_propagators.append(errors)


###########################################################################
# PLOT RESULTS ############################################################
###########################################################################

plt.figure()
for errors in errors_for_propagators:
    plt.plot(errors.keys(), errors.values(),label=available_propagators_names[errors_for_propagators.index(errors)])
plt.yscale('log')
plt.xlabel('Time [s]')
plt.ylabel('Difference')
plt.title('Difference between benchmark and regular simulation')
plt.legend()
plt.grid()
plt.show()



for i in range(len(available_propagators)):
    unprocessed_state = propagated_states[i]
    times = list(unprocessed_state.keys())
    states = np.array(list(unprocessed_state.values()))
    state_names = variable_names[i]
    fig,axs = plt.subplots(2,3)
    for j in range(6):
        state = states[:,j]
        axs[j//3,j%3].plot(times,state)
        axs[j//3,j%3].set_title('State ' +state_names[j])
        axs[j//3,j%3].grid()
        fig.suptitle(available_propagators_names[i])
        #axs[j//2,j%2].title = available_propagators_names[i]
    plt.show()