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
timesteps = np.logspace(0, 6, num=13, base=2.0, dtype=float)
#timesteps = [1,2,4,8,16,32,64]
print(timesteps)

tolerances = [1e-5,5e-6, 1e-6,5e-7, 1e-7,5e-8, 1e-8,5e-9, 1e-9,5e-10, 1e-10]

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
bodies_to_create = ['Earth', 'Moon', 'Sun', 'Mars', 'Venus', 'Mercury', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
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

propagator_settings = Util.get_propagator_settings_benchmark(shape_parameters,
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

regular_step_size = 8.
regular_integrator_coefs = propagation_setup.integrator.CoefficientSets.rkf_56
regular_integrator = propagation_setup.integrator.runge_kutta_fixed_step(
    regular_step_size,
    regular_integrator_coefs
)



# set parameters for time at which initial data is extracted from spice
initial_time = 12345
# set parameters for defining the rotation between frames
original_frame = "J2000"
target_frame = "IAU_Earth_Simplified"
target_frame_spice = "IAU_Earth"
# create rotation model settings and assign to body settings of "Earth"
simple_from_spice = environment_setup.rotation_model.simple_from_spice(
original_frame, target_frame, target_frame_spice, initial_time)

# define parameters describing the rotation between frames
original_frame = "J2000"
target_frame = "IAU_Earth"
# create rotation model settings and assign to body settings of "Earth"
spice_direct = environment_setup.rotation_model.spice(
original_frame, target_frame)

precession_nutation_theory = environment_setup.rotation_model.IAUConventions.iau_2006
original_frame = "J2000"
# create rotation model settings and assign to body settings of "Earth"
high_accuracy = environment_setup.rotation_model.gcrs_to_itrs(
precession_nutation_theory, original_frame)

rotation_models = [simple_from_spice, spice_direct, high_accuracy]

# define parameters of an invariant exponential atmosphere model
density_at_zero_altitude = 1.225
density_scale_height = 7.2E3
constant_temperature = 290
# create atmosphere settings and add to body settings of "Earth"
exponential = environment_setup.atmosphere.exponential(
     density_scale_height, density_at_zero_altitude)

exponential_predefined = environment_setup.atmosphere.exponential_predefined("Earth")
us76 = environment_setup.atmosphere.us76()

nrlmsise00 = environment_setup.atmosphere.nrlmsise00()  

atmosphere_models = [exponential, exponential_predefined, us76, nrlmsise00]

state_histories_rotation = []

for rotation_model in rotation_models:
    body_settings.get("Earth").rotation_model_settings = rotation_model
    bodies = environment_setup.create_system_of_bodies(body_settings)

    propagator_settings = Util.get_propagator_settings_benchmark(shape_parameters,
                                                                 bodies,
                                                                 simulation_start_epoch,
                                                                termination_settings,
                                                                dependent_variables_to_save)



    propagator_settings.integrator_settings = regular_integrator

    regular_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        propagator_settings )
    
    regular_state_history = regular_dynamics_simulator.state_history

    state_histories_rotation.append(regular_state_history)

chosen_rotation_model_index = 2

body_settings.get("Earth").rotation_model_settings = rotation_models[chosen_rotation_model_index]

state_histories_atmosphere = []

#test out the different atmosphere models
for atmospheric_model in atmosphere_models:
    body_settings.get("Earth").atmosphere_settings = atmospheric_model
    bodies = environment_setup.create_system_of_bodies(body_settings)

    propagator_settings = Util.get_propagator_settings_benchmark(shape_parameters,
                                                                 bodies,
                                                                 simulation_start_epoch,
                                                                termination_settings,
                                                                dependent_variables_to_save)

    propagator_settings.integrator_settings = regular_integrator

    regular_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        propagator_settings )

    regular_state_history = regular_dynamics_simulator.state_history

    state_histories_atmosphere.append(regular_state_history)

chosen_atmosphere_model_index = 3

