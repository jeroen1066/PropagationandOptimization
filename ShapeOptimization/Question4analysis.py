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

end_benchmark = list(benchmark_state_history.keys())[-1]

###########################################################################
# REGULAR SIMULATION ######################################################
###########################################################################

regular_step_size = 8.
regular_integrator_coefs = propagation_setup.integrator.CoefficientSets.rkf_56
regular_integrator = propagation_setup.integrator.runge_kutta_fixed_step(
    regular_step_size,
    regular_integrator_coefs
)

perturbing_bodies = ['Moon', 'Sun', 'Jupiter']


# set parameters for time at which initial data is extracted from spice
initial_time = 12345
# set parameters for defining the rotation between frames
original_frame = "J2000"
target_frame = "IAU_Earth"
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

rotation_models = [simple_from_spice, spice_direct]

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

state_errors_rotation = []

for rotation_model in rotation_models:
    body_settings.get("Earth").rotation_model_settings = rotation_model
    bodies = environment_setup.create_system_of_bodies(body_settings)
    Util.add_capsule_to_body_system(bodies,
                                shape_parameters,
                                capsule_density)

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

    errors = {}

    for epoch in regular_state_history.keys():
        if epoch > end_benchmark:
            continue
        errors[epoch] = np.linalg.norm(regular_state_history[epoch] - benchmark_interpolator.interpolate(epoch))

    state_errors_rotation.append(errors)

chosen_rotation_model_index = 1

body_settings.get("Earth").rotation_model_settings = rotation_models[chosen_rotation_model_index]

state_errors_atmosphere = []

#test out the different atmosphere models
for atmospheric_model in atmosphere_models:
    body_settings.get("Earth").atmosphere_settings = atmospheric_model
    bodies = environment_setup.create_system_of_bodies(body_settings)
    Util.add_capsule_to_body_system(bodies,
                                shape_parameters,
                                capsule_density)

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

    errors = {}

    for epoch in regular_state_history.keys():
        if epoch > end_benchmark:
            continue
        errors[epoch] = np.linalg.norm(regular_state_history[epoch] - benchmark_interpolator.interpolate(epoch))

    state_errors_atmosphere.append(errors)

chosen_atmosphere_model_index = 3

# Define bodies that are propagated and their central bodies of propagation
bodies_to_propagate = ['Capsule']
central_bodies = ['Earth']

# Define accelerations acting on capsule
acceleration_settings_on_vehicle = {
    'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(64,64),
                propagation_setup.acceleration.aerodynamic()],
    'Moon': [propagation_setup.acceleration.point_mass_gravity()],
    'Sun': [propagation_setup.acceleration.point_mass_gravity()],
    'Jupiter': [propagation_setup.acceleration.point_mass_gravity()]
}
# Create acceleration models.

perturbing_errors = []
for perturbing_body in perturbing_bodies:
    current_acceleration_settings = {
        'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(64,64),
                    propagation_setup.acceleration.aerodynamic()]
    }
    current_perturbing_bodies =[]
    for perturbing_body_inner in perturbing_bodies:
        if perturbing_body_inner != perturbing_body:
            current_acceleration_settings[perturbing_body_inner] = [propagation_setup.acceleration.point_mass_gravity()]
            current_perturbing_bodies.append(perturbing_body_inner)

    print('for the analysis of perturbing body',perturbing_body, 'the other perturbing bodies are:')
    print(current_perturbing_bodies)

    print(current_acceleration_settings)

    acceleration_settings = {'Capsule': current_acceleration_settings}
    acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)

    new_angles = np.array([shape_parameters[5], 0.0, 0.0])
    new_angle_function = lambda time : new_angles
    bodies.get_body('Capsule').rotation_model.reset_aerodynamic_angle_function( new_angle_function )




    # Retrieve initial state
    initial_state = Util.get_initial_state(simulation_start_epoch,bodies)

    # Create propagation settings for the translational dynamics. NOTE: these are not yet 'valid', as no
    # integrator settings are defined yet
    current_propagator = propagation_setup.propagator.cowell
    propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                     acceleration_models,
                                                                     bodies_to_propagate,
                                                                     initial_state,
                                                                     simulation_start_epoch,
                                                                     None,
                                                                     termination_settings,
                                                                     current_propagator,
                                                                     output_variables=dependent_variables_to_save)
    propagator_settings.integrator_settings = regular_integrator

    regular_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        propagator_settings )
    
    errors = {}

    regular_state_history = regular_dynamics_simulator.state_history
    for epoch in regular_state_history.keys():
        if epoch > end_benchmark:
            continue
        errors[epoch] = np.linalg.norm(regular_state_history[epoch] - benchmark_interpolator.interpolate(epoch))
    perturbing_errors.append(errors)

print('the number of errors is', len(perturbing_errors))

###########################################################################
# PLOTTING ################################################################
###########################################################################

# Plot the results

for errors in state_errors_rotation:

    plt.plot(errors.keys(),errors.values())
plt.xlabel('Time [s]')
plt.ylabel('State error [m]')
plt.title('State error for rotation model')
plt.yscale('log')
plt.grid()
plt.show()

for errors in state_errors_atmosphere:
    
    plt.plot(errors.keys(),errors.values())
plt.xlabel('Time [s]')
plt.ylabel('State error [m]')
plt.title('State error for atmosphere model')
plt.yscale('log')
plt.grid()
plt.show()

for errors in perturbing_errors:
    plt.plot(errors.keys(),errors.values())
    print('oneplot')
plt.xlabel('Time [s]')
plt.ylabel('State error [m]')
plt.title('State error for perturbing bodies')
plt.yscale('log')
plt.grid()
plt.show()
