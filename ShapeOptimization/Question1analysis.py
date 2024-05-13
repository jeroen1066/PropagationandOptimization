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

time_exponents = np.arange(-3, 6, 1)
time_steps = np.power(2., time_exponents)



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

propagator = propagation_setup.propagator.cowell

integrator_index = 1
state_histories = []
differences = []
interpolation_errors = []

for current_time_step in time_steps:
    current_integrator_settings = Util.get_integrator_settings(0,
                                                                            integrator_index,
                                                                            current_time_step,
                                                                            simulation_start_epoch)

    coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_56
    integrator = propagation_setup.integrator
    integrator_settings = integrator.runge_kutta_fixed_step_size(
            current_time_step,
            coefficient_set,
            integrator.OrderToIntegrate.lower
    )

    propagator_settings.integrator_settings = integrator_settings

    returnlist = Util.generate_benchmarks_rk5(
        current_time_step,
        simulation_start_epoch,
        bodies,
        propagator_settings,
        False
    )

    times_comparison = returnlist[0].keys()
    comparison_end =  list(times_comparison)[-1]

    states = returnlist[1]
    comparisonstates = returnlist[0]
    
    times = states.keys()
    error = {}
    for time in times:
        if time > comparison_end:
            continue
        error[time] = np.linalg.norm(states[time] - comparisonstates[time])
    
    differences.append(error)

    interpolation_error = {}

    interpolator_settings = interpolators.lagrange_interpolation( 8 )
    interpolation = interpolators.create_one_dimensional_vector_interpolator( states, interpolator_settings )

    for time in times_comparison:
        if time in times:
            continue
        interpolated_state = interpolation.interpolate(time)
        interpolation_error[time] = np.linalg.norm(interpolated_state - comparisonstates[time])

    interpolation_errors.append(interpolation_error)

    

    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                    bodies, propagator_settings )
    
    state_history = dynamics_simulator.state_history
    state_histories.append(state_history) 
    

###########################################################################
# PLOT RESULTS ############################################################
###########################################################################

plt.figure()
for i in range( len(differences)):
    plt.plot(list(differences[i].keys()), list(differences[i].values()), label=str(time_steps[i]*2))
plt.axhline(10,color='black', label= 'required accuracy')
plt.xlabel('Time [s]')
plt.ylabel('Difference in state [m]')
plt.grid()
plt.yscale('log')
plt.legend()
plt.show()
    
maxerrors = []
for i in range(len(time_steps)):
    states_differences = differences[i]
    maxerror = np.max(np.array(list(states_differences.values())))
    maxerrors.append(maxerror)

plt.plot(time_steps*2, maxerrors, 'o-')
plt.xlabel('Time step [s]')
plt.ylabel('Maximum error [m]')
plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.show()

plt.figure()
for i in range( len(interpolation_errors)):
    plt.plot(list(interpolation_errors[i].keys()), list(interpolation_errors[i].values()), label=str(time_steps[i]*2))
plt.xlabel('Time [s]')
plt.ylabel('Interpolation error [m]')
plt.grid()
plt.yscale('log')
plt.legend()
plt.show()