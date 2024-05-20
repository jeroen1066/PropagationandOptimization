# General imports
import os
import datetime

# Tudatpy imports
from tudatpy.astro import element_conversion
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

benchmark_step_size = 8.

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

propagator_settings_00 = Util.get_propagator_settings_sphericalharmonics(shape_parameters,
                                                                 bodies,
                                                                 simulation_start_epoch,
                                                                 termination_settings,
                                                                 dependent_variables_to_save,
                                                                 0,
                                                                 0)
propagator_settings_20 = Util.get_propagator_settings_sphericalharmonics(shape_parameters,
                                                                    bodies,
                                                                    simulation_start_epoch,
                                                                    termination_settings,
                                                                    dependent_variables_to_save,
                                                                    2,
                                                                    0)
propagator_settings_22 = Util.get_propagator_settings_sphericalharmonics(shape_parameters,
                                                                    bodies,
                                                                    simulation_start_epoch,
                                                                    termination_settings,
                                                                    dependent_variables_to_save,
                                                                    2,
                                                                    2)
propagator_settings_44 = Util.get_propagator_settings_sphericalharmonics(shape_parameters,
                                                                    bodies,
                                                                    simulation_start_epoch,
                                                                    termination_settings,
                                                                    dependent_variables_to_save,
                                                                    4,
                                                                    4)
propagator_settings_88 = Util.get_propagator_settings_sphericalharmonics(shape_parameters,
                                                                    bodies,
                                                                    simulation_start_epoch,
                                                                    termination_settings,
                                                                    dependent_variables_to_save,
                                                                    8,
                                                                    8)
propagator_settings_1616 = Util.get_propagator_settings_sphericalharmonics(shape_parameters,
                                                                    bodies,
                                                                    simulation_start_epoch,
                                                                    termination_settings,
                                                                    dependent_variables_to_save,
                                                                    16,
                                                                    16)
propagator_settings_3232 = Util.get_propagator_settings_sphericalharmonics(shape_parameters,
                                                                    bodies,
                                                                    simulation_start_epoch,
                                                                    termination_settings,
                                                                    dependent_variables_to_save,
                                                                    32,
                                                                    32)
propagator_settings_6464 = Util.get_propagator_settings_sphericalharmonics(shape_parameters,
                                                                    bodies,
                                                                    simulation_start_epoch,
                                                                    termination_settings,
                                                                    dependent_variables_to_save,
                                                                    64,
                                                                    64)
propagator_settings_128128 = Util.get_propagator_settings_sphericalharmonics(shape_parameters,
                                                                    bodies,
                                                                    simulation_start_epoch,
                                                                    termination_settings,
                                                                    dependent_variables_to_save,
                                                                    128,
                                                                    128)
propagator_settings_200200 = Util.get_propagator_settings_sphericalharmonics(shape_parameters,
                                                                    bodies,
                                                                    simulation_start_epoch,
                                                                    termination_settings,
                                                                    dependent_variables_to_save,
                                                                    200,
                                                                    200)



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

propagator_settings_00.integrator_settings = integrator_settings
propagator_settings_20.integrator_settings = integrator_settings
propagator_settings_22.integrator_settings = integrator_settings
propagator_settings_44.integrator_settings = integrator_settings
propagator_settings_88.integrator_settings = integrator_settings
propagator_settings_1616.integrator_settings = integrator_settings
propagator_settings_3232.integrator_settings = integrator_settings
propagator_settings_6464.integrator_settings = integrator_settings
propagator_settings_128128.integrator_settings = integrator_settings
propagator_settings_200200.integrator_settings = integrator_settings

#propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
#    benchmark_step_size,
#    propagation_setup.integrator.CoefficientSets.rkf_56)
#propagator_settings_00.print_settings.print_dependent_variable_indices = True
starttime = datetime.datetime.now()
benchmark_dynamics_simulator_00 = numerical_simulation.create_dynamics_simulator(
    bodies,
    propagator_settings_00 )
after00 = datetime.datetime.now()
benchmark_dynamics_simulator_20 = numerical_simulation.create_dynamics_simulator(
    bodies,
    propagator_settings_20 )
after20 = datetime.datetime.now()
benchmark_dynamics_simulator_22 = numerical_simulation.create_dynamics_simulator(
    bodies,
    propagator_settings_22 )
after22 = datetime.datetime.now()
benchmark_dynamics_simulator_44 = numerical_simulation.create_dynamics_simulator(
    bodies,
    propagator_settings_44 )
after44 = datetime.datetime.now()
benchmark_dynamics_simulator_88 = numerical_simulation.create_dynamics_simulator(
    bodies,
    propagator_settings_88 )
after88 = datetime.datetime.now()
benchmark_dynamics_simulator_1616 = numerical_simulation.create_dynamics_simulator(
    bodies,
    propagator_settings_1616 )
after1616 = datetime.datetime.now()
benchmark_dynamics_simulator_3232 = numerical_simulation.create_dynamics_simulator(
    bodies,
    propagator_settings_3232 )
after3232 = datetime.datetime.now()
benchmark_dynamics_simulator_6464 = numerical_simulation.create_dynamics_simulator(
    bodies,
    propagator_settings_6464 )
after6464 = datetime.datetime.now()
benchmark_dynamics_simulator_128128 = numerical_simulation.create_dynamics_simulator(
    bodies,
    propagator_settings_128128 )
after128128 = datetime.datetime.now()
benchmark_dynamics_simulator_200200 = numerical_simulation.create_dynamics_simulator(
    bodies,
    propagator_settings_200200 )
after200200 = datetime.datetime.now()

print('point gravity propagation time: ', after00 - starttime)
print('2 0 gravity propagation time: ', after20 - after00)
print('2 2 gravity propagation time: ', after22 - after20)
print('4 4 gravity propagation time: ', after44 - after22)
print('8 8 gravity propagation time: ', after88 - after44)
print('16 16 gravity propagation time: ', after1616 - after88)
print('32 32 gravity propagation time: ', after3232 - after1616)
print('64 64 gravity propagation time: ', after6464 - after3232)
print('128 128 gravity propagation time: ', after128128 - after6464)
print('200 200 gravity propagation time: ', after200200 - after128128)



benchmark_state_history_00 = benchmark_dynamics_simulator_00.state_history
benchmark_state_history_20 = benchmark_dynamics_simulator_20.state_history
benchmark_state_history_22 = benchmark_dynamics_simulator_22.state_history
benchmark_state_history_44 = benchmark_dynamics_simulator_44.state_history
benchmark_state_history_88 = benchmark_dynamics_simulator_88.state_history
benchmark_state_history_1616 = benchmark_dynamics_simulator_1616.state_history
benchmark_state_history_3232 = benchmark_dynamics_simulator_3232.state_history
benchmark_state_history_6464 = benchmark_dynamics_simulator_6464.state_history
benchmark_state_history_128128 = benchmark_dynamics_simulator_128128.state_history
benchmark_state_history_200200 = benchmark_dynamics_simulator_200200.state_history



###########################################################################
# PLOTTING ################################################################
###########################################################################

# Plot the results
end = np.inf
if list(benchmark_state_history_00.keys())[-1] < end:
    end = list(benchmark_state_history_00.keys())[-1]
    timesteps = list(benchmark_state_history_00.keys())
if list(benchmark_state_history_20.keys())[-1] < end:
    end = list(benchmark_state_history_20.keys())[-1]
    timesteps = list(benchmark_state_history_20.keys())
if list(benchmark_state_history_22.keys())[-1] < end:
    end = list(benchmark_state_history_22.keys())[-1]
    timesteps = list(benchmark_state_history_22.keys())
if list(benchmark_state_history_44.keys())[-1] < end:
    end = list(benchmark_state_history_44.keys())[-1]
    timesteps = list(benchmark_state_history_44.keys())
if list(benchmark_state_history_88.keys())[-1] < end:
    end = list(benchmark_state_history_88.keys())[-1]
    timesteps = list(benchmark_state_history_88.keys())
if list(benchmark_state_history_1616.keys())[-1] < end:
    end = list(benchmark_state_history_1616.keys())[-1]
    timesteps = list(benchmark_state_history_1616.keys())
if list(benchmark_state_history_3232.keys())[-1] < end:
    end = list(benchmark_state_history_3232.keys())[-1]
    timesteps = list(benchmark_state_history_3232.keys())
if list(benchmark_state_history_6464.keys())[-1] < end:
    end = list(benchmark_state_history_6464.keys())[-1]
    timesteps = list(benchmark_state_history_6464.keys())
if list(benchmark_state_history_128128.keys())[-1] < end:
    end = list(benchmark_state_history_128128.keys())[-1]
    timesteps = list(benchmark_state_history_128128.keys())
if list(benchmark_state_history_200200.keys())[-1] < end:
    end = list(benchmark_state_history_200200.keys())[-1]
    timesteps = list(benchmark_state_history_200200.keys())




improvement_20_radial = []
improvement_22_radial = []
improvement_44_radial = []
improvement_88_radial = []
improvement_1616_radial = []
improvement_3232_radial = []
improvement_6464_radial = []
improvement_128128_radial = []
improvement_200200_radial = []

improvement_20_velocity = []
improvement_22_velocity = []
improvement_44_velocity = []
improvement_88_velocity = []
improvement_1616_velocity = []
improvement_3232_velocity = []
improvement_6464_velocity = []
improvement_128128_velocity = []
improvement_200200_velocity = []

radii00 = []
radii20 = []
radii22 = []
radii44 = []
radii88 = []
radii1616 = []
radii3232 = []
radii6464 = []
radii128128 = []
radii200200 = []

cartesianerror_00 = []
cartesianerror_20 = []
cartesianerror_22 = []
cartesianerror_44 = []
cartesianerror_88 = []
cartesianerror_1616 = []
cartesianerror_3232 = []
cartesianerror_6464 = []
cartesianerror_128128 = []



for step in timesteps:
    cartesianerror_00.append(np.linalg.norm(benchmark_state_history_00[step] - benchmark_state_history_200200[step]))
    cartesianerror_20.append(np.linalg.norm(benchmark_state_history_20[step] - benchmark_state_history_200200[step]))
    cartesianerror_22.append(np.linalg.norm(benchmark_state_history_22[step] - benchmark_state_history_200200[step]))
    cartesianerror_44.append(np.linalg.norm(benchmark_state_history_44[step] - benchmark_state_history_200200[step]))
    cartesianerror_88.append(np.linalg.norm(benchmark_state_history_88[step] - benchmark_state_history_200200[step]))
    cartesianerror_1616.append(np.linalg.norm(benchmark_state_history_1616[step] - benchmark_state_history_200200[step]))
    cartesianerror_3232.append(np.linalg.norm(benchmark_state_history_3232[step] - benchmark_state_history_200200[step]))
    cartesianerror_6464.append(np.linalg.norm(benchmark_state_history_6464[step] - benchmark_state_history_200200[step]))
    cartesianerror_128128.append(np.linalg.norm(benchmark_state_history_128128[step] - benchmark_state_history_200200[step]))


    sphericalstate_00 = element_conversion.cartesian_to_spherical(benchmark_state_history_00[step])
    sphericalstate_20 = element_conversion.cartesian_to_spherical(benchmark_state_history_20[step])
    sphericalstate_22 = element_conversion.cartesian_to_spherical(benchmark_state_history_22[step])
    sphericalstate_44 = element_conversion.cartesian_to_spherical(benchmark_state_history_44[step])
    sphericalstate_88 = element_conversion.cartesian_to_spherical(benchmark_state_history_88[step])
    sphericalstate_1616 = element_conversion.cartesian_to_spherical(benchmark_state_history_1616[step])
    sphericalstate_3232 = element_conversion.cartesian_to_spherical(benchmark_state_history_3232[step])
    sphericalstate_6464 = element_conversion.cartesian_to_spherical(benchmark_state_history_6464[step])
    sphericalstate_128128 = element_conversion.cartesian_to_spherical(benchmark_state_history_128128[step])
    sphericalstate_200200 = element_conversion.cartesian_to_spherical(benchmark_state_history_200200[step])

    

    radii00.append(sphericalstate_00[0])
    radii20.append(sphericalstate_20[0])
    radii22.append(sphericalstate_22[0])
    radii44.append(sphericalstate_44[0])
    radii88.append(sphericalstate_88[0])
    radii1616.append(sphericalstate_1616[0])
    radii3232.append(sphericalstate_3232[0])
    radii6464.append(sphericalstate_6464[0])
    radii128128.append(sphericalstate_128128[0])
    radii200200.append(sphericalstate_200200[0])

    improvement_20_radial.append(np.abs(sphericalstate_00[0] - sphericalstate_200200[0]))
    improvement_22_radial.append(np.abs(sphericalstate_20[0] - sphericalstate_200200[0]))
    improvement_44_radial.append(np.abs(sphericalstate_22[0] - sphericalstate_200200[0]))
    improvement_88_radial.append(np.abs(sphericalstate_44[0] - sphericalstate_200200[0]))
    improvement_1616_radial.append(np.abs(sphericalstate_88[0] - sphericalstate_200200[0]))
    improvement_3232_radial.append(np.abs(sphericalstate_1616[0] - sphericalstate_200200[0]))
    improvement_6464_radial.append(np.abs(sphericalstate_3232[0] - sphericalstate_200200[0]))
    improvement_128128_radial.append(np.abs(sphericalstate_6464[0] - sphericalstate_200200[0]))
    improvement_200200_radial.append(np.abs(sphericalstate_128128[0] - sphericalstate_200200[0]))

    improvement_20_velocity.append(np.abs(sphericalstate_00[3] - sphericalstate_200200[3]))
    improvement_22_velocity.append(np.abs(sphericalstate_20[3] - sphericalstate_200200[3]))
    improvement_44_velocity.append(np.abs(sphericalstate_22[3] - sphericalstate_200200[3]))
    improvement_88_velocity.append(np.abs(sphericalstate_44[3] - sphericalstate_200200[3]))
    improvement_1616_velocity.append(np.abs(sphericalstate_88[3] - sphericalstate_200200[3]))
    improvement_3232_velocity.append(np.abs(sphericalstate_1616[3] - sphericalstate_200200[3]))
    improvement_6464_velocity.append(np.abs(sphericalstate_3232[3] - sphericalstate_200200[3]))
    improvement_128128_velocity.append(np.abs(sphericalstate_6464[3] - sphericalstate_200200[3]))
    improvement_200200_velocity.append(np.abs(sphericalstate_128128[3] - sphericalstate_200200[3]))

                                       
                                     
plt.plot(timesteps, radii00, label='00')
plt.plot(timesteps, radii20, label='20')
plt.plot(timesteps, radii22, label='22')
plt.plot(timesteps, radii44, label='44')
plt.plot(timesteps, radii88, label='88')
plt.plot(timesteps, radii1616, label='1616')
plt.plot(timesteps, radii3232, label='3232')
plt.plot(timesteps, radii6464, label='6464')
plt.plot(timesteps, radii128128, label='128128')
plt.plot(timesteps, radii200200, label='200200')

plt.yscale('log')
plt.legend()
plt.grid()
plt.show()

plt.plot(timesteps, cartesianerror_00, label='00')
plt.plot(timesteps, cartesianerror_20, label='20')
plt.plot(timesteps, cartesianerror_22, label='22')
plt.plot(timesteps, cartesianerror_44, label='44')
plt.plot(timesteps, cartesianerror_88, label='88')
plt.plot(timesteps, cartesianerror_1616, label='1616')
plt.plot(timesteps, cartesianerror_3232, label='3232')
plt.plot(timesteps, cartesianerror_6464, label='6464')
plt.plot(timesteps, cartesianerror_128128, label='128128')

plt.yscale('log')
plt.legend()
plt.grid()
plt.show()

plt.plot(timesteps, improvement_20_radial, label='00')
plt.plot(timesteps, improvement_22_radial, label='20')
plt.plot(timesteps, improvement_44_radial, label='22')
plt.plot(timesteps, improvement_88_radial, label='44')
plt.plot(timesteps, improvement_1616_radial, label='88')
plt.plot(timesteps, improvement_3232_radial, label='1616')
plt.plot(timesteps, improvement_6464_radial, label='3232')
plt.plot(timesteps, improvement_128128_radial, label='6464')
plt.plot(timesteps, improvement_200200_radial, label='128128')

plt.yscale('log')
plt.legend()
plt.grid()
plt.show()

plt.plot(timesteps, improvement_20_velocity, label='00')
plt.plot(timesteps, improvement_22_velocity, label='20')
plt.plot(timesteps, improvement_44_velocity, label='22')
plt.plot(timesteps, improvement_88_velocity, label='44')
plt.plot(timesteps, improvement_1616_velocity, label='88')
plt.plot(timesteps, improvement_3232_velocity, label='1616')
plt.plot(timesteps, improvement_6464_velocity, label='3232')
plt.plot(timesteps, improvement_128128_velocity, label='6464')
plt.plot(timesteps, improvement_200200_velocity, label='128128')

plt.yscale('log')
plt.legend()
plt.grid() 
plt.show()