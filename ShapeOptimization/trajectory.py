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
from tudatpy.util import result2array

# Problem-specific imports
import CapsuleEntryUtilities as Util

# General python imports
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle

import pygmo as pg

import matplotlib.pyplot as plt

data_file = 'results/tuning_v2_maco/maco_tuned_seed_169.dat'
file = open(data_file,'rb')
data = pickle.load(file)[0]
file.close()
inputs = data[0]

spice_interface.load_standard_kernels()

simulation_start_epoch = 0.0

bodies_to_create = ['Earth', 'Moon', 'Sun']

# Define coordinate system
global_frame_origin = 'Earth'
global_frame_orientation = 'J2000'

# Create body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

nrlmsise00 = environment_setup.atmosphere.nrlmsise00()
equitorial_radius = 6378137.0
flattening = 1 / 298.25
body_settings.get('Earth').shape_settings = environment_setup.shape.oblate_spherical(equitorial_radius, flattening)

body_settings.get('Earth').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
    'Earth', simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Sun'),
    frame_orientation='J2000')
body_settings.get('Moon').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
    'Moon', simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Earth'),
    frame_orientation='J2000')
body_settings.get('Sun').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
    'Sun', simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Sun'),
    frame_orientation='J2000')

body_settings.get('Earth').atmosphere_settings = nrlmsise00

# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Retrieve termination settings
maximum_duration = constants.JULIAN_DAY  # s
termination_altitude = 25.0E3  # m
termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                          maximum_duration,
                                                          termination_altitude)

dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration_norm('Capsule'),
    propagation_setup.dependent_variable.density('Capsule', 'Earth'),
    propagation_setup.dependent_variable.airspeed('Capsule', 'Earth'),
    propagation_setup.dependent_variable.altitude('Capsule', 'Earth'),
    propagation_setup.dependent_variable.mach_number('Capsule', 'Earth')]

integrator_settings = Util.get_integrator_settings(3,3,1e-8,simulation_start_epoch)

initial_state = Util.get_initial_state(simulation_start_epoch,bodies)

fig, axs = plt.subplots(2,2)

bounce_fitnesses = []
counter = 0
for i in range(len(inputs)):
    shape_parameters = inputs[i][:6]
    density = inputs[i][6]

    Util.add_capsule_to_body_system(bodies,shape_parameters,density)

    propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                       bodies,
                                                       simulation_start_epoch,
                                                       termination_settings,
                                                       dependent_variables_to_save)

    propagator_settings.integrator_settings = integrator_settings

    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        propagator_settings)

    state_history = dynamics_simulator.state_history
    dependent_variable_history = dynamics_simulator.dependent_variable_history
    dependent_varibable_array = result2array(dependent_variable_history)
    times = np.array(list(state_history.keys()))
    if data[1][i][0] <= 8:
        if data[1][i][1] <= 8:
            if data[1][i][2] <= 8:
                counter += 1
                g_load = dependent_varibable_array[:, 1] / 9.81
                density = dependent_varibable_array[:, 2]
                airspeed = dependent_varibable_array[:, 3]
                altitude = dependent_varibable_array[:, 4]
                mach = dependent_varibable_array[:, 5]

                n = 0.2
                heatflux = (density ** (1 - n) * airspeed ** 3) / (shape_parameters[0] ** n)

                axs[(0, 0)].plot(times, altitude, label='run ' + str(i + 1))
                axs[(0, 0)].grid()
                axs[(0, 0)].set_xlabel('time [s]')
                axs[(0, 0)].set_ylabel('altitude [m]')
                axs[(0, 0)].legend()

                axs[(1, 0)].plot(times, g_load, label='run ' + str(i + 1))
                axs[(1, 0)].grid()
                axs[(1, 0)].set_xlabel('time [s]')
                axs[(1, 0)].set_ylabel('g-load [g]')
                axs[(1, 0)].legend()

                axs[(0, 1)].plot(times, mach, label='run ' + str(i + 1))
                axs[(0, 1)].grid()
                axs[(0, 1)].set_xlabel('time [s]')
                axs[(0, 1)].set_ylabel('Mach number [m]')
                axs[(0, 1)].legend()

                axs[(1, 1)].plot(times, heatflux, label='run ' + str(i + 1))
                axs[(1, 1)].grid()
                axs[(1, 1)].set_xlabel('time [s]')
                axs[(1, 1)].set_ylabel('heat flux [m]')
                axs[(1, 1)].legend()

axs[(0,0)].grid()
axs[(1,0)].grid()
axs[(0,1)].grid()
axs[(1,1)].grid()
plt.show()
print(counter)