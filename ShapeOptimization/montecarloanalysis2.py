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
import datetime
import pickle

spice_interface.load_standard_kernels()

###########################################################################
# PARAMETERS ##############################################################
###########################################################################

numsimulations = 500
numparameters = 7

n = 0.2 #0.2 for turbulent boundary, 0.5 for laminar boundary

#constraints
heat_flux_constraint = 5e6
heat_load_constraint = 200e6
stability_constraint = 0

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
body_settings.get('Earth').shape_settings = environment_setup.shape.oblate_spherical(
    equitorial_radius, flattening)

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
termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                     maximum_duration,
                                                     termination_altitude)
# Retrieve dependent variables to save
dependent_variables_to_save = Util.get_dependent_variable_save_settings()
# Check whether there is any
are_dependent_variables_to_save = False if not dependent_variables_to_save else True

propagator = propagation_setup.propagator.cowell

integrator_settings = Util.get_integrator_settings(3,
                                                    3,
                                                    1e-8,
                                                    simulation_start_epoch)

initial_state = Util.get_initial_state(simulation_start_epoch,bodies)

###########################################################################
# MONTE CARLO ANALYSIS ####################################################
###########################################################################

default_shape_parameters = [7.6983117481,
                    2.0923385955,
                    1.7186406196,
                    -0.255984141,
                    0.1158838553,
                    0.3203083369]
np.random.seed(42)
range_per_parameter = [[0,20],
                       [0,5],
                       [0,5],
                        [-1,1],
                        [-1,1],
                        [-1,1],
                        [150,500]
                       ]

inputs = np.empty((numparameters,numsimulations),dtype=object)
#time_histories = np.empty((numparameters,numsimulations),dtype=object)
objectives = np.empty((numparameters,numsimulations),dtype=object)
constraints = np.empty((numparameters,numsimulations),dtype=object)

parameters = np.empty((numparameters,numsimulations),dtype=float)

for i in range (numparameters):

    range_of_parameters = np.random.uniform(range_per_parameter[i][0],range_per_parameter[i][1],numsimulations)
    parameters[i] = range_of_parameters

starttime = datetime.datetime.now()
for j in range (numsimulations):
    
    shape_parameters = default_shape_parameters
    shape_parameters = parameters[:6,j]

    shape_parameters = default_shape_parameters
    capsule_density = parameters[-1,j]


    # Create shape model
    Util.add_capsule_to_body_system(bodies,
                            shape_parameters,
                            capsule_density)
    
    propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                bodies,
                                                                simulation_start_epoch,
                                                                termination_settings,
                                                                dependent_variables_to_save )
    
    propagator_settings.integrator_settings = integrator_settings
    
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies,
    propagator_settings )

    state_history = dynamics_simulator.state_history
    dependent_variable_history = dynamics_simulator.dependent_variable_history
    state_history_array = np.array(list(state_history.values()))
    dependent_variable_history_array = np.array(list(dependent_variable_history.values()))
    times = np.array(list(state_history.keys()))
    inputs[i,j] = shape_parameters

    #dependent variable will eventaully have 5 members: 
    # 3d aerodynamic acceleration,
    # 1d g-force,
    # heat flux
    # skipping is defined by having a height increase during propagation
    max_ld = 0
    max_g_load = 0
    max_heat_flux = 0
    total_heat_load = 0
    has_skipped = False
    stability = -1000
    
    for t in times:
        aerodynamic_acceleration = dependent_variable_history[t][:3]
        velocity = state_history[t][3:]
        velocitynorm = np.linalg.norm(velocity)
        g_load = np.linalg.norm(aerodynamic_acceleration)/9.81
        density = dependent_variable_history[t][4]
        heat_flux = (density**(1-n)*velocitynorm**3)/(shape_parameters[0]**n)

        drag = np.dot(aerodynamic_acceleration,velocity)/np.linalg.norm(velocity)
        helpervec = state_history[t][:3]/np.linalg.norm(state_history[t][:3])
        orthogonal = np.cross(helpervec,velocity)
        lift_direction = np.cross(velocity,orthogonal)
        lift = np.dot(aerodynamic_acceleration,lift_direction)/np.linalg.norm(lift_direction)

        if g_load > max_g_load:
            max_g_load = g_load
        if heat_flux > max_heat_flux:
            max_heat_flux = heat_flux
        if drag > 0.1:
            if lift/drag > max_ld:
                max_ld = lift/drag
        total_heat_load += heat_flux
        if velocity[2] > 0:
            has_skipped = True
    
    objectives[i,j] = [max_ld,max_g_load]
    constraints[i,j] = [max_heat_flux,total_heat_load,has_skipped]
    if j % 100 == 0:
        print('Time for 100 simulations: ',datetime.datetime.now()-starttime)
        print('Number of simulations done: ',j, 'out of ',numsimulations)
        starttime = datetime.datetime.now()

within_constraints = []
outside_constraints = []

for j in range(numsimulations):
    within_heat_flux = constraints[i,j][0] < heat_flux_constraint
    within_heat_load = constraints[i,j][1] < heat_load_constraint
    within_stability = constraints[i,j][2] < stability_constraint
    within_skip = constraints[i,j][3] == False

    if within_heat_flux and within_heat_load and within_stability and within_skip:
        within_constraints.append([inputs[i,j],objectives[i,j]])
    else:
        outside_constraints.append([inputs[i,j],objectives[i,j]])

savedata = [within_constraints,outside_constraints]

with open('mcdata','wb') as f:
    pickle.dump(savedata,f)


    





    