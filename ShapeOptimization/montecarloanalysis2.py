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

plot = True

numsimulations = 1000
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


np.random.seed(42)
range_per_parameter = [[0,5],
                       [2,5],
                       [0,3],
                        [-np.pi/2,0.],
                        [2.5,5.5],
                        [-0.25,0.5],
                        [50,400]
                       ]
invalid_inputs = []
inputs = np.empty((numsimulations),dtype=object)
#time_histories = np.empty((numparameters,numsimulations),dtype=object)
objectives = np.empty((numsimulations),dtype=object)
constraints = np.empty((numsimulations),dtype=object)

parameters = np.empty((numparameters,numsimulations),dtype=float)

for i in range (numparameters):

    range_of_parameters = np.random.uniform(range_per_parameter[i][0],range_per_parameter[i][1],numsimulations)
    parameters[i] = range_of_parameters

print(parameters[:,68])
#print(parameters[:,100])

starttime = datetime.datetime.now()
for j in range (numsimulations):
    if j == 68:
        print('Starting simulation 68')
    #    continue
    
    shape_parameters = parameters[:6,j]
    if j == 68:
        print(shape_parameters)

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
    
    if shape_parameters[2] < 0:
        print('Negative length')
        invalid_inputs.append(parameters[:,j])
        continue
    if shape_parameters[1] < 0:
        invalid_inputs.append(parameters[:,j])
        print('Negative radius')
        continue
    
    propagator_settings.integrator_settings = integrator_settings
    
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies,
    propagator_settings )

    state_history = dynamics_simulator.state_history
    dependent_variable_history = dynamics_simulator.dependent_variable_history
    state_history_array = np.array(list(state_history.values()))
    dependent_variable_history_array = np.array(list(dependent_variable_history.values()))
    times = np.array(list(state_history.keys()))
    input_values = parameters[:,j]
    inputs[j] = input_values

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
    mass_capsule = bodies.get_body('Capsule').mass
    volume_capsule = mass_capsule/capsule_density
    last_t = times[0]
    succesfull_completion = dynamics_simulator.integration_completed_successfully

    
    for t in times:
        timestep = t - last_t
        aerodynamic_acceleration = dependent_variable_history[t][:3]
        airspeed = dependent_variable_history[t][3]
        velocity = state_history[t][3:]
        velocitynorm = np.linalg.norm(velocity)
        g_load = np.linalg.norm(aerodynamic_acceleration)/9.81
        density = dependent_variable_history[t][4]
        heat_flux = (density**(1-n)*airspeed**3)/(shape_parameters[0]**n)

        drag = -np.dot(aerodynamic_acceleration,velocity)/np.linalg.norm(velocity)
        helpervec = state_history[t][:3]/np.linalg.norm(state_history[t][:3])
        orthogonal = np.cross(helpervec,velocity)
        lift_direction = np.cross(velocity,orthogonal)
        lift = np.dot(aerodynamic_acceleration,lift_direction)/np.linalg.norm(lift_direction)

        if g_load > max_g_load:
            max_g_load = g_load
        if heat_flux > max_heat_flux:
            max_heat_flux = heat_flux
        if drag > 0.0001:
            if lift/drag > max_ld:
                max_ld = lift/drag
        total_heat_load += heat_flux*timestep
        if np.inner(velocity,state_history[t][:3]) > 0:
            has_skipped = True
        last_t = t
    
    objectives[j] = [mass_capsule,max_ld,max_g_load]
    constraints[j] = [max_heat_flux,total_heat_load,stability,has_skipped,succesfull_completion]
    if j+1 % 100 == 0:
        print('Time for 100 simulations: ',datetime.datetime.now()-starttime)
        print('Number of simulations done: ',j+1, 'out of ',numsimulations)
        starttime = datetime.datetime.now()

within_constraints = []
outside_constraints = []

for j in range(numsimulations):
    if constraints[j] == None:
        continue
    within_heat_flux = constraints[j][0] < heat_flux_constraint
    within_heat_load = constraints[j][1] < heat_load_constraint
    within_stability = constraints[j][2] < stability_constraint
    within_skip = constraints[j][3] == False
    succesfull_completion = constraints[j][4]

    if within_heat_flux and within_heat_load and within_stability and within_skip and succesfull_completion:
        within_constraints.append([inputs[j],objectives[j]])
    else:
        outside_constraints.append([inputs[j],objectives[j]])

savedata = [within_constraints,outside_constraints]

with open('mcdata.dat','wb') as f:
    pickle.dump(savedata,f)

if plot:
    objective1_withinconstraints = [within_constraints[j][1][0] for j in range(len(within_constraints))]
    objective2_withinconstraints = [within_constraints[j][1][1] for j in range(len(within_constraints))]
    objective3_withinconstraints = [within_constraints[j][1][2] for j in range(len(within_constraints))]

    objective1_outsideconstraints = [outside_constraints[j][1][0] for j in range(len(outside_constraints))]
    objective2_outsideconstraints = [outside_constraints[j][1][1] for j in range(len(outside_constraints))]
    objective3_outsideconstraints = [outside_constraints[j][1][2] for j in range(len(outside_constraints))]

    # 3 by 1 figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    #fig.suptitle('Parameter ' + str(i) + ' against objectives')
    axs[0].scatter(objective1_withinconstraints, objective2_withinconstraints,color='blue', label='Within constraints')
    axs[0].scatter(objective1_outsideconstraints, objective2_outsideconstraints,color='red', label='Outside constraints')
    #axs[0].set_title('Volume')
    axs[0].set_xlabel('Volume')
    axs[0].set_ylabel('L/D')
    axs[0].legend()
    axs[1].scatter(objective2_withinconstraints, objective3_withinconstraints,color='blue', label='Within constraints')
    axs[1].scatter(objective2_outsideconstraints, objective3_outsideconstraints,color='red', label='Outside constraints')
    #axs[1].set_title('L/D')
    axs[1].set_xlabel('L/D')
    axs[1].set_ylabel('Max G-load')
    axs[1].legend()
    axs[2].scatter(objective1_withinconstraints, objective3_withinconstraints,color='blue', label='Within constraints')
    axs[2].scatter(objective1_outsideconstraints, objective3_outsideconstraints,color='red', label='Outside constraints')
    #axs[2].set_title('Max G-load')
    axs[2].set_xlabel('Volume')
    axs[2].set_ylabel('Max G-load')
    axs[2].legend()
    plt.show()    





        