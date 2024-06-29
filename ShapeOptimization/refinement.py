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

parameternames = ['Nose radius', 'Middle radius', 'Rear length', 'Rear angle', 'Side radius','Angle of Attack','Density']


#load parameters from previous optimization
#chosen id's are 18, 12, 0, 4

xval_choice_names = ['mass','l- d','g-load','average_fitness','single_optimizer' ]

xvals_forprocessing = []
with open('ShapeOptimization/results/tuning_v2_maco/maco_tuned_seed_169.dat', 'rb') as f:
    data = pickle.load(f)
    data = data[0]
    xvals = data[0]
    yvals = data[1]
    xvals_forprocessing.append(xvals[0])
    xvals_forprocessing.append(xvals[4])
    xvals_forprocessing.append(xvals[12])
    xvals_forprocessing.append(xvals[18])


###########################################################################
# PARAMETERS ##############################################################
###########################################################################

numsimulations = 100
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



single_shape_parameters = [4.96696412e+00,  
                             3.75143635e+00,  
                             2.98836607e+00, 
                             -1.55543028e-03,
                             4.33821906e+00,  
                             1.62819329e-01,  
                             2.28097245e+02]

xvals_forprocessing.append(single_shape_parameters)

np.random.seed(42)

new_range_per_parameter = [[-0.1,0.1],
                           [-0.1,0.1],
                           [-0.1,0.1],
                           [-0.1,0.1],
                           [-0.1,0.1],
                           [-0.01,0.01],
                           [-10,10]
                           ]


inputs = np.empty((numparameters,numsimulations),dtype=object)
#time_histories = np.empty((numparameters,numsimulations),dtype=object)
objectives = np.empty((numparameters,numsimulations),dtype=object)
constraints = np.empty((numparameters,numsimulations),dtype=object)
xvalindex = 0
for x in xvals_forprocessing:
    if xvalindex == 0:
        xvalindex += 1
        continue
    selected_shape_parameters = x
    for i in range (numparameters):

        shape_variation = np.random.uniform(new_range_per_parameter[i][0],new_range_per_parameter[i][1],numsimulations)

        for j in range (numsimulations):
            if i != 6:
                shape_parameters = selected_shape_parameters[:6]
                shape_parameters[i] = selected_shape_parameters[i] + shape_variation[j]
                capsule_density = selected_shape_parameters[6]

            else:
                shape_parameters = selected_shape_parameters[:6]
                capsule_density = selected_shape_parameters[i] + shape_variation[j]


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
            input_value = shape_parameters[:]
            input_value = list(input_value)
            input_value.append(capsule_density)
            inputs[i,j] = input_value[i]

            #dependent variable will eventaully have 5 members: 
            # 3d aerodynamic acceleration,
            # 1d g-force,
            # density
            # skipping is defined by having a height increase during propagation
            max_ld = -np.inf
            max_g_load = -np.inf
            max_heat_flux = -np.inf
            total_heat_load = 0
            has_skipped = False
            stability = -1000
            mass_capsule = bodies.get_body('Capsule').mass
            volume_capsule = mass_capsule/capsule_density
            last_t = times[0]
            succesfull_completion = dynamics_simulator.integration_completed_successfully

            for t in times:
                timestep = t-last_t
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
            
            objectives[i,j] = [mass_capsule,max_ld,max_g_load]
            constraints[i,j] = [max_heat_flux,total_heat_load,stability,has_skipped,succesfull_completion]
            last_t = t

    within_constraints = []
    outside_constraints = []
    for i in range(numparameters):
        within_constraints_parameter = []
        outside_constraints_parameter = []
        for j in range(numsimulations):
            within_heat_flux = constraints[i,j][0] < heat_flux_constraint
            within_heat_load = constraints[i,j][1] < heat_load_constraint
            within_stability = constraints[i,j][2] < stability_constraint
            within_skip = constraints[i,j][3] == False
            succesfull_completion = constraints[i,j][4]

            if within_heat_flux and within_heat_load and within_stability and within_skip and succesfull_completion:
                input_value = inputs[i,j]
                objective_value = objectives[i,j]
                within_constraints_parameter.append([input_value,objective_value])
            else:
                input_value = inputs[i,j]
                objective_value = objectives[i,j]
                outside_constraints_parameter.append([input_value,objective_value])
        within_constraints.append(within_constraints_parameter)
        outside_constraints.append(outside_constraints_parameter)


    #plotting input against objectives
    for i in range(numparameters):
        plotname = xval_choice_names[xvalindex]
        input_withinconstraints = [within_constraints[i][j][0] for j in range(len(within_constraints[i]))]
        input_outsideconstraints = [outside_constraints[i][j][0] for j in range(len(outside_constraints[i]))]

        objective1_withinconstraints = [within_constraints[i][j][1][0] for j in range(len(within_constraints[i]))]
        objective2_withinconstraints = [within_constraints[i][j][1][1] for j in range(len(within_constraints[i]))]
        objective3_withinconstraints = [within_constraints[i][j][1][2] for j in range(len(within_constraints[i]))]

        objective1_outsideconstraints = [outside_constraints[i][j][1][0] for j in range(len(outside_constraints[i]))]
        objective2_outsideconstraints = [outside_constraints[i][j][1][1] for j in range(len(outside_constraints[i]))]
        objective3_outsideconstraints = [outside_constraints[i][j][1][2] for j in range(len(outside_constraints[i]))]

        # 3 by 1 figure with 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        fig.suptitle('Parameter ' + parameternames[i] + ' against objectives')
        axs[0].scatter(input_withinconstraints, objective1_withinconstraints,color='blue', label='Within constraints')
        axs[0].scatter(input_outsideconstraints, objective1_outsideconstraints,color='red', label='Outside constraints')
        axs[0].set_title('Volume')
        axs[0].set_xlabel('Parameter ' + parameternames[i])
        axs[0].set_ylabel('Volume')
        axs[0].legend()
        axs[0].grid()
        axs[1].scatter(input_withinconstraints, objective2_withinconstraints,color='blue', label='Within constraints')
        axs[1].scatter(input_outsideconstraints, objective2_outsideconstraints,color='red', label='Outside constraints')
        axs[1].set_title('L/D')
        axs[1].set_xlabel('Parameter ' + parameternames[i])
        axs[1].set_ylabel('L/D')
        axs[1].legend()
        axs[1].grid()
        axs[2].scatter(input_withinconstraints, objective3_withinconstraints,color='blue', label='Within constraints')
        axs[2].scatter(input_outsideconstraints, objective3_outsideconstraints,color='red', label='Outside constraints')
        axs[2].set_title('Max G-load')
        axs[2].set_xlabel('Parameter ' + parameternames[i])
        axs[2].set_ylabel('Max G-load')
        axs[2].legend()
        axs[2].grid()
        fig.savefig('results/plots/'+plotname+'_Param_' + parameternames[i] + '_obj' + '.png')
        plt.clf()

        #plot constraints against inputs:
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        fig.suptitle('Parameter ' + parameternames[i] + ' against constraints')
        heat_flux = [constraints[i][j][0] for j in range(numsimulations)]
        heat_load = [constraints[i][j][1] for j in range(numsimulations)]
        stability = [constraints[i][j][2] for j in range(numsimulations)]
        skipped = [constraints[i][j][3] for j in range(numsimulations)]
        succesfull = [constraints[i][j][4] for j in range(numsimulations)]
        axs[0].scatter(inputs[i], heat_flux)
        axs[0].set_title('Heat flux')
        axs[0].set_xlabel('Parameter ' + parameternames[i])
        axs[0].set_ylabel('Heat flux')
        axs[0].grid()
        axs[1].scatter(inputs[i], heat_load)
        axs[1].set_title('Heat load')
        axs[1].set_xlabel('Parameter ' +parameternames[i])
        axs[1].set_ylabel('Heat load')
        axs[1].grid()
        axs[2].scatter(inputs[i], skipped)
        axs[2].set_title('Skipped')
        axs[2].set_xlabel('Parameter ' + parameternames[i])
        axs[2].set_ylabel('Skipped')
        axs[2].grid()
        #axs[3].scatter(inputs[i], succesfull)
        #axs[3].set_title('Succesfull')
        #axs[3].set_xlabel('Parameter ' + parameternames[i])
        #axs[3].set_ylabel('Succesfull')
        #axs[3].grid()
        fig.savefig('results/plots/'+plotname+'_Param_' + parameternames[i] + '_con'+ '.png')
        plt.clf()






    xvalindex += 1