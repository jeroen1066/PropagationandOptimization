"""
Copyright (c) 2010-2021, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

AE4866 Propagation and Optimization in Astrodynamics
Shape Optimization
First name: ***Jonah***
Last name: ***Pedra***
Student number: ***5003768***

This module computes the dynamics of a capsule re-entering the atmosphere of the Earth, using a variety of integrator
and propagator settings.  For each run, the differences w.r.t. a benchmark propagation are computed, providing a proxy
for setting quality. The benchmark settings are currently defined semi-randomly, and are to be analyzed/modified.

The trajectory of the capsule is heavily dependent on the shape and orientation of the vehicle. Here, the shape is
determined here by the five parameters, which are used to compute the aerodynamic accelerations on the vehicle using a
modified Newtonian flow (see Dirkx and Mooij, "Conceptual Shape Optimization of Entry Vehicles" 2018). The bank angle
and sideslip angles are set to zero. The vehicle shape and angle of attack are defined by values in the vector shape_parameters.

The vehicle starts 120 km above the surface of the planet, with a speed of 7.83 km/s in an Earth-fixed frame (see
getInitialState function).

The propagation is terminated as soon as one of the following conditions is met (see 
get_propagation_termination_settings() function):
- Altitude < 25 km
- Propagation time > 24 hr

This propagation assumes only point mass gravity by the Earth and aerodynamic accelerations.

The entries of the vector 'shape_parameters' contains the following:
- Entry 0:  Nose radius
- Entry 1:  Middle radius
- Entry 2:  Rear length
- Entry 3:  Rear angle
- Entry 4:  Side radius
- Entry 5:  Constant Angle of Attack

Details on the outputs written by this file can be found:
- benchmark data: comments for 'generateBenchmarks' function
- results for integrator/propagator variations: comments under "RUN SIMULATION FOR VARIOUS SETTINGS"
- files defining the points and surface normals of the mesg used for the aerodynamic analysis (save_vehicle_mesh_to_file)

Frequent warnings and/or errors that might pop up:
* One frequent warning could be the following (mock values):
    "Warning in interpolator, requesting data point outside of boundaries, requested data at 7008 but limit values are
    0 and 7002, applying extrapolation instead."
It can happen that the benchmark ends earlier than the regular simulation, due to the smaller step size. Therefore,
the code will be forced to extrapolate the benchmark states (or dependent variables) to compare them to the
simulation output, producing a warning. This warning can be deactivated by forcing the interpolator to use the boundary
value instead of extrapolating (extrapolation is the default behavior). This can be done by setting:

    interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation = interpolators.extrapolate_at_boundary)

* One frequent error could be the following:
    "Error, propagation terminated at t=4454.723896, returning propagation data up to current time."
    This means that an error occurred with the given settings. Typically, this implies that the integrator/propagator
    combination is not feasible. It is part of the assignment to figure out why this happens.

* One frequent error can be one of:
    "Error in RKF integrator, step size is NaN"
    "Error in ABM integrator, step size is NaN"
    "Error in BS integrator, step size is NaN"

This means that a variable time-step integrator wanting to take a NaN time step. In such cases, the selected
integrator settings are unsuitable for the problem you are considering.

NOTE: When any of the above errors occur, the propagation results up to the point of the crash can still be extracted
as normal. It can be checked whether any issues have occured by using the function

dynamics_simulator.integration_completed_successfully

which returns a boolean (false if any issues have occured)

* A frequent issue can be that a simulation with certain settings runs for too long (for instance if the time steo
becomes excessively small). To prevent this, you can add an additional termination setting (on top of the existing ones!)

    cpu_tim_termination_settings = propagation_setup.propagator.cpu_time_termination(
        maximum_cpu_time )

where maximum_cpu_time is a varaiable (float) denoting the maximum time in seconds that your simulation is allowed to
run. If the simulation runs longer, it will terminate, and return the propagation results up to that point.

* Finally, if the following error occurs, you can NOT extract the results up to the point of the crash. Instead,
the program will immediately terminate

    SPICE(DAFNEGADDR) --

    Negative value for BEGIN address: -214731446

This means that a state is extracted from Spice at a time equal to NaN. Typically, this is indicative of a
variable time-step integrator wanting to take a NaN time step, and the issue not being caught by Tudat.
In such cases, the selected integrator settings are unsuitable for the problem you are considering.
"""

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################
#
# import sys
# sys.path.insert(0, "/home/dominic/Tudat/tudat-bundle/tudat-bundle/cmake-build-default/tudatpy")

# General imports
import numpy as np
import os
import time

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

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()
# NOTE TO STUDENTS: INPUT YOUR PARAMETER SET HERE, FROM THE INPUT FILES
# ON BRIGHTSPACE, FOR YOUR SPECIFIC STUDENT NUMBER
shape_parameters_1 = [9.0071362922,	2.9990405154,	2.4752705714,	-0.512795019,	0.3084274431,	0.5128362015]
shape_parameters_2 = [3.8686343422, 2.6697460404, 0.6877576649, -0.7652400717, 0.3522259173, 0.2548030601]
shape_parameters_3 = [7.6983117481, 2.0923385955, 1.7186406196, -0.255984141, 0.1158838553, 0.3203083369]

shape_parameters_to_test = [shape_parameters_1, shape_parameters_2, shape_parameters_3]

# Choose whether output of the propagation is written to files
write_results_to_file = True
# Get path of current directory
current_dir = os.path.dirname(__file__)

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

# Initialize dictionary to save simulation output
simulation_results = dict()


generate_data = True


if generate_data:
    # Loop over different model settings
    for test_shape in range(len(shape_parameters_to_test)):
        # Set shape parameters
        shape_parameters = shape_parameters_to_test[test_shape]
        print('')
        print('Running Shape: ', test_shape)

        # Define settings for celestial bodies
        bodies_to_create = ['Earth','Moon','Sun']
        # Define coordinate system
        global_frame_origin = 'Earth'
        global_frame_orientation = 'J2000'

        # Create body settings
        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create,
            global_frame_origin,
            global_frame_orientation) 
        
        body_settings.get('Earth').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
            'Earth', simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Sun'),
            frame_orientation='J2000')
        body_settings.get('Moon').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
            'Moon', simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Earth'),
            frame_orientation='J2000')
        body_settings.get('Sun').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
            'Sun', simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Sun'),
            frame_orientation='J2000')

        body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.nrlmsise00()

        body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.simple_from_spice(
                global_frame_orientation, 'IAU_Earth', 'IAU_Earth', simulation_start_epoch)

        equitorial_radius = 6378137.0
        flattening = 1 / 298.25
        body_settings.get('Earth').shape_settings = environment_setup.shape.oblate_spherical(
            equitorial_radius, flattening)
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


        # Create propagator settings for benchmark (Cowell)
        propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                        bodies,
                                                        simulation_start_epoch,
                                                        termination_settings,
                                                        dependent_variables_to_save,
                                                        current_propagator=propagation_setup.propagator.cowell,
                                                        )
        
        # Create integrator settings
        settings_index = 2
        initial_step_size = 1.0

        integrator = propagation_setup.integrator
        current_tolerance = 10.0 ** (-8.0)
        current_coefficient_set = propagation_setup.integrator.CoefficientSets.rkdp_87

        propagator_settings.integrator_settings = integrator.runge_kutta_variable_step_size(
                simulation_start_epoch,
                initial_step_size,
                current_coefficient_set,
                np.finfo(float).eps,
                np.inf,
                current_tolerance,
                current_tolerance)
        
        #Simulation Start time
        sim_start_time = time.time()

        # Create Shape Optimization Problem object
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies, propagator_settings )
        
        #Simulation End time
        sim_end_time = time.time()

        #Simulation Duration time
        sim_duration = sim_end_time - sim_start_time

        ### OUTPUT OF THE SIMULATION ###
        # Retrieve propagated state and dependent variables
        state_history = dynamics_simulator.state_history
        dependent_variable_history = dynamics_simulator.dependent_variable_history

        state_history_array = np.array(list(state_history.values()))
        dependent_variable_history_array = np.array(list(dependent_variable_history.values()))
        times = np.array(list(state_history.keys()))
        input_value = shape_parameters[:]
        input_value.append(capsule_density)

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

        n = 0.2

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
        
        objectives = [mass_capsule,max_ld,max_g_load]
        print('Mass:',mass_capsule)
        print('Max L/D:',max_ld)
        print('Max g-load:',max_g_load)
        print('Max heat flux:',max_heat_flux)
        print('Total heat load:',total_heat_load)
        print('Has skipped:',has_skipped)
        print('Stability:',stability)
        print('Succesfull completion:',succesfull_completion)


        # Save results to a dictionary
        simulation_results[test_shape] = [state_history, dependent_variable_history]

        # Get the number of function evaluations (for comparison of different integrators)
        function_evaluation_dict = dynamics_simulator.cumulative_number_of_function_evaluations
        number_of_function_evaluations = list(function_evaluation_dict.values())[-1]
        # Add it to a dictionary
        dict_to_write = {'Number of function evaluations (ignore the line above)': number_of_function_evaluations}
        # Check if the propagation was run successfully
        propagation_outcome = dynamics_simulator.integration_completed_successfully
        dict_to_write['Propagation run successfully'] = propagation_outcome
        # Note if results were written to files
        dict_to_write['Results written to file'] = write_results_to_file
        # Note the simulation duration
        dict_to_write['Simulation duration'] = sim_duration
        # Note if dependent variables were present
        dict_to_write['Dependent variables present'] = are_dependent_variables_to_save

        subdirectory = '/Test_shape_' + str(test_shape) + '/'

        # Decide if output writing is required
        if write_results_to_file:
            output_path = current_dir + subdirectory
        else:
            output_path = None

        # If desired, write output to a file
        if write_results_to_file:
            save2txt(state_history, 'state_history.dat', output_path)
            print("Saved state history for shape", test_shape, "to file")
            save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)
            print("Saved dependent variable history for shape", test_shape, "to file")
            save2txt(dict_to_write, 'ancillary_simulation_info.txt',   output_path)
            print("Saved ancillary simulation info for shape", test_shape, "to file")

