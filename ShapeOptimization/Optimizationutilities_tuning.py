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

import pygmo as pg

spice_interface.load_standard_kernels()






###########################################################################
# DEFINE PROBLEM ##########################################################
###########################################################################

class CapsuleEntryProblem:
    def __init__(
            self,
            simulation_start_epoch:float,
            initial_state:np.ndarray,
            integrator_settings:numerical_simulation.propagation_setup.integrator.IntegratorSettings,
            termination_settings :numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings,
            bodies:environment.SystemOfBodies,
            bounds:list[list[float]]

        )->None:
        self.simulation_start_epoch = simulation_start_epoch
        self.initial_state = initial_state
        self.integrator_settings = integrator_settings
        self.termination_settings = termination_settings
        self.bodies = bodies
        self.bounds = bounds
        self.constraints = [5e6,2e8]
        dependent_variables_to_save = Util.get_dependent_variable_save_settings()
        self.dependent_variables_to_save = lambda: dependent_variables_to_save

        
    def get_bounds(self)->list[list[float]]:
        return self.bounds
    
    def get_number_of_parameters(self)->int:
        return len(self.bounds)
    
    def get_nobj(self)->int:
        return 3
    
    def fitness(self, parameters):
        shape_parameters = parameters[:6]
        density = parameters[6]
        # Create vehicle

        bodies = self.bodies()
        termination_settings = self.termination_settings()
        integrator_settings = self.integrator_settings()

        Util.add_capsule_to_body_system(bodies,
                            shape_parameters,
                            density)
        
        dependent_variables_to_save = self.dependent_variables_to_save()
    
        propagator_settings = Util.get_propagator_settings(shape_parameters,
                                                                bodies,
                                                                self.simulation_start_epoch,
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

        n = 0.2
        max_ld = 0
        max_g_load = 0
        max_heat_flux = 0
        total_heat_load = 0
        has_skipped = False
        stability = -1000
        mass_capsule = bodies.get_body('Capsule').mass
        volume_capsule = mass_capsule/density
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

        fitness = np.array([1e5/mass_capsule,1/(max_ld+0.01),max_g_load])
        if not succesfull_completion or has_skipped:
            fitness = np.array([1e10,1e10,1e10])
        if max_heat_flux > self.constraints[0]:
            fitness *= max_heat_flux/self.constraints[0]*100
        if total_heat_load > self.constraints[1]:
            fitness *= total_heat_load/self.constraints[1]*100

        
        return fitness
    
class optimization:
    def __init__(self,range_per_parameter:list[list[float]],optimizer_name = str)->None:
        self.simulation_start_epoch = 0.0
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
            'Earth', self.simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Sun'),
            frame_orientation='J2000')
        body_settings.get('Moon').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
            'Moon', self.simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Earth'),
            frame_orientation='J2000')
        body_settings.get('Sun').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
            'Sun', self.simulation_start_epoch, spice_interface.get_body_gravitational_parameter('Sun'),
            frame_orientation='J2000')

        body_settings.get('Earth').atmosphere_settings = nrlmsise00
        # Create bodies
        self.bodies = environment_setup.create_system_of_bodies(body_settings)

        # Retrieve termination settings
        maximum_duration = constants.JULIAN_DAY  # s
        termination_altitude = 25.0E3  # m
        self.termination_settings = Util.get_termination_settings(self.simulation_start_epoch,
                                                            maximum_duration,
                                                            termination_altitude)

        dependent_variables_to_save = Util.get_dependent_variable_save_settings()
        # Check whether there is any
        are_dependent_variables_to_save = False if not dependent_variables_to_save else True

        self.integrator_settings = Util.get_integrator_settings(3,
                                                            3,
                                                            1e-8,
                                                            self.simulation_start_epoch)

        self.initial_state = Util.get_initial_state(self.simulation_start_epoch,self.bodies)
        self.range_per_parameter = range_per_parameter
        
        if optimizer_name == 'moead_gen':
            self.optimizer = pg.moead_gen
        else:
            raise ValueError('Optimizer not recognized, invalid input name')
        
    def optimize(self,numpops:int,numgens:int,seed:int = 42, numneighbours=20, test_CR=1, test_F=0.5, test_eta_m=20, test_realb=0.9)->None:
        self.results = []
        integrator = lambda: self.integrator_settings
        termination = lambda: self.termination_settings
        bodies = lambda: self.bodies

        problem_definition = CapsuleEntryProblem(self.simulation_start_epoch,
                                             self.initial_state,
                                             integrator,
                                             termination,
                                             bodies,
                                             self.range_per_parameter)
        problem = pg.problem(problem_definition)
        
        algo = pg.algorithm(self.optimizer(seed=seed, 
                                            gen=numgens,
                                            neighbours=numneighbours,
                                            CR = test_CR,
                                            F = test_F,
                                            eta_m = test_eta_m,
                                            realb = test_realb
                                           ))
        pop = pg.population(problem,numpops,seed=seed)
        self.results.append(pop)
        return None
        
    def plot(self)->None:
        #todo, some time at some point

        return None