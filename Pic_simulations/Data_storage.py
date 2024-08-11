import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pygame


class Basic_data_storage:
    def __init__(self, Particle_stored, Num_iteration, Inject_frequ1, Inject_frequ2):
        #Particle_variables
        N_tot_variables = Particle_stored.N_particles+ Num_iteration*(Inject_frequ1+Inject_frequ2)
        self.X = np.zeros([N_tot_variables, 2, Num_iteration])
        self.V = np.zeros([N_tot_variables, 2, Num_iteration])
        self.A = np.zeros([N_tot_variables, 2, Num_iteration])
        self.charge_mass = np.zeros([N_tot_variables, 2, Num_iteration])
        #Mesh_variables
        self.density = np.zeros([2, Particle_stored.mesh_size, Particle_stored.mesh_size, Num_iteration])
        self.Pot_field = np.zeros([2, Particle_stored.mesh_size, Particle_stored.mesh_size, Num_iteration])
        self.E_field = np.zeros([2, Particle_stored.mesh_size, Particle_stored.mesh_size, Num_iteration])
    
    def store_data(self, Particle_stored, i):
        self.X[:Particle_stored.N_particles,:, i] = np.stack((Particle_stored.X, Particle_stored.Y), axis=1)
        self.V[:Particle_stored.N_particles,:, i] = np.stack((Particle_stored.VX, Particle_stored.VY), axis=1)
        self.A[:Particle_stored.N_particles,:, i] = np.stack((Particle_stored.AX, Particle_stored.AY), axis=1)
