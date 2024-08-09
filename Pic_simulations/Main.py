import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pygame

from Units import electron, Nitrogen_pos, Nitrogen_neg
from Mesh_grid_initiation import PDE_matrix
from Particle_initation import Particle_distribution
import Functions

#Simulation position parameters
Mesh_grid_size = 50
Mesh_distace = 1.38*10**(-6)

#Simulation time parameters
Time_step = 1*10**(-9)
N_iterations = 1000

Particle_matrix = PDE_matrix(Mesh_grid_size, Mesh_distace,[0 ,0, 0])
Potential_field_matrix = PDE_matrix(Mesh_grid_size, Mesh_distace, [1, -2, 1])
Electric_field_matrix = PDE_matrix(Mesh_grid_size, Mesh_distace, [-1, 0 ,1])

#initial conditions for dual particle simulation
#molecule 1
Type_1 = Nitrogen_pos
Num_particles_1  =  1000
Particle_sigma_1 = 1
Particle_pos_0_1 = [0.3*Mesh_grid_size*Mesh_distace, 0.5*Mesh_grid_size*Mesh_distace]
Particle_vel_0_1 = [100, 0]
Particle_acc_0_1 = [0, 0]

#molecule 2
Type_2 = Nitrogen_pos
Num_particles_2  =  1000
Particle_sigma_2 = 1
Particle_pos_0_2 = [0.6*Mesh_distace*Mesh_grid_size, 0.5*Mesh_distace*Mesh_grid_size]
Particle_vel_0_2 = [-100, 0]
Particle_acc_0_2 = [0, 0]

#Physical parameters
Multiplication_factor = 3.125*10**15

#Initial fields
E_initial = [1*10**(-7), 0]

#Loop variables 
Sim_position = []

pygame.init()

screen = pygame.display.set_mode((1280,720))

clock = pygame.time.Clock()

while True:
    # Process player inputs.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExitx

    molecule_1_distribution= Particle_distribution(Type_1, Num_particles_1, Particle_pos_0_1[0], Particle_pos_0_1[1], Particle_vel_0_1[0], Particle_vel_0_1[1], Particle_acc_0_1[0], Particle_acc_0_1[1], Particle_sigma_1, Particle_matrix, Multiplication_factor)
    molecule_2_distribution= Particle_distribution(Type_2, Num_particles_2, Particle_pos_0_2[0], Particle_pos_0_2[1], Particle_vel_0_2[0], Particle_vel_0_2[1], Particle_acc_0_2[0], Particle_acc_0_2[1], Particle_sigma_2, Particle_matrix, Multiplication_factor)
    molecule_combined = Particle_distribution(Type_1, Num_particles_1+Num_particles_2, Particle_pos_0_2[0], Particle_pos_0_2[1], Particle_vel_0_2[0], Particle_vel_0_2[1], Particle_acc_0_2[0], Particle_acc_0_2[1], Particle_sigma_2, Particle_matrix, Multiplication_factor)
    molecule_combined.Multiple_molecules(molecule_1_distribution, molecule_2_distribution)
    molecule_combined.calculate_density()
    molecule_combined.plot_density(True, Num_particles_1, Num_particles_2)

    molecule_combined, Sim_position_X, Sim_position_Y, Sim_temperature = Functions.simulate(molecule_combined, Potential_field_matrix, Electric_field_matrix, Time_step, N_iterations, Num_particles_1+Num_particles_2, E_initial, screen, clock, Mesh_grid_size, Mesh_distace, Num_particles_1, Num_particles_2)
    Functions.animate_particles(molecule_combined, Sim_position_X, Sim_position_Y,Sim_temperature, N_iterations, Mesh_grid_size, Mesh_distace, Num_particles_1, Num_particles_2)

# Example usage
# Sim_position_X, Sim_position_Y = simulate(Nitrogen_distribution, Potential_field_matrix, Electric_field_matrix, Time_step, N_iterations, Num_particles)
# animate_particles(Nitrogen_distribution, Sim_position_X, Sim_position_Y, N_iterations, mesh_size, mesh_separation)
