import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pygame

from Units import electron, Nitrogen_pos, Nitrogen_neg
from Particle_initation import Particle_distribution
import Functions

# Class to store all simulation parameters
class SimulationParameters:
    def __init__(self):
        # Simulation position parameters
        self.Mesh_grid_size = 50
        self.Mesh_distance = 1.38 * 10**(-7)

        # Simulation time parameters
        self.Time_step = 1*10**(-9)
        self.N_iterations = 10000

        # Initial conditions for molecule 1
        self.Type_1 = Nitrogen_neg
        self.Num_particles_1 = 1
        self.Particle_sigma_1 = 0.5
        self.Particle_pos_0_1 = [0.2 * self.Mesh_grid_size * self.Mesh_distance, 0.5 * self.Mesh_grid_size * self.Mesh_distance]
        self.Particle_vel_0_1 = [200, -200]
        self.Particle_acc_0_1 = [0, 0]
        self.Particle_inject_0_1 = 2
        self.Particle_color_0_1 = (0, 0  ,255)

        # Initial conditions for molecule 2
        self.Type_2 = Nitrogen_pos
        self.Num_particles_2 = 1
        self.Particle_sigma_2 = 0.5
        self.Particle_pos_0_2 = [0.8 * self.Mesh_distance * self.Mesh_grid_size, 0.5 * self.Mesh_distance * self.Mesh_grid_size]
        self.Particle_vel_0_2 = [-200, 200]
        self.Particle_acc_0_2 = [0, 0]
        self.Particle_inject_0_2 = 2
        self.Particle_color_0_2 = (255, 0  ,0)
        # Physical parameters
        self.Multiplication_factor = 3.125 * 10**15

        # Initial electric field
        self.E_initial = [1*10**(-8) , 0]

        # Loop variables
        self.Sim_position = []

        

def main():
    # Initialize the simulation parameters
    sim_params = SimulationParameters()

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((900, 900))
    clock = pygame.time.Clock()

    while True:
        # Process player inputs.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # Initialize particle distributions for molecule 1 and molecule 2
        molecule_1_distribution = Particle_distribution(
            sim_params.Type_1, sim_params.Num_particles_1,
            sim_params.Particle_pos_0_1[0], sim_params.Particle_pos_0_1[1],
            sim_params.Particle_vel_0_1[0], sim_params.Particle_vel_0_1[1],
            sim_params.Particle_acc_0_1[0], sim_params.Particle_acc_0_1[1],
            sim_params.Particle_sigma_1, sim_params.Mesh_distance, sim_params.Mesh_grid_size, 
            sim_params.Multiplication_factor, sim_params.Particle_color_0_1
        )

        molecule_2_distribution = Particle_distribution(
            sim_params.Type_2, sim_params.Num_particles_2,
            sim_params.Particle_pos_0_2[0], sim_params.Particle_pos_0_2[1],
            sim_params.Particle_vel_0_2[0], sim_params.Particle_vel_0_2[1],
            sim_params.Particle_acc_0_2[0], sim_params.Particle_acc_0_2[1],
            sim_params.Particle_sigma_2, sim_params.Mesh_distance,sim_params.Mesh_grid_size,   
            sim_params.Multiplication_factor, sim_params.Particle_color_0_2
        )

        # Combine molecule distributions
        molecule_combined = Particle_distribution(
            sim_params.Type_1, sim_params.Num_particles_1 + sim_params.Num_particles_2,
            sim_params.Particle_pos_0_2[0], sim_params.Particle_pos_0_2[1],
            sim_params.Particle_vel_0_2[0], sim_params.Particle_vel_0_2[1],
            sim_params.Particle_acc_0_2[0], sim_params.Particle_acc_0_2[1],
            sim_params.Particle_sigma_2, sim_params.Mesh_distance, sim_params.Mesh_grid_size,  
            sim_params.Multiplication_factor,sim_params.Particle_color_0_2
        )
        molecule_combined.Multiple_molecules(molecule_1_distribution, molecule_2_distribution)
        molecule_combined.calculate_density()
        molecule_combined.plot_density(True, sim_params.Num_particles_1, sim_params.Num_particles_2)

        # Run simulation and animate particles
        molecule_combined, Sim_position_X, Sim_position_Y, Sim_temperature = Functions.simulate(molecule_1_distribution, molecule_2_distribution,
            molecule_combined, sim_params.Time_step, sim_params.N_iterations, sim_params.Num_particles_1 + sim_params.Num_particles_2,
            sim_params.E_initial, screen, clock, sim_params.Mesh_grid_size, sim_params.Mesh_distance,
            sim_params.Num_particles_1, sim_params.Num_particles_2, sim_params.Particle_inject_0_1, sim_params.Particle_inject_0_2
        )
        Functions.animate_particles(
            molecule_combined, Sim_position_X, Sim_position_Y, Sim_temperature,
            sim_params.N_iterations, sim_params.Mesh_grid_size, sim_params.Mesh_distance,
            sim_params.Num_particles_1, sim_params.Num_particles_2
        )

if __name__ == "__main__":
    main()