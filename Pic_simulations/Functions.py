import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import pygame 

#Declare esential variables
Epsilon_0 = 8.8541878128 * 10**(-12)  # F/m

# Main simulation loop
def simulate(Molecule1, Molecule2, Particle_distribution, Time_step, N_iterations, E_field, screen, clock, mesh_size, mesh_separation, inject_1, inject_2, sim_data_storage):

    # Initialize Laplacian matrices
    laplacian_matrix_x = np.zeros([Particle_distribution.mesh_size, Particle_distribution.mesh_size])
    laplacian_matrix_y = np.zeros([Particle_distribution.mesh_size, Particle_distribution.mesh_size])

    for i in range(N_iterations):
        print(f"Iteration {i+1}/{N_iterations}")
        
        # Inject new particles into the system
        Particle_distribution.inject(inject_1, inject_2, Molecule1, Molecule2)

        # Calculate the density of particles in the mesh
        Particle_distribution.calculate_density()

        # Calculate the acceleration and electric field matrix
        Particle_distribution, laplacian_matrix_x, laplacian_matrix_y, E_field_X, E_field_Y = Acceleration_calculation(Particle_distribution, laplacian_matrix_x, laplacian_matrix_y, E_field)

        # Update the positions of the particles
        Particle_distribution = Forward_step_function(Particle_distribution, Time_step)

        # Record the current simulation state
        sim_data_storage.store_data(Particle_distribution, i,Particle_distribution.density, laplacian_matrix_x, laplacian_matrix_y, E_field_X, E_field_Y)

        # Update Pygame screen for visualization
        update_pygame_screen(screen, Particle_distribution, mesh_size, mesh_separation)

        # Control the simulation speed
        clock.tick(30)  # Adjust the value to control the speed of the simulation
    #This plots the final results of the simulations
    sim_data_storage.plot_data()

    #This stores the final results of the simulation for future use
    sim_data_storage.data_storage("Simulation_01")

    return Particle_distribution, sim_data_storage


# Gauss-Seidel method to solve Laplace equation
def Gauss_Seidel(phi, Delta_x, b, mesh_size):
    for j in range(1, mesh_size - 1):
        for k in range(1, mesh_size - 1):
            phi[j, k] = 0.25 * (phi[j-1, k] + phi[j+1, k] + phi[j, k-1] + phi[j, k+1] - Delta_x**2 * b[j, k])
    return phi

# Calculate the acceleration of particles based on the electric field
def Acceleration_calculation(Particle_distribution, laplacian_matrix_x, laplacian_matrix_y, E_field):
    mesh_size = Particle_distribution.mesh_size
    mesh_separation = Particle_distribution.mesh_separation
    
    # Solve for Laplacian matrices using Gauss-Seidel method
    laplacian_matrix_X = Gauss_Seidel(laplacian_matrix_x, mesh_separation, Particle_distribution.density, mesh_size)
    laplacian_matrix_Y = Gauss_Seidel(laplacian_matrix_y, mesh_separation, np.transpose(Particle_distribution.density), mesh_size)
    
    # Debug prints for checking Laplacian matrices
    print(np.max(laplacian_matrix_X))
    print(np.max(laplacian_matrix_Y))
    
    # Initialize electric field matrices
    E_field_matrix_X = np.zeros([mesh_size, mesh_size])
    E_field_matrix_Y = np.zeros([mesh_size, mesh_size])
    
    # Calculate the electric field based on the Laplacian matrices
    for j in range(1, mesh_size - 1):
        for k in range(1, mesh_size - 1):
            E_field_matrix_X[j, k] = -(laplacian_matrix_X[k, j+1] - laplacian_matrix_X[k, j-1]) / (2 * mesh_separation) + E_field[0]
            E_field_matrix_Y[j, k] = -(laplacian_matrix_Y[j, k+1] - laplacian_matrix_Y[j, k-1]) / (2 * mesh_separation) + E_field[1]
   
    # Initialize arrays for particle-specific electric fields and accelerations
    E_field_particle_X = np.zeros(Particle_distribution.N_particles)
    E_field_particle_Y = np.zeros(Particle_distribution.N_particles)
    Particle_distribution.AX = np.zeros(Particle_distribution.N_particles)
    Particle_distribution.AY = np.zeros(Particle_distribution.N_particles)

    # Assign electric field values to particles and calculate accelerations
    for j in range(Particle_distribution.N_particles):
        closest_x_index = int(Particle_distribution.X[j] // mesh_separation)
        closest_y_index = int(Particle_distribution.Y[j] // mesh_separation)

        closest_x_index = np.clip(closest_x_index, 0, mesh_size - 1)
        closest_y_index = np.clip(closest_y_index, 0, mesh_size - 1)
        
        E_field_particle_X[j] = E_field_matrix_X[closest_x_index, closest_y_index]
        E_field_particle_Y[j] = E_field_matrix_Y[closest_x_index, closest_y_index]

        # Correct sign in the acceleration calculation
        Particle_distribution.AX[j] = -E_field_particle_X[j] * (Particle_distribution.charge[j]) / (Particle_distribution.mass[j] * Epsilon_0)
        Particle_distribution.AY[j] = -E_field_particle_Y[j] * (Particle_distribution.charge[j]) / (Particle_distribution.mass[j] * Epsilon_0)

    # Debug plot for electric field matrix (optional)
    # plt.plot(E_field_matrix_X)
    # plt.show()
    
    return Particle_distribution, laplacian_matrix_x, laplacian_matrix_y, E_field_matrix_X, E_field_matrix_Y

# Update the positions and velocities of particles in a forward step
def Forward_step_function(Particle_distribution, time_step):
    Particle_distribution.VX += time_step * Particle_distribution.AX 
    Particle_distribution.VY += time_step * Particle_distribution.AY 
    Mesh_limit = Particle_distribution.mesh_size * Particle_distribution.mesh_separation

    VX = np.average(np.abs(Particle_distribution.VX))
    VY = np.average(np.abs(Particle_distribution.VY))

    # Check for high velocities and reset if needed
    for j in range(Particle_distribution.N_particles):
        if np.abs(Particle_distribution.VX[j]) > 10 * VX or np.abs(Particle_distribution.VY[j]) > 10 * VY:
            Particle_distribution.VX[j] = 0
            Particle_distribution.VY[j] = 0
            Particle_distribution.X[j] = 0
            Particle_distribution.Y[j] = 0

    # Reflect particles at the boundaries
    for j in range(Particle_distribution.N_particles):
        if Particle_distribution.X[j] + Particle_distribution.mesh_separation * 3 > Mesh_limit:
            Particle_distribution.X[j] = Mesh_limit - Particle_distribution.mesh_separation * 3
            Particle_distribution.VX[j] = -Particle_distribution.VX[j]
        elif Particle_distribution.X[j] - Particle_distribution.mesh_separation * 3 < 0:
            Particle_distribution.X[j] = Particle_distribution.mesh_separation * 3
            Particle_distribution.VX[j] = -Particle_distribution.VX[j]
        if Particle_distribution.Y[j] + Particle_distribution.mesh_separation * 3 > Mesh_limit:
            Particle_distribution.Y[j] = Mesh_limit - Particle_distribution.mesh_separation * 3
            Particle_distribution.VY[j] = -Particle_distribution.VY[j]
        elif Particle_distribution.Y[j] - Particle_distribution.mesh_separation * 3 < 0:
            Particle_distribution.Y[j] = Particle_distribution.mesh_separation * 3
            Particle_distribution.VY[j] = -Particle_distribution.VY[j]

    # Update particle positions based on velocities
    Particle_distribution.Y += time_step * Particle_distribution.VY
    Particle_distribution.X += time_step * Particle_distribution.VX
    
    return Particle_distribution



# Function to update the Pygame screen with particle positions
def update_pygame_screen(screen, Particle_distribution, mesh_size, mesh_separation):
    screen.fill((0, 0, 0))  # Clear screen

    # Scaling factor for visualization
    scale = 900 / (mesh_size * mesh_separation)

    # Draw particles on the screen
    for idx in range(len(Particle_distribution.X)):
        x = int(Particle_distribution.X[idx] * scale)
        y = int(Particle_distribution.Y[idx] * scale)
        if 0 <= x < screen.get_width() and 0 <= y < screen.get_height():
            pygame.draw.circle(screen, (Particle_distribution.color_R[idx], Particle_distribution.color_G[idx], Particle_distribution.color_B[idx]), (x, y), 2)

    # Update the display
    pygame.display.flip()
