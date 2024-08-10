import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import pygame 

Epsilon_0 =8.8541878128* 10**(-12) #F/m

def Gauss_Seidel(phi, Delta_x, b, mesh_size):

    for j in range(1, mesh_size - 1):
            for k in range(1, mesh_size - 1):
                phi[j, k] = 0.25 * (phi[j-1, k] + phi[j+1, k] + phi[j, k-1] + phi[j, k+1] - Delta_x**2 * b[j, k])
    return phi

def Acceleration_calculation(Nitrogen_distribution, laplacian_matrix_x, laplacian_matrix_y, E_field):
    mesh_size = Nitrogen_distribution.mesh_size
    mesh_separation = Nitrogen_distribution.mesh_separation
    print(Nitrogen_distribution.density)
    
    laplacian_matrix_X = Gauss_Seidel(laplacian_matrix_x, mesh_separation, Nitrogen_distribution.density, mesh_size)
    laplacian_matrix_Y = Gauss_Seidel(laplacian_matrix_y, mesh_separation, np.transpose(Nitrogen_distribution.density), mesh_size)
    
    # Debug prints for checking Laplacian matrices
    print(np.max(laplacian_matrix_X))
    print(np.max(laplacian_matrix_Y))
    
    E_field_matrix_X = np.zeros([mesh_size, mesh_size])
    E_field_matrix_Y = np.zeros([mesh_size, mesh_size])
    
    for j in range(1, mesh_size - 1):
        for k in range(1, mesh_size - 1):
            E_field_matrix_X[j, k] = -(laplacian_matrix_X[k, j+1] - laplacian_matrix_X[k, j-1]) / (2 * mesh_separation)+ E_field[0]
            E_field_matrix_Y[j, k] = -(laplacian_matrix_Y[j, k+1] - laplacian_matrix_Y[j, k-1]) / (2 * mesh_separation)+ E_field[1]
   
    E_field_particle_X = np.zeros(Nitrogen_distribution.N_particles)
    E_field_particle_Y = np.zeros(Nitrogen_distribution.N_particles)
    Nitrogen_distribution.AX = np.zeros(Nitrogen_distribution.N_particles)
    Nitrogen_distribution.AY = np.zeros(Nitrogen_distribution.N_particles)

    for j in range(Nitrogen_distribution.N_particles):
        closest_x_index = int(Nitrogen_distribution.X[j] // mesh_separation)
        closest_y_index = int(Nitrogen_distribution.Y[j] // mesh_separation)

        closest_x_index = np.clip(closest_x_index, 0, mesh_size - 1)
        closest_y_index = np.clip(closest_y_index, 0, mesh_size - 1)
        
        E_field_particle_X[j] = E_field_matrix_X[closest_x_index, closest_y_index]
        E_field_particle_Y[j] = E_field_matrix_Y[closest_x_index, closest_y_index]

        # Correct sign in the acceleration calculation
        Nitrogen_distribution.AX[j] = -E_field_particle_X[j] * (Nitrogen_distribution.charge[j]) / (Nitrogen_distribution.mass[j] * Epsilon_0)
        Nitrogen_distribution.AY[j] = -E_field_particle_Y[j] * (Nitrogen_distribution.charge[j]) / (Nitrogen_distribution.mass[j] * Epsilon_0)

    #plt.plot(E_field_matrix_X)
    plt.show()
    
    return Nitrogen_distribution, laplacian_matrix_x, laplacian_matrix_y

def Forward_step_function(Nitrogen_distribution, time_step):
    Nitrogen_distribution.VX += time_step * Nitrogen_distribution.AX*0.5
    Nitrogen_distribution.VY += time_step * Nitrogen_distribution.AY*0.5
    Mesh_limit = Nitrogen_distribution.mesh_size * Nitrogen_distribution.mesh_separation

    VX = np.average(np.abs(Nitrogen_distribution.VX))
    VY = np.average(np.abs(Nitrogen_distribution.VY))
    for j in range(Nitrogen_distribution.N_particles):
        if np.abs(Nitrogen_distribution.VX[j])>10*VX or np.abs(Nitrogen_distribution.VY[j])>10*VY:
            Nitrogen_distribution.VX[j] = 0
            Nitrogen_distribution.VY[j] = 0
            Nitrogen_distribution.X[j] = 0
            Nitrogen_distribution.Y[j] = 0

    # Reflect particles at the boundaries
    Nitrogen_distribution.mesh_separation
    for j in range(Nitrogen_distribution.N_particles):
        if Nitrogen_distribution.X[j] + Nitrogen_distribution.mesh_separation*3 > Mesh_limit:
            Nitrogen_distribution.X[j] = Mesh_limit - Nitrogen_distribution.mesh_separation*3
            Nitrogen_distribution.VX[j] = -Nitrogen_distribution.VX[j] * 0.9
        elif Nitrogen_distribution.X[j] - Nitrogen_distribution.mesh_separation*3 < 0:
            Nitrogen_distribution.X[j] = Nitrogen_distribution.mesh_separation*3
            Nitrogen_distribution.VX[j] = -Nitrogen_distribution.VX[j] * 0.9
        if Nitrogen_distribution.Y[j] + Nitrogen_distribution.mesh_separation*3 > Mesh_limit:
            Nitrogen_distribution.Y[j] = Mesh_limit - Nitrogen_distribution.mesh_separation*3
            Nitrogen_distribution.VY[j] = -Nitrogen_distribution.VY[j] * 0.9
        elif Nitrogen_distribution.Y[j] - Nitrogen_distribution.mesh_separation*3 < 0:
            Nitrogen_distribution.Y[j] = Nitrogen_distribution.mesh_separation*3
            Nitrogen_distribution.VY[j] = -Nitrogen_distribution.VY[j] * 0.9
    Nitrogen_distribution.Y += time_step * Nitrogen_distribution.VY
    Nitrogen_distribution.X += time_step * Nitrogen_distribution.VX
    return Nitrogen_distribution

def simulate(Molecule1, Molecule2, Nitrogen_distribution, Time_step, N_iterations, Num_particles, E_field, screen, clock, mesh_size, mesh_separation, num_particle_1, num_particle_2, inject_1, inject_2):
    Sim_position_X = []
    Sim_position_Y = []
    Sim_temperature = []
    Nitrogen_distribution.Temperature_Velocity_Calc()
    Sim_temperature.append(Nitrogen_distribution.temperature.copy())

    laplacian_matrix_x = np.zeros([Nitrogen_distribution.mesh_size, Nitrogen_distribution.mesh_size])
    laplacian_matrix_y = np.zeros([Nitrogen_distribution.mesh_size, Nitrogen_distribution.mesh_size])

    for i in range(N_iterations):
        print(f"Iteration {i+1}/{N_iterations}")
        
        Nitrogen_distribution.inject(inject_1, inject_2, Molecule1, Molecule2)

        Nitrogen_distribution.calculate_density()
        Nitrogen_distribution.Temperature_Velocity_Calc()

        # Calculate the acceleration and electric field matrix
        Nitrogen_distribution, laplacian_matrix_x, laplacian_matrix_y = Acceleration_calculation(Nitrogen_distribution, laplacian_matrix_x, laplacian_matrix_y, E_field)

        # Update the positions of the particles
        Nitrogen_distribution = Forward_step_function(Nitrogen_distribution, Time_step)

        # Append the current positions to Sim_position
        Sim_position_X.append(Nitrogen_distribution.X.copy())
        Sim_position_Y.append(Nitrogen_distribution.Y.copy())
        Sim_temperature.append(Nitrogen_distribution.temperature.copy())

        # Update Pygame screen
        update_pygame_screen(screen, Nitrogen_distribution, mesh_size, mesh_separation)

        # Print the maximum x-coordinate of the particles
        print(f"Max X: {max(Nitrogen_distribution.X)}, Max Y: {max(Nitrogen_distribution.Y)}")
        print(f"Max VX: {max(Nitrogen_distribution.VX)}, Max VY: {max(Nitrogen_distribution.VY)} Temperature: {np.average(Nitrogen_distribution.temperature)},")

        # Control the simulation speed
        clock.tick(30)  # Adjust the value to control the speed of the simulation

    return Nitrogen_distribution, Sim_position_X, Sim_position_Y, Sim_temperature

import pygame

def update_pygame_screen(screen, Nitrogen_distribution, mesh_size, mesh_separation):
    screen.fill((0, 0, 0))  # Clear screen

    # Scaling factor for visualization
    scale = 900 / (mesh_size * mesh_separation)

    # Draw density plot
    density = Nitrogen_distribution.density
    max_density = np.max(density)
    min_density = np.min(density)
    """
    for i in range(mesh_size):
        for j in range(mesh_size):
            color_intensity = int(255 * (density[i, j] - min_density) / (max_density - min_density))
            color = (color_intensity, color_intensity, 255 - color_intensity)
            pygame.draw.rect(screen, color, pygame.Rect(int(j * scale), int(i * scale), int(scale), int(scale)))

    """
    for idx in range(len(Nitrogen_distribution.X)):
        x = int(Nitrogen_distribution.X[idx] * scale)
        y = int(Nitrogen_distribution.Y[idx] * scale)
        if 0 <= x < screen.get_width() and 0 <= y < screen.get_height():
            pygame.draw.circle(screen, (Nitrogen_distribution.color_R[idx], Nitrogen_distribution.color_G[idx], Nitrogen_distribution.color_B[idx]), (x, y), 2)

    # Update the display
    pygame.display.flip()