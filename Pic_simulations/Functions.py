import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import pygame 

#Declare esential variables
Epsilon_0 = 8.8541878128 * 10**(-12)  # F/m

# Main simulation loop
def simulate(Molecule1, Molecule2, Particle_distribution, Time_step, N_iterations, E_field, screen, clock, mesh_size, mesh_separation, inject_1, inject_2, sim_data_storage):

    # Initialize Laplacian matrices
    laplacian_matrix_x = np.zeros([Particle_distribution.mesh_size, Particle_distribution.mesh_size])
    laplacian_matrix_y = np.zeros([Particle_distribution.mesh_size, Particle_distribution.mesh_size])

    screen_images = []

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

        # Update Pygame screen for visualiz.ation
        screen_images = update_pygame_screen(screen, Particle_distribution, mesh_size, mesh_separation, False, screen_images)

    #This plots the final results of the simulations
    screen_images = update_pygame_screen(screen, Particle_distribution, mesh_size, mesh_separation, True, screen_images)
    sim_data_storage.plot_data()

    #This stores the final results of the simulation for future use
    sim_data_storage.data_storage("Simulation_10")

    return Particle_distribution, sim_data_storage


def Gauss_Seidel_2D(phi, mesh_separation, density, mesh_size, max_iterations=100000, tolerance=10**(-6)):
    # Precompute constants
    idx2 = 1.0 / (mesh_separation ** 2)
    idy2 = 1.0 / (mesh_separation ** 2)
    coeff = 1 / (2 * (idx2 + idy2))
    epsilon_0 = 8.854187817e-12
    
    # Iterative solver
    for iteration in range(max_iterations):
        # Perform the update on the interior points
        phi_new = phi.copy()
        phi_new[1:-1, 1:-1] = coeff * (
            density[1:-1, 1:-1] / epsilon_0 +
            idx2 * (phi[0:-2, 1:-1] + phi[2:, 1:-1]) +
            idy2 * (phi[1:-1, 0:-2] + phi[1:-1, 2:])
        )

        # Successive Over-Relaxation (SOR) with relaxation factor 1.4
        phi_new[1:-1, 1:-1] = phi[1:-1, 1:-1] + 1 * (phi_new[1:-1, 1:-1] - phi[1:-1, 1:-1])
        
        # Check for convergence every 25 iterations
        if iteration % 25 == 0:
            # Calculate the residual
            residual = (-phi_new[1:-1, 1:-1] * (2 * idx2 + 2 * idy2) +
                        density[1:-1, 1:-1] / epsilon_0 +
                        idx2 * (phi_new[0:-2, 1:-1] + phi_new[2:, 1:-1]) +
                        idy2 * (phi_new[1:-1, 0:-2] + phi_new[1:-1, 2:]))
            
            # Calculate L2 norm of the residual
            L2 = np.sqrt(np.sum(residual**2) / (mesh_size * mesh_size))
            if L2 < tolerance:
                print(f"Converged after {iteration} iterations with L2 norm = {L2:.2e}")
                return phi_new
        
        phi = phi_new
    
    print(f"Max iterations reached with L2 norm = {L2:.2e}")
    return phi 

def computeEF_2D(phi, mesh_separation):
    return np.gradient(-phi, mesh_separation)

def Acceleration_calculation(Particle_distribution, laplacian_matrix_x, laplacian_matrix_y, E_field):
    
    # Solve for Laplacian matrices using Gauss-Seidel method
    laplacian_matrix = Gauss_Seidel_2D(laplacian_matrix_x, Particle_distribution.mesh_separation, 
                                       Particle_distribution.density, Particle_distribution.mesh_size)
    
    # Initialize electric field matrices
    E_field_Y, E_field_X = computeEF_2D(laplacian_matrix, Particle_distribution.mesh_separation)
    print(np.min(E_field_X))

    E_field_Y = E_field_Y
    E_field_X = E_field_X

    # Initialize arrays for particle-specific electric fields and accelerations
    Particle_distribution.AX = np.zeros(Particle_distribution.N_particles)
    Particle_distribution.AY = np.zeros(Particle_distribution.N_particles)

    # Assign electric field values to particles and calculate accelerations
    #plt.scatter(Particle_distribution.X, Particle_distribution.Y)
    #plt.contour(E_field_Y)
    #plt.show()
    # Initialize arrays for particle-specific electric fields and accelerations
    E_field_particle_X = np.zeros(Particle_distribution.N_particles)
    E_field_particle_Y = np.zeros(Particle_distribution.N_particles)
    Particle_distribution.AX = np.zeros(Particle_distribution.N_particles)
    Particle_distribution.AY = np.zeros(Particle_distribution.N_particles)
    # Assign electric field values to particles and calculate accelerations
    for j in range(Particle_distribution.N_particles):
        closest_x_index = int(Particle_distribution.X[j] // Particle_distribution.mesh_separation)
        closest_y_index = int(Particle_distribution.Y[j] // Particle_distribution.mesh_separation)
        closest_x_index = np.clip(closest_x_index, 0, Particle_distribution.mesh_size - 1)
        closest_y_index = np.clip(closest_y_index, 0, Particle_distribution.mesh_size - 1)
        
        E_field_particle_X[j] = E_field_X[closest_y_index, closest_x_index]
        E_field_particle_Y[j] = E_field_Y[closest_y_index, closest_x_index]
        # Correct sign in the acceleration calculation
        Particle_distribution.AX[j] = E_field_particle_X[j] * (Particle_distribution.charge[j]) / (Particle_distribution.mass[j])
        Particle_distribution.AY[j] = E_field_particle_Y[j] * (Particle_distribution.charge[j]) / (Particle_distribution.mass[j])
    """
    heatmap_ax, xedges, yedges = np.histogram2d(
        Particle_distribution.X, 
        Particle_distribution.Y, 
        bins=Particle_distribution.mesh_size, 
        range=[[0, Particle_distribution.mesh_size * Particle_distribution.mesh_separation],
               [0, Particle_distribution.mesh_size * Particle_distribution.mesh_separation]],
        weights=Particle_distribution.AX
    )

    heatmap_ay, _, _ = np.histogram2d(
        Particle_distribution.X, 
        Particle_distribution.Y, 
        bins=Particle_distribution.mesh_size, 
        range=[[0, Particle_distribution.mesh_size * Particle_distribution.mesh_separation],
               [0, Particle_distribution.mesh_size * Particle_distribution.mesh_separation]],
        weights=Particle_distribution.AY
    )

    # Normalize by the number of particles in each bin to get the average acceleration in each bin
    particle_count, _, _ = np.histogram2d(
        Particle_distribution.X, 
        Particle_distribution.Y, 
        bins=Particle_distribution.mesh_size, 
        range=[[0, Particle_distribution.mesh_size * Particle_distribution.mesh_separation],
               [0, Particle_distribution.mesh_size * Particle_distribution.mesh_separation]]
    )

    # Avoid division by zero
    heatmap_ax = np.divide(heatmap_ax, particle_count, out=np.zeros_like(heatmap_ax), where=particle_count!=0)
    heatmap_ay = np.divide(heatmap_ay, particle_count, out=np.zeros_like(heatmap_ay), where=particle_count!=0)

    # Plotting the heatmap for AX
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_ax.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='viridis')
    plt.colorbar(label='Acceleration X Component')
    plt.title('Heat Map of Acceleration X Component')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()

    # Plotting the heatmap for AY
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_ay.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='viridis')
    plt.colorbar(label='Acceleration Y Component')
    plt.title('Heat Map of Acceleration Y Component')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()
"""
    return Particle_distribution, laplacian_matrix, laplacian_matrix_y, E_field_X, E_field_Y
def Forward_step_function(Particle_distribution, time_step):
    Particle_distribution.VX += time_step * Particle_distribution.AX 
    Particle_distribution.VY += time_step * Particle_distribution.AY 
    Mesh_limit = Particle_distribution.mesh_size * Particle_distribution.mesh_separation

    VX = np.average(np.abs(Particle_distribution.VX))
    VY = np.average(np.abs(Particle_distribution.VY))

    # Check for high velocities and reset if needed
    """
    for j in range(Particle_distribution.N_particles):
        if np.abs(Particle_distribution.VX[j]) > 10 * VX or np.abs(Particle_distribution.VY[j]) > 10 * VY:
            Particle_distribution.VX[j] = 0
            Particle_distribution.VY[j] = 0
            Particle_distribution.X[j] = 0
            Particle_distribution.Y[j] = 0
    """
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

def update_pygame_screen(screen, Particle_distribution, mesh_size, mesh_separation, plot, screen_images):
    # Initialize a list to store screen images

    # Scaling factor for visualization
    scale = 900 / (mesh_size * mesh_separation)
    
    # Get screen dimensions
    screen_width, screen_height = screen.get_size()
    
    # Precompute colors to avoid accessing Particle_distribution repeatedly inside the loop
    colors = list(zip(Particle_distribution.color_R, Particle_distribution.color_G, Particle_distribution.color_B))
    

    screen.fill((0, 0, 0))  # Clear screen
        
        # Draw particles on the screen
    for idx in range(len(Particle_distribution.X)):
            # Convert positions to pixel coordinates
            x = int(Particle_distribution.X[idx] * scale)
            y = int(Particle_distribution.Y[idx] * scale)

            # Check if the particle is within the screen bounds before drawing
            if 0 <= x < screen_width and 0 <= y < screen_height:
                pygame.draw.circle(screen, colors[idx], (x, y), 2)
        
        # Store the current screen image
    image = pygame.surfarray.array3d(pygame.display.get_surface())
    screen_images.append(np.transpose(image, (1, 0, 2)))  # Transpose for matplotlib compatibility

    # Plot all stored images


    if plot:

        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            ax.imshow(screen_images[frame])
            ax.axis('off')
            return ax
        ani = animation.FuncAnimation(fig, update, frames=len(screen_images), repeat=False)
        ani.save('animation.gif', writer='ffmpeg', fps=30)  # Save the animation as an MP4 file
        plt.show()

    
    return screen_images