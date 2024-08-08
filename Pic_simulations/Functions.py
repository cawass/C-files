import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
Epsilon_0 =8.8541878128* 10**(-12) #F/m

def Gauss_Seidel(phi, Delta_x, b, mesh_size):

    for j in range(1, mesh_size - 1):
            for k in range(1, mesh_size - 1):
                phi[j, k] = 0.25 * (phi[j-1, k] + phi[j+1, k] + phi[j, k-1] + phi[j, k+1] - Delta_x**2 * b[j, k])
    return phi

def Acceleration_calculation(Nitrogen_distribution, Electric_field_matrix, laplacian_matrix_x, laplacian_matrix_y):
    mesh_size = Nitrogen_distribution.Meshgrid.mesh_size
    mesh_separation = Nitrogen_distribution.Meshgrid.mesh_separation
    print(Nitrogen_distribution.density)
    
    laplacian_matrix_X = Gauss_Seidel(laplacian_matrix_x, mesh_separation, Nitrogen_distribution.density, mesh_size)
    laplacian_matrix_Y = Gauss_Seidel(laplacian_matrix_y, mesh_separation, Nitrogen_distribution.density, mesh_size)*0
    
    # Debug prints for checking Laplacian matrices
    print(np.max(laplacian_matrix_X))
    print(np.max(laplacian_matrix_Y))
    
    E_field_matrix_X = np.zeros([mesh_size, mesh_size])
    E_field_matrix_Y = np.zeros([mesh_size, mesh_size])
    
    for j in range(1, mesh_size - 1):
        for k in range(1, mesh_size - 1):
            E_field_matrix_X[j, k] = -(laplacian_matrix_X[j+1, k] - laplacian_matrix_X[j-1, k]) / (2 * mesh_separation)
            E_field_matrix_Y[j, k] = -(laplacian_matrix_Y[j, k+1] - laplacian_matrix_Y[j, k-1]) / (2 * mesh_separation)
   
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
        Nitrogen_distribution.AY[j] = E_field_particle_Y[j] * (Nitrogen_distribution.charge[j]) / (Nitrogen_distribution.mass[j] * Epsilon_0)

    #plt.plot(E_field_matrix_X)
    plt.show()
    
    return Nitrogen_distribution, laplacian_matrix_x, laplacian_matrix_y

def Forward_step_function(Nitrogen_distribution, time_step):
    Nitrogen_distribution.VX += time_step * Nitrogen_distribution.AX*0.5
    Nitrogen_distribution.VY += time_step * Nitrogen_distribution.AY*0.5
    Mesh_limit = Nitrogen_distribution.Meshgrid.mesh_size * Nitrogen_distribution.Meshgrid.mesh_separation

    VX = np.average(np.abs(Nitrogen_distribution.VX))
    VY = np.average(np.abs(Nitrogen_distribution.VY))
    for j in range(Nitrogen_distribution.N_particles):
        if np.abs(Nitrogen_distribution.VX[j])>10*VX or np.abs(Nitrogen_distribution.VY[j])>10*VY:
            Nitrogen_distribution.VX[j] = 0
            Nitrogen_distribution.VY[j] = 0
            Nitrogen_distribution.X[j] = 0
            Nitrogen_distribution.Y[j] = 0

    # Reflect particles at the boundaries
    Nitrogen_distribution.Meshgrid.mesh_separation
    for j in range(Nitrogen_distribution.N_particles):
        if Nitrogen_distribution.X[j] + Nitrogen_distribution.Meshgrid.mesh_separation*3 > Mesh_limit:
            Nitrogen_distribution.X[j] = Mesh_limit - Nitrogen_distribution.Meshgrid.mesh_separation*3
            Nitrogen_distribution.VX[j] = -Nitrogen_distribution.VX[j] * 0.9
        elif Nitrogen_distribution.X[j] - Nitrogen_distribution.Meshgrid.mesh_separation*3 < 0:
            Nitrogen_distribution.X[j] = Nitrogen_distribution.Meshgrid.mesh_separation*3
            Nitrogen_distribution.VX[j] = -Nitrogen_distribution.VX[j] * 0.9
        if Nitrogen_distribution.Y[j] + Nitrogen_distribution.Meshgrid.mesh_separation*3 > Mesh_limit:
            Nitrogen_distribution.Y[j] = Mesh_limit - Nitrogen_distribution.Meshgrid.mesh_separation*3
            Nitrogen_distribution.VY[j] = -Nitrogen_distribution.VY[j] * 0.9
        elif Nitrogen_distribution.Y[j] - Nitrogen_distribution.Meshgrid.mesh_separation*3 < 0:
            Nitrogen_distribution.Y[j] = Nitrogen_distribution.Meshgrid.mesh_separation*3
            Nitrogen_distribution.VY[j] = -Nitrogen_distribution.VY[j] * 0.9
    Nitrogen_distribution.Y += time_step * Nitrogen_distribution.VY
    Nitrogen_distribution.X += time_step * Nitrogen_distribution.VX
    return Nitrogen_distribution
def simulate(Nitrogen_distribution, Potential_field_matrix, Electric_field_matrix, Time_step, N_iterations, Num_particles):
    Sim_position_X = []
    Sim_position_Y = []
    Sim_temperature = []
    Nitrogen_distribution.Temperature_Velocity_Calc()
    Sim_temperature.append(Nitrogen_distribution.temperature.copy())

    laplacian_matrix_x = np.zeros([Nitrogen_distribution.Meshgrid.mesh_size, Nitrogen_distribution.Meshgrid.mesh_size])
    laplacian_matrix_y = np.zeros([Nitrogen_distribution.Meshgrid.mesh_size, Nitrogen_distribution.Meshgrid.mesh_size])
    for i in range(N_iterations):
        print(f"Iteration {i+1}/{N_iterations}")
        Nitrogen_distribution.calculate_density()
        Nitrogen_distribution.Temperature_Velocity_Calc()
        
        # Calculate the acceleration and electric field matrix
        Nitrogen_distribution, laplacian_matrix_x, laplacian_matrix_y = Acceleration_calculation(Nitrogen_distribution, Electric_field_matrix, laplacian_matrix_x, laplacian_matrix_y)
        # Update the positions of the particles
        Nitrogen_distribution = Forward_step_function(Nitrogen_distribution, Time_step)
        
        # Print the maximum x-coordinate of the particles
        print(f"Max X: {max(Nitrogen_distribution.X)}, Max Y: {max(Nitrogen_distribution.Y)}" )
        print(f"Max VX: {max(Nitrogen_distribution.VX)}, Max VY: {max(Nitrogen_distribution.VY)} Temperature: {np.average(Nitrogen_distribution.temperature)}," )
        
        # Append the current positions to Sim_position
        Sim_position_X.append(Nitrogen_distribution.X.copy())
        Sim_position_Y.append(Nitrogen_distribution.Y.copy())
        Sim_temperature.append(Nitrogen_distribution.temperature.copy())
        # Recalculate the density
    return Nitrogen_distribution, Sim_position_X, Sim_position_Y, Sim_temperature

def animate_particles(Nitrogen_distribution, Sim_position_X, Sim_position_Y, Sim_temperature, N_iterations, mesh_size, mesh_separation, num_particle_1, num_particle_2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    X_axis = np.linspace(0, mesh_size * mesh_separation, mesh_size)
    Y_axis = np.linspace(0, mesh_size * mesh_separation, mesh_size)
    X, Y = np.meshgrid(X_axis, Y_axis)
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        Nitrogen_distribution.X = Sim_position_X[frame]
        Nitrogen_distribution.Y = Sim_position_Y[frame]
        
        if frame % 2 == 0:
            Nitrogen_distribution.calculate_density()

        # Update particle position plot
        ax1.contourf(X_axis, Y_axis, Nitrogen_distribution.density, 20, cmap='viridis')

        # Plot particles if Particle_plot is True
        ax1.scatter(Nitrogen_distribution.X[0:num_particle_1], Nitrogen_distribution.Y[0:num_particle_1], marker=".", color = "blue")
        ax1.scatter(Nitrogen_distribution.X[num_particle_1:num_particle_2+num_particle_1], Nitrogen_distribution.Y[num_particle_1:num_particle_2+num_particle_1], marker=".", color = "red")
        ax1.scatter(Nitrogen_distribution.X_0, Nitrogen_distribution.Y_0, marker="o")
        ax1.set_title(f"Iteration {frame + 1}/{N_iterations}")

        # Update temperature plot
        avg_temp = [np.mean(temp) for temp in Sim_temperature[:frame+1]]
        ax2.plot(avg_temp, color='blue')
        ax2.set_xlim(0, N_iterations)
        ax2.set_ylim(0, max(avg_temp) * 1.1)
        ax2.set_title("Average Particle Temperature")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Temperature (K)")
        
        return ax1, ax2
    ani = FuncAnimation(fig, update, frames=N_iterations, interval=10, blit=False)
    plt.show()