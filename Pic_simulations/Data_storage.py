import os
import matplotlib.pyplot as plt
import numpy as np
class Basic_data_storage:
    def __init__(self, Particle_stored, Num_iteration, Inject_frequ1, Inject_frequ2):
        # Particle variables
        N_tot_variables = Particle_stored.N_particles + Num_iteration * (Inject_frequ1 + Inject_frequ2)
        self.X = np.zeros([N_tot_variables, 2, Num_iteration])
        self.V = np.zeros([N_tot_variables, 2, Num_iteration])
        self.A = np.zeros([N_tot_variables, 2, Num_iteration])
        self.charge_mass = np.zeros([N_tot_variables, 2, Num_iteration])
        # Mesh variables
        self.density = np.zeros([Particle_stored.mesh_size, Particle_stored.mesh_size, Num_iteration])
        self.Pot_field = np.zeros([2, Particle_stored.mesh_size, Particle_stored.mesh_size, Num_iteration])
        self.E_field = np.zeros([2, Particle_stored.mesh_size, Particle_stored.mesh_size, Num_iteration])

    def store_data(self, Particle_stored, i, density, laplacian_X, laplacian_Y, E_field_X, E_field_Y):
        self.X[:Particle_stored.N_particles,:, i] = np.stack((Particle_stored.X, Particle_stored.Y), axis=1)
        self.V[:Particle_stored.N_particles,:, i] = np.stack((Particle_stored.VX, Particle_stored.VY), axis=1)
        self.A[:Particle_stored.N_particles,:, i] = np.stack((Particle_stored.AX, Particle_stored.AY), axis=1)
        self.density[:,:, i] = density
        self.Pot_field[:, :,:, i] = np.stack((laplacian_X, laplacian_Y))
        self.E_field[:, :,:, i] = np.stack((E_field_X, E_field_Y))
        
    def plot_data(self):
        # Generate subplots and plot the data
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        # Position, Velocity, and Acceleration plots
        avg_X = np.nanmean(np.where(self.X != 0, self.X, np.nan), axis=0)
        avg_V = np.nanmean(np.where(self.V != 0, self.V, np.nan), axis=0)
        avg_A = np.nanmean(np.where(self.A != 0, self.A, np.nan), axis=0)
        
        time_steps = avg_X.shape[1]

        axs[0, 0].plot(range(time_steps), avg_X[0, :], label='X-axis')
        axs[0, 0].plot(range(time_steps), avg_X[1, :], label='Y-axis')
        axs[0, 0].set_title('Average Position Over Time')
        axs[0, 0].set_xlabel('Time step')
        axs[0, 0].set_ylabel('Position')
        axs[0, 0].legend()

        axs[0, 1].plot(range(time_steps), avg_V[0, :], label='X-axis')
        axs[0, 1].plot(range(time_steps), avg_V[1, :], label='Y-axis')
        axs[0, 1].set_title('Average Velocity Over Time')
        axs[0, 1].set_xlabel('Time step')
        axs[0, 1].set_ylabel('Velocity')
        axs[0, 1].legend()

        axs[0, 2].plot(range(time_steps), avg_A[0, :], label='X-axis')
        axs[0, 2].plot(range(time_steps), avg_A[1, :], label='Y-axis')
        axs[0, 2].set_title('Average Acceleration Over Time')
        axs[0, 2].set_xlabel('Time step')
        axs[0, 2].set_ylabel('Acceleration')
        axs[0, 2].legend()

        # Heatmaps for Density, Potential Field, and Electric Field
        im1 = axs[1, 0].imshow(self.density[:, :, -1], aspect='auto', cmap='viridis')
        axs[1, 0].set_title('Density at Last Time Step')
        fig.colorbar(im1, ax=axs[1, 0])

        im2 = axs[1, 1].imshow(self.Pot_field[0, :, :, -1], aspect='auto', cmap='viridis')
        axs[1, 1].set_title('Potential Field (X-axis) at Last Time Step')
        fig.colorbar(im2, ax=axs[1, 1])

        im3 = axs[1, 2].imshow(self.E_field[0, :, :, -1], aspect='auto', cmap='viridis')
        axs[1, 2].set_title('Electric Field (X-axis) at Last Time Step')
        fig.colorbar(im3, ax=axs[1, 2])

        plt.tight_layout()
        return fig  # Return the figure object
    def data_storage(self, simulation_name):
        # Create a directory for the simulation
        if not os.path.exists(simulation_name):
            os.makedirs(simulation_name)
        
        # Save arrays as CSV files
        np.savetxt(os.path.join(simulation_name, "X.csv"), self.X.reshape(-1, self.X.shape[-1]), delimiter=",")
        np.savetxt(os.path.join(simulation_name, "V.csv"), self.V.reshape(-1, self.V.shape[-1]), delimiter=",")
        np.savetxt(os.path.join(simulation_name, "A.csv"), self.A.reshape(-1, self.A.shape[-1]), delimiter=",")
        np.savetxt(os.path.join(simulation_name, "density.csv"), self.density.reshape(-1, self.density.shape[-1]), delimiter=",")
        np.savetxt(os.path.join(simulation_name, "Pot_field.csv"), self.Pot_field.reshape(-1, self.Pot_field.shape[-1]), delimiter=",")
        np.savetxt(os.path.join(simulation_name, "E_field.csv"), self.E_field.reshape(-1, self.E_field.shape[-1]), delimiter=",")
        
        # Generate the plot and save it
        fig = self.plot_data()  # Ensure the plot is fully generated
        plt.savefig(os.path.join(simulation_name, f"{simulation_name}_figure.png"))
        plt.close(fig)  # Close the figure to free up memory