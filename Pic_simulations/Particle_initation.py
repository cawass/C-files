import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

class Particle_distribution:
    def __init__(self, atom, N_particles, X, Y, VX, VY, AX, AY, sigma, Meshgrid, multiplication_factor):
        self.N_particles = N_particles
        self.Meshgrid = Meshgrid
        self.X_0 = X
        self.Y_0 = Y
        self.sigma = sigma
        self.atom = atom
        self.index = np.zeros(N_particles)
        self.X = np.zeros(N_particles)
        self.Y = -np.zeros(N_particles)
        self.VX = np.zeros(N_particles)
        self.VY = np.zeros(N_particles)
        self.AX = np.zeros(N_particles)
        self.AY = np.zeros(N_particles)
        self.charge = np.ones(N_particles)*self.atom.charge*multiplication_factor
        self.mass = np.ones(N_particles)*self.atom.mass*multiplication_factor
        
        for i in range(N_particles):
            self.index[i] = i
            self.X[i] = X+np.random.normal(0, sigma)*Meshgrid.mesh_separation
            self.Y[i] = Y+np.random.normal(0, sigma)*Meshgrid.mesh_separation
            self.VX[i] = VX
            self.VY[i] = VY
            self.AX[i] = AX
            self.AY[i] = AY
        """
        clear = True
        while clear:
            clear = False
            print("Checking spacing")
            for i in range(N_particles):
                for j in range(N_particles):
                    if ((self.X[i]-self.X[j])**2+(self.Y[i]-self.Y[j])**2)**(1/2)<0.3 and i != j:
                        self.X[i] = X+np.random.normal(0, sigma)
                        self.Y[i] = Y+np.random.normal(0, sigma)
                        print(((self.X[i]-self.X[j])**2+(self.Y[i]-self.Y[j])**2)**(1/2))
                        clear = True"""
    def Multiple_molecules(self, molecule_1_distribution, molecule_2_distribution):
        for i in range(molecule_1_distribution.N_particles):
#            self.atom.name[i] = molecule_1_distribution.atom.name[i]
            self.X[i] = molecule_1_distribution.X[i]
            self.Y[i] = molecule_1_distribution.Y[i]
            self.VX[i] = molecule_1_distribution.VX[i]
            self.VY[i] = molecule_1_distribution.VY[i]
            self.AX[i] = molecule_1_distribution.AX[i]
            self.AY[i] = molecule_1_distribution.AY[i]
            self.charge[i] = molecule_1_distribution.charge[i]
            self.mass[i] = molecule_1_distribution.mass[i]
        for i in range(molecule_2_distribution.N_particles):
#            self.atom.name[i+molecule_2_distribution.N_particles] = molecule_2_distribution.atom.name[i]
            self.X[i+molecule_2_distribution.N_particles] = molecule_2_distribution.X[i]
            self.Y[i+molecule_2_distribution.N_particles] = molecule_2_distribution.Y[i]
            self.VX[i+molecule_2_distribution.N_particles] = molecule_2_distribution.VX[i]
            self.VY[i+molecule_2_distribution.N_particles] = molecule_2_distribution.VY[i]
            self.AX[i+molecule_2_distribution.N_particles] = molecule_2_distribution.AX[i]
            self.AY[i+molecule_2_distribution.N_particles] = molecule_2_distribution.AY[i]
            self.charge[i+molecule_2_distribution.N_particles] = molecule_2_distribution.charge[i]
            self.mass[i+molecule_2_distribution.N_particles] = molecule_2_distribution.mass[i]
    def plot_particle(self):
        plt.scatter(self.X, self.Y, marker=".")
        plt.scatter(self.Meshgrid.mesh_size*1/2*self.Meshgrid.mesh_separation +self.X_0, self.Meshgrid.mesh_size*1/2*self.Meshgrid.mesh_separation+ self.Y_0, marker="o")
        plt.xscale = 3*self.sigma
        plt.yscale = 3*self.sigma
        plt.show() 

    """
    def calculate_density(self):
        # Create meshgrid arrays
        x = np.linspace(0, self.Meshgrid.mesh_size * self.Meshgrid.mesh_separation, self.Meshgrid.mesh_size)
        y = np.linspace(0, self.Meshgrid.mesh_size * self.Meshgrid.mesh_separation, self.Meshgrid.mesh_size)
        xx, yy = np.meshgrid(x, y)
        
        # Flatten the meshgrid arrays
        grid_coords = np.vstack([xx.ravel(), yy.ravel()])
        
        # Stack the particle positions
        positions = np.vstack([self.X, self.Y])
        
        # Separate positive and negative charges
        positive_mask = self.charge > 0
        negative_mask = self.charge < 0
        
        # Perform kernel density estimation for positive and negative charges separately
        if np.any(positive_mask):
            kde_positive = gaussian_kde(positions[:, positive_mask], weights=self.charge[positive_mask])
            charge_density_positive = kde_positive(grid_coords).reshape(self.Meshgrid.mesh_size, self.Meshgrid.mesh_size)
        else:
            charge_density_positive = np.zeros((self.Meshgrid.mesh_size, self.Meshgrid.mesh_size))
        
        if np.any(negative_mask):
            kde_negative = gaussian_kde(positions[:, negative_mask], weights=-self.charge[negative_mask])
            charge_density_negative = kde_negative(grid_coords).reshape(self.Meshgrid.mesh_size, self.Meshgrid.mesh_size)
        else:
            charge_density_negative = np.zeros((self.Meshgrid.mesh_size, self.Meshgrid.mesh_size))
        
        # Combine positive and negative charge densities
        self.density = charge_density_positive - charge_density_negative
        
        print("Charge density:")
        print(self.density)
        print("Maximum density:", np.max(self.density))
        print("Minimum density:", np.min(self.density))
    """
    def calculate_density(self):
        # Define the meshgrid dimensions and spacing
        mesh_size = self.Meshgrid.mesh_size
        mesh_sep = self.Meshgrid.mesh_separation
        
        # Create meshgrid arrays
        x = np.linspace(0, (mesh_size - 1) * mesh_sep, mesh_size)
        y = np.linspace(0, (mesh_size - 1) * mesh_sep, mesh_size)
        xx, yy = np.meshgrid(x, y)
        
        # Flatten the meshgrid arrays
        grid_coords = np.vstack([xx.ravel(), yy.ravel()]).T
        
        # Initialize density grid
        density_grid = np.zeros((mesh_size, mesh_size))
        
        # Store critical mesh points and their corresponding densities
        critical_points = []
        critical_densities = []
        
        # Loop through each particle
        for px, py, charge in zip(self.X, self.Y, self.charge):
            # Calculate distances from the particle to all grid nodes
            distances = np.sqrt((grid_coords[:, 0] - px) ** 2 + (grid_coords[:, 1] - py) ** 2)
            
            # Find the index of the closest node
            closest_node_idx = np.argmin(distances)
            
            # Convert the linear index to a 2D index

    def calculate_density(self):
        # Create meshgrid arrays
        x = np.linspace(0, self.Meshgrid.mesh_size * self.Meshgrid.mesh_separation, self.Meshgrid.mesh_size)
        y = np.linspace(0, self.Meshgrid.mesh_size * self.Meshgrid.mesh_separation, self.Meshgrid.mesh_size)
        xx, yy = np.meshgrid(x, y)
        
        # Flatten the meshgrid arrays
        grid_coords = np.vstack([xx.ravel(), yy.ravel()]).T
        
        # Initialize density grid
        density_grid = np.zeros((self.Meshgrid.mesh_size, self.Meshgrid.mesh_size))
        
        # Loop through each particle
        for px, py, q in zip(self.X, self.Y, self.charge):
            # Calculate distances from the particle to all grid nodes
            distances = np.sqrt((grid_coords[:, 0] - px) ** 2 + (grid_coords[:, 1] - py) ** 2)
            
            # Find the index of the closest node
            closest_node_idx = np.argmin(distances)
            
            # Convert the linear index to a 2D index
            closest_node_2d_idx = np.unravel_index(closest_node_idx, (self.Meshgrid.mesh_size, self.Meshgrid.mesh_size))
            
            # Increment the density of the closest node by the charge of the particle
            density_grid[closest_node_2d_idx] += q
        
        self.density = density_grid
        
        print("Node density (charge density):")
        print(self.density)
        print("Maximum density:", np.max(self.density))
        print("Minimum density:", np.min(self.density))


    def plot_density(self, Particle_plot, num_particle_1, num_particle_2):
        # Create meshgrid arrays for plotting
        X_axis = np.linspace(0, self.Meshgrid.mesh_size * self.Meshgrid.mesh_separation, self.Meshgrid.mesh_size)
        Y_axis = np.linspace(0, self.Meshgrid.mesh_size * self.Meshgrid.mesh_separation, self.Meshgrid.mesh_size)
        
        # Plot the density using contourf
        plt.contourf(X_axis, Y_axis, self.density, 20, cmap='viridis')
        
        # Plot particles if Particle_plot is True
        if Particle_plot:
            plt.scatter(self.X[0:num_particle_1], self.Y[0:num_particle_1], marker=".", color = "blue")
            plt.scatter(self.X[num_particle_1:num_particle_2+num_particle_1], self.Y[num_particle_1:num_particle_2+num_particle_1], marker=".", color = "red")
            plt.scatter(self.X_0, self.Y_0, marker="o")
        
        # Set axis limits to ensure proper centering
        plt.xlim(X_axis[0], X_axis[-1])
        plt.ylim(Y_axis[0], Y_axis[-1])
        
        # Show the plot
        plt.show()
    
    
    def Temperature_Velocity_Calc(self):
        blotzman_constant = 1.380649*10**(-23) #J/K
        self.temperature = 1/3*self.atom.mass*(self.VX**2+ self.VY**2)/blotzman_constant
        