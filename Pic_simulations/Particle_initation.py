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
        self.VX_0 = VX
        self.VY_0 = VY
        self.AX_0 = AX
        self.AY_0 = AY
        self.sigma = sigma
        self.atom = atom
        self.charge_0 = self.atom.charge*multiplication_factor
        self.mass_0 = self.atom.mass*multiplication_factor
       
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

    def inject(self, inject_1, inject_2, original_1, original_2):
        # Adjust the total number of particles in the main distribution and original distributions
        self.N_particles += inject_1 + inject_2
        original_1.N_particles += inject_1
        original_2.N_particles += inject_2

        # Inject new particles from original_1
        for i in range(inject_1):
            self.X = np.append(self.X, (original_1.X_0  +np.random.normal(0, original_1.sigma)*original_1.Meshgrid.mesh_separation))
            self.Y = np.append(self.Y, (original_1.Y_0 + +np.random.normal(0, original_1.sigma)*original_1.Meshgrid.mesh_separation))
            self.VX = np.append(self.VX, original_1.VX_0)
            self.VY = np.append(self.VY, original_1.VY_0)
            self.AX = np.append(self.AX, original_1.AX_0)
            self.AY = np.append(self.AY, original_1.AY_0)
            self.charge = np.append(self.charge, original_1.charge_0)
            self.mass = np.append(self.mass, original_1.mass_0)

        # Inject new particles from original_2
        for i in range(inject_2):
            self.X = np.append(self.X, (original_2.X_0 +np.random.normal(0, original_2.sigma)*original_2.Meshgrid.mesh_separation))
            self.Y = np.append(self.Y, (original_2.Y_0+np.random.normal(0, original_2.sigma)*original_2.Meshgrid.mesh_separation))
            self.VX = np.append(self.VX, original_2.VX_0)
            self.VY = np.append(self.VY, original_2.VY_0)
            self.AX = np.append(self.AX, original_2.AX_0)
            self.AY = np.append(self.AY, original_2.AY_0)
            self.charge = np.append(self.charge, original_2.charge_0)
            self.mass = np.append(self.mass, original_2.mass_0)

        
        