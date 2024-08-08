import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class PDE_matrix:
    def __init__(self, mesh_size, mesh_separation, matrix_indices):
      self.mesh_size = mesh_size
      self.mesh_separation = mesh_separation
      self.e = np.zeros([self.mesh_size, self.mesh_size]) 
      for j in range(mesh_size):
        for i in range(mesh_size):
            if i == j:
                 self.e[j, i] = matrix_indices[1]
            if i == j-1:
                 self.e[j, i] = matrix_indices[0]
            if i == j+1:
                 self.e[j, i] = matrix_indices[2]
      self.e[0, self.mesh_size-1] = matrix_indices[0]
      self.e[self.mesh_size-1, 0] = matrix_indices[2]


      

