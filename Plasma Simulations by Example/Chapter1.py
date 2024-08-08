import numpy as np
import scipy as sp


#Physiscs constants
plank_constant = 6.62607015*10**(-34) #m^2 kg/s
blotzman_constant = 1.380649*10**(-23) #J/K
Epsilon_0 = 8.8541878128* 10**(-12) #F/m
Electron_charge = 1.602176634*10**(-19) #C

#Isa condition
Pressure_isa = 101325 #Pa
Temperature_isa = 293 #K
Critical_efield_atmos = 2.6*10**6 #V/m



#Maxwell-Boltzmann Distribution Function
#v_d = Mean drift velocity 
#v_th = Thermal velocity
def Maxwell_Boltzmann_Distribution_Function(v_th, v, v_d):
    return 1/(np.sqrt(np.pi)*v_th)*np.exp((-(v-v_d)**2))/(v_th**2)

#Thermal velocity
def Thermal_velocity(T, m):
    return np.sqrt(2*blotzman_constant*T/m)

#Mean free path
#sigma = collision cross-section
#n = number of paricles 
def Mean_free_path(Sigma, n):
    return 1/(Sigma*n)

#Knusden number
#L = Characteristic Lenght(ejm, Ionization chamber diameter)
#K_n << 1, continuum flow, Maxwellian VDf is valid
#K_n >> 1, free molecular flow
#K_n = 1, rarefield gas
def Knusden_number(Lambda, L):
    return Lambda/L

#Macroparticle weight 
def Macroparticle_weight(N_real, N_sim):
    return N_real/N_sim

#Lorentz force
#q = particle charge
#E = efield vector
#v = velocity vector
#B = magnetic field vector

def Lorentz_force(q, E, v, B):
    return q*(E+np.cross(v, B))

#Charge density Rho
#q_s = charge per species 
#n_s = charge density
def Charge_density(q_s, n_s):
    return np.sum(q_s*n_s)

#Current density j
#v_s = particle velocity
def Current_desity(q_s, n_s, v_s):
    return np.sum(q_s*n_s*v_s)

#Population based charge density
#Z_i = Average ionization state
#n_i = number of ions
#n_e = number of electrons

def Charge_density_species(Z_i, n_i, n_e):
    return Electron_charge*(Z_i*n_i-n_e)

#Uniform cartesian mesh
#x_0 = starting position
#c_i = number of cells in the i direction
#Delta_x = cell_spacing
def Uniform_cartesian_mesh(x_0, c_i, Delta_x):
    return x_0+ c_i*Delta_x

#Gauss-Seidel scheme
#phi = array containing the values of phi
#b =array containing the boundary conditions and the iration logic
def Gauss_Seidel(phi, Delta_x, b):
    for j in range(Delta_x[:]):
        phi[j] = (phi[j-1]+ phi[j+1]-Delta_x*Delta_x*b[j])/2

#Successive Over Relaxation
#y_1 = previus value of the node potential 
#y_2 = new value of the node potential
#w = relaxation value
#w > 1 -> succesive over relaxation
#w < 1 -> succesive under relaxation
def Successive_Over_Relaxation(y_1, w, y_2):
    return y_1 +w*(y_2-y_1)

#Gauss-Seidel_Relaxation(optimal value of w = 1.4)
def Gauss_Seidel_Relaxation(phi, Delta_x, b, w):
    for j in range(Delta_x[:]):
        g = (phi[j-1]+ phi[j+1]-Delta_x*Delta_x*b[j])/2
        phi[j] = phi[j] + w*(g- phi[j])