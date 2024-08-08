import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#Physiscs constants
plank_constant = 6.62607015*10**(-34) #m^2 kg/s
blotzman_constant = 1.380649*10**(-23) #J/K
Epsilon_0 = 1# 8.8541878128* 10**(-12) #F/m


#Isa condition
Pressure_isa = 101325 #Pa
Temperature_isa = 293 #K
Critical_efield_atmos = 2.6*10**6 #V/m

#"C:\Users\carlo\Ion-plane-project\NC-files\Research\Scientific paper\Electric discharges\Cooray_-Discharge-Physics-Course.pdf"
Volt  = 1.602176634*10**(-19)#Electron volt
Ionization_cross_section_units = 10**(-11) #cm^2
pressure = 1 #atm
V_over_N = 10**(-6) #cm^3/s
E_over_N = 10**(-4) #Vcm^2
Number_critical_particles = 10**(8) #NU
Active_region_steamer = 0.2*10**(-3) #m
Propagation_voltage_positive = 5*10**(5) #V/m
Propagation_voltage_negative = 1.5 * 10**(6) #V/m
Steamer_speed = 10**6 #m/s

def Mean_free_path(density, cross_section):
    return 1/(density*cross_section)

def Surviving_electrons(Mean_free_path, initial_num_electrons, X):
    return initial_num_electrons*np.exp(-X/Mean_free_path)

def Microscopic_cross_section(elastic_cross_section, exitation_cross_section, ionization_cross_section, attachment_cross_section, other_cross_section):
    return elastic_cross_section + exitation_cross_section + ionization_cross_section+ attachment_cross_section + other_cross_section
   
def Macroscopic_cross_section(elastic_cross_section, exitation_cross_section, ionization_cross_section, attachment_cross_section, other_cross_section):
    return elastic_cross_section + exitation_cross_section + ionization_cross_section+ attachment_cross_section + other_cross_section

def Inelastic_collision(Energy_incoming_atom, Energy_stationary_initial, Energy_stationary_final, Potential_stationary):
    return Energy_stationary_initial+ Energy_stationary_final+ Potential_stationary

def Moment_transfer(Mass_1, Mass_2, V_1):
    return (np.divide(Mass_2*Mass_1*V_1**2, (Mass_1 + Mass_2)*2))

def Drift_velocity(particle_movility, Electric_field):
    return particle_movility*Electric_field

def Number_ionization_collision(concentration_atoms, microscopic_cross_section):
    return concentration_atoms*microscopic_cross_section

def Ionization_cross_section(atom, potential_energy):
    average_molecule = np.array([
        [15, 0.02],
        [16, 0.05],
        [17, 0.1],
        [19, 0.2],
        [20, 0.35],
        [30, 1],
        [40, 2],
        [50, 2.5],
        [100, 3.5],
        [200, 3],
        [500, 2],
        [1000, 1.35]
    ])
    average_molecule[:, 0] = average_molecule[:, 0] * Volt
    average_molecule[:, 1] = average_molecule[:, 1] * Ionization_cross_section_units

    plt.plot(average_molecule[:, 0], average_molecule[:, 1], marker='o')
    plt.xlabel('Potential Energy (V)')
    plt.ylabel('Ionization Cross Section')
    plt.title(f'Ionization Cross Section for {atom}')
    plt.show()

def Photo_ionisation(ionization_energy):
    return ionization_energy/plank_constant

def Saha_equation(beta_initial_guess, ionization_energy, pressure, temperature):
    def solver(beta, ionization_energy, pressure, temperature):
        a = beta**2/(1-beta**2)-np.divide(2.4*10**(-4)*(temperature)**2.5*np.exp(-ionization_energy/(blotzman_constant*temperature)), (pressure))
        return a
    return sp.optimize.fsolve(solver, beta_initial_guess, args=(ionization_energy,  pressure, temperature ))
#print(Saha_equation(0.6, 12.5*Volt, 1, 9000))

def Attachment_frequency(density_free_electrons, lifetime_free_electron, time_passed):
    return density_free_electrons*np.exp(-lifetime_free_electron*time_passed)

def Critical_efield(print):
    E_N = np.array([8, 9, 10, 11, 12, 13, 14])*E_over_N
    nu_a_N = np.array([7e-12, 7.3e-12, 7.6e-12, 7.9e-12, 8.2e-12, 7.9e-12, 7.9e-12])*V_over_N
    nu_i_N = np.array([1e-12, 2e-12, 3e-12, 5.75e-12, 7e-12, 8e-12, 1e-11])*V_over_N
    interp_nu_a_N = sp.interpolate.interp1d(E_N, nu_a_N)
    interp_nu_i_N = sp.interpolate.interp1d(E_N, nu_i_N)
    E_N_interp = np.linspace(0.0008, 0.0014, 100)
    nu_a_N_interp = interp_nu_a_N(E_N_interp)
    nu_i_N_interp = interp_nu_i_N(E_N_interp)
    if print:
        plt.figure()
        plt.plot(E_N, nu_a_N, 'o', label='Data Points νa/N')
        plt.plot(E_N_interp, nu_a_N_interp, '-', label='Interpolated νa/N')
        plt.plot(E_N, nu_i_N, 'o', label='Data Points νi/N')
        plt.plot(E_N_interp, nu_i_N_interp, '-', label='Interpolated νi/N')
        plt.yscale('log')
        plt.xlabel('E/N (10^-16 V m^2)')
        plt.ylabel('ν/N (m^3/s)')
        plt.legend()
        plt.show()
    return interp_nu_a_N, interp_nu_i_N
# a, b = Critical_efield(True)

def Critical_efield_distance(print):
    Distance = np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3])
    E_field = np.array([0.4, 0.278, 0.264, 0.26, 0.258, 0.257, 0.2565])*1000
    interp_E_field = sp.interpolate.interp1d(Distance, E_field)
    E_N_interp = np.linspace(0.0, 0.3, 100)
    E_N_interp_plot = interp_E_field(E_N_interp)
    if print:
        plt.figure()
        plt.plot(Distance, E_field,  'o', label='Data Points νa/N')
        plt.plot(E_N_interp, E_N_interp_plot, '-', label='Interpolated νa/N')
        plt.yscale('log')
        plt.xlabel('d (m)')
        plt.ylabel('E (m^3/s)')
        plt.legend()
        plt.show()
    return interp_E_field
# a = Critical_efield_distance(True)

#Non sustained discharge, primary ionization
def Primary_ionization_stage(alpha, distance, electrons_leaving_cathode, Initial_curent):
    return electrons_leaving_cathode*np.exp(alpha*distance), Initial_curent*np.exp(alpha*distance)

#Non sustained discharge, secondary ionization
def Secondary_ionization_stage(alpha, distance, electrons_leaving_cathode, Initial_curent, Townsend_coefficient):
    return electrons_leaving_cathode*np.exp(alpha*distance)/(1-Townsend_coefficient*(np.exp(alpha*distance)-1)), Initial_curent*np.exp(alpha*distance)/(1-Townsend_coefficient*(np.exp(alpha*distance)-1))

#Non sustained discharge, secondary ionization, particle recombination
def Secondary_ionization_stage_recombination(alpha, distance, electrons_leaving_cathode, Initial_curent, Townsend_coefficient, reattachment_number):
    coefficent = (alpha/(alpha-reattachment_number)*np.exp(alpha-reattachment_number)-reattachment_number/(alpha-reattachment_number))/(1-Townsend_coefficient*alpha/(alpha-reattachment_number)*(np.exp((alpha-reattachment_number)*distance)-1))
    return electrons_leaving_cathode*coefficent, Initial_curent*coefficent

#Townsend´s derivation of alpha
def Toensends_alpha(A, B, Efield, pressure):
    return A*np.exp(-B/(Efield/pressure))

#Paschen´s law experimental data
def Paschens_law_experimental(print):
    Distance_pressure = np.array([10**(-6), 10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1)]) #bar m
    E_field = np.array([200, 350, 1100, 3000, 50000, 500000]) #Volts
    interp_E_field = sp.interpolate.interp1d(Distance_pressure, E_field)
    E_N_interp = np.linspace(10**(-6), 10**(-1), 100)
    E_N_interp_plot = interp_E_field(E_N_interp)
    if print:
        plt.figure()
        plt.plot(Distance_pressure, E_field,  'o', label='Data Points νa/N')
        plt.plot(E_N_interp, E_N_interp_plot, '-', label='Interpolated νa/N')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('dp (m bar)')
        plt.ylabel('P (V)')
        plt.legend()
        plt.show()
    return interp_E_field
#Paschens_law(True)

def Paschens_law_numerical_der(A, B, pressure, distance, K):
    return (B*pressure*distance)/(np.log(A*pressure*distance/K))

def Average_radial_distance_diffusion(D, t):
    return np.sqrt(4*D*t)

def E_field_head_avalanche(alpha, x, drift_velocity, D):
    return np.exp(alpha*x)*drift_velocity/(16*np.pi()^2*Epsilon_0*D*x)

def inception_criterion(alpha, X_f):
    return sp.integrate.quad(lambda x: alpha(x), 0, X_f)>18

def Electrical_breakdown_non_atmosphere(p, T):
    return Critical_efield_atmos*(p*Temperature_isa)/(Pressure_isa*T)

#"C:\Users\carlo\Ion-plane-project\NC-files\Research\Scientific paper\Space ion engines\Fundamentals of Electric.pdf"
def Temperature_Velocity_Calc(m, vx, vy):
    return 1/2*m*(vx**2+ vy**2)/blotzman_constant