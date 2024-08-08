import numpy as np
import scipy as sp
import matplotlib


class Elem_particle:
    def __init__(self, name ,mass, charge):
        self.name = name
        self.mass = mass
        self.charge = charge

electron = Elem_particle("electron",  9.1093837015 * 10**(-31), -1.602176634*10**(-19))
proton = Elem_particle("proton", 1.67262192369*10**(-27), 1.602176634*10**(-19))
neutron = Elem_particle("neutron", 1.67492749804*10**(-27), 0)

class Elem_atom: 
    def __init__(self, name , n_proton, n_neutron, n_electron):
        self.name = name
        self.n_proton = n_proton
        self.n_neutron = n_neutron
        self.n_electron = n_electron
        self.mass = n_proton*proton.mass+n_neutron*neutron.mass+n_electron*electron.mass
        self.charge = n_proton*proton.charge+n_neutron*neutron.charge+n_electron*electron.charge
    def print_particle(self):
        print(f"{self.name}, P: {self.n_proton}, E: {self.n_electron}, N: {self.n_neutron} ")
    def ionize_particle(self, electron_movemnt, proton_movent, neutron_movent):
        self.n_electron = self.n_electron + electron_movemnt
        self.n_proton = self.n_proton + proton_movent
        self.n_neutron = self.n_neutron + neutron_movent
        self.mass = self.mass + electron_movemnt*electron.mass+proton_movent*proton.mass+neutron.mass*neutron_movent
        self.charge = self.charge + electron_movemnt*electron.charge + proton_movent*proton.charge

Nitrogen_pos = Elem_atom("N+", 6, 7, 7)
Nitrogen_neg = Elem_atom("N-", 8, 7, 7)
Nitrogen = Elem_atom("N", 7, 7, 7)
Oxygen = Elem_atom("O", 8, 8, 8)
Hidrogen = Elem_atom("H", 1, 1, 1)
Helium = Elem_atom("He", 2, 2,2)
Carbon = Elem_atom("C", 6 ,6 ,6)

class Molecule:
    def __init__(self, name, particles):
        self.name = name
        self.particles = particles
        self.mass = np.sum([particle.mass for particle in particles])
        self.charge = np.sum([particle.charge for particle in particles])
    def print_particle(self):
        print(f"{self.name}, Particles: {[particle.name for particle in self.particles]} ")

H2 = Molecule("H2", [Hidrogen, Hidrogen])
O2 = Molecule("O2",[Oxygen, Oxygen])
N2 = Molecule("N2", [Nitrogen, Nitrogen])
CH4 = Molecule("Methane", [Carbon, Hidrogen, Hidrogen, Hidrogen, Hidrogen])
H2O = Molecule("Water", [Hidrogen, Hidrogen, Oxygen])