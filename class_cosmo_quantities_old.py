import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import interpolate
from scipy.misc import derivative
#from function_z import calculate_integrand, calculate_D1
from functools import partial
import camb


class cosmo_quantities:

    def __init__(self, z, Omega_m, H_0, Omega_Lambda):
        self.z = z
        self.Omega_m = Omega_m
        self.H_0 = H_0
        self.Omega_Lambda = Omega_Lambda
        
        self.comoving_distance = np.vectorize(self.comoving_distance_)

    def calculate_Hubble(self):
        # In units of Mpc/km/s
        H = self.H_0 * np.sqrt(self.Omega_m * (1 + self.z) ** 3 + self.Omega_Lambda)
        return H

    def calculate_Hubble_cal(self):
        # In units of Mpc/km/s
        H_cal = (self.H_0 / (1 + np.array(self.z))) * np.sqrt(self.Omega_m * (1 + np.array(self.z)) ** 3 + self.Omega_Lambda)
        return H_cal


    def calculate_Hubble_cal_star(self):
        H_cal = (self.H_0 / (1 + 10)) * np.sqrt(self.Omega_m * (1 + 10) ** 3 + self.Omega_Lambda)
        return H_cal

    def calculate_Hubble_cal_dot(self):
        #def Hubble_cal_wrapper(z):
        #    return self.calculate_Hubble_cal(z)

        #H_cal_dot = derivative(Hubble_cal_wrapper, self.z, dx=1e-6)
        
        H_cal_dot = - (self.H_0**2 / 2) *  (self.Omega_m*(1+np.array(self.z)) - 2*(1-self.Omega_m)/(1+np.array(self.z))**2)
        #D: This is the dot derivative of the comoving Hubble parameter. Above formula computes just the z-derivative.
        #D: I prefer to have analytical expressions when possible.       
        return H_cal_dot
    
    def comoving_distance_(self, z, clight=299792.458):
        # In units of Mpc
        Oml = 1 - self.Omega_m
        # Comoving distance
        result = quad(lambda x: 1/(self.H_0*np.sqrt(self.Omega_m * (1+x)**3 + Oml)), 0, z)
        value = clight * result[0]
        return np.array(value)
    
    def calculate_comoving_distance(self):
        return self.comoving_distance(self.z)

    def calculate_integrand(self, x):
        H = self.H_0 * np.sqrt(self.Omega_m * (1 + x) ** 3 + self.Omega_Lambda)
        return (1 + x) * (1 / (H / self.H_0) ** 3)

    def calculate_D1_single(self, z):
        H = self.H_0 * np.sqrt(self.Omega_m * (1 + z) ** 3 + self.Omega_Lambda)
        ini = quad(self.calculate_integrand, 0, z)[0]
        D1 = ini * (5 * self.Omega_m * H) / (2 * self.H_0)
        return D1

    def calculate_D1(self):
        if isinstance(self.z, (list, np.ndarray)):
            D1 = np.array([self.calculate_D1_single(zi) for zi in self.z])
        else:
            D1 = self.calculate_D1_single(self.z)
        return D1

    def calculate_f(self):
        # D: Careful here. This computes the derivative with respect to z of D1, not the dot derivative.
        if isinstance(self.z, (list, np.ndarray)):
            dz = self.z[1] - self.z[0]
            f_etap = np.array([derivative(self.calculate_D1_single, zi, dx=dz) for zi in self.z])
            D1 = self.calculate_D1()
            f = f_etap * (-1) * (1 + np.array(self.z)) / D1
        else:
            f_etap = derivative(self.calculate_D1_single, self.z, dx=1e-6)
            D1 = self.calculate_D1()
            f = f_etap * (-1) * (1 + np.array(self.z)) / D1
        return f

    def calculate_G(self):
        G = self.calculate_D1() * self.calculate_f()
        return G

    def derivative_5_point_stencil(self, func, z, delta_z):
        """
        Compute the derivative of a function using the 5-point stencil method.

        Parameters:
            func : callable
                Function for which the derivative is to be computed.
            z : float
                Point at which to compute the derivative.
            delta_z : float
                Step size for the stencil method.

        Returns:
            float
                Numerical approximation of the derivative of func at z.
        """
        return - ((1 + z) / (12 * delta_z)) * (
                -func(z + 2 * delta_z) + 8 * func(z + delta_z) - 8 * func(z - delta_z) + func(z - 2 * delta_z))

    def calculate_G_dot(self, delta_z=1e-6):
        # As it is now, this computes the derivative of G with respect to z, not the dot derivative.
        # D: we need to revisit the calculation of G_dot by means of the 5-pt stencil when using more than one redshift
        # Output is wrong when using more than one redshift
        def G_func(z):
            # D: I think the issue is here. calculate_f() gives a vector. Then, G_func is a vector * value of D1 
            return self.calculate_D1_single(z) * self.calculate_f()

        if isinstance(self.z, (list, np.ndarray)):
            G_dot = np.array([self.derivative_5_point_stencil(G_func, zi, delta_z) for zi in self.z])
        else:
            G_dot = np.array([self.derivative_5_point_stencil(G_func, self.z, delta_z)])
        return G_dot
    
    def calculate_G_dot_(self, delta_z = 0.05):
        # Preliminary version. Interpolates G and then computes the derivative.
        # We have to think on the z-binning. We need to compute the derivative at the center of the bin.
        # Here we are extrapolating to obtain the derivative at the edges. This might not be fully correct.
        
        G_arr = self.calculate_G()
        G_f = interpolate.interp1d(self.z, G_arr, kind='cubic', fill_value='extrapolate')
        pref = - (1+np.array(self.z)) * self.calculate_Hubble_cal() / (12 * delta_z)
               
        G_dot = pref * (-G_f(np.array(self.z) + 2*delta_z) + 8*G_f(np.array(self.z) + delta_z) - 8*G_f(np.array(self.z) - delta_z) + G_f(np.array(self.z) - 2*delta_z))
        return G_dot

    def calculate_Omega_mz(self):
        Omega_mz = (self.Omega_m * (1 + np.array(self.z)) ** 3) / (self.Omega_m * (1 + np.array(self.z)) ** 3 + (1 - self.Omega_m))
        return Omega_mz

    def calculate_I(self):
        I = self.calculate_Omega_mz() * self.calculate_D1()
        return I


if __name__ == '__main__':
    Omega_m = 0.31
    Omega_Lambda = 0.69
    H_0 = 67.66
    z_min = 0
    z_max = 10
    
    z = np.linspace(z_min, z_max, 100)

    my_cosmo = cosmo_quantities(z, Omega_m, H_0, Omega_Lambda)

    I = my_cosmo.calculate_I()

    plt.figure()
    plt.title(r'D1')
    plt.xlabel('z')
    plt.ylabel(r'Int')
    plt.plot(z+1, I, c='m')
    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.show()
    # plt.xscale('log')
