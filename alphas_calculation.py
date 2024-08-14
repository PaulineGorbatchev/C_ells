'''%matplotlib inline
%config InlineBackend.figure_format = 'retina' '''
import sys, platform, os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
import pickle as pkl

# We can generate the C ell's with the source terms specified above
from camb.sources import GaussianSourceWindow
from class_cosmo_quantities_old import *


#Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
#This file is then in the docs folders
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))


class alphas:

    def __init__(self, z, Omega_m, H_0, Omega_Lambda, ombh2, omch2, mnu, omk, tau, As, ns, r, sigma, bias_F, bias_B, lmax):
        self.z = z
        self.Omega_m = Omega_m
        self.H_0 = H_0
        self.Omega_Lambda = Omega_Lambda
        self.ombh2 = ombh2
        self.omch2 = omch2
        self.mnu = mnu
        self.omk = omk
        self.tau = tau
        self.As = As
        self.ns = ns
        self.r = r
        self.sigma = sigma
        self.bias_F = bias_F
        self.bias_B = bias_B
        self.lmax = lmax
        
    def sigma8(self, zin=10., zfin=0.):
        
        """
        Computes the Sigma8 function.

        Parameters\n        ----------\n   
             
        zin : float
        The maximum redshift.
        
        zfin : float, the latest redshift.
        
        Returns
        -------
        interp1d
        A cubic spline interpolation object of Sigma8.
        """
        
        pars = camb.CAMBparams()
        # Set the cosmological parameters
        pars.set_cosmology(H0=self.H_0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk, tau=self.tau)
        # Set the parameter values for the initial power spectrum
        pars.InitPower.set_params(As=self.As, ns=self.ns, r=self.r)
        # Enable transfer functions
        zs = np.linspace(zfin, zin, 100)
        pars.set_matter_power(redshifts=zs, kmax=2.0)
        
        # Compute results
        results = camb.get_results(pars)

        sigma_8_uns = results.get_sigma8()
        sigma8_sorted = np.flip(sigma_8_uns)
        
        return interp1d(zs, sigma8_sorted, kind='cubic', fill_value='extrapolate')

    def calculate_alpha_1(self):
# Set up the CAMB parameters
        pars = camb.CAMBparams()

        # Set the cosmological parameters
        pars.set_cosmology(H0=self.H_0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk, tau=self.tau)
        # Set the parameter values for the initial power spectrum
        pars.InitPower.set_params(As=self.As, ns=self.ns, r=self.r)
        # Enable transfer functions
        pars.set_matter_power(redshifts=[self.z], kmax=2.0)


# We can choose which source terms to include: for now, we just choose density and RSD
        pars.SourceTerms.counts_density = True
        pars.SourceTerms.counts_redshift = False
        pars.SourceTerms.counts_lensing = False
        pars.SourceTerms.counts_velocity = False
        pars.SourceTerms.counts_radial = False
        pars.SourceTerms.counts_timedelay = False
        pars.SourceTerms.counts_ISW = False
        pars.SourceTerms.counts_potential = False
        pars.SourceTerms.counts_evolve = False
        pars.SourceTerms.use_21cm_mK = False


        # Let's double-check that we have indeed included the terms that we want
        #print(pars.SourceTerms)

        #GENERATE THE CL's


        # Set up the window functions that will be correlated, later labelled W1, W2 and so on. Here we choose two gaussian windows, with a chosen central redshift and width.
        # You can also specify the galaxy bias and, to allow for multiple populations of galaxies, you can choose two identical windows with different bias values.
        pars.SourceWindows = [
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_F, sigma=self.sigma),
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_B, sigma=self.sigma)]
        results = camb.get_results(pars)
        cls = results.get_source_cls_dict(lmax = self.lmax, raw_cl = True) # raw_cl gives the C ell's without the factor 2pi/(l(l + 1)). The output corresponds to ell from 0 to lmax.



        W1W1 = cls['W1xW1']

        sigma_8 = results.get_sigma8()

        result = W1W1 / (sigma_8**2 * self.bias_B * self.bias_F)

        return result, W1W1


    def calculate_alpha_2(self):
# Set up the CAMB parameters
        pars = camb.CAMBparams()

        # Set the cosmological parameters
        pars.set_cosmology(H0=self.H_0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk, tau=self.tau)
        # Set the parameter values for the initial power spectrum
        pars.InitPower.set_params(As=self.As, ns=self.ns, r=self.r)
        # Enable transfer functions
        pars.set_matter_power(redshifts=[self.z], kmax=2.0)




# We can choose which source terms to include: for now, we just choose density and RSD
        pars.SourceTerms.counts_density = False
        pars.SourceTerms.counts_redshift = True
        pars.SourceTerms.counts_lensing = False
        pars.SourceTerms.counts_velocity = False
        pars.SourceTerms.counts_radial = False
        pars.SourceTerms.counts_timedelay = False
        pars.SourceTerms.counts_ISW = False
        pars.SourceTerms.counts_potential = False
        pars.SourceTerms.counts_evolve = False
        pars.SourceTerms.use_21cm_mK = False


        # Let's double-check that we have indeed included the terms that we want
        #print(pars.SourceTerms)

        #GENERATE THE CL's


        # Set up the window functions that will be correlated, later labelled W1, W2 and so on. Here we choose two gaussian windows, with a chosen central redshift and width.
        # You can also specify the galaxy bias and, to allow for multiple populations of galaxies, you can choose two identical windows with different bias values.
        pars.SourceWindows = [
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_F, sigma=self.sigma),
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_B, sigma=self.sigma)]
        results = camb.get_results(pars)
        cls = results.get_source_cls_dict(lmax = self.lmax, raw_cl = True) # raw_cl gives the C ell's without the factor 2pi/(l(l + 1)). The output corresponds to ell from 0 to lmax.



        W1W1 = cls['W1xW1']

        sigma_8 = results.get_sigma8()

        my_cosmo = cosmo_quantities(self.z, self.Omega_m, self.H_0, self.Omega_Lambda)
        G = my_cosmo.calculate_G()
        D1 = my_cosmo.calculate_D1()

        result = - W1W1 * ((sigma_8**2 * G ** 2 / (D1 ** 2))) ** (-1)

        return result, W1W1

    def calculate_alpha_3(self):
# Set up the CAMB parameters
        pars1 = camb.CAMBparams()

        # Set the cosmological parameters
        pars1.set_cosmology(H0=self.H_0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk, tau=self.tau)
        # Set the parameter values for the initial power spectrum
        pars1.InitPower.set_params(As=self.As, ns=self.ns, r=self.r)
        # Enable transfer functions
        pars1.set_matter_power(redshifts=[self.z], kmax=2.0)

# We can choose which source terms to include: for now, we just choose density and RSD
        pars1.SourceTerms.counts_density = True
        pars1.SourceTerms.counts_redshift = True
        pars1.SourceTerms.counts_lensing = False
        pars1.SourceTerms.counts_velocity = False
        pars1.SourceTerms.counts_radial = False
        pars1.SourceTerms.counts_timedelay = False
        pars1.SourceTerms.counts_ISW = False
        pars1.SourceTerms.counts_potential = False
        pars1.SourceTerms.counts_evolve = False
        pars1.SourceTerms.use_21cm_mK = False


        # Let's double-check that we have indeed included the terms that we want
        #print(pars.SourceTerms)

        #GENERATE THE CL's


        # Set up the window functions that will be correlated, later labelled W1, W2 and so on. Here we choose two gaussian windows, with a chosen central redshift and width.
        # You can also specify the galaxy bias and, to allow for multiple populations of galaxies, you can choose two identical windows with different bias values.
        pars1.SourceWindows = [
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_F, sigma=self.sigma),
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_B, sigma=self.sigma)]
        results1 = camb.get_results(pars1)
        cls1 = results1.get_source_cls_dict(lmax = self.lmax, raw_cl = True) # raw_cl gives the C ell's without the factor 2pi/(l(l + 1)). The output corresponds to ell from 0 to lmax.

        W1W11 = cls1['W1xW1']


        pars2 = camb.CAMBparams()

        # Set the cosmological parameters
        pars2.set_cosmology(H0=self.H_0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk, tau=self.tau)
        # Set the parameter values for the initial power spectrum
        pars2.InitPower.set_params(As=self.As, ns=self.ns, r=self.r)
        # Enable transfer functions
        pars2.set_matter_power(redshifts=[self.z], kmax=2.0)

# We can choose which source terms to include: for now, we just choose density and RSD
        pars2.SourceTerms.counts_density = True
        pars2.SourceTerms.counts_redshift = False
        pars2.SourceTerms.counts_lensing = False
        pars2.SourceTerms.counts_velocity = False
        pars2.SourceTerms.counts_radial = False
        pars2.SourceTerms.counts_timedelay = False
        pars2.SourceTerms.counts_ISW = False
        pars2.SourceTerms.counts_potential = False
        pars2.SourceTerms.counts_evolve = False
        pars2.SourceTerms.use_21cm_mK = False


        # Let's double-check that we have indeed included the terms that we want
        #print(pars.SourceTerms)

        #GENERATE THE CL's


        # Set up the window functions that will be correlated, later labelled W1, W2 and so on. Here we choose two gaussian windows, with a chosen central redshift and width.
        # You can also specify the galaxy bias and, to allow for multiple populations of galaxies, you can choose two identical windows with different bias values.
        pars2.SourceWindows = [
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_F, sigma=self.sigma),
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_B, sigma=self.sigma)]
        results2 = camb.get_results(pars2)
        cls2 = results2.get_source_cls_dict(lmax = self.lmax, raw_cl = True) # raw_cl gives the C ell's without the factor 2pi/(l(l + 1)). The output corresponds to ell from 0 to lmax.

        W1W12 = cls2['W1xW1']



        pars3 = camb.CAMBparams()

        # Set the cosmological parameters
        pars3.set_cosmology(H0=self.H_0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk, tau=self.tau)
        # Set the parameter values for the initial power spectrum
        pars3.InitPower.set_params(As=self.As, ns=self.ns, r=self.r)
        # Enable transfer functions
        pars3.set_matter_power(redshifts=[self.z], kmax=2.0)

# We can choose which source terms to include: for now, we just choose density and RSD
        pars3.SourceTerms.counts_density = False
        pars3.SourceTerms.counts_redshift = True
        pars3.SourceTerms.counts_lensing = False
        pars3.SourceTerms.counts_velocity = False
        pars3.SourceTerms.counts_radial = False
        pars3.SourceTerms.counts_timedelay = False
        pars3.SourceTerms.counts_ISW = False
        pars3.SourceTerms.counts_potential = False
        pars3.SourceTerms.counts_evolve = False
        pars3.SourceTerms.use_21cm_mK = False


        # Let's double-check that we have indeed included the terms that we want
        #print(pars.SourceTerms)

        #GENERATE THE CL's


        # Set up the window functions that will be correlated, later labelled W1, W2 and so on. Here we choose two gaussian windows, with a chosen central redshift and width.
        # You can also specify the galaxy bias and, to allow for multiple populations of galaxies, you can choose two identical windows with different bias values.
        pars3.SourceWindows = [
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_F, sigma=self.sigma),
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_B, sigma=self.sigma)]
        results3 = camb.get_results(pars3)
        cls3 = results3.get_source_cls_dict(lmax = self.lmax, raw_cl = True) # raw_cl gives the C ell's without the factor 2pi/(l(l + 1)). The output corresponds to ell from 0 to lmax.

        W1W13 = cls3['W1xW1']

        res = W1W11 - W1W12 - W1W13



        sigma_8_1 = results1.get_sigma8()
        sigma_8_2 = results2.get_sigma8()


        my_cosmo = cosmo_quantities(self.z, self.Omega_m, self.H_0, self.Omega_Lambda)
        G = my_cosmo.calculate_G()
        D1 = my_cosmo.calculate_D1()

        result = -  res * ((sigma_8_1 * G / D1 ) * (bias_B * sigma_8_1 + bias_F * sigma_8_1))**(-1)

        return result,  res


    def calculate_alpha_4(self):
# Set up the CAMB parameters
        pars1 = camb.CAMBparams()

        # Set the cosmological parameters
        pars1.set_cosmology(H0=self.H_0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk, tau=self.tau)
        # Set the parameter values for the initial power spectrum
        pars1.InitPower.set_params(As=self.As, ns=self.ns, r=self.r)
        # Enable transfer functions
        pars1.set_matter_power(redshifts=[self.z], kmax=2.0)

# We can choose which source terms to include: for now, we just choose density and RSD
        pars1.SourceTerms.counts_density = True
        pars1.SourceTerms.counts_redshift = False
        pars1.SourceTerms.counts_lensing = False
        pars1.SourceTerms.counts_velocity = True
        pars1.SourceTerms.counts_radial = False
        pars1.SourceTerms.counts_timedelay = False
        pars1.SourceTerms.counts_ISW = False
        pars1.SourceTerms.counts_potential = False
        pars1.SourceTerms.counts_evolve = False
        pars1.SourceTerms.use_21cm_mK = False


        # Let's double-check that we have indeed included the terms that we want
        #print(pars.SourceTerms)

        #GENERATE THE CL's


        # Set up the window functions that will be correlated, later labelled W1, W2 and so on. Here we choose two gaussian windows, with a chosen central redshift and width.
        # You can also specify the galaxy bias and, to allow for multiple populations of galaxies, you can choose two identical windows with different bias values.
        pars1.SourceWindows = [
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_F, sigma=self.sigma),
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_B, sigma=self.sigma)]
        results1 = camb.get_results(pars1)
        cls1 = results1.get_source_cls_dict(lmax = self.lmax, raw_cl = True) # raw_cl gives the C ell's without the factor 2pi/(l(l + 1)). The output corresponds to ell from 0 to lmax.

        W1W11 = cls1['W1xW1']


        pars2 = camb.CAMBparams()

        # Set the cosmological parameters
        pars2.set_cosmology(H0=self.H_0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk, tau=self.tau)
        # Set the parameter values for the initial power spectrum
        pars2.InitPower.set_params(As=self.As, ns=self.ns, r=self.r)
        # Enable transfer functions
        pars2.set_matter_power(redshifts=[self.z], kmax=2.0)

# We can choose which source terms to include: for now, we just choose density and RSD
        pars2.SourceTerms.counts_density = True
        pars2.SourceTerms.counts_redshift = False
        pars2.SourceTerms.counts_lensing = False
        pars2.SourceTerms.counts_velocity = False
        pars2.SourceTerms.counts_radial = False
        pars2.SourceTerms.counts_timedelay = False
        pars2.SourceTerms.counts_ISW = False
        pars2.SourceTerms.counts_potential = False
        pars2.SourceTerms.counts_evolve = False
        pars2.SourceTerms.use_21cm_mK = False


        # Let's double-check that we have indeed included the terms that we want
        #print(pars.SourceTerms)

        #GENERATE THE CL's


        # Set up the window functions that will be correlated, later labelled W1, W2 and so on. Here we choose two gaussian windows, with a chosen central redshift and width.
        # You can also specify the galaxy bias and, to allow for multiple populations of galaxies, you can choose two identical windows with different bias values.
        pars2.SourceWindows = [
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_F, sigma=self.sigma),
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_B, sigma=self.sigma)]
        results2 = camb.get_results(pars2)
        cls2 = results2.get_source_cls_dict(lmax = self.lmax, raw_cl = True) # raw_cl gives the C ell's without the factor 2pi/(l(l + 1)). The output corresponds to ell from 0 to lmax.

        W1W12 = cls2['W1xW1']



        pars3 = camb.CAMBparams()

        # Set the cosmological parameters
        pars3.set_cosmology(H0=self.H_0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk, tau=self.tau)
        # Set the parameter values for the initial power spectrum
        pars3.InitPower.set_params(As=self.As, ns=self.ns, r=self.r)
        # Enable transfer functions
        pars3.set_matter_power(redshifts=[self.z], kmax=2.0)

# We can choose which source terms to include: for now, we just choose density and RSD
        pars3.SourceTerms.counts_density = False
        pars3.SourceTerms.counts_redshift = False
        pars3.SourceTerms.counts_lensing = False
        pars3.SourceTerms.counts_velocity = True
        pars3.SourceTerms.counts_radial = False
        pars3.SourceTerms.counts_timedelay = False
        pars3.SourceTerms.counts_ISW = False
        pars3.SourceTerms.counts_potential = False
        pars3.SourceTerms.counts_evolve = False
        pars3.SourceTerms.use_21cm_mK = False


        # Let's double-check that we have indeed included the terms that we want
        #print(pars.SourceTerms)

        #GENERATE THE CL's


        # Set up the window functions that will be correlated, later labelled W1, W2 and so on. Here we choose two gaussian windows, with a chosen central redshift and width.
        # You can also specify the galaxy bias and, to allow for multiple populations of galaxies, you can choose two identical windows with different bias values.
        pars3.SourceWindows = [
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_F, sigma=self.sigma),
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_B, sigma=self.sigma)]
        results3 = camb.get_results(pars3)
        cls3 = results3.get_source_cls_dict(lmax = self.lmax, raw_cl = True) # raw_cl gives the C ell's without the factor 2pi/(l(l + 1)). The output corresponds to ell from 0 to lmax.

        W1W13 = cls3['W1xW1']


        res = W1W11 - W1W12 - W1W13


        pars4 = camb.CAMBparams()
        # Set the cosmological parameters
        pars4.set_cosmology(H0=self.H_0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk, tau=self.tau)
        # Set the parameter values for the initial power spectrum
        pars4.InitPower.set_params(As=self.As, ns=self.ns, r=self.r)
        # Enable transfer functions
        pars4.set_matter_power(redshifts=[10], kmax=2.0)

        results4 = camb.get_results(pars4)

        sigma_8_1 = results1.get_sigma8()
        sigma_8_2 = results4.get_sigma8()


        my_cosmo = cosmo_quantities(self.z, self.Omega_m, self.H_0, self.Omega_Lambda)
        G = my_cosmo.calculate_G()
        D1 = my_cosmo.calculate_D1()
        D1_star = my_cosmo.calculate_D1_single(10)
        H_cal_star = my_cosmo.calculate_Hubble_cal_star()
        H_cal = my_cosmo.calculate_Hubble_cal()
        H_cal_dot = my_cosmo.calculate_Hubble_cal_dot()
        chi = results1.comoving_radial_distance(self.z)
        beta = 1 - 2/(chi * H_cal) - (H_cal_dot/H_cal**2)
        I = my_cosmo.calculate_I()

        result =  res * ((1 / H_cal_star) * (H_cal * G * sigma_8_1 / D1) * ((self.bias_B * sigma_8_1 + self.bias_F * sigma_8_1) - (self.bias_B * sigma_8_1 * beta + self.bias_F * sigma_8_1 * beta)))**(-1)

        delta_z = 0.01
        #G_dot_hat = - (1 / (12 * delta_z)) * (-G(self.z + 2 * delta_z) * sigma_8_1 / D1 + 8 * G(self.z + delta_z) * sigma_8_1 / D1 - 8 * G(self.z - delta_z) * sigma_8_1 / D1 + G(self.z - 2 * delta_z) * sigma_8_1 / D1)
        G_dot_hat = H_cal * my_cosmo.calculate_G_dot(0.01) * sigma_8_2 / D1_star

        res1 = result / H_cal_star * (((self.bias_B * sigma_8_1 + self.bias_F * sigma_8_1) * ((3 * H_cal * I * sigma_8_1 / D1 * 2) - (H_cal_dot * (G * sigma_8_1 / D1) + H_cal * G_dot_hat)/ H_cal) - (H_cal * G * sigma_8_1 / D1) * (self.bias_B * sigma_8_1 * beta + self.bias_F * sigma_8_1 * beta)))

        return result, res1



    def calculate_alpha_5(self):
# Set up the CAMB parameters
        pars1 = camb.CAMBparams()

        # Set the cosmological parameters
        pars1.set_cosmology(H0=self.H_0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk, tau=self.tau)
        # Set the parameter values for the initial power spectrum
        pars1.InitPower.set_params(As=self.As, ns=self.ns, r=self.r)
        # Enable transfer functions
        pars1.set_matter_power(redshifts=[self.z], kmax=2.0)

# We can choose which source terms to include: for now, we just choose density and RSD
        pars1.SourceTerms.counts_density = False
        pars1.SourceTerms.counts_redshift = True
        pars1.SourceTerms.counts_lensing = False
        pars1.SourceTerms.counts_velocity = True
        pars1.SourceTerms.counts_radial = False
        pars1.SourceTerms.counts_timedelay = False
        pars1.SourceTerms.counts_ISW = False
        pars1.SourceTerms.counts_potential = False
        pars1.SourceTerms.counts_evolve = False
        pars1.SourceTerms.use_21cm_mK = False


        # Let's double-check that we have indeed included the terms that we want
        #print(pars.SourceTerms)

        #GENERATE THE CL's


        # Set up the window functions that will be correlated, later labelled W1, W2 and so on. Here we choose two gaussian windows, with a chosen central redshift and width.
        # You can also specify the galaxy bias and, to allow for multiple populations of galaxies, you can choose two identical windows with different bias values.
        pars1.SourceWindows = [
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_F, sigma=self.sigma),
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_B, sigma=self.sigma)]
        results1 = camb.get_results(pars1)
        cls1 = results1.get_source_cls_dict(lmax = self.lmax, raw_cl = True) # raw_cl gives the C ell's without the factor 2pi/(l(l + 1)). The output corresponds to ell from 0 to lmax.

        W1W11 = cls1['W1xW1']


        pars2 = camb.CAMBparams()

        # Set the cosmological parameters
        pars2.set_cosmology(H0=self.H_0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk, tau=self.tau)
        # Set the parameter values for the initial power spectrum
        pars2.InitPower.set_params(As=self.As, ns=self.ns, r=self.r)
        # Enable transfer functions
        pars2.set_matter_power(redshifts=[self.z], kmax=2.0)

# We can choose which source terms to include: for now, we just choose density and RSD
        pars2.SourceTerms.counts_density = False
        pars2.SourceTerms.counts_redshift = True
        pars2.SourceTerms.counts_lensing = False
        pars2.SourceTerms.counts_velocity = False
        pars2.SourceTerms.counts_radial = False
        pars2.SourceTerms.counts_timedelay = False
        pars2.SourceTerms.counts_ISW = False
        pars2.SourceTerms.counts_potential = False
        pars2.SourceTerms.counts_evolve = False
        pars2.SourceTerms.use_21cm_mK = False


        # Let's double-check that we have indeed included the terms that we want
        #print(pars.SourceTerms)

        #GENERATE THE CL's


        # Set up the window functions that will be correlated, later labelled W1, W2 and so on. Here we choose two gaussian windows, with a chosen central redshift and width.
        # You can also specify the galaxy bias and, to allow for multiple populations of galaxies, you can choose two identical windows with different bias values.
        pars2.SourceWindows = [
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_F, sigma=self.sigma),
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_B, sigma=self.sigma)]
        results2 = camb.get_results(pars2)
        cls2 = results2.get_source_cls_dict(lmax = self.lmax, raw_cl = True) # raw_cl gives the C ell's without the factor 2pi/(l(l + 1)). The output corresponds to ell from 0 to lmax.

        W1W12 = cls2['W1xW1']



        pars3 = camb.CAMBparams()

        # Set the cosmological parameters
        pars3.set_cosmology(H0=self.H_0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk, tau=self.tau)
        # Set the parameter values for the initial power spectrum
        pars3.InitPower.set_params(As=self.As, ns=self.ns, r=self.r)
        # Enable transfer functions
        pars3.set_matter_power(redshifts=[self.z], kmax=2.0)

# We can choose which source terms to include: for now, we just choose density and RSD
        pars3.SourceTerms.counts_density = False
        pars3.SourceTerms.counts_redshift = False
        pars3.SourceTerms.counts_lensing = False
        pars3.SourceTerms.counts_velocity = True
        pars3.SourceTerms.counts_radial = False
        pars3.SourceTerms.counts_timedelay = False
        pars3.SourceTerms.counts_ISW = False
        pars3.SourceTerms.counts_potential = False
        pars3.SourceTerms.counts_evolve = False
        pars3.SourceTerms.use_21cm_mK = False


        # Let's double-check that we have indeed included the terms that we want
        #print(pars.SourceTerms)

        #GENERATE THE CL's


        # Set up the window functions that will be correlated, later labelled W1, W2 and so on. Here we choose two gaussian windows, with a chosen central redshift and width.
        # You can also specify the galaxy bias and, to allow for multiple populations of galaxies, you can choose two identical windows with different bias values.
        pars3.SourceWindows = [
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_F, sigma=self.sigma),
            GaussianSourceWindow(redshift=self.z, source_type='counts', bias=self.bias_B, sigma=self.sigma)]
        results3 = camb.get_results(pars3)
        cls3 = results3.get_source_cls_dict(lmax = self.lmax, raw_cl = True) # raw_cl gives the C ell's without the factor 2pi/(l(l + 1)). The output corresponds to ell from 0 to lmax.

        W1W13 = cls3['W1xW1']


        res = W1W11 - W1W12 - W1W13



        sigma_8_1 = results1.get_sigma8()
        sigma_8_2 = results2.get_sigma8()


        my_cosmo = cosmo_quantities(self.z, self.Omega_m, self.H_0, self.Omega_Lambda)
        G = my_cosmo.calculate_G()
        D1 = my_cosmo.calculate_D1()
        H_cal_star = my_cosmo.calculate_Hubble_cal_star()
        H_cal = my_cosmo.calculate_Hubble_cal()
        H_cal_dot = my_cosmo.calculate_Hubble_cal_dot()
        chi = results1.comoving_radial_distance(self.z)
        beta = 1 - 2/(chi * H_cal) - (H_cal_dot/H_cal**2)

        result = res * ((beta+beta)*(H_cal * G ** 2 * sigma_8_1 ** 2 / D1 ** 2) / H_cal_star)**(-1)

        return result, res


if __name__ == '__main__':
    
    Omega_m = 0.31
    Omega_Lambda = 0.69
    H_0 = 67.66
    ombh2=0.02236
    omch2=0.1200
    mnu=0.06
    omk=0
    tau=0.0544
    As=2.100549e-9
    ns=0.9652
    r=0
    z = 0.15
    sigma=0.02
    bias_F=1.
    bias_B=1.5
    lmax = 1000
    
    # Check if the plots target folder exists, creates one if doesn't
    plot_dir = 'plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    my_alphas = alphas(z, Omega_m, H_0, Omega_Lambda, ombh2, omch2, mnu, omk, tau, As, ns, r, sigma, bias_F, bias_B, lmax)
    alpha_1, Cell1 = my_alphas.calculate_alpha_1()
    alpha_2, Cell2 = my_alphas.calculate_alpha_2()
    alpha_3, Cell3 = my_alphas.calculate_alpha_3()
    alpha_4, Cell4 = my_alphas.calculate_alpha_4()
    alpha_5, Cell5 = my_alphas.calculate_alpha_5()
    
    sigma8 = my_alphas.sigma8()
    with open('sigma8.pkl', 'wb') as file:
        pkl.dump(sigma8, file)

    ell = np.arange(len(alpha_1))
    
    # Store alphas
    np.savez('alpha_integrals.npz', alpha_1 = alpha_1, alpha_2 = alpha_2, alpha_3 = alpha_3, alpha_4 = alpha_4, alpha_5 = alpha_5)
    
    
    # Plot alpha_i
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_1, label=r'$\alpha_1$')
    #plt.plot(-alpha_2, label=r'$\alpha_2$')
    plt.plot(alpha_3, label=r'$\alpha_3$')
    plt.plot(alpha_4, label=r'$\alpha_4$')
    plt.plot(alpha_5, label=r'$\alpha_5$')
    # Plot alpha_2 with log scale and handling negative values
    alpha_2_neg = alpha_2 < 0
    plt.plot(ell[alpha_2_neg], -alpha_2[alpha_2_neg], 'r--', label=r'$\alpha_2$ (neg)', linewidth=1)  # Dashed red line for negative values
    plt.plot(ell[~alpha_2_neg], alpha_2[~alpha_2_neg], label=r'$\alpha_2$',linewidth=1, color = 'm')  # Normal plot for non-negative values

    plt.yscale('log')
    plt.xlabel(r'$\ell$', fontsize=14)
    plt.ylabel(r'$\alpha_i$', fontsize=14)
    plt.title(r'Plot of $\alpha_i$', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/alphas_plot.pdf')
    plt.show()

    # Plot Cells
    plt.figure(figsize=(10, 6))
    plt.plot(Cell1, label=r'$C_\ell^{density}$')
    plt.plot(Cell2, label=r'$C_\ell^{RSD}$')
    plt.plot(Cell3, label=r'$C_\ell^{density-RSD}$')
    plt.plot(Cell4, label=r'$C_\ell^{density-rel}$')
    plt.plot(Cell5, label=r'$C_\ell^{RSD-rel}$')
    plt.yscale('log')
    plt.xlabel(r'$\ell$', fontsize=14)
    plt.ylabel(r'$C_\ell$', fontsize=14)
    plt.title(r'Plot of $C_\ell$', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Cells_plot.pdf')
    plt.show()


    ell = np.arange(len(Cell1))  # Assuming ell values are indices
    ell_factor = ell * (ell + 1) / (2 * np.pi)

    # List of C_ell arrays and their corresponding labels
    Cell_arrays = [Cell1, Cell2, Cell3, Cell4, Cell5]
    labels = [r'$C_\ell^{density}$', r'$C_\ell^{RSD}$', r'$C_\ell^{density-RSD}$', r'$C_\ell^{density-rel}$', r'$C_\ell^{RSD-rel}$']

    # Plot ell * (ell + 1) / (2 * pi) * C_ell for each Cell array on the same plot
    plt.figure(figsize=(10, 6))
    for i, Cell in enumerate(Cell_arrays):
        # Compute transformed values
        transformed_values = ell_factor * Cell

        # Plot transformed values
        plt.plot(transformed_values, label=f'{labels[i]}')

    plt.yscale('log')
    plt.xlabel(r'$\ell$', fontsize=14)
    plt.ylabel(r'$\ell (\ell + 1) / (2 \pi) \times C_\ell$', fontsize=14)
    plt.title(r'$\ell (\ell + 1) / (2 \pi) \times C_\ell$', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Transformed_Cells_plot_combined.pdf')
    plt.show()