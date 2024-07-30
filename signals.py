import numpy as np
import pickle as pkl
import itertools as it
from class_cosmo_quantities_old import cosmo_quantities

class Signals(cosmo_quantities):
    
    def __init__(self, z, Omega_m, H_0, Omega_Lambda, bias_B, bias_F, lmax, pop = ['B', 'F']):
        
        # Initialize the parent class (cosmo_quantities)
        super().__init__(z, Omega_m, H_0, Omega_Lambda)
        
        # Initialize Signals class attributes
        self.bias_B = np.array(bias_B)
        self.bias_F = np.array(bias_F)
        self.lmax = lmax
        self.pop = pop
        
        # Load the alphas. Call them with self.alphas[alpha_i]
        self.alphas = np.load('alpha_integrals.npz')
        
        # Load sigma8 function
        with open('sigma8.pkl', 'rb') as file:
            self.s8 = pkl.load(file)
            
        self.bias = {
            'B' : self.bias_B * self.s8(self.z), 
            'F' : self.bias_F * self.s8(self.z)
            }
        
    def Cells_new(self):
        pass
    
    def Cell1_new(self):

        b1 = self.bias_B * self.s8(self.z)
        b2 = self.bias_F * self.s8(self.z)
        
        product = np.outer(b1, b2)
        
        # Create a mask for the upper triangle including the diagonal
        mask = np.triu(np.ones_like(product), k=0).astype(bool)
        factor_z = product[mask]

        alpha1 = self.alphas['alpha_1']
        
        # Scale factor_z by alpha1 and create the final result
        result = alpha1[np.newaxis,:] * factor_z[:,np.newaxis]
        
        return result

    def Cell1_new_alt(self, which_pop = 'default'):
        
        Cell_combs = []
        if which_pop == 'default':
            which_pop = list(it.combinations_with_replacement(self.pop,2))
        else:
            which_pop = self.pop
        
        for i, comb in enumerate(which_pop):
            b1 = self.bias[comb[0]]
            b2 = self.bias[comb[1]]
        
            product = np.outer(b1,b2).flatten()
            # Create a mask for the upper triangle including the diagonal
            mask = np.triu(np.ones_like(product), k=0).astype(bool)
            factor_z = product[mask]
            
            alpha1 = self.alphas['alpha_1']
            
            Cell_combs += [alpha1[np.newaxis,:] * factor_z[:,np.newaxis]]
        
        return  Cell_combs