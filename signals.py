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
    
    def Cells_DD(self, which_pop = 'default'):
        """
        Compute and return the Density-Density Cell's for the different combinations of populations at different redshifts.

        This method computes the Cell's using the bias vectors of two populations,
        calculates the outer product of each pair of bias vectors, and then applies an upper
        triangle mask to these products, removing repeated elements. 
        Finally, it scales the resulting values by the corresponding alpha integral and stores them in an output array.

        Parameters
        ----------
        which_pop : str or list of tuples, optional
            Specifies the population pairs for which to compute the Cell combinations.
            If 'default', uses all combinations with replacement of `self.pop` (default is 'default').

        Returns
        -------
        np.ndarray
            A 3D NumPy array of shape (num_combs, num_z_combinations, alpha_length),
            where:
            - `num_combs` is the number of population pairs.
            - `num_z_combinations` is the number of redshift bin combinations.
            - `alpha_length` is the length of the alpha integral (`self.alphas['alpha_1']`).

            Each element `Cell_combs[i, j, k]` represents the scaled value of the `j`-th element
            in the upper triangle of the outer product of the bias vectors for the `i`-th
            population pair, scaled by the `k`-th element of `alpha1`.

        Example
        -------
        Given:
        - self.pop = ['pop1', 'pop2']
        - self.bias = {'pop1': np.array([1, 2]), 'pop2': np.array([3, 4])}
        - self.alphas = {'alpha_1': np.array([0.5, 1.5])}
        
        Calling `Cells_DD()` will compute the combinations, outer products, and return
        the results in a 3D NumPy array.
        """
        
        if which_pop == 'default':
            which_pop = list(it.combinations_with_replacement(self.pop, 2))
        
        alpha1 = self.alphas['alpha_1']
        num_combs = len(which_pop)
        z_length = len(self.z)
        
        # Precompute the upper triangle mask
        triu_idx = np.triu_indices(z_length)
        
        # Initialize an array to hold the results
        Cell_combs = np.zeros((num_combs, len(triu_idx[0]), len(alpha1)))
        
        for i, comb in enumerate(which_pop):
            b1 = self.bias[comb[0]]
            b2 = self.bias[comb[1]]
            
            # Compute the outer product and apply the upper triangle mask
            product = np.outer(b1, b2)[triu_idx]
            
            # Compute the final result using broadcasting
            Cell_combs[i] = alpha1[np.newaxis, :] * product[:, np.newaxis]
        
        return Cell_combs
    
    def Cells_RR(self, which_pop = [['B','B']]):
        """
        Compute and return the RSD-RSD Cell's at different redshifts.

        This method computes the Cell's,
        calculates the outer product for each redshift bin, and then applies an upper
        triangle mask to these products, removing repeated elements. 
        Finally, it scales the resulting values by the corresponding alpha integral and stores them in an output array.

        Parameters
        ----------
        which_pop : str or list of tuples, optional
            Specifies the population pairs for which to compute the Cell combinations.
            If 'default', uses all combinations with replacement of `self.pop` (default is 'default').

        Returns
        -------
        np.ndarray
            A 3D NumPy array of shape (num_combs, num_z_combinations, alpha_length),
            where:
            - `num_combs` is the number of population pairs.
            - `num_z_combinations` is the number of redshift bin combinations.
            - `alpha_length` is the length of the alpha integral (`self.alphas['alpha_1']`).

            Each element `Cell_combs[i, j, k]` represents the scaled value of the `j`-th element
            in the upper triangle of the outer product of the bias vectors for the `i`-th
            population pair, scaled by the `k`-th element of `alpha1`.

        Example
        -------
        Given:
        - self.pop = ['pop1', 'pop2']
        - self.bias = {'pop1': np.array([1, 2]), 'pop2': np.array([3, 4])}
        - self.alphas = {'alpha_1': np.array([0.5, 1.5])}
        
        Calling `Cells_RR()` will compute the combinations, outer products, and return
        the results in a 3D NumPy array.
        """
        
        alpha2 = self.alphas['alpha_2']
        num_combs = len(which_pop)
        z_length = len(self.z)
        
        Ghat = self.calculate_G() * self.s8(self.z) / self.calculate_D1()
        
        # Precompute the upper triangle mask
        triu_idx = np.triu_indices(z_length)
        
        # Initialize an array to hold the results
        Cell_combs = np.zeros((num_combs, len(triu_idx[0]), len(alpha2)))
        
        for i,_ in enumerate(which_pop):

            # Compute the outer product and apply the upper triangle mask
            product = -np.outer(Ghat, Ghat)[triu_idx]
            
            # Compute the final result using broadcasting
            Cell_combs[i] = alpha2[np.newaxis, :] * product[:, np.newaxis]
        
        return Cell_combs
    
    def Cells_DR(self, which_pop='default'):
        """
        Compute and return the Density-RSD Cell's at different redshifts.

        This method computes the Cell's,
        calculates the outer product for each redshift bin, and then applies an upper
        triangle mask to these products, removing repeated elements. 
        Finally, it scales the resulting values by the corresponding alpha integral and stores them in an output array.

        Parameters
        ----------
        which_pop : str or list of tuples, optional
            Specifies the population pairs for which to compute the Cell combinations.
            If 'default', uses all combinations with replacement of `self.pop` (default is 'default').

        Returns
        -------
        np.ndarray
            A 3D NumPy array of shape (num_combs, num_z_combinations, alpha_length),
            where:
            - `num_combs` is the number of population pairs.
            - `num_z_combinations` is the number of redshift bin combinations.
            - `alpha_length` is the length of the alpha integral (`self.alphas['alpha_1']`).

            Each element `Cell_combs[i, j, k]` represents the scaled value of the `j`-th element
            in the upper triangle of the outer product of the bias vectors for the `i`-th
            population pair, scaled by the `k`-th element of `alpha1`.

        Example
        -------
        Given:
        - self.pop = ['pop1', 'pop2']
        - self.bias = {'pop1': np.array([1, 2]), 'pop2': np.array([3, 4])}
        - self.alphas = {'alpha_1': np.array([0.5, 1.5])}
        
        Calling `Cells_DR()` will compute the combinations, outer products, and return
        the results in a 3D NumPy array.
        """
        if which_pop == 'default':
            which_pop = list(it.combinations_with_replacement(self.pop, 2))
        
        alpha3 = self.alphas['alpha_3']
        Ghat = self.calculate_G() * self.s8(self.z) / self.calculate_D1()
        num_combs = len(which_pop)
        z_length = len(self.z)
        
        # Precompute the upper triangle mask
        triu_idx = np.triu_indices(z_length)
        
        # Initialize an array to hold the results
        Cell_combs = np.zeros((num_combs, len(triu_idx[0]), len(alpha3)))
        
        for i, comb in enumerate(which_pop):
            b1 = self.bias[comb[0]]
            b2 = self.bias[comb[1]]
            
            # Compute the outer product and apply the upper triangle mask
            product_RD = - np.outer(b1, Ghat)[triu_idx]
            product_DR = - np.outer(Ghat, b2)[triu_idx]
            
            product = product_RD + product_DR
            
            # Compute the final result using broadcasting
            Cell_combs[i] = alpha3[np.newaxis, :] * product[:, np.newaxis]
        
        return Cell_combs
    
    def Cells_DRel(self, which_pop='default'):
        """
        Compute and return the Density-Relativistic Cell's at different redshifts.

        This method computes the Cell's,
        calculates the outer product for each redshift bin, and then applies an upper
        triangle mask to these products, removing repeated elements. 
        Finally, it scales the resulting values by the corresponding alpha integral and stores them in an output array.

        Parameters
        ----------
        which_pop : str or list of tuples, optional
            Specifies the population pairs for which to compute the Cell combinations.
            If 'default', uses all combinations with replacement of `self.pop` (default is 'default').

        Returns
        -------
        np.ndarray
            A 3D NumPy array of shape (num_combs, num_z_combinations, alpha_length),
            where:
            - `num_combs` is the number of population pairs.
            - `num_z_combinations` is the number of redshift bin combinations.
            - `alpha_length` is the length of the alpha integral (`self.alphas['alpha_1']`).

            Each element `Cell_combs[i, j, k]` represents the scaled value of the `j`-th element
            in the upper triangle of the outer product of the bias vectors for the `i`-th
            population pair, scaled by the `k`-th element of `alpha1`.

        Example
        -------
        Given:
        - self.pop = ['pop1', 'pop2']
        - self.bias = {'pop1': np.array([1, 2]), 'pop2': np.array([3, 4])}
        - self.alphas = {'alpha_1': np.array([0.5, 1.5])}
        
        Calling `Cells_DRel()` will compute the combinations, outer products, and return
        the results in a 3D NumPy array.
        """
        if which_pop == 'default':
            which_pop = list(it.combinations_with_replacement(self.pop, 2))
        
        alpha4 = self.alphas['alpha_4']
        Ghat = self.calculate_G() * self.s8(self.z) / self.calculate_D1()
        Hz = self.calculate_Hubble_cal(self.z)
        Ghat_dot = -(1+self.z) * Hz * self.calculate_G_dot() # This is incorrect. Review the derivative_5_point_stencil method.
        Evolz = Hz / self.calculate_Hubble_cal(10.)
        Hzdot = self.calculate_Hubble_cal_dot()
        chi = self.calculate_comoving_distance()
        Ihat = self.calculate_I() * self.s8(self.z) / self.calculate_D1()
        
        num_combs = len(which_pop)
        z_length = len(self.z)
        
        # Precompute the upper triangle mask
        triu_idx = np.triu_indices(z_length)
        
        # Initialize an array to hold the results
        Cell_combs = np.zeros((num_combs, len(triu_idx[0]), len(alpha4)))
        
        for i, comb in enumerate(which_pop):
            b1 = self.bias[comb[0]]
            b2 = self.bias[comb[1]]
            
            # Neglect magnification and evolution biases
            beta1 = 1 - 2/(chi * Hz) - Hzdot/(Hz**2)
            beta2 = 1 - 2/(chi * Hz) - Hzdot/(Hz**2)
            
            # Compute the outer product and apply the upper triangle mask
            productDI = np.outer(b1, 3/2 * Evolz * Ihat)[triu_idx]
            productDG = - np.outer(b1, Evolz * beta2 * Ghat)[triu_idx]
            productDGdot = - np.outer(b1, (Hzdot/Hz) * Ghat + Ghat_dot)[triu_idx]
            
            productID = np.outer(3/2 * Evolz * Ihat, b2)[triu_idx]
            productGD = - np.outer(Evolz * beta1 * Ghat, b2)[triu_idx]
            productGdotD = - np.outer((Hzdot/Hz) * Ghat + Ghat_dot, b2)[triu_idx]
            
            productDRel = productDI + productDG + productDGdot
            productRelD = productID + productGD + productGdotD
            
            product = productDRel + productRelD
            # Compute the final result using broadcasting
            Cell_combs[i] = alpha4[np.newaxis, :] * product[:, np.newaxis]
        
        return Cell_combs

    def Cells_RRel(self, which_pop='default'):
        """
        Compute and return the RSD-Relativistic Cell's at different redshifts.

        This method computes the Cell's,
        calculates the outer product for each redshift bin, and then applies an upper
        triangle mask to these products, removing repeated elements. 
        Finally, it scales the resulting values by the corresponding alpha integral and stores them in an output array.

        Parameters
        ----------
        which_pop : str or list of tuples, optional
            Specifies the population pairs for which to compute the Cell combinations.
            If 'default', uses all combinations with replacement of `self.pop` (default is 'default').

        Returns
        -------
        np.ndarray
            A 3D NumPy array of shape (num_combs, num_z_combinations, alpha_length),
            where:
            - `num_combs` is the number of population pairs.
            - `num_z_combinations` is the number of redshift bin combinations.
            - `alpha_length` is the length of the alpha integral (`self.alphas['alpha_1']`).

            Each element `Cell_combs[i, j, k]` represents the scaled value of the `j`-th element
            in the upper triangle of the outer product of the bias vectors for the `i`-th
            population pair, scaled by the `k`-th element of `alpha1`.

        Example
        -------
        Given:
        - self.pop = ['pop1', 'pop2']
        - self.bias = {'pop1': np.array([1, 2]), 'pop2': np.array([3, 4])}
        - self.alphas = {'alpha_1': np.array([0.5, 1.5])}
        
        Calling `Cells_RRel()` will compute the combinations, outer products, and return
        the results in a 3D NumPy array.
        """
        if which_pop == 'default':
            which_pop = list(it.combinations_with_replacement(self.pop, 2))
        
        alpha5 = self.alphas['alpha_5']
        Ghat = self.calculate_G() * self.s8(self.z) / self.calculate_D1()
        Hz = self.calculate_Hubble_cal(self.z)
        Evolz = Hz / self.calculate_Hubble_cal(10.)
        Hzdot = self.calculate_Hubble_cal_dot()
        chi = self.calculate_comoving_distance()
   
        
        num_combs = len(which_pop)
        z_length = len(self.z)
        
        # Precompute the upper triangle mask
        triu_idx = np.triu_indices(z_length)
        
        # Initialize an array to hold the results
        Cell_combs = np.zeros((num_combs, len(triu_idx[0]), len(alpha5)))
        
        for i, comb in enumerate(which_pop):
            # Neglect magnification and evolution biases
            beta1 = 1 - 2/(chi * Hz) - Hzdot/(Hz**2)
            beta2 = 1 - 2/(chi * Hz) - Hzdot/(Hz**2)
            
            # Compute the outer product and apply the upper triangle mask
            productRRel = np.outer(beta1 * Ghat, Evolz * Ghat)[triu_idx]
            productRelR = np.outer(Evolz * Ghat, beta2 * Ghat)[triu_idx]

            product = productRRel + productRelR
            
            # Compute the final result using broadcasting
            Cell_combs[i] = alpha5[np.newaxis, :] * product[:, np.newaxis]
        
        return Cell_combs