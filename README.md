The code uses:
CAMB
numpy
matplotlib
scipy

Code structure (Currently in development):

- class_cosmo_quantities_old.py: contains a class that computes cosmological quantities.

- alphas_calculation.py: computes the alpha's integrals using CAMB (Note: update code to have correlations between redshift bins). 

- signals.py: computes the signals taking into account, Density (D), RSD (R) and relativistic contributions (Rel). Computes all the cross correlations for two populations of galaxies (Bright (B) and Faint (F)).

- derivatives.py: computes the derivatives (Note: analytic expressions).

- covariance.py: computes the Covariance Matrix. 


 