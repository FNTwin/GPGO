# GPGO - Gaussian Process GO
My own implementation of a Bayesian Black box Optimization with Gaussian Process as a surrogate model.
It is still in development as I'm using it for my Master degree thesis to achieve a bottom up optimization of the Dissipative
Particle Dynamics force field for a complex system of polymers chains functionalized gold nanoparticles in a water solvent. 

# Hyperparameters
The Hyperparameters of the GP are optimized by the common technique of maximizing the Log Marginal Likelihood. In this repository this is achieved by using a search grid (although not in an efficient way) or by using the scipy optimizer module (L-BFGS-B, TNC, SLSCP).
The analytical gradient is implemented for the Radial Basis Function kernel and it is possible to use the derivate of the Log Marginal Likelihood to optimize the hyperparameters.
<a href="https://ibb.co/D8yvW3x"><img src="https://i.ibb.co/pR8MwCt/Figure-6.png" alt="Figure-6" border="0"></a>

# Acquisition function
As it is there are two different acquisition function implemented right now:

-Expected Improvement (EI)

-UCB (Upper Confidence Bound)

# Maximizing the Acquisition function 
In this little package right now there are 3 ways to run an optimization task with Gaussian Processes:

-NAIVE : AkA sampling the acquisition function with a grid of some kind or a quasi random methods as LHS (require smt package)

-BFGS : optimize the Acquisition function by using the L-BFGS-B optimizer

-DIRECT : optimize the Acquisition function by using the DIRECT optimizer (require DIRECT python package)
<a href="https://ibb.co/GPSM0cm"><img src="https://i.ibb.co/f0wN24J/Figure-7.png" alt="Figure-7" border="0"></a>

# TODO

-Tutorials and Examples

-Good code practice maybe 

-An integration with LAMMPS using the pyLammps routine





