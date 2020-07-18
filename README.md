# Bayesian-Optimization

My own implementation of Bayesian Black box Optimization with Gaussian Process as a surrogate model.
It is still in development as I'm using it for my Master degree thesis to achieve a bottom up optimization of the Dissipative
Particle Dynamics force field for a complex system of polymers functionalized gold nanoparticles in water solvent. 

# What it can do
Well, it can do Gaussian Process and Optimization on it without relying on the usual sklearn GaussianProcessRegressor package.
Right now it even has an optimization routine for the hyperparameters!

# Maximizing the Acquisition function (EI only for now)
In this little package right now there are 3 ways to run an optimization task with Gaussian Processes:

-NAIVE : AkA sampling the acquisition function with a grid of some kind or a quasi random methods as LHS

-BFGS : Find the Maxima of the Acquisition function by using the L-BFGS-B optimizer

-DIRECT : Find the Maxiam of the Acquisition function by using the DIRECT optimizer


