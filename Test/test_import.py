import numpy as np
from BayesianOptimization import GP
from BayesianOptimization import RBF
from BayesianOptimization import BayesianOptimization

x=np.random.uniform(0,10,3)[:,None]
y=x*3
a=GP(x,y,RBF())
print(a)
