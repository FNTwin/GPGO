import numpy as np
from GPGO import GP
from GPGO import RBF
from GPGO import BayesianOptimization

x=np.random.uniform(0,10,3)[:,None]
y=x*3
a=GP(x,y,RBF())
print(a)
