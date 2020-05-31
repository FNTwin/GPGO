import time

import matplotlib.pyplot as plt
import numpy as np
from GaussianProcess.GP import GP, generate_grid
from GaussianProcess.Kernel.RBF import RBF
from Opt import BayesianOptimization

def min_2D():
    dim_test = 2
    dim_out = 1
    n_train_p = 10
    X = np.random.uniform(0,10,(10,2))
    boundaries = [[-5, 10], [0, 15]]

    def f(x):
        x1, x2 = x[:, 0], x[:, 1]
        return (1 * (x2 - (5.1 / (4 * np.pi ** 2)) * x1 ** 2 + 5 / np.pi *
                     x1 - 6) ** 2 + 10 * (1 - (1 / (8 * np.pi))) * np.cos(x1) + 10)

    Z = f(X)[:, None]
    gp = GP(X, Z, noise=0.01)
    gp.fit()
    BayOpt = BayesianOptimization(X, Z, gp, f, err=1e-2)
    # best=BayOpt.bayesian_run(100,  [[-1,4] for i in range(dim_test)] , iteration=30, optimization=False)
    err = BayOpt.bayesian_run_min(200,
                                   boundaries,
                                   iteration=100,
                                   optimization=False,
                                   epsilon=0.01,
                                   func=np.random.uniform)
    print(err)
    plt.plot(np.arange(1,11,1), err)

def min_6D():
    dim = 6
    points = 10
    x = np.random.uniform(0, 1, (10, 6))

    def f(x):
        alpha = np.array([[1.], [1.2], [3.], [3.2]])

        A = np.array([[10, 3, 17, 3.50, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])

        P = 10 ** -4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                 [2329, 4135, 8307, 3736, 1004, 9991],
                                 [2348, 1451, 3522, 2883, 3047, 6650],
                                 [4047, 8828, 8732, 5743, 1091, 381]])

        def comp(i):
            tot = 0
            for j in range(6):
                tot += A[i][j] * (x.T[j] - P[i][j]) ** 2
            return np.exp(-tot)

        f = 0
        for i in range(4):
            f += -(alpha[i] * comp(i))

        return f[:, None]

    y = f(x)

    gp = GP(x, y, noise=0.01)
    gp.fit()
    BayOpt = BayesianOptimization(x,y, gp, f, err=1e-2)
    # best=BayOpt.bayesian_run(100,  [[-1,4] for i in range(dim_test)] , iteration=30, optimization=False)
    err = BayOpt.bayesian_run_DIRECT(10,
                          [[0, 1] for i in range(6)],
                          iteration=100,
                          optimization=False,
                          epsilon=0.01,
                          func=np.random.uniform)
    print("BEST", err)

def one_run_test():

    x=np.array([[-2],[2],[-3],[1]])
    gp = GP(x, np.array([[2],[3],[2.3],[0.5]]), noise=0.005)
    gp.fit()
    BayOpt = BayesianOptimization(x,  np.array([[-2],[3],[-2.3],[0.5]]), gp, err=1e-3)
    gp.plot(np.linspace(-3,3,1000)[:,None])


    print(BayOpt.bayesian_run_min(200,
                        [[-3,3]],
                        optimization=False,
                        minimization=True,
                        epsilon=0.1,
                        opt_constrain=[[2, 30], [2, 30]],
                        n_opt_points=100,
                        func=np.random.uniform))
min_6D()