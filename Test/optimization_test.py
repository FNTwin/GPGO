import time

import matplotlib.pyplot as plt
import numpy as np
from .GaussianProcess.GP import GP, generate_grid
from .GaussianProcess.Kernel.RBF import RBF
from .Opt import BayesianOptimization


def min_2D():
    dim_test = 2
    dim_out = 1
    n_train_p = 10
    X = np.random.uniform(0,10,(1,2))
    boundaries = [[-5, 10], [0, 15]]

    def f(x):
        x1, x2 = x[:, 0], x[:, 1]
        return (1 * (x2 - (5.1 / (4 * np.pi ** 2)) * x1 ** 2 + 5 / np.pi *
                     x1 - 6) ** 2 + 10 * (1 - (1 / (8 * np.pi))) * np.cos(x1) + 10)

    Z = f(X)[:, None]
    """gp = GP(X, Z, noise=0.01)
    gp.fit()
    BayOpt = GPGO(X, Z, gp, f, err=1e-2)
    # best=BayOpt.bayesian_run(100,  [[-1,4] for i in range(dim_test)] , iteration=30, optimization=False)
    err = BayOpt.bayesian_run_min(100,
                                   boundaries,
                                   #minima=0.397887,
                                   iteration=50,
                                   optimization=False,
                                   epsilon=0.01,
                                   func=np.random.uniform)"""

    BayesianOptimization.test_long(x=X,y=Z,f=f, n_search_points=300, boundaries=boundaries, iter=50,minima= 0.397887)



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

    gp = GP(x, y, noise=0.0002, kernel=RBF(sigma_l=0.7,l=0.52))
    gp.fit()
    BayOpt = BayesianOptimization(x,y, gp, f, err=1e-4)
    best=BayOpt.bayesian_run_BFGL(n_search=10,
                                  iteration=80,
                                  boundaries=[[0,1] for i in range(6)],
                                  minimization=True)


    print(best)


    # best=BayOpt.bayesian_run(100,  [[-1,4] for i in range(dim_test)] , iteration=30, optimization=False)
    """err = BayOpt.direct_test(10,
                          [[0, 1] for i in range(6)],
                          minima=-3.32237,
                          iteration=100,
                          optimization=False,
                          epsilon=0.01,
                          func=np.random.uniform)"""
    #print("BEST", err)
    #GPGO.test_long(x=x, y=y, f=f, n_search_points=6, boundaries=[[0, 1] for i in range(6)]
                                 #  , iter=100, minima=-3.32237, opt=False)

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
                                  sampling=np.random.uniform))

def test_minimization_1D():
    dim_test = 1
    dim_out = 1
    n_train_p = 3

    np.random.seed(1)

    X = np.random.uniform(0,1,3)[:,None]

    def f(X):
        return (6* X - 2)**2 * np.sin (12 * X - 4)

    Z = f(X)

    BayesianOptimization.test_long(x=X, y=Z, f=f, n_search_points=6, boundaries=[[0, 1]]
                                   , iter=10, minima=-3.32237, opt=False)
    """gp = GP(X, Z, noise=0.0005)
    gp.fit()
    BayOpt = GPGO(X, Z, gp, f)
    #gp.optimize(constrains=[[2,100],[2,100]],n_points=150)

    # best=BayOpt.bayesian_run(100,  [[-1,4] for i in range(dim_test)] , iteration=30, optimization=False)
    best = BayOpt.bayesian_run_min(1000,
                                    [[0,1]],
                                    iteration=10,
                                    optimization=True,
                                    opt_constrain=[[0.1, 20], [0.1, 20]],
                                    epsilon=0.1,
                                    func=np.linspace,
                                    plot=True)

    #print("bay:", best)
    """
def test_GP_2D(optimize=True, function=np.linspace):
    dim_test = 2
    dim_out = 1
    n_train_p = 7
    #X = np.random.uniform(-2, 2, (40, 2))
    #Z = ((X[:, 1] ** 2 * X[:, 0] ** 2) * np.sin((X[:, 1] ** 2 + X[:, 0] ** 2)))[:, None]
    data=np.loadtxt("/Users/gab/Desktop/data_test_reg.txt")

    gp = GP((data[:,0:2]-np.mean(data[:,0:2]))/np.std(data[:,0:2]), data[:,2][:,None], kernel=RBF(2,2), noise=0.0002)
    gp.fit()
    plot = generate_grid(dim_test, 100, [[-3, 3] for i in range(dim_test)])

    gp.plot(plot)
    # gp.static_compute_marg()
    print("Old marg likelihood :", gp.get_marg(),
          "\n Hyperparameters: ", gp.get_kernel().gethyper())

    if optimize:
        gp.optimize_grid(constrains=[[0.1, 20], [0.5, 30]], n_points=200, function=function)
        #pred = gp.predict(plot)
        gp.plot(plot)
        print("New marg likelihood :", gp.get_marg(),
              "\n Hyperparameters: ", gp.get_kernel().gethyper())

def test_GP_1D(optimize=False):
    x = np.array([ -1,0.5, 1, 3])[:, None]

    def f(X):
        return np.sin(X)

    def noise(x, alpha=1):
        return f(x) + np.random.uniform(-1, 1, size=x.shape) * alpha

    y = noise(x, alpha=0)

    gp = GP(x, y, noise=0.0002, kernel=RBF(sigma_l=0.5135, l=1.26))
    gp.fit()

    plot = np.linspace(-1.5, 3.5, 1000)

    pred_old, var_old = gp.predict(plot[:, None])
    #plt.plot(plot[:,None],f(plot), color="saddlebrown", linestyle="-.", label="True Function")
    gp.plot(plot[:, None])


    gp.log_marginal_likelihood()
    print("Old marg likelihood :", gp.get_marg(), "\n Hyperparameters: ",
          gp.get_kernel().gethyper())
    if optimize:
        new = gp.grid_search_optimization(constrains=[[0.2, 3], [0.2, 3]],
                                          n_points=500,
                                          function=np.random.uniform)

        optimized = GP(x, y, noise=0.000005, kernel=RBF(sigma_l=new["hyper"][0], l=new["hyper"][1]))
        optimized.fit()
        pred, var = optimized.predict(plot[:, None])

        optimized.plot(plot[:, None])
        optimized.log_marginal_likelihood()
        print("New marg likelihood :", optimized.get_marg(),
              "\n Hyperparameters: ", optimized.get_kernel().gethyper())


min_6D()