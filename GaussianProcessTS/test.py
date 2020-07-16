import time
import matplotlib.pyplot as plt
import numpy as np
from GaussianProcess.GP import GP, generate_grid
from GaussianProcess.Kernel.RBF import RBF
from GaussianProcess.Kernel.Matern import Matern
from Opt import BayesianOptimization


def test_GP_1D(optimize=True):
    #x =  np.arange(-3, 5, 1)[:, None]
    x=np.random.uniform(0,1,300)[:,None]

    def f(X):
        #return np.sin(X)
        return (6 * X - 2) ** 2 * np.sin(12 * X - 4)

    def noise(x, alpha=1):
        return f(x) + np.random.randn(*x.shape) * alpha

    #y = noise(x, alpha=2)
    x = np.random.uniform(-5, 5, (300,1))
    y = np.array([0.4 * (a*a)*np.sin(a-4) + (.6 * np.random.randn()) for a in x])

    epsilon = 1  # Energy minimum
    sigma = 1  # Distance to zero crossing point
    x=np.random.uniform(0.8,3, 30)[:,None]
    y = 4 * epsilon * ((sigma / x) ** 12 - (sigma / x) ** 6)

    gp = GP(x, y, kernel=RBF(sigma_l=0.2, l= 1, noise= 1e-2, gradient=True), normalize_y=True)
    gp.fit()

    plot = np.linspace(0.1,3.5, 1000)

    pred_old, var_old = gp.predict(plot[:, None])

    #gp.plot(plot[:, None])
    gp.static_compute_marg()
    print("Old marg likelihood :", gp.get_marg(), "\n Hyperparameters: ",
          gp.get_kernel().gethyper())
    if optimize:
        """new = gp.grid_search_optimization(constrains=[[1, 30], [1, 30],[0.00001,1]],
                                          n_points=100,
                                          function=np.linspace)"""


        #gp.opt(n_restarts=30)
        gp.opt(n_restarts=100,verbose=True)
        #optimized.fit()
        pred, var = gp.predict(plot[:, None])

        gp.plot(plot[:, None])
        gp.static_compute_marg()
        print("New marg likelihood :", gp.get_marg(),
              "\n Hyperparameters: ", gp.get_kernel().gethyper())


def test_GP_2D(optimize=True, function=np.linspace):
    dim_test = 2
    dim_out = 1
    n_train_p = 7
    X = np.random.uniform(-2, 2, (40, 2))
    Z = ((X[:, 1] ** 2 * X[:, 0] ** 2) * np.sin((X[:, 1] ** 2 + X[:, 0] ** 2)))[:, None]
    gp = GP(X, Z, kernel=RBF())
    gp.fit()
    plot = generate_grid(dim_test, 30, [[-2, 2] for i in range(dim_test)])

    pred = gp.predict(plot)
    #gp.plot(plot)
    # gp.static_compute_marg()
    print("Old marg likelihood :", gp.get_marg(),
          "\n Hyperparameters: ", gp.get_kernel().gethyper())

    if optimize:
        gp.opt(n_restarts=50)
        pred = gp.predict(plot)
        gp.plot(plot)
        print("New marg likelihood :", gp.get_marg(),
              "\n Hyperparameters: ", gp.get_kernel().gethyper())


def test_GP_4D(optimize=False):
    x = generate_grid(4, 3, [[-2, 2] for i in range(4)], np.random.uniform)


    def f(x):
        return x[:, 1] ** 2 - x[:, 3] * x[:, 0]

    y = f(x)[:, None]
    plot = generate_grid(4, 5, [[-2, 2] for i in range(4)], np.linspace)
    gp = GP(x, y,  kernel=RBF(sigma_l=2, l=2))
    gp.fit()
    mean, var = gp.predict(plot)
    print("Old marg likelihood :", gp.get_marg(),
          "\n Hyperparameters: ", gp.get_kernel().gethyper())

    if optimize:
        gp.optimize(constrains=[[1, 3], [2, 100],[0,30]], n_points=100, function=np.random.uniform)
        mean, var = gp.predict(plot)
        print("New marg likelihood :", gp.get_marg(),
              "\n Hyperparameters: ", gp.get_kernel().gethyper())

    return mean, var


def test_minimization_2D():
    dim_test = 2
    dim_out = 1
    n_train_p = 3
    X = np.array([[-0.9, 0.3],[0.5,.5]])
    boundaries = [[-5, 10], [0, 15]]

    def f(x):
        x1, x2 = x[:, 0], x[:, 1]
        return (1 * (x2 - (5.1 / (4 * np.pi ** 2)) * x1 ** 2 + 5 / np.pi *
                     x1 - 6) ** 2 + 10 * (1 - (1 / (8 * np.pi))) * np.cos(x1) + 10)

    Z = f(X)[:, None]
    gp = GP( X, Z, RBF(gradient=True))
    gp.fit()
    settings = {"type": "BFGL",
                "n_search": 10,
                "boundaries": boundaries,
                "epsilon": 0.01,
                "iteration": 50,
                "minimization": True,
                "optimization": True,
                "n_restart": 10,
                "sampling": np.random.uniform}

    BayOpt = BayesianOptimization(X,Z, settings, gp, f)
    best=BayOpt.run()

    print("bay:", best)

    # plot = generate_grid(dim_test, 30, [[-5, 5] for i in range(dim_test)])
    """plot = generate_grid(dim_test, 30, boundaries, np.linspace)
    ax = plt.axes(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Z, color='red', marker="x")
    ax.scatter(plot[:, 0], plot[:, 1], f(plot))
    ax.scatter(best[0][0], best[0][1], best[1], color="red")
    ax.scatter(gp.get_X()[:, 0], gp.get_X()[:, 1], gp.get_Y(), marker="x", color="black")
    plt.show()"""


def test_minimization_1D():
    dim_test = 1
    dim_out = 1
    n_train_p = 3

    X = np.array([[1.3],[2.1]])


    def f(X):
        #return (6* X - 2)**2 * np.sin (12 * X - 4)
        return 4 * 100 * ((.9 / X) ** 12 - (.9 / X) ** 6)

    Z = f(X)

    gp = GP(X, Z, RBF(gradient=True), normalize_y=True)
    settings={"type":"DIRECT",
              "n_search": 10,
              "boundaries": [[0.85,3]],
              "epsilon": 0.01,
              "iteration": 10,
              "minimization":True,
              "optimization":True,
              "n_restart":10,
              "sampling":np.linspace}

    BayOpt = BayesianOptimization(X, Z, settings, gp, f)
    best=BayOpt.run()
    # best=BayOpt.bayesian_run(100,  [[-1,4] for i in range(dim_test)] , iteration=30, optimization=False)
    """best = BayOpt.bayesian_run_min(250,
                                   [[0,1]],
                                   iteration=30,
                                   optimization=False,
                                   plot=False,
                                   n_restart=10,
                                   epsilon=0.01,
                                   sampling=np.linspace)"""

    print("bay:", best)


def test_Hartmann_6D():
    dim = 6
    points = 10
    x = np.random.uniform(0, 1, (10, 6))

    def hartmann_6D(x):
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

    y = hartmann_6D(x)

    gp = GP(x, y, RBF(gradient=False))
    gp.fit()

    settings = {"type": "BFGL",
                "n_search": 10,
                "boundaries": [[0, 1] for i in range(6)],
                "epsilon": 0.01,
                "iteration": 50,
                "minimization": True,
                "optimization": True,
                "n_restart": 10,
                "sampling": np.random.uniform}

    BayOpt = BayesianOptimization(x, y, settings, gp, hartmann_6D)

    n_p = 10

    best = BayOpt.run()

    print("Number of points sampled in an iteration: ", n_p ** dim)
    print("bay:", best)


def test_GP_print():
    dim_test = 2
    dim_out = 1
    n_train_p = 7
    X = np.random.uniform(-2, 2, (40, 2))
    Z = ((X[:, 1] ** 2 * X[:, 0] ** 2) * np.sin((X[:, 1] ** 2 + X[:, 0] ** 2)))[:, None]
    gp = GP(X, Z, RBF())
    gp.get_kernel().plot()
    gp.fit()
    plot = generate_grid(dim_test, 5, [[-3, 3] for i in range(dim_test)])
    gp.optimize(n_points=5, verbose=True)
    gp.get_kernel().plot()
    pred = gp.predict(plot)
    print(gp)
    gp.save_model("/home/merk/Desktop/GP.txt")


a = time.time()
test_minimization_1D()
print("Finished: ", time.time() - a)


