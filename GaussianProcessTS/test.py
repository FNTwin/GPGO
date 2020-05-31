import time

import matplotlib.pyplot as plt
import numpy as np
from GaussianProcess.GP import GP, generate_grid
from GaussianProcess.Kernel.RBF import RBF
from Opt import BayesianOptimization


def test_GP_1D(optimize=False):
    x = np.array([-4, -3, -2, -1, 1])[:, None]

    def f(X):
        return np.sin(X)

    def noise(x, alpha=1):
        return f(x) + np.random.uniform(-1, 1, size=x.shape) * alpha

    y = noise(x, alpha=0)

    gp = GP(x, y, noise=0.0000005, kernel=RBF(sigma_l=1.0, l=1))
    gp.fit()

    plot = np.linspace(-5, 5, 1000)

    pred_old, var_old = gp.predict(plot[:, None])

    gp.plot(plot[:, None])

    gp.static_compute_marg()
    print("Old marg likelihood :", gp.get_marg(), "\n Hyperparameters: ",
          gp.get_kernel().gethyper())
    if optimize:
        new = gp.grid_search_optimization(constrains=[[1, 30], [1, 30]],
                                          n_points=300,
                                          function=np.random.uniform)

        optimized = GP(x, y, noise=0.1, kernel=RBF(sigma_l=new["hyper"][0], l=new["hyper"][1]))
        optimized.fit()
        pred, var = optimized.predict(plot[:, None])

        optimized.plot(plot[:, None])
        optimized.static_compute_marg()
        print("New marg likelihood :", optimized.get_marg(),
              "\n Hyperparameters: ", optimized.get_kernel().gethyper())


def test_GP_2D(optimize=True, function=np.linspace):
    dim_test = 2
    dim_out = 1
    n_train_p = 7
    X = np.random.uniform(-2, 2, (40, 2))
    Z = ((X[:, 1] ** 2 * X[:, 0] ** 2) * np.sin((X[:, 1] ** 2 + X[:, 0] ** 2)))[:, None]
    gp = GP(X, Z, kernel=RBF(), noise=0.2)
    gp.fit()
    plot = generate_grid(dim_test, 30, [[-3, 3] for i in range(dim_test)])

    pred = gp.predict(plot)
    gp.plot(plot)
    # gp.static_compute_marg()
    print("Old marg likelihood :", gp.get_marg(),
          "\n Hyperparameters: ", gp.get_kernel().gethyper())

    if optimize:
        gp.optimize(n_points=100, function=function)
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
    gp = GP(x, y, noise=0.01, kernel=RBF(sigma_l=2, l=2))
    gp.fit()
    mean, var = gp.predict(plot)
    print("Old marg likelihood :", gp.get_marg(),
          "\n Hyperparameters: ", gp.get_kernel().gethyper())

    if optimize:
        gp.optimize(constrains=[[1, 3], [2, 100]], n_points=100, function=np.random.uniform)
        mean, var = gp.predict(plot)
        print("New marg likelihood :", gp.get_marg(),
              "\n Hyperparameters: ", gp.get_kernel().gethyper())

    return mean, var


def test_minimization_2D():
    dim_test = 2
    dim_out = 1
    n_train_p = 3
    X = np.array([[-0.9, 0.3]])
    boundaries = [[-5, 10], [0, 15]]

    def f(x):
        x1, x2 = x[:, 0], x[:, 1]
        return (1 * (x2 - (5.1 / (4 * np.pi ** 2)) * x1 ** 2 + 5 / np.pi *
                     x1 - 6) ** 2 + 10 * (1 - (1 / (8 * np.pi))) * np.cos(x1) + 10)

    Z = f(X)[:, None]
    gp = GP( X, Z, noise=0.01 )
    gp.fit()
    BayOpt = BayesianOptimization( X, Z, gp, f )
    # best=BayOpt.bayesian_run(100,  [[-1,4] for i in range(dim_test)] , iteration=30, optimization=False)
    best = BayOpt.bayesian_run_DIRECT(200,
                                   boundaries,
                                   iteration=100,
                                   optimization=False,
                                   epsilon=0.01,
                                   func=np.random.uniform)

    print("bay:", best)

    # plot = generate_grid(dim_test, 30, [[-5, 5] for i in range(dim_test)])
    plot = generate_grid(dim_test, 30, boundaries, np.linspace)
    ax = plt.axes(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Z, color='red', marker="x")
    ax.scatter(plot[:, 0], plot[:, 1], f(plot))
    ax.scatter(best[0][0], best[0][1], best[1], color="red")
    ax.scatter(gp.get_X()[:, 0], gp.get_X()[:, 1], gp.get_Y(), marker="x", color="black")
    plt.show()


def test_minimization_1D():
    dim_test = 1
    dim_out = 1
    n_train_p = 3

    X = np.array([[0.1]])

    def f(X):
        return (6* X - 2)**2 * np.sin (12 * X - 4)

    Z = f(X)

    gp = GP(X, Z, noise=0.00001)
    gp.fit()
    BayOpt = BayesianOptimization(X, Z, gp, f)
    # best=BayOpt.bayesian_run(100,  [[-1,4] for i in range(dim_test)] , iteration=30, optimization=False)
    best = BayOpt.bayesian_run_min( 250,
                                    [[0,1]],
                                    iteration=10,
                                    optimization=False,
                                    opt_constrain=[[1, 20], [2, 20]],
                                    epsilon=0.1,
                                    func=np.linspace,
                                    plot=True)

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

    gp = GP(x, y, noise=0.005)
    gp.fit()
    BayOpt = BayesianOptimization(x, y, gp, hartmann_6D)

    n_p = 10

    best = BayOpt.bayesian_run_min(n_p,
                                   [[0, 1] for i in range(6)],
                                   iteration=100,
                                   optimization=False,
                                   epsilon=0.01,
                                   func=np.random.uniform)

    print("Number of points sampled in an iteration: ", n_p ** dim)
    print("bay:", best)


def test_GP_print():
    dim_test = 2
    dim_out = 1
    n_train_p = 7
    X = np.random.uniform(-2, 2, (40, 2))
    Z = ((X[:, 1] ** 2 * X[:, 0] ** 2) * np.sin((X[:, 1] ** 2 + X[:, 0] ** 2)))[:, None]
    gp = GP(X, Z, noise=0.2)
    gp.fit()
    plot = generate_grid(dim_test, 30, [[-3, 3] for i in range(dim_test)])
    pred = gp.predict(plot)
    print(gp)


a = time.time()
test_Hartmann_6D()
print("Finished: ", time.time() - a)


