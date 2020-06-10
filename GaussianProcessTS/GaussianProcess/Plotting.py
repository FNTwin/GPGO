import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def plot_GP_1D(X_train, Y_train, lin, mean, var):
    plt.scatter(X_train, Y_train,
                color="red", marker="x", label="Train")

    plt.plot(lin, mean,
             color="black", linestyle="-", label="GP")

    plt.plot(lin, mean + 2 * var,
             color="Navy")

    plt.plot(lin, mean - 2 * var,
             color="Navy")

    plt.fill_between(lin.ravel(), (mean + 2 * var).ravel(), (mean - 2 * var).ravel(),
                     alpha=0.2, color="Navy", label="confidence")
    return plt


def plot_GP_2D(X_train, Y_train, grid, mean, var):
    ax = plt.axes(projection='3d')

    ax.scatter(X_train[:, 0], X_train[:, 1], Y_train,
               marker=".", s=1, color="red", label="Train")

    ax.scatter(grid[:, 0], grid[:, 1], mean,
               s=0.6, color="black", label="GP")

    ax.scatter(grid[:, 0], grid[:, 1], mean + 2 * var,
               s=0.5, alpha=0.3, color="Navy", label="Confidence")

    ax.scatter(grid[:, 0], grid[:, 1], mean - 2 * var,
               s=0.5, alpha=0.3, color="Navy")
    return plt


def plot_improvement(X_train, Y_train, grid, mean, var, improvement):
    fig, ax = plt.subplots(nrows=2, ncols=1)

    ax[0].scatter(X_train, Y_train,
                  marker="x", color="red", label="Train")

    ax[0].plot(grid, mean,
               linestyle="--", color="black", label="GP")

    ax[0].plot(grid, mean + 2 * var,
               alpha=0.3, color="navy")

    ax[0].plot(grid, mean - 2 * var,
               alpha=0.3, color="navy")

    ax[0].fill_between(grid.ravel(), (mean + 2 * var).ravel(), (mean - 2 * var).ravel(),
                       alpha=0.2, color="navy", label="confidence")


    ax[1].plot(grid, improvement,
               alpha=0.3, color="orange")
    ax[1].fill_between(grid.ravel(), improvement.ravel(), np.full(grid.ravel().shape, 0),
                       color="orange", alpha=0.2, label="EI")

    return plt


def plot_BayOpt(X_train, Y_train, grid, mean, var, improvement=None):
    dim = X_train[0].shape[0]

    if improvement is None:
        args = [X_train, Y_train, grid, mean, var]

        if dim == 1:
            return plot_GP_1D(*args)
        elif dim == 2:
            return plot_GP_2D(*args)
        else:
            raise ValueError("Cannot plot more than 2 dimension")

    else:
        args = [X_train, Y_train, grid, mean, var, improvement]

        if dim == 1:
            return plot_improvement(*args)
        elif dim == 2:
            raise ValueError("TODO")
        else:
            raise ValueError("Cannot plot more than 2 dimension")
