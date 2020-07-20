import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.font_manager import FontProperties


def plot_GP_1D(X_train, Y_train, lin, mean, var):
    """Function to plot a GP 1-D Object"""

    plt.fill_between(lin.ravel(), (mean + 2 * var).ravel(), (mean - 2 * var).ravel(),
                     alpha=1, color="lavender", label="GP Variance")

    plt.scatter(X_train, Y_train,
                color="red", marker="x", label="Train")

    plt.plot(lin, mean,
             color="black",  label="GP Mean")

    plt.xlabel("x value")
    plt.ylabel("y value")
    plt.title("Gaussian Process Regression", pad=28)
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
    plt.tight_layout()
    return plt


def plot_GP_2D(X_train, Y_train, grid, mean, var):
    n=int(np.sqrt(grid.shape[0]))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    cp=ax[0].contourf(grid[:, 0].reshape(n,n), grid[:, 1].reshape(n,n), mean.reshape(n,n), 15,cmap="plasma")
    ax[0].scatter(X_train[:,0],X_train[:,1],color="red",marker="x")

    cp1=ax[1].contourf(grid[:, 0].reshape(n, n), grid[:, 1].reshape(n, n), (2*var).reshape(n, n),15,cmap="Blues")
    ax[1].scatter(X_train[:, 0], X_train[:, 1], color="red", marker="x")

    fig.colorbar(cp,ax=ax[0])
    fig.colorbar(cp1,ax=ax[1])
    ax[0].set_title("Mean Plot", pad=23)
    ax[0].set_xlabel("X1")
    ax[0].set_ylabel("X2")

    ax[1].set_title("Variance plot", pad=23)
    ax[1].set_xlabel("X1")
    ax[1].set_ylabel("X2")


    plt.show()

    """ax.scatter(X_train[:, 0], X_train[:, 1], Y_train,
               marker=".", s=1, color="red", label="Train")

    ax.scatter(grid[:, 0], grid[:, 1], mean,
               s=0.6, color="black", label="GP")

    ax.scatter(grid[:, 0], grid[:, 1], mean + 2 * var,
               s=0.5, alpha=0.3, color="Navy", label="Confidence")

    ax.scatter(grid[:, 0], grid[:, 1], mean - 2 * var,
               s=0.5, alpha=0.3, color="Navy")"""
    return plt


def plot_improvement(X_train, Y_train, grid, mean, var, improvement, x_improvement):

    fig, ax = plt.subplots(nrows=2, ncols=1)

    ax[0].fill_between(grid.ravel(), (mean + 2 * var).ravel(), (mean - 2 * var).ravel(),
                       alpha=1, color="lavender", label="GP Variance")

    ax[0].scatter(X_train, Y_train,
                  marker="x", color="red", label="Train")

    ax[0].plot(grid, mean,
                color="black", label="GP Mean")

    ax[0].axvline(x_improvement, label="Suggested Point", linestyle="--",
                  color="red", alpha=0.8)

    ax[1].plot(grid, improvement,
               alpha=0.4, color="orange")

    ax[1].fill_between(grid.ravel(), improvement.ravel(), np.full(grid.ravel().shape, 0),
                       color="orange", alpha=0.4, label="EI")

    ax[1].scatter(x_improvement, np.max(improvement),label="Maxima",color="red",marker="*",s=20)


    ax[0].set_title("Surrogate Model",pad=23)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y value")
    ax[0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=4)

    ax[1].set_title("Acquisition Function",pad=23)
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("EI Value")
    ax[1].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower center", borderaxespad=0, ncol=2)

    plt.tight_layout()
    plt.show()


def plot_improvement_2D(X_train, Y_train, grid, mean, var, improvement, x_improvement):
    n=int(np.sqrt(grid.shape[0]))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    cp=ax[0].contourf(grid[:, 0].reshape(n,n), grid[:, 1].reshape(n,n), mean.reshape(n,n), 15,cmap="plasma")
    ax[0].scatter(X_train[:,0],X_train[:,1],color="red",marker="x")
    ax[0].axhline(x_improvement[1], color="red")
    ax[0].axvline(x_improvement[0], color="red")

    cp1=ax[1].contourf(grid[:,0].reshape(n, n), grid[:, 1].reshape(n, n), improvement.reshape(n, n),15,cmap="OrRd")

    fig.colorbar(cp,ax=ax[0])
    fig.colorbar(cp1,ax=ax[1])
    ax[0].set_title("Mean Plot", pad=23)
    ax[0].set_xlabel("X1")
    ax[0].set_ylabel("X2")

    ax[1].set_title("Variance plot", pad=23)
    ax[1].set_xlabel("X1")
    ax[1].set_ylabel("X2")
    plt.show()




def plot_BayOpt(X_train, Y_train, grid, mean, var, improvement=None, x_improve=None):
    dim = X_train[0].shape[0]

    if improvement is None or x_improve is None:
        args = [X_train, Y_train, grid, mean, var]

        if dim == 1:
            return plot_GP_1D(*args)
        elif dim == 2:
            return plot_GP_2D(*args)
        else:
            raise ValueError("Cannot plot more than 2 dimension")

    else:
        args = [X_train, Y_train, grid, mean, var, improvement, x_improve]

        if dim == 1:
            return plot_improvement(*args)
        elif dim == 2:
            return plot_improvement_2D(*args)
        else:
            raise ValueError("Cannot plot more than 2 dimension")
