from .Kernel import Kernel
from numpy import exp
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt


class RBF(Kernel):
    """
    RBF Kernel object. Type: Kernel, Subtype: RBF
    ...

    Attributes:
    -----------
        hyper      : dict
            Dictionary with sigma_l,l,noise as key
        subtype    : str
            subtype of the kernel , the general type is always kernel while the subtype is RBF
        eval_grad  : bool
            flag to compute the kernel gradient
    Methods
    -----------
    """

    def __init__(self, sigma_l=1., l=1., noise=1e-6, gradient=False):
        """
        sigma_l  : float (default 1.)
            float value for the variance of the Kernel
        l        : float (default 1.)
            float value for the lengthscale of the Kernel
        noise    : float (default 1e-6)
            float value for noise of the Gaussian Process
        gradient : bool (default False)
            if set to True it compute the gradient when called to maximize the marginal likelihood"""
        super().__init__()
        self.hyper = {"sigma": sigma_l, "l": l, "noise": noise}
        self.subtype = "rbf"
        self.eval_grad = gradient

    def kernel_var(self):
        sigma, _, noise = self.gethyper()
        return sigma ** 2 + noise ** 2

    @staticmethod
    def kernel_(sigma, l, noise, pair_matrix):
        """
        Static method for computing the Graham Matrix using the kernel, it is require to be defined as it
        is passed in the GP module to optimize the hyperparameters.
        :param sigma: float
            Variance hyperparameter value
        :param l: float
            Variance hyperparameter value
        :param noise: float
            Variance hyperparameter value
        :param pair_matrix: NxM np.array
            Distance matrix between pairs of points
        :return: NxM np.array
            Graham matrix of the Kernel
        """
        K = sigma ** 2 * np.exp(-.5 / l ** 2 * pair_matrix)
        K[np.diag_indices_from(K)] += noise ** 2
        return K

    @staticmethod
    def kernel_eval_grad_(sigma, l, noise, pair_matrix):
        """
            Static method for computing the derivates of the Kernel, it is require to be defined if __eval_grad is set
            to True as it is passed in the GP module to optimize the hyperparameters with the use of the gradient.
            :param sigma: float
                Variance hyperparameter value
            :param l: float
                Variance hyperparameter value
            :param noise: float
                Variance hyperparameter value
            :param pair_matrix: NxM np.array
                Distance matrix between pairs of points
            :return: NxM np.array
                Graham matrix of the Kernel
        """
        K = np.exp(-.5 / l ** 2 * pair_matrix)
        K_norm = sigma ** 2 * K
        K_norm[np.diag_indices_from(K)] += noise ** 2
        K_1 = 2 * sigma * K
        K_2 = sigma ** 2 * K * pair_matrix / l ** 3
        K_3 = np.eye(pair_matrix.shape[0]) * noise * 2
        return (K_norm, K_1, K_2, K_3)

    def product(self, x1, x2=0):
        """
        Methods used to compute the kernel values on single points.
        x1 : np.array
        x2 : np.array (default 0)
        :return: kernel value between two data point x1,x2
        """
        sigma, l, noise = self.gethyper()
        return sigma ** 2 * exp((-.5 * (norm(x1 - x2, axis=1) ** 2)) / l ** 2)

    def kernel_product(self, X1, X2):
        """
        Vectorized method to compute the kernel product between X1 and X2 by creating a distance matrix of the X1 and X2
        entry and then computing the Kernel. The Kernel is computing respect the X2 points.
        X1: np.array of shape N,.
            N points data
        X2: np.array of shape M,.
           M points data
        return : np.array of dimension NxM
            return the Graham Matrix of the Kernel
        """
        sigma, l, noise = self.gethyper()
        dist = np.sum(X1 ** 2, axis=1)[:, None] + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
        return sigma ** 2 * np.exp(-.5 / l ** 2 * dist)

    def plot(self):
        """
        Plot the Kernel in 1 Dimension
        """
        X = np.linspace(-4, 4, 100)
        plt.plot(X, self.product(X[:, None]))
        plt.show()

    def sethyper(self, sigma, l, noise):
        """
        Set new hyperparameters value
        :param sigma: Variance
        :param l: Lengthscale
        :param noise: Noise
        """
        self.hyper["sigma"] = sigma
        self.hyper["l"] = l
        self.hyper["noise"] = noise

    '#Get methods'
    def getsubtype(self):
        return self.subtype

    def gethyper_dict(self):
        return self.hyper

    def gethyper(self):
        return tuple(self.hyper.values())

    def get_noise(self):
        return self.hyper["noise"]

    def get_eval(self):
        return self.eval_grad

    def __str__(self):
        kernel_info = f'Kernel type: {self.getsubtype()}\n'
        hyper_info = f'Hyperparameters: {self.gethyper_dict()}\n\n'
        return kernel_info + hyper_info
