from .Kernel import Kernel
from numpy import sum, exp
from numpy.linalg import norm
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt

class RBF(Kernel):
    """
    RBF Kernel type class. Type: Kernel, Subtype: RBF
    Init method require the hyperparameters as an input (variance, lengthscale, noise)
    """

    def __init__(self, sigma_l=1., l=1. , noise=1e-6):
        super().__init__()
        self.__hyper = {"sigma": sigma_l, "l": l, "noise": noise}
        self.__subtype = "rbf"

    @staticmethod
    def kernel_(sigma,l,pair_matrix):
        return sigma ** 2 * np.exp(-.5 / l ** 2 * pair_matrix)

    def product(self, x1, x2=0):
        """
        Kernel product between two parameters
        """
        sigma, l , noise= self.gethyper()
        return sigma**2 * exp((-.5 * (norm(x1 - x2,axis=1) ** 2)) / l**2)


    def kernel_product(self, X1, X2):
        """
        Function to compute the vectorized kernel product between X1 and X2
        :param X1: Train data
        :param X2: Distance is computed from those
        :return: np.array
        """
        sigma, l , noise= self.gethyper()
        dist = np.sum(X1 ** 2, axis=1)[:, None] + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
        return sigma ** 2 * np.exp(-.5 / l ** 2 * dist)

    def plot(self):
        """
        Plot the kernel as a 1-D figure
        """
        X=np.linspace(-4,4,100)
        plt.plot(X,self.product(X[:,None]))

    def sethyper(self, sigma, l, noise):
        """
        Set new hyperparameters value
        :param sigma: Variance
        :param l: Lengthscale
        :param noise: Noise
        """
        self.__hyper["sigma"] = sigma
        self.__hyper["l"] = l
        self.__hyper["noise"]= noise

    '#Get methods '

    def getsubtype(self):
        """
        :return: Subtype of the kernel RBF
        """
        return self.__subtype

    def gethyper_dict(self):
        return self.__hyper

    def gethyper(self):
        return tuple(self.__hyper.values())

    def get_noise(self):
        return self.__hyper["noise"]

    def __str__(self):
        kernel_info=f'Kernel type: {self.getsubtype()}\n'
        hyper_info=f'Hyperparameters: {self.gethyper_dict()}\n\n'
        return kernel_info+hyper_info
