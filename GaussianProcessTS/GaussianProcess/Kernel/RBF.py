from .Kernel import Kernel
from numpy import sum, exp
from numpy.linalg import norm
from numpy import ndarray
import numpy as np
from numba import jit

class RBF(Kernel):
    """
    RBF Kernel type class. Type: Kernel, Subtype: RBF
    Init method require the hyperparameters as an input (normal is sigma:2 , l:2)
    """

    def __init__(self, sigma_l=2., l=2.):
        super().__init__()
        self.__hyper = {"sigma": sigma_l, "l": l}
        self.__subtype = "rbf"

    def product(self, x1, x2=0):
        """
        Kernel product between two parameters
        """
        sigma, l = self.gethyper()
        return sigma * exp((-.5 * (norm(x1 - x2) ** 2)) / l)


    def kernel_product(self, X1, X2):
        """
        Function to compute the vectorized kernel product between X1 and X2
        :param X1: Train data
        :param X2: Distance is computed from those
        :return: np.array
        """
        sigma, l = self.gethyper()
        dist = np.sum(X1 ** 2, axis=1)[:, None] + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
        return sigma ** 2 * np.exp(-.5 / l ** 2 * dist)

    def sethyper(self, sigma, l):
        self.__hyper["sigma"] = sigma
        self.__hyper["l"] = l

    '#Get methods '

    def getsubtype(self):
        return self.__subtype

    def gethyper_dict(self):
        return self.__hyper

    def gethyper(self):
        return tuple(self.__hyper.values())

    def __str__(self):
        kernel_info=f'Kernel type: {self.getsubtype()}\t'
        hyper_info=f'Hyperparameters: {self.gethyper_dict()}\n\n'
        return kernel_info+hyper_info
