from .Kernel import Kernel
from numpy import sum, exp
from numpy.linalg import norm
from numpy import ndarray
import numpy as np


class Matern(Kernel):
    """
    RBF Kernel type class. Type: Kernel, Subtype: RBF
    Init method require the hyperparameters as an input (normal is sigma:2 , l:2)
    """

    def __init__(self, sigma_l=2., l=2.):
        super().__init__()
        self.__hyper = {"sigma": sigma_l, "l": l}
        self.__subtype = "matern"

    def product(self, x1, x2=0):
        """
        Kernel product between two parameters
        """
        sigma, l = self.gethyper()
        return sigma ** 2 * (1 + (5 ** .5) * abs(x1-x2) / l +
                                 5 / 3 *
                             abs(x1-x2) ** 2 / l ** 2) * exp(- (5 ** .5) * abs(x1-x2/ l))

    def kernel_product(self, X1, X2):
        """
        Function to compute the vectorized kernel product between X1 and X2
        :param X1: Train data
        :param X2: Distance is computed from those
        :return: np.array
        """

        sigma, l = self.gethyper()
        sq=np.sqrt(5)
        dist = np.sum(X1 ** 2, axis=1)[:, None] + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
        return sigma ** 2 * np.exp(- sq * dist / l ) * (1 +
                                                         sq * dist + 5/ 3 * dist**2) /l

    def sethyper(self, sigma, l):
        self.__hyper["sigma"] = sigma
        self.__hyper["l"] = l

    '#Get methods '

    def getsubtype(self):
        return self.__subtype

    def gethyper(self):
        return tuple(self.__hyper.values())











