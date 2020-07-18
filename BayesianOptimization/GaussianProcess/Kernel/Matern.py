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

    def __init__(self, sigma_l=2.,  noise=1e-2, gradient=False):
        super().__init__()
        self.hyper = {"sigma": sigma_l,  "noise": noise}
        self.subtype = "matern"
        self.eval_grad= gradient

    def kernel_var(self):
        sigma,  noise = self.gethyper()
        return sigma ** 2 + noise**2

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

        sigma, noise= self.gethyper()
        dist = np.sum(X1 ** 2 , axis=1)[:, None] + np.sum(X2 ** 2 , axis=1) - 2 * np.dot(X1, X2.T)
        K = dist* np.sqrt(5) /sigma
        return (1. + K + K**2 / 3.0) * np.exp(-K)


    @staticmethod
    def kernel_(sigma, noise, pair_matrix):
        K = pair_matrix * np.sqrt(5) / sigma
        return (1. + K + K ** 2 / 3.0) * np.exp(-K) + noise**2


    def sethyper(self, sigma, noise):
        self.hyper["sigma"] = sigma
        self.hyper["noise"] = noise

    '#Get methods '

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











