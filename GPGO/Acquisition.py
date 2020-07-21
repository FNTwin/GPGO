import logging
import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


class Acquistion():
    def __init__(self, type, gp, epsilon, minimization):
        functions = {"EI": Expected_Improvement,
                     "UCB": Upper_Confidence_Bound}
        self.minimization = minimization
        self.type = type
        self.ac_function = functions[type](gp, epsilon, minimization)

    def call(self, point, best):
        if self.minimization:
            return self.ac_function.func_min(point, best)
        else:
            return self.ac_function.func_max(point, best)


class Expected_Improvement():
    def __init__(self, gp, type, epsilon=0.01, minimization=True):
        self.minimization = minimization
        self.epsilon = epsilon
        self.gp = gp

    def func_max(self, point, best):
        point = np.atleast_2d(point)
        mean, variance = self.gp.predict(point)
        Z = lambda h, m, v, e: (m - h - e) / v
        EI = (mean - best - self.epsilon) * norm.cdf(Z(best, mean, variance, self.epsilon)) \
             + variance * norm.pdf(Z(best, mean, variance, self.epsilon))
        EI[variance == 0.] = 0
        return -EI

    def func_min(self, point, best):
        point = np.atleast_2d(point)
        mean, variance = self.gp.predict(point)
        Z = lambda h, m, v, e: (-m + h + e) / v
        EI = (-mean + best + self.epsilon) * norm.cdf(Z(best, mean, variance, self.epsilon)) \
             + variance * norm.pdf(Z(best, mean, variance, self.epsilon))
        EI[variance == 0] = 0
        return EI


class Upper_Confidence_Bound():
    def __init__(self, gp, type, epsilon=0.01, minimization=True):
        self.minimization = minimization
        self.epsilon = epsilon
        self.gp = gp

    def func_max(self, point, best):
        point = np.atleast_2d(point)
        mean, variance = self.gp.predict(point)
        UCB = (mean + self.epsilon * variance)
        UCB[UCB < 0] = 0
        return UCB

    def func_min(self, point, best):
        point = np.atleast_2d(point)
        mean, variance = self.gp.predict(point)
        UCB = (mean - self.epsilon * variance)
        UCB[UCB > 0] = 0
        return -UCB
