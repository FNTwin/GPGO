import numpy as np
from scipy.linalg import solve
from .Kernel.Kernel import Kernel
from .Kernel.RBF import RBF
from .Plotting import plot_BayOpt
from numba import jit
from smt.sampling_methods import LHS

class GP():
    """
    Gaussian Process class
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, kernel: Kernel = RBF(), noise=1e-10, cov: np.ndarray = None):
        self.__dim_input = X[0].shape[0]
        self.__dim_output = Y[0].shape[0]
        self.__dimension = X.shape[0]
        self.__X = X
        self.__Y = Y
        self.__data = {"X": self.__X, "Y": self.__Y}
        self.__kernel = kernel
        self.__noise = noise
        self.__cov = cov
        self.__marg = None
        self.__stat = False


    def static_compute_marg(self):
        Y = self.get_Y()
        K = np.linalg.inv(
            self.get_kernel().kernel_product(self.get_X(), self.get_X()) + self.get_noise() ** 2 * np.eye(Y.shape[0]))

        marg = - .5 * Y.T.dot(K.dot(Y)) - .5 * np.log(np.linalg.det(K))

        self.set_marg(np.squeeze(marg))


    def compute_marg(self, X, Y, pair_matrix, hyper):
        sigma, l = hyper
        kernel = sigma ** 2 * np.exp(-.5 / l ** 2 * pair_matrix) + np.eye(X.shape[0]) * self.get_noise() ** 2
        K = np.linalg.inv(kernel)
        marg = - .5 * Y.T.dot(K.dot(Y)) - .5 * np.log(np.linalg.det(K))

        return marg

    def grid_search_optimization(self, constrains, n_points, function=np.linspace):
        def check_best(best, new):
            return best["marg"] < new

        def set_dict(new_marg, new_hyper):
            best["marg"] = new_marg
            best["hyper"] = new_hyper

        if self.get_marg() is not None:
            best = {"marg": self.get_marg(), "hyper": self.get_kernel().gethyper()}
        else:
            self.static_compute_marg()
            best = {"marg": self.get_marg(), "hyper": self.get_kernel().gethyper()}

        X, Y = self.get_X(), self.get_Y()
        pair_matrix = np.sum(X ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1) - 2 * np.dot(X, X.T)
        hyper_grid = generate_grid(2, n_points, constrains, function)

        for i in hyper_grid:
            tmp_marg = self.compute_marg(X, Y, pair_matrix, i)
            if check_best(best, tmp_marg):
                set_dict(tmp_marg, i)
        return best


    def optimize(self, constrains=[[2, 30], [2, 30]], n_points=100, function=np.linspace):
        new = self.grid_search_optimization(constrains, n_points, function)
        self.set_marg(new["marg"])
        self.set_hyper(new["hyper"][0], new["hyper"][1])
        self.fit()


    def fit(self):
        ker = self.get_kernel()
        n = self.get_dim_data()
        try:
            self.__cov = np.linalg.cholesky(
                ker.kernel_product(self.get_X(), self.get_X()) + 
                self.get_noise() ** 2 * np.eye(self.get_X().shape[0]))

        except:
            raise ValueError("Increase Noise level")
        self.__stat = True


    def predict(self, X):
        ker = self.get_kernel()
        K_sample = ker.kernel_product(self.get_X(), X)
        inv_cov = np.linalg.solve(self.get_cov(), K_sample)
        # cov_sample = ker.kernel_product(X,X)
        # var = np.diag(cov_sample) - np.sum(inv_cov ** 2, axis=0)
        # var = ker.gethyper()[0]**2 - np.sum(inv_cov ** 2, axis=0)
        var =  ker.gethyper()[0] ** 2 - np.sum(inv_cov ** 2, axis=0)
        return np.dot(inv_cov.T, np.linalg.solve(self.get_cov(),self.get_Y()),), 2 * np.sqrt(var)[:, None]
        return mean, np.sqrt(var)


    def plot(self, X):
        mean, var = self.predict(X)
        dim = self.get_dim_data()
        args = [self.get_X(), self.get_Y(), X, mean, var]
        plt = plot_BayOpt(*args)
        plt.legend()
        plt.show()

    def augment_dataset(self, dataset, new_data):
        try:
            dataset.shape[1] == new_data.shape[1]
            return np.concatenate((dataset, new_data))
        except:
            try:
                dataset.shape[1] == new_data.shape[0]
                return np.concatenate((dataset, np.expand_dims(new_data, axis=0)))
            except:
                raise ValueError(f'Data dimensions are wrong: {dataset.shape} != {new_data.shape}')

    def augment_X(self, new_data):
        self.__X = self.augment_dataset(self.get_X(), new_data)

    def augment_Y(self, new_data):
        self.__Y = self.augment_dataset(self.get_Y(), new_data)

    def augment_XY(self, new_data_X, new_data_Y):
        self.augment_X(new_data_X)
        self.augment_Y(new_data_Y)

    def get_kernel(self):
        return self.__kernel

    def get_state(self):
        return self.__stat

    def get_marg(self):
        return self.__marg

    def get_cov(self):
        return self.__cov

    def get_X(self):
        return self.__X

    def get_Y(self):
        return self.__Y

    def get_dim_data(self):
        return self.__dimension

    def get_dim_outspace(self):
        return self.__dim_output

    def get_noise(self):
        return self.__noise

    def set_marg(self, marg):
        self.__marg = marg

    def set_hyper(self, sigma=None, l=None):
        self.get_kernel().sethyper(sigma, l)

    def increase_noise(self):
        self.__noise += 5e-6

    def __str__(self):
        header="#=========================GP===========================\n"
        tail="#======================================================"
        X,Y=self.get_X(),self.get_Y()
        kernel_info=str(self.get_kernel())
        train_info=f'Train values: {self.get_dim_data()}\n'
        train_info+=f'\t\t\t\tX_train\t\t\t\tY_train\n'
        train_info+=f'{np.hstack((X,Y))}\n\n'
        gp_info=f'Noise: {self.get_noise()}\n'
        return header+kernel_info+gp_info+train_info+tail





def generate_grid(n_dim: int, n_points: int, constrains: list, function=np.linspace):
    """
    Build a grid of dim dimension in forms of vectors
    n_dim : Number of dimensions, AKA number of columns for the output vector (int)
    n_points : Number of points to divide the linear grids (int)
    constrains : list containing [min,max] lists to constrain the linear grids
    Output : np.array of dimension n_points**n_dim x n_dim

    =================================Example======================================
    n_dim = 3 , n_points = 4, constrains=[[0,2],[0,2],[0,2]]

    dimension_grid(n_dim , n_points , constrains)

   np.array(array([[0.         , 0.        , 0.       ],
                   [0.        , 0.        , 0.66666667],
                   [0.        , 0.        , 1.33333333],
                   ...,
                   [2.        , 2.        , 0.66666667],
                   [2.        , 2.        , 1.33333333],
                   [2.        , 2.        , 2.        ]]))

    shape: (64,3)
    ==============================================================================
    """

    if n_dim != len(constrains):
        raise ValueError(f'Space dimensions and constrains don\'t match! {n_dim} != {len(constrains)}')
    l = lambda x, y, z: function(x, y, z)
    l_dim = [l(*j, n_points) for i, j in list(zip(range(n_dim), constrains))]
    return np.vstack(np.meshgrid(*l_dim)).reshape(n_dim, -1).T

