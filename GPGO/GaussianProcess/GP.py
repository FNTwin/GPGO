import logging

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from .Kernel import Kernel
from .Kernel import RBF
from .Plotting import plot_BayOpt

logger = logging.getLogger(__name__)


class GP():
    """
    Gaussian Process class
    ...
    Attributes
    ----------
    dim_input : int
        Dimension of the input space
    dim_output : int
        Dimension of the output space
    dimension : int
        Number of the training points
    X : np.array
        Training sample points written as column vector. 5 Training points of 2 dimensions -> shape(5,2)
                                                         1 Training point of 2 dimensions -> shape(1,1)
    Y : np.array
        Training target points written as column vector. 5 Target points of 1 dimension -> shape(5,1)
                                                         1 Target point of 1 dimension -> shape(1,1)
    data : dict
        Dictionary with X and Y as keys that holds the values of the respective array
    kernel : Kernel object (default RBF)
        Chosen kernel for the Gaussian Process
    cov : np.array (default None)
        Covariance of the GP
    normalize_y : bool (default True)
        Flag to normalize the target values to 0 mean and 1 variance
    marg : float (default None)
        Log Marginal Likelihood value
    _stat : bool (default False)
        Flag for the fit

    Example
    ---------
        x=np.random.uniform(0,3,10)[:,None]
        y=np.sin(x)
        gp = GP(x, y)
        gp.optimize()

    Methods
    ---------
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, kernel: Kernel = RBF(), cov: np.ndarray = None, normalize_y=True):
        """
            X : np.array
                Training sample points written as column vector.
            Y : np.array
                Training target points written as column vector.
            kernel : Kernel object
            cov : np.array (default None)
                Covariance of the GP after the fit. A covariance matrix can be passed to use for the prediction.
            normalize_y : bool (default True)
                Flag to normalize the target values to 0 mean and 1 variance
            """
        self.dim_input = X[0].shape[0]
        self.dim_output = Y[0].shape[0]
        self.dimension = X.shape[0]
        self.X_train = X
        self.Y_train = Y
        self.data = {"X": self.X_train, "Y": self.Y_train}
        self.kernel = kernel
        self.cov = cov
        self.normalize_y = normalize_y
        self.marg = None
        self._stat = False

    def log_marginal_likelihood(self):
        """
        Compute the Log Marginal Likelihood for the current set of Training points and the current
        hyperparameters while storing the result value in __marg.
        By default it tries to use the cholesky decomposition to compute the log marginal likelihood.
        If there is some numerical error (usually by too litlle noise) it tries to use the standard
        inversion routine of the Graham Matrix. If it happens the noise of the GP should be increased.
        """

        Y = self.get_Y()
        if self.normalize_y:
            self._mean_y = np.mean(self.get_Y(), axis=0)
            self._std_y = np.std(self.get_Y(), axis=0)
            Y = (self.get_Y() - self._mean_y) / self._std_y

        kernel = self.get_kernel().kernel_product(self.get_X(), self.get_X())
        kernel[np.diag_indices_from(kernel)] += self.get_noise() ** 2

        try:
            if self._stat:
                K = self.get_cov()
            else:
                K = np.linalg.cholesky(kernel)

            marg = - .5 * Y.T.dot(np.linalg.lstsq(K.T, np.linalg.lstsq(K, Y, rcond=None)[0], rcond=None)[0]) \
                   - .5 * np.log(np.diag(K)).sum() \
                   - .5 * K.shape[0] * np.log(2 * np.pi)

        except np.linalg.LinAlgError as exc:

            logging.info(exc, "\nComputing as a normal inverse\n")
            K = np.linalg.inv(kernel)
            marg = - .5 * Y.T.dot(K.dot(Y)) \
                   - .5 * np.log(np.diag(K)).sum() \
                   - .5 * K.shape[0] * np.log(2 * np.pi)

        self.set_marg(marg)

    def compute_log_marginal_likelihood(self, X, Y, pair_matrix, hyper, verbose=False):
        """
        Routine to calculate the negative log marginal likelihood for a given set of hyperparameters.
        :param X: np.array
            Training points for the calculation of the Log Marginal likelihood
        :param Y: np.array
            Target points for the calculation
        :param pair_matrix: np.array
            Distance matrix between X and Y
        :param hyper: array-like
            Array of hyperparameters
        :param verbose: bool (default False)
            print the process of the optimizers on the console
        :return: The value of the negative log marginal likelihood
            If eval_grad attribute of the kernel is set on True, the method will use the gradient informations.
            With eval_grad=True it will return an array of the negative value for the log marginal likelihood and the
             needed derivates.
        It tries to calculate using the Cholesky decomposition. If it fails it also tries to
        compute the normal inversion of the Graham matrix. If the set of hyperparameters causes the calculations to fail
        for a np.linalg.error, it sets the marginal likelihood to infinity to discard the set of parameters.
        """

        logger.debug(hyper)

        if self.normalize_y:
            self._mean_y = np.mean(self.get_Y(), axis=0)
            self._std_y = np.std(self.get_Y(), axis=0)
            Y = (self.get_Y() - self._mean_y) / self._std_y

        kernel = self.kernel.kernel_(*hyper, pair_matrix)

        try:
            K = np.linalg.cholesky(kernel)
            marg = - .5 * Y.T.dot(np.linalg.lstsq(K.T, np.linalg.lstsq(K, Y, rcond=None)[0], rcond=None)[0]) \
                   - .5 * np.log(np.diag(K)).sum() \
                   - .5 * K.shape[0] * np.log(2 * np.pi)

        except np.linalg.LinAlgError as exc:
            try:
                logger.info(exc, "\nComputing as a normal inverse\n")
                K = np.linalg.inv(kernel)
                marg = - .5 * Y.T.dot(K.dot(Y)) \
                       - .5 * np.log(np.diag(K)).sum() \
                       - .5 * K.shape[0] * np.log(2 * np.pi)

            except np.linalg.LinAlgError as exc2:
                if verbose:
                    logger.warning(exc2)
                marg = np.inf

        if self.get_kernel().get_eval():
            marg_grad = self.compute_log_marginal_likelihood_gradient(X, Y, pair_matrix, hyper, verbose=False)
            return -marg, marg_grad
        else:
            return -marg

    def compute_log_marginal_likelihood_gradient(self, X, Y, pair_matrix, hyper, verbose=False):
        """
        Routine to calculate the gradient of log marginal likelihood for a set of hyperparameters.
        :param X: np.array
            Training points for the calculation of the Log Marginal likelihood
        :param Y: np.array
            Target points for the calculation
        :param pair_matrix: np.array
            Distance matrix between X and Y
        :param hyper: array-like
            Array of hyperparameters
        :param verbose: bool (default False)
            print the process of the optimizers on the console
        :return: The value of the gradients of negative log marginal likelihood.
        """

        logger.debug("GRADIENT", hyper)
        hyper = np.squeeze(hyper)
        m_g = lambda h: -(.5 * Y.T.dot(K.dot(h.dot(K.dot(Y)))) - .5 * np.diag(K.dot(h)).sum())

        kernel = self.kernel.kernel_eval_grad_(*hyper, pair_matrix)
        K = np.linalg.inv(kernel[0])
        try:
            grad_kernel = [m_g(i) for i in kernel[1:]]

        except np.linalg.LinAlgError as exc:
            if verbose:
                logger.warning(exc)

        return np.array(grad_kernel)

    def grid_search_optimization(self, constrains, n_points, function=np.linspace, verbose=False):
        """
        Routine called by optimize_grid to handle the optimization.
        :param constrains: list
            Array-like with the constrains for the hyperparameters. Ex: [[lb,ub],[lb,ub],[lb,ub]]
        :param n_points: int
            Number of points sampled in each dimension. The number of total evaluations will be n_points^d(hyper)
        :param function: callable (default np.linspace)
            function that will be used to generate the sampling point.
            np.random.uniform : randomized grid search
            np.linspace : normal grid search
        :param verbose: bool (default False)
            print the process of the optimizers on the console
        """
        self.__optimizer = "Grid search"

        def check_best(best, new):
            return best["marg"] < new

        def set_dict(new_marg, new_hyper):
            best["marg"] = new_marg
            best["hyper"] = new_hyper

        if self.get_marg() is not None:
            best = {"marg": self.get_marg(), "hyper": self.get_kernel().gethyper()}
        else:
            self.log_marginal_likelihood()
            best = {"marg": self.get_marg(), "hyper": self.get_kernel().gethyper()}

        X, Y = self.get_X(), self.get_Y()

        pair_matrix = np.sum(X ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1) - 2 * np.dot(X, X.T)
        hyper_grid = generate_grid(3, n_points, constrains, function)

        for i in hyper_grid:
            tmp_marg = self.compute_log_marginal_likelihood(X, Y, pair_matrix, i, verbose)
            if check_best(best, -tmp_marg):
                set_dict(-tmp_marg, i)

        return best

    def optimize_grid(self, constrains=[[1e-5, 30], [1e-5, 30], [1e-5, 10]], n_points=100, function=np.linspace,
                      verbose=False):
        """
        Routine for optimizing the hyperparameters by maximizing the log marginal Likelihood by a naive grid search.
        :param constrains: list (default [[1e-5, 30], [1e-5, 30], [1e-5, 10]])
            Array-like with the constrains for the hyperparameters. Ex: [[lb,ub],[lb,ub],[lb,ub]]
        :param n_points: int (default 100)
            Number of points sampled in each dimension. The number of total evaluations will be n_points^3
        :param function: func (default np.linspace)
            function that will be used to generate the sampling point.
            np.random.uniform : randomized grid search
            np.linspace : normal grid search
        :param verbose: bool (default False)
            print the process of the optimizers on the console
        """
        args = (constrains, n_points, function, verbose)
        new = self.grid_search_optimization(*args)
        self.set_marg(new["marg"])
        # fare un buon metodo set hyper
        self.set_hyper(*new["hyper"])
        self.fit()

    def optimize(self, n_restarts=10, optimizer="L-BFGS-B", verbose=False):
        """
        Optimization Routine for the hyperparameters of the GP by Minimizing the negative log Marginal Likelihood.
        n_restarts : int (default 10)
            Number of restart points for the optimizer
        optimizer : str (default L-BFGS-B)
            Type of Optimizer chosen for the task. Because the Optimization is bounded, the following
            optimizers can be used: L-BFGS-B, SLSCP, TNC
            For a better documentation of the optimizers follow the Scipy documentation
        verbose : bool (default False)
            print the process of the optimizers on the console
        If eval_grad of the Kernel is set on True , the
        gradient of the Log Marginal Likelihood will be calculated and used for the optimization. This usually speed up
        the convergence but it also makes the Optimizer more prone to find local minima.
        """

        self.__optimizer = optimizer
        self.log_marginal_likelihood()
        old_marg = self.get_marg()
        old_hyper = self.get_kernel().gethyper()
        hyper_dim = len(old_hyper)

        boundaries = self.get_boundary()
        eval_grad = self.get_kernel().get_eval()

        logger.debug("Starting Log Marginal Likelihood Value: ", old_marg)

        X_d, Y_d = self.get_X(), self.get_Y()
        pair_matrix = np.sum(X_d ** 2, axis=1)[:, None] + np.sum(X_d ** 2, axis=1) - 2 * np.dot(X_d, X_d.T)

        new_hyper = None
        new_marg = None
        it = 0

        # Optimization Loop
        for i in np.random.uniform(boundaries[:, 0], boundaries[:, 1], size=(n_restarts, hyper_dim)):
            it += 1

            logger.debug("RESTART :", it, i)

            res = minimize(lambda h: self.compute_log_marginal_likelihood(X=X_d,
                                                                          Y=Y_d,
                                                                          pair_matrix=pair_matrix,
                                                                          verbose=verbose,
                                                                          hyper=(np.squeeze(
                                                                              h.reshape(-1, hyper_dim)))),
                           x0=i,
                           bounds=((1e-5, None), (1e-5, None), (1e-5, None)),
                           method=self.__optimizer,
                           jac=eval_grad)

            if not res.success:
                continue

            try:
                if new_marg is None or res.fun[0] < new_marg:
                    new_marg = res.fun[0]
                    new_hyper = res.x
            except:
                pass
                logger.info(res.fun)

        # If the optimization doesn't converge set the value to old parameters
        if new_marg is None:
            if verbose:
                logger.warning("Using Old Log Marg Likelihood")
            new_marg = old_marg
            new_hyper = old_hyper

        logger.info("New Log Marginal Likelihood Value: ", -new_marg, new_hyper)
        # Update and fit the new gp model
        self.set_hyper(*new_hyper)
        self.set_marg(-new_marg)
        self.fit()

    def fit(self):
        """
        Compute the Graham Matrix for the kernel and the cholesky decomposition required for the prediction.
        It stores the result in cov and change the flag of stat in True
        """
        ker = self.get_kernel()
        n = self.get_dim_data()
        try:
            kernel = self.get_kernel().kernel_product(self.get_X(), self.get_X())
            kernel[np.diag_indices_from(kernel)] += self.get_noise() ** 2
            self.cov = np.linalg.cholesky(kernel)
            self._stat = True

        except np.linalg.LinAlgError as exc:
            raise ValueError("Cholesky decomposition encountered a numerical error\nIncrease the Noise Level\n", exc)

    def predict(self, X):
        """
        Make prediction on X samples
        :param X: np.array
        :return: tuple of 2 elements
            GP mean and GP Variance
        """

        # Check if we require to normalize the target values
        # normalizer = np.sqrt(2 * np.pi * ker.gethyper()[0] ** 2)
        ker = self.get_kernel()
        K_sample = ker.kernel_product(self.get_X(), X)
        inv_cov = np.linalg.solve(self.get_cov(), K_sample)
        if not self.normalize_y:
            # cov_sample = ker.kernel_product(X,X)
            # Mean
            mean = np.dot(inv_cov.T, np.linalg.solve(self.get_cov(), self.get_Y()))
            # general case
            # Variance
            # var = ker.gethyper()[0] ** 2 + ker.gethyper()[2] - np.sum(inv_cov ** 2, axis=0)
            var = ker.kernel_var() - np.sum(inv_cov ** 2, axis=0)
            var_neg = var < 0
            var[var_neg] = 0.
            var = np.sqrt(var)[:, None]
            return mean, var

        else:
            # Normalize Y
            self._mean_y = np.mean(self.get_Y(), axis=0)
            self._std_y = np.std(self.get_Y(), axis=0)
            Y = (self.get_Y() - self._mean_y) / self._std_y
            # MEAN
            y_mean = np.dot(inv_cov.T, np.linalg.solve(self.get_cov(), Y))
            # DESTANDARDIZE
            y_mean = (self._std_y * y_mean + self._mean_y)
            # VARIANCE
            # var = ker.gethyper()[0] ** 2 + self.get_noise() - np.sum(inv_cov ** 2, axis=0)
            var = ker.kernel_var() - np.sum(inv_cov ** 2, axis=0)
            # REMOVE VARIANCE VALUE LESS THAN 0
            var_neg = var < 0
            var[var_neg] = 0.
            # DeStandardize
            var = np.sqrt(var * self._std_y ** 2)[:, None]
            return y_mean, var

    def plot(self, X):
        """
        Function that handles the prediction of the GP object and the plotting without returning anything.
        For visual reasons it only works with input data of 1 dimension and 2 dimension
        :param X: np.array
            Data to use to predict and plot
        """
        mean, var = self.predict(X)
        dim = self.get_dim_data()
        args = [self.get_X(), self.get_Y(), X, mean, var]
        plt = plot_BayOpt(*args)
        plt.show()

    def prepare_fold(self, n):
        kf = KFold(n_splits=n)
        kf.get_n_splits(self.get_dim_data())
        folds = []
        for train, test in kf.split(self.get_X()):
            folds.append([train, test])
        return folds

    @staticmethod
    def RMSE(gp_model, dataset_X, dataset_Y, test_index):
        def func(pred, test):
            return np.sqrt(np.mean(np.squeeze((pred - test) ** 2)))

        gp_model.fit()
        test_x = dataset_X[test_index]
        test_y = dataset_Y[test_index]
        pred = gp_model.predict(test_x)
        return func(pred, test_y)

    def k_fold_cv(self, n_fold, hyper):
        index = self.prepare_fold(n_fold)
        RMSE = 0
        for i in range(n_fold):
            X_train, X_test = self.get_X()[index[i][0]], self.get_X()[index[i][1]]
            Y_train, Y_test = self.get_Y()[index[i][0]], self.get_Y()[index[i][1]]
            cv_gp = GP(X_train, Y_train, kernel=RBF(*hyper))
            RMSE += GP.RMSE(cv_gp, self.get_X(), self.get_Y(), index[i][1])
        return RMSE / n_fold

    def augment_dataset(self, dataset, new_data):
        """
        General function to augment an array with a shape check
        :param dataset: Dataset to augment
        :param new_data: Data to agument the dataset with
        :return: new array
        """
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
        self.X_train = self.augment_dataset(self.get_X(), new_data)

    def augment_Y(self, new_data):
        self.Y_train = self.augment_dataset(self.get_Y(), new_data)

    def augment_XY(self, new_data_X, new_data_Y):
        self.augment_X(new_data_X)
        self.augment_Y(new_data_Y)

    def get_kernel(self):
        return self.kernel

    def get_state(self):
        return self._stat

    def get_marg(self):
        return self.marg

    def get_cov(self):
        return self.cov

    def get_X(self):
        return self.X_train

    def get_Y(self):
        return self.Y_train

    def get_dim_data(self):
        return self.dimension

    def get_dim_outspace(self):
        return self.dim_output

    def set_marg(self, marg):
        self.marg = marg

    def get_boundary(self):
        """
        Get the boundaries for the hyperparameters optimization routine.
        :return: np.array
            Array of boundaries
        If no boundaries were previously set it creates some dummy boundaries (1e-5,10) for all hyperparameters
        in the kernel.
        """
        try:
            return self.__boundary
        except AttributeError:
            self.__boundary = np.asarray([[1e-4, 10] for i in range(len(self.get_kernel().gethyper()))])
            return self.__boundary

    def set_boundary(self, array):
        """
        Create the boundaries for the hyperparameters optimization
        :param array: list
            [lb,ub] type of list to set the boundaries, it can be the same dimensions of the number of hyperparameters
            or just have one array inside

        If the array has the same length required for the hyperparameters space the space will be bounded.
        If the array only has one dimension, then it will set the space as the hypercube of the array lb and ub
        All others types of input are not supported
        _____________________________________________
        Example
        set_boundary([[1e-5,4]]) on RBF ---> boundaries: (1e-5,4),(1e-5,4),(1e-5,4)
        set_boundary([[1e-5,4],[1,10],[2,3]]) on RBF ---> boundaries: (1e-5,4),(1,10),(2,3)
        """
        n = len(self.get_kernel().gethyper())

        if len(array) == 1:
            self.__boundary = np.asarray([array[0] for i in range(n)])
        if len(array) == n:
            self.__boundary = np.asarray(array)

    def set_hyper(self, sigma=None, l=None, noise=None):

        self.get_kernel().sethyper(sigma, l, noise)

    def get_noise(self):
        ker = self.get_kernel()
        return ker.get_noise()

    def save_model(self, path):
        with open(path, "w") as file:
            file.write(str(self))

    def __str__(self):
        header = "#=========================GP===========================\n"
        tail = "#======================================================"
        X, Y = self.get_X(), self.get_Y()
        try:
            gp_info = f"Optimizer: {self.__optimizer}\n"
        except:
            gp_info = f"Optimizer: No Optimized\n"
        kernel_info = str(self.get_kernel())
        train_info = f'Train values: {self.get_dim_data()}\nInput Dimension: {self.dim_input} Output Dimension: {self.get_dim_outspace()}\n'
        train_info += f'\t\t\t\tX_train\t\t\t\tY_train\n'
        train_info += f'{np.hstack((X, Y))}\n\n'
        gp_info += f'Noise: {self.get_noise()}\nLog Marginal Likelihood: {self.get_marg()}\n'
        surprise = "\n░░░░░░░░▄▄▄▀▀▀▄▄███▄░░░░░░░░░░░░░░\n░░░░░▄▀▀░░░░░░░▐░▀██▌░░░░░░░░░░░░░\n░░░▄▀░░░░▄▄███░▌▀▀░▀█░░░░░░░░░░░░░\n░░▄█░░▄▀▀▒▒▒▒▒▄▐░░░░█▌░░░░░░░░░░░░\n░▐█▀▄▀▄▄▄▄▀▀▀▀▌░░░░░▐█▄░░░░░░░░░░░\n░▌▄▄▀▀░░░░░░░░▌░░░░▄███████▄░░░░░░\n░░░░░░░░░░░░░▐░░░░▐███████████▄░░░\n░░░░░le░░░░░░░▐░░░░▐█████████████▄\n░░░░toucan░░░░░░▀▄░░░▐█████████████▄\n░░░░░░has░░░░░░░░▀▄▄███████████████\n░░░░░arrived░░░░░░░░░░░░█▀██████░░\n"
        return header + kernel_info + gp_info + train_info + tail + surprise

    def __repr__(self):
        header = "#=========================GP===========================\n"
        tail = "#======================================================"
        X, Y = self.get_X(), self.get_Y()
        try:
            gp_info = f"Optimizer: {self.__optimizer}\n"
        except:
            gp_info = f"Optimizer: No Optimized\n"
        kernel_info = str(self.get_kernel())
        train_info = f'Train values: {self.get_dim_data()}\nInput Dimension: {self.dim_input} Output Dimension: {self.get_dim_outspace()}\n'
        train_info += f'\t\t\t\tX_train\t\t\t\tY_train\n'
        train_info += f'{np.hstack((X, Y))}\n\n'
        gp_info += f'Noise: {self.get_noise()}\nLog Marginal Likelihood: {self.get_marg()}\n'
        surprise = "░░░░░░░░▄▄▄▀▀▀▄▄███▄░░░░░░░░░░░░░░\n░░░░░▄▀▀░░░░░░░▐░▀██▌░░░░░░░░░░░░░\n░░░▄▀░░░░▄▄███░▌▀▀░▀█░░░░░░░░░░░░░\n░░▄█░░▄▀▀▒▒▒▒▒▄▐░░░░█▌░░░░░░░░░░░░\n░▐█▀▄▀▄▄▄▄▀▀▀▀▌░░░░░▐█▄░░░░░░░░░░░\n░▌▄▄▀▀░░░░░░░░▌░░░░▄███████▄░░░░░░ ░░░░░░░░░░░░░▐░░░░▐███████████▄░░░\n░░░░░le░░░░░░░▐░░░░▐█████████████▄\n░░░░toucan░░░░░░▀▄░░░▐█████████████▄\n░░░░░░has░░░░░░░░▀▄▄███████████████\n░░░░░arrived░░░░░░░░░░░░█▀██████░░\n"
        return header + kernel_info + gp_info + train_info + tail + surprise


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
