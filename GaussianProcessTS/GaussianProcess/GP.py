import numpy as np
from scipy.linalg import solve
from .Kernel.Kernel import Kernel
from .Kernel.RBF import RBF
from .Plotting import plot_BayOpt
from scipy.optimize import minimize
from scipy.linalg import cho_solve, cholesky
from sklearn.model_selection import train_test_split, KFold
#from smt.sampling_methods import LHS

class GP():
    """
    Gaussian Process class
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, kernel: Kernel = RBF(), cov: np.ndarray = None,  normalize_y=True):
        self.__dim_input = X[0].shape[0]
        self.__dim_output = Y[0].shape[0]
        self.__dimension = X.shape[0]
        self.__X = X
        self.__Y = Y
        self.__data = {"X": self.__X, "Y": self.__Y}
        self.__kernel = kernel
        self.__cov = cov
        self.__normalize_y=True
        self.__marg = None
        self.__stat = False


    def static_compute_marg(self):

        Y = self.get_Y()
        K = np.linalg.cholesky(self.get_kernel().kernel_product(self.get_X(), self.get_X())
                          + self.get_noise()**2 * np.eye(self.get_X().shape[0]))

        marg = - .5 * Y.T.dot(np.linalg.lstsq(K.T, np.linalg.lstsq(K, Y)[0])[0]) \
               -.5 * np.log(np.diag(K)).sum()\
               -.5* K.shape[0]* np.log(2*np.pi)

        self.set_marg(marg)


    def compute_marg(self, X, Y, pair_matrix, hyper, verbose=False):
        if verbose:
            print(hyper)
        sigma, l , noise= hyper
        #noise=self.get_noise()

        if self.__normalize_y:
            self._mean_y = np.mean(self.get_Y(), axis=0)
            self._std_y = np.std(self.get_Y(), axis=0)
            Y = (self.get_Y() - self._mean_y) / self._std_y

        #kernel = sigma ** 2 * np.exp(-.5 / l ** 2 * pair_matrix)
        kernel = self.__kernel.kernel_(sigma,l,pair_matrix)
        kernel += np.eye(X.shape[0]) * noise ** 2

        try:
            K = np.linalg.cholesky(kernel)
            marg = - .5 * Y.T.dot(np.linalg.lstsq(K.T, np.linalg.lstsq(K, Y)[0])[0]) \
                   - .5 * np.log(np.diag(K)).sum() \
                   - .5 * K.shape[0] * np.log(2 * np.pi)

        except np.linalg.LinAlgError as exc:
            print(exc, "\nComputing as a normal inverse\n")
            K=np.linalg.inv(kernel)
            marg = - .5 * Y.T.dot(K.dot( Y)) \
                   - .5 * np.log(np.diag(K)).sum() \
                   - .5 * K.shape[0] * np.log(2 * np.pi)

        return marg

    def prepare_fold(self, n):
        kf= KFold(n_splits=n)
        kf.get_n_splits(self.get_dim_data())
        folds=[]
        for train, test in kf.split(self.get_X()):
            folds.append([train,test])
        return folds

    @staticmethod
    def RMSE(gp_model,dataset_X,dataset_Y,test_index):
        def func(pred,test):
            return np.sqrt(np.mean(np.squeeze((pred-test)**2)))
        gp_model.fit()
        test_x=dataset_X[test_index]
        test_y=dataset_Y[test_index]
        pred=gp_model.predict(test_x)
        return func(pred,test_y)

    def k_fold_cv(self, n_fold, hyper):
        index=self.prepare_fold(n_fold)
        RMSE=0
        for i in range(n_fold):
            X_train , X_test = self.get_X()[index[i][0]], self.get_X()[index[i][1]]
            Y_train , Y_test = self.get_Y()[index[i][0]], self.get_Y()[index[i][1]]
            cv_gp=GP(X_train,Y_train,kernel=RBF(*hyper))
            RMSE+=GP.RMSE(cv_gp,self.get_X(),self.get_Y(),index[i][1])
        return RMSE/n_fold

    def grid_search_optimization(self, constrains, n_points, function=np.linspace, verbose=False):
        self.__optimizer="Grid search"
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
        hyper_grid = generate_grid(3, n_points, constrains, function)

        for i in hyper_grid:
            tmp_marg = self.compute_marg(X, Y, pair_matrix, i, verbose)
            if check_best(best, tmp_marg):
                set_dict(tmp_marg, i)

        return best

    def optimize(self, constrains=[[1e-5, 30],[1e-5, 30],[1e-5,10]], n_points=100, function=np.linspace, verbose=False):
        args=(constrains,n_points,function,verbose)
        new = self.grid_search_optimization(*args)
        self.set_marg(new["marg"])
        self.set_hyper(new["hyper"][0], new["hyper"][1], self.get_noise())
        self.fit()

    def opt(self, constrains=[[1e-4,10],[1e-4,10],[1e-4,10]], n_restarts=10, verbose=False):
        self.__optimizer="L-BFGS-B"
        boundaries=np.asarray(constrains)
        self.static_compute_marg()
        marg = self.get_marg()

        print("Starting Log Marginal Likelihood Value: ", marg)

        X_d, Y_d = self.get_X(), self.get_Y()
        pair_matrix = np.sum(X_d ** 2, axis=1)[:, None] + np.sum(X_d ** 2, axis=1) - 2 * np.dot(X_d, X_d.T)

        new_hyper = None
        new_marg = None
        it=0

        for i in np.random.uniform(boundaries[:, 0], boundaries[:, 1], size=(n_restarts, 3)):
            it+=1
            print("RESTART :", it)
            res = minimize(lambda h: -self.compute_marg(X=X_d,
                                                        Y=Y_d,
                                                        pair_matrix=pair_matrix,
                                                        verbose=verbose,
                                                        hyper=np.squeeze(h.reshape(-1, 3))),
                           x0=i,
                           bounds=((1e-5,None),(1e-5,None),(1e-5,None)),
                           method='L-BFGS-B')

            if not res.success:
                continue
            if new_marg is None or res.fun[0]<new_marg:
                new_marg = res.fun[0]
                new_hyper = res.x


        print("New Log Marginal Likelihood Value: ", -new_marg)
        print(new_hyper)

        #Update and fit the new gp model
        self.set_hyper(*new_hyper)
        self.set_marg(-new_marg)
        self.fit()

    def fit(self):
        ker = self.get_kernel()
        n = self.get_dim_data()
        try:
            self.__cov = np.linalg.cholesky((ker.kernel_product(self.get_X(), self.get_X()) +
                                  self.get_noise()**2  * np.eye(self.get_X().shape[0])))

            self.__stat = True

        except np.linalg.LinAlgError as exc:
            raise ValueError("Cholesky decomposition encountered a numerical error\nIncrease the Noise Level")



    def predict(self, X):
        #Check if we require to normalize the target values
        if not self.__normalize_y:
            ker = self.get_kernel()
            K_sample = ker.kernel_product(self.get_X(), X)
            inv_cov = np.linalg.solve(self.get_cov(), K_sample)

            #Mean
            mean=np.dot(inv_cov.T, np.linalg.solve(self.get_cov(), self.get_Y()))
            #general case
            # cov_sample = ker.kernel_product(X,X)
            #Variance
            var = ker.gethyper()[0]**2 + ker.gethyper()[2] - np.sum(inv_cov ** 2, axis=0)
            var_neg = var < 0
            var[var_neg] = 0.
            var=np.sqrt(var)[:,None]

            #normalizer = np.sqrt(2 * np.pi * ker.gethyper()[0] ** 2)
            return mean, var


        else:
            #Normalize Y
            self._mean_y= np.mean(self.get_Y(),axis=0)
            self._std_y=np.std(self.get_Y(), axis=0)
            Y=(self.get_Y()-self._mean_y)/self._std_y

            ker = self.get_kernel()
            K_sample = ker.kernel_product(self.get_X(), X)
            inv_cov = np.linalg.solve(self.get_cov(), K_sample)
            #MEAN
            y_mean = np.dot(inv_cov.T, np.linalg.solve(self.get_cov(), Y))
            #DESTANDARDIZE
            y_mean=(self._std_y*y_mean + self._mean_y)
            #VARIANCE
            var = ker.gethyper()[0] ** 2 + self.get_noise() - np.sum(inv_cov ** 2, axis=0)
            #REMOVE VARIANCE VALUE LESS THAN 0
            var_neg = var < 0
            var[var_neg] = 0.
            #DeStandardize
            var = np.sqrt(var*self._std_y**2)[:, None]

            #normalizer = np.sqrt(2 * np.pi * ker.gethyper()[0] ** 2)
            return y_mean ,  var



    def plot(self, X):
        """
        Function that handles the prediction of the GP object and the Plotting.
        For visual reasons it only works with input data of 1D and 2D
        :param X: Data to predict and plot
        """
        mean, var = self.predict(X)
        dim = self.get_dim_data()
        args = [self.get_X(), self.get_Y(), X, mean, var]
        plt = plot_BayOpt(*args)
        plt.legend()
        plt.show()

    def augment_dataset(self, dataset, new_data):
        """
        General function to augment an array with a shape check
        :param dataset: Dataset to augment
        :param new_data: Data to agument dataset with
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

    def set_marg(self, marg):
        self.__marg = marg

    def set_hyper(self, sigma=None, l=None, noise=None):
        self.get_kernel().sethyper(sigma, l, noise)

    def get_noise(self):
        ker=self.get_kernel()
        return ker.get_noise()

    def save_model(self,path):
        with open(path,"w") as file:
            file.write(str(self))

    def __str__(self):
        header="#=========================GP===========================\n"
        tail="#======================================================"
        X,Y=self.get_X(),self.get_Y()
        if not hasattr(self, "__optimizer"):
            gp_info=f"Optimizer: {self.__optimizer}\n"
        kernel_info=str(self.get_kernel())
        train_info=f'Train values: {self.get_dim_data()}\nInput Dimension: {self.__dim_input} Output Dimension: {self.get_dim_outspace()}\n'
        train_info+=f'\t\t\t\tX_train\t\t\t\tY_train\n'
        train_info+=f'{np.hstack((X,Y))}\n\n'
        gp_info+=f'Noise: {self.get_noise()}\nLog Marginal Likelihood: {self.get_marg()}\n'
        surprise = "\n░░░░░░░░▄▄▄▀▀▀▄▄███▄░░░░░░░░░░░░░░\n░░░░░▄▀▀░░░░░░░▐░▀██▌░░░░░░░░░░░░░\n░░░▄▀░░░░▄▄███░▌▀▀░▀█░░░░░░░░░░░░░\n░░▄█░░▄▀▀▒▒▒▒▒▄▐░░░░█▌░░░░░░░░░░░░\n░▐█▀▄▀▄▄▄▄▀▀▀▀▌░░░░░▐█▄░░░░░░░░░░░\n░▌▄▄▀▀░░░░░░░░▌░░░░▄███████▄░░░░░░\n░░░░░░░░░░░░░▐░░░░▐███████████▄░░░\n░░░░░le░░░░░░░▐░░░░▐█████████████▄\n░░░░toucan░░░░░░▀▄░░░▐█████████████▄\n░░░░░░has░░░░░░░░▀▄▄███████████████\n░░░░░arrived░░░░░░░░░░░░█▀██████░░\n"
        return header+kernel_info+gp_info+train_info+tail+surprise

    def __repr__(self):
        header="#=========================GP===========================\n"
        tail="#======================================================"
        X,Y=self.get_X(),self.get_Y()
        if not hasattr(self, "__optimizer"):
            gp_info=f"Optimizer: {self.__optimizer}\n"
        kernel_info=str(self.get_kernel())
        train_info=f'Train values: {self.get_dim_data()}\nInput Dimension: {self.__dim_input} Output Dimension: {self.get_dim_outspace()}\n'
        train_info+=f'\t\t\t\tX_train\t\t\t\tY_train\n'
        train_info+=f'{np.hstack((X,Y))}\n\n'
        gp_info+=f'Noise: {self.get_noise()}\nLog Marginal Likelihood: {self.get_marg()}\n'
        surprise="░░░░░░░░▄▄▄▀▀▀▄▄███▄░░░░░░░░░░░░░░\n░░░░░▄▀▀░░░░░░░▐░▀██▌░░░░░░░░░░░░░\n░░░▄▀░░░░▄▄███░▌▀▀░▀█░░░░░░░░░░░░░\n░░▄█░░▄▀▀▒▒▒▒▒▄▐░░░░█▌░░░░░░░░░░░░\n░▐█▀▄▀▄▄▄▄▀▀▀▀▌░░░░░▐█▄░░░░░░░░░░░\n░▌▄▄▀▀░░░░░░░░▌░░░░▄███████▄░░░░░░ ░░░░░░░░░░░░░▐░░░░▐███████████▄░░░\n░░░░░le░░░░░░░▐░░░░▐█████████████▄\n░░░░toucan░░░░░░▀▄░░░▐█████████████▄\n░░░░░░has░░░░░░░░▀▄▄███████████████\n░░░░░arrived░░░░░░░░░░░░█▀██████░░\n"
        return header+kernel_info+gp_info+train_info+tail+surprise





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

