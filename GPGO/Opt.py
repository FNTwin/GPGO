import copy
import logging
import numpy as np
from .GaussianProcess import GP, generate_grid, time_log, plot_BayOpt, Observer
from scipy.optimize import minimize
from .Acquisition import Acquistion

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BayesianOptimization():
    """
    Bayesian Optimization class based on a Gaussian Process surrogate model
    ...

    Attributes:
    -----------
        dim_input : int
            Dimension of the input space
        dim_output : int
            Dimension of the output space
        dimension : int
            Number of the starting Training Points for the optimization
        X_train : np.array
            Starting training sample points written as column vector. 5 Training points of 2 dimensions -> shape(5,2)
                                                             1 Training point of 1 dimensions -> shape(1,1)
        Y_train : np.array
            Starting training sample points written as column vector. 5 Training points of 2 dimensions -> shape(5,2)
                                                            1 Training point of 1 dimensions -> shape(1,1)
        settings : dict (default None)
            Dictionary with the setting needed for the Bayesian Optimization
        GP : GP object (default None)
            Gaussian Process object as documented in GP.py
        func : callable (default None)
            Function to call to evaluate new points
        _err : float (default 1e-4)
            Error value to check between the sampling of the proposal
        _it : int (default None)
            Number of iterations utilized in the optimization run
        _time_logger : time_log object
            Handler for the time measurement
        _helper : Observer object
            Handler for the observations and the convergence plots
        _old_dataset : list
            List containing the starting training dataset
        _plot : bool
            Only for 1 Dimensional Bayesian Optimization.
            If True it plots the EI function and the GP model at each iteration step. Use set_plotter to abilitate it.

    Settings
    --------
        The settings in this object are passed by a dictionary with the following keys:
            "type": str
                Set the type of optimization for the Acquisition Function. Right now there are 3 types of methods:
                "DIRECT" , The Bayesian Optimization utilizes the Dividing Rectangle Technique to maximize the
                           Acquisition Function
                "BFGS" , The Bayesian Optimization utilizes the L-BFGS-B optimizer to maximize the Acquisition Function
                "NAIVE" , The Bayesian Optimization utilizes some sampling techniques to calculate the Acquisition
                          Function maxima
            "ac_type": str
                Set the acquisition function. Right now there are 3 types of methods:
                "EI" , Expected Improvement
                "UCB" , Upper Confidence Bound (if Minimization is True the LCB will be calculated)
            "n_search": int
                Number of search of the optimizer or in the NAIVE case and for the NAIVE optimization
                the number of sampling points generated per dimension
            "boundaries": list
                List of bounderies compatible with the input shape.
                Ex: X: shape(3,6) , Boundaries=[[0,1] for i in range(6)]
            "epsilon": float (standard should be 0.01)
                Exploration-Exploitation trade off value
            "iteration": int
                Number of iteration of the optimization
            "minimization": bool
                Flag for choosing the minimization or maximization
            "optimization": bool
                Flag for the optimization of the Gaussian Process model
            "n_restart": int
                Number of restart in the hyperparameters of the Gaussian Process optimization routine
            "sampling": callable or str
                Sampling methods of the space: np.random.uniform
                                               np.linspace
                                               "LHS" : Quasi Random Latin Hypercube Sampling

    Example
    ________
        X=np.random.uniform(0,3,5)[:,None]
        Y=np.sin(x)
        GaussProcess=GP(X,Y)
        f=np.sin
        settings={"type":"NAIVE",
                  "ac_type": "EI",
                  "n_search": 1000,
                  "boundaries": [[0,3]],
                  "epsilon": 0.01,
                  "iteration": 10,
                  "minimization":True,
                  "optimization":True,
                  "n_restart":10,
                  "sampling":np.linspace}
        BayOpt=GPGO(X,Y,settings,GaussProcess,f)
        BayOpt.set_plotter()
        BayOpt.run()
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, settings=None, GP=None, func=None, err=1e-4):
        self.dim_input = X[0].shape[0]
        self.dim_output = Y[0].shape[0]
        self.dimension = X.shape[0]
        self.X_train = X
        self.Y_train = Y
        self.settings = settings
        self.GP = GP
        self.func = func
        self._err = err
        self._it = None
        self._time_logger = time_log()
        self._helper = Observer(self.get_info("type"),self.get_info("minimization"))
        self._old_dataset = [X, Y]

    def run(self):
        """
        Handler for running an optimization task
        :return: [X_array,Y_array] of the optimization
        """
        # print(self.settings)
        try:
            if self.settings is not None:
                copied_settings = copy.copy(self.settings)
                bay_opt_methods = {"DIRECT": self.direct,
                                   "BFGS": self.bfgs,
                                   "NAIVE": self.naive}

                self._optimizer = bay_opt_methods[self.get_info("type")]
                self._helper.type = self.get_info("type")

                del copied_settings["type"]
                return self.bayesian_run(**copied_settings)
            else:
                logger.warning("Settings not specified")
                raise ValueError
        except BaseException as exc:
            logger.warning("Error on inizialing the Bayesian Optimization\n")
            raise exc

    def suggest_location(self):
        """
        Method to suggest the new sample points without calling the evaluation routine
        :return: np.array
            Proposal for the new set of points to sample the function
        It create a shallow copy of the GPGO object with a _no_evaluation flag that will be eliminated
        at the end of the run
        """
        # self._no_evaluation=True
        # self.settings["iteration"]=1
        tmp = copy.copy(self)
        tmp._no_evaluation = True
        proposal = tmp.run()
        del tmp
        return proposal

    def naive(self, n_search, boundaries, sampling, grid_bounds):
        dim = self.get_dim_inputspace()
        if sampling == "LHS":
            try:
                from smt.sampling_methods import LHS
                lhs_generator = LHS(xlimits=boundaries)
                search_grid = lhs_generator(n_search)
            except ImportError as exc:
                raise ImportError("To use the Latin Hypercube Sampling install the smt python package\n",exc)
        else:
            search_grid = generate_grid(dim, n_search, grid_bounds, sampling)

        if self.get_info("minimization"):
            best = np.min(self.get_Y())
        else:
            best = np.max(self.get_Y())

        improvement = self._acquistion.call(search_grid, best=best)
        new_prediction = search_grid[np.argmax(improvement)]
        print(new_prediction)

        if hasattr(self, "_plot"):
            mean, variance = self.get_GP().predict(search_grid)
            args = [self.get_X(), self.get_Y(), search_grid, mean, variance, improvement, new_prediction]
            plot_BayOpt(*args)

        return np.atleast_2d(new_prediction)

    def bfgs(self, boundaries, n_search):
        dim = self.get_dim_inputspace()
        min_val = None
        if self.get_info("minimization"):
            best = np.min(self.get_Y())
        else:
            best = np.max(self.get_Y())
        # improvement = EI_func(n_search_points, max, gp, epsilon)
        min_x = None
        c=0
        for i in np.random.uniform(boundaries[:, 0], boundaries[:, 1], size=(n_search, dim)):
            c+=1
            logging.info("RESTART: %s",c)
            res = minimize(lambda X: -self._acquistion.call(np.atleast_2d(X), best=best),
                           x0=i, bounds=boundaries, method='L-BFGS-B')
            if not res.success:
                continue
            if self.get_info("minimization"):
                if min_val is None or res.fun[0] < min_val:
                    min_val = res.fun[0]
                    min_x = res.x
            else:
                if min_val is None or res.fun[0] > min_val:
                    min_val = res.fun[0]
                    min_x = res.x

        return np.atleast_2d(min_x)

    def direct(self, boundaries, max_iter=3000):
        try:
            from DIRECT import solve
        except ImportError as exc:
            raise ImportError("To use the DIRECT optimization install the Python DIRECT wrapper\n", exc)
        def wrapper(f):
            def g(x, user_data):
                return -f(np.array([x]), user_data), 0

            return g

        if self.get_info("minimization"):
            best = np.min(self.get_Y())
        else:
            best = np.max(self.get_Y())
        lb = boundaries[:, 0]
        ub = boundaries[:, 1]
        # maxf= 80000
        #max_iter=6000,1000
        x, val, _ = solve(wrapper(self._acquistion.call), lb, ub, maxT=max_iter, user_data=best, algmethod=1)
        logger.info("DIRECT:", x, val)
        return np.atleast_2d(x)

    def next_sample_validation(self, new_sample, boundaries):
        if np.any(np.sqrt(np.sum((self.get_X() - new_sample) ** 2, axis=1)) < self.get_err()):
            logger.debug("----------Evaluations too close---------\n", new_sample, new_sample.shape[0])
            return np.random.uniform(boundaries[:, 0], boundaries[:, 1], (1, self.get_dim_inputspace()))
        else:
            return new_sample

    def compute_new_sample(self, new_sample):
        func = self.get_func()
        try:
            sample_Y = func(new_sample)

        except:
            sample_Y = func(np.expand_dims(new_sample, axis=0))

        return sample_Y

    '#===================================Bayesian RUN======================================='

    def bayesian_run(self,
                     ac_type,
                     n_search,
                     boundaries,
                     iteration=10,
                     epsilon=0.01,
                     minimization=True,
                     optimization=False,
                     n_restart=100,
                     sampling=np.random.uniform):

        if GP is None:
            raise ValueError("Gaussian Process not existing. Define one before running a " +
                             "Bayesian Optimization")
        else:
            gp = self.get_GP()
            dim = self.get_dim_inputspace()
            tm = self.get_time_logger()
            boundaries_array = np.asarray(boundaries)
            self._acquistion = Acquistion(ac_type, gp, epsilon, minimization)
            kwargs = {"type": self.get_info("type"),
                      "n_search": n_search,
                      "boundaries": boundaries_array,
                      "sampling": sampling,
                      "grid_bounds": boundaries}

            for i in range(1, iteration + 1):
                logger.info("Iteration: ", i)
                gp.fit()
                if optimization:
                    gp.optimize(n_restarts=n_restart)
                    logger.info("Optimization: ", i, " completed")

                # Compute the EI and the new theoretical best
                # Choose type of optimizer
                # predicted_best_X = self._optimizer(boundaries_array,n_search)
                predicted_best_X = self._optimizer(*BayesianOptimization.kwargs_check(**kwargs))
                # Check if it is a duplicate
                predicted_best_X = self.next_sample_validation(predicted_best_X, boundaries_array)

                if hasattr(self, "_no_evaluation"):
                    return predicted_best_X
                else:
                    predicted_best_Y = self.compute_new_sample(predicted_best_X)
                    # Pass the result to the loggers
                    self.observe(predicted_best_Y, predicted_best_X)
                    logger.info(i, "It: ", predicted_best_X, " , Y: ", predicted_best_Y)

                    # Augment the dataset of the BO and the GP objects
                    self.augment_XY(predicted_best_X, predicted_best_Y)
                    gp.augment_XY(predicted_best_X, predicted_best_Y)

            if minimization:
                best_index = np.argmin(self.get_Y())
            else:
                best_index = np.argmax(self.get_Y())
            self.observer_plot()
            return self.get_X()[best_index], self.get_Y()[best_index]

    '#=====================================UTILITIES============================================'

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
        self.X_train = self.augment_dataset(self.get_X(), new_data)

    def augment_Y(self, new_data):
        self.Y_train = self.augment_dataset(self.get_Y(), new_data)

    def augment_XY(self, new_data_X, new_data_Y):
        self.augment_X(new_data_X)
        self.augment_Y(new_data_Y)

    @staticmethod
    def kwargs_check(**kwargs):
        if kwargs["type"] == "NAIVE":
            return kwargs["n_search"], kwargs["boundaries"], kwargs["sampling"], kwargs["grid_bounds"]
        if kwargs["type"] == "BFGS":
            return kwargs["boundaries"], kwargs["n_search"]
        if kwargs["type"] == "DIRECT":
            return [kwargs["boundaries"]]

    def set_func(self, func):
        self.func = func

    def get_X(self):
        return self.X_train

    def get_Y(self):
        return self.Y_train

    def get_GP(self):
        return self.GP

    def get_func(self):
        return self.func

    def get_dim_data(self):
        return self.dimension

    def get_dim_inputspace(self):
        return self.dim_input

    def get_dim_outspace(self):
        return self.dim_output

    def get_time_logger(self):
        return self._time_logger

    def get_err(self):
        return self._err

    def get_info(self, key):
        return self.settings[key]

    def set_err(self, err):
        self._err = err

    def set_plotter(self):
        self._plot = True

    def observe(self, value, propose):
        self._helper.observe(value, propose)

    def observer_plot(self):
        self._helper.plot()

    def plot_convergence(self):
        # TODO
        try:
            self._helper.plot_conv()
        except:
            logger.warning("No Optimization task done")

    def __str__(self):
        header = "============================================================================\n"
        old_data = f'Bayesian Run initialized with: {self._it} iterations\nDATASET\n{self._old_dataset}\n'
        old_data += "----------------------------------------------------------------------------\n\n"
        time = f'Time: {self.get_time_logger().total()}\n'
        time += "Acquisition function optimizer: " + str(self._helper) + "\n\n"
        count = 0
        datanew = f'Iteration data\n\n'
        X, Y = self.get_X(), self.get_Y()
        for i in range(X.shape[0]):
            count += 1
            datanew += f'{count}\t{X[i]}\t{Y[i]}\n'

        return header + old_data + time + datanew + header + str(self.get_GP())
