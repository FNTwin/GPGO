from GaussianProcessTS.GaussianProcess.GP import GP, generate_grid
from GaussianProcessTS.GaussianProcess.Kernel.RBF import RBF
from GaussianProcessTS.Opt import BayesianOptimization
import numpy as np
import os
import csv

"""
I have to read a log file to get the RDF, process the RDF with my loss function to get my Y ---> write on utile
Then I write this Y on the db file where I store all the different coefficients. ---> .csv?
Read the file so I have the data. Initialize the gp with those data. ---> I can just write on util, kernel will be RBF
Run a single bo that return the new sample ----> bayesian_run_min_single
Store that sample (X) on the db file ---> I can just write on util
return a formatted file for lammps.in ---> .txt file
"""

truth_table=[False, False, True, True, True, False, True , True,
             False, True, False, False, False, False , False,
             True, True, True, False, True , True,
             True, True, False, True , True,
             True, False, True, True,
             True, False, False,
             False, False,
             False]

bounds= ["! AuE\t\tAuE", "! AuE\t\tAuI","! AuE\t\tC", "! AuE\t\tN", "! AuE\t\tL", "! AuE\t\tS", "! AuE\t\tW",
         "! AuE\t\tCl", "! AuI\t\tAuI", "! AuI\t\tC", "! AuI\t\tN", "! AuI\t\tL", "! AuI\t\tS", "! AuI\t\tW",
         "! AuI\t\tCl", "! C\t\t\tC", "! C\t\t\tN", "! C\t\t\tL", "! C\t\t\tS", "! C\t\t\tW", "! C\t\t\tCl",
         "! N\t\t\tN", "! N\t\t\tL", "! N\t\t\tS", "! N\t\t\tW", "! N\t\t\tCl", "! L\t\t\tL", "! L\t\t\tS",
         "! L\t\t\tW", "! L\t\t\tCl", "! S\t\t\tS", "! S\t\t\tW", "! S\t\t\tCl", "! W\t\t\tW", "! W\t\t\tCl",
         "! Cl\t\tCl"]

def read_interaction(path):
    """Read the dpd interaction file and return the array of the interactions parameters"""
    with open(path,"r") as f:
        coeffs=[]
        for row in f:
            a=row.split()
            if "!" in a:
                coeffs.append(float(a[3]))
        return np.array(coeffs)

def write_interaction(path, array):
    "write an interaction file in path"
    path=os.path.join(path,"interactions_test.txt")
    with open(path,"w") as f:
        for i in range(len(bounds)):
            f.write(f'{bounds[i]}\t\t{np.squeeze(array[i])}\t\t{4.5}\n')

def write_db(path, array):
    with open(path, "ab") as f:
        np.savetxt(f, array, fmt="%2.3f", header="Iteration 2")

def read_db(path):
    return np.loadtxt(path)

def process_info_rdf():
    """Load RDF file, process it, write_db for y"""
    pass

def process():
    """read db, run bayopt, process new sample to db and write interaction"""

import time

import matplotlib.pyplot as plt



def test_dim(n_p, n_train, dim, min=False):
    def f(X):
        return np.sin(np.sum(X**2, axis=1))
    best=[]
    tm=[]
    for i in range(dim):
        a = time.time()

        boundaries = [[-3, 3] for j in range(i+1)]
        x = np.random.uniform(-3, 3, (n_train, i+1))
        y = f(x)[:,None]

        gp = GP(np.float32(x), np.float32(y) , noise=0.05)
        BayOpt = BayesianOptimization(x, y, gp, f)

        best_tmp = BayOpt.bayesian_run_min(n_p,
                                   boundaries,
                                   iteration=4,
                                   optimization=min,
                                   func=np.random.uniform)

        best.append(best_tmp)
        tm.append(time.time()-a)
        print("Dimension: ", i+1)
        print("Number of points sampled in an iteration: ", n_p ** (i+1))
        print("bay:", best_tmp)
        print("--------------------------------------\n")

    print("BEST:", best)
    return best,tm

arr=test_dim(5, 100 , 10)
arr2=test_dim(3, 100 , 10)
plt.scatter(np.arange(1,11,1),arr[1], marker="x", color="black")
plt.plot(np.arange(1,11,1),arr[1], color="blue", alpha=0.7 , label="Number of points : 3")
plt.scatter(np.arange(1,11,1),arr2[1], marker="x", color="black")
plt.plot(np.arange(1,11,1),arr2[1], color="red", alpha=0.7 , label="Number of points : 2")
plt.fill_between(np.arange(1,14,1), arr[1], arr2[1],
                       color="red", alpha=0.2)
plt.xlabel('Dimensions', fontsize=18)
plt.ylabel('Time', fontsize=16)
plt.legend()
plt.show()

