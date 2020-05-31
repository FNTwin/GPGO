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

bounds_index=["2\t\t2","1\t\t2","2\t\t3","2\t\t5","2\t\t4","2\t\t6","2\t\t7","2\t\t8","1\t\t1","1\t\t3","1\t\t5",
         "1\t\t4","1\t\t6","1\t\t7","1\t\t8","3\t\t3","3\t\t5","3\t\t4","3\t\t6","3\t\t7","3\t\t8","5\t\t5","4\t\t5",
         "5\t\t6","5\t\t7","5\t\t8","4\t\t4","4\t\t6","4\t\t7","4\t\t8","6\t\t6","6\t\t7","6\t\t8","7\t\t7","7\t\t8",
         "8\t\t8"]

def get_right_coeffs(array):
    truth_index=[2,3,6,7,9,15,16,19,20,21,24,25,30]
    return np.atleast_2d(np.squeeze(array)[truth_index])


def fill_spots(array):
    truth_index = [2, 3, 6, 7, 9, 15, 16, 19, 20, 21, 24, 25, 30]
    copy_to = [4, 10, 11, 13, 14, 17, 22, 26, 28, 29]
    copy_from = [2, 3, 2, 6, 7, 15, 16, 15, 19, 21]
    coeffs=np.zeros(36)
    coeffs[truth_index]=np.squeeze(array)
    coeffs[copy_to] = coeffs[copy_from]
    return np.atleast_2d(coeffs)


def read_interaction(path):
    """Read the dpd interaction file and return the array of the interactions parameters"""
    path=os.path.join(path, "full_G11_326N16.solv.inputinteg.txt")
    with open(path,"r") as f:
        coeffs=[]
        for row in f:
            a=row.split()
            if "pair_coeffs" in a:
                coeffs.append(float(a[3]))
        return np.atleast_2d(coeffs)

def write_interaction(path, array):
    #Array will be less than 36 bounds
    bounds = ["! AuE\t\tAuE", "! AuE\t\tAuI", "! AuE\t\tC", "! AuE\t\tN", "! AuE\t\tL", "! AuE\t\tS", "! AuE\t\tW",
              "! AuE\t\tCl", "! AuI\t\tAuI", "! AuI\t\tC", "! AuI\t\tN", "! AuI\t\tL", "! AuI\t\tS", "! AuI\t\tW",
              "! AuI\t\tCl", "! C\t\tC", "! C\t\tN", "! C\t\tL", "! C\t\tS", "! C\t\tW", "! C\t\tCl",
              "! N\t\tN", "! N\t\tL", "! N\t\tS", "! N\t\tW", "! N\t\tCl", "! L\t\tL", "! L\t\tS",
              "! L\t\tW", "! L\t\tCl", "! S\t\tS", "! S\t\tW", "! S\t\tCl", "! W\t\tW", "! W\t\tCl",
              "! Cl\t\tCl"]

    bounds_index = ["2\t2", "1\t2", "2\t3", "2\t5", "2\t4", "2\t6", "2\t7", "2\t8", "1\t1", "1\t3",
                    "1\t5",
                    "1\t4", "1\t6", "1\t7", "1\t8", "3\t3", "3\t5", "3\t4", "3\t6", "3\t7", "3\t8",
                    "5\t5", "4\t5",

                    "5\t6", "5\t7", "5\t8", "4\t4", "4\t6", "4\t7", "4\t8", "6\t6", "6\t7", "6\t8",
                    "7\t7", "7\t8",
                    "8\t8"]

    bound_mask=[0,1,5,8,12,18,23,27,31,32,33,34,35]
    mask_value=[51.6, 51.6, -10., 51.6 , 40., 72.,68.9,72., 80.,80.,51.6,51.6, 51.6]
    n_bounds=36
    n_real_bounds=13
    array=np.squeeze(array)

    "write an interaction file in path"
    path=os.path.join(path,"full_G11_326N16.solv.inputinteg.txt")
    with open(path,"w") as f:
        f.write("\n# Atom Types used: AuE: 2, AuI: 1, C: 3, Cl: 8, L: 4, N: 5, S: 6, W: 7, \n\n")
        f.write("# pair_coeff, to be imported in the lammps input file...\n")
        for i in range(len(bounds)):
            if i in bound_mask:
                f.write(f'pair_coeffs\t{bounds_index[i]}\t{mask_value[bound_mask.index(i)]:.4f}\t\t{4.5:.4f}\t#{bounds[i]}\n')
            else:
                f.write(f'pair_coeffs\t{bounds_index[i]}\t{np.squeeze(array[i]):.4f}\t\t{4.5:.4f}\t#{bounds[i]}\n')

def write_db(path, array):
    with open(path, "ab") as f:
        np.savetxt(f, array, fmt="%2.3f", header="#----------------------")

def read_db(path):
    return np.atleast_2d(np.loadtxt(path))

def process_info_rdf(a1,a2):
    """Load RDF file, process it, write_db for y"""

    loss= np.linalg.norm
    pass

def main(path_interaction, path_x, path_y):
    #read rdf and calculate Y
    new_Y=np.random.randint(-10,3, (1,1))
    #Write X and Y
    X_old=get_right_coeffs(read_interaction(path_interaction))
    write_db(path_x, X_old)
    write_db(path_y, new_Y)

    X,Y=read_db(path_x), read_db(path_y).reshape(-1,1)
    print(X)
    print(Y)
    #bo run
    boundaries=[[-10,140] for i in range(X.shape[1])]
    gp = GP(X, Y, noise=0.02)
    gp.fit()

    bayesian_optimizer=BayesianOptimization(X,Y,gp, err=1e-4)
    proposal=bayesian_optimizer.bayesian_run_single(3,
                                                    boundaries,
                                                    optimization=False,
                                                    minimization=True,
                                                    epsilon=0.01,
                                                    opt_constrain=[[2, 100], [1, 50]],
                                                    n_opt_points=200,
                                                    func=np.random.uniform)

    #Write new file
    write_interaction(path_interaction, fill_spots(proposal))


#write_interaction("/home/merk/Desktop/", np.random.randint(0,90,size=len(bounds_index)))
main("/home/merk/Desktop/",
     "/home/merk/Desktop/test.txt",
     "/home/merk/Desktop/test (copy).txt")