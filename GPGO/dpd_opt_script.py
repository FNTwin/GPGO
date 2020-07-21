from GPGO.GaussianProcess.GP import GP, generate_grid
from GPGO.GaussianProcess.Kernel.RBF import RBF
from GPGO.Opt import BayesianOptimization
import numpy as np
import os
import argparse


def get_right_coeffs(array):
    # truth_index=[2,3,6,7,9,15,16,19,20,21,24,25,30]
    truth_index = [2, 6, 7, 9, 15, 19, 20, 30]

    return np.atleast_2d(np.squeeze(array)[truth_index])


def fill_spots(array):
    # truth_index = [2, 3, 6, 7, 9, 15, 16, 19, 20, 21, 24, 25, 30]
    truth_index = [2, 6, 7, 9, 15, 19, 20, 30]

    copy_to = [4, 10, 11, 13, 14, 17, 22, 26, 28, 29]
    copy_from = [2, 3, 2, 6, 7, 15, 16, 15, 19, 20]
    coeffs = np.zeros(36)
    coeffs[truth_index] = np.squeeze(array)
    N = [3, 16, 21, 24, 25]
    coeffs[N] = np.array([127.19, 2.51, -4.3, 124.4, 4.5])
    coeffs[copy_to] = coeffs[copy_from]

    return np.atleast_2d(coeffs)


def read_interaction(path):
    """Read the dpd interaction file and return the array of the interactions parameters"""
    path = os.path.join(path, "full_G11_326N16.solv.inputinteg.txt")
    with open(path, "r") as f:
        coeffs = []
        for row in f:
            a = row.split()
            if "pair_coeffs" in a:
                coeffs.append(float(a[3]))
        return np.atleast_2d(coeffs)


def write_interaction(path, array):
    # Array will be less than 36 bounds
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

    # bound_mask=[0,1,5,8,12,18,23,27,31,32,33,34,35]
    # mask_value=[51.6, 51.6, -10., 51.6 , 40., 72.,68.9,72., 80.,80.,51.6,51.6, 51.6]
    # N beads fixed
    # bound_mask=[0, 1, 3, 5, 8, 12, 16, 18, 21, 23,24,25 ,27,31,32,33,34,35]
    # mask_value=[51.6, 51.6, 127.19, -10., 51.6 , 40.,2.5, 72.,-4.3,68.9,124.4,4.53,72., 80.,80.,51.6,51.6, 51.6]

    bound_mask = [0, 1, 3, 5, 8, 12, 16, 18, 21, 23, 24, 25, 27, 31, 32, 33, 34, 35]
    mask_value = [51.6, 51.6, 127.19, -10., 51.6, 40., 2.5, 72., -4.3, 68.9, 124.4, 4.53, 72., 80., 80., 51.6, 51.6,
                  51.6]
    n_bounds = 36
    # n_real_bounds=13
    n_real_bounds = 8
    array = np.squeeze(array)

    "write an interaction file in path"
    path = os.path.join(path, "full_G11_326N16.solv.inputinteg.txt")
    with open(path, "w") as f:
        f.write("\n# Atom Types used: AuE: 2, AuI: 1, C: 3, Cl: 8, L: 4, N: 5, S: 6, W: 7, \n\n")
        f.write("# pair_coeff, to be imported in the lammps input file...\n")
        for i in range(len(bounds)):
            if i in bound_mask:
                f.write(
                    f'pair_coeff\t{bounds_index[i]}\tdpd\t{mask_value[bound_mask.index(i)]:.4f}\t\t{4.5:.4f}\t#{bounds[i]}\n')
            else:
                f.write(f'pair_coeff\t{bounds_index[i]}\tdpd\t{np.squeeze(array[i]):.4f}\t\t{4.5:.4f}\t#{bounds[i]}\n')


def write_db(path, array):
    with open(path, "ab") as f:
        np.savetxt(f, array, fmt="%2.3f", header="#----------------------")


def read_db(path):
    return np.atleast_2d(np.loadtxt(path))


def parse_cmd():
    """
    Function that parse the input from command line.
    Read three flags: -f , -o, -c
    -f: path to the input file [Required]
    -o: Path to the output file [Optional]
    Output: args object containing the input path, the outputh path and the dictionary of the charges
    """

    parser = argparse.ArgumentParser(description="Prepare the lammps pair coeffs")

    parser.add_argument('-f', '--file', dest='file',
                        action='store', type=str, help="Path to input fie")

    args = parser.parse_args()

    return args


def main():
    args = parse_cmd()
    path_interaction = args.file
    # Write X and Y
    # X_old=get_right_coeffs(read_interaction(path_interaction))
    path_x = "/home/merk/Desktop/optimization_run/data_X.txt"
    path_y = "/home/merk/Desktop/optimization_run/data_Y.txt"

    X, Y = read_db(path_x), read_db(path_y).reshape(-1, 1)
    print(X.shape)
    tmp = []
    for i in X:
        tmp.append(get_right_coeffs(i))
    X = np.asarray(np.squeeze(tmp))
    dim = X[0].shape[0]
    print(X.shape)
    # bo run
    # mean, var=np.mean(X), np.std(X)
    # X= (X - mean)/var
    # low, up =(-10-mean)/var , (140 - mean)/var
    boundaries = [[-10, 140] for i in range(dim)]
    gp = GP(X, Y, RBF(), normalize_y=True)
    settings = {"type": "BFGS",
                "ac_type": "EI",
                "n_search": 100,
                "boundaries": boundaries,
                "epsilon": 0.1,
                "iteration": 1,
                "minimization": True,
                "optimization": True,
                "n_restart": 30,
                "sampling": np.linspace}

    BayOpt = BayesianOptimization(X, Y, settings, gp, func=None)
    proposal = BayOpt.suggest_location()

    # Write new file
    # proposal= proposal *var + mean

    print(proposal)
    write_interaction(path_interaction, fill_spots(proposal))


if __name__ == "__main__":
    main()
