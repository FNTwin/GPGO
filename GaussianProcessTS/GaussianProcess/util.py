import numpy as np
import os
import time as tm


def log_gp(X_train, Y_train, lin, mean, var, gp, path=None):
    name = "GP_log.txt"
    if path is not None:
        name=os.path.join(path,name)
    else:
        with open(name, "w") as f:
            f.write(f'#===============LOG of Gaussian Process==============\n\n')
            f.write(str(gp))


def log_bo(bayOpt):
    path=os.path.join(r"C:\Users\Cristian\Desktop\benchmark", str(np.random.randn(1))+"_test.txt")
    f = open(path, "w")
    f.write(str(bayOpt))
    f.close()


def write_data_file():
    pass

def read_data_file():
    pass

class time_log():

    def __init__(self, start=tm.time()):
        self.start=start
        self.intervall=[start]
        self.end=None
        self.count=0


    def time(self, t=tm.time()):
        self.intervall.append(t-self.intervall[self.count])
        self.count+=1

    def time_end(self, t=tm.time()):
        self.time(t=t)
        self.end=t

    def total(self):
        if self.end is None:
            self.time_end(tm.time())
        return str(self.end-self.start)

    def __str__(self):
        start=f'{str(self.end-self.start)}\n'
        return start