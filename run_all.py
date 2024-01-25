from run_estimations.n1 import run_n_1
from run_estimations.ngeq2 import run_n_geq_2

import numpy.random as np_rand


if __name__ == "__main__":
	np_rand.seed(3)
	run_n_1()
	run_n_geq_2(2)
	run_n_geq_2(10)
