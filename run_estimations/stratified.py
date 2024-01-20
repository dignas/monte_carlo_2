from generators.brownian_general import sample_trajectory

from model.call_option import call_price

from estimators.crude_monte_carlo import cmc_estimator
from estimators.stratified_estimator import stratified_estimator
from estimators.variance_estimators import variance_estimator

import numpy as np
import numpy.typing as npt
from typing import Tuple


# This function runs the stratified testing for call option. m is the number of equally probable strata.
# R = [R_1, ..., R_m] is the number of observations expected for each strata.
# returns (I, sigma), where
#		I = [I_1, ..., I_no_tests] are the results in consecutive tests
#		sigma = [sigma_1, ..., sigma_m] are empirical variance for each of the strata
def run_stratified(no_tests: int, n: int, R: npt.NDArray[np.int32], m: int) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

	results = np.zeros(no_tests)
	sigma = np.zeros(m)
	p = np.full(m, 1/m)

	for t in range(no_tests):

		I = np.zeros(m)

		for i in range(m):

			Y = np.zeros(R[i])

			for r in range(R[i]):

				trajectory = sample_trajectory(n, m, i + 1)
				Y[r] = call_price(trajectory)

			I[i] = cmc_estimator(Y)
			sigma[i] = variance_estimator(Y)

		results[t] = stratified_estimator(p, I)

	return results, sigma
