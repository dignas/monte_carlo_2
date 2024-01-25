from generators.brownian_general import sample_trajectory

from model.call_option import call_price

from estimators.crude_monte_carlo import cmc_estimator
from estimators.stratified_estimator import stratified_estimator
from estimators.variance_estimators import variance_estimator

import numpy as np
import numpy.typing as npt
from typing import Tuple


# This function runs the stratified testing for call option. m is the number of equally probable strata.
# R = [R_1, ..., R_m] is the number of observations expected for each stratum.
# returns (I, sigma, strata_sigma), where
#		I is the estimated result
#		sigma is the computed variance of the estimator
#		strata_sigma = [s_1, ..., s_m] is the computed variance in each stratum
def run_stratified(n: int, R: npt.NDArray[np.int32], m: int) -> Tuple[float, float, npt.NDArray[np.float64]]:

	strata_sigma = np.zeros(m)
	p = np.full(m, 1/m)

	I = np.zeros(m)

	for i in range(m):

		Y = np.zeros(R[i])

		for r in range(R[i]):

			trajectory = sample_trajectory(n, m, i + 1)
			Y[r] = call_price(trajectory)

		I[i], _ = cmc_estimator(Y)
		strata_sigma[i] = variance_estimator(Y)

	result = stratified_estimator(p, I)

	sigma = np.sum(np.square(p) / R * strata_sigma)

	return result, sigma, strata_sigma
