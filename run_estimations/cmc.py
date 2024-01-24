from model.call_option import call_price

from generators.brownian_general import sample_trajectory

from estimators.crude_monte_carlo import cmc_estimator

import numpy as np
from typing import Tuple


# This function runs the Crude Monte Carlo estimator for call option
# R = [R_1, ..., R_m] is the number of observations expected for each stratum.
# returns (I, sigma), where
#		I is the estimated result
#		sigma is the computed variance of the estimator
def run_cmc(n: int, R: int) -> Tuple[float, float]:

	Y = np.zeros(R)

	for i in range(R):
		trajectory = sample_trajectory(n)
		Y[i] = call_price(trajectory)

	return cmc_estimator(Y)
