from model.call_option import call_price

from generators.brownian_general import sample_trajectory

from estimators.crude_monte_carlo import cmc_estimator

import numpy as np
import numpy.typing as npt


def run_cmc(no_tests: int, n: int, R: int) -> npt.NDArray[np.float64]:

	results = np.zeros(no_tests)

	for t in range(no_tests):

		Y = np.zeros(R)

		for i in range(R):
			trajectory = sample_trajectory(n)
			Y[i] = call_price(trajectory)

		results[t] = cmc_estimator(Y)

	return results
