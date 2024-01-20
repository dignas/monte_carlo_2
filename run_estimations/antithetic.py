from model.call_option import call_price

from generators.antithetic import sample_normal_antithetic

from estimators.antithetic_estimator import antithetic_estimator

import numpy as np
import numpy.typing as npt


# This functions runs the antithetic estimator for n = 1.
def run_antithetic(no_tests: int, R: int) -> npt.NDArray[np.float64]:

	results = np.zeros(no_tests)

	for t in range(no_tests):

		Y = np.zeros(R)
		
		trajectories = sample_normal_antithetic(R)
		Y = np.array([call_price(np.array(trajectory)) for trajectory in trajectories])

		results[t] = antithetic_estimator(Y)

	return results
