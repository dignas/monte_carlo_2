from generators.standard_normal import sample_normal

from model.call_option import call_price

from estimators.control_variate import control_variate_estimator

import numpy as np
import numpy.typing as npt


# This function runs no_test tests of control_variate estimator for n = 1
def run_control_variate(no_tests: int, R: int) -> npt.NDArray[np.float64]:

	results = np.zeros(no_tests)

	for t in range(no_tests):
		
		X = sample_normal(R)
		Y = np.array([call_price(np.array(B)) for B in X])

		results[t] = control_variate_estimator(X, Y, 0)

	return results
