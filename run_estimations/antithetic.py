from model.call_option import call_price

from generators.antithetic import sample_normal_antithetic

from estimators.antithetic_estimator import antithetic_estimator
from estimators.variance_estimators import variance_estimator, covariance_estimator

import numpy as np
import numpy.typing as npt
from typing import Tuple


# This functions runs the antithetic estimator for n = 1.
# It returns (I, sigma), where
#		I is the estimated value
#		sigma is the calculated variance of the estimator
def run_antithetic(R: int) -> Tuple[float, float]:

	trajectories = sample_normal_antithetic(R)
	Y1 = np.array([call_price(np.array([trajectory])) for trajectory in trajectories[::2]])
	Y2 = np.array([call_price(np.array([trajectory])) for trajectory in trajectories[1::2]])

	cov = covariance_estimator(Y1, Y2)
	var1 = variance_estimator(Y1)
	var2 = variance_estimator(Y2)

	corr = cov / np.sqrt(var1 * var2)

	result, sigma_cmc = antithetic_estimator(np.append(Y1, Y2))

	sigma = sigma_cmc * (1 - corr)

	return result, sigma
