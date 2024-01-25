from generators.standard_normal import sample_normal

from model.call_option import call_price

from estimators.control_variate import control_variate_estimator

import numpy as np
from typing import Tuple


# This function runs control_variate estimator for n = 1
# It returns (I, sigma), where
#		I is the estimated value
#		sigma is the calculated variance of the estimator
def run_control_variate(R: int) -> Tuple[float, float]:
		
	X = sample_normal(R)
	Y = np.array([call_price(np.array([B])) for B in X])

	return control_variate_estimator(X, Y, 0)
