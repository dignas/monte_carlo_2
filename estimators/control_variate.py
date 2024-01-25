from estimators.crude_monte_carlo import cmc_estimator
from estimators.variance_estimators import covariance_estimator, variance_estimator

import numpy as np
import numpy.typing as npt
from typing import Tuple


# control variate estimator
# X = [X_1, X_2, ..., X_n], Y = [Y_1, Y_2, ..., Y_n], where (X_1, Y_1), ..., (X_n, Y_n) are i.i.d. pairs
# EX – expected value of X should be known
def control_variate_estimator(X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64], EX: float) -> Tuple[float, float]:
	Y_cmc, var_Y_cmc = cmc_estimator(Y)
	X_cmc, _ = cmc_estimator(X)
	varY = variance_estimator(Y)
	varX = variance_estimator(X)
	covXY = covariance_estimator(X, Y)
	c = -covXY / varX
	corr = covXY / np.sqrt(varX * varY)
	return Y_cmc + c * (X_cmc - EX), var_Y_cmc * (1 - corr**2)
