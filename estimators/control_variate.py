import numpy as np
import numpy.typing as npt

from estimators.crude_monte_carlo import cmc_estimator
from estimators.variance_estimators import covariance_estimator, variance_estimator


# control variate estimator
# X = [X_1, X_2, ..., X_n], Y = [Y_1, Y_2, ..., Y_n], where (X_1, Y_1), ..., (X_n, Y_n) are i.i.d. pairs
# EX – expected value of X should be known
def control_variate_estimator(X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64], EX: float) -> float:
	covXY = covariance_estimator(X, Y)
	varX = variance_estimator(X)
	c = -covXY / varX
	return cmc_estimator(Y) + c * (cmc_estimator(X) - EX)
