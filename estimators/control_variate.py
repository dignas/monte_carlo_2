import numpy as np
import numpy.typing as npt

from estimators.crude_monte_carlo import crude_monte_carlo


# control variate estimator
# X = [X_1, X_2, ..., X_n], Y = [Y_1, Y_2, ..., Y_n], where (X_1, Y_1), ..., (X_n, Y_n) are i.i.d. pairs
# c – estimated value of -Cov(X, Y)/VarX
# EX – expected value of X should be known
def control_variate_estimator(X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64], c: float, EX: float) -> float:
	return crude_monte_carlo(Y) - c * (crude_monte_carlo(X) - EX)
