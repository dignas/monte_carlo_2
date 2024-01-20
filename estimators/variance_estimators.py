import numpy as np
import numpy.typing as npt


def variance_estimator(X: npt.NDArray[np.float64]):
	return covariance_estimator(X, X)
	

def covariance_estimator(X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64]):
	R = min(len(X), len(Y))
	if R <= 1:
		return 0
	
	X = X[:R]
	Y = Y[:R]

	EX = np.mean(X)
	EY = np.mean(Y)

	return np.dot(X - EX, Y - EY) / (R - 1)
