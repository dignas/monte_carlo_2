import numpy as np
import numpy.typing as npt


def variance_estimator(X: npt.ArrayLike[float]):
	return covariance_estimator(X, X)
	

def covariance_estimator(X: npt.ArrayLike[float], Y: npt.ArrayLike[float]):
	R = min(len(X), len(Y))
	if R <= 1:
		return 0
	
	X = X[:R]
	Y = Y[:R]

	EX = np.mean(X)
	EY = np.mean(Y)

	return np.dot(X - EX, Y - EY) / (R - 1)
