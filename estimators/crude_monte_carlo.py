import numpy as np
import numpy.typing as npt


# CMC (Crude Monte Carlo) estimator
# Y=[Y_1, Y_2, ..., Y_R] should be a list of i.i.d. samples of the estimated value
def cmc(Y: npt.ArrayLike[float]) -> float:
	return np.mean(Y)
