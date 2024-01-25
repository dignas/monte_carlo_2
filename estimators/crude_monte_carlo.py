from estimators.variance_estimators import variance_estimator

import numpy as np
import numpy.typing as npt
from typing import Tuple


# CMC (Crude Monte Carlo) estimator
# Y=[Y_1, Y_2, ..., Y_R] should be a list of i.i.d. samples of the estimated value
def cmc_estimator(Y: npt.NDArray[np.float64]) -> Tuple[float, float]:
	R, = np.shape(Y)
	return np.mean(Y), variance_estimator(Y) / R
