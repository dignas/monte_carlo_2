import numpy as np
import numpy.typing as npt


# Stratified estimator
# p = [p_1, p_2, ..., p_m] – probablilities of each stratum
# I = [I_1, I_2, ..., I_m] – estimated values in each stratum
def stratified_estimator(p: npt.NDArray[np.float64], I: npt.NDArray[np.float64]) -> float:
	return np.dot(p, I)
