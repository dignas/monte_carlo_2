import numpy as np
import numpy.typing as npt
from scipy.stats.distributions import chi2

from standard_normal import sample_normal


# This generator can be used to sample a trajectory of brownian motion.
# It can be used for CMC, stratified sampling and CV.
# n specifies the number of sampled points (returns [B(1/n), B(2/n), ..., B(1)]).
# m, i are the parameters for stratified sampling. m – number of strata, i – index od stratum.
def sample_trajectory(n: int, m: int = 1, i: int = 1) -> npt.ArrayLike[int]:
	ksi = sample_normal(n)
	X = ksi / np.linalg.norm(ksi)

	U = np.random.uniform()
	D = np.sqrt(chi2.ppf(i / m + U / m, df=n))
	Z = D * X

	A = np.tril(np.full((n, n), fill_value=1/np.sqrt(n)))

	return A * Z
