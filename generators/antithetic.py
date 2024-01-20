import numpy.typing as npt
import numpy as np
from math import ceil

from generators.standard_normal import sample_normal


# antithetic sampling of the normal distribution
# n - length of the sample (suggested to be even)
# returns (Z_1, -Z_1, Z_2, -Z_2, ..., Z_{n/2}, -Z_{n/2}), where Z_1, ..., Z_{n/2} are i.i.d.
def sample_normal_antithetic(n: int) -> npt.ArrayLike[float]:
	n_half = ceil(n / 2)

	sample = sample_normal(n_half)

	result = np.zeros(n)
	result[::2] = sample

	if n % 2 == 0:
		result[1::2] = -sample
	else:
		result[1::2] = -sample[:-1]

	return result
