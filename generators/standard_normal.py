import numpy as np
import numpy.typing as npt


# sample n i.i.d. variables of standard normal distribution
def sample_normal(n: int) -> npt.ArrayLike[float]:
	return np.random.normal(size=n)
