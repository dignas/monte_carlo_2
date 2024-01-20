import numpy as np
import numpy.typing as npt


# sample n i.i.d. variables of standard normal distribution
def sample_normal(n: int) -> npt.NDArray[np.float64]:
	return np.random.normal(size=n)
