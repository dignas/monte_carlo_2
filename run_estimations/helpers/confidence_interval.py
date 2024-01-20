from model.params import quantile

import numpy as np
import numpy.typing as npt
from typing import Tuple


def calculate_confidence_interval(result: npt.NDArray[np.float64], variance: float) -> Tuple[float, float]:
	centre = np.mean(result)
	err = np.sqrt(variance) * quantile

	return centre - err, centre + err
