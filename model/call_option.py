# This module provides a parametrizable function that calculates a call option price.
# The function takes a simulated trajectory of a Brownian motion at times 1/n, 2/n, ..., 1.
# In case n=1 this is an European call option, otherwise it's Asian call option.

import numpy as np
import numpy.typing as npt

import model.params as p


def build_call_price(r: float, sigma: float, S_0: float, K: float):

	mu_star = r - sigma**2 / 2

	def call_price(brownian_trajectory: npt.ArrayLike[float]):
		n = len(brownian_trajectory)
		t = np.arange(1, n + 1)
		
		S = S_0 * np.exp(mu_star * t + sigma * brownian_trajectory)
		An = np.mean(S)

		return np.exp(-r) * np.max(0, An - K)

	return call_price


call_price = build_call_price(p.r, p.sigma, p.S_0, p.K)
