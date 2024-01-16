# This module provides an exact value of the call price in case n=1 (European)

from scipy.stats import norm
from math import log, exp

import model.params as p


def get_exact_call_price_for_params(r, sigma, S_0, K):

	d1 = 1 / sigma * (log(S_0 / K) + r + sigma**2 / 2)
	d2 = d1 - sigma

	return S_0 * norm.cdf(d1) - K * exp(-r) * norm.cdf(d2)


exact_call_price = get_exact_call_price_for_params(p.r, p.sigma, p.S_0, p.K)
