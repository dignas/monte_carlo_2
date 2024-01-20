from run_estimations.cmc import run_cmc
from run_estimations.antithetic import run_antithetic
from run_estimations.cv import run_control_variate
from run_estimations.stratified import run_stratified
import run_estimations.params as p

from estimators.variance_estimators import variance_estimator

from run_estimations.helpers.confidence_interval import calculate_confidence_interval
import run_estimations.helpers.plot_styles as plt_styles

from model.black_scholes import exact_call_price

import numpy as np
import matplotlib.pyplot as plt


__no_tests = 100
__n = 1


def run_n_1():
	global __no_tests, __n

	variance_cmc = np.zeros_like(p.Rs, dtype=np.float64)
	variance_stratified_proportional = np.zeros_like(p.Rs, dtype=np.float64)
	variance_stratified_optimal = np.zeros_like(p.Rs, dtype=np.float64)
	variance_antithetic = np.zeros_like(p.Rs, dtype=np.float64)
	variance_cv = np.zeros_like(p.Rs, dtype=np.float64)

	last_R_index = np.shape(p.Rs)[0] - 1

	for i, R in enumerate(p.Rs):
		results_cmc = run_cmc(__no_tests, __n, R)
		variance_cmc[i] = variance_estimator(results_cmc)

		stratified_R_proportional = np.full(p.m, np.ceil(R / p.m), dtype=np.int32)
		results_stratified_proportional, sigma_strat = run_stratified(__no_tests, __n, stratified_R_proportional, p.m)
		variance_stratified_proportional[i] = variance_estimator(results_stratified_proportional)

		stdev_strat_prop = np.sqrt(sigma_strat)
		stratified_R_optimal = np.ceil(R * stdev_strat_prop / np.sum(stdev_strat_prop)).astype(np.int32)
		results_stratified_optimal, _ = run_stratified(__no_tests, __n, stratified_R_optimal, p.m)
		variance_stratified_optimal[i] = variance_estimator(results_stratified_optimal)

		results_antithetic = run_antithetic(__no_tests, R)
		variance_antithetic[i] = variance_estimator(results_antithetic)

		results_cv = run_control_variate(__no_tests, R)
		variance_cv[i] = variance_estimator(results_cv)

	_, ax = plt.subplots(1, 1)

	ax.plot(np.full_like(results_cmc, 1), results_cmc, label="Monte Carlo estimation", **plt_styles.monte_carlo_estimation)
	ax.plot(np.full_like(results_stratified_proportional, 2), results_stratified_proportional, **plt_styles.monte_carlo_estimation)
	ax.plot(np.full_like(results_stratified_optimal, 3), results_stratified_optimal, **plt_styles.monte_carlo_estimation)
	ax.plot(np.full_like(results_antithetic, 4), results_antithetic, **plt_styles.monte_carlo_estimation)
	ax.plot(np.full_like(results_cv, 5), results_cv, **plt_styles.monte_carlo_estimation)

	ax.plot([1, 1], calculate_confidence_interval(results_cmc, variance_cmc[last_R_index]), label="confidence interval", **plt_styles.confidence_interval)
	ax.plot([2, 2], calculate_confidence_interval(results_stratified_proportional, variance_stratified_proportional[last_R_index]), **plt_styles.confidence_interval)
	ax.plot([3, 3], calculate_confidence_interval(results_stratified_optimal, variance_stratified_optimal[last_R_index]), **plt_styles.confidence_interval)
	ax.plot([4, 4], calculate_confidence_interval(results_antithetic, variance_antithetic[last_R_index]), **plt_styles.confidence_interval)
	ax.plot([5, 5], calculate_confidence_interval(results_cv, variance_cv[last_R_index]), **plt_styles.confidence_interval)

	ax.set_xticks([1, 2, 3, 4, 5], ["Crude\nMonte\nCarlo", "Stratified\nproportional", "Stratified\noptimal", "Antithetic", "Control\nvariate"])
	ax.set_yticks(np.append(ax.get_yticks(), exact_call_price))

	ax.hlines(exact_call_price, 0, 6, label="exact value", **plt_styles.exact_value_hline)

	ax.legend()

	plt.margins(x=0)
	plt.subplots_adjust(bottom=0.15, top=0.95)
	plt.savefig("plots/estimation_result_n_1.png", dpi=300)

	_, ax = plt.subplots(1, 1)

	ax.set_xlabel("R – number of simulations")
	ax.set_xlabel("Estimator variance")

	ax.plot(p.Rs, variance_cmc, label="Crude Monte Carlo", color="black")
	ax.plot(p.Rs, variance_stratified_proportional, label="Stratified proportional", color="navy")
	ax.plot(p.Rs, variance_stratified_optimal, label="Stratified optimal", color="lightskyblue")
	ax.plot(p.Rs, variance_antithetic, label="Antithetic", color="goldenrod")
	ax.plot(p.Rs, variance_cv, label="Control variate", color="crimson")

	ax.legend()

	plt.savefig("plots/variance_n_1.png", dpi=300)
