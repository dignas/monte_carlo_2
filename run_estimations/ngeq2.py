from run_estimations.cmc import run_cmc
from run_estimations.stratified import run_stratified
import run_estimations.helpers.params as p

from estimators.variance_estimators import variance_estimator

from run_estimations.helpers.confidence_interval import calculate_confidence_interval
import run_estimations.helpers.plot_styles as plt_styles

import numpy as np
import matplotlib.pyplot as plt


def run_n_geq_2(n: int):
	variance_cmc = np.zeros_like(p.Rs, dtype=np.float64)
	variance_stratified_proportional = np.zeros_like(p.Rs, dtype=np.float64)
	variance_stratified_optimal = np.zeros_like(p.Rs, dtype=np.float64)

	last_R_index = np.shape(p.Rs)[0] - 1

	for i, R in enumerate(p.Rs):
		results_cmc, variance_cmc[i] = run_cmc(n, R)

		stratified_R_proportional = np.full(p.m, np.ceil(R / p.m), dtype=np.int32)
		results_stratified_proportional, variance_stratified_proportional[i], sigma_strat = run_stratified(n, stratified_R_proportional, p.m)

		stdev_strat_prop = np.sqrt(sigma_strat)
		stratified_R_optimal = np.ceil(R * stdev_strat_prop / np.sum(stdev_strat_prop)).astype(np.int32)
		results_stratified_optimal, variance_stratified_optimal[i], _ = run_stratified(n, stratified_R_optimal, p.m)

	_, ax = plt.subplots(1, 1)

	ax.plot(1, results_cmc, label="Monte Carlo estimation", **plt_styles.monte_carlo_estimation)
	ax.plot(2, results_stratified_proportional, **plt_styles.monte_carlo_estimation)
	ax.plot(3, results_stratified_optimal, **plt_styles.monte_carlo_estimation)

	ax.plot([1, 1], calculate_confidence_interval(results_cmc, variance_cmc[last_R_index]), label="confidence interval", **plt_styles.confidence_interval)
	ax.plot([2, 2], calculate_confidence_interval(results_stratified_proportional, variance_stratified_proportional[last_R_index]), **plt_styles.confidence_interval)
	ax.plot([3, 3], calculate_confidence_interval(results_stratified_optimal, variance_stratified_optimal[last_R_index]), **plt_styles.confidence_interval)

	ax.set_ylabel("Estimated value")
	ax.set_xticks([1, 2, 3], ["Crude\nMonte\nCarlo", "Stratified\nproportional", "Stratified\noptimal"])

	ax.legend()

	plt.margins(x=0.5)
	plt.subplots_adjust(bottom=0.15, top=0.95)
	plt.savefig(f"plots/estimation_result_n_{n}.png", dpi=300)

	_, ax = plt.subplots(1, 1)

	ax.set_xlabel("R – number of simulations")
	ax.set_ylabel("Estimator variance")

	ax.plot(p.Rs, variance_cmc, label="Crude Monte Carlo", color="black")
	ax.plot(p.Rs, variance_stratified_proportional, label="Stratified proportional", color="navy")
	ax.plot(p.Rs, variance_stratified_optimal, label="Stratified optimal", color="lightskyblue")

	ax.legend()

	plt.savefig(f"plots/variance_n_{n}.png", dpi=300)
