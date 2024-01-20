from estimators.crude_monte_carlo import cmc_estimator

# antithetic estimator is the same as CMC with a proper choice of generator
antithetic_estimator = cmc_estimator
