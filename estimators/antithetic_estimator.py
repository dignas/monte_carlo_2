from estimators.crude_monte_carlo import crude_monte_carlo

# antithetic estimator is the same as CMC with a proper choice of generator
antithetic_estimator = crude_monte_carlo
