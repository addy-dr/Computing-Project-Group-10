import numpy as np
import STOM_higgs_tools

### FUNCTION DEFINITIONS FOR MAIN.PY ###


def negative_log_likelihood(params, x_data, y_data):
    A, lamb = params
    if lamb <= 0 or A <= 0:
        return np.inf

    mu = A * np.exp(-x_data / lamb)
    # Poisson log-likelihood (ignoring constant log(n!))
    nll = np.sum(mu - y_data * np.log(mu + 1e-9))  # small term to avoid log(0)
    return nll


def chi2_estimate(A_values, lamb_values, x, y):
    "χ² calculation using binned background data"

    chi2_grid = np.zeros((len(A_values), len(lamb_values)))

    for i, A_trial in enumerate(A_values):
        for j, lamb_trial in enumerate(lamb_values):
            # Calculate expected background
            B_expected = STOM_higgs_tools.get_B_expectation(x, A_trial, lamb_trial)
            chi2 = np.sum((y - B_expected)**2 / B_expected)
            chi2_grid[i, j] = chi2

    min_index = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
    A_estimate, lamb_estimate = A_values[min_index[0]], lamb_values[min_index[1]]
    min_chi2 = chi2_grid[min_index]
    print(f"χ² Results: A = {A_estimate:.1f}, λ = {lamb_estimate:.1f}, χ²/DoF = {min_chi2:.2f}")
    return A_estimate, lamb_estimate, min_chi2