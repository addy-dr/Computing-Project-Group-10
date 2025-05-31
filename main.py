import STOM_higgs_tools
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2 as sp_chi2 # prevent ambiguity with the variable chi2

## CONSTANTS

BINS = 30
RANGE = [104.0, 155.0]

GAUSS_AMP = 700
GAUSS_MEAN = 125
GAUSS_STD = 1.5

## FUNCTION DEFINITIONS

def negative_log_likelihood(params, x_data, y_data):
    '''
    Gets the Poisson log-likelihood of a set of data for an exponential model.
    '''
    A, lamb = params
    if lamb <= 0 or A <= 0:
        return np.inf

    mu = A * np.exp(-x_data / lamb)
    # Poisson log-likelihood (ignoring constant log(n!))
    nll = np.sum(mu - y_data * np.log(mu + 1e-9))  # small term to avoid log(0)
    return nll


def chi2_estimate(A_values, lamb_values, xs, ys):
    '''
    Performs a χ² search for the best values of A and lambda.
    '''
    chi2_grid = np.zeros((len(A_values), len(lamb_values)))

    for i, A_trial in enumerate(A_values):
        for j, lamb_trial in enumerate(lamb_values):
            # Calculate expected background
            B_expected = STOM_higgs_tools.get_B_expectation(xs, A_trial, lamb_trial)
            chi2 = np.sum((ys - B_expected)**2 / B_expected)
            chi2_grid[i, j] = chi2

    min_index = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
    A_estimate, lamb_estimate = A_values[min_index[0]], lamb_values[min_index[1]]
    min_chi2 = chi2_grid[min_index]
    print(f"χ² Results: A = {A_estimate:.1f}, λ = {lamb_estimate:.1f}, χ²/DoF = {min_chi2:.2f}")
    return A_estimate, lamb_estimate, min_chi2


def main():
    "The main data analysis"

    # Generate data
    vals = np.array(STOM_higgs_tools.generate_data(n_signals=400))

    # Histogram
    bin_height, bin_edges = np.histogram(vals, range=RANGE, bins=BINS)
    mean = (bin_edges[:-1] + bin_edges[1:]) / 2
    ystd = np.sqrt(bin_height)
    xstd = (bin_edges[1:] - bin_edges[:-1]) / 2

    # Background bin selection using dynamic masking
    mask = (mean < 121.0) | (mean > 129.0)
    mean_background = mean[mask]
    bin_height_background = bin_height[mask]


    # MLE fit with bounds and status check
    initial_guess = [1800, 30]
    result = minimize(
        lambda p: negative_log_likelihood(p, mean_background, bin_height_background),
        x0=initial_guess,
        bounds=[(0, None), (0, None)]  # Prevent negative parameters
    )

    if not result.success:
        raise ValueError(f"MLE failed: {result.message}")

    A_mle, lamb_mle = result.x
    print(f"MLE Results: A = {A_mle:.1f}, λ = {lamb_mle:.1f}")


    # χ² grid search range near MLE results
    A_values = np.linspace(0.5*A_mle, 1.5*A_mle, 50)
    lamb_values = np.linspace(0.5*lamb_mle, 1.5*lamb_mle, 50)

    #χ² calculation using binned background data. Use function defined in statistics_functions
    print("\nStart calculation for background data:")
    A_chi2, lamb_chi2, chi2_min = chi2_estimate(A_values, lamb_values, mean_background, bin_height_background)
    p_value = 1 - sp_chi2.cdf(chi2_min, (len(mean_background) - 2))

    # background only hypothesis test (including signal region)
    print("\nWhat happens when we include our signal region?")
    A_chi2_with_signal, lamb_chi2_with_signal, chi2_min_with_signal = chi2_estimate(A_values, lamb_values, mean, bin_height)
    p_value = 1 - sp_chi2.cdf(chi2_min_with_signal, (len(mean) - 2))
    print(f"With signal (Background Only): p_value = {p_value} << 5%, should be rejected.")

    # signal plus background hypothesis
    x = np.linspace(104, 155, 500)
    y_gaussian = STOM_higgs_tools.get_SB_expectation(x, A_mle, lamb_mle, GAUSS_MEAN, GAUSS_STD, GAUSS_AMP)



    # Plot both fits
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'figure.dpi': 400})
    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.linspace(104, 155, 500)
    ax.plot(x, STOM_higgs_tools.get_B_expectation(x, A_mle, lamb_mle), label="MLE Fit", color='red')
    ax.plot(x, STOM_higgs_tools.get_B_expectation(x, A_chi2, lamb_chi2), '--', label="χ² Fit (background)", color='blue')
    ax.plot(x, STOM_higgs_tools.get_B_expectation(x, A_chi2_with_signal, lamb_chi2_with_signal), '--', label="χ² Fit (background + signal)", color='yellow')
    ax.errorbar(mean, bin_height, yerr=ystd, xerr=xstd, fmt='o', markersize=3, color='black')

    ax.set_xlim(104, 155)
    ax.set_xlabel("Rest mass (GeV)")
    ax.set_ylabel("Number of entries")
    ax.legend()
    plt.savefig("histogram.png", bbox_inches='tight')
    plt.show()


main()


