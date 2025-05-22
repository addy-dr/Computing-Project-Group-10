import numpy as np

def negative_log_likelihood(params, x_data, y_data):
    A, lamb = params
    if lamb <= 0 or A <= 0:
        return np.inf

    mu = A * np.exp(-x_data / lamb)
    # Poisson log-likelihood (ignoring constant log(n!))
    nll = np.sum(mu - y_data * np.log(mu + 1e-9))  # small term to avoid log(0)
    return nll
