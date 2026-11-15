"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
from scipy.special import logsumexp
from copy import deepcopy


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a Gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current Gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    log_likelihood = 0.0
    post = np.zeros((n, K))

    for i in range(n):
        log_probs = np.zeros(K)
        for j in range(K):
            mask = (X[i] != 0)
            X_masked = X[i, mask]
            mu_masked = mixture.mu[j, mask]

            # Compute log probability for the j-th Gaussian
            log_prob = -0.5 * np.sum(((X_masked - mu_masked) ** 2) / mixture.var[j])
            log_prob -= 0.5 * np.sum(mask) * np.log(2 * np.pi * mixture.var[j])
            log_prob += np.log(mixture.p[j] + 1e-16)  # to avoid log(0)
            log_probs[j] = log_prob

        # Compute the log-sum-exp for normalization
        log_likelihood_i = logsumexp(log_probs)
        log_likelihood += log_likelihood_i
    


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    # Update the mixture weights (p)
    nk = np.sum(post, axis=0)
    p = nk / n

    # Update the means (mu)
    mu = np.zeros((K, d))
    for j in range(K):
        for l in range(d):
            mask = X[:, l] != 0
            mu[j, l] = np.sum(post[mask, j] * X[mask, l]) / np.sum(post[mask, j])

    # Update the variances (var)
    var = np.zeros(K)
    for j in range(K):
        sse = 0  # sum of squared errors
        for l in range(d):
            mask = X[:, l] != 0
            sse += np.sum(post[mask, j] * (X[mask, l] - mu[j, l]) ** 2)
        var[j] = sse / (nk[j] * d)
        var[j] = max(var[j], min_variance)

    return GaussianMixture(mu, var, p)
    



def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model using the EM algorithm until convergence.

    Args:
        X: (n, d) array holding the data.
        post: (n, K) array holding the soft counts for all components for all examples.
        mixture: the initial Gaussian mixture.

    Returns:
        GaussianMixture: the new Gaussian mixture after convergence.
        np.ndarray: (n, K) array holding the soft counts for all components for all examples.
        float: the log-likelihood of the final assignment.
    """
    prev_log_likelihood = None
    #log_likelihood = None
    threshold = 1e-6

    # Deep copy of X to avoid altering the original data
    X = deepcopy(X)

    while True:
        # E-step
        post, log_likelihood = estep(X, mixture)

        # M-step
        mixture = mstep(X, post, mixture)

        # Check for convergence
        if prev_log_likelihood is not None and (log_likelihood - prev_log_likelihood) <= threshold * abs(log_likelihood):
            break

        prev_log_likelihood = log_likelihood

    return mixture, post, log_likelihood



def initialize_mixture(X: np.ndarray, K: int, seed: int = 0) -> Tuple[GaussianMixture, np.ndarray]:
    """
    Initializes the Gaussian mixture model.

    Args:
        X: (n, d) array holding the data
        K: number of components
        seed: random seed for reproducibility

    Returns:
        GaussianMixture: the initialized gaussian mixture
        np.ndarray: (n, K) array holding the soft counts for all components for all examples
    """
    np.random.seed(seed)
    n, d = X.shape
    
    # Initialize the means (mu) by randomly choosing K data points
    indices = np.random.choice(n, K, replace=False)
    mu = X[indices]
    
    # Initialize the variances (var) with the same value (e.g., variance of the entire data)
    var = np.var(X, axis=0).mean() * np.ones(K)
    
    # Initialize the mixing proportions (p) equally
    p = np.full(K, 1 / K)
    
    # Initialize the post (responsibilities) matrix uniformly
    post = np.full((n, K), 1 / K)
    
    # Create the GaussianMixture instance
    mixture = GaussianMixture(mu, var, p)
    
    return mixture, post