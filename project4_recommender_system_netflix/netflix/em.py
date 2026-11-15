"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


import numpy as np
from typing import Tuple

def gaussian_pdf(X: np.ndarray, mean: np.ndarray, var: float) -> np.ndarray:
    """
    Computes the probability density function of a multivariate Gaussian distribution.
    
    Args:
    - X: (n, d) array where each row is a d-dimensional data point
    - mean: (d,) array representing the mean vector of the Gaussian
    - var: scalar representing the variance of the Gaussian
    
    Returns:
    - (n,) array of probability densities
    """
    d = X.shape[1]
    
    # Compute the normalization factor
    norm_factor = 1 / np.sqrt((2 * np.pi) ** d * var ** d)
    
    # Compute the exponent
    diff = X - mean
    exponent = -0.5 * np.sum(diff ** 2, axis=1) / var
    
    return norm_factor * np.exp(exponent)

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """
    E-step: Softly assigns each data point to a gaussian component
    
    Args:
    - X: (n, d) array holding the data
    - mixture: the current gaussian mixture
    
    Returns:
    - np.ndarray: (n, K) array holding the soft counts for all components for all examples
    - float: log-likelihood of the assignment
    """
    
    n, d = X.shape
    K = mixture.mu.shape[0]
    p = mixture.p
    mu = mixture.mu
    var = mixture.var
    
    # Initialize the responsibilities matrix
    resp = np.zeros((n, K))
    
    # Compute the responsibilities for each Gaussian
    for k in range(K):
        resp[:, k] = p[k] * gaussian_pdf(X, mu[k], var[k])
    
    # Calculate the total responsibilities
    resp_sum = np.sum(resp, axis=1, keepdims=True)
    
    # Normalize to get the posterior probabilities
    resp = resp / resp_sum
    
    # Calculate log-likelihood
    log_likelihood = np.sum(np.log(resp_sum))
    
    return resp, log_likelihood



def mstep(X: np.ndarray, resp: np.ndarray) -> GaussianMixture:
    """
    M-step: Updates the gaussian mixture by maximizing the log-likelihood
    
    Args:
    - X: (n, d) array holding the data
    - resp: (n, K) array holding the soft counts for all components for all examples
    
    Returns:
    - GaussianMixture: the new gaussian mixture
    """
    
    n, d = X.shape
    K = resp.shape[1]
    
    # Calculate effective number of samples for each component
    n_k = np.sum(resp, axis=0)
    
    # Update the mixing coefficients
    p = n_k / n
    
    # Update the means
    mu = np.dot(resp.T, X) / n_k[:, np.newaxis]
    
    # Update the variances
    var = np.zeros(K)
    for k in range(K):
        diff = X - mu[k]
        var[k] = np.sum(resp[:, k] * np.sum(diff ** 2, axis=1)) / (n_k[k] * d)
    
    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture, post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """
    Runs the mixture model using the EM algorithm until convergence.

    Args:
        X: (n, d) array holding the data
        mixture: the initial gaussian mixture
        post: (n, K) array holding the soft counts for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts for all components for all examples
        float: log-likelihood of the current assignment
    """
    
    prev_log_likelihood = None
    tolerance = 1e-6
    max_iter = 100  # Maximum number of iterations (optional)
    
    for _ in range(max_iter):
        # E-step: Update the responsibilities based on the current mixture
        post, log_likelihood = estep(X, mixture)
        
        # Check for convergence
        if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < tolerance:
            break
        
        prev_log_likelihood = log_likelihood
        
        # M-step: Update the mixture parameters based on the current responsibilities
        mixture = mstep(X, post)
    
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



def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
