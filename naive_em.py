"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    K, _ = mixture.mu.shape
    n, d = X.shape
    gprob = lambda x, m, s: (1 / (2*np.pi*s)**(d/2)) * (np.exp(-((x-m)**2).sum(axis=1) / (2*s)))
    soft_counts, ll_ = np.empty((0,K)), np.empty((0,K))

    for i in range(n):
        prob = gprob(np.tile(X[i], (K,1)), mixture.mu, mixture.var)
        prob_ll = prob.reshape(1, K)
        prob_post = (prob*mixture.p)/(prob*mixture.p).sum()
        soft_counts = np.append(soft_counts, prob_post, axis=0)
        ll_ =  np.append(ll_, prob_ll, axis=0)
    ll = np.log((ll_*mixture.p).sum(axis=1)).sum()

    return soft_counts, ll


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    K, _ = post.shape
    n, d = X.shape
    up_mu, up_var, up_p = np.empty((0, K)), np.empty(K), np.empty(K)
    for i in range(n):
        temp_mu = post[i] * X[i] / (post[i].sum*())
        up_mu = np.append(up_mu, temp_mu)
        temp_var = (post[i] * (X[i] - temp_mu)**2) / (post[i].sum*())
        up_var = np.append(up_var, temp_var)
        temp_p = 1/n * post[i].sum()
        up_p = np.append(up_p, temp_p)

    mixture = GaussianMixture(mu=up_mu, var=up_var, p=up_p)

    return mixture


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """

    post, ll = estep(X, mixture)

    return post, ll


