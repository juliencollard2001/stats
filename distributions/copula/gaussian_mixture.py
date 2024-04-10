import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jax import grad, jit, vmap
import numpy.random as npr

from .base import Copula

class GaussianMixture(Copula):
    
    def __init__(self, d :int, nb_components : int = 5) -> None:
        self.d = d
        self.nb_components = nb_components
        self.weights = jnp.ones((nb_components,)) / nb_components
        self.means = jnp.zeros((nb_components, d))
        self.covs = jnp.array([jnp.eye(d) for _ in range(nb_components)])

    def fit(self, data: Array, weights: Array | None = None) -> None:
        n,d = data.shape

        self.weights = jnp.ones((self.nb_components,)) / self.nb_components
        self.means = jnp.array([data[npr.choice(n)] for _ in range(self.nb_components)])
        self.covs = jnp.array([jnp.eye(d) for _ in range(self.nb_components)])

        for _ in range(100):
            # E-step
            gamma = self.expectation(data)
            # M-step
            self.maximization(data, gamma)

    def expectation(self, data: Array) -> Array:
        n = data.shape[0]
        gamma = jnp.zeros((n, self.nb_components))
        for k in range(self.nb_components):
            gamma = gamma.at[:, k].set(self.weights[k] * jsp.stats.multivariate_normal.pdf(data, self.means[k], self.covs[k]))
        gamma = gamma / jnp.sum(gamma, axis=1, keepdims=True)
        return gamma
    
    def maximization(self, data: Array, gamma: Array) -> None:
        n = data.shape[0]
        for k in range(self.nb_components):
            Nk = jnp.sum(gamma[:,k])
            self.weights = self.weights.at[k].set(Nk / n)
            self.means = self.means.at[k].set(jnp.sum(gamma[:,k,None] * data, axis=0) / Nk)
            self.covs = self.covs.at[k].set(jnp.dot((gamma[:,k,None] * (data - self.means[k])).T, data - self.means[k]) / Nk)

    def get_params(self) -> Array:
        return jnp.concatenate([self.weights.ravel(), self.means.ravel(), self.covs.ravel()])
    
    def set_params(self, params : Array) -> None:
        b = self.nb_components
        d = self.d
        raveled_weights = params[:b]
        raveled_means = params[b:b+b*d]
        raveled_covs = params[b+b*d:]
        self.weights = raveled_weights
        self.means = raveled_means.reshape((b, d))
        self.covs = raveled_covs.reshape((b, d, d))

    def pdf(self, x : Array) -> Array:
        return jnp.sum(jnp.array([self.weights[k] * jsp.stats.multivariate_normal.pdf(x, self.means[k], self.covs[k]) for k in range(self.nb_components)]), axis=0)
    
    def logpdf(self, x : Array) -> Array:
        return jnp.log(self.pdf(x))
    

        