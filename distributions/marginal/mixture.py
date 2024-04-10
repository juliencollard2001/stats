import jax.numpy as jnp
from jax import Array
from jax import grad, jit, vmap
import jax.random

from jax.scipy.special import logsumexp

from .base import Marginal

class Mixture(Marginal):
        
        def __init__(self, distributions: list[Marginal], weights: Array|None = None, max_iter : int = 100) -> None:
            self.distributions = distributions
            if weights is None:
                weights = jnp.ones(len(distributions)) / len(distributions)
            self.weights = weights
            self.max_iters = max_iter

            key = jax.random.key(6837)

            for dist in self.distributions:
                params = dist.get_params()
                key, subkey = jax.random.split(key)
                perturb = jax.random.exponential(subkey, shape=params.shape) / 10
                dist.set_params(params + perturb)
        
        def fit(self, data: Array, weights: Array | None = None) -> None:

            for _ in range(self.max_iters):
                # E step
                log_likelihoods = jnp.array([dist.logpdf(data) for dist in self.distributions])
                log_weights = jnp.log(self.weights)
                log_posteriors = log_likelihoods + log_weights[:, None] - logsumexp(log_likelihoods + log_weights[:, None], axis=0)

                # M step
                self.weights = jnp.exp(log_posteriors).mean(axis=1)
                for i, dist in enumerate(self.distributions):
                    dist.fit(data, weights=jnp.exp(log_posteriors[i]))

        
        def get_params(self) -> Array:
            params = []
            for marginal in self.distributions:
                params.append(marginal.get_params())
            return jnp.concatenate(params)
        
        def set_params(self, params : Array) -> None:
            start = 0
            for marginal in self.distributions:
                end = start + len(marginal.get_params())
                marginal.set_params(params[start:end])
                start = end
        
        def pdf(self, x : Array) -> Array:
            pdfs = []
            for marginal in self.distributions:
                pdfs.append(marginal.pdf(x))
            return jnp.dot(jnp.array(pdfs).T, self.weights)
        
        def logpdf(self, x : Array) -> Array:
            return jnp.log(self.pdf(x))
        
        def cdf(self, x : Array) -> Array:
            cdfs = []
            for marginal in self.distributions:
                cdfs.append(marginal.cdf(x))
            return jnp.dot(jnp.array(cdfs).T, self.weights)
        
        def mean(self) -> Array:
            means = []
            for marginal in self.distributions:
                means.append(marginal.mean())
            return jnp.dot(jnp.array(means).T, self.weights)
        
        def var(self) -> Array:
            vars = []
            for marginal in self.distributions:
                vars.append(marginal.var())
            return jnp.dot(jnp.array(vars).T, self.weights)
        
        def std(self) -> Array:
            return jnp.sqrt(self.var())
        