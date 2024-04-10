import jax.numpy as jnp
from jax import Array
from jax import grad, jit, vmap

from jax.scipy.stats import gamma

from .base import Marginal

class Gamma(Marginal):
        
        def __init__(self, shape: Array = 1, scale: Array = 1) -> None:
            self.shape = shape
            self.scale = scale
        
        def fit(self, data: Array, weights: Array | None = None) -> None:
            if weights is None:
                weights = jnp.ones_like(data) / len(data)
            self.shape = jnp.sum(data * weights) / jnp.sum(weights)
            self.scale = jnp.sum((data / self.shape) * weights) / jnp.sum(weights)

        def get_params(self) -> Array:
            return jnp.array([self.shape, self.scale])
        
        def set_params(self, params : Array) -> None:
            self.shape = params[0]
            self.scale = params[1]

        def pdf(self, x : Array) -> Array:
            return gamma.pdf(x, loc=self.shape, scale=self.scale)
        
        def logpdf(self, x : Array) -> Array:
            return gamma.logpdf(x, loc=self.shape, scale=self.scale)
        
        def cdf(self, x : Array) -> Array:
            return gamma.cdf(x, loc=self.shape, scale=self.scale)
        
        def mean(self) -> Array:
            return self.shape * self.scale
        
        def var(self) -> Array:
            return self.shape * self.scale**2
        
        def std(self) -> Array:
            return jnp.sqrt(self.var())
        
        def mode(self) -> Array:
            return jnp.maximum(self.shape - 1, 0) * self.scale
        
        