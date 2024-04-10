import jax.numpy as jnp
from jax import Array
from jax import grad, jit, vmap

from jax.scipy.stats import uniform

from .base import Marginal

class Uniform(Marginal):
        
        def __init__(self, a: Array = 0, b: Array = 1) -> None:
            self.a = a
            self.b = b
        
        def fit(self, data: Array, weights: Array | None = None) -> None:
            if weights is None:
                weights = jnp.ones_like(data) / len(data)
            self.a = jnp.min(data)
            self.b = jnp.max(data)
        
        def get_params(self) -> Array:
            return jnp.array([self.a, self.b])
        
        def set_params(self, params : Array) -> None:
            self.a = params[0]
            self.b = params[1]
    
        def pdf(self, x : Array) -> Array:
            return uniform.pdf(x, loc=self.a, scale=self.b - self.a)
        
        def logpdf(self, x : Array) -> Array:
            return uniform.logpdf(x, loc=self.a, scale=self.b - self.a)
        
        def cdf(self, x : Array) -> Array:
            return uniform.cdf(x, loc=self.a, scale=self.b - self.a)
        
        def mean(self) -> Array:
            return (self.a + self.b) / 2
        
        def var(self) -> Array:
            return (self.b - self.a)**2 / 12
        
        def std(self) -> Array:
            return jnp.sqrt(self.var())
        
        def mode(self) -> Array:
            return self.a
        
        def median(self) -> Array:
            return (self.a + self.b) / 2
        
        def skew(self) -> Array:
            return 0
        
        def kurtosis(self) -> Array:
            return -6 / 5