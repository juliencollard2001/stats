import jax.numpy as jnp
from jax import Array
from jax import grad, jit, vmap

from jax.scipy.stats import norm

from .base import Marginal

class Normal(Marginal):
    
        def __init__(self, mean: Array = 0, std: Array = 1) -> None:
            self.mean = mean
            self.std = std
    
        def fit(self, data: Array, weights: Array | None = None) -> None:
            if weights is None:
                weights = jnp.ones_like(data) / len(data)
            self.mean = jnp.sum(data * weights)
            self.std = jnp.sqrt(jnp.sum((data - self.mean)**2 * weights))
    
        def get_params(self) -> Array: 
            return jnp.array([self.mean, self.std])
    
        def set_params(self, params : Array) -> None:
            self.mean = params[0]
            self.std = params[1]
    
        def pdf(self, x : Array) -> Array:
            return norm.pdf(x, loc=self.mean, scale=self.std)
        
        def logpdf(self, x : Array) -> Array:
            return norm.logpdf(x, loc=self.mean, scale=self.std)
    
        def cdf(self, x : Array) -> Array:
            return norm.cdf(x, loc=self.mean, scale=self.std)
        
        def ppf(self, q : Array) -> Array:
            return norm.ppf(q, loc=self.mean, scale=self.std)
    
    
        def mode(self) -> Array:
            return self.mean
    
        def median(self) -> Array:
            return self.mean
        
        def mean(self) -> Array:
            return self.mean
        
        def var(self) -> Array:
            return self.std**2
        
        def std(self) -> Array:
            return self.std
    
        def skew(self) -> Array:
            return 0
    
        def kurtosis(self) -> Array:
            return 3
    