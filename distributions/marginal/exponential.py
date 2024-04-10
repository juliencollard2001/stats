import jax.numpy as jnp
from jax import Array
from jax import grad, jit

from .base import Marginal

class Exponential(Marginal):
            
    def __init__(self, rate: Array = 1) -> None:
        self.rate = rate
    
    def fit(self, data: Array, weights: Array | None = None) -> None:
        if weights is None:
            weights = jnp.ones_like(data) / len(data)
        self.rate = jnp.sum(data * weights) / jnp.sum(weights)

    def get_params(self) -> Array:
        return jnp.array([self.rate])
    
    def set_params(self, params : Array) -> None:
        self.rate = params[0]

    def pdf(self, x : Array) -> Array:
        return jnp.exp(-self.rate * x)
    
    def logpdf(self, x : Array) -> Array:
        return -self.rate * x
    
    def cdf(self, x : Array) -> Array:
        return 1 - jnp.exp(-self.rate * x)
    
    def mean(self) -> Array:
        return 1 / self.rate
    
    def var(self) -> Array:
        return 1 / self.rate**2
    
    def std(self) -> Array:
        return jnp.sqrt(self.var())
    
    def mode(self) -> Array:
        return 0
    
    def median(self) -> Array:
        return jnp.log(2) / self.rate
    
    def skew(self) -> Array:
        return 2
    
    def kurtosis(self) -> Array:
        return 6