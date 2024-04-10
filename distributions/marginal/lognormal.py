import jax.numpy as jnp
from jax import Array
from jax import grad, jit, vmap

from .base import Marginal

class LogNormal(Marginal):
        
        def __init__(self, mean: Array = 0, std: Array = 1) -> None:
            self.mean = mean
            self.std = std
        
        def fit(self, data: Array, weights: Array | None = None) -> None:
            if weights is None:
                weights = jnp.ones_like(data) / len(data)
            self.mean = jnp.sum(jnp.log(data) * weights)
            self.std = jnp.sqrt(jnp.sum((jnp.log(data) - self.mean)**2 * weights))
        
        def get_params(self) -> Array: 
            return jnp.array([self.mean, self.std])
        
        def set_params(self, params : Array) -> None:
            self.mean = params[0]
            self.std = params[1]
        
        def pdf(self, x : Array) -> Array:
            return jnp.exp(-0.5 * ((jnp.log(x) - self.mean) / self.std)**2) / (x * self.std * jnp.sqrt(2 * jnp.pi))
        
        def logpdf(self, x : Array) -> Array:
            return -0.5 * ((jnp.log(x) - self.mean) / self.std)**2 - jnp.log(x) - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(self.std)
        
        def cdf(self, x : Array) -> Array:
            return 0.5 * (1 + jnp.erf((jnp.log(x) - self.mean) / (self.std * jnp.sqrt(2))))
        
        def ppf(self, q : Array) -> Array:
            return jnp.exp(self.mean + self.std * jnp.sqrt(2) * jnp.erfinv(2 * q - 1))
        
        def sample(self, n_samples: int) -> Array:
            return jnp.exp(self.mean + self.std * jnp.random.normal(size=(n_samples,)))
        
        def mode(self) -> Array:
            return jnp.exp(self.mean - self.std**2)
        
        def median(self) -> Array:
            return jnp.exp(self.mean)
            
        def mean(self) -> Array:
            return jnp.exp(self.mean + 0.5 * self.std**2)
            
        def var(self) -> Array:
            return (jnp.exp(self.std**2) - 1) * jnp.exp(2 * self.mean + self.std**2)
            
        def std(self) -> Array:
            return jnp.sqrt(self.var())
        
        def skew(self) -> Array:
            return (jnp.exp(self.std**2) + 2)