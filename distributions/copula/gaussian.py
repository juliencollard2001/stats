import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jax import grad, jit, vmap
import numpy.random as npr
from jax.scipy.special import erfinv, erfc
from jax.scipy.stats import norm

from .base import Copula

class Gaussian(Copula):

    def __init__(self, d :int) -> None:
        self.d = d
        self.cov = jnp.eye(d)

    def fit(self, data: Array, weights: Array | None = None) -> None:
        n,d = data.shape
        if weights is None:
            weights = jnp.ones(n)
        else:
            weights = weights / jnp.sum(weights)
        back_transf_data = erfinv(data)
        self.cov = jnp.cov(back_transf_data, rowvar=False, aweights=weights)
    
    def pdf(self, x: Array) -> Array:
        return norm.pdf(erfinv(x), loc=jnp.zeros(self.d), scale=self.cov)
    
    def logpdf(self, x: Array) -> Array:
        return norm.logpdf(erfinv(x), loc=jnp.zeros(self.d), scale=self.cov)
    
    def get_params(self) -> Array:
        return self.cov.flatten()
    
    def set_params(self, params: Array) -> None:
        self.cov = params.reshape(self.d, self.d)

    def get_nb_params(self) -> int:
        return self.d * self.d
    
    
        
    