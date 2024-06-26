import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jax import grad, jit, vmap
import numpy.random as npr
from jax.scipy.special import erfinv, erfc
from jax.scipy.stats import multivariate_normal as norm

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
        back_transf_data = self.gaussian_univariate_inverse_cdf(data)
        max_value = jnp.nanmax(back_transf_data[jnp.isfinite(back_transf_data)])
        back_transf_data = jnp.where(jnp.isinf(back_transf_data), max_value, back_transf_data)
        self.cov = jnp.cov(back_transf_data, rowvar=False, aweights=weights)

    def pdf(self, x: Array) -> Array:
        back_transf_x = self.gaussian_univariate_inverse_cdf(x)
        max_value = jnp.nanmax(back_transf_x[jnp.isfinite(back_transf_x)])
        back_transf_x = jnp.where(jnp.isinf(back_transf_x), max_value, back_transf_x)
        return norm.pdf(back_transf_x, mean=jnp.zeros(self.d), cov=self.cov)
    
    def logpdf(self, x: Array) -> Array:
        return jnp.log(self.pdf(x))
    
    
    def get_params(self) -> Array:
        return self.cov.flatten()
    
    def set_params(self, params: Array) -> None:
        self.cov = params.reshape(self.d, self.d)

    def get_nb_params(self) -> int:
        return self.d * self.d

    def gaussian_univariate_cdf(self, x: Array) -> Array:
        return 0.5 * erfc(-x / jnp.sqrt(2))
    
    def gaussian_univariate_inverse_cdf(self, x: Array) -> Array:
        return jnp.sqrt(2) * erfinv(2 * x - 1)
    
    
        
    