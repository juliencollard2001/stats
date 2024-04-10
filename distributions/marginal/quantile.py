import jax.numpy as jnp
from jax import Array
from jax import grad, jit, vmap
from jax.scipy.stats import beta

from .base import Marginal

class Quantile(Marginal):

    def __init__(self, quantiles : Array|None = None) -> None:
        if quantiles is None:
            quantiles = jnp.linspace(0,1,100)
        self.quantiles = quantiles
        self.values = jnp.zeros_like(quantiles)
        self.density = jit(vmap(grad(self.cdf)))

    def fit(self, data: Array, weights: Array | None = None) -> None:
        if weights is None:
            weights = jnp.ones_like(data) / len(data)

        # Sort the data and weights in ascending order
        sorted_indices = jnp.argsort(data)
        sorted_data = data[sorted_indices]
        sorted_weights = weights[sorted_indices]

        # Compute the cumulative sum of the weights
        cum_weights = jnp.cumsum(sorted_weights)

        # Normalize the cumulative weights to get the weighted ECDF
        ecdf = cum_weights / cum_weights[-1]

        # Interpolate the data at the quantiles
        self.values = jnp.interp(self.quantiles, ecdf, sorted_data)

        self.density = jit(vmap(grad(self.cdf)))

    def get_params(self) -> Array: 
        return jnp.concatenate([self.quantiles, self.values])

    def set_params(self, params : Array) -> None:
        self.quantiles = params[:len(params)//2]
        self.values = params[len(params)//2:]
        self.density = jit(vmap(grad(self.cdf)))

    def pdf(self, x : Array) -> Array:
        return self.density(x)

    def cdf(self, x : Array) -> Array:
        return jnp.interp(x, self.values, self.quantiles)
    
    def ppf(self, q : Array) -> Array:
        return jnp.interp(q, self.quantiles, self.values)
    
    def mode(self) -> Array:
        return self.values[jnp.argmax(jnp.diff(self.quantiles))]
    
    def median(self) -> Array:
        return self.ppf(0.5)
    
    def mean(self) -> Array:
        def integrand(x):
            return x * self.density(x)

        return jnp.trapezoid(integrand, self.values[0], self.values[-1])

    def var(self) -> Array:
        def integrand(x):
            return x**2 * self.density(x)

        return jnp.trapezoid(integrand, self.values[0], self.values[-1]) - self.mean()**2
    
    @staticmethod
    def beta_quantiles(a=3, n=100):
        x = jnp.linspace(0, 1, n)
        return  beta.cdf(x, a, a)

    @staticmethod
    def uniform_quantiles(n=100):
        x = jnp.linspace(0, 1, n)
        return x
