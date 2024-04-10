import jax.numpy as jnp
from jax import Array
from jax import vmap

from .base import Joint

class Gaussian(Joint):
        
    def __init__(self, d : int):
        self.d = d
        self.mu = jnp.zeros(d)
        self.sigma = jnp.eye(d)

    def fit(self, data : Array, weights : Array|None = None) -> None:
        self.mu = jnp.mean(data, axis=0)
        self.sigma = jnp.cov(data, rowvar=False)

    def get_params(self) -> Array:
        return jnp.concatenate([self.mu, self.sigma.ravel()])

    def set_params(self, params : Array) -> None:
        self.mu = params[:self.d]
        self.sigma = jnp.reshape(params[self.d:], (self.d, self.d))

    def pdf(self, x : Array) -> Array:
        def one_x_pdf(x):
            return jnp.exp(-0.5 * (x - self.mu).T @ jnp.linalg.inv(self.sigma) @ (x - self.mu)) / jnp.sqrt(jnp.linalg.det(2*jnp.pi*self.sigma))
        vectorised_one_x_pdf = vmap(one_x_pdf)
        return vectorised_one_x_pdf(x)

    def logpdf(self, x : Array) -> Array:
        def one_x_logpdf(x):
            return -0.5 * (x - self.mu).T @ jnp.linalg.inv(self.sigma) @ (x - self.mu) - 0.5*jnp.log(jnp.linalg.det(2*jnp.pi*self.sigma))
        vectorised_one_x_logpdf = vmap(one_x_logpdf)
        return vectorised_one_x_logpdf(x)

    def get_nb_params(self) -> int:
        return len(self.get_params())