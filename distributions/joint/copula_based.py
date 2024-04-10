import jax.numpy as jnp
from jax import Array

from .base import Joint
from ..marginal import Marginal
from ..copula import Copula

class CopulaBased(Joint):
    def __init__(self, copula : Copula, marginals : list[Marginal]):
        self.copula = copula
        self.marginals = marginals
        self.d = len(marginals)

    def fit(self, data : Array, weights : Array|None = None) -> None:
        data_transf = jnp.zeros_like(data)
        for i in range(self.d):
            X = data[:,i]
            marginal = self.marginals[i]
            marginal.fit(X, weights)
            C = marginal.cdf(X)
            data_transf = data_transf.at[:, i].set(C)
        self.copula.fit(data_transf, weights)

    def to_copula(self, data : Array) -> Array:
        data_transf = jnp.zeros_like(data)
        for i in range(self.d):
            X = data[:,i]
            marginal = self.marginals[i]
            C = marginal.cdf(X)
            data_transf = data_transf.at[:, i].set(C)
        return data_transf


    def get_params(self) -> Array:
        return jnp.concatenate([self.copula.get_params(), jnp.concatenate([marginal.get_params() for marginal in self.marginals])])

    def set_params(self, params : Array) -> None:
        copula_params = self.copula.get_params()
        self.copula.set_params(params[:len(copula_params)])
        start = len(copula_params)
        for marginal in self.marginals:
            end = start + marginal.get_nb_params()
            marginal.set_params(params[start:end])
            start = end

    def pdf(self, x : Array) -> Array:
        c = jnp.column_stack([marginal.cdf(x[:,i]) for i, marginal in enumerate(self.marginals)])
        return self.copula.pdf(c) * jnp.prod(jnp.array([marginal.pdf(x[:,i]) for i, marginal in enumerate(self.marginals)]), axis=0)

    def logpdf(self, x : Array) -> Array:
        c = jnp.column_stack([marginal.cdf(x[:,i]) for i, marginal in enumerate(self.marginals)])
        return self.copula.logpdf(c) + jnp.sum(jnp.array([marginal.logpdf(x[:,i]) for i, marginal in enumerate(self.marginals)]), axis=0)

    def get_nb_params(self) -> int:
        return len(self.get_params())