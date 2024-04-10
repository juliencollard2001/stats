import jax.numpy as jnp
from jax import Array

from .base import Joint
from ..marginal import Marginal

class Independent(Joint):
    
        def __init__(self, marginals : list[Marginal]):
            self.marginals = marginals
            self.d = len(marginals)
    
        def fit(self, data : Array, weights : Array|None = None) -> None:
            for i in range(self.d):
                X = data[:,i]
                marginal = self.marginals[i]
                marginal.fit(X, weights)
    
        def get_params(self) -> Array:
            return jnp.concatenate([marginal.get_params() for marginal in self.marginals])
    
        def set_params(self, params : Array) -> None:
            start = 0
            for marginal in self.marginals:
                end = start + marginal.get_nb_params()
                marginal.set_params(params[start:end])
                start = end
    
        def pdf(self, x : Array) -> Array:
            return jnp.prod(jnp.array([self.marginals[i].pdf(x[:,i]) for i in range(self.d)]), axis=0)
    
        def logpdf(self, x : Array) -> Array:
            return jnp.sum(jnp.array([self.marginals[i].logpdf(x[:,i]) for i in range(self.d)]), axis=0)
    
        def get_nb_params(self) -> int:
            return sum([marginal.get_nb_params() for marginal in self.marginals])