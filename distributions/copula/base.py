from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import Array

class Copula(ABC):

    @abstractmethod
    def fit(self, data : Array, weights : Array|None = None) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Array: 
        pass

    @abstractmethod
    def set_params(self, params : Array) -> None:
        pass

    @abstractmethod
    def pdf(self, x : Array) -> Array:
        pass

    def logpdf(self, x : Array) -> Array:
        return jnp.log(self.pdf(x))
    
    def get_nb_params(self) -> int:
        return len(self.get_params())
