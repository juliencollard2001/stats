import jax.numpy as jnp
from jax import Array

from .base import Joint
from .copula_based import CopulaBased
from ..marginal import Quantile
from ..copula import GaussianMixture, Copula

class QuantilesCopula(CopulaBased):
    def __init__(self, d: int, copula : Copula, quantiles : Array | None = None) -> None:
        self.d = d
        if quantiles is None:
            marginals = [Quantile() for _ in range(d)]
        else:
            marginals = [Quantile(Quantile.beta_quantiles()) for _ in range(d)]
        super().__init__(copula, marginals)