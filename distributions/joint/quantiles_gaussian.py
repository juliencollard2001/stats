import jax.numpy as jnp
from jax import Array

from .base import Joint
from .copula_based import CopulaBased
from ..marginal import Quantile
from ..copula import GaussianMixture

class QuantilesGaussian(CopulaBased):
    def __init__(self, d: int, quantiles : Array | None = None, nb_components : int | None = None) -> None:
        self.d = d
        if quantiles is None:
            marginals = [Quantile() for _ in range(d)]
        else:
            marginals = [Quantile(quantiles) for _ in range(d)]
        if nb_components is None:
            nb_components = 5*d
        copula = GaussianMixture(d, nb_components)
        super().__init__(copula, marginals)