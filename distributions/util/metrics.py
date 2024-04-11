import jax.numpy as jnp
from jax import Array
from jax import vmap

from ..joint import Joint

def log_likelihood(joint : Joint, data : Array) -> Array:
    return jnp.sum(joint.logpdf(data))

def BIC(joint : Joint, data : Array):
    LL, k, N = log_likelihood(joint, data), joint.get_nb_params(), data.shape[0]
    return - 2 * LL + jnp.log(N) * k

def AIC(joint : Joint, data : Array):
    LL, k, N = log_likelihood(joint, data), joint.get_nb_params(), data.shape[0]
    return - 2/N * LL + 2 * k / N