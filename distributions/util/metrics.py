import jax.numpy as jnp
from jax import Array
from jax import vmap

from ..joint import Joint

def log_likelihood(joint : Joint, data : Array) -> Array:
    return jnp.sum(joint.logpdf(data)) / data.shape[0]