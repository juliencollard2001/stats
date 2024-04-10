import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jax import grad, jit, vmap

from .base import Copula

class Histogram(Copula):
    
    def __init__(self, d :int, nb_bins : int = 50, smoothing_sigma : float = 1.) -> None:
        self.d = d
        self.nb_bins = nb_bins
        self.bins = [nb_bins] * d
        self.H = None
        self.edges = None
        self.centers = None
        self.smoothing_sigma = smoothing_sigma
    
    def fit(self, data: Array, weights: Array | None = None) -> None:
        n,d = data.shape

        assert d == self.d

        if weights is None:
            weights = jnp.ones((n,)) / n

        self.H, self.edges = jnp.histogramdd(data, bins=self.bins, weights=weights, range=[(0, 1)] * d, density=True)
        self.H = self.gaussian_smoothing(self.H, self.smoothing_sigma)
        self.edges = jnp.array(self.edges)
        self.centers = [(self.edges[i][1:] + self.edges[i][:-1]) / 2 for i in range(d)]
    
    def get_params(self) -> Array:
        return jnp.concatenate([self.H.ravel(), self.edges.ravel()])
    
    def set_params(self, params : Array) -> None:
        b = self.nb_bins
        d = self.d
        raveled_H = params[:b**d]
        raveld_edges = params[b**d:]
        self.H = raveled_H.reshape((b,) * d)
        self.edges = raveld_edges.reshape((d, b+1))
        self.centers = [(self.edges[i][1:] + self.edges[i][:-1]) / 2 for i in range(self.d)]
    
    def pdf(self, x : Array) -> Array:
        # Interpolate the histogram (step for now)
        # Find the bin index for each dimension
        def one_x_pdf(x):
            indices = [jnp.searchsorted(self.centers[i], x[i]) for i in range(self.d)]
            return self.H[tuple(indices)]
        return vmap(one_x_pdf)(x)


    
    ### Additional methods for the histogram copula
    
    def gaussian_smoothing(self, input_array, sigma):
        # Create a Gaussian kernel
        kernel_size = int(2 * 3.0 * sigma + 1)
        x = jnp.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        x = x / sigma
        kernel_base = jnp.exp(-0.5 * x**2)
        kernel = jnp.copy(kernel_base)
        # Make it d-dimensional
        for i in range(1, self.d):
            kernel = jnp.outer(kernel, kernel_base)
        kernel = kernel / jnp.sum(kernel)

        # Convolve the input array with the Gaussian kernel
        smoothed_array = jsp.signal.convolve(input_array, kernel, mode='same')
        return smoothed_array
    
    def get_indices_and_weight(self, x, dim):
        dim_centers = self.centers[dim]
        #find the center before x
        idx = jnp.searchsorted(dim_centers, x)
        i, j = idx - 1, idx
