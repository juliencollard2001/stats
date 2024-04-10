import jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt

from ..joint import Joint, CopulaBased

def plot_marginals(model : CopulaBased, data : Array) -> None:
    d = model.d
    fig, axs = plt.subplots(d, 2, figsize=(20, 5*d))
    for i in range(d):
        X = data[:,i]
        marginal = model.marginals[i]
        x = jnp.linspace(jnp.min(X), jnp.max(X), 100)
        y = marginal.pdf(x)
        axs[i,0].plot(x, y)
        axs[i,0].hist(X, bins=50, density=True, alpha=0.5)
        axs[i,0].set_title(f'Marginal {i+1} PDF')
        y = marginal.cdf(x)
        axs[i,1].plot(x, y)
        axs[i,1].hist(X, bins=50, density=True, alpha=0.5, cumulative=True)
        axs[i,1].set_title(f'Marginal {i+1} CDF')
    plt.show()

