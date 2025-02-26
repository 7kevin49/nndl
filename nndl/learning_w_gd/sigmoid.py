import numpy as np
import jax.numpy as jnp

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def jax_sigmoid(z):
    return 1.0 / (1.0 + jnp.exp(-z))

def jax_sigmoid_prime(z):
    return jax_sigmoid(z) * (1 - jax_sigmoid(z))