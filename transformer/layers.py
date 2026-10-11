"""Building blocks shared by the encoder and the decoder.

Convention used in this package: a module is a small Python class that stores its hyperparameters.
    init(key)              -> parameters (a dict, or an array for Embedding)
    __call__(params, ...)  -> pure function of the parameters and the inputs
Keeping the parameters outside the class lets jax.jit and jax.grad work on everything directly.
"""

import jax
import jax.numpy as jnp


def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


# --------------------------------------------------------------------------- #
# Linear layer
# --------------------------------------------------------------------------- #
def init_linear(key, d_in, d_out):
    w = jax.nn.initializers.glorot_uniform()(key, (d_in, d_out))
    b = jnp.zeros((d_out,))
    return {"w": w, "b": b}


def linear(params, x):
    return x @ params["w"] + params["b"]


# --------------------------------------------------------------------------- #
# Dropout (paper: rate 0.1 on every sub-layer output and on embeddings + PE)
# --------------------------------------------------------------------------- #
def dropout(x, rate, rng):
    """Inverted dropout. rng=None means evaluation mode: x is returned unchanged."""
    if rng is None or rate == 0.0:
        return x
    keep = jax.random.bernoulli(rng, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


def split_rng(rng, n):
    """Like jax.random.split, but passes None (evaluation mode) through."""
    return [None] * n if rng is None else list(jax.random.split(rng, n))


# --------------------------------------------------------------------------- #
# Positional encoding (fixed, not learned)
#   PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
# --------------------------------------------------------------------------- #
def positional_encoding(max_len, d_model):
    pos = jnp.arange(max_len)[:, None]                       # (L, 1)
    i = jnp.arange(0, d_model, 2)[None, :]                    # (1, d_model/2)
    angles = pos / jnp.power(10000.0, i / d_model)            # (L, d_model/2)
    pe = jnp.zeros((max_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(angles))
    pe = pe.at[:, 1::2].set(jnp.cos(angles))
    return pe


class Embedding:
    """Token embedding scaled by sqrt(d_model), plus the positional encoding. Parameters: one (vocab, d_model) array."""

    def __init__(self, vocab_size, d_model, max_len):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pe = positional_encoding(max_len, d_model)

    def init(self, key):
        return jax.random.normal(key, (self.vocab_size, self.d_model)) * self.d_model**-0.5

    def __call__(self, table, tokens):
        """tokens (B, L) -> (B, L, d_model)"""
        L = tokens.shape[1]
        return table[tokens] * jnp.sqrt(self.d_model) + self.pe[:L]


class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps

    def init(self, key):
        del key  # no randomness needed
        return {"gamma": jnp.ones((self.d_model,)), "beta": jnp.zeros((self.d_model,))}

    def __call__(self, params, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_hat = (x - mean) / jnp.sqrt(var + self.eps)
        return params["gamma"] * x_hat + params["beta"]


class FeedForward:
    """Position-wise FFN: max(0, x W1 + b1) W2 + b2, applied to every token independently."""

    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff

    def init(self, key):
        k1, k2 = jax.random.split(key)
        return {
            "w1": init_linear(k1, self.d_model, self.d_ff),
            "w2": init_linear(k2, self.d_ff, self.d_model),
        }

    def __call__(self, params, x):
        return linear(params["w2"], jax.nn.relu(linear(params["w1"], x)))
