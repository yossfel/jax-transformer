"""Scaled dot-product attention, multi-head attention and attention masks."""

import jax
import jax.numpy as jnp

from .layers import init_linear, linear


# --------------------------------------------------------------------------- #
# Scaled dot-product attention
#   Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
# --------------------------------------------------------------------------- #
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q: (..., L_q, d_k)
    k: (..., L_k, d_k)
    v: (..., L_k, d_v)
    mask: broadcastable to (..., L_q, L_k); True = "may attend", False = "blocked"

    Returns (output (..., L_q, d_v), weights (..., L_q, L_k)).
    """
    d_k = q.shape[-1]
    # How similar is every query to every key?  (..., L_q, L_k)
    scores = q @ jnp.swapaxes(k, -1, -2) / jnp.sqrt(d_k)
    if mask is not None:
        # Blocked positions get a huge negative score -> ~0 after softmax.
        scores = jnp.where(mask, scores, -1e9)
    # Each row (one query) becomes a probability distribution over the keys.
    weights = jax.nn.softmax(scores, axis=-1)
    # Output = weighted average of the values.
    return weights @ v, weights


# --------------------------------------------------------------------------- #
# Multi-head attention
#   head_i = Attention(Q W_q^i, K W_k^i, V W_v^i)
#   MultiHead(Q, K, V) = Concat(head_1..head_h) W_o
# --------------------------------------------------------------------------- #
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

    def init(self, key):
        kq, kk, kv, ko = jax.random.split(key, 4)
        return {
            "w_q": init_linear(kq, self.d_model, self.d_model),
            "w_k": init_linear(kk, self.d_model, self.d_model),
            "w_v": init_linear(kv, self.d_model, self.d_model),
            "w_o": init_linear(ko, self.d_model, self.d_model),
        }

    def split_heads(self, x):
        """(B, L, d_model) -> (B, H, L, d_k)"""
        B, L, _ = x.shape
        x = x.reshape(B, L, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def merge_heads(self, x):
        """(B, H, L, d_k) -> (B, L, d_model)"""
        B, H, L, d_k = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, L, H * d_k)

    def __call__(self, params, x_q, x_kv, mask=None):
        """
        x_q:  (B, L_q, d_model)  - where the queries come from
        x_kv: (B, L_k, d_model)  - where keys/values come from
              (self-attention: x_q is x_kv; cross-attention in the decoder: x_kv = encoder output)
        mask: broadcastable to (B, H, L_q, L_k)

        Returns (output (B, L_q, d_model), weights (B, H, L_q, L_k)).
        """
        q = self.split_heads(linear(params["w_q"], x_q))
        k = self.split_heads(linear(params["w_k"], x_kv))
        v = self.split_heads(linear(params["w_v"], x_kv))

        out, weights = scaled_dot_product_attention(q, k, v, mask)

        out = linear(params["w_o"], self.merge_heads(out))
        return out, weights


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #
def make_padding_mask(tokens, pad_id=0):
    """tokens (B, L) -> (B, 1, 1, L): every query may attend to every non-pad key."""
    return (tokens != pad_id)[:, None, None, :]


def make_causal_mask(length):
    """(1, 1, L, L) lower-triangular: position i may only attend to positions <= i (used by the decoder)."""
    return jnp.tril(jnp.ones((length, length), dtype=bool))[None, None, :, :]
