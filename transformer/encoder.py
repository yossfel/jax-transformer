"""Transformer encoder (post-LayerNorm, as in the paper).

Encoder layer:
    x = LayerNorm(x + Dropout(MultiHeadAttention(x, x, x)))
    x = LayerNorm(x + Dropout(FeedForward(x)))
Encoder:
    x = Dropout(Embedding(tokens) * sqrt(d_model) + PositionalEncoding)
    x = N x EncoderLayer(x)

`rng` is a PRNG key while training (dropout on) and None for evaluation (dropout off).
"""

import jax

from .attention import MultiHeadAttention
from .layers import Embedding, FeedForward, LayerNorm, dropout, split_rng


class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        self.dropout_rate = dropout_rate
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def init(self, key):
        k_attn, k_ffn, k_n1, k_n2 = jax.random.split(key, 4)
        return {
            "attn": self.attn.init(k_attn),
            "ffn": self.ffn.init(k_ffn),
            "norm1": self.norm1.init(k_n1),
            "norm2": self.norm2.init(k_n2),
        }

    def __call__(self, params, x, mask=None, rng=None):
        r_attn, r_ffn = split_rng(rng, 2)
        # Sub-layer 1: self-attention + residual + norm
        attn_out, weights = self.attn(params["attn"], x, x, mask)
        x = self.norm1(params["norm1"], x + dropout(attn_out, self.dropout_rate, r_attn))
        # Sub-layer 2: feed-forward + residual + norm
        ffn_out = self.ffn(params["ffn"], x)
        x = self.norm2(params["norm2"], x + dropout(ffn_out, self.dropout_rate, r_ffn))
        return x, weights


class Encoder:
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, max_len=512,
                 dropout_rate=0.1):
        self.dropout_rate = dropout_rate
        self.embedding = Embedding(vocab_size, d_model, max_len)
        self.layers = [EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]

    def init(self, key):
        k_emb, *k_layers = jax.random.split(key, len(self.layers) + 1)
        return {
            "embedding": self.embedding.init(k_emb),
            "layers": [layer.init(k) for layer, k in zip(self.layers, k_layers)],
        }

    def embed(self, params, tokens):
        """tokens (B, L) -> (B, L, d_model): token embedding + positional encoding."""
        return self.embedding(params["embedding"], tokens)

    def __call__(self, params, tokens, mask=None, rng=None):
        """
        tokens: (B, L) int ids
        mask:   (B, 1, 1, L) padding mask, see attention.make_padding_mask

        Returns (output (B, L, d_model), list of attention weights per layer, each (B, H, L, L)).
        """
        r_emb, *r_layers = split_rng(rng, len(self.layers) + 1)
        x = dropout(self.embed(params, tokens), self.dropout_rate, r_emb)
        all_weights = []
        for layer, layer_params, r in zip(self.layers, params["layers"], r_layers):
            x, weights = layer(layer_params, x, mask, r)
            all_weights.append(weights)
        return x, all_weights
