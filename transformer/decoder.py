"""Transformer decoder (post-LayerNorm, as in the paper).

Decoder layer:
    x = LayerNorm(x + Dropout(MaskedMultiHeadAttention(x, x, x)))        # only previous target tokens
    x = LayerNorm(x + Dropout(MultiHeadAttention(x, enc_out, enc_out)))  # cross-attention to the source
    x = LayerNorm(x + Dropout(FeedForward(x)))
Decoder:
    x = Dropout(Embedding(tokens) * sqrt(d_model) + PositionalEncoding)
    x = N x DecoderLayer(x, enc_out)
    logits = x W_out            # or x E^T when the output projection is tied to the embedding
"""

import jax

from .attention import MultiHeadAttention
from .layers import Embedding, FeedForward, LayerNorm, dropout, init_linear, linear, split_rng


class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        self.dropout_rate = dropout_rate
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def init(self, key):
        k_sa, k_ca, k_ffn, k_n1, k_n2, k_n3 = jax.random.split(key, 6)
        return {
            "self_attn": self.self_attn.init(k_sa),
            "cross_attn": self.cross_attn.init(k_ca),
            "ffn": self.ffn.init(k_ffn),
            "norm1": self.norm1.init(k_n1),
            "norm2": self.norm2.init(k_n2),
            "norm3": self.norm3.init(k_n3),
        }

    def __call__(self, params, x, enc_out, self_mask=None, cross_mask=None, rng=None):
        """
        x:          (B, L_tgt, d_model)
        enc_out:    (B, L_src, d_model)
        self_mask:  (B, 1, L_tgt, L_tgt) causal AND target padding
        cross_mask: (B, 1, 1, L_src)     source padding
        rng:        PRNG key for dropout (training) or None (evaluation)

        Returns (x, self-attention weights (B, H, L_tgt, L_tgt), cross-attention weights (B, H, L_tgt, L_src)).
        """
        r_sa, r_ca, r_ffn = split_rng(rng, 3)
        # Sub-layer 1: masked self-attention (queries, keys, values all from the target)
        sa_out, self_weights = self.self_attn(params["self_attn"], x, x, self_mask)
        x = self.norm1(params["norm1"], x + dropout(sa_out, self.dropout_rate, r_sa))
        # Sub-layer 2: cross-attention (queries from the target, keys/values from the encoder)
        ca_out, cross_weights = self.cross_attn(params["cross_attn"], x, enc_out, cross_mask)
        x = self.norm2(params["norm2"], x + dropout(ca_out, self.dropout_rate, r_ca))
        # Sub-layer 3: feed-forward
        ffn_out = self.ffn(params["ffn"], x)
        x = self.norm3(params["norm3"], x + dropout(ffn_out, self.dropout_rate, r_ffn))
        return x, self_weights, cross_weights


class Decoder:
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, max_len=512,
                 dropout_rate=0.1, tie_output=False):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.tie_output = tie_output  # True: logits = x @ embedding^T (no separate output matrix)
        self.embedding = Embedding(vocab_size, d_model, max_len)
        self.layers = [DecoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]

    def init(self, key):
        k_emb, k_out, *k_layers = jax.random.split(key, len(self.layers) + 2)
        params = {
            "embedding": self.embedding.init(k_emb),
            "layers": [layer.init(k) for layer, k in zip(self.layers, k_layers)],
        }
        if not self.tie_output:
            params["out"] = init_linear(k_out, self.d_model, self.vocab_size)
        return params

    def embed(self, params, tokens):
        """tokens (B, L) -> (B, L, d_model): token embedding + positional encoding."""
        return self.embedding(params["embedding"], tokens)

    def __call__(self, params, tokens, enc_out, self_mask=None, cross_mask=None, rng=None):
        """
        tokens:  (B, L_tgt) int ids (shifted right: starts with <bos>)
        enc_out: (B, L_src, d_model)
        rng:     PRNG key for dropout (training) or None (evaluation)

        Returns (logits (B, L_tgt, vocab_size), list of self-attention weights, list of cross-attention weights).
        """
        r_emb, *r_layers = split_rng(rng, len(self.layers) + 1)
        x = dropout(self.embed(params, tokens), self.dropout_rate, r_emb)
        all_self, all_cross = [], []
        for layer, layer_params, r in zip(self.layers, params["layers"], r_layers):
            x, self_weights, cross_weights = layer(layer_params, x, enc_out, self_mask, cross_mask, r)
            all_self.append(self_weights)
            all_cross.append(cross_weights)
        if self.tie_output:
            logits = x @ params["embedding"].T
        else:
            logits = linear(params["out"], x)
        return logits, all_self, all_cross
