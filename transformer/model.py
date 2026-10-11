"""Full encoder-decoder Transformer: source tokens -> encoder -> decoder -> next-token logits.

share_embeddings=True is the paper's setup for a shared vocabulary: one embedding matrix is used by
the encoder, the decoder and (transposed) as the pre-softmax output projection.
"""

import jax
import jax.numpy as jnp

from .attention import make_causal_mask, make_padding_mask
from .decoder import Decoder
from .encoder import Encoder
from .layers import split_rng


class Transformer:
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048,
                 num_layers=6, max_len=512, dropout_rate=0.1, share_embeddings=False, pad_id=0):
        if share_embeddings and src_vocab_size != tgt_vocab_size:
            raise ValueError("share_embeddings needs src_vocab_size == tgt_vocab_size")
        self.pad_id = pad_id
        self.share_embeddings = share_embeddings
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout_rate)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout_rate,
                               tie_output=share_embeddings)

    def init(self, key):
        k_enc, k_dec = jax.random.split(key)
        params = {"encoder": self.encoder.init(k_enc), "decoder": self.decoder.init(k_dec)}
        if self.share_embeddings:
            params["shared_embedding"] = params["encoder"].pop("embedding")
            del params["decoder"]["embedding"]
        return params

    def _split_params(self, params):
        """Encoder / decoder params, with the shared embedding put back where each expects it."""
        if not self.share_embeddings:
            return params["encoder"], params["decoder"]
        emb = params["shared_embedding"]
        return {**params["encoder"], "embedding": emb}, {**params["decoder"], "embedding": emb}

    def make_masks(self, src, tgt_in):
        src_mask = make_padding_mask(src, self.pad_id)                  # (B, 1, 1, L_src)
        tgt_mask = make_padding_mask(tgt_in, self.pad_id) & make_causal_mask(tgt_in.shape[1])  # (B, 1, L_tgt, L_tgt)
        return src_mask, tgt_mask

    def __call__(self, params, src, tgt_in, rng=None):
        """
        src:    (B, L_src) source token ids
        tgt_in: (B, L_tgt) decoder input = <bos> + target[:-1]
        rng:    PRNG key for dropout (training) or None (evaluation)

        Returns (logits (B, L_tgt, tgt_vocab), dict of attention weights for visualization).
        """
        enc_p, dec_p = self._split_params(params)
        r_enc, r_dec = split_rng(rng, 2)
        src_mask, tgt_mask = self.make_masks(src, tgt_in)
        enc_out, enc_weights = self.encoder(enc_p, src, src_mask, r_enc)
        logits, self_weights, cross_weights = self.decoder(dec_p, tgt_in, enc_out, tgt_mask, src_mask, r_dec)
        return logits, {"encoder": enc_weights, "decoder_self": self_weights, "decoder_cross": cross_weights}

    def generate(self, params, src, max_new_tokens, bos_id=1, temperature=0.0, rng=None):
        """
        Autoregressive decoding: encode once, then produce one target token at a time.
        temperature=0 -> greedy (argmax); otherwise sample from softmax(logits / temperature) with `rng`.
        Jit-friendly: fixed-size buffer + lax.fori_loop. Returns (B, max_new_tokens) token ids.
        """
        enc_p, dec_p = self._split_params(params)
        B = src.shape[0]
        src_mask = make_padding_mask(src, self.pad_id)
        enc_out, _ = self.encoder(enc_p, src, src_mask)
        L = max_new_tokens  # decoder input <bos> + out[:-1] has the same length as the output
        # Positions after t are still <pad>, but the causal mask already hides them from position t.
        causal = make_causal_mask(L)
        dec_in = jnp.full((B, L), self.pad_id, dtype=jnp.int32).at[:, 0].set(bos_id)
        out = jnp.zeros((B, L), dtype=jnp.int32)

        def step(t, state):
            dec_in, out = state
            logits, _, _ = self.decoder(dec_p, dec_in, enc_out, causal, src_mask)
            logits_t = logits[:, t]
            if temperature > 0.0:
                nxt = jax.random.categorical(jax.random.fold_in(rng, t), logits_t / temperature)
            else:
                nxt = jnp.argmax(logits_t, axis=-1)
            nxt = nxt.astype(jnp.int32)
            # Feed the new token back as the next decoder input (dropped on the last step: index L).
            dec_in = dec_in.at[:, t + 1].set(nxt, mode="drop")
            return dec_in, out.at[:, t].set(nxt)

        _, out = jax.lax.fori_loop(0, max_new_tokens, step, (dec_in, out))
        return out
