"""Step-by-step walkthrough of attention and the encoder, with figures saved to figures/.

Run:  uv run visualize_encoder.py

The model is untrained (random weights), so the attention patterns are not
"meaningful" yet - the point is to see the shapes, the mechanics and the masks.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from transformer import (
    Encoder,
    make_padding_mask,
    positional_encoding,
    scaled_dot_product_attention,
)
from transformer.layers import linear
from viz import FIG_DIR, heatmap, save

# --------------------------------------------------------------------------- #
# Toy data
# --------------------------------------------------------------------------- #
words = ["the", "cat", "sat", "on", "the", "mat", "<pad>", "<pad>"]
vocab = {"<pad>": 0, "the": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5}
tokens = jnp.array([[vocab[w] for w in words]])  # (B=1, L=8)

D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS = 64, 4, 128, 2

encoder = Encoder(len(vocab), D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, max_len=100)
params = encoder.init(jax.random.PRNGKey(0))
mask = make_padding_mask(tokens)


print(f"tokens: {tokens.shape} {tokens.tolist()}")

# --------------------------------------------------------------------------- #
# Step 1: embeddings + positional encoding
# --------------------------------------------------------------------------- #
print("\n[1] Embedding + positional encoding")
pe = positional_encoding(100, D_MODEL)
emb = params["embedding"][tokens[0]] * jnp.sqrt(D_MODEL)
x = encoder.embed(params, tokens)  # (1, L, d_model)
print(f"   positional encoding: {pe.shape}, embedded input x: {x.shape}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
heatmap(axes[0], pe, "Positional encoding (100 positions x d_model)", cmap="RdBu")
axes[0].set_xlabel("dimension")
axes[0].set_ylabel("position")
heatmap(axes[1], emb, "Token embeddings (note: both 'the' rows are identical)", ylabels=words, cmap="RdBu")
heatmap(axes[2], x[0], "Embedding + PE (now the two 'the' differ)", ylabels=words, cmap="RdBu")
save(fig, "01_embedding_positional_encoding.png")

# --------------------------------------------------------------------------- #
# Step 2: scaled dot-product attention, opened up (layer 0, head 0)
# --------------------------------------------------------------------------- #
print("\n[2] Scaled dot-product attention internals (layer 0, head 0)")
mha = encoder.layers[0].attn
p_attn = params["layers"][0]["attn"]

q = mha.split_heads(linear(p_attn["w_q"], x))  # (B, H, L, d_k)
k = mha.split_heads(linear(p_attn["w_k"], x))
v = mha.split_heads(linear(p_attn["w_v"], x))
print(f"   Q, K, V after split_heads: {q.shape}  (B, H, L, d_k)")

q0, k0, v0 = q[0, 0], k[0, 0], v[0, 0]
scores = q0 @ k0.T / jnp.sqrt(mha.d_k)
weights_nomask = jax.nn.softmax(scores, axis=-1)
out0, weights0 = scaled_dot_product_attention(q0, k0, v0, mask[0, 0])
print(f"   scores: {scores.shape}, weights: {weights0.shape}, output: {out0.shape}")
print(f"   every row of the weights sums to 1: {jnp.allclose(weights0.sum(-1), 1.0)}")

fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
heatmap(axes[0], q0, "Q = x W_q  (L x d_k)", ylabels=words, cmap="RdBu")
heatmap(axes[1], k0, "K = x W_k  (L x d_k)", ylabels=words, cmap="RdBu")
heatmap(axes[2], scores, "Q K^T / sqrt(d_k)", words, words, cmap="RdBu")
heatmap(axes[3], weights0, "softmax (with padding mask)", words, words)
heatmap(axes[4], out0, "output = weights @ V", ylabels=words, cmap="RdBu")
axes[3].set_xlabel("key (attended to)")
axes[3].set_ylabel("query (attending)")
save(fig, "02_attention_internals.png")

# --------------------------------------------------------------------------- #
# Step 3: what the padding mask does
# --------------------------------------------------------------------------- #
print("\n[3] Padding mask")
print(f"   mask {mask.shape}: {mask[0, 0, 0].tolist()}")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
heatmap(axes[0], weights_nomask, "No mask: queries attend to <pad>", words, words)
heatmap(axes[1], weights0, "With padding mask: <pad> columns are 0", words, words)
save(fig, "03_padding_mask.png")

# --------------------------------------------------------------------------- #
# Step 4: multi-head attention - every head has its own pattern
# --------------------------------------------------------------------------- #
print("\n[4] Multi-head attention")
mha_out, mha_weights = mha(p_attn, x, x, mask)
print(f"   output: {mha_out.shape}, weights: {mha_weights.shape}  (B, H, L_q, L_k)")
fig, axes = plt.subplots(1, NUM_HEADS, figsize=(5 * NUM_HEADS, 4.5))
for h in range(NUM_HEADS):
    heatmap(axes[h], mha_weights[0, h], f"Layer 0 - head {h}", words, words)
save(fig, "04_multi_head_attention.png")

# --------------------------------------------------------------------------- #
# Step 5: the full encoder (jit-compiled)
# --------------------------------------------------------------------------- #
print("\n[5] Full encoder")
encode = jax.jit(encoder.__call__)
enc_out, all_weights = encode(params, tokens, mask)
print(f"   encoder output: {enc_out.shape}")
for i, w in enumerate(all_weights):
    print(f"   layer {i} attention weights: {w.shape}")

fig, axes = plt.subplots(NUM_LAYERS, NUM_HEADS, figsize=(5 * NUM_HEADS, 4.5 * NUM_LAYERS))
for layer in range(NUM_LAYERS):
    for h in range(NUM_HEADS):
        heatmap(axes[layer, h], all_weights[layer][0, h], f"Layer {layer} - head {h}", words, words)
save(fig, "05_encoder_all_attention.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
heatmap(axes[0], x[0], "Encoder input (embedding + PE)", ylabels=words, cmap="RdBu")
heatmap(axes[1], enc_out[0], "Encoder output (contextual representations)", ylabels=words, cmap="RdBu")
save(fig, "06_encoder_input_vs_output.png")

print(f"\nDone. Figures are in {FIG_DIR}")
