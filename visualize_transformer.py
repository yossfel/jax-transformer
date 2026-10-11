"""Graph of the whole encoder-decoder network + step-by-step decoder visualization.

Run:  uv run visualize_transformer.py

Figures (in figures/):
    07_architecture.png        every module of the network, with parameter counts
    08_parameter_count.png     where the parameters live
    09_decoder_masks.png       causal mask, target padding mask, and their combination
    10_decoder_self_attention.png
    11_decoder_cross_attention.png
    12_output_distribution.png

The model is untrained (random weights): look at shapes and masks, not at patterns.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, Patch

from transformer import Transformer, count_params, make_causal_mask, make_padding_mask
from viz import COLORS, FIG_DIR, GRID, INK, MUTED, heatmap, save

# Small config. VOCAB_SIZE = 68 matches a char-level Tiny Shakespeare tokenizer (65 chars + 3 specials).
VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS = 68, 64, 4, 128, 2

model = Transformer(VOCAB_SIZE, VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, max_len=256)  # separate embeddings
params = model.init(jax.random.PRNGKey(0))

KIND_LABELS = {
    "embedding": "Embedding + positional encoding",
    "attention": "Multi-head attention",
    "norm": "Add (residual) & LayerNorm",
    "ffn": "Feed-forward",
    "output": "Output projection",
}


# --------------------------------------------------------------------------- #
# 07: architecture graph
# --------------------------------------------------------------------------- #
def draw_block(ax, x, y, w, h, kind, text):
    color = COLORS[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=to_rgba(color, 0.22), ec=color, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, color=INK)
    return y + h


def arrow(ax, x0, y0, x1, y1, color=MUTED, **kw):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1, shrinkA=0, shrinkB=0, **kw))


def draw_architecture():
    enc_p, dec_p = params["encoder"], params["decoder"]
    W, ATT_H, FFN_H, NORM_H, GAP = 4.2, 0.75, 0.75, 0.38, 0.28
    EX, DX = 0.4, 7.2                     # left edge of the encoder / decoder columns
    BUS_X = (EX + W + DX) / 2             # vertical connector carrying the encoder output
    attn_txt = f"{NUM_HEADS} heads x d_k={D_MODEL // NUM_HEADS}  ·  W_q W_k W_v W_o: {D_MODEL}x{D_MODEL}"
    ffn_txt = f"{D_MODEL} -> {D_FF} -> ReLU -> {D_MODEL}"

    def enc_blocks(lp):
        return [("attention", ATT_H, f"Self-attention  ({count_params(lp['attn']):,})\n{attn_txt}"),
                ("norm", NORM_H, f"Add & LayerNorm  ({count_params(lp['norm1']):,})"),
                ("ffn", FFN_H, f"Feed-forward  ({count_params(lp['ffn']):,})\n{ffn_txt}"),
                ("norm", NORM_H, f"Add & LayerNorm  ({count_params(lp['norm2']):,})")]

    def dec_blocks(lp):
        return [("attention", ATT_H, f"Masked self-attention  ({count_params(lp['self_attn']):,})\n"
                                     "causal + padding mask"),
                ("norm", NORM_H, f"Add & LayerNorm  ({count_params(lp['norm1']):,})"),
                ("attention", ATT_H, f"Cross-attention  ({count_params(lp['cross_attn']):,})\n"
                                     "Q from decoder, K V from encoder"),
                ("norm", NORM_H, f"Add & LayerNorm  ({count_params(lp['norm2']):,})"),
                ("ffn", FFN_H, f"Feed-forward  ({count_params(lp['ffn']):,})\n{ffn_txt}"),
                ("norm", NORM_H, f"Add & LayerNorm  ({count_params(lp['norm3']):,})")]

    # Every layer gets the height of a decoder layer, so encoder layer i and decoder layer i line up.
    layer_h = sum(h for _, h, _ in dec_blocks(dec_p["layers"][0])) + 5 * GAP

    def draw_layer(x, y0, blocks):
        """Draw blocks spread evenly over layer_h. Returns (top y, list of block bottoms)."""
        gap = (layer_h - sum(h for _, h, _ in blocks)) / (len(blocks) - 1)
        cx, y, bottoms = x + W / 2, y0, []
        for j, (kind, h, text) in enumerate(blocks):
            if j:
                arrow(ax, cx, y, cx, y + gap)
                y += gap
            bottoms.append(y)
            y = draw_block(ax, x, y, W, h, kind, text)
        return y, bottoms

    def layer_group(x, y0, y1, title, side):
        ax.add_patch(FancyBboxPatch((x - 0.25, y0 - 0.15), W + 0.5, y1 - y0 + 0.3,
                                    boxstyle="round,pad=0,rounding_size=0.15", fc="none", ec=MUTED, lw=1, ls="--"))
        if side == "left":
            ax.text(x - 0.35, (y0 + y1) / 2, title, rotation=90, ha="right", va="center", fontsize=8.5, color=MUTED)
        else:
            ax.text(x + W + 0.35, (y0 + y1) / 2, title, rotation=270, ha="left", va="center", fontsize=8.5, color=MUTED)

    def draw_column(x, side, label, side_p, blocks_fn, name):
        """Input label, embedding and layers. Returns (top y, list of block bottoms per layer)."""
        cx = x + W / 2
        ax.text(cx, 0.0, label, ha="center", fontsize=9, color=INK)
        arrow(ax, cx, 0.3, cx, 0.6)
        y = draw_block(ax, x, 0.6, W, ATT_H, "embedding",
                       f"Embedding {VOCAB_SIZE}x{D_MODEL} ({count_params(side_p['embedding']):,})\n"
                       "x sqrt(d_model) + sinusoidal PE (0 params)")
        all_bottoms = []
        for i, lp in enumerate(side_p["layers"]):
            y0 = y + GAP + 0.25
            arrow(ax, cx, y, cx, y0)
            top, bottoms = draw_layer(x, y0, blocks_fn(lp))
            all_bottoms.append(bottoms)
            layer_group(x, y0, top, f"{name} layer {i}", side)
            y = top + 0.15
        return y, all_bottoms

    fig, ax = plt.subplots(figsize=(13, 10))
    enc_top, _ = draw_column(EX, "left", "source tokens (B, L_src)", enc_p, enc_blocks, "Encoder")
    dec_top, dec_bottoms = draw_column(DX, "right", "target tokens shifted right: <bos> + tgt[:-1]",
                                       dec_p, dec_blocks, "Decoder")

    # Encoder output: up, right to the connector, down, then into every cross-attention block.
    cx = EX + W / 2
    out_y = enc_top + GAP + 0.25
    ax.plot([cx, cx, BUS_X, BUS_X], [enc_top, out_y, out_y, dec_bottoms[0][2] + ATT_H / 2],
            color=COLORS["attention"], lw=1.2)
    for bottoms in dec_bottoms:
        yc = bottoms[2] + ATT_H / 2  # block 2 of a decoder layer is the cross-attention
        arrow(ax, BUS_X, yc, DX, yc, color=COLORS["attention"])
    ax.text(cx, out_y + 0.12, f"encoder output (B, L_src, {D_MODEL})\nkeys and values for cross-attention",
            ha="center", va="bottom", fontsize=9, color=INK)

    # Decoder output projection
    cx = DX + W / 2
    proj_y = dec_top + GAP + 0.25
    arrow(ax, cx, dec_top, cx, proj_y)
    top = draw_block(ax, DX, proj_y, W, ATT_H, "output",
                     f"Linear {D_MODEL} -> {VOCAB_SIZE}  ({count_params(dec_p['out']):,})\nlogits -> softmax")
    arrow(ax, cx, top, cx, top + 0.35)
    ax.text(cx, top + 0.42, f"next-token probabilities (B, L_tgt, {VOCAB_SIZE})",
            ha="center", va="bottom", fontsize=9, color=INK)

    y_max = top + 0.9
    ax.set_xlim(-0.6, DX + W + 0.6)
    ax.set_ylim(-0.4, y_max)
    fig.set_size_inches(13, (y_max + 0.4) * 0.82 + 0.6)  # height follows the content
    ax.axis("off")
    ax.set_title(f"Encoder-decoder Transformer  ·  d_model={D_MODEL}, heads={NUM_HEADS}, d_ff={D_FF}, "
                 f"layers={NUM_LAYERS}  ·  {count_params(params):,} parameters", fontsize=11, color=INK, loc="left")
    handles = [Patch(fc=to_rgba(COLORS[k], 0.22), ec=COLORS[k], label=KIND_LABELS[k]) for k in KIND_LABELS]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False, bbox_to_anchor=(0, 1.0))
    return fig


# --------------------------------------------------------------------------- #
# 08: parameter count per module
# --------------------------------------------------------------------------- #
def draw_param_count():
    rows = [("encoder", "Embedding", params["encoder"]["embedding"])]
    for i, lp in enumerate(params["encoder"]["layers"]):
        rows += [("encoder", f"L{i} self-attention", lp["attn"]), ("encoder", f"L{i} feed-forward", lp["ffn"]),
                 ("encoder", f"L{i} LayerNorms", [lp["norm1"], lp["norm2"]])]
    rows.append(("decoder", "Embedding", params["decoder"]["embedding"]))
    for i, lp in enumerate(params["decoder"]["layers"]):
        rows += [("decoder", f"L{i} masked self-attention", lp["self_attn"]),
                 ("decoder", f"L{i} cross-attention", lp["cross_attn"]),
                 ("decoder", f"L{i} feed-forward", lp["ffn"]),
                 ("decoder", f"L{i} LayerNorms", [lp["norm1"], lp["norm2"], lp["norm3"]])]
    rows.append(("decoder", "Output projection", params["decoder"]["out"]))

    side_color = {"encoder": COLORS["attention"], "decoder": COLORS["ffn"]}
    counts = [count_params(p) for _, _, p in rows]
    fig, ax = plt.subplots(figsize=(9, 7))
    ys = range(len(rows))[::-1]
    ax.barh(list(ys), counts, height=0.7, color=[side_color[s] for s, _, _ in rows], edgecolor="white", linewidth=2)
    ax.set_yticks(list(ys), [f"{s[:3]}. {name}" for s, name, _ in rows], fontsize=8.5, color=INK)
    for yy, c in zip(ys, counts):
        ax.text(c + max(counts) * 0.01, yy, f"{c:,}", va="center", fontsize=8, color=MUTED)
    ax.set_xlabel("parameters", color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(counts) * 1.15)
    ax.set_title(f"Parameters per module  ·  total {sum(counts):,}", loc="left", color=INK)
    ax.legend(handles=[Patch(color=side_color["encoder"], label="Encoder"),
                       Patch(color=side_color["decoder"], label="Decoder")], frameon=False, loc="lower right")
    return fig


print(f"[7] Architecture graph ({count_params(params):,} parameters)")
save(draw_architecture(), "07_architecture.png")
print("[8] Parameter count per module")
save(draw_param_count(), "08_parameter_count.png")

# --------------------------------------------------------------------------- #
# Toy translation example for the decoder (ids are just labels here)
# --------------------------------------------------------------------------- #
src_words = ["the", "cat", "sat", "on", "the", "mat", "<pad>", "<pad>"]
tgt_words = ["le", "chat", "s'est", "assis", "<eos>", "<pad>", "<pad>"]
words = sorted(set(src_words + tgt_words) - {"<pad>", "<eos>"})
vocab = {w: i for i, w in enumerate(["<pad>", "<bos>", "<eos>"] + words)}
src = jnp.array([[vocab[w] for w in src_words]])
tgt_in_words = ["<bos>"] + tgt_words[:-1]         # shifted right
tgt_in = jnp.array([[vocab[w] for w in tgt_in_words]])
tgt_out = jnp.array([[vocab[w] for w in tgt_words]])

# --------------------------------------------------------------------------- #
# 09: decoder masks
# --------------------------------------------------------------------------- #
print("\n[9] Decoder masks")
src_mask, tgt_mask = model.make_masks(src, tgt_in)
causal = make_causal_mask(tgt_in.shape[1])[0, 0]
pad = jnp.broadcast_to(make_padding_mask(tgt_in)[0, 0], causal.shape)
print(f"   src mask {src_mask.shape}, tgt mask {tgt_mask.shape}")
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
heatmap(axes[0], causal, "Causal mask: position i sees only <= i", tgt_in_words, tgt_in_words, vmin=0, vmax=1)
heatmap(axes[1], pad, "Target padding mask", tgt_in_words, tgt_in_words, vmin=0, vmax=1)
heatmap(axes[2], tgt_mask[0, 0], "Combined = causal AND padding", tgt_in_words, tgt_in_words, vmin=0, vmax=1)
for ax in axes:
    ax.set_xlabel("key (attended to)")
    ax.set_ylabel("query (attending)")
save(fig, "09_decoder_masks.png")

# --------------------------------------------------------------------------- #
# 10 + 11: decoder self-attention and cross-attention
# --------------------------------------------------------------------------- #
print("\n[10/11] Full forward pass (jit)")
forward = jax.jit(model.__call__)
logits, attn = forward(params, src, tgt_in)
print(f"   logits: {logits.shape}  (B, L_tgt, vocab)")
for i in range(NUM_LAYERS):
    print(f"   layer {i}: decoder self-attn {attn['decoder_self'][i].shape}, "
          f"cross-attn {attn['decoder_cross'][i].shape}")

fig, axes = plt.subplots(NUM_LAYERS, NUM_HEADS, figsize=(5 * NUM_HEADS, 4.5 * NUM_LAYERS))
for layer in range(NUM_LAYERS):
    for h in range(NUM_HEADS):
        heatmap(axes[layer, h], attn["decoder_self"][layer][0, h], f"Decoder self-attn  L{layer} head {h}",
                tgt_in_words, tgt_in_words, vmin=0, vmax=1)
save(fig, "10_decoder_self_attention.png")

fig, axes = plt.subplots(NUM_LAYERS, NUM_HEADS, figsize=(5 * NUM_HEADS, 4.5 * NUM_LAYERS))
for layer in range(NUM_LAYERS):
    for h in range(NUM_HEADS):
        ax = axes[layer, h]
        heatmap(ax, attn["decoder_cross"][layer][0, h], f"Cross-attn  L{layer} head {h}",
                src_words, tgt_in_words, vmin=0, vmax=1)
        ax.set_xlabel("source (encoder) token")
        ax.set_ylabel("target (decoder) token")
save(fig, "11_decoder_cross_attention.png")

# --------------------------------------------------------------------------- #
# 12: output distribution + initial loss sanity check
# --------------------------------------------------------------------------- #
print("\n[12] Output distribution")
log_probs = jax.nn.log_softmax(logits, axis=-1)
nll = -jnp.take_along_axis(log_probs, tgt_out[..., None], axis=-1)[..., 0]
non_pad = tgt_out != 0
loss = (nll * non_pad).sum() / non_pad.sum()
print(f"   cross-entropy at init: {loss:.3f}   (uniform guess would be ln({VOCAB_SIZE}) = {jnp.log(VOCAB_SIZE):.3f})")

pos = tgt_in_words.index("chat")
probs = jnp.exp(log_probs[0, pos])
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.bar(range(VOCAB_SIZE), probs, width=0.8, color=COLORS["attention"])
ax.axhline(1 / VOCAB_SIZE, color=MUTED, ls="--", lw=1)
ax.text(VOCAB_SIZE - 0.5, 1 / VOCAB_SIZE, " uniform 1/V", va="bottom", ha="right", fontsize=8, color=MUTED)
ax.set_xlabel("token id", color=MUTED)
ax.set_ylabel("probability", color=MUTED)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title(f"P(next token | decoder input up to '{tgt_in_words[pos]}')  ·  untrained, so close to uniform",
             loc="left", fontsize=10, color=INK)
save(fig, "12_output_distribution.png")

print(f"\nDone. Figures are in {FIG_DIR}")
