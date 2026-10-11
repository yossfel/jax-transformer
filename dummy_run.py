"""Dummy run: check that every piece of the Transformer is correct, then train briefly.

Run:
    uv run dummy_run.py            # parts A, B and C (about 13 min on CPU)
    uv run dummy_run.py --quick    # parts A and B only (about 1 min)

Part A - correctness checks (each prints PASS / FAIL)
Part B - overfit one batch (the model must be able to memorise it) + greedy generation
Part C - short training on Tiny Shakespeare, loss curve -> figures/13_dummy_training.png
"""

import argparse
import math
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from data import BOS_ID, PAD_ID, CharTokenizer, load_text, make_seq2seq_pairs
from train import (
    cross_entropy,
    evaluate,
    make_eval_step,
    make_optimizer,
    make_train_step,
    noam_schedule,
    random_batches,
)
from transformer import Transformer, count_params
from viz import COLORS, GRID, MUTED, save

sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser(description="Correctness checks and a short training run.")
parser.add_argument("--quick", action="store_true", help="skip part C (the training run)")
args = parser.parse_args()

# Small config for a CPU
BLOCK, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, DROPOUT = 64, 128, 4, 512, 2, 0.1
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# =========================================================================== #
print("\n=== Part A: correctness checks ===")
# =========================================================================== #
text = load_text()  # downloads + verifies size and MD5 (raises otherwise)
check("dataset is Karpathy's input.txt", len(text) == 1_115_394 and len(set(text)) == 65,
      f"{len(text):,} chars, {len(set(text))} unique, md5 verified")

tok = CharTokenizer(text)
check("tokenizer round-trip", tok.decode(tok.encode(text[:5000])) == text[:5000], f"vocab {tok.vocab_size}")

split = int(len(text) * 0.9)
train_pairs = make_seq2seq_pairs(tok.encode(text[:split]), BLOCK)
val_pairs = make_seq2seq_pairs(tok.encode(text[split:]), BLOCK)
src, tgt_in, tgt_out = train_pairs
check("seq2seq pairs are consistent",
      (tgt_in[:, 0] == BOS_ID).all() and (tgt_in[:, 1:] == tgt_out[:, :-1]).all() and (src[1:] == tgt_out[:-1]).all(),
      f"train {src.shape}, val {val_pairs[0].shape}")

model = Transformer(tok.vocab_size, tok.vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS,
                    max_len=256, dropout_rate=DROPOUT, share_embeddings=True)
params = model.init(jax.random.PRNGKey(0))
print(f"  model: {count_params(params):,} parameters")

B = 4
s, ti, to = (jnp.asarray(a[:B]) for a in train_pairs)
logits, attn = model(params, s, ti)
check("forward shapes", logits.shape == (B, BLOCK, tok.vocab_size)
      and attn["encoder"][0].shape == (B, NUM_HEADS, BLOCK, BLOCK)
      and attn["decoder_cross"][0].shape == (B, NUM_HEADS, BLOCK, BLOCK), f"logits {logits.shape}")

loss0, _ = cross_entropy(logits, to)
check("initial loss close to ln(vocab)", abs(float(loss0) - math.log(tok.vocab_size)) < 1.0,
      f"{float(loss0):.3f} vs ln({tok.vocab_size}) = {math.log(tok.vocab_size):.3f}")

# Causality: changing decoder inputs at positions > t must not change the logits at positions <= t.
t = 20
ti_changed = ti.at[:, t + 1:].set((ti[:, t + 1:] + 7) % tok.vocab_size + 3)
logits_changed, _ = model(params, s, ti_changed)
check("decoder is causal (no peeking at future tokens)",
      jnp.allclose(logits[:, :t + 1], logits_changed[:, :t + 1], atol=1e-5)
      and not jnp.allclose(logits[:, t + 1:], logits_changed[:, t + 1:]),
      "first 21 positions identical, later ones differ")

# Source padding: appending <pad> to the source must not change anything.
s_padded = jnp.concatenate([s, jnp.full((B, 16), PAD_ID, dtype=s.dtype)], axis=1)
logits_padded, _ = model(params, s_padded, ti)
check("source <pad> tokens are ignored", jnp.allclose(logits, logits_padded, atol=1e-4),
      f"max diff {float(jnp.abs(logits - logits_padded).max()):.1e}")

# Dropout: off when rng=None, on (and reproducible) when an rng is given.
k1, k2 = jax.random.PRNGKey(1), jax.random.PRNGKey(2)
la = model(params, s, ti, rng=k1)[0]
check("dropout: eval deterministic, train random but reproducible",
      jnp.allclose(model(params, s, ti)[0], logits)
      and not jnp.allclose(la, logits)
      and jnp.allclose(la, model(params, s, ti, rng=k1)[0])
      and not jnp.allclose(la, model(params, s, ti, rng=k2)[0]))

check("embeddings shared (encoder, decoder, output projection)",
      "shared_embedding" in params and "embedding" not in params["encoder"]
      and "embedding" not in params["decoder"] and "out" not in params["decoder"])


def loss_fn(p):
    return cross_entropy(model(p, s, ti, rng=k1)[0], to, label_smoothing=0.1)[0]


grads = jax.grad(loss_fn)(params)
leaves = jax.tree_util.tree_leaves_with_path(grads)
bad = [jax.tree_util.keystr(path) for path, g in leaves if not jnp.isfinite(g).all() or float(jnp.abs(g).max()) == 0.0]
check("gradients finite and non-zero for every parameter", not bad,
      f"{len(leaves)} tensors" if not bad else f"bad: {bad[:3]}")

# =========================================================================== #
print("\n=== Part B: overfit a single batch ===")
# =========================================================================== #
model_nodrop = Transformer(tok.vocab_size, tok.vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS,
                           max_len=256, dropout_rate=0.0, share_embeddings=True)
batch = tuple(jnp.asarray(a[:8]) for a in train_pairs)
opt = make_optimizer(3e-3)
step_fn = make_train_step(model_nodrop, opt, label_smoothing=0.0)
p, opt_state = params, opt.init(params)
t0 = time.time()
for i in range(300):
    p, opt_state, loss, acc, _ = step_fn(p, opt_state, batch, None)
    if i % 100 == 0 or i == 299:
        print(f"  step {i:3d}  loss {float(loss):.4f}  acc {float(acc):.3f}")
check("can memorise one batch", float(loss) < 0.1, f"final loss {float(loss):.4f} in {time.time() - t0:.1f}s")

generate = jax.jit(model_nodrop.generate, static_argnums=(2,))
gen = generate(p, batch[0], BLOCK)
gen_acc = float((gen == batch[2]).mean())
check("greedy generation reproduces the memorised targets", gen_acc > 0.95, f"token accuracy {gen_acc:.3f}")
print(f"  source   : {tok.decode(np.asarray(batch[0][0]))!r}")
print(f"  generated: {tok.decode(np.asarray(gen[0]))!r}")

# =========================================================================== #
# Part C: short training on Tiny Shakespeare
# =========================================================================== #
def train_briefly():
    print("\n=== Part C: short training on Tiny Shakespeare ===")
    STEPS, BATCH, WARMUP = 1500, 32, 300
    schedule = noam_schedule(D_MODEL, warmup_steps=WARMUP)
    opt = make_optimizer(schedule)
    train_step = make_train_step(model, opt, label_smoothing=0.1)
    eval_step = make_eval_step(model)
    p, opt_state = params, opt.init(params)
    batches = random_batches(train_pairs, BATCH)
    rng = jax.random.PRNGKey(42)

    val_before, _ = evaluate(eval_step, p, val_pairs)
    peak_lr = float(schedule(WARMUP - 1))
    print(f"  val loss before training: {val_before:.3f}   (peak lr {peak_lr:.4f} at step {WARMUP})")
    history = {"step": [], "train": [], "val_step": [], "val": []}
    t0 = time.time()
    for step in range(STEPS):
        rng, step_rng = jax.random.split(rng)
        p, opt_state, loss, acc, gnorm = train_step(p, opt_state, next(batches), step_rng)
        history["step"].append(step)
        history["train"].append(float(loss))
        if (step + 1) % 250 == 0:
            vl, va = evaluate(eval_step, p, val_pairs)
            history["val_step"].append(step)
            history["val"].append(vl)
            print(f"  step {step + 1:4d}  train {float(loss):.3f}  val {vl:.3f}  val acc {va:.3f}  "
                  f"lr {float(schedule(step)):.5f}  |grad| {float(gnorm):.2f}  {time.time() - t0:.0f}s")
    val_after = history["val"][-1]
    check("training lowers validation loss", val_after < val_before - 1.5, f"{val_before:.3f} -> {val_after:.3f}")
    check("no NaNs during training", all(np.isfinite(history["train"])))

    # Loss curve
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 25
    smooth = np.convolve(history["train"], np.ones(w) / w, mode="valid")
    ax.plot(history["step"][w - 1:], smooth, color=COLORS["attention"], lw=2,
            label="train (label-smoothed, moving avg)")
    ax.plot(history["val_step"], history["val"], color=COLORS["ffn"], lw=2, marker="o", ms=8, label="validation")
    ax.axhline(math.log(tok.vocab_size), color=MUTED, ls="--", lw=1)
    ax.text(STEPS, math.log(tok.vocab_size), " uniform guess", va="bottom", ha="right", fontsize=8, color=MUTED)
    ax.set_xlabel("step", color=MUTED)
    ax.set_ylabel("cross-entropy (nats / char)", color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False)
    ax.set_title(f"Dummy training run  ·  {count_params(params):,} params  ·  char-level Tiny Shakespeare",
                 loc="left", fontsize=10)
    save(fig, "13_dummy_training.png")

    # Sample: the model was trained to map one 64-char chunk to the next one, so generate chunk by chunk:
    # the decoder writes BLOCK chars, which become the encoder input for the next chunk.
    # (Asking for more than BLOCK chars at once would use positions the decoder never saw in training.)
    generate = jax.jit(model.generate, static_argnums=(2,), static_argnames=("temperature",))
    chunk = jnp.asarray(val_pairs[0][:1])
    print("\n  encoder input (validation chunk):\n" + tok.decode(np.asarray(chunk[0])))
    pieces = []
    for i in range(4):
        chunk = generate(p, chunk, BLOCK, temperature=0.8, rng=jax.random.PRNGKey(7 + i))
        pieces.append(tok.decode(np.asarray(chunk[0])))
    print(f"  generated continuation ({4 * BLOCK} chars in {BLOCK}-char chunks, temperature 0.8):\n" + "".join(pieces))


if not args.quick:
    train_briefly()

# =========================================================================== #
n_fail = sum(not ok for _, ok in results)
print(f"\n=== {len(results) - n_fail}/{len(results)} checks passed ===")
sys.exit(1 if n_fail else 0)
