"""Train the Transformer on Tiny Shakespeare (CPU or GPU - JAX picks the device automatically).

Training setup from the paper (section 5):
    loss  = cross-entropy against smoothed targets (eps_ls = 0.1), <pad> positions ignored
    Adam(beta1=0.9, beta2=0.98, eps=1e-9)
    lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)

Run:
    uv run train.py                          # default GPU-sized config
    uv run train.py --steps 200 --d-model 64 # quick CPU smoke test
    uv run train.py --help                   # all options
Checkpoints go to checkpoints/<run-name>/ (params + config + tokenizer + loss history).
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from data import load_dataset
from transformer import Transformer, count_params


def cross_entropy(logits, targets, pad_id=0, label_smoothing=0.0):
    """Mean token-level cross-entropy over non-pad targets. Returns (loss, accuracy)."""
    vocab = logits.shape[-1]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(targets, vocab)
    # Label smoothing: put (1 - eps) on the right token, spread eps over the whole vocab.
    soft = one_hot * (1.0 - label_smoothing) + label_smoothing / vocab
    nll = -(soft * log_probs).sum(-1)                       # (B, L)
    mask = (targets != pad_id).astype(logits.dtype)
    n = jnp.maximum(mask.sum(), 1.0)
    loss = (nll * mask).sum() / n
    acc = ((logits.argmax(-1) == targets) * mask).sum() / n
    return loss, acc


def noam_schedule(d_model, warmup_steps=4000, scale=1.0):
    """Linear warmup, then decay with 1/sqrt(step). `scale` multiplies the whole curve."""
    def schedule(count):
        step = count + 1  # optax counts from 0
        return scale * d_model**-0.5 * jnp.minimum(step**-0.5, step * warmup_steps**-1.5)
    return schedule


def make_optimizer(learning_rate, clip_norm=1.0):
    """Paper's Adam settings. Gradient clipping is not in the paper but makes small runs safer."""
    return optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adam(learning_rate, b1=0.9, b2=0.98, eps=1e-9),
    )


def make_train_step(model, optimizer, label_smoothing=0.1):
    def loss_fn(params, batch, rng):
        src, tgt_in, tgt_out = batch
        logits, _ = model(params, src, tgt_in, rng=rng)
        return cross_entropy(logits, tgt_out, model.pad_id, label_smoothing)

    @jax.jit
    def train_step(params, opt_state, batch, rng):
        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, rng)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, acc, optax.global_norm(grads)

    return train_step


def make_eval_step(model):
    """Dropout off, no label smoothing: the 'true' cross-entropy."""
    @jax.jit
    def eval_step(params, batch):
        src, tgt_in, tgt_out = batch
        logits, _ = model(params, src, tgt_in)
        return cross_entropy(logits, tgt_out, model.pad_id)
    return eval_step


def random_batches(pairs, batch_size, seed=0):
    """Endless iterator of random (src, tgt_in, tgt_out) batches."""
    src, tgt_in, tgt_out = pairs
    rng = np.random.default_rng(seed)
    while True:
        idx = rng.integers(0, len(src), batch_size)
        yield jnp.asarray(src[idx]), jnp.asarray(tgt_in[idx]), jnp.asarray(tgt_out[idx])


def evaluate(eval_step, params, pairs, batch_size=64, max_batches=20):
    src, tgt_in, tgt_out = pairs
    losses, accs = [], []
    for i in range(min(max_batches, len(src) // batch_size)):
        s = slice(i * batch_size, (i + 1) * batch_size)
        loss, acc = eval_step(params, (jnp.asarray(src[s]), jnp.asarray(tgt_in[s]), jnp.asarray(tgt_out[s])))
        losses.append(float(loss))
        accs.append(float(acc))
    return float(np.mean(losses)), float(np.mean(accs))


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path, params, config, tokenizer, step, history):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "checkpoint.pkl", "wb") as f:
        pickle.dump({"params": jax.device_get(params), "config": config, "tokenizer": tokenizer, "step": step}, f)
    (path / "history.json").write_text(json.dumps(history))


def load_checkpoint(path):
    """Returns (model, params, tokenizer, config)."""
    with open(Path(path) / "checkpoint.pkl", "rb") as f:
        ckpt = pickle.load(f)
    model = build_model(ckpt["config"], ckpt["tokenizer"].vocab_size)
    params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
    return model, params, ckpt["tokenizer"], ckpt["config"]


def build_model(config, vocab_size):
    """Shared embeddings: source and target use the same tokenizer."""
    return Transformer(vocab_size, vocab_size, config["d_model"], config["num_heads"], config["d_ff"],
                       config["num_layers"], max_len=config["block"], dropout_rate=config["dropout"],
                       share_embeddings=True)


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="shakespeare")
    p.add_argument("--tokenizer", choices=["char", "bpe", "byte"], default="char")
    p.add_argument("--bpe-vocab", type=int, default=1024)
    p.add_argument("--block", type=int, default=128, help="tokens per source / target chunk")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--lr-scale", type=float, default=1.0, help="multiplies the paper's schedule")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    config = dict(vars(args))
    out_dir = Path("checkpoints") / args.run_name
    print(f"devices: {jax.devices()}")

    tok, train_pairs, val_pairs = load_dataset(tokenizer=args.tokenizer, block=args.block, bpe_vocab=args.bpe_vocab)
    print(f"tokenizer: {args.tokenizer} (vocab {tok.vocab_size})  train pairs {train_pairs[0].shape}  "
          f"val pairs {val_pairs[0].shape}")

    model = build_model(config, tok.vocab_size)
    params = model.init(jax.random.PRNGKey(args.seed))
    print(f"parameters: {count_params(params):,}")

    schedule = noam_schedule(args.d_model, args.warmup, args.lr_scale)
    optimizer = make_optimizer(schedule)
    opt_state = optimizer.init(params)
    train_step = make_train_step(model, optimizer, args.label_smoothing)
    eval_step = make_eval_step(model)
    batches = random_batches(train_pairs, args.batch_size, args.seed)
    rng = jax.random.PRNGKey(args.seed + 1)

    history = {"step": [], "train_loss": [], "val_step": [], "val_loss": [], "val_acc": []}
    best_val = float("inf")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        rng, step_rng = jax.random.split(rng)
        params, opt_state, loss, acc, gnorm = train_step(params, opt_state, next(batches), step_rng)
        history["step"].append(step)
        history["train_loss"].append(float(loss))

        if step % args.eval_every == 0 or step == args.steps:
            val_loss, val_acc = evaluate(eval_step, params, val_pairs)
            history["val_step"].append(step)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            tokens_per_s = step * args.batch_size * args.block / (time.time() - t0)
            print(f"step {step:6d}  train {float(loss):.3f}  val {val_loss:.3f}  val acc {val_acc:.3f}  "
                  f"lr {float(schedule(step - 1)):.2e}  |grad| {float(gnorm):.2f}  {tokens_per_s:,.0f} tok/s")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(out_dir, params, config, tok, step, history)
                print(f"  saved best checkpoint -> {out_dir}")
    print(f"done in {time.time() - t0:.0f}s, best val loss {best_val:.3f}")


if __name__ == "__main__":
    main()
