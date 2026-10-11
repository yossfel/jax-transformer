"""Generate Shakespeare from a trained checkpoint.

Run:
    uv run generate.py                                  # checkpoints/shakespeare, random prompt
    uv run generate.py --prompt "ROMEO:" --chunks 8 --temperature 0.8

The model maps one `block`-token chunk to the next, so text is produced chunk by chunk:
each generated chunk becomes the encoder input for the next one.
"""

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np

from data import PAD_ID, load_text
from train import load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/shakespeare")
    p.add_argument("--prompt", default=None, help="encoder input; default: a random chunk of the validation text")
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    model, params, tok, config = load_checkpoint(args.checkpoint)
    block = config["block"]

    if args.prompt is None:
        text = load_text()
        val_ids = tok.encode(text[int(0.9 * len(text)):])
        start = np.random.default_rng(args.seed).integers(0, len(val_ids) - block)
        ids = val_ids[start:start + block]
    else:
        ids = tok.encode(args.prompt)[-block:]
    src = jnp.array([ids + [PAD_ID] * (block - len(ids))], dtype=jnp.int32)  # left-aligned, padded

    generate = jax.jit(model.generate, static_argnums=(2,), static_argnames=("temperature",))
    print(tok.decode(ids), end="", flush=True)
    rng = jax.random.PRNGKey(args.seed)
    for _ in range(args.chunks):
        rng, r = jax.random.split(rng)
        src = generate(params, src, block, temperature=args.temperature, rng=r)
        print(tok.decode(np.asarray(src[0]).tolist()), end="", flush=True)
    print()


if __name__ == "__main__":
    main()
