"""Tiny Shakespeare + tokenizers + encoder-decoder training pairs.

The text is Karpathy's char-rnn `input.txt`, downloaded once from his GitHub repo to
data/input.txt and verified (size + MD5) so we know we train on the exact same file.

Seq2seq framing (so the encoder-decoder has something to do on plain text):
    src = a chunk of `block` tokens          -> encoder
    tgt = the next chunk of `block` tokens   -> what the decoder must produce
    tgt_in  = <bos> + tgt[:-1]               -> decoder input (teacher forcing)
    tgt_out = tgt                            -> labels

Run `uv run data.py [path/to/text.txt]` to compare the tokenizers on the text.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_TEXT_PATH = Path(__file__).parent / "data" / "input.txt"
EXPECTED_BYTES = 1_115_394
EXPECTED_MD5 = "6fb458f1232090904fb40fe944165e91"

SPECIALS = ["<pad>", "<bos>", "<eos>"]
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2


def verify_tiny_shakespeare(raw: bytes):
    """Raise if `raw` is not byte-for-byte Karpathy's input.txt."""
    md5 = hashlib.md5(raw).hexdigest()
    if len(raw) != EXPECTED_BYTES or md5 != EXPECTED_MD5:
        raise ValueError(f"Not the expected Tiny Shakespeare file: {len(raw):,} bytes, md5 {md5} "
                         f"(expected {EXPECTED_BYTES:,} bytes, md5 {EXPECTED_MD5})")


def download_tiny_shakespeare(path=DEFAULT_TEXT_PATH):
    """Download input.txt once (skipped if it already exists) and verify it."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {DATA_URL}")
        with urllib.request.urlopen(DATA_URL) as r:
            raw = r.read()
        verify_tiny_shakespeare(raw)
        path.write_bytes(raw)  # write bytes so Windows keeps the original line endings
    verify_tiny_shakespeare(path.read_bytes())
    return path


def load_text(path=None):
    """Default: Tiny Shakespeare (downloaded + verified). Otherwise read the given file."""
    path = download_tiny_shakespeare() if path is None else Path(path)
    text = path.read_text(encoding="utf-8")  # text mode also turns Windows CRLF into LF
    if not text:
        raise ValueError(f"{path} is empty.")
    return text


# --------------------------------------------------------------------------- #
# Tokenizers - all share the same interface and the same special-token ids
# --------------------------------------------------------------------------- #
class CharTokenizer:
    """One token per character (Karpathy's choice). Vocab = 3 specials + 65 chars."""

    def __init__(self, text):
        self.itos = SPECIALS + sorted(set(text))
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i >= len(SPECIALS))


class ByteTokenizer:
    """One token per UTF-8 byte. Vocab = 3 specials + 256 bytes; can encode any text."""

    vocab_size = len(SPECIALS) + 256

    def encode(self, text):
        return [b + len(SPECIALS) for b in text.encode("utf-8")]

    def decode(self, ids):
        return bytes(i - len(SPECIALS) for i in ids if i >= len(SPECIALS)).decode("utf-8", errors="replace")


class BPETokenizer:
    """Byte-level BPE (same algorithm as GPT-2) trained on our own corpus with a small vocab."""

    def __init__(self, text, vocab_size=1024):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIALS,  # first -> ids 0, 1, 2
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator([text], trainer)
        self.tok = tok
        self.vocab_size = tok.get_vocab_size()

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, ids):
        return self.tok.decode([i for i in ids if i >= len(SPECIALS)])


# --------------------------------------------------------------------------- #
# Encoder-decoder pairs
# --------------------------------------------------------------------------- #
def make_seq2seq_pairs(ids, block):
    """Consecutive chunks: (src = chunk i, tgt = chunk i+1). Returns src, tgt_in, tgt_out, each (N, block)."""
    ids = np.asarray(ids, dtype=np.int32)
    n = (len(ids) // block) - 1
    chunks = ids[: (n + 1) * block].reshape(n + 1, block)
    src, tgt_out = chunks[:-1], chunks[1:]
    tgt_in = np.concatenate([np.full((n, 1), BOS_ID, dtype=np.int32), tgt_out[:, :-1]], axis=1)
    return src, tgt_in, tgt_out


def load_dataset(path=None, tokenizer="char", block=64, val_frac=0.1, bpe_vocab=1024):
    """Returns (tokenizer, train_pairs, val_pairs); each pairs = (src, tgt_in, tgt_out)."""
    text = load_text(path)
    if tokenizer == "char":
        tok = CharTokenizer(text)
    elif tokenizer == "byte":
        tok = ByteTokenizer()
    elif tokenizer == "bpe":
        tok = BPETokenizer(text, bpe_vocab)
    else:
        raise ValueError(f"unknown tokenizer {tokenizer!r}, expected 'char', 'byte' or 'bpe'")
    split = int(len(text) * (1 - val_frac))
    train = make_seq2seq_pairs(tok.encode(text[:split]), block)
    val = make_seq2seq_pairs(tok.encode(text[split:]), block)
    return tok, train, val


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # BPE pieces contain non-cp1252 chars on Windows
    text = load_text(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Dataset: {len(text):,} characters, {len(set(text))} unique\n")
    sample = text[:200]

    candidates = [("char", CharTokenizer(text)), ("byte", ByteTokenizer())]
    candidates += [(f"bpe-{v}", BPETokenizer(text, v)) for v in (512, 1024, 2048, 4096)]

    print(f"{'tokenizer':<10} {'vocab':>6} {'tokens':>10} {'chars/token':>12} {'emb params (d=128)':>20}  round-trip")
    for name, tok in candidates:
        n = len(tok.encode(text))
        ok = tok.decode(tok.encode(sample)) == sample
        print(f"{name:<10} {tok.vocab_size:>6} {n:>10,} {len(text) / n:>12.2f} {tok.vocab_size * 128:>20,}  {ok}")

    bpe = BPETokenizer(text, 1024)
    pieces = [bpe.tok.id_to_token(i) for i in bpe.encode(sample)]
    print(f"\nbpe-1024 splits the first 200 chars into {len(pieces)} tokens ('Ġ' = space, 'Ċ' = newline):\n{pieces}")
