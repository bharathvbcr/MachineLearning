"""
nanolab.data — datasets & tokenization (guide §3).

"Pick the dataset for the job — not Shakespeare." Roles:
  Phase 0  — watch a small model actually learn: TinyStories (coherent English
             from a tiny model), or char-level Shakespeare / enwik8.
  Phase 1  — real pretraining: FineWeb-edu (the default for small models; ~8x
             more sample-efficient), OpenWebText, Cosmopedia.

Everything is tokenized once into a flat uint16 ``.bin`` of token ids and read
with a memmap — the "efficient (pre-tokenized, packed) data loader" the guide
asks for (§7.4). ``get_batch`` samples random contiguous windows.

Hard rule (§3): never let validation data leak into training. We tokenize the
official train/val splits separately into ``train.bin`` / ``val.bin``.

Reuse: ``--dataset fineweb_bin --fineweb_pattern '...*.bin'`` reads this repo's
existing pre-tokenized FineWeb SentencePiece shards directly (the format from
parameter-golf/train_gpt_sprint_native.py: 256xint32 header, magic 20240520,
uint16 payload).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def _gpt2_tokenizer():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


# ---------------------------------------------------------------------------
# Dataset preparation -> uint16 .bin
# ---------------------------------------------------------------------------
_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt")


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_bin(path: Path, ids: np.ndarray):
    ids.astype(np.uint16).tofile(path)


def prepare_text8(data_dir: Path, which: str = "text8"):
    """enwik8 / text8 — the classic char-level benchmark scored in bits-per-char
    (guide §3). 100M chars of Wikipedia; standard 90M/5M/5M split. ``text8`` is
    lowercased a-z+space (27 symbols); ``enwik8`` is raw bytes (256 symbols)."""
    import zipfile

    d = _ensure_dir(data_dir / which)
    if (d / "train.bin").exists() and (d / "val.bin").exists():
        chars = np.load(d / "char_vocab.npy")
        tok = CharTokenizer("".join(chars.tolist()))
        return d, tok.vocab_size, tok
    url = f"http://mattmahoney.net/dc/{which}.zip"
    zp = d / f"{which}.zip"
    try:
        if not zp.exists():
            urllib.request.urlretrieve(url, zp)
        with zipfile.ZipFile(zp) as z:
            raw = z.read(which)
        text = raw.decode("latin-1") if which == "enwik8" else raw.decode("utf-8")
    except Exception:
        text = ("the quick brown fox jumps over the lazy dog " * 50000)
    n = len(text)
    tok = CharTokenizer(text)
    # standard 90/5/5 split (we keep train + val; test == last 5M held out)
    a, b = int(0.90 * n), int(0.95 * n)
    _write_bin(d / "train.bin", np.array(tok.encode(text[:a])))
    _write_bin(d / "val.bin", np.array(tok.encode(text[a:b])))
    np.save(d / "char_vocab.npy", np.array(sorted(set(text))))
    return d, tok.vocab_size, tok


def prepare_shakespeare(data_dir: Path, tokenizer: str):
    """Char or GPT-2 BPE tiny-Shakespeare. Falls back to a synthetic corpus if
    offline so the smoke test always runs."""
    d = _ensure_dir(data_dir / ("shakespeare_char" if tokenizer == "char"
                                else "shakespeare_bpe"))
    # idempotent: reuse existing bins (avoids re-tokenizing & concurrent-write races)
    if (d / "train.bin").exists() and (d / "val.bin").exists():
        if tokenizer == "char":
            chars = np.load(d / "char_vocab.npy")
            tok = CharTokenizer("".join(chars.tolist()))
            return d, tok.vocab_size, tok
        return d, 50304, None
    raw = d / "input.txt"
    if not raw.exists():
        try:
            urllib.request.urlretrieve(_SHAKESPEARE_URL, raw)
        except Exception:
            # offline fallback — deterministic synthetic text
            txt = ("To be, or not to be, that is the question. " * 4000)
            raw.write_text(txt, encoding="utf-8")
    text = raw.read_text(encoding="utf-8")
    n = len(text)
    train_txt, val_txt = text[: int(0.9 * n)], text[int(0.9 * n):]

    if tokenizer == "char":
        tok = CharTokenizer(text)
        meta = {"vocab_size": tok.vocab_size}
        _write_bin(d / "train.bin", np.array(tok.encode(train_txt)))
        _write_bin(d / "val.bin", np.array(tok.encode(val_txt)))
        np.save(d / "char_vocab.npy", np.array(sorted(set(text))))
        return d, meta["vocab_size"], tok
    else:
        enc = _gpt2_tokenizer()
        _write_bin(d / "train.bin", np.array(enc.encode_ordinary(train_txt)))
        _write_bin(d / "val.bin", np.array(enc.encode_ordinary(val_txt)))
        return d, 50304, None


def prepare_fineweb_parquet(data_dir: Path, repo: str, subdir: str, tag: str,
                            max_train_tokens: int = 50_000_000):
    """Robust FineWeb(-edu) prep that sidesteps the pyarrow *streaming* reader
    (which segfaults on Windows): download one parquet shard with
    ``hf_hub_download`` and tokenize it non-streaming, row-group by row-group.

    ``subdir`` e.g. ``sample/10BT`` for fineweb-edu's sample-10BT config.

    IMPORTANT: the existence check runs BEFORE importing pyarrow, because torch +
    pyarrow in one Windows process segfault (clashing native DLLs). When the bins
    already exist (the common case once tokenized via ``nanolab.prep_fineweb``),
    this returns without ever importing pyarrow, so the trainer is safe."""
    d = _ensure_dir(data_dir / tag)
    if (d / "train.bin").exists() and (d / "val.bin").exists():
        return d, 50304, None

    # only import the native parquet/HF stack on the tokenization path. Prefer
    # running the dedicated, torch-free ``python -m nanolab.prep_fineweb`` first.
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if subdir in f and f.endswith(".parquet"))
    if not files:
        raise SystemExit(f"no parquet shards under {subdir} in {repo}")

    enc = _gpt2_tokenizer()
    eot = enc.eot_token
    # Stream-write to a temp .bin in small chunks so we never hold 50M Python
    # ints in memory (that ~1.4 GB list + pyarrow buffers segfaults on Windows).
    tmp = d / "all.tmp.bin"
    written = 0
    chunk: list[int] = []
    CHUNK = 1_000_000

    def flush(fh):
        nonlocal chunk, written
        if chunk:
            np.asarray(chunk, dtype=np.uint16).tofile(fh)
            written += len(chunk)
            chunk = []

    with open(tmp, "wb") as fh:
        done = False
        for shard in files:                   # usually one shard is plenty
            local = hf_hub_download(repo, shard, repo_type="dataset")
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=256, columns=["text"]):
                for text in batch.column("text").to_pylist():
                    if text:
                        chunk.extend(enc.encode_ordinary(text))
                        chunk.append(eot)
                if len(chunk) >= CHUNK:
                    flush(fh)
                if written >= max_train_tokens:
                    done = True
                    break
            if done:
                break
        flush(fh)

    ids = np.memmap(tmp, dtype=np.uint16, mode="r")
    n = min(len(ids), max_train_tokens)
    split = int(0.995 * n)
    ids[:split].tofile(d / "train.bin")
    ids[split:n].tofile(d / "val.bin")
    del ids
    tmp.unlink()
    return d, 50304, None


def prepare_hf(data_dir: Path, hf_dataset: str, hf_config: str, name_tag: str,
               max_train_tokens: int = 200_000_000):
    """Tokenize a HuggingFace text dataset with GPT-2 BPE into train/val .bin.

    TinyStories / OpenWebText work as-is; FineWeb-edu / Cosmopedia need a subset
    config (pass ``hf_config``, e.g. ``sample-10BT``) — exactly the two-line
    change the guide mentions (§3).

    The existence check runs BEFORE importing ``datasets`` (which pulls pyarrow):
    torch + pyarrow in one Windows process segfault, so a cached dataset must
    return without importing the native stack. To *create* the bins on Windows,
    run the torch-free ``python -m nanolab.prep_fineweb`` first."""
    tag = name_tag or hf_dataset.replace("/", "_")
    d = _ensure_dir(data_dir / tag)
    if (d / "train.bin").exists() and (d / "val.bin").exists():
        return d, 50304, None

    from datasets import load_dataset           # heavy/native: import only here

    enc = _gpt2_tokenizer()
    eot = enc.eot_token
    kwargs = {"name": hf_config} if hf_config else {}
    ds = load_dataset(hf_dataset, split="train", streaming=True, **kwargs)

    # stream-write to disk in chunks — never hold the whole token list in memory
    tmp = d / "all.tmp.bin"
    written, chunk = 0, []
    with open(tmp, "wb") as fh:
        for ex in ds:
            text = ex.get("text") or ex.get("content") or ""
            if text:
                chunk.extend(enc.encode_ordinary(text))
                chunk.append(eot)
            if len(chunk) >= 1_000_000:
                np.asarray(chunk, dtype=np.uint16).tofile(fh)
                written += len(chunk)
                chunk = []
            if written >= max_train_tokens:
                break
        if chunk:
            np.asarray(chunk, dtype=np.uint16).tofile(fh)
            written += len(chunk)
    ids = np.memmap(tmp, dtype=np.uint16, mode="r")
    n = min(len(ids), max_train_tokens)
    split = int(0.995 * n)
    ids[:split].tofile(d / "train.bin")
    ids[split:n].tofile(d / "val.bin")
    del ids
    tmp.unlink()
    return d, 50304, None


# ---------------------------------------------------------------------------
# Reuse: this repo's pre-tokenized FineWeb shards (int32 header + uint16 body)
# ---------------------------------------------------------------------------
def load_repo_shard(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520, f"bad magic in {path}"
    ntok = int(header[2])
    return np.fromfile(path, dtype=np.uint16, offset=256 * 4, count=ntok)


# ---------------------------------------------------------------------------
# Unified dataset resolution
# ---------------------------------------------------------------------------
def get_dataset(cfg):
    """Returns (data_dir, vocab_size, char_tokenizer_or_None)."""
    data_dir = Path(cfg.data_dir)
    if cfg.dataset == "shakespeare":
        return prepare_shakespeare(data_dir, cfg.tokenizer)
    if cfg.dataset in ("text8", "enwik8"):
        return prepare_text8(data_dir, cfg.dataset)
    if cfg.dataset == "tinystories":
        return prepare_hf(data_dir, cfg.hf_dataset or "roneneldan/TinyStories",
                          cfg.hf_config, "tinystories")
    if cfg.dataset == "hf":
        if not cfg.hf_dataset:
            raise SystemExit("--dataset hf requires --hf_dataset")
        # FineWeb / FineWeb-edu: use the non-streaming parquet path (streaming
        # pyarrow segfaults on Windows). sample-10BT -> sample/10BT subdir.
        if "fineweb" in cfg.hf_dataset.lower():
            sub = {"sample-10BT": "sample/10BT", "sample-100BT": "sample/100BT",
                   "sample-350BT": "sample/350BT"}.get(cfg.hf_config, cfg.hf_config or "data")
            tag = cfg.hf_dataset.replace("/", "_")
            return prepare_fineweb_parquet(data_dir, cfg.hf_dataset, sub, tag)
        return prepare_hf(data_dir, cfg.hf_dataset, cfg.hf_config, "")
    if cfg.dataset == "fineweb_bin":
        return _prepare_fineweb_bin(cfg)
    raise SystemExit(f"unknown dataset {cfg.dataset}")


def _prepare_fineweb_bin(cfg):
    import glob
    files = sorted(glob.glob(cfg.fineweb_pattern))
    if not files:
        raise SystemExit(f"no shards match {cfg.fineweb_pattern}")
    out = _ensure_dir(Path(cfg.data_dir) / "fineweb_bin")
    if not (out / "train.bin").exists():
        toks = np.concatenate([load_repo_shard(Path(f)) for f in files])
        split = int(0.995 * len(toks))
        _write_bin(out / "train.bin", toks[:split])
        _write_bin(out / "val.bin", toks[split:])
    return out, cfg.vocab_size, None


# ---------------------------------------------------------------------------
# Batch iterator
# ---------------------------------------------------------------------------
# If a tokenized split is below this many tokens we keep it resident on the GPU
# (stored as int32 -> ~4 bytes/token, so 150M tokens ≈ 0.6 GB) and sample windows
# on-device, eliminating the CPU->GPU copy that otherwise stalls the tensor cores
# between steps (guide §7.4: an efficient, packed loader is part of maximizing
# MFU). Larger corpora fall back to the pinned-memory async path. Kept well below
# 8 GB so the resident data never crowds out the model/optimizer/activations.
# 8 GB cards: 150M int32 tokens ≈ 0.6 GB. H100/GH200-class cards hold the
# whole FineWeb shard on device (~2 GB for 500M tokens) so the host gather
# never stalls tensor cores.
_GPU_RESIDENT_MAX_TOKENS = 150_000_000
_GPU_RESIDENT_LARGE_VRAM = 40 * (1024 ** 3)


def should_gpu_resident(n_tokens: int, device: str) -> bool:
    """Keep the token buffer on GPU when the copy is cheap relative to VRAM."""
    if not device.startswith("cuda"):
        return False
    bytes_needed = int(n_tokens) * 4  # int32 resident
    if bytes_needed <= _GPU_RESIDENT_MAX_TOKENS * 4:
        return True
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception:
        return False
    if total < _GPU_RESIDENT_LARGE_VRAM:
        return False
    return bytes_needed < 0.2 * free and bytes_needed < 8 * (1024 ** 3)


class Batcher:
    """Random-window sampler. Two paths:
      * GPU-resident  — small datasets live on-device; zero per-step H2D copy.
      * pinned async   — large datasets stay in a memmap, copied with pinned
                         memory + ``non_blocking`` so the copy overlaps compute.
    ``overfit`` restricts to a tiny slice for the overfitting demo (guide §6.3).
    """

    def __init__(self, data_dir: Path, split: str, cfg, device, overfit=0):
        path = Path(data_dir) / f"{split}.bin"
        data = np.memmap(path, dtype=np.uint16, mode="r")
        if overfit:
            data = data[: overfit]
        self.block_size = cfg.block_size
        self.batch_size = cfg.batch_size
        self.device = device
        self.gen = torch.Generator().manual_seed(cfg.seed + (0 if split == "train" else 1))

        self.gpu_resident = should_gpu_resident(len(data), device)
        self._pin = [None, None]
        self._pin_i = 0
        if self.gpu_resident:
            # one copy, then everything is on-device (int32 keeps it compact)
            self.data = torch.as_tensor(np.asarray(data, dtype=np.int32),
                                        device=device)
            self.gen = torch.Generator(device=device).manual_seed(
                cfg.seed + (0 if split == "train" else 1))
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def batch(self, block_size=None, frontier=1.0):
        """``block_size`` overrides the default (sequence-length curriculum, §3).
        ``frontier`` (0..1] limits sampling to the first fraction of the corpus —
        with difficulty-sorted data this is the easy→hard *difficulty* curriculum
        (§3): the window of reachable data grows over training."""
        bs = block_size or self.block_size
        n = max(1, int(len(self.data) * frontier) - bs - 1)
        n = max(1, n)
        if self.gpu_resident:
            ix = torch.randint(n, (self.batch_size,), generator=self.gen,
                               device=self.device)
            offsets = ix[:, None] + torch.arange(bs + 1, device=self.device)[None, :]
            seq = self.data[offsets].long()              # (B, T+1) on-GPU gather
            return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()
        ix = torch.randint(n, (self.batch_size,), generator=self.gen).tolist()
        width = bs + 1
        buf = np.empty((self.batch_size, width), dtype=np.uint16)
        for row, start in enumerate(ix):
            buf[row] = self.data[start:start + width]
        seq = torch.from_numpy(np.asarray(buf, dtype=np.int64))
        if self.device.startswith("cuda"):
            slot = self._pin_i
            pin = self._pin[slot]
            if pin is None or pin.shape != seq.shape:
                pin = torch.empty(seq.shape, dtype=torch.long).pin_memory()
                self._pin[slot] = pin
            pin.copy_(seq)
            gpu = pin.to(self.device, non_blocking=True)
            self._pin_i ^= 1
            return gpu[:, :-1], gpu[:, 1:]
        return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()

    def iterator(self):
        while True:
            yield self.batch()
