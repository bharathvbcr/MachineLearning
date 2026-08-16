"""
Standalone HuggingFace tokenizer — run as its OWN process (never imports torch).

On Windows, importing torch and pyarrow into the same interpreter can segfault
(clashing native OpenMP/MKL DLLs), and so can importing torch and `datasets`.
This script imports ONLY pyarrow / datasets + tiktoken + numpy, tokenizes text
with GPT-2 BPE, and writes ``train.bin`` / ``val.bin`` that ``nanolab.train``
then reads in a separate (torch) process.

Two modes:
  * FineWeb(-edu) parquet (default) — the streaming pyarrow reader segfaults on
    Windows, so we download one shard with ``hf_hub_download`` and read it
    non-streaming, row-group by row-group.
  * any other HF text dataset (``--hf_dataset``) — TinyStories, OpenWebText,
    Cosmopedia, … via ``datasets.load_dataset(streaming=True)``.

    python -m nanolab.prep_fineweb --config sample-10BT --max_tokens 50000000
    python -m nanolab.prep_fineweb --hf_dataset roneneldan/TinyStories --max_tokens 30000000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken

SUBDIR = {"sample-10BT": "sample/10BT", "sample-100BT": "sample/100BT",
          "sample-350BT": "sample/350BT"}


def download_with_retry(repo, path, retries=8):
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    for attempt in range(retries):
        try:
            return hf_hub_download(repo, path, repo_type="dataset")
        except Exception as e:                       # resumes from cache
            print(f"  download attempt {attempt} failed ({type(e).__name__}); retrying")
            time.sleep(5)
    raise SystemExit(f"could not download {path}")


def iter_parquet_text(repo, sub, local_parquet):
    """Yield document text from FineWeb parquet shard(s), non-streaming."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi
    if local_parquet:
        local_files = [local_parquet]
    else:
        files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                       if sub in f and f.endswith(".parquet"))
        if not files:
            raise SystemExit(f"no parquet shards under {sub} in {repo}")
        local_files = [download_with_retry(repo, files[0])]
    for local in local_files:
        print(f"  reading {local}")
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=256, columns=["text"]):
            yield from batch.column("text").to_pylist()


def iter_hf_text(hf_dataset, hf_config):
    """Yield document text from any HF text dataset via streaming."""
    from datasets import load_dataset
    kwargs = {"name": hf_config} if hf_config else {}
    ds = load_dataset(hf_dataset, split="train", streaming=True, **kwargs)
    for ex in ds:
        yield ex.get("text") or ex.get("content") or ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--config", default="sample-10BT")
    p.add_argument("--hf_dataset", default="",
                   help="any HF text dataset (TinyStories, OpenWebText...); uses datasets streaming")
    p.add_argument("--max_tokens", type=int, default=50_000_000)
    p.add_argument("--data_dir", default="nanolab/data")
    p.add_argument("--val_frac", type=float, default=0.005)
    p.add_argument("--local_parquet", default="",
                   help="use an already-downloaded parquet file; skip all HF network calls")
    p.add_argument("--tag", default="",
                   help="output subdir under data_dir (default: derived from repo/hf_dataset). "
                        "Set to match the trainer preset, e.g. 'tinystories'.")
    p.add_argument("--sort_difficulty", action="store_true",
                   help="order documents easy->hard by zlib compression ratio for the "
                        "difficulty curriculum (train with --curriculum difficulty)")
    args = p.parse_args()

    if args.hf_dataset:
        tag = args.tag or args.hf_dataset.replace("/", "_")
        text_iter = iter_hf_text(args.hf_dataset, args.config if args.config != "sample-10BT" else "")
    else:
        sub = SUBDIR.get(args.config, args.config)
        text_iter = iter_parquet_text(args.repo, sub, args.local_parquet)
        tag = args.tag or args.repo.replace("/", "_")

    out = Path(args.data_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.bin").exists() and (out / "val.bin").exists():
        print(f"already prepared at {out}")
        return

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    tmp = out / "all.tmp.bin"
    t0 = time.time()

    if args.sort_difficulty:
        # difficulty curriculum: buffer documents, rank by compression ratio
        # (zlib bytes / raw bytes — lower = more repetitive/simpler = easier),
        # then write easy→hard so the trainer's growing frontier sees easy first.
        import zlib
        docs, total = [], 0
        for text in text_iter:
            if not text:
                continue
            b = text.encode("utf-8", "ignore")
            ratio = len(zlib.compress(b, 1)) / max(1, len(b))
            docs.append((ratio, enc.encode_ordinary(text) + [eot]))
            total += len(docs[-1][1])
            if total >= args.max_tokens:
                break
        docs.sort(key=lambda r: r[0])           # easy (low ratio) first
        # hold out val as a STRATIFIED sample (every Nth doc across the difficulty
        # range) so val isn't biased to the hard tail; train stays easy→hard.
        stride = max(2, int(1 / max(args.val_frac, 1e-6)))
        train_docs = [d for i, d in enumerate(docs) if i % stride]
        val_docs = [d for i, d in enumerate(docs) if not i % stride]
        with open(out / "train.bin", "wb") as fh:
            tr = sum((len(t) for _, t in train_docs))
            for _, toks in train_docs:
                np.asarray(toks, dtype=np.uint16).tofile(fh)
        with open(out / "val.bin", "wb") as fh:
            for _, toks in val_docs:
                np.asarray(toks, dtype=np.uint16).tofile(fh)
        print(f"PREPARED {out}  train={tr}  val(stratified)={len(val_docs)} docs  "
              f"difficulty-sorted in {round(time.time()-t0)}s")
        return
    else:
        written, chunk = 0, []
        with open(tmp, "wb") as fh:
            for text in text_iter:
                if text:
                    chunk.extend(enc.encode_ordinary(text))
                    chunk.append(eot)
                if len(chunk) >= 1_000_000:
                    np.asarray(chunk, dtype=np.uint16).tofile(fh)
                    written += len(chunk)
                    chunk = []
                    print(f"    {written/1e6:.1f}M tokens  "
                          f"({written/max(1,time.time()-t0)/1e3:.0f}K tok/s)")
                    sys.stdout.flush()
                if written >= args.max_tokens:
                    break
            if chunk:
                np.asarray(chunk, dtype=np.uint16).tofile(fh)
                written += len(chunk)

    ids = np.memmap(tmp, dtype=np.uint16, mode="r")
    n = min(len(ids), args.max_tokens)
    split = int((1 - args.val_frac) * n)
    ids[:split].tofile(out / "train.bin")
    ids[split:n].tofile(out / "val.bin")
    del ids
    tmp.unlink()
    print(f"PREPARED {out}  train={split}  val={n - split}  "
          f"in {round(time.time() - t0)}s")


if __name__ == "__main__":
    main()
