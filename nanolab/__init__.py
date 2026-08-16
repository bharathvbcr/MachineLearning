"""
nanolab — the companion package for ``modern-small-lm-training-guide.md``.

A clean, instrumented, single-GPU (RTX 3070 Ti / 8 GB) implementation of the
modern small-LM training stack: RoPE · RMSNorm · QK-Norm · SwiGLU · tied
embeddings · zero-init · Muon/AdamW/Lion/Schedule-Free · cosine/WSD/plateau
schedules · pluggable attention/mingru/mamba2/gdn mixers.

Every module maps to a section of the guide; see ``nanolab/README.md``.
"""

__all__ = ["config", "model", "mixers", "optim", "schedules", "data", "train",
           "experiments", "utils"]

__version__ = "0.1.0"
