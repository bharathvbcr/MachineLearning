"""Custom CUDA flash attention: build-on-demand extension + autograd wrapper.

The kernel is ``nanolab/csrc/flash_attn_cuda.cu`` -- a port of this repo's Metal
FA-2 kernels, same layout, same taped LSE, same backward algebra. See that file's
header for what it does and does not do.

Two implementations live there, forward and backward each: an FMA one (every
dtype, every supported head dim) and a tensor-core one on
``mma.sync.aligned.m16n8k16`` (f16/bf16, sm_80+; the backward additionally wants
head_dim 32 or 64). The tensor-core path is taken by default when it is
available, and "available" means its fragment layout was checked against the
device at load -- see ``_probe_mma``.

Nothing here is on by default. The extension is compiled the first time it is
actually asked for, and every gate is explicit:

    from nanolab import flash_cuda
    if flash_cuda.supports(q, k, v):          # dtype / head_dim / layout
        y = flash_cuda.flash_attn(q, k, v, scale)

``NANOLAB_FLASH_CUDA=0`` in the environment refuses to build it at all, which is
the escape hatch when nvcc is present but you do not want a compile in the middle
of a run.

Self-check on a CUDA box (correctness first, then speed vs SDPA):

    python -m nanolab.flash_cuda
"""

from __future__ import annotations

import logging
import math
import os
import pathlib

import torch

log = logging.getLogger("nanolab.flash_cuda")

SUPPORTED_HEAD_DIMS = (32, 64, 128)
SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_SOURCE = pathlib.Path(__file__).parent / "csrc" / "flash_attn_cuda.cu"

MMA_DTYPES = (torch.float16, torch.bfloat16)   # m16n8k16 has no f32 form
# The tensor-core backward stops at head_dim 64: at 128 a dK/dV warp's fragments
# and accumulators run past the register file and spill. See the note above
# NANOLAB_ON_HEAD_DIM_MMA_BWD in the .cu.
MMA_BWD_HEAD_DIMS = (32, 64)

_ext = None            # the loaded module, once built
_load_error: str | None = None
_load_attempted = False
_mma_ok = False        # set by the layout probe, once, at load
_mma_error: str | None = None


def _probe_mma(ext) -> tuple[bool, str]:
    """Check the kernel's mma fragment layout against the hardware.

    ``flash_attn_cuda.cu`` hard-codes the PTX ISA's register-to-matrix mapping
    for ``mma.sync.aligned.m16n8k16.row.col``. Everything else in that file is
    verified on CPU by ``csrc/flash_attn_sim.py``, but a layout table cannot be:
    the mirror and the kernel would just agree with each other. So multiply
    known matrices through the kernel's own fragment helpers and compare with a
    plain matmul. If the table is wrong the answer is wrong by order 1, not by
    an ulp, so this is not a close call -- and getting it wrong silently would
    corrupt training. Fails closed: the tensor-core path stays off.
    """
    try:
        major, minor = torch.cuda.get_device_capability()
        if major < 8:
            return False, (f"mma.m16n8k16 needs sm_80+, this device is "
                           f"sm_{major}{minor}")
        for dtype in MMA_DTYPES:
            torch.manual_seed(0)
            a = torch.randn(16, 16, device="cuda", dtype=dtype)
            b = torch.randn(16, 8, device="cuda", dtype=dtype)
            got = ext.mma_probe(a, b)
            want = a.float() @ b.float()   # exact from the same rounded inputs
            err = (got - want).abs().max().item()
            if not err < 1e-2:
                return False, (f"fragment layout probe failed for {dtype}: "
                               f"max |mma - matmul| = {err:.3e}")
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _load():
    """Build (or fetch from torch's extension cache) the CUDA extension.

    Returns the module, or None with ``_load_error`` set. Never raises: callers
    gate on ``is_available()`` and fall back, and a build failure on a machine
    without nvcc is an expected outcome, not a crash.
    """
    global _ext, _load_error, _load_attempted, _mma_ok, _mma_error
    if _load_attempted:
        return _ext
    _load_attempted = True

    if os.environ.get("NANOLAB_FLASH_CUDA", "1") == "0":
        _load_error = "disabled by NANOLAB_FLASH_CUDA=0"
        return None
    if not torch.cuda.is_available():
        _load_error = "no CUDA device"
        return None
    if not _SOURCE.exists():
        _load_error = f"kernel source missing: {_SOURCE}"
        return None

    try:
        from torch.utils.cpp_extension import load
        log.info("building nanolab_flash_attn from %s (first use; ~1-2 min)", _SOURCE)
        _ext = load(
            name="nanolab_flash_attn",
            sources=[str(_SOURCE)],
            # No --use_fast_math: it would change the softmax numerics, and
            # suite 19's lesson was that the accumulation precision is the whole
            # ballgame. -lineinfo costs nothing and makes ncu profiles readable.
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )
    except Exception as e:                      # nvcc missing, arch mismatch, ...
        _ext = None
        _load_error = f"{type(e).__name__}: {e}"
        log.warning("nanolab_flash_attn did not build (%s); falling back to SDPA",
                    _load_error)
        return None

    _mma_ok, _mma_error = _probe_mma(_ext)
    if not _mma_ok:
        log.info("tensor-core path off (%s); using the FMA kernel", _mma_error)
    return _ext


def is_available() -> bool:
    """True once the extension is built and loadable on this machine."""
    return _load() is not None


def unavailable_reason() -> str:
    """Why ``is_available()`` is False -- empty string when it is True."""
    _load()
    return "" if _ext is not None else (_load_error or "unknown")


def mma_available() -> bool:
    """True when the tensor-core forward is built AND its layout probe passed."""
    return _load() is not None and _mma_ok


def mma_unavailable_reason() -> str:
    """Why ``mma_available()`` is False -- empty string when it is True."""
    if _load() is None:
        return unavailable_reason()
    return "" if _mma_ok else (_mma_error or "unknown")


def effective_dtype(q: torch.Tensor, k: torch.Tensor,
                    v: torch.Tensor) -> torch.dtype | None:
    """The dtype the kernel would actually run at, or None if there isn't one.

    Under an enabled CUDA autocast this is the autocast dtype, mirroring what
    autocast does to SDPA's inputs. That mirroring is not a nicety: nanolab's
    RoPE multiplies bf16 q/k by fp32 cos/sin tables, so q and k arrive at
    attention as **fp32 while v is still bf16**. SDPA survives that only because
    it is an autocast-listed op and its inputs get cast for it; a custom
    autograd.Function gets no such treatment. Without this the kernel would
    decline every step of the default bf16 recipe -- the flag would be dead.
    """
    if q.is_cuda and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    if q.dtype == k.dtype == v.dtype:
        return q.dtype
    return None


def supports(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Whether this (q, k, v) triple can go through the kernel.

    Shape / dtype / device only -- it does not build the extension, so it is safe
    to call on the hot path before deciding to fall back. Layout is not part of
    the question: ``flash_attn`` owns the cast-and-contiguous normalisation, so
    a non-contiguous input is something it fixes, not something it refuses.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if effective_dtype(q, k, v) not in SUPPORTED_DTYPES:
        return False
    if q.shape[-1] not in SUPPORTED_HEAD_DIMS:
        return False
    if k.shape[:2] != q.shape[:2] or v.shape[:2] != q.shape[:2]:
        return False
    if k.shape[2] != v.shape[2] or k.shape[3] != q.shape[3] or v.shape[3] != q.shape[3]:
        return False
    if k.shape[2] == 0 or q.shape[2] % k.shape[2] != 0:
        return False
    return True


class _FlashAttnFn(torch.autograd.Function):
    """Causal GQA attention on [B,T,H,D] with the taped LSE carried to backward."""

    @staticmethod
    def forward(ctx, q, k, v, scale, use_mma):
        ext = _load()
        if ext is None:
            raise RuntimeError(
                f"nanolab flash_cuda is unavailable ({unavailable_reason()})")
        o, lse = ext.fwd(q, k, v, float(scale), bool(use_mma))
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.scale = float(scale)
        # The backward follows the forward's choice, narrowed by head_dim. The
        # two are independent as far as correctness goes -- both passes
        # recompute scores from the taped LSE, so either backward works with
        # either forward's tape -- but there is no reason to want them split.
        ctx.use_mma = bool(use_mma) and q.shape[-1] in MMA_BWD_HEAD_DIMS
        return o

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = _load().bwd(dout.contiguous(), q, k, v, o, lse, ctx.scale,
                                 ctx.use_mma)
        return dq, dk, dv, None, None


def flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               scale: float | None = None,
               use_mma: bool | None = None) -> torch.Tensor:
    """Causal GQA flash attention. q [B,T,H,D], k/v [B,T,Hkv,D] -> [B,T,H,D].

    No mask argument: this is the causal path only. GQA is native -- k/v keep
    their own head count, no ``repeat_interleave``. Raises if the inputs are not
    supported rather than silently falling back; call ``supports()`` first.

    ``use_mma`` picks the kernels: None (default) takes the tensor-core ones
    whenever they are available and the dtype is 16-bit, True demands them and
    raises if they cannot be had, False forces the FMA ones. It covers both
    passes -- the backward additionally needs head_dim 32 or 64 and quietly
    keeps the FMA backward at 128, since that is a register-file limit rather
    than a choice. Every combination produces the same tape and the same
    gradients up to accumulation order, so this is a speed/precision question,
    not a semantic one.
    """
    if not supports(q, k, v):
        raise ValueError(
            "flash_cuda.flash_attn: unsupported inputs "
            f"(q={tuple(q.shape)}/{q.dtype}, k={tuple(k.shape)}/{k.dtype}, "
            f"v={tuple(v.shape)}/{v.dtype}); head_dim must be one of "
            f"{SUPPORTED_HEAD_DIMS} and the effective dtype one of "
            f"{SUPPORTED_DTYPES}")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    # One owner for normalisation. The casts are autograd ops, so gradients flow
    # back to the original (possibly fp32) tensors exactly as they do through
    # autocast's own casts on SDPA. Both calls are no-ops when nothing changes.
    dtype = effective_dtype(q, k, v)
    q = q.to(dtype).contiguous()
    k = k.to(dtype).contiguous()
    v = v.to(dtype).contiguous()

    if use_mma is None:
        use_mma = dtype in MMA_DTYPES and mma_available()
    elif use_mma:
        if dtype not in MMA_DTYPES:
            raise ValueError(
                f"flash_cuda: use_mma=True but the effective dtype is {dtype}; "
                f"mma.m16n8k16 exists only for {MMA_DTYPES}")
        if not mma_available():
            raise RuntimeError(
                f"flash_cuda: use_mma=True but the tensor-core path is "
                f"unavailable ({mma_unavailable_reason()})")
    return _FlashAttnFn.apply(q, k, v, scale, use_mma)


def sdpa_reference(q, k, v, scale):
    """What nanolab does today: widen KV heads, transpose, call SDPA.

    Here so the benchmark compares against the path this would actually replace,
    `repeat_interleave` included, rather than an idealised one.
    """
    import torch.nn.functional as F
    group = q.shape[2] // k.shape[2]
    qs = q.transpose(1, 2)
    ks = k.repeat_interleave(group, dim=2).transpose(1, 2)
    vs = v.repeat_interleave(group, dim=2).transpose(1, 2)
    y = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, scale=scale)
    return y.transpose(1, 2).contiguous()


def _bench_once(fn, iters: int, warmup: int = 5) -> float:
    """Median milliseconds per call, GPU-synchronised."""
    import statistics
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main() -> int:
    """Correctness against SDPA, then a speed comparison. CUDA only."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--kv_heads", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument("--dtype", default="bf16", choices=("fp32", "fp16", "bf16"))
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device -- nothing to run")
        return 1
    if not is_available():
        print(f"extension unavailable: {unavailable_reason()}")
        return 1

    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[args.dtype]
    B, T, H, Hkv, D = args.batch, args.seq, args.heads, args.kv_heads, args.head_dim
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, Hkv, D, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, Hkv, D, device="cuda", dtype=dtype, requires_grad=True)
    scale = 1.0 / math.sqrt(D)

    print(f"shape B{B} T{T} H{H}/{Hkv} D{D} {args.dtype}")
    if mma_available():
        print("  tensor-core path: on (layout probe passed)")
    else:
        print(f"  tensor-core path: OFF -- {mma_unavailable_reason()}")

    mma_usable = mma_available() and dtype in MMA_DTYPES
    arms = [("SDPA", lambda: sdpa_reference(q, k, v, scale)),
            ("fma", lambda: flash_attn(q, k, v, scale, use_mma=False))]
    if mma_usable:
        arms.append(("mma", lambda: flash_attn(q, k, v, scale, use_mma=True)))

    # ---- correctness, before any timing is worth reading ----
    y_ref = sdpa_reference(q, k, v, scale)
    do = torch.randn_like(y_ref)
    gr = torch.autograd.grad(y_ref, (q, k, v), do, retain_graph=False)
    for name, fn in arms[1:]:
        y = fn()
        g = torch.autograd.grad(y, (q, k, v), do, retain_graph=False)
        fe = (y.float() - y_ref.float()).abs().max().item()
        be = max((a.float() - b.float()).abs().max().item() for a, b in zip(g, gr))
        print(f"  {name:4s} vs SDPA:  max |dO| {fe:.3e}   max |dgrad| {be:.3e}")

    # ---- speed ----
    def _fwd_bwd(fn):
        torch.autograd.grad(fn(), (q, k, v), do, retain_graph=False)

    print(f"  {'arm':4s} {'fwd ms':>10s} {'fwd+bwd ms':>12s} {'vs SDPA fwd':>13s}")
    base = None
    for name, fn in arms:
        t_f = _bench_once(fn, args.iters)
        t_fb = _bench_once(lambda fn=fn: _fwd_bwd(fn), args.iters)
        base = base if base is not None else t_f
        print(f"  {name:4s} {t_f:10.3f} {t_fb:12.3f} {base / t_f:12.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
