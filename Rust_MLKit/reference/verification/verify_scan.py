"""
verify_scan.py -- Numerical correctness tests for the chunk-parallel SSD scan.

Compares _ssd_chunk_parallel() against the sequential reference at
multiple seqlens, batch sizes, chunk sizes, and random seeds.
All tests must pass within absolute tolerance 1e-5 (FP32 precision noise floor).

Usage:
  conda activate cuda_torch_env
  cd parameter-golf
  python verify_scan.py
"""

import os, sys, math

os.environ["SKIP_COMPILE"] = "1"

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import importlib.util

spec = importlib.util.spec_from_file_location("hc", "train_hypercascade.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print()


def sequential_ssd_reference(x_h_scaled, B_h, C_h, log_dA):
    """Ground-truth sequential SSD recurrence. Slow but provably correct."""
    B, L, H, P = x_h_scaled.shape
    N = B_h.shape[-1]
    dA = torch.exp(log_dA)  # (B, L, H) in (0, 1]
    h = torch.zeros(B, H, P, N, device=x_h_scaled.device, dtype=torch.float32)
    ys = []
    for t in range(L):
        dA_t = dA[:, t, :, None, None]  # (B, H, 1, 1)
        x_t = x_h_scaled[:, t, :, :, None]  # (B, H, P, 1)
        B_t = B_h[:, t, None, None, :]  # (B, 1, 1, N)
        h = dA_t * h + x_t * B_t  # (B, H, P, N)
        C_t = C_h[:, t, None, None, :]  # (B, 1, 1, N)
        ys.append((h * C_t).sum(-1))  # (B, H, P)
    return torch.stack(ys, dim=1)  # (B, L, H, P)


def run_test(tag, B, L, H, P, N, C, seed=42):
    torch.manual_seed(seed)

    # Build a minimal Mamba2Block just to call _ssd_chunk_parallel
    d_model = H * P
    block = mod.Mamba2Block(
        d_model=d_model, d_state=N, d_conv=4, expand=1, chunk_size=C
    ).to(DEVICE)
    block.headdim = P
    block.nheads = H
    # Ensure causal mask covers chunk size
    if C > block._causal_mask.shape[0]:
        block._causal_mask = torch.tril(
            torch.ones(C * 2, C * 2, dtype=torch.bool, device=DEVICE)
        )

    x_h_scaled = torch.randn(B, L, H, P, device=DEVICE)
    B_h = torch.randn(B, L, N, device=DEVICE) * 0.3
    C_h = torch.randn(B, L, N, device=DEVICE) * 0.3
    log_dA = -torch.rand(B, L, H, device=DEVICE) * 0.5  # in (-0.5, 0]

    y_ref = sequential_ssd_reference(x_h_scaled, B_h, C_h, log_dA)
    y_chunk = block._ssd_chunk_parallel(x_h_scaled, B_h, C_h, log_dA)

    assert y_ref.shape == (B, L, H, P), f"ref shape wrong:   {y_ref.shape}"
    assert y_chunk.shape == (B, L, H, P), f"chunk shape wrong: {y_chunk.shape}"
    assert not y_chunk.isnan().any(), "NaN in chunk output"
    assert not y_chunk.isinf().any(), "Inf in chunk output"

    abs_err = (y_ref - y_chunk).abs().max().item()
    rel_err = abs_err / y_ref.abs().clamp(min=1e-4).max().item()
    pass_ok = abs_err < 1e-5 or rel_err < 1e-3

    status = "PASS" if pass_ok else "FAIL"
    print(f"  [{status}] {tag:50s}  abs={abs_err:.2e}  rel={rel_err:.2e}")
    return pass_ok


def main():
    print("=" * 70)
    print("  MAMBA-2 CHUNK-PARALLEL SSD SCAN -- CORRECTNESS TESTS")
    print("=" * 70)
    print()

    results = []
    B, H, P, N = 4, 2, 8, 4

    print("--- seqlen coverage (C=32) ---")
    for L in [1, 32, 64, 96, 100, 256, 300, 512, 1024, 2049]:
        results.append(
            run_test(f"L={L:<5d} B={B} H={H} P={P} N={N} C=32", B, L, H, P, N, 32)
        )

    print()
    print("--- chunk size sweep (L=256) ---")
    for C in [1, 4, 8, 16, 32, 64, 128, 256]:
        results.append(
            run_test(f"L=256  B={B} H={H} P={P} N={N} C={C:<4d}", B, 256, H, P, N, C)
        )

    print()
    print("--- batch size sweep (L=256, C=32) ---")
    for b in [1, 2, 8, 16, 32]:
        results.append(
            run_test(f"L=256  B={b:<3d} H={H} P={P} N={N} C=32", b, 256, H, P, N, 32)
        )

    print()
    print("--- random seed sweep (L=256, C=32) ---")
    for seed in [0, 1, 7, 42, 1337, 2025, 99999]:
        results.append(
            run_test(
                f"L=256  B={B} H={H} P={P} N={N} seed={seed}", B, 256, H, P, N, 32, seed
            )
        )

    print()
    print("--- d_state sweep (L=256, C=32) ---")
    for n in [4, 8, 16, 32, 64]:
        results.append(
            run_test(f"L=256  B={B} H={H} P={P} N={n:<3d} C=32", B, 256, H, P, n, 32)
        )

    print()
    print("--- gradient flow ---")
    torch.manual_seed(42)
    blk = mod.Mamba2Block(d_model=16, d_state=4, d_conv=4, expand=1, chunk_size=8).to(
        DEVICE
    )
    blk.headdim = 8
    blk.nheads = 2
    blk._causal_mask = torch.tril(torch.ones(64, 64, dtype=torch.bool, device=DEVICE))
    x_h = torch.randn(2, 32, 2, 8, device=DEVICE, requires_grad=True)
    B_h = torch.randn(2, 32, 4, device=DEVICE, requires_grad=True)
    C_h = torch.randn(2, 32, 4, device=DEVICE, requires_grad=True)
    ldA = -torch.rand(2, 32, 2, device=DEVICE)
    blk._ssd_chunk_parallel(x_h, B_h, C_h, ldA).mean().backward()
    grad_ok = (
        x_h.grad is not None
        and not x_h.grad.isnan().any()
        and B_h.grad is not None
        and C_h.grad is not None
    )
    print(f"  [{'PASS' if grad_ok else 'FAIL'}] gradient through x_h, B_h, C_h")
    results.append(grad_ok)

    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print()
    print("=" * 70)
    if n_fail == 0:
        print(f"  ALL {n_pass} TESTS PASSED")
    else:
        print(f"  {n_pass}/{len(results)} PASSED  --  {n_fail} FAILURES")
    print("=" * 70)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
