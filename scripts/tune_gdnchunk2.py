"""Does GDN's chunk width change the MATH, or only the floating-point order?

Chunk width is the largest single-arm speed-up in the sprint: at the board shape
`mixer_chunk` 128 runs 1.98x the configured 32 (53.2K vs 26.9K tok/s). Whether
that is adoptable depends entirely on this question, and the first attempt to
answer it was VACUOUS: it compared `gdn(x)` against `gdn._sequential(x)` on a
freshly built model whose output projection is zero-initialised, so both sides
were all zeros and every chunk width "agreed" to 0.000e+00 with a nan relative
RMS. A check that cannot fail is not a check.

This one compares the kernel itself -- `gdn_chunked` on one fixed set of
projections -- across widths, against the configured width 32 and against the
O(T) recurrence, and asserts the reference is non-trivial before believing any
agreement. Both `gdn_rule` variants, since E28 turns on that flag.

    python scripts/tune_gdnchunk2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

OUT = REPO / "nanolab/out/_tune"
WIDTHS = (16, 32, 64, 128, 256)


def sequential_ref(q, k, v, alpha, beta, rule):
    """O(T) delta rule on [B,H,L,D] projections -- the chunk-independent truth."""
    B, H, L, D = k.shape
    S = torch.zeros(B, H, D, D, dtype=torch.float32, device=k.device)
    ys = []
    for t in range(L):
        kt, vt = k[:, :, t], v[:, :, t]
        at = alpha[:, :, t].unsqueeze(-1)
        bt = beta[:, :, t].unsqueeze(-1)
        pred = torch.einsum("bhpn,bhp->bhn", S, kt)
        if rule == "published":
            pred = at * pred
        delta = (vt - pred) * bt
        S = at.unsqueeze(-1) * S + torch.einsum("bhp,bhn->bhpn", kt, delta)
        ys.append(torch.einsum("bhpn,bhp->bhn", S, q[:, :, t]))
    return torch.stack(ys, dim=2)


def main() -> None:
    from nanolab.mixers import gdn_chunked
    torch.manual_seed(3)
    B, H, L, D = 2, 12, 512, 64          # the board's head count and context
    dev = "cuda"
    q = torch.randn(B, H, L, D, device=dev)
    k = torch.randn(B, H, L, D, device=dev)
    v = torch.randn(B, H, L, D, device=dev)
    alpha = torch.rand(B, H, L, device=dev).clamp(0.90, 0.999)
    beta = torch.rand(B, H, L, device=dev).clamp(0.0, 1.0)

    res = {}
    for rule in ("repo", "published"):
        ref_seq = sequential_ref(q, k, v, alpha, beta, rule)
        scale = float(ref_seq.pow(2).mean().sqrt())
        # The guard the first probe lacked: if the reference is ~0, agreement is
        # meaningless and this must fail loudly rather than print zeros.
        assert scale > 1e-3, f"reference is trivial (rms={scale:.3e}); check is vacuous"
        base = gdn_chunked(q, k, v, alpha, beta, chunk=32, rule=rule)
        res[rule] = {"ref_rms": scale, "widths": {}}
        print(f"\n=== rule={rule}  (reference RMS {scale:.4f}) ===")
        print(f"  {'chunk':>6} {'max|y-seq|':>12} {'relRMS vs seq':>15} "
              f"{'max|y-chunk32|':>16}")
        for c in WIDTHS:
            y = gdn_chunked(q, k, v, alpha, beta, chunk=c, rule=rule)
            d_seq = float((y - ref_seq).abs().max())
            r_seq = float((y - ref_seq).pow(2).mean().sqrt() / scale)
            d_32 = float((y - base).abs().max())
            res[rule]["widths"][c] = {"max_abs_vs_sequential": d_seq,
                                      "rel_rms_vs_sequential": r_seq,
                                      "max_abs_vs_chunk32": d_32}
            print(f"  {c:>6} {d_seq:>12.3e} {r_seq:>15.3e} {d_32:>16.3e}")
    (OUT / "gdnchunk2.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT/'gdnchunk2.json'}")
    print("\nReading: agreement at ~1e-5 relative means the chunk width changes only")
    print("the floating-point association order, not the operator -- so adopting a")
    print("wider chunk is a numerics change of that size, not a different model.")


if __name__ == "__main__":
    main()
