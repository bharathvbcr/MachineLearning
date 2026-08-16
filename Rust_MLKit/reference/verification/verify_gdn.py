"""
verify_gdn.py -- Numerical correctness tests for GatedDeltaNetBlock.

Tests the delta rule recurrence against a brute-force reference,
gradient flow, and forward/backward at multiple configurations.

Usage:
  conda activate cuda_torch_env
  cd parameter-golf
  python verify_gdn.py
"""

import os, sys

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


def delta_rule_reference(q, k, v, alpha, beta):
    """
    Ground-truth sequential delta rule recurrence.
    S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1} @ k_t) @ k_t^T
    y_t = S_t @ q_t
    All inputs in FP32.  k must already be L2-normalised.
    """
    B, L, H, P = q.shape
    S = torch.zeros(B, H, P, P, device=q.device, dtype=torch.float32)
    ys = []
    for t in range(L):
        k_t = k[:, t, :, :, None]  # (B, H, P, 1)
        v_t = v[:, t, :, :, None]  # (B, H, P, 1)
        q_t = q[:, t, :, :, None]  # (B, H, P, 1)
        a_t = alpha[:, t, :, None, None]
        b_t = beta[:, t, :, None, None]
        pred = torch.matmul(S, k_t)  # (B, H, P, 1)
        error = v_t - pred
        S = a_t * S + b_t * torch.matmul(error, k_t.transpose(-1, -2))
        ys.append(torch.matmul(S, q_t).squeeze(-1))  # (B, H, P)
    return torch.stack(ys, dim=1)  # (B, L, H, P)


def run_test(tag, B, L, H, P, seed=42):
    torch.manual_seed(seed)
    d_model = H * P

    block = mod.GatedDeltaNetBlock(d_model=d_model, n_heads=H, head_dim=P).to(DEVICE)
    for m in block.modules():
        if isinstance(m, mod.CastedLinear):
            m.float()
    mod.restore_low_dim_params_to_fp32(block)
    block.eval()

    x = torch.randn(B, L, d_model, device=DEVICE)

    # --- Reference: extract Q, K, V, gates from the same in_proj ---
    with torch.no_grad():
        proj = block.in_proj(x.float())
        q_, k_, v_, gates_ = proj.split([d_model, d_model, d_model, 2 * H], dim=-1)
        q_ = q_.view(B, L, H, P).float()
        k_ = k_.view(B, L, H, P).float()
        v_ = v_.view(B, L, H, P).float()
        alpha_g, beta_g = gates_.split([H, H], dim=-1)
        alpha_ = torch.sigmoid(
            alpha_g.float() + block.decay_bias.float()[None, None, :]
        )
        beta_ = torch.sigmoid(beta_g.float() + block.update_bias.float()[None, None, :])
        k_norm = F.normalize(k_, p=2, dim=-1)  # must match GDN forward
        y_ref = delta_rule_reference(q_, k_norm, v_, alpha_, beta_)  # (B,L,H,P)

        # Reproduce the GDN inner path: chunked delta rule then out_norm + out_proj
        # This tests that the chunked implementation matches the naive loop.
        q_c = q_.transpose(1, 2)  # (B, H, L, P)
        k_c = k_norm.transpose(1, 2)
        v_c = v_.transpose(1, 2)
        alpha_c = alpha_.transpose(1, 2)  # (B, H, L)
        beta_c = beta_.transpose(1, 2)
        y_chunked = mod.chunk_gated_delta_rule(
            q_c, k_c, v_c, alpha_c, beta_c, chunk_size=block.chunk_size
        )
        y_chunked = y_chunked.transpose(1, 2)  # (B, L, H, P)

        # Compare chunked vs naive reference (the core correctness check)
        inner_diff = (y_ref - y_chunked).abs().max().item()
        ok_inner = inner_diff < 1e-4  # FP32 tolerance

    # --- GDN full forward (includes out_norm + out_proj) ---
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        y_out = block(x)

    # Shape and sanity checks
    ok_shape = y_out.shape == (B, L, d_model)
    ok_nan = not y_out.isnan().any()
    ok_inf = not y_out.isinf().any()
    ok_mag = y_ref.abs().max().item() < 1e6

    ok = ok_shape and ok_nan and ok_inf and ok_mag and ok_inner
    status = "PASS" if ok else "FAIL"
    details = f"shape={ok_shape} nan={ok_nan} inf={ok_inf} mag={ok_mag} inner_diff={inner_diff:.2e}"
    print(f"  [{status}] {tag:50s}  {details}")
    return ok


def run_gradient_test():
    torch.manual_seed(0)
    B, L, H, P = 2, 16, 2, 8
    d_model = H * P
    block = mod.GatedDeltaNetBlock(d_model=d_model, n_heads=H, head_dim=P).to(DEVICE)
    for m in block.modules():
        if isinstance(m, mod.CastedLinear):
            m.float()
    mod.restore_low_dim_params_to_fp32(block)

    x = torch.randn(B, L, d_model, device=DEVICE, requires_grad=True)
    y = block(x)
    y.mean().backward()

    ok_x = x.grad is not None and not x.grad.isnan().any()
    ok_inproj = block.in_proj.weight.grad is not None
    ok_decay = block.decay_bias.grad is not None
    ok_update = block.update_bias.grad is not None
    ok = ok_x and ok_inproj and ok_decay and ok_update
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {'gradient flow (x, in_proj, decay_bias, update_bias)':50s}  "
        f"x={ok_x} in_proj={ok_inproj} decay={ok_decay} update={ok_update}"
    )
    return ok


def run_full_model_test(layer_types_str, tag, mtp_lambda=0.0):
    torch.manual_seed(42)
    ltypes = [t.strip() for t in layer_types_str.split(",")]
    n = len(ltypes)
    dim = 128
    m = (
        mod.HyperCascadeGPT(
            vocab_size=1024,
            num_layers=n,
            model_dim=dim,
            num_heads=4,
            num_kv_heads=4,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.005,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
            layer_types=ltypes,
            gdn_head_dim=32,
            ssm_d_state=16,
            ssm_expand=1,
            ssm_d_conv=4,
            ssm_chunk_size=32,
            mtp_lambda=mtp_lambda,
            vocab_size_mtp=1024,
        )
        .to(DEVICE)
        .bfloat16()
    )
    for mod2 in m.modules():
        if isinstance(mod2, mod.CastedLinear):
            mod2.float()
    mod.restore_low_dim_params_to_fp32(m)

    x = torch.randint(0, 1024, (4, 32), device=DEVICE)
    y = torch.randint(0, 1024, (4, 32), device=DEVICE)

    m.train()
    with torch.autocast("cuda", torch.bfloat16):
        loss = m(x, y)

    ok_loss = not loss.isnan() and not loss.isinf() and loss.item() > 0
    loss.backward()

    # Check logits shape
    m.eval()
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        logits = m.forward_logits(x)
    ok_logits = logits.shape == (4, 32, 1024) and not logits.isnan().any()

    n_params = sum(p.numel() for p in m.parameters())
    ok = ok_loss and ok_logits
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {tag:50s}  loss={loss.item():.4f}  params={n_params:,}  logits_ok={ok_logits}"
    )
    return ok


def main():
    print("=" * 70)
    print("  GATED DELTANET (GDN) -- CORRECTNESS TESTS")
    print("=" * 70)
    print()

    results = []

    print("--- GDN block: shape, no NaN, no Inf ---")
    for B, L, H, P in [(2, 16, 2, 8), (4, 32, 4, 32), (8, 64, 2, 16), (2, 256, 4, 32)]:
        ok = run_test(f"B={B} L={L} H={H} P={P}", B, L, H, P)
        results.append(ok)

    print()
    print("--- GDN block: multiple seeds ---")
    for seed in [0, 7, 42, 1337]:
        ok = run_test(f"B=2 L=32 H=2 P=8 seed={seed}", 2, 32, 2, 8, seed=seed)
        results.append(ok)

    print()
    print("--- GDN block: gradient flow ---")
    results.append(run_gradient_test())

    print()
    print("--- Full model: all 4 configs ---")
    results.append(run_full_model_test("attn,attn,attn,attn", "toy_attn   (control)"))
    results.append(
        run_full_model_test("mamba,mamba,mamba,attn", "toy_hybrid (Mamba+Attn)")
    )
    results.append(run_full_model_test("gdn,gdn,gdn,attn", "toy_gdn_hybrid (GDN+Attn)"))
    results.append(
        run_full_model_test(
            "gdn,gdn,gdn,attn", "toy_deltahybrid (GDN+Attn+MTP)", mtp_lambda=0.3
        )
    )

    print()
    print("--- Full model: mixed GDN+Mamba ---")
    results.append(run_full_model_test("gdn,mamba,gdn,attn", "mixed GDN+Mamba"))

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
