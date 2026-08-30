"""
nanolab.tests — a fast, dependency-light regression suite.

Run it after any change:  ``python -m nanolab.tests``

Everything here is CPU-runnable and tiny (<~30 s), so it is a cheap pre-flight
that locks in the invariants this package was built around: the fused
cross-entropy numerics, every mixer/optimizer/schedule, μP scaling, the
curricula, MoE routing, the diffusion objective, and checkpoint round-trips.
No pytest needed — it is a plain assert harness that prints PASS/FAIL and exits
non-zero on any failure (so it works in CI or a bare shell).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
import json
import math
import statistics
from pathlib import Path

import torch

from .config import MIXERS, OPTIMIZERS, SCHEDULES, build_config
from .model import build_model, fused_linear_cross_entropy
from .optim import (
    MONA, SOAP, MiMuon, Muown, NorMuon, build_optimizers, is_lr_free,
    zeropower_via_newtonschulz5, zeropower_via_polar_express,
)
from .schedules import apply_lr, make_schedule
from .optimizer_funnel import CANDIDATES
from .crossover_replicate import _collect, load_run_timing, timing_summary
from .native_funnel import (
    _mean_ci95, _rank_candidates, _read_result, _t_critical_95, advance,
    champion_argv, unlock_from_gate, write_champion,
)


# ---------------------------------------------------------------------------
# tiny harness
# ---------------------------------------------------------------------------
_RESULTS = []


def test(fn):
    _RESULTS.append(fn)
    return fn


def _cfg(**kw):
    base = dict(vocab_size=256, device="cpu", dtype="fp32", n_layer=2, d_model=64,
                n_head=4, head_dim=16, block_size=32, batch_size=4, compile=False)
    base.update(kw)
    return build_config(None, base)


def _toy_model(**kw):
    torch.manual_seed(0)
    cfg = _cfg(**kw)
    return build_model(cfg), cfg


def _batch(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.block_size), generator=g)
    return x, x.clone()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
@test
def fused_ce_matches_reference():
    torch.manual_seed(0)
    N, D, V = 128, 64, 256
    x = torch.randn(N, D)
    W = torch.randn(V, D) * 0.02
    tgt = torch.randint(0, V, (N,))
    xr, Wr = x.clone().requires_grad_(), W.clone().requires_grad_()
    ref = torch.nn.functional.cross_entropy(xr @ Wr.T, tgt)
    ref.backward()
    xf, Wf = x.clone().requires_grad_(), W.clone().requires_grad_()
    out = fused_linear_cross_entropy(xf, Wf, tgt, n_chunks=4)
    out.backward()
    assert abs(out.item() - ref.item()) < 1e-4, "fused CE value mismatch"
    assert (xf.grad - xr.grad).abs().max() < 1e-4, "fused CE grad_x mismatch"
    assert (Wf.grad - Wr.grad).abs().max() < 1e-4, "fused CE grad_W mismatch"


@test
def all_mixers_fwd_bwd():
    for mx in MIXERS:
        m, cfg = _toy_model(mixer=mx)
        x, y = _batch(cfg)
        _, loss = m(x, y)
        loss.backward()
        assert torch.isfinite(loss), f"{mx} loss not finite"
        g = sum(p.grad.abs().sum() for p in m.parameters() if p.grad is not None)
        assert g > 0, f"{mx} has zero gradient"


@test
def gdn_chunked_matches_sequential():
    # the chunk-parallel GDN kernel must match the O(T) sequential reference in
    # both output and the gradient flowing back to the input — including a
    # block_size NOT divisible by the chunk size (exercises the tail padding).
    for T, chunk in ((32, 8), (30, 8)):
        m, cfg = _toy_model(mixer="gdn", block_size=T, mixer_chunk=chunk)
        gdn = m.blocks[0].mixer
        torch.manual_seed(3)
        x = torch.randn(2, T, cfg.d_model, requires_grad=True)
        xs = x.detach().clone().requires_grad_()
        y_chunk, _ = gdn(x)
        y_seq, _ = gdn._sequential(xs)
        assert (y_chunk - y_seq).abs().max() < 1e-4, f"GDN chunked!=seq output (T={T})"
        y_chunk.square().sum().backward()
        y_seq.square().sum().backward()
        assert (x.grad - xs.grad).abs().max() < 1e-4, f"GDN chunked!=seq grad (T={T})"


@test
def mamba2_chunked_matches_sequential():
    # the chunk-parallel SSD scan must match the O(T) sequential reference in
    # output and input-gradient, including a non-chunk-divisible length (padding).
    for T, chunk in ((32, 8), (30, 8)):
        m, cfg = _toy_model(mixer="mamba2", block_size=T, mixer_chunk=chunk)
        mamba = m.blocks[0].mixer
        torch.manual_seed(4)
        x = torch.randn(2, T, cfg.d_model, requires_grad=True)
        xs = x.detach().clone().requires_grad_()
        y_chunk, _ = mamba(x)
        y_seq, _ = mamba._sequential(xs)
        assert (y_chunk - y_seq).abs().max() < 1e-4, f"Mamba2 chunked!=seq out (T={T})"
        y_chunk.square().sum().backward()
        y_seq.square().sum().backward()
        assert (x.grad - xs.grad).abs().max() < 1e-4, f"Mamba2 chunked!=seq grad (T={T})"


@test
def mingru_value_residual_threads_v0():
    # Layer 0 seeds raw_v; layer>0 blends v0 @ W_v0_up into h_pre via vr_lambda.
    from .mixers import MinGRU

    m, cfg = _toy_model(
        mixer="mingru", n_layer=2, n_kv_head=2, value_residual=True,
        zero_init_proj=False,
    )
    mingru0, mingru1 = m.blocks[0].mixer, m.blocks[1].mixer
    assert isinstance(mingru0, MinGRU) and mingru0.value_residual

    B, T = 2, cfg.block_size
    x = torch.randn(B, T, cfg.d_model)
    _, raw_v0 = mingru0(x)
    assert raw_v0 is not None
    assert raw_v0.shape == (B, T, 2, cfg.head_dim)

    out_base, _ = mingru1(x, v0=None)
    out_vr, _ = mingru1(x, v0=raw_v0.detach())
    assert not torch.allclose(out_base, out_vr), "VR blend should change layer>0 output"

    idx = torch.randint(0, cfg.vocab_size, (B, T))
    h = m.forward_hidden(idx)
    assert torch.isfinite(h).all()

    m_off, _ = _toy_model(mixer="mingru", value_residual=False)
    assert not getattr(m_off.blocks[0].mixer, "value_residual", False)


@test
def all_ffn_types_fwd_bwd():
    for ffn in ("swiglu", "relu2", "gelu", "moe"):
        m, cfg = _toy_model(ffn=ffn)
        x, y = _batch(cfg)
        _, loss = m(x, y)
        loss.backward()
        assert torch.isfinite(loss), f"ffn {ffn} not finite"


@test
def all_optimizers_step_and_converge():
    # overfit one batch: a working optimizer must drive the loss down. Each gets
    # a realistic LR (SGD needs far more than Adam-family — guide §4.2).
    lr_for = {"sgd_momentum": 0.5, "lion": 1e-3,
              "cautious_lion": 1e-3, "prodigy": 1.0}
    for opt in OPTIMIZERS:
        m, cfg = _toy_model(optimizer=opt, lr=lr_for.get(opt, 5e-3))
        opts = build_optimizers(m, cfg)
        sched = make_schedule(cfg)
        x, y = _batch(cfg)
        first = last = None
        for step in range(80):
            if not is_lr_free(cfg):
                apply_lr(opts, sched(step), cfg)
            _, loss = m(x, y)
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            if opt == "sophia" and step % 5 == 0:
                opts[0].update_hessian()
            for o in opts:
                o.step()
            first = first if first is not None else loss.item()
            last = loss.item()
        # smoke check that the optimizer makes real progress overfitting one
        # batch (not a convergence benchmark). 15% in 80 CPU steps cleanly
        # separates a working optimizer from a broken/flat one; sign-based Lion
        # is the slowest mover so it sets the bar.
        assert last < first * 0.85, f"{opt} did not reduce loss ({first:.2f}->{last:.2f})"


@test
def matrix_optimizer_oracles_have_distinct_state_and_updates():
    torch.manual_seed(11)
    g = torch.randn(12, 8)
    ns3 = zeropower_via_newtonschulz5(g, steps=3)
    ns5 = zeropower_via_newtonschulz5(g, steps=5)
    polar = zeropower_via_polar_express(g)
    assert torch.isfinite(polar).all()
    assert not torch.allclose(ns3, ns5), "NS3 was silently mapped to NS5"
    assert not torch.allclose(polar, ns5), "Polar Express was silently mapped to NS5"

    p = torch.nn.Parameter(torch.randn(12, 8) * 0.02)
    p.grad = g.clone()
    nor = NorMuon([p], lr=1e-3)
    nor.step()
    state = nor.state[p]
    assert state["momentum_buffer"].shape == p.shape
    assert state["second_moment"].shape == (12, 1)

    p2 = torch.nn.Parameter(torch.randn(12, 8) * 0.02)
    p2.grad = g.clone()
    mona = MONA([p2], lr=1e-3)
    mona.step()
    assert {"momentum_buffer", "acceleration", "prev_grad"} <= mona.state[p2].keys()

    p3 = torch.nn.Parameter(torch.randn(12, 8) * 0.02)
    p3.grad = g.clone()
    mixed = MiMuon([p3], lr=1e-3, singular_gap=float("inf"))
    mixed.step()
    assert mixed.state[p3]["used_sgd"] == 1

    p4 = torch.nn.Parameter(torch.randn(12, 8) * 0.02)
    p4.grad = g.clone()
    muown = Muown([p4], lr=1e-3)
    muown.step()
    assert torch.isfinite(p4).all()
    assert muown.state[p4]["direction"].shape == p4.shape


@test
def soap_memory_estimate_counts_full_and_axis_state():
    p = torch.nn.Parameter(torch.zeros(12, 8))
    # m/v: 2*96 values; covariance+basis: 2*(12^2+8^2) values.
    assert SOAP.estimated_state_bytes([p], max_precond_dim=16) == 4 * (192 + 416)


@test
def mup_scaling_rules():
    # at 2x base width: hidden LR halves, output_mult halves, embeddings unchanged.
    m1, c1 = _toy_model(mup=True, mup_base_width=64, d_model=64)
    m2, c2 = _toy_model(mup=True, mup_base_width=64, d_model=128, n_head=8, head_dim=16)
    assert abs(m1.width_mult - 1.0) < 1e-6 and abs(m2.width_mult - 2.0) < 1e-6
    assert abs(m2.output_mult - 0.5) < 1e-6, "output_mult should be 1/width"
    o1 = build_optimizers(m1, c1)[0].param_groups[0]["lr"]
    o2 = build_optimizers(m2, c2)[0].param_groups[0]["lr"]
    assert abs(o2 - o1 / 2) < 1e-9, "hidden LR should scale 1/width"


@test
def per_layer_sp_scales_hidden_lr_by_sqrt_width():
    # Everett et al. (arXiv:2407.05872) Table 1, Standard row, Adam LR column:
    # hidden LR ~ 1/sqrt(n) while the embedding LR is width-constant. Contrast with
    # muP, which scales the hidden LR by 1/n -- the two must not agree at width > 1.
    base = _toy_model(per_layer_sp=True, mup_base_width=64, d_model=64, optimizer="adamw")
    wide = _toy_model(per_layer_sp=True, mup_base_width=64, d_model=256,
                      n_head=8, head_dim=32, optimizer="adamw")
    lr_base = build_optimizers(*base)[0].param_groups[0]["lr"]
    lr_wide = build_optimizers(*wide)[0].param_groups[0]["lr"]
    # width ratio 4 -> hidden LR divided by sqrt(4) = 2
    assert abs(lr_wide - lr_base / 2) < 1e-9, (
        f"hidden LR should scale 1/sqrt(width): {lr_base} -> {lr_wide}")
    mu = _toy_model(mup=True, mup_base_width=64, d_model=256, n_head=8, head_dim=32,
                    optimizer="adamw")
    lr_mup = build_optimizers(*mu)[0].param_groups[0]["lr"]
    assert abs(lr_mup - lr_base / 4) < 1e-9, "muP should scale 1/width, not 1/sqrt(width)"
    assert abs(lr_wide - lr_mup) > 1e-9, "per-layer SP must differ from muP at width > 1"


@test
def per_layer_sp_and_mup_are_mutually_exclusive():
    m, cfg = _toy_model(per_layer_sp=True, mup=True, mup_base_width=64, d_model=128)
    try:
        build_optimizers(m, cfg)
    except ValueError:
        return
    raise AssertionError("enabling both per_layer_sp and mup must raise")


@test
def embed_lr_mult_moves_only_the_embedding_group():
    # Kalra & Barkeshli probe: raise the embedding LR alone and change nothing else.
    # Checked on the Muon hybrid, where embeddings sit on the AdamW half.
    m0, c0 = _toy_model(optimizer="muon_ns5_adamw", d_model=128)
    m4, c4 = _toy_model(optimizer="muon_ns5_adamw", d_model=128, embed_lr_mult=4.0)
    muon0, adam0 = build_optimizers(m0, c0)
    muon4, adam4 = build_optimizers(m4, c4)
    assert abs(muon0.param_groups[0]["lr"] - muon4.param_groups[0]["lr"]) < 1e-12, \
        "matrix (Muon) LR must be untouched by embed_lr_mult"
    assert abs(adam4.param_groups[0]["lr"] - 4.0 * adam0.param_groups[0]["lr"]) < 1e-12, \
        "embedding/head LR should scale by embed_lr_mult"
    assert abs(adam0.param_groups[1]["lr"] - adam4.param_groups[1]["lr"]) < 1e-12, \
        "scalar group must be untouched by embed_lr_mult"


@test
def embed_lr_mult_survives_the_schedule():
    # apply_lr rescales from each group's initial_lr, so the ablation must persist
    # across the cosine rather than being flattened at the first step.
    m, cfg = _toy_model(optimizer="muon_ns5_adamw", d_model=128, embed_lr_mult=4.0)
    opts = build_optimizers(m, cfg)
    sched = make_schedule(cfg)
    ratios = []
    for step in (0, cfg.max_steps // 2, cfg.max_steps - 1):
        apply_lr(opts, sched(step), cfg)
        ratios.append(opts[1].param_groups[0]["lr"] / opts[1].param_groups[1]["lr"])
    assert max(ratios) - min(ratios) < 1e-9, f"embed/scalar ratio drifted: {ratios}"
    assert abs(ratios[0] - 4.0) < 1e-9, f"ratio should stay 4x, got {ratios[0]}"


@test
def apply_lr_preserves_group_ratios():
    # the Muon hybrid: matrix group and adam group must keep their LR ratio as
    # the schedule scales both.
    m, cfg = _toy_model(optimizer="muon_ns5_adamw")
    opts = build_optimizers(m, cfg)
    sched = make_schedule(cfg)
    apply_lr(opts, sched(0), cfg)
    r0 = opts[0].param_groups[0]["lr"] / opts[1].param_groups[0]["lr"]
    apply_lr(opts, sched(cfg.max_steps // 2), cfg)
    r1 = opts[0].param_groups[0]["lr"] / opts[1].param_groups[0]["lr"]
    assert abs(r0 - r1) < 1e-6, "Muon/Adam LR ratio drifted under schedule"


@test
def schedules_shapes():
    for name in SCHEDULES:
        cfg = _cfg(schedule=name, warmup_steps=10, max_steps=100)
        s = make_schedule(cfg)
        assert s(0) < s(10), f"{name} should ramp during warmup"
        peak = s(11)
        assert peak > 0, f"{name} non-positive peak"
        if name in ("cosine", "wsd"):
            assert s(99) < peak, f"{name} should decay below peak by the end"


@test
def moe_router_gets_gradient():
    m, cfg = _toy_model(ffn="moe", moe_experts=8, moe_top_k=2)
    x, y = _batch(cfg)
    _, loss = m(x, y)
    loss.backward()
    g = m.blocks[0].ffn.gate.weight.grad
    assert g is not None and g.norm() > 0, "MoE gate received no gradient"
    assert m.blocks[0].ffn.aux is not None, "MoE aux loss not set"


@test
def curriculum_growth():
    from .train import _curriculum_frontier, _curriculum_len
    cs = _cfg(curriculum="seqlen", block_size=128, curriculum_start_len=32,
              curriculum_frac=0.5, max_steps=40)
    lens = [_curriculum_len(cs, s) for s in (0, 20, 40)]
    assert lens[0] == 32 and lens[-1] == 128 and lens[0] < lens[1] < lens[2] + 1
    cd = _cfg(curriculum="difficulty", curriculum_frac=0.5, max_steps=40)
    fr = [_curriculum_frontier(cd, s) for s in (0, 20, 40)]
    assert abs(fr[0] - 0.1) < 1e-6 and abs(fr[-1] - 1.0) < 1e-6 and fr[0] < fr[1]


@test
def diffusion_masked_loss():
    from .diffusion import MASK_ID, diffusion_loss, mask_tokens
    # vocab must include MASK_ID (50257); keep real targets small so a clean
    # token never coincides with the mask id.
    m, cfg = _toy_model(vocab_size=MASK_ID + 16)
    m.set_causal(False)
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 200, (cfg.batch_size, cfg.block_size), generator=g)
    t = torch.rand(cfg.batch_size) * 0.99 + 0.01
    xm, mask = mask_tokens(x, t)
    loss = diffusion_loss(m, xm, x, t, mask, m.lm_head.weight)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0, "diffusion loss invalid"
    assert (xm == MASK_ID).any(), "no tokens were masked"
    # clean target != masked input at masked positions (the collapse-bug guard)
    assert (xm[mask] != x[mask]).all(), "masked positions should differ from target"


@test
def set_causal_toggles_attention():
    m, cfg = _toy_model(mixer="attention")
    assert m.blocks[0].mixer.causal is True
    m.set_causal(False)
    assert m.blocks[0].mixer.causal is False


@test
def block_causal_mask_structure():
    # block_len=1 -> plain causal; block_len>=T -> fully bidirectional.
    from .mixers import block_causal_mask
    T = 6
    c = block_causal_mask(T, 1, "cpu", {})[0, 0]
    assert torch.equal(c, torch.tril(torch.ones(T, T, dtype=torch.bool))), "block_len=1 != causal"
    full = block_causal_mask(T, T, "cpu", {})[0, 0]
    assert full.all(), "block_len>=T should attend everywhere"
    b = block_causal_mask(T, 2, "cpu", {})[0, 0]   # blocks {0,1},{2,3},{4,5}
    assert bool(b[1, 0]) and bool(b[0, 1]), "within-block must be bidirectional"
    assert bool(b[2, 1]) and not bool(b[1, 2]), "across-block must be causal"


@test
def set_block_attention_spans_modes():
    # block_len=1 must reproduce causal AR output; block_len>=T the bidirectional
    # output — proving one weight set spans AR <-> block <-> full diffusion.
    m, cfg = _toy_model(mixer="attention")
    x, _ = _batch(cfg)
    m.set_causal(True)
    causal = m.forward_hidden(x)
    m.set_causal(False)
    bidir = m.forward_hidden(x)
    m.set_block_attention(1)
    assert torch.allclose(m.forward_hidden(x), causal, atol=1e-5), "block_len=1 != causal"
    m.set_block_attention(cfg.block_size)
    assert torch.allclose(m.forward_hidden(x), bidir, atol=1e-5), "block_len>=T != bidirectional"
    m.set_block_attention(0)         # disabled -> falls back to causal flag
    assert m.blocks[0].mixer.block_attn == 0


@test
def kv_cached_blockwise_matches_uncached():
    # The KV-cached semi-AR sampler must produce identical greedy output to the
    # full-trunk reference sampler — that's the proof the cache invariant holds.
    import contextlib
    from .diffusion import MASK_ID, sample_blockwise, sample_blockwise_cached

    class _Enc:
        def encode_ordinary(self, s): return [5, 6, 7]
        def decode(self, ids): return ",".join(map(str, ids))

    torch.manual_seed(0)
    m, cfg = _toy_model(vocab_size=MASK_ID + 16, n_layer=2, d_model=64, n_head=4,
                        head_dim=16, block_size=64, n_kv_head=2, value_residual=True,
                        gated_attention=True)
    m.eval()
    enc, nc = _Enc(), contextlib.nullcontext()
    kw = dict(gen_len=13, block_len=4, device="cpu", autocast=nc, temperature=0.0)
    ref = sample_blockwise(m, enc, "hi", steps_per_block=3, **kw)
    cac = sample_blockwise_cached(m, enc, "hi", steps_per_block=3, **kw)
    assert ref == cac, f"cached sampler diverged from reference:\n  ref={ref}\n  cac={cac}"


@test
def kv_cached_selfspec_is_lossless():
    # Cached self-speculation must equal (a) the uncached self-spec sampler and
    # (b) plain greedy AR decoding — the lossless guarantee, now with a KV cache.
    import contextlib
    from .diffusion import MASK_ID, sample_selfspec, sample_selfspec_cached

    class _Enc:
        def encode_ordinary(self, s): return [5, 6, 7]
        def decode(self, ids): return ",".join(map(str, ids))

    torch.manual_seed(0)
    m, cfg = _toy_model(vocab_size=MASK_ID + 16, n_layer=2, d_model=64, n_head=4,
                        head_dim=16, block_size=64, n_kv_head=2, value_residual=True,
                        gated_attention=True)
    m.eval()
    enc, nc = _Enc(), contextlib.nullcontext()
    kw = dict(gen_len=13, block_len=4, device="cpu", autocast=nc, temperature=0.0)
    ref = sample_selfspec(m, enc, "hi", draft_steps=3, **kw)
    cac = sample_selfspec_cached(m, enc, "hi", draft_steps=3, **kw)
    assert ref == cac, f"cached self-spec diverged:\n  ref={ref}\n  cac={cac}"

    # independent ground truth: greedy AR continuation of the same prompt
    P = 3
    ids = [5, 6, 7]
    m.set_block_attention(0)
    m.set_causal(True)
    for _ in range(13):
        lg = torch.nn.functional.linear(m.forward_hidden(torch.tensor([ids])),
                                        m.lm_head.weight)[0]
        lg[:, MASK_ID] = -1e30
        ids.append(int(lg[-1].argmax()))
    greedy = ",".join(map(str, ids[P:P + 13]))
    assert cac == greedy, f"self-spec not lossless vs greedy AR:\n  ss={cac}\n  ar={greedy}"


@test
def checkpoint_roundtrip():
    m, cfg = _toy_model()
    x, y = _batch(cfg)
    _, l0 = m(x, y)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ck.pt"
        torch.save({"model": m.state_dict(), "cfg": cfg.to_dict()}, p)
        m2 = build_model(build_config(None, cfg.to_dict()))
        m2.load_state_dict(torch.load(p, weights_only=False)["model"])
    _, l1 = m2(x, y)
    assert abs(l0.item() - l1.item()) < 1e-5, "checkpoint did not restore exactly"


@test
def tied_embeddings_shared():
    m, cfg = _toy_model(tie_embeddings=True)
    assert m.lm_head.weight is m.tok_emb.weight, "embeddings not tied"


@test
def fused_ce_equals_unfused_in_model():
    # the model's fused-CE path must match the plain-logits path.
    m, cfg = _toy_model(fused_ce=False)
    x, y = _batch(cfg)
    _, plain = m(x, y)
    m.cfg.fused_ce = True
    _, fused = m(x, y)
    assert abs(plain.item() - fused.item()) < 1e-3, "fused vs plain CE mismatch in model"


@test
def optimizer_funnel_native_surface_is_explicit():
    native = [candidate for candidate in CANDIDATES if candidate.native_ready]
    blocked = [candidate for candidate in CANDIDATES if not candidate.native_ready]
    assert len(native) == 14
    assert {candidate.name for candidate in blocked} == {"mimuon_adamw", "soap_adamw"}
    assert all(candidate.native_block_reason for candidate in blocked)


@test
def native_funnel_collects_metrics_and_guards_champion_recipe():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "metrics.jsonl").write_text(
            '{"step":0,"loss":4.0,"grad_norm_global":1.0,"step_ms":9.0,'
            '"optimizer_ms":2.0,"current_physical_mb":100.0,"dispatches":50,'
            '"research":{"nonfinite_values":0}}\n'
            '{"final_ema_sliding_bpb":2.0}\n',
            encoding="utf-8",
        )
        result = _read_result(path)
    assert result["finite"] and result["validation_bpb"] == 2.0
    argv = champion_argv("muon_ns5_adamw", 0.025, Path("data"), Path("bytes.json"))
    assert argv[argv.index("--total-steps") + 1] == "2000"
    assert argv[argv.index("--final-warmdown") + 1] == "350"
    assert argv[argv.index("--checkpoint-every") + 1] == "250"
    assert "--no-final-weight-save" not in argv


@test
def native_funnel_classifies_json_null_numerics_as_failed():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "metrics.jsonl").write_text(
            '{"step":50,"loss":null,"grad_norm_global":null,"step_ms":9.0,'
            '"optimizer_ms":2.0,"current_physical_mb":100.0,"dispatches":50}\n'
            '{"final_ema_sliding_bpb":NaN}\n',
            encoding="utf-8",
        )
        result = _read_result(path)
    assert not result["finite"]
    assert result["validation_bpb"] is None


@test
def native_funnel_applies_system_tiebreakers_only_on_ci_overlap():
    # Five seeds per arm: the interval is informative, so the declared systems
    # tie breakers may decide between arms whose intervals overlap.  This used
    # to be written with two seeds per arm, which cannot support the overlap
    # test at all -- see the underpowered test below.
    ranked = _rank_candidates([
        *(_funnel_job("lower_mean_slower", seed, bpb, 20.0)
          for seed, bpb in zip((42, 2026, 1337, 100, 777),
                               (1.99, 2.01, 2.00, 2.005, 1.995))),
        *(_funnel_job("overlap_faster", seed, bpb, 10.0)
          for seed, bpb in zip((42, 2026, 1337, 100, 777),
                               (2.001, 2.009, 2.005, 2.003, 2.007))),
        *(_funnel_job("clearly_worse", seed, bpb, 1.0)
          for seed, bpb in zip((42, 2026, 1337, 100, 777),
                               (2.50, 2.51, 2.49, 2.505, 2.495))),
    ])
    assert all(item["ci95_informative"] for item in ranked)
    assert ranked[0]["candidate"] == "overlap_faster"
    assert ranked[0]["tiebreakers_applied"] is True
    assert ranked[-1]["candidate"] == "clearly_worse"
    assert ranked[-1]["confidence_interval_overlaps_best"] is False


def _funnel_job(candidate, seed, bpb, step_ms):
    return {
        "candidate": candidate,
        "seed": seed,
        "status": "completed",
        "result": {
            "finite": True,
            "validation_bpb": bpb,
            "mean_logged_step_ms": step_ms,
            "max_current_physical_mb": 100.0,
            "training_loss_trace": [{"step": 9, "loss": 3.0}],
        },
    }


@test
def crossover_timing_reads_measured_wall_clock():
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory) / "cx32_attention_s1337"
        run.mkdir()
        (run / "metrics.jsonl").write_text(
            '{"event":"train","step":10,"tok_s":120000.0,"mfu":0.31}\n'
            '{"event":"train","step":20,"tok_s":124000.0,"mfu":0.33}\n'
            '{"event":"done","best_val":4.22,"tokens":50000000,'
            '"elapsed_s":900.0,"mean_tok_s":55555.6}\n',
            encoding="utf-8")
        row = load_run_timing(run)
    assert row["source"] == "measured"
    assert row["elapsed_s"] == 900.0
    assert abs(row["median_tok_s"] - 122000.0) < 1e-6
    assert abs(row["median_mfu"] - 0.32) < 1e-9


@test
def crossover_timing_estimates_pre_2026_08_runs_and_labels_them():
    # Runs finished before Logger.done persisted elapsed_s carry no wall clock.
    # They must be estimated from median tok/s AND flagged, never counted as a
    # measurement.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        legacy = root / "cx50_mingru_s42"
        legacy.mkdir()
        (legacy / "metrics.jsonl").write_text(
            '{"event":"train","step":10,"tok_s":100000.0,"mfu":0.25}\n'
            '{"event":"done","best_val":4.45,"tokens":50000000}\n',
            encoding="utf-8")
        untimed = root / "cx50_gdn_s42"
        untimed.mkdir()
        (untimed / "metrics.jsonl").write_text(
            '{"event":"eval","step":10,"val_loss":5.0,"tokens":163840}\n',
            encoding="utf-8")

        row = load_run_timing(legacy)
        assert row["source"] == "estimated_from_median_tok_s"
        assert abs(row["elapsed_s"] - 500.0) < 1e-6

        blank = load_run_timing(untimed)
        assert blank["source"] == "missing"
        assert blank["elapsed_s"] is None

        summary = timing_summary(root)
    assert summary["runs_seen"] == 2
    assert summary["runs_measured"] == 0
    assert summary["runs_estimated"] == 1
    assert summary["runs_untimed"] == 1
    # The untimed run contributes nothing rather than silently reading as zero.
    assert abs(summary["gpu_hours_total"] - 500.0 / 3600.0) < 1e-9
    assert summary["gpu_hours_measured"] == 0.0


@test
def native_funnel_ci95_uses_student_t_not_the_normal_quantile():
    # Two samples 0.02 apart: sd = 0.0141421, se = 0.01, so a z-interval would
    # report 0.0196 and a correct df=1 t-interval reports 0.127062.  The old
    # code shipped the z value, which is ~6.5x too narrow.
    half_width, df, informative = _mean_ci95([2.04, 2.06])
    assert df == 1
    assert not informative
    assert abs(half_width - 0.12706205) < 1e-6, half_width
    assert abs(_t_critical_95(1) - 12.706205) < 1e-6
    assert abs(_t_critical_95(4) - 2.776445) < 1e-6
    # Past the tabulated range the normal quantile is the documented fallback.
    assert abs(_t_critical_95(500) - 1.959964) < 1e-6


@test
def crossover50m_summary_is_not_stale_against_its_runs():
    # summary.json is an intermediate: the manuscript's ten-arm board is built
    # from it, not from the runs.  It went stale once and nothing noticed.  An
    # older aligner took the *exact intersection* of seed token-grids; three of
    # mamba2's five seeds ran a 49.17M eval cadence and two ran 49.99M, so the
    # intersection collapsed to three points and the arm's published "50M"
    # figure was measured at 40.16M -- 80% of the horizon of the other nine.
    # `_align_at_or_after` fixed the aligner, but the committed artifact was
    # never regenerated, so the paper kept quoting the stale file.
    #
    # This asserts the committed summary is what the current aligner produces
    # from the committed runs.  Regenerate summary.json when this fails.
    root = Path(__file__).resolve().parents[1]
    out = root / "nanolab" / "out" / "crossover50m"
    stored = json.loads((out / "summary.json").read_text())
    fresh = _collect(out)

    assert set(stored["arms"]) == set(fresh["arms"]), "arm set differs"
    for name, want in fresh["arms"].items():
        got = stored["arms"][name]
        assert got["n"] == want["n"], f"{name}: n {got['n']} != {want['n']}"
        assert len(got["tokens"]) == len(want["tokens"]), (
            f"{name}: {len(got['tokens'])} grid points stored, "
            f"{len(want['tokens'])} derived from the runs -- summary.json is stale")
        for i, (a, b) in enumerate(zip(got["tokens"], want["tokens"])):
            assert abs(a - b) < 1.0, f"{name}[{i}]: token grid drift {a} != {b}"
        for key in ("mean", "lo", "hi"):
            for i, (a, b) in enumerate(zip(got[key], want[key])):
                assert abs(a - b) < 1e-6, f"{name}[{i}]: {key} {a} != {b}"

    # And the board must stay close to a single horizon; the stale artifact put
    # one arm 20% short of the rest while the table read "at 50M tokens".
    horizons = [a["tokens"][-1] for a in stored["arms"].values()]
    spread = (max(horizons) - min(horizons)) / max(horizons)
    assert spread < 0.05, (
        f"board arms span {spread:.1%} of horizon; that is a measurement "
        f"difference, not a ranking at one token count")


@test
def paper_matched_batch_board_uses_student_t_not_the_normal_quantile():
    # The manuscript's matched batch-32 board was the last place in the repo
    # still computing a 95% interval with z = 1.96.  At n = 5 the correct
    # multiplier is t_4 = 2.776445, so every interval on that board was
    # reported 1.42x too narrow -- the same defect the funnel fixed above,
    # sitting in the very script that checks the manuscript for drift.
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_derive_figures", root / "paper" / "derive_figures.py")
    df = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(df)

    board = df.matched_batch_board()
    assert board is not None, "crossover50m_matched32 artifacts are missing"
    assert board["rows"], "matched batch-32 board derived no rows"

    # Recompute each arm's sample straight off disk.  Deriving the standard
    # error from the reported half-width instead would make this test a
    # tautology: half/(half/t) == t for any half, so it would pass against the
    # very z-quantile code it exists to catch.
    #
    # Read the FINAL EVALUATION, not the `done` record's best_val.  This test
    # originally sampled best_val, which is the minimum over all evaluations --
    # the quantity the manuscript's own section 3.2 rule says is not a paired
    # snapshot and is never reported as a ranking.  The board was corrected to
    # stop using it (it disagreed with the eval-aligned main board by 0.018 BPB
    # on the same arm), so a test still sampling best_val compares the board's
    # interval against the standard error of a different sample and fails for
    # the wrong reason.
    src = Path(df.OUT) / "crossover50m_matched32"
    samples: dict[str, list[float]] = {}
    for run_dir in sorted(src.iterdir()):
        if not run_dir.is_dir() or "_s" not in run_dir.name:
            continue
        arm = run_dir.name.rsplit("_s", 1)[0].replace("cx32_", "")
        evals: dict[int, float] = {}
        for line in (run_dir / "metrics.jsonl").open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "eval" and row.get("val_loss") is not None:
                evals[row["tokens"]] = row["val_loss"]
        if evals:
            samples.setdefault(arm, []).append(evals[max(evals)])

    checked = 0
    for row in board["rows"]:
        n = row["n"]
        assert row["df"] == n - 1, row
        xs = samples[row["arm"]]
        assert len(xs) == n, f"{row['arm']}: board n={n}, disk n={len(xs)}"
        # Pin the QUANTITY as well as the multiplier: if the board regresses to
        # best_val the mean moves and this fires, rather than only the interval
        # silently changing shape.
        assert abs(row["final"] - statistics.fmean(xs)) < 1e-9, (
            f"{row['arm']}: board mean {row['final']:.6f} is not the mean of the "
            f"final evaluations {statistics.fmean(xs):.6f} -- the board is "
            f"tabulating a different quantity (best_val?)")
        sem = statistics.stdev(xs) / math.sqrt(n)
        half = (row["ci"][1] - row["ci"][0]) / 2
        multiplier = half / sem
        assert abs(multiplier - _t_critical_95(n - 1)) < 1e-4, (
            f"{row['arm']}: interval uses multiplier {multiplier:.4f}, expected "
            f"t_{n - 1} = {_t_critical_95(n - 1):.4f} "
            f"(1.96 here is the normal-quantile defect)")
        checked += 1
    assert checked >= 8, f"only {checked} arms checked"


@test
def native_funnel_single_sample_interval_is_undefined_not_zero():
    # A one-seed arm has no interval.  Reporting 0.0 made an unmeasured arm
    # look infinitely precise and let it be declared separated from every rival.
    half_width, df, informative = _mean_ci95([2.42])
    assert half_width == math.inf
    assert df == 0
    assert not informative


@test
def native_funnel_skips_tiebreakers_when_seeds_are_underpowered():
    # Two seeds per arm.  A correct df=1 interval on the leader spans roughly
    # [0.73, 3.27], so even "clearly_worse" overlaps it -- and the systems tie
    # breakers would hand the stage to the fastest arm regardless of quality.
    # The overlap test cannot run here, so it must not be reported as having
    # run, and ranking falls back to mean validation BPB.
    ranked = _rank_candidates([
        _funnel_job("lower_mean_slower", 42, 1.9, 20.0),
        _funnel_job("lower_mean_slower", 2026, 2.1, 20.0),
        _funnel_job("clearly_worse", 42, 2.5, 1.0),
        _funnel_job("clearly_worse", 2026, 2.5, 1.0),
    ])
    assert [item["candidate"] for item in ranked] == [
        "lower_mean_slower", "clearly_worse",
    ]
    for item in ranked:
        assert item["confidence_interval_overlaps_best"] is None
        assert item["tiebreakers_applied"] is False
        assert "n_seeds=[2]" in item["tiebreaker_skip_reason"]


@test
def champion_unlock_requires_winner_specific_exact_gate():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        champion = root / "champion.json"
        gate = root / "gate.json"
        write_champion(champion, "adamw", 6e-4, 1.9, locked=True)
        # Ambient residual swap is allowed; induced growth is not.
        gate.write_text(
            '{"passed":true,"optimizer":"adamw","parameter_count":128367988,'
            '"current_physical_mb":13000,"dispatches":1700,'
            '"swap_before_mb":0.25,"swap_after_mb":0.25,"swap_delta_mb":0.0,'
            '"swap_pressure":false}',
            encoding="utf-8",
        )
        unlock_from_gate(champion, gate, Path("data"), Path("bytes.json"))
        value = json.loads(champion.read_text(encoding="utf-8"))
    assert not value["locked"]
    assert value["command_argv"]


@test
def champion_unlock_rejects_induced_swap_pressure():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        champion = root / "champion.json"
        gate = root / "gate.json"
        write_champion(champion, "adamw", 6e-4, 1.9, locked=True)
        gate.write_text(
            '{"passed":true,"optimizer":"adamw","parameter_count":128367988,'
            '"current_physical_mb":13000,"dispatches":1700,'
            '"swap_before_mb":0.25,"swap_after_mb":16.0}',
            encoding="utf-8",
        )
        try:
            unlock_from_gate(champion, gate, Path("data"), Path("bytes.json"))
            raise AssertionError("expected induced swap to reject unlock")
        except RuntimeError as exc:
            assert "swap pressure increased" in str(exc)


@test
def funnel_advance_excludes_failed_candidate_without_blocking_survivors():
    def result(bpb):
        return {
            "finite": True,
            "validation_bpb": bpb,
            "mean_logged_step_ms": 10.0,
            "max_current_physical_mb": 100.0,
            "training_loss_trace": [{"step": 9, "loss": 3.0}],
        }

    candidates = ["adamw", "muon_ns5_adamw", "a", "b", "c", "d", "e", "prodigy"]
    jobs = []
    for index, candidate in enumerate(candidates):
        jobs.append({
            "id": f"lr__{candidate}", "stage": "lr_sweep_16m", "candidate": candidate,
            "seed": 1337, "steps": 100, "lr": 0.001 + index * 0.001,
            "status": "completed", "result": result(4.0 + index), "output": "unused",
        })
        stable = {
            "id": f"stable__{candidate}", "stage": "stable_500", "candidate": candidate,
            "seed": 1337, "steps": 500, "lr": 0.001 + index * 0.001,
            "status": "completed", "result": result(2.0 + index * 0.01), "output": "unused",
        }
        if candidate == "prodigy":
            stable["status"] = "failed"
            stable["result"] = {"finite": False, "validation_bpb": None,
                                "failure_reason": "numerical failure"}
        jobs.append(stable)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan = root / "plan.json"
        champion = root / "champion.json"
        plan.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
        advance(plan, champion)
        value = json.loads(plan.read_text(encoding="utf-8"))
    next_jobs = [job for job in value["jobs"] if job["stage"] == "advance_1000"]
    assert next_jobs and all(job["candidate"] != "prodigy" for job in next_jobs)
    assert value["history"][-1]["excluded_candidates"][0]["candidate"] == "prodigy"


@test
def parse_layer_mixers_uniform_and_star_syntax():
    from .config import parse_layer_mixers

    uniform = _cfg(n_layer=12, mixer="mingru")
    assert parse_layer_mixers(uniform) == ("mingru",) * 12

    hybrid = _cfg(n_layer=12, mixer="gdn", layer_mixers="gdn*10,attention*2")
    kinds = parse_layer_mixers(hybrid)
    assert kinds == ("gdn",) * 10 + ("attention",) * 2

    aliased = _cfg(n_layer=12, mixer="gdn",
                   layer_mixers="gdn*3,attn,gdn*3,attn,gdn*3,attn")
    kinds = parse_layer_mixers(aliased)
    assert kinds.count("attention") == 3 and kinds.count("gdn") == 9
    assert kinds[3] == kinds[7] == kinds[11] == "attention"


@test
def every_registered_arm_expands_to_exactly_n_layer():
    """Every Arm in the suite registry must be a legal 12-layer stack.

    An arm whose spec does not sum to n_layer is not caught until a job launches,
    which on a rented GPU means it is caught after it has been billed. E10 added
    four arms by hand; this is the check that a fifth cannot be added wrong.
    """
    from .config import build_config, parse_layer_mixers
    from .crossover_replicate import ARMS, RATIO_ARMS
    for arm in ARMS:
        cfg = build_config("crossover50m", {"run_name": "probe", "mixer": arm.mixer,
                                            "layer_mixers": arm.layer_mixers})
        kinds = parse_layer_mixers(cfg)
        assert len(kinds) == cfg.n_layer, (
            f"{arm.name}: {len(kinds)} layers, expected {cfg.n_layer}")
    names = {a.name for a in ARMS}
    missing = [n for n in RATIO_ARMS if n not in names]
    assert not missing, f"RATIO_ARMS names no registered arm: {missing}"
    # the E10 set must actually vary the ratio, or it is four copies of one point
    ratios = {}
    for arm in ARMS:
        if arm.name not in RATIO_ARMS:
            continue
        cfg = build_config("crossover50m", {"run_name": "probe", "mixer": arm.mixer,
                                            "layer_mixers": arm.layer_mixers})
        kinds = parse_layer_mixers(cfg)
        ratios[arm.name] = sum(1 for k in kinds if k == "attention")
    assert len(set(ratios.values())) >= 3, (
        f"E10 is meant to sweep the attention ratio; got {ratios}")


@test
def smoke_runs_never_land_in_a_suite_directory():
    """A 40-step smoke run writes a real `done` record with a real best_val, and
    `_collect` reads every directory under the out root. Writing smoke beside the
    suite broke the aligner: the committed summary said n=5 for attention while a
    fresh _collect found 0. Same defect docs/GPU_BUNDLE.md records for the other
    runner, fixed there and left here."""
    import argparse
    from . import crossover_replicate as cr
    seen = {}

    def fake_run(job, out_root, smoke=False, **kw):
        seen[job["id"]] = Path(out_root)
        return 0

    orig = cr.run_job if hasattr(cr, "run_job") else None
    with tempfile.TemporaryDirectory() as td:
        suite = Path(td) / "crossover50m"
        (suite / "cx50_attention_s1337").mkdir(parents=True)
        ns = argparse.Namespace(out=str(suite), arm=None, seed=None)
        target = cr.Path(ns.out) / "_smoke"
        assert target.parent == suite, "smoke must nest under the suite root"
        assert target != suite, "smoke must NOT be the suite root itself"
        # the real cmd_smoke computes exactly this path
        src = (Path(__file__).parent / "crossover_replicate.py").read_text()
        assert 'out_root = Path(args.out) / "_smoke"' in src, (
            "cmd_smoke writes into the suite directory again")


@test
def isolates_runs_exactly_its_documented_three_stages():
    """`isolates` iterated the whole stage tuple, so adding E10's stage to it
    would have silently enlarged an existing command by 20 unrelated jobs."""
    from .crossover_replicate import ISOLATE_SEQUENCE, ISOLATE_STAGES, stage_by_name
    assert ISOLATE_SEQUENCE == ("matched20", "bs8", "matched32")
    assert "ratio32" not in ISOLATE_SEQUENCE
    assert len(ISOLATE_STAGES) > len(ISOLATE_SEQUENCE), (
        "this test is only meaningful while a stage exists outside the sequence")
    for name in ISOLATE_SEQUENCE:
        assert stage_by_name(name)["name"] == name


@test
def stage_subcommands_resolve_by_name_not_position():
    """They were ISOLATE_STAGES[0]/[1]/[2]: inserting a stage anywhere but the end
    re-pointed a subcommand at another experiment's recipe."""
    from . import crossover_replicate as cr
    for name in ("matched20", "bs8", "matched32", "ratio32"):
        assert cr.stage_by_name(name)["name"] == name
    try:
        cr.stage_by_name("nope")
    except SystemExit as e:
        assert "unknown stage" in str(e)
    else:
        raise AssertionError("an unknown stage name must fail closed")
    # behavioural, not a source scan: reorder the tuple and every subcommand must
    # still resolve to the same recipe. Positional wiring cannot survive this.
    orig = cr.ISOLATE_STAGES
    try:
        cr.ISOLATE_STAGES = tuple(reversed(orig))
        for name in ("matched20", "bs8", "matched32", "ratio32"):
            assert cr.stage_by_name(name)["name"] == name
            assert cr.stage_by_name(name)["out"] == \
                next(x["out"] for x in orig if x["name"] == name)
    finally:
        cr.ISOLATE_STAGES = orig


@test
def wallclock_budgets_match_the_clock_and_anneal_over_their_own_budget():
    """E11 phase 2: every arm trains the same WALL CLOCK, not the same tokens.

    Two properties carry the experiment. Each arm's budget must land on the target
    wall clock (else the arms are not time-matched, which is the whole point), and
    each must anneal its cosine over the budget it actually gets. Phase 1 could not
    do the second: it re-read curves annealed over 50M and stopped early, which
    penalises slow arms exactly as section 4.3 measures.
    """
    import os
    from pathlib import Path
    from . import crossover_replicate as cr
    stage = {x["name"]: x for x in cr.ISOLATE_STAGES}["wallclock32"]
    # Rates must come from the tenancy the stage will actually run at, and must be
    # tokens per WALL-CLOCK second rather than per stepping second -- the two
    # mistakes that produced a 1.70x spread on a suite defined by equal wall clock.
    rates = {a: r for a, r in
             ((a, cr.effective_rate_by_arm(tenancy=stage["workers"]).get(a))
              for a in cr.WALLCLOCK_ARMS)}
    assert all(rates.values()), (
        f"missing single-tenant throughput: {rates}")
    budgets = cr.wallclock_budgets(stage["wall_clock_s"], cr.WALLCLOCK_ARMS,
                                   batch=stage["batch"], rates=rates)
    tps = stage["batch"] * 512
    for arm, tok in budgets.items():
        assert tok % tps == 0, f"{arm}: budget {tok} is not a whole number of steps"
        secs = tok / rates[arm]
        assert abs(secs - stage["wall_clock_s"]) <= 2.0, (
            f"{arm}: {secs:.1f}s against a {stage['wall_clock_s']}s target")
    assert len(set(budgets.values())) == len(budgets), (
        "arms of different speed must get different budgets")

    prev = os.environ.get("CROSSOVER_BUDGET_BY_ARM")
    try:
        cr.apply_isolate(stage)
        assert cr.budget_by_arm() == budgets
        for arm in cr.WALLCLOCK_ARMS:
            cfg = cr.job_config({"id": "p", "arm": arm, "mixer": "attention",
                                 "layer_mixers": "", "seed": 1337}, Path("/tmp/x"))
            got = cfg.batch_size * cfg.grad_accum * cfg.block_size * cfg.max_steps
            assert got == budgets[arm], f"{arm}: runs {got}, budget {budgets[arm]}"
            assert cfg.lr_max_steps == cfg.max_steps, (
                f"{arm}: cosine spans {cfg.lr_max_steps} steps but the run is "
                f"{cfg.max_steps} -- it must anneal over its OWN budget")
        assert cr.current_recipe()["budget_by_arm"] == budgets, (
            "the budget map must be in the recipe, or lock_recipe cannot stop a "
            "wall-clock run sharing a directory with a token-matched one")
    finally:
        if prev is None:
            os.environ.pop("CROSSOVER_BUDGET_BY_ARM", None)
        else:
            os.environ["CROSSOVER_BUDGET_BY_ARM"] = prev
    assert stage["workers"] == 1, (
        "phase 2's artifact is elapsed_s; a co-located job measures the scheduler")


@test
def wallclock_budgets_fail_closed_without_measured_throughput():
    """An arm with no committed rate cannot be time-matched. Guessing one would
    put the wall clock out of match silently, which is the failure the whole
    experiment is designed to detect."""
    from . import crossover_replicate as cr
    try:
        cr.wallclock_budgets(600.0, ("attention", "not_a_real_arm"),
                             rates={"attention": 59452.0})
    except SystemExit as e:
        assert "not_a_real_arm" in str(e)
    else:
        raise AssertionError("a missing throughput must fail closed")


@test
def ctx2048_stage_holds_the_suite26_token_cadence():
    """E9 varies sequence length and NOTHING else.

    batch 8 x ctx 2048 = 16,384 tokens/step, the same as suite 26's bs32 x ctx512,
    so eval markers land on identical token counts and the curves are comparable.
    If the cadence drifts, E9 measures sequence length confounded with cadence.
    """
    from .crossover_replicate import ISOLATE_STAGES, scale_to_token_budget
    by = {x["name"]: x for x in ISOLATE_STAGES}
    a, e9 = by["matched32"], by["ctx2048"]
    sa = scale_to_token_budget(a["batch"], block_size=a.get("block", 512),
                               token_budget=a["token_budget"],
                               lr_horizon_tokens=a["lr_horizon"])
    se = scale_to_token_budget(e9["batch"], block_size=e9["block"],
                               token_budget=e9["token_budget"],
                               lr_horizon_tokens=e9["lr_horizon"])
    for field in ("tokens_per_step", "max_steps", "eval_interval", "warmup_steps",
                  "lr_max_steps"):
        assert sa[field] == se[field], (
            f"{field}: matched32={sa[field]} ctx2048={se[field]}")
    assert e9["block"] == 2048 and a.get("block", 512) == 512
    assert e9["out"] != a["out"] and e9["prefix"] != a["prefix"]


@test
def block_size_is_a_recipe_field_so_two_contexts_cannot_share_a_directory():
    """`lock_recipe` compares `current_recipe()`. If block_size were absent from
    it, a 2048-context run would drop into a 512-context suite dir and be averaged
    into the same board -- the confound this paper is about, committed silently."""
    import os
    from . import crossover_replicate as cr
    prev = os.environ.get("CROSSOVER_BLOCK")
    try:
        os.environ["CROSSOVER_BLOCK"] = "512"
        r512 = cr.current_recipe()
        os.environ["CROSSOVER_BLOCK"] = "2048"
        r2048 = cr.current_recipe()
    finally:
        if prev is None:
            os.environ.pop("CROSSOVER_BLOCK", None)
        else:
            os.environ["CROSSOVER_BLOCK"] = prev
    assert "block_size" in r512, "block_size missing from the recorded recipe"
    assert r512 != r2048, "two context lengths produce the same recipe fingerprint"
    assert r512["block_size"] == 512 and r2048["block_size"] == 2048


@test
def ratio32_stage_matches_matched32_in_every_recipe_field():
    """E10's rows join section 4.5's board, so only the arm list may differ.

    If any other recipe field drifts, the new rows are measured at a different
    recipe than the board they are placed in -- the paper's own thesis, committed
    by the runner rather than reported by it.
    """
    from .crossover_replicate import ISOLATE_STAGES
    by = {s["name"]: s for s in ISOLATE_STAGES}
    a, b = by["matched32"], by["ratio32"]
    for field in ("batch", "eval_iters", "token_budget", "lr_horizon"):
        assert a[field] == b[field], (
            f"ratio32.{field}={b[field]!r} but matched32.{field}={a[field]!r}")
    assert a["out"] != b["out"], "ratio32 must not share matched32's out dir"
    assert a["prefix"] != b["prefix"], "ratio32 needs its own job prefix"
    assert set(a["arms"].split(",")) != set(b["arms"].split(","))


@test
def parse_layer_mixers_fails_closed_on_bad_specs():
    from .config import parse_layer_mixers

    try:
        _cfg(n_layer=12, layer_mixers="gdn*8,attention*2")
        raise AssertionError("length mismatch must fail")
    except ValueError as e:
        assert "12" in str(e)

    try:
        parse_layer_mixers(_cfg(n_layer=4, mixer="gdn", layer_mixers="gdn*4,nope"))
        raise AssertionError("unknown mixer must fail")
    except ValueError as e:
        assert "unknown mixer" in str(e)

    try:
        parse_layer_mixers(_cfg(n_layer=4, mixer="gdn", layer_mixers="gdn*0,attention*4"))
        raise AssertionError("zero repeat must fail")
    except ValueError as e:
        assert "repeat count" in str(e)


@test
def hybrid_stack_fwd_bwd_uses_per_layer_mixers():
    from .mixers import Attention, GatedDeltaNet, MinGRU

    m, cfg = _toy_model(
        mixer="gdn", n_layer=4, layer_mixers="gdn*2,attention,mingru",
        mixer_chunk=8, value_residual=True,
    )
    assert [type(b.mixer) for b in m.blocks] == [GatedDeltaNet, GatedDeltaNet, Attention, MinGRU]
    x, y = _batch(cfg)
    _, loss = m(x, y)
    loss.backward()
    assert torch.isfinite(loss)
    g = sum(p.grad.abs().sum() for p in m.parameters() if p.grad is not None)
    assert g > 0
    # attention FLOPs counted only for the attention layer, not all 4
    assert m.flops_per_token() > 0


@test
def crossover_interpolation_and_ci():
    from .crossover_replicate import all_crossover_tokens, first_crossover_tokens, mean_ci

    tokens = [1.0, 2.0, 3.0, 4.0]
    # arch A better early (lower loss), B better late; gap A-B crosses 0 at 2.5
    a = [1.0, 0.8, 0.7, 0.65]
    b = [1.2, 0.9, 0.6, 0.50]
    tok = first_crossover_tokens(tokens, a, b)
    assert abs(tok - 2.5) < 1e-6, tok

    # two flips: A better, B better, A better again
    tokens2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    a2 = [1.0, 0.9, 0.8, 0.5, 0.4]
    b2 = [1.2, 0.85, 0.7, 0.6, 0.55]
    flips = all_crossover_tokens(tokens2, a2, b2)
    assert len(flips) == 2, flips
    assert abs(flips[0] - 1.8) < 1e-6, flips
    assert abs(flips[1] - 3.5) < 1e-6, flips
    assert first_crossover_tokens(tokens2, a2, b2) == flips[0]

    mean, lo, hi = mean_ci([1.0, 1.0, 1.0, 1.0, 1.0])
    assert abs(mean - 1.0) < 1e-12
    assert lo <= mean <= hi
    # n=5, zero variance -> interval collapses to the mean
    assert abs(hi - lo) < 1e-12

    mean, lo, hi = mean_ci([0.0, 1.0, 2.0, 3.0, 4.0])
    assert abs(mean - 2.0) < 1e-12
    assert lo < mean < hi


@test
def scale_to_token_budget_keeps_suite14_token_cadence():
    from .crossover_replicate import scale_to_token_budget, SUITE14_EVAL_TOKENS, TOKEN_BUDGET
    s = scale_to_token_budget(64)
    assert s["tokens_per_step"] == 64 * 512
    assert s["max_steps"] == TOKEN_BUDGET // s["tokens_per_step"]
    assert s["eval_interval"] * s["tokens_per_step"] == SUITE14_EVAL_TOKENS
    s8 = scale_to_token_budget(8)
    assert s8["tokens_per_step"] == 4096
    assert s8["max_steps"] == 12207
    try:
        scale_to_token_budget(0)
        raise AssertionError("zero batch must fail")
    except ValueError:
        pass
    s96 = scale_to_token_budget(96)
    assert s96["tokens_per_step"] == 96 * 512
    cadence = s96["eval_interval"] * s96["tokens_per_step"]
    assert abs(cadence - SUITE14_EVAL_TOKENS) < s96["tokens_per_step"]


@test
def claim_job_skips_held():
    import json
    from .crossover_replicate import claim_job, QUEUE_NAME
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        q = root / QUEUE_NAME
        q.write_text(json.dumps({"jobs": [
            {"id": "held_job", "status": "held", "arm": "gdn", "seed": 1},
            {"id": "open_job", "status": "pending", "arm": "mla", "seed": 1},
        ]}), encoding="utf-8")
        got = claim_job(q, 0, root)
        assert got is not None and got["id"] == "open_job"
        state = json.loads(q.read_text(encoding="utf-8"))
        by_id = {j["id"]: j["status"] for j in state["jobs"]}
        assert by_id["held_job"] == "held"
        assert by_id["open_job"] == "running"
        assert claim_job(q, 1, root) is None


@test
def claim_job_skips_failed():
    import json
    from .crossover_replicate import claim_job, QUEUE_NAME
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        q = root / QUEUE_NAME
        q.write_text(json.dumps({"jobs": [
            {"id": "dead", "status": "failed", "arm": "gdn", "seed": 1},
            {"id": "next", "status": "pending", "arm": "mla", "seed": 1},
        ]}), encoding="utf-8")
        got = claim_job(q, 0, root)
        assert got is not None and got["id"] == "next"
        state = json.loads(q.read_text(encoding="utf-8"))
        by_id = {j["id"]: j["status"] for j in state["jobs"]}
        assert by_id["dead"] == "failed"


@test
def gdn_saturated_alpha_has_finite_grads():
    # Pre-fix: exp(A_{t-1}/A_j) was computed on the full CxC grid; j>t overflowed
    # to inf and autograd NaN'd (GH200 gdn_s1337: finite loss through step 50,
    # nan gnorm at 55). Keys are L2-normalized as in GatedDeltaNet._project.
    from .mixers import gdn_chunked
    torch.manual_seed(1)
    B, H, L, D = 2, 4, 128, 16
    q = torch.randn(B, H, L, D, requires_grad=True)
    k = torch.nn.functional.normalize(torch.randn(B, H, L, D), dim=-1)
    k.requires_grad_(True)
    v = torch.randn(B, H, L, D, requires_grad=True)
    alpha = torch.sigmoid(torch.randn(B, H, L) * 30).detach().requires_grad_(True)
    beta = torch.sigmoid(torch.randn(B, H, L) * 5).detach().requires_grad_(True)
    y = gdn_chunked(q, k, v, alpha, beta, chunk=32)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    for t in (q, k, v, alpha, beta):
        assert t.grad is not None and torch.isfinite(t.grad).all()



@test
def gdn_jobs_cap_batch_at_32():
    from .crossover_replicate import job_batch
    import os
    os.environ["CROSSOVER_BATCH"] = "96"
    assert job_batch({"arm": "gdn", "mixer": "gdn", "layer_mixers": ""}) == 32
    assert job_batch({"arm": "hybrid_gdn_periodic", "mixer": "gdn",
                      "layer_mixers": "gdn*3,attention"}) == 32
    assert job_batch({"arm": "mla", "mixer": "mla", "layer_mixers": ""}) == 96
    assert job_batch({"arm": "hybrid_mamba10_attn2", "mixer": "mamba2",
                      "layer_mixers": "mamba2*10,attention*2"}) == 96
    os.environ.pop("CROSSOVER_BATCH", None)


@test
def locked20_recipe_is_one_batch_two_arms():
    import os
    from .crossover_replicate import (
        LOCKED20_TOKEN_BUDGET, TOKEN_BUDGET, expand_grid, job_config,
        scale_to_token_budget, selected_arms,
    )
    os.environ["CROSSOVER_ARMS"] = "attention,mingru"
    os.environ["CROSSOVER_JOB_PREFIX"] = "cx20"
    os.environ["CROSSOVER_BATCH"] = "32"
    os.environ["CROSSOVER_EVAL_ITERS"] = "20"
    os.environ["CROSSOVER_TOKEN_BUDGET"] = str(LOCKED20_TOKEN_BUDGET)
    os.environ["CROSSOVER_LR_HORIZON"] = str(TOKEN_BUDGET)
    try:
        names = [a.name for a in selected_arms()]
        assert names == ["attention", "mingru"]
        jobs = expand_grid()
        assert len(jobs) == 10
        assert all(j["id"].startswith("cx20_") for j in jobs)
        assert {j["arm"] for j in jobs} == {"attention", "mingru"}
        scaled = scale_to_token_budget(
            32, token_budget=LOCKED20_TOKEN_BUDGET, lr_horizon_tokens=TOKEN_BUDGET)
        assert scaled["batch_size"] == 32
        assert scaled["max_steps"] == LOCKED20_TOKEN_BUDGET // (32 * 512)
        long = scale_to_token_budget(32, token_budget=TOKEN_BUDGET)
        assert scaled["lr_max_steps"] == long["max_steps"]
        assert scaled["lr_max_steps"] > scaled["max_steps"]
        cfg = job_config(jobs[0], Path("/tmp"), smoke=False)
        assert cfg.batch_size == 32
        assert cfg.eval_iters == 20
        assert cfg.max_steps == scaled["max_steps"]
        assert cfg.lr_max_steps == scaled["lr_max_steps"]
        assert cfg.compile is False
    finally:
        for key in ("CROSSOVER_ARMS", "CROSSOVER_JOB_PREFIX", "CROSSOVER_BATCH",
                    "CROSSOVER_EVAL_ITERS", "CROSSOVER_TOKEN_BUDGET",
                    "CROSSOVER_LR_HORIZON"):
            os.environ.pop(key, None)


@test
def isolate_recipes_do_not_share_batch_or_horizon():
    import os
    from .crossover_replicate import (
        DRIFTED_ARMS, ISOLATE_STAGES, SUITE14_TOKEN_BUDGET, TOKEN_BUDGET,
        apply_isolate, expand_grid, job_batch, job_config, lock_recipe,
        scale_to_token_budget,
    )
    try:
        apply_isolate(ISOLATE_STAGES[0])
        jobs = expand_grid()
        assert len(jobs) == 10
        assert all(j["id"].startswith("cx20h_") for j in jobs)
        cfg = job_config(jobs[0], Path("/tmp"), smoke=False)
        assert cfg.batch_size == 32
        assert cfg.max_steps == 20_000_000 // (32 * 512)
        assert cfg.lr_max_steps == TOKEN_BUDGET // (32 * 512)
        assert cfg.eval_iters == 20

        apply_isolate(ISOLATE_STAGES[1])
        jobs = expand_grid()
        assert len(jobs) == 10
        assert all(j["id"].startswith("cx8_") for j in jobs)
        cfg = job_config(jobs[0], Path("/tmp"), smoke=False)
        assert cfg.batch_size == 8
        assert cfg.max_steps == 2000
        assert cfg.lr_max_steps == 2000
        assert job_batch({"arm": "gdn", "mixer": "gdn", "layer_mixers": ""}) == 8
        s = scale_to_token_budget(8, token_budget=SUITE14_TOKEN_BUDGET)
        assert s["max_steps"] == 2000

        apply_isolate(ISOLATE_STAGES[2])
        jobs = expand_grid()
        assert len(jobs) == len(DRIFTED_ARMS) * 5
        assert {j["arm"] for j in jobs} == set(DRIFTED_ARMS)
        assert "attention" not in {j["arm"] for j in jobs}
        cfg = job_config(jobs[0], Path("/tmp"), smoke=False)
        assert cfg.batch_size == 32
        assert cfg.eval_iters == 20
        assert cfg.max_steps == TOKEN_BUDGET // (32 * 512)
        assert cfg.lr_max_steps == cfg.max_steps

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_recipe(root)
            os.environ["CROSSOVER_BATCH"] = "8"
            try:
                lock_recipe(root)
                raise AssertionError("recipe mismatch must abort")
            except SystemExit:
                pass
    finally:
        for key in ("CROSSOVER_ARMS", "CROSSOVER_JOB_PREFIX", "CROSSOVER_BATCH",
                    "CROSSOVER_EVAL_ITERS", "CROSSOVER_TOKEN_BUDGET",
                    "CROSSOVER_LR_HORIZON"):
            os.environ.pop(key, None)


@test
def collect_discovers_disk_runs_and_keeps_mixed_batch_evals():
    import json
    from .crossover_replicate import _collect

    def write_run(root: Path, job_id: str, mixer: str, points: list[tuple[float, float]]):
        d = root / job_id
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({
            "mixer": mixer, "layer_mixers": "", "batch_size": 32,
            "grad_accum": 1, "block_size": 512,
        }), encoding="utf-8")
        lines = []
        for i, (tok, val) in enumerate(points):
            lines.append(json.dumps({
                "event": "eval", "step": i + 1, "tokens": tok, "val_loss": val,
            }))
        lines.append("{not json")
        (d / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dense = [(0.8e6, 6.8), (4.1e6, 5.9), (8.2e6, 5.4), (12.3e6, 5.1), (19.7e6, 4.7)]
        sparse = [(1.6e6, 6.5), (8.2e6, 5.3), (19.7e6, 4.9)]
        write_run(root, "cx20_attention_s1337", "attention", dense)
        write_run(root, "cx20_mingru_s1337", "mingru", sparse)
        write_run(root, "cx20_mingru_s42", "mingru",
                  [(0.8e6, 6.7), (4.1e6, 5.6), (8.2e6, 5.2), (12.3e6, 5.15), (19.7e6, 4.85)])
        summary = _collect(root)
        attn = summary["arms"]["attention"]
        gru = summary["arms"]["mingru"]
        assert attn["n"] == 1
        assert gru["n"] == 2
        assert len(attn["tokens"]) == 5, attn["tokens"]
        assert len(gru["mean"]) == 5, gru["tokens"]
        # mixed grids must not collapse to a single shared 8.2M point
        assert gru["mean"][0] != gru["mean"][-1]
        flips = summary["all_crossovers_vs_attention"]["mingru"]
        assert len(flips) >= 1


@test
def require_finite_fails_closed_on_nan_and_inf():
    from .train import require_finite
    require_finite(3, loss=1.0, gnorm=0.5)
    try:
        require_finite(55, loss=1.2, gnorm=float("nan"))
        raise AssertionError("NaN gnorm must raise")
    except RuntimeError as exc:
        assert "step 55" in str(exc) and "gnorm" in str(exc)
    try:
        require_finite(9, val_loss=float("inf"))
        raise AssertionError("inf val must raise")
    except RuntimeError as exc:
        assert "val_loss" in str(exc)


@test
def gdn_chunked_survives_odd_lengths_and_tiny_gates():
    from .mixers import gdn_chunked
    torch.manual_seed(0)
    for L, chunk in ((1, 32), (17, 8), (33, 32), (64, 32)):
        B, H, D = 2, 2, 8
        q = torch.randn(B, H, L, D)
        k = torch.nn.functional.normalize(torch.randn(B, H, L, D), dim=-1)
        v = torch.randn(B, H, L, D)
        alpha = torch.full((B, H, L), 1e-8)
        beta = torch.rand(B, H, L)
        y = gdn_chunked(q, k, v, alpha, beta, chunk=chunk)
        assert y.shape == (B, H, L, D)
        assert torch.isfinite(y).all(), (L, chunk)


@test
def cpu_batcher_windows_are_contiguous_and_shifted():
    import numpy as np
    from .data import Batcher, should_gpu_resident
    assert should_gpu_resident(10_000, "cpu") is False
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ids = np.arange(2000, dtype=np.uint16)
        ids.tofile(root / "train.bin")
        ids.tofile(root / "val.bin")
        cfg = _cfg(batch_size=4, block_size=8, seed=0)
        batcher = Batcher(root, "train", cfg, "cpu")
        x, y = batcher.batch()
        assert x.shape == (4, 8) and y.shape == (4, 8)
        assert torch.equal(y[:, :-1], x[:, 1:])


# ---------------------------------------------------------------------------
# E11 phase 2: tenancy, the wall-clock guard, and marker honesty.
# Each of these fails against the code that produced the 1.70x-spread board.
# ---------------------------------------------------------------------------
def _wc_suite(root, arms_elapsed, target_budget=1000, workers=None):
    """Write a fake wall-clock suite: {arm: [elapsed_s per seed]}."""
    import json
    for arm, elapsed in arms_elapsed.items():
        for i, el in enumerate(elapsed):
            d = root / f"cxwc_{arm}_s{1337 + i}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text(json.dumps(
                {"batch_size": 32, "block_size": 512, "mixer": arm}), encoding="utf-8")
            (d / "metrics.jsonl").write_text("\n".join([
                json.dumps({"event": "train", "step": 1, "tok_s": 1000.0,
                            "tokens": 1000}),
                json.dumps({"event": "done", "best_val": 4.0 + 0.01 * i,
                            "final_val": 4.0 + 0.01 * i + 0.02,
                            "tokens": target_budget, "elapsed_s": el,
                            "mean_tok_s": target_budget / el}),
            ]) + "\n", encoding="utf-8")
    if workers is not None:
        (root / "recipe.json").write_text(
            json.dumps({"workers": workers, "budget_by_arm": {a: target_budget
                                                              for a in arms_elapsed}}),
            encoding="utf-8")


@test
def effective_rate_is_below_step_rate_and_by_an_arm_specific_margin():
    """Wall-clock seconds include eval and startup; stepping seconds do not.

    If the overhead were a constant factor across arms, sizing from step rate
    would still match the clock. It is not: measured single-tenant, attention
    realises ~81% of its step rate over a whole run and gdn_bookend ~89%.
    """
    from .crossover_replicate import (
        effective_rate_by_arm, measured_rate_by_arm, WALLCLOCK_ARMS)
    eff = effective_rate_by_arm(tenancy=1)
    step = measured_rate_by_arm(tenancy=1)
    have = [a for a in WALLCLOCK_ARMS if a in eff and a in step]
    assert len(have) == len(WALLCLOCK_ARMS), f"missing single-tenant rates: {have}"
    realised = {a: eff[a] / step[a] for a in have}
    for a, frac in realised.items():
        assert 0.5 < frac < 1.0, f"{a}: effective/step = {frac:.3f}"
    assert max(realised.values()) - min(realised.values()) > 0.03, (
        "if overhead were uniform across arms this guard would be unnecessary; "
        f"spread was {realised}")


@test
def wallclock_budgets_land_on_the_target_using_effective_rates():
    from .crossover_replicate import (
        effective_rate_by_arm, wallclock_budgets, WALLCLOCK_ARMS,
        WALLCLOCK_SECONDS, WALLCLOCK_TOLERANCE)
    eff = effective_rate_by_arm(tenancy=1)
    budgets = wallclock_budgets(WALLCLOCK_SECONDS, WALLCLOCK_ARMS,
                                batch=32, block=512, tenancy=1)
    for arm, tok in budgets.items():
        predicted = tok / eff[arm]
        off = abs(predicted - WALLCLOCK_SECONDS) / WALLCLOCK_SECONDS
        assert off <= WALLCLOCK_TOLERANCE, (
            f"{arm}: predicted {predicted:.1f}s against {WALLCLOCK_SECONDS}s")


@test
def mfu_is_none_when_the_hardware_peak_is_unknown():
    """A GH200 normalised by a laptop's peak logged 'mfu 169.3%' as a measurement."""
    import os
    from .train import _mfu
    prev = os.environ.get("PEAK_FLOPS")
    try:
        os.environ.pop("PEAK_FLOPS", None)
        assert _mfu(954e6, 16384, 0.2, "cuda:0") is None, (
            "an unknown peak must read as unknown, not as a laptop's number")
        assert _mfu(954e6, 16384, 0.2, "cpu") == 0.0
        os.environ["PEAK_FLOPS"] = str(989.5e12)
        got = _mfu(954e6, 16384, 0.2, "cuda:0")
        assert 0.0 < got < 1.0, got
    finally:
        if prev is None:
            os.environ.pop("PEAK_FLOPS", None)
        else:
            os.environ["PEAK_FLOPS"] = prev


@test
def step_logger_records_an_unknown_mfu_without_crashing():
    import json
    from .utils import Logger
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        log = Logger(root, _cfg(run_name="probe"))
        log.step(1, loss=4.0, lr=1e-3, grad_norm=0.5, tok_s=1000.0,
                 mfu=None, tokens=16384)
        log.step(2, loss=3.9, lr=1e-3, grad_norm=0.5, tok_s=1000.0,
                 mfu=0.079, tokens=32768)
        rows = [json.loads(x) for x in
                (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rows[0]["mfu"] is None, "unmeasured must persist as null, not 0.0"
        assert abs(rows[1]["mfu"] - 0.079) < 1e-9


@test
def status_on_a_missing_suite_reports_nothing_not_a_default_grid():
    from .crossover_replicate import cmd_status
    with tempfile.TemporaryDirectory() as directory:
        missing = Path(directory) / "never_launched"
        try:
            cmd_status(argparse.Namespace(out=str(missing)))
        except SystemExit as e:
            assert "does not exist" in str(e), e
        else:
            raise AssertionError(
                "an unlaunched suite must not report a full queue of pending jobs")


@test
def stage_subcommands_have_no_default_tenancy_of_their_own():
    """argparse default=2 was indistinguishable from an explicit --workers 2."""
    from .crossover_replicate import build_parser
    parser = build_parser()
    ns = parser.parse_args(["wallclock32"])
    assert ns.workers is None, (
        f"unspecified tenancy must stay unspecified, got {ns.workers}")


@test
def wallclock_stage_defaults_to_its_own_tenancy_and_refuses_another():
    """The budgets are sized for one jobs-per-GPU; the launcher must not pick 2.

    The stage declared workers=1 while the launcher defaulted to 2 and honoured
    only an explicit flag, so the correct tenancy depended on the operator
    remembering to type it.
    """
    from . import crossover_replicate as cr
    stage = {x["name"]: x for x in cr.ISOLATE_STAGES}["wallclock32"]
    seen = {}
    real = cr._stage_launch
    cr._stage_launch = lambda args, st: seen.update(workers=args.workers)
    try:
        cr.cmd_wallclock32(argparse.Namespace(detach=True, workers=None))
        assert seen["workers"] == stage["workers"], seen
        try:
            cr.cmd_wallclock32(argparse.Namespace(detach=True, workers=3))
        except SystemExit as e:
            assert "wall-clock match" in str(e), e
        else:
            raise AssertionError("a mismatched tenancy must not launch")
    finally:
        cr._stage_launch = real


@test
def verify_wallclock_rejects_the_suite_that_missed_its_own_clock():
    from .crossover_replicate import verify_wallclock
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        # The real numbers from the first attempt: a 691s target hit at 387.8s
        # and 661.1s. That board was emitted without complaint.
        _wc_suite(root, {"attention": [387.8] * 5, "hybrid_gdn_bookend": [661.1] * 5})
        v = verify_wallclock(root, 691.0)
        assert v["ok"] is False, "a 1.70x spread must not pass as equal wall clock"
        assert abs(v["spread"] - 661.1 / 387.8) < 1e-6
        assert "43" in v["reason"] or "44" in v["reason"], v["reason"]


@test
def verify_wallclock_accepts_a_suite_that_actually_matched():
    from .crossover_replicate import verify_wallclock
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _wc_suite(root, {"attention": [688.0, 693.0], "hybrid_gdn_bookend": [700.0]})
        v = verify_wallclock(root, 691.0)
        assert v["ok"] is True, v["reason"]


@test
def verify_wallclock_fails_closed_when_no_run_records_a_duration():
    from .crossover_replicate import verify_wallclock
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "cxwc_attention_s1337").mkdir(parents=True)
        (root / "cxwc_attention_s1337" / "metrics.jsonl").write_text(
            '{"event": "done", "best_val": 4.0}\n', encoding="utf-8")
        v = verify_wallclock(root, 691.0)
        assert v["ok"] is False, "unmeasured must never read the same as verified"


@test
def wallclock_budgets_refuse_to_size_without_a_tenancy():
    from .crossover_replicate import wallclock_budgets
    try:
        wallclock_budgets(691.0, ("attention",), batch=32, block=512)
    except SystemExit as e:
        assert "tenancy" in str(e), e
    else:
        raise AssertionError(
            "sizing a budget from tenancy-blind rates is the defect that put "
            "attention 43.9% under its own wall-clock target")


@test
def wallclock_budgets_still_accept_explicit_rates():
    from .crossover_replicate import wallclock_budgets
    got = wallclock_budgets(100.0, ("attention",), batch=32, block=512,
                            rates={"attention": 10_000.0})
    tps = 32 * 512
    assert got["attention"] % tps == 0
    assert got["attention"] <= 1_000_000


@test
def measured_rate_by_arm_ignores_suites_run_at_another_tenancy():
    import json
    from .crossover_replicate import measured_rate_by_arm
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        for name, workers, tok_s in (("solo", 1, 100_000.0), ("packed", 3, 50_000.0)):
            suite = base / "nanolab" / "out" / name
            run = suite / f"cx_attention_s1337"
            run.mkdir(parents=True)
            (run / "config.json").write_text(
                json.dumps({"batch_size": 32, "block_size": 512}), encoding="utf-8")
            (run / "metrics.jsonl").write_text(json.dumps(
                {"event": "train", "tok_s": tok_s}) + "\n", encoding="utf-8")
            (suite / "recipe.json").write_text(
                json.dumps({"workers": workers}), encoding="utf-8")
        suites = ("nanolab/out/solo", "nanolab/out/packed")
        solo = measured_rate_by_arm(root=base, tenancy=1, suites=suites)
        packed = measured_rate_by_arm(root=base, tenancy=3, suites=suites)
        assert solo == {"attention": 100_000.0}, solo
        assert packed == {"attention": 50_000.0}, packed


@test
def suite_tenancy_prefers_the_record_then_falls_back_to_worker_logs():
    import json
    from .crossover_replicate import _suite_tenancy
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assert _suite_tenancy(root) is None, "unknown tenancy must not read as 1"
        (root / "worker_0.log").write_text("", encoding="utf-8")
        (root / "worker_1.log").write_text("", encoding="utf-8")
        assert _suite_tenancy(root) == 2
        (root / "recipe.json").write_text(json.dumps({"workers": 3}), encoding="utf-8")
        assert _suite_tenancy(root) == 3


@test
def lock_recipe_never_backfills_tenancy_from_the_current_launch():
    import json, os
    from .crossover_replicate import lock_recipe, current_recipe
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        legacy = {k: v for k, v in current_recipe().items() if k != "workers"}
        (root / "recipe.json").write_text(json.dumps(legacy), encoding="utf-8")
        # Suite plainly ran two-wide; this launch claims one.
        (root / "worker_0.log").write_text("", encoding="utf-8")
        (root / "worker_1.log").write_text("", encoding="utf-8")
        prev = os.environ.get("CROSSOVER_WORKERS")
        os.environ["CROSSOVER_WORKERS"] = "1"
        try:
            try:
                lock_recipe(root)
            except SystemExit as e:
                assert "tenanc" in str(e), e
            else:
                raise AssertionError(
                    "backfilling workers=1 would write a guess about a run that "
                    "already happened into its own record")
        finally:
            if prev is None:
                os.environ.pop("CROSSOVER_WORKERS", None)
            else:
                os.environ["CROSSOVER_WORKERS"] = prev


@test
def marker_window_is_one_eval_interval_not_a_fixed_fraction():
    from .crossover_replicate import _marker_window
    # ctx2048 steps 0.819M per eval and answers the 0.8M marker with 0.836M.
    grid = [0.836e6 + i * 0.819e6 for i in range(20)]
    assert abs(0.836e6 - 0.8e6) <= _marker_window(grid, 0.8e6)
    # An arm that stopped at 18.9M is not answering a 50M marker.
    short = [i * 0.819e6 for i in range(1, 24)]
    assert abs(max(short) - 50e6) > _marker_window(short, 50e6)


@test
def table_refuses_a_token_grid_for_a_wall_clock_suite():
    import json
    from .crossover_replicate import cmd_table
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _wc_suite(root, {"attention": [690.0]}, workers=1)
        try:
            cmd_table(argparse.Namespace(out=str(root)))
        except SystemExit as e:
            assert "wcboard" in str(e), e
        else:
            raise AssertionError(
                "arms that stop at different token counts cannot share a "
                "token-grid column")


@test
def wcboard_refuses_to_publish_a_board_whose_clock_did_not_match():
    from .crossover_replicate import cmd_wcboard
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _wc_suite(root, {"attention": [387.8] * 3, "hybrid_gdn_bookend": [661.1] * 3},
                  workers=1)
        args = argparse.Namespace(out=str(root), seconds=691.0,
                                  allow_unmatched=False)
        try:
            cmd_wcboard(args)
        except SystemExit as e:
            assert "REFUSING" in str(e), e
        else:
            raise AssertionError("an unmatched clock must not yield a board")
        # The diagnostic escape hatch still prints, and says why.
        cmd_wcboard(argparse.Namespace(out=str(root), seconds=691.0,
                                       allow_unmatched=True))


# ---------------------------------------------------------------------------
@test
def board_reads_the_end_of_schedule_loss_not_the_running_minimum():
    """best_val is a min over an arm-dependent number of evals.

    The fast arm takes more steps, so it draws more evals, so its minimum sits
    lower -- a bias pointing the same way as the throughput advantage the suite
    exists to measure. On the 2026-08-27 suite it was -0.0133 for attention
    (89 evals) against -0.0032 for hybrid_gdn_bookend (24).
    """
    import json
    from .crossover_replicate import _final_by_seed
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "cxwc_attention_s1337"
        run.mkdir(parents=True)
        (run / "metrics.jsonl").write_text(json.dumps(
            {"event": "done", "best_val": 4.0721, "final_val": 4.1401,
             "tokens": 1000, "elapsed_s": 690.0}) + "\n", encoding="utf-8")
        got = _final_by_seed(root)["attention"]["1337"]
        assert abs(got - 4.1401) < 1e-9, (
            f"board must read the end of the schedule, got {got}")


@test
def board_refuses_a_run_with_no_end_of_schedule_loss_to_recover():
    """Fail closed. Substituting best_val silently is the confounded board."""
    import json
    from .crossover_replicate import _final_by_seed
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "cxwc_attention_s1337"
        run.mkdir(parents=True)
        (run / "metrics.jsonl").write_text(json.dumps(
            {"event": "done", "best_val": 4.0721,
             "tokens": 1000, "elapsed_s": 690.0}) + "\n", encoding="utf-8")
        try:
            _final_by_seed(root)          # no final_val, no final.pt
        except SystemExit as e:
            assert "best_val" in str(e), e
        else:
            raise AssertionError(
                "a run with no recoverable final loss must not fall back to "
                "best_val")


@test
def done_record_carries_the_end_of_schedule_loss_alongside_the_minimum():
    import json
    from .utils import Logger
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        log = Logger(root, _cfg(run_name="probe"))
        log.done(4.0721, "11m", 73203712, elapsed_s=684.7, final_val=4.1401)
        rec = [json.loads(x) for x in
               (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()][-1]
        assert abs(rec["final_val"] - 4.1401) < 1e-9, rec
        assert abs(rec["best_val"] - 4.0721) < 1e-9, rec


@test
def the_two_peak_flops_constants_agree_and_exceed_what_was_measured():
    """494.7e12 vs 989.5e12 sat in the tree for months, 2x apart.

    A peak below an achieved rate is not a calibration question, it is
    arithmetic: the 8192^3 dense BF16 matmul measured on the GH200 sustains
    786.6 TFLOP/s, so any 'peak' under that is refuted outright.
    """
    import importlib.util
    from .crossover_replicate import GH200_PEAK_FLOPS, MEASURED_GH200_DENSE_BF16

    assert GH200_PEAK_FLOPS > MEASURED_GH200_DENSE_BF16, (
        f"peak {GH200_PEAK_FLOPS:.4g} is below the measured "
        f"{MEASURED_GH200_DENSE_BF16:.4g} achieved on the same device")
    frac = MEASURED_GH200_DENSE_BF16 / GH200_PEAK_FLOPS
    assert 0.55 <= frac <= 0.95, (
        f"a big dense matmul realising {frac:.1%} of peak means the peak is "
        f"wrong in one direction or the other")

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "gpu_bundle", root / "scripts" / "gpu_bundle.py")
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load scripts/gpu_bundle.py to cross-check")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    other = mod.DEVICE_PEAK_FLOPS["gh200"]
    assert abs(other - GH200_PEAK_FLOPS) < 1e9, (
        f"two peaks for one device: crossover_replicate says "
        f"{GH200_PEAK_FLOPS:.4g}, gpu_bundle says {other:.4g}")



@test
def mup_attention_temperature_is_the_arm_asymmetric_term():
    """muP's 1/d logit scale, without its companion q/k init, flattens attention.

    The rule is correct only when q.k grows as Theta(d). This tree keeps a fixed
    std=0.02 init, so q.k grows as Theta(sqrt(d)) and 1/d leaves the attention
    distribution at ~99.8% of uniform entropy at init (d_model=768, head_dim=64)
    against ~89% under SP. minGRU has no attention logits, so the handicap lands
    on one arm only -- which is why every e1_mup cell in the GPU bundle scored
    NOT A VALID COMPARATOR with attention hurt 4-7x more than minGRU, and why
    re-tuning the LR in e1_mup_tuned could not rescue it.

    Measured through a real forward pass: the attention input is a normed hidden
    state, not a raw Gaussian, and the effect does not reproduce on synthetic
    input.
    """
    import torch
    import torch.nn.functional as F
    from .config import Config
    from .model import GPT

    real_sdpa = F.scaled_dot_product_attention
    seen: list[float] = []

    def spy(q, k, v, *a, **kw):
        scale = kw.get("scale")
        if scale is not None:
            logits = (q.float() @ k.float().transpose(-2, -1)) * scale
            t = logits.shape[-1]
            causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=logits.device))
            pr = logits.masked_fill(~causal, float("-inf")).softmax(-1)
            ent = -(pr * torch.log(pr.clamp_min(1e-12))).sum(-1).mean()
            uni = torch.log(torch.arange(1, t + 1, dtype=torch.float32,
                                         device=logits.device)).mean()
            seen.append(float(ent / uni))
        return real_sdpa(q, k, v, *a, **kw)

    def entropy_frac(mup, sqrt_attn):
        seen.clear()
        cfg = Config(mixer="attention", d_model=768, n_head=12, head_dim=64,
                     n_layer=4, block_size=128, vocab_size=1024, mup=mup,
                     mup_base_width=256, mup_sqrt_attn_scale=sqrt_attn,
                     zero_init_proj=False)
        torch.manual_seed(0)
        model = GPT(cfg).eval()
        tokens = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        F.scaled_dot_product_attention = spy
        try:
            with torch.no_grad():
                model(tokens)
        finally:
            F.scaled_dot_product_attention = real_sdpa
        assert seen, "the spy never fired; attention did not go through SDPA"
        return sum(seen) / len(seen)

    sp = entropy_frac(False, False)
    mup = entropy_frac(True, False)
    ablated = entropy_frac(True, True)

    assert mup > 0.99, (
        f"muP attention should sit at ~uniform entropy at init, got {mup:.3f}")
    assert sp < 0.95, f"SP attention should not be near-uniform, got {sp:.3f}"
    assert abs(ablated - sp) < 0.01, (
        f"ablating the 1/d scale must restore SP's temperature: "
        f"{ablated:.3f} vs {sp:.3f}")



def _mqar(**over):
    from .config import Config
    from .mqar import vocab_for
    d = dict(n_pairs=4, n_queries=2, n_keys=8, n_values=8)
    d.update({k: v for k, v in over.items() if k in d})
    cfg = Config(mixer="attention", batch_size=4, seed=1337,
                 mqar_n_pairs=d["n_pairs"], mqar_n_queries=d["n_queries"],
                 mqar_n_keys=d["n_keys"], mqar_n_values=d["n_values"],
                 block_size=over.get("block_size",
                                     2 * (d["n_pairs"] + d["n_queries"]) - 1),
                 vocab_size=vocab_for(d["n_keys"], d["n_values"]))
    return cfg


@test
def mqar_supervises_only_the_query_positions():
    """The training loss IS recall loss: everything else is ignore_index."""
    from .mqar import MQARBatcher, IGNORE
    b = MQARBatcher(_mqar(), "cpu")
    x, y = b.batch()
    p, q = b.n_pairs, b.n_queries
    supervised = (y != IGNORE)
    assert int(supervised.sum()) == x.shape[0] * q, (
        f"expected {q} supervised positions per row, got "
        f"{int(supervised.sum()) / x.shape[0]}")
    # they must be the even offsets inside the query block, nowhere else
    want = torch.zeros_like(supervised)
    want[:, torch.arange(2 * p, b.seq_len - 1, 2)] = True
    assert torch.equal(supervised, want), "supervision landed off the query positions"


@test
def mqar_is_solvable_by_lookup_and_its_labels_agree_with_its_context():
    """An exact-match oracle scores 1.0.

    This is the difference between a hard task and a broken one: if the answer
    were not recoverable from the context, a recall probe would measure noise
    and every arm would look equally bad.
    """
    from .mqar import MQARBatcher, IGNORE
    b = MQARBatcher(_mqar(n_pairs=6, n_queries=3, n_keys=12, n_values=12), "cpu")
    x, y = b.batch()
    p = b.n_pairs
    hits = seen = 0
    for row in range(x.shape[0]):
        table = {int(x[row, 2 * i]): int(x[row, 2 * i + 1]) for i in range(p)}
        assert len(table) == p, "keys are not unique; a query has two right answers"
        for t in range(x.shape[1]):
            if y[row, t] == IGNORE:
                continue
            seen += 1
            hits += table[int(x[row, t])] == int(y[row, t])
    assert seen and hits == seen, f"oracle scored {hits}/{seen}, so the labels "\
                                  f"do not follow from the context"


@test
def mqar_refuses_a_task_that_has_no_right_answer():
    from .mqar import MQARBatcher
    for over, why in ((dict(n_queries=9, n_pairs=4), "no stored pair"),
                      (dict(n_pairs=9, n_keys=8), "two right answers")):
        cfg = _mqar(**over)
        try:
            MQARBatcher(cfg, "cpu")
        except ValueError:
            pass
        else:
            raise AssertionError(f"should refuse: {why}")


@test
def mqar_refuses_a_block_size_the_task_cannot_fill():
    """Silently padding would train the arms on a different task than reported."""
    from .mqar import MQARBatcher
    try:
        MQARBatcher(_mqar(block_size=512), "cpu")
    except ValueError as e:
        assert "block_size" in str(e), e
    else:
        raise AssertionError("a mismatched block_size must not be padded around")


@test
def mqar_recall_accuracy_separates_an_oracle_from_a_guesser():
    """CE cannot stand in for this: the metric is exact match at the query."""
    import contextlib
    from .mqar import MQARBatcher, recall_accuracy, IGNORE

    b = MQARBatcher(_mqar(n_pairs=6, n_queries=3, n_keys=12, n_values=12), "cpu")
    V = b.vocab_size

    class Oracle(torch.nn.Module):
        """Reads the pair table out of the context. Scores 1.0 by construction."""
        training = False
        def eval(self): return self
        def train(self, mode=True): return self
        def forward(self, x, y=None):
            out = torch.zeros(*x.shape, V)
            for r in range(x.shape[0]):
                tab = {int(x[r, 2 * i]): int(x[r, 2 * i + 1])
                       for i in range(b.n_pairs)}
                for t in range(x.shape[1]):
                    out[r, t, tab.get(int(x[r, t]), 0)] = 1.0
            return out, None

    class Constant(torch.nn.Module):
        """Always answers with one value -- the strongest key-blind strategy."""
        training = False
        def eval(self): return self
        def train(self, mode=True): return self
        def forward(self, x, y=None):
            out = torch.zeros(*x.shape, V)
            out[..., 1 + b.n_keys] = 1.0
            return out, None

    acc = recall_accuracy(Oracle(), b, contextlib.nullcontext(), iters=3)
    assert acc == 1.0, f"oracle must score 1.0, got {acc}"
    blind = recall_accuracy(Constant(), b, contextlib.nullcontext(), iters=8)
    assert blind < 0.35, (
        f"a key-blind constant scored {blind:.3f}; with {b.n_values} values "
        f"chance is {1 / b.n_values:.3f} and the task is not testing recall")



@test
def e8_reports_a_rate_because_the_outcomes_are_bimodal():
    """Mean recall over a bimodal sample describes no model that exists.

    Identical config, init alone varying: 0.553 / 0.542 / 0.957. The head forms
    or it does not. The board must report how OFTEN it forms.
    """
    from .mqar_suite import board, SOLVED
    recs = [{"run": f"mqar_attention_s{i}", "recall": r, "solved": r >= SOLVED}
            for i, r in enumerate([0.95, 0.96, 0.54, 0.97, 0.55])]
    recs += [{"run": f"mqar_mingru_s{i}", "recall": r, "solved": r >= SOLVED}
             for i, r in enumerate([0.55, 0.54, 0.56, 0.55, 0.54])]
    rows = {r["arm"]: r for r in board(recs)}
    assert rows["attention"]["solved"] == 3 and rows["attention"]["n"] == 5, rows
    assert rows["mingru"]["solved"] == 0, rows
    # the means differ by only ~0.2 while the rates differ by 0.6
    assert rows["attention"]["rate"] == 0.6 and rows["mingru"]["rate"] == 0.0


@test
def e8_wilson_interval_is_binomial_and_never_leaves_zero_one():
    from .mqar_suite import _wilson
    lo, hi = _wilson(0, 15)
    assert lo == 0.0 and 0.0 < hi < 0.30, (lo, hi)
    lo, hi = _wilson(15, 15)
    assert hi == 1.0 and 0.70 < lo < 1.0, (lo, hi)
    lo, hi = _wilson(8, 15)
    assert 0.0 < lo < 0.53 < hi < 1.0, (lo, hi)


@test
def e8_runs_untied_because_tying_caps_the_reference_arm():
    """Not a tuning choice: tied, attention itself caps near 0.55."""
    from .mqar_suite import e8_config
    cfg = e8_config("attention", 1)
    assert cfg.tie_embeddings is False, "E8 must untie; see nanolab/mqar.py"
    assert cfg.fused_ce is False, "recall_accuracy cannot score the fused path"
    assert cfg.block_size == 2 * (cfg.mqar_n_pairs + cfg.mqar_n_queries) - 1


@test
def e8_carries_the_layer_pattern_of_each_hybrid_family():
    from .mqar_suite import e8_config
    hyb = e8_config("hybrid_mingru10_attn2", 1)
    assert hyb.layer_mixers == "mingru*10,attention*2", hyb.layer_mixers
    assert e8_config("attention", 1).layer_mixers == ""



@test
def e8_batch_is_calibrated_not_defaulted():
    """At batch 32 the reference arm never forms the head; at 256 it always does.

    Same depth, same 3000 steps, same wall clock to within 10% -- the sequence is
    15 tokens, so a small batch spends the step on kernel launches. Measured on
    the GH200 2026-08-28: 0.552/0.550 at bs=32 against 1.000/1.000 at bs=256
    (L=4), and 0.553/0.708 against 0.999/1.000 (L=12). The probe's answer to
    "can this architecture recall" is decided by batch before any architecture is
    compared, so the default must not drift back to the corpus suites' 32.
    """
    from .mqar_suite import e8_config
    assert e8_config("attention", 1).batch_size == 256
    assert e8_config("attention", 1).max_steps == 3000
    assert e8_config("attention", 1, batch_size=32).batch_size == 32



@test
def e8_run_name_carries_the_recipe_so_resume_cannot_mix_configs():
    """A name that does not carry the recipe is not an identifier.

    The first sweep skipped an attention seed as "done, recall 0.542" that had
    been trained at batch 32 for 6000 steps, into a board being built at batch
    256 for 3000 -- and batch is precisely the variable that decides whether the
    head forms. Resume keys on the run name, so the name must differ whenever
    the recipe does.
    """
    from .mqar_suite import e8_config
    a = e8_config("attention", 1)
    b = e8_config("attention", 1, batch_size=32)
    c = e8_config("attention", 1, steps=6000)
    d = e8_config("attention", 1, n_pairs=6, n_queries=6)
    assert len({a.run_name, b.run_name, c.run_name, d.run_name}) == 4, (
        a.run_name, b.run_name, c.run_name, d.run_name)
    assert e8_config("attention", 1).run_name == a.run_name, "must be stable"


@test
def launch_records_the_tenancy_it_actually_runs_at():
    """`launch --workers N` must write N into recipe.json, not the env default.

    `cluster_workers()` reads CROSSOVER_WORKERS; `cmd_launch` spawns
    `args.workers`. Those were two different sources, so a suite launched with
    --workers 3 recorded `workers: 1` while running three jobs to a GPU. Every
    downstream rate model trusts that field -- `_suite_tenancy`, wall-clock budget
    sizing, and `scripts/gpu_bundle.py`'s cost basis -- and a rate measured at one
    tenancy does not transfer to another. Caught in the release audit, on a suite
    whose record had already been published.

    `subprocess.Popen` is stubbed: the assertion is about what `lock_recipe`
    writes, and an unstubbed run spawns real training workers into a temp dir.
    """
    import argparse, os, tempfile, json as _json, subprocess
    from . import crossover_replicate as cr

    class _NoProc:
        def __init__(self, *a, **k): self.pid = -1
        def wait(self, *a, **k): return 0
        def poll(self): return 0
        def terminate(self): pass
        def kill(self): pass

    prior_env = os.environ.get("CROSSOVER_WORKERS")
    prior_batch = os.environ.get("CROSSOVER_BATCH")
    real_popen = subprocess.Popen
    try:
        os.environ.pop("CROSSOVER_WORKERS", None)
        # Batch 32, as the suite that exposed this ran at. The default 96 trips
        # the >=64 VRAM clamp, which correctly forces workers to 1 -- so a test
        # left on the default asserts against the clamp, not against the bug.
        os.environ["CROSSOVER_BATCH"] = "32"
        subprocess.Popen = _NoProc
        with tempfile.TemporaryDirectory() as d:
            args = argparse.Namespace(out=d, workers=3, arm=None, seed=None,
                                      detach=True, unhold=False)
            try:
                cr.cmd_launch(args)
            except (Exception, SystemExit):
                pass          # the assertion is about recipe.json, not the run
            rec = os.path.join(d, "recipe.json")
            assert os.path.exists(rec), "launch wrote no recipe.json"
            got = _json.loads(open(rec).read()).get("workers")
            assert got == 3, f"recipe records workers={got}, launch ran 3"
    finally:
        subprocess.Popen = real_popen
        for k, v in (("CROSSOVER_WORKERS", prior_env), ("CROSSOVER_BATCH", prior_batch)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main():
    torch.set_num_threads(2)
    passed = failed = 0
    for fn in _RESULTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except (Exception, SystemExit) as e:
            # SystemExit is BaseException, so an uncaught one used to kill the
            # runner mid-list: no FAIL line, no summary, exit 1 -- indistinguish-
            # able at a glance from a clean run whose output scrolled. Code under
            # test raises SystemExit deliberately, so it is a failure, not an exit.
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
