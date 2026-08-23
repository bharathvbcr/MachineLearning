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

import sys
import tempfile
import json
import math
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
from .crossover_replicate import load_run_timing, timing_summary
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
def main():
    torch.set_num_threads(2)
    passed = failed = 0
    for fn in _RESULTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
