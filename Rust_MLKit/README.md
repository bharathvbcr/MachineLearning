# Rust_MLKit — Mac ML Kit Core-Level Architectures for Rust Port

Extracted from the [Parameter Golf](../README.md) experimentation workspace.
These are the three architectures selected for a Rust rewrite targeting Apple's
Core ML / Metal Performance Shaders stack.

---

## Architectures

| # | Name | BPB | Source | Status |
|---|---|---|---|---|
| 01 | **Gated Attention + Value Residual** | **1.985** | `train_gpt_sprint_native.py` | ★ Champion |
| 02 | **Value Residual Only** | 1.987 | `train_gpt_sprint_native.py` | Runner-up (simplest) |
| 03 | **APRDH Adaptive Raw-Byte** | experimental | `train_toy_adaptive.py` | Most promising |

### Selection Rationale

- **arch_01** is the ablation champion — highest quality at competition scale.
- **arch_02** is within seed noise of arch_01 but removes the gating module,
  making it the **simplest to port** while retaining nearly identical quality.
  **Recommended starting point for the Rust port.**
- **arch_03** is the most architecturally novel — byte-level input, linear-time
  recurrence, adaptive compute. It trades BPB optimization for properties that
  are uniquely valuable on Apple Silicon: no tokenizer, O(1) memory inference,
  and workload-proportional compute.

---

## Directory Layout

```
Rust_MLKit/
├── README.md                                ← you are here
├── arch_01_gated_value_resid/
│   ├── README.md                            ← architecture spec + porting notes
│   ├── train_gpt_sprint_native.py           ← full source (run with GATED_ATTENTION=1 VALUE_RESIDUAL=1)
│   └── submission_packaging.py              ← single-file builder + SDPA fallback prelude
├── arch_02_value_resid/
│   ├── README.md                            ← architecture spec + porting notes
│   └── train_gpt_sprint_native.py           ← same source (run with VALUE_RESIDUAL=1 GATED_ATTENTION=0)
├── arch_03_aprdh_adaptive/
│   ├── README.md                            ← architecture spec + porting notes
│   └── train_toy_adaptive.py                ← byte-level adaptive recurrent architecture
└── reference/
    ├── ablation_results/
    │   ├── champion_arch_ladder.json         ← final champion selection
    │   ├── champion_followup.json            ← follow-up ablation champion
    │   ├── arch_ladder_summary.txt           ← full short→mid→long ranking
    │   ├── arch_ladder_long_results.json     ← per-seed long-stage raw data
    │   ├── followup_summary.txt              ← value-resid variant follow-up
    │   └── ablation_suites.json              ← all 8 ablation suite definitions
    └── verification/
        ├── verify_scan.py                    ← Mamba-2 SSD correctness tests
        ├── verify_gdn.py                     ← Gated DeltaNet correctness tests
        └── verify_gdn_wy.py                  ← Delta-rule WY representation tests
```

---

## Porting Roadmap (suggested)

### Phase 1 — Core Inference (arch_02, simplest path)
1. Port `RMSNorm`, `Rotary (RoPE)`, `apply_rotary_emb` → Metal compute shaders
2. Port `CausalSelfAttention` (GQA + QK-Norm + Value Residual) → MPS or custom Metal
3. Port `MLP` (squared leaky-ReLU) → fused Metal kernel
4. Port `Block` (residual mix + attn/mlp scaling) → host-side orchestration
5. Port `GPT._forward_baseline_backbone` → the full forward pass
6. Implement int6/int8 weight dequantization for inference-from-checkpoint

### Phase 2 — Auxiliary Modules
7. Port `SmearGate`, `BigramHashEmbedding`, `ValueEmbedding`
8. Port Cross-Self-Attention (XSA) subtraction
9. Port `Gated Attention` module (upgrade from arch_02 to arch_01)

### Phase 3 — Training Loop (optional)
10. Port `Muon` optimizer with batched Newton-Schulz (NS5)
11. Port `CastedLinear` with QAT STE bypass
12. Port sliding-window evaluation and BPB scoring

### Phase 4 — APRDH (arch_03, research)
13. Port Gated DeltaNet chunked scan → Metal compute shader
14. Port span-mixer patching and engram hash memory
15. Port Gumbel-routed compute controller

---

## Key Numeric Constraints

- **FP32 accumulation** is required inside all recurrence loops (Mamba-2 SSD,
  GDN delta-rule). BF16/FP16 will silently produce wrong results.
- **Newton-Schulz (NS5)** runs in BF16 and is numerically stable at 5 iterations
  with coefficients `a=3.4445, b=-4.7750, c=2.0315`.
- **int6 range** is `[-31, 31]` (6-bit signed), NOT `[-32, 31]`.
- Verification scripts in `reference/verification/` test to **1e-5 tolerance** —
  port these first and use them as your Rust test suite.

---

## Provenance

All architecture code is original work by Bharath Vaddaram, built on top of the
[openai/parameter-golf](https://github.com/openai/parameter-golf) competition
repository. The upstream baseline (`train_gpt.py`), data pipeline, and leaderboard
submissions remain the property of their original authors.
