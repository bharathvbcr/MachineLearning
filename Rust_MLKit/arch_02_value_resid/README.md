# Arch 02 — Value Residual Only (Runner-Up)

**Rank:** #2 — Within noise of champion  
**Calibrated BPB:** 1.9875 (3,000 steps, seeds 1337+42)  
**Delta from champion:** +0.0027 BPB (statistically insignificant)  
**Source:** `train_gpt_sprint_native.py` with `VALUE_RESIDUAL=1 GATED_ATTENTION=0`

---

## Configuration

Identical to arch_01 except:
```
GATED_ATTENTION  = 0    # ← OFF (no per-head gating)
VALUE_RESIDUAL   = 1    # ← ON  (v0 forwarding through depth)
```

---

## Why This Is the Second Pick

| Variant | Cal. BPB | Δ from champion |
|---|---|---|
| gated_value_resid (arch_01) | 1.9847 | — |
| **value_resid (arch_02)** | **1.9875** | **+0.0027** |
| gated_attn only | 2.0887 | +0.1040 |
| control (neither) | 2.2231 | +0.2384 |

The gap between arch_01 and arch_02 (0.003 BPB) is **within seed variance**.
The gap between arch_02 and gated-only (0.101 BPB) confirms that **value residual
is the dominant architectural contribution** — gating alone hurts.

Per-seed detail (from `long.results.json`):
- seed 1337: value_resid = 1.9918, gated_value_resid = 1.9806 (Δ 0.011)
- seed 42:   value_resid = 1.9832, gated_value_resid = 1.9889 (Δ −0.006 — value_resid wins!)

---

## Key Module: Value Residual

The single mechanism that separates this from the baseline:

```python
# In CausalSelfAttention.forward():
#   v0 is cached from the FIRST block's raw value projection
#   All subsequent blocks mix v0 into their local values:

if self.value_residual and v0 is not None:
    lam = self.vr_lambda.to(dtype=v.dtype)  # learned [0.5, 0.5]
    v = lam[0] * v0 + lam[1] * v
```

### Why it works

Value residual prevents **representational drift** across depth. In a standard
transformer, each layer computes fresh V projections — by the time the model is
11 layers deep, the value space may have shifted significantly from the original
token semantics. Mixing in v0 provides a **direct gradient highway** from the
output back to the first layer's value projection, improving both convergence
speed and final quality.

---

## Rust/Metal Porting Notes

This is the **simpler** of the two top architectures — it removes the gating
module entirely, making the attention path:

```
Q, K, V = project(x)
if layer > 0: V = λ₀·V₀ + λ₁·V
Q, K = rms_norm(Q), rms_norm(K)
Q, K = apply_rope(Q, K)
Q = Q * per_head_gain
Y = flash_attn(Q, K, V, causal=True)
if xsa_enabled: Y = Y - proj_V(Y)
output = W_out · Y
```

No gating sigmoid, no extra linear layer — fewer ops per token, simpler Metal
kernel, and nearly identical quality. **This is the recommended starting point
for a Rust port** due to its simplicity.
