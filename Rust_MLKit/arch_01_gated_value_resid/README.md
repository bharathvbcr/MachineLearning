# Arch 01 — Gated Attention + Value Residual (CHAMPION)

**Rank:** #1 — Best architecture from staged ablation  
**Calibrated BPB:** 1.9847 (3,000 steps, seeds 1337+42)  
**Artifact size:** ~1.34 MB (int6 quantized)  
**Source:** `train_gpt_sprint_native.py` with `GATED_ATTENTION=1 VALUE_RESIDUAL=1`

---

## Configuration

```
NUM_LAYERS       = 11
MODEL_DIM        = 512
NUM_HEADS        = 8
NUM_KV_HEADS     = 4        # GQA ratio 2:1
MLP_MULT         = 3.0      # hidden_dim = 1536
VOCAB_SIZE       = 1024     # SentencePiece BPE
TRAIN_SEQ_LEN    = 2048
ROPE_DIMS        = 16       # partial RoPE
BIGRAM_DIM       = 48       # auxiliary bigram hash embedding
VE_DIM           = 24       # value embedding dim
VE_LAYERS        = 9,10     # inject value embeddings at layers 9-10
XSA_MODE         = paper    # cross-self-attention on last 4 layers
XSA_LAST_N       = 4
GATED_ATTENTION  = 1        # ★ champion toggle
VALUE_RESIDUAL   = 1        # ★ champion toggle
```

---

## Key Modules to Port to Rust

### 1. Gated Attention (`CausalSelfAttention`)
```
gate = σ(W_gate · x + b_gate)          # shape: (B, T, num_heads)
y = y * gate.unsqueeze(-1)             # per-head gating after attention
```
- `W_gate`: Linear(dim → num_heads), zero-init weights, bias init = 4.0
- Applied after FlashAttention / SDPA, before output projection

### 2. Value Residual (v0 forwarding)
```
# First block stores raw_v as v0
# All subsequent blocks mix:
v_l = λ₀ · v0 + λ₁ · v_l_raw
```
- `vr_lambda`: nn.Parameter([0.5, 0.5]) — learned per-layer

### 3. Cross-Self-Attention (XSA) — last 4 layers
```
# Subtract self-value projection (GQA-aware reshape):
y_xsa = y - proj_V(y)     # forces external context modeling
```

### 4. SmearGate (temporal smoothing)
```
g = σ(gate_param)
x_out = (1 - g) · x_current + g · x_previous_token
```

### 5. Squared Leaky ReLU MLP
```
u = W_up · x
h = leaky_relu(u, slope=0.5)²
output = W_down · h
```

### 6. Parameter Banking (batched Muon optimization)
All Q/K/V/O/MLP weights are stored as 3D bank tensors:
- `qo_bank`: shape (2·L, d_model, d_model)
- `kv_bank`: shape (2·L, kv_dim, d_model)
- `mlp_up_bank`, `mlp_down_bank`: shape (L, hidden_dim, d_model)

This enables batched Newton-Schulz orthogonalization in the Muon optimizer.

### 7. Quantization: Per-row int6
```
for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
    clip row-wise → scale = clip / 31
    q = clamp(round(w / scale), -31, 31)
    pick pct with lowest MSE
```
- MLP + attention weights → int6
- Everything else → int8
- Small tensors (≤65536 elements) → fp16 passthrough

---

## Rust/Metal Porting Notes

- **RoPE**: Partial rotary (16 dims), base 10000 — pure arithmetic, no external deps
- **RMSNorm**: `x / sqrt(mean(x²) + eps)` — single Metal kernel
- **FlashAttention**: Map to Metal Performance Shaders or custom compute shader
- **GQA**: `repeat_interleave` for K/V heads can be fused into attention kernel
- **QK-Norm**: RMSNorm on Q and K before RoPE — stabilizes attention logits
- **Weight banks**: Natural fit for Metal buffer arrays indexed by layer
