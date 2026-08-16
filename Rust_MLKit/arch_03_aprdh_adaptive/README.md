# Arch 03 — APRDH: Adaptive Raw-Byte Recurrent (Most Promising Experimental)

**Rank:** Experimental — most architecturally novel  
**Source:** `train_toy_adaptive.py`  
**Status:** Research prototype; not ablation-ranked against the sprint architectures  

---

## Why This Is "Most Promising"

The arch_01/02 architectures are optimized dense transformers. APRDH explores a
fundamentally different design space:

- **Byte-level** (no tokenizer dependency — operates directly on UTF-8 bytes)
- **Single shared block** reused recurrently 2–5 passes (depth via reuse, not params)
- **Adaptive compute** — a learned controller decides how many passes per token
- **Gated DeltaNet** (linear-time recurrence) instead of quadratic attention

This matters for a Rust/Metal port because:
1. **No tokenizer shipping** — the model reads raw bytes
2. **Constant memory** — linear recurrence, no KV cache growth
3. **Adaptive compute** — on Apple Silicon, this means the model can spend fewer
   cycles on easy tokens (whitespace, common words) and more on hard ones

---

## Architecture Components

### 1. Projected Gated DeltaNet (core mixer)
A linear-time recurrence with an exact chunked dual-scan implemented as a custom
`torch.autograd.Function`:
```
# Delta rule: S_t = S_{t-1} + β_t · (v_t ⊗ k_t - S_{t-1} · (k_t ⊗ k_t))
# Gated:     h_t = α_t · h_{t-1} + β_t · v_t · k_t^T
# Chunked for GPU parallelism, FP32 accumulation for numerical stability
```

### 2. DeepSeek-style MLA (Multi-head Latent Attention)
```
# Compress K/V into low-rank latent:
c = W_dkv · x                          # (B, T, latent_dim)
k = W_uk · c                           # decompress to keys
v = W_uv · c                           # decompress to values
# Top-k routing: only use k most informative latent dimensions
```

### 3. Span-Mixer Patching
Groups raw bytes into learned "patches" (variable-length spans):
```
# boundary_logits = f(byte_features)
# patches = group_by_boundaries(bytes, boundary_logits)
# Operate on patches (compressed sequence) then scatter back
```

### 4. N-gram Hash "Engram" Memory
A tiny external lookup table keyed by byte n-grams:
```
hash = polynomial_hash(last_n_bytes) % table_size
memory_vector = engram_table[hash]
x = x + memory_vector    # cheap external context
```

### 5. Marginal-Gain Compute Controller
Gumbel-routed adaptive computation:
```
# For each token, predict: "would another pass improve output?"
gain_logit = W_ctrl · hidden_state
route = gumbel_softmax(gain_logit)      # soft → hard annealing
if route == CONTINUE: run another pass
if route == STOP:     emit output
# Budget penalty prevents always choosing max passes
```

### 6. Fast-Weight Adapters
Per-pass weight modulation:
```
# Each recurrent pass gets a pass-specific adapter:
W_adapted = W_shared + pass_embedding · W_adapter
# Enables specialization across passes without duplicating full weights
```

---

## Training Hardening

- **Gradient spike skip/rollback**: if grad_norm > threshold, skip the update
- **Soft→hard Gumbel**: tau anneals from 1.0 → 0.1 over training
- **Tau floors**: prevent Gumbel from collapsing to deterministic too early

---

## Rust/Metal Porting Notes

### Advantages for Apple Silicon
- **No KV cache**: linear recurrence = O(1) memory per token at inference
- **No tokenizer**: reads raw UTF-8 bytes — simpler deployment
- **Adaptive compute**: Neural Engine / GPU can early-exit on easy tokens
- **Small footprint**: weight-shared block means far fewer unique parameters

### Challenges
- **Custom autograd Function**: The chunked dual-scan GDN kernel needs a custom
  Metal compute shader (reference: `verify_gdn.py`, `verify_gdn_wy.py`)
- **FP32 accumulation**: The recurrence MUST use FP32 inside the scan loop
  (the project discovered this bug empirically — see learning-notes/08)
- **Gumbel routing**: Stochastic during training, argmax at inference — the
  inference path is simpler
- **Span boundaries**: Need efficient byte-level boundary detection on GPU

### Verification
The `reference/verification/` folder contains:
- `verify_scan.py` — Mamba-2 SSD chunk-parallel scan vs sequential reference
- `verify_gdn.py` — Gated DeltaNet recurrence vs brute-force reference
- `verify_gdn_wy.py` — Delta-rule chunkwise (WY representation) benchmark

All verify to 1e-5 tolerance. **Port these first** — if the Rust recurrence
doesn't match, nothing downstream will work.
