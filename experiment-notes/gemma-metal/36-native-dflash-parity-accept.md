# 36: Native DFlash parity / accept gates

## Executive summary

- **Question:** Porting MLX DFlash block-verify into gemma-metal (GPU draft + M>1 verify) would match greedy exactness and approach MLX mean_accept≈3 @ bs=5 while clearing product ≥15 / ≥25.
- **Result:** Mini steered gates are green; honest 31B native DFlash still far from MLX accept≈3 and ≥15 tok/s — treat latest exact=true with mode-locked streams as partial, not ship clearance.
- **Implication:** Do not promote this beyond the completed stages; the missing or failed confirmation is decision-relevant.
- **Status:** `partial`; evidence confidence **Low**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `36-native-dflash-parity-accept` |
| Dates | `2026-07-14` – `2026-07-14` |
| Hardware | Apple M5 Pro · gemma-metal Hot 31B + DFlash draft |
| Status | `partial` |

## Hypothesis

Porting MLX DFlash block-verify into gemma-metal (GPU draft + M>1 verify) would match greedy exactness and approach MLX mean_accept≈3 @ bs=5 while clearing product ≥15 / ≥25.

## Setup

- Trainer / preset: `bench --dflash-31b` / parity gate harness writing `latest_dflash_parity_gates.json`
- Fixed knobs: hazard skip-auto on product 31B; always-on + MASK-steer **synthetic mini only**; draft FA `1/√d`, target FA `1.0`, embed `√H`
- Env flags: `GEMMA_METAL_31B_VERIFY_DIAG=1` for GEMM≡seq checks; dual act-scratch ×`VERIFY_MAX_M`

## Variants

| Variant | Change |
|---------|--------|
| Synthetic mini | Steered draft; M×GEMV verify (H=256) |
| Real 31B Hot | Q4 GEMM + FA(Tq=M) when cols>256 |
| MLX golden reference | exact verify PASS, mean_accept≈3.0 |

## Results

| Rank | Run / config | Metric | Notes |
|------|--------------|--------|-------|
| Mini (latest) | DFlash best **764.8** tok/s | ≥ hazard greedy **739.4**; exact **PASS**; mean_accept@exact **2.0** | Steered — not HF-ready |
| Real 31B (latest gates) | best DFlash **1.91** @ bs=3 | greedy **4.57**; sweep mean_accept **0.77** (bs=3) / **1.0** (bs=5) | `exact_vs_greedy: true` but streams mode-lock (unique small) |
| Honest pre dual-norm | DFlash **~1.17** @ bs=5 | mean_accept **≈0**; exact **FAIL** | `run_dflash_parity_gates_1784001103.json` |
| MLX reference | **~28–37** tok/s | mean_accept **≈3.0**; exact **PASS** | Product bar |
| Product gates | ≥15 / ≥25 | **false** / **false** | in `latest_dflash_parity_gates.json` |

Port progress (`dflash_port.md`): GEMM dual-norm + `layer_scalar` ×M **landing/landed**; device capture; conditioner Q8 g64. Remaining: drafts still ≪ MLX proposals (anchor **929≠531** on short ids); DFlash still ~2× slower than greedy under AO-while-capture tax.

**Interpretation boundary.** Synthetic throughput and steered acceptance are not product evidence. Real 31B acceptance, stream diversity, exactness, and tok/s must all pass together before promotion.

## Failures

- Prior mean_accept≈3.8 claim invalid (NaN-collapsed target) — discard `run_dflash_parity_gates_final.json`
- Vacuous AO exact PASS when unique≤4 — do not claim product readiness
- Hazard exactness historically FAIL near-ties; mid-verify host sync under hazard not used for honest score
- Product 31B decode ≥15 / MTP ≥25 unmet on native

## Lesson

**Mini steered gates are green; honest 31B native DFlash still far from MLX accept≈3 and ≥15 tok/s — treat latest exact=true with mode-locked streams as partial, not ship clearance.**

## Reproduction

- Replay: From `Rust_MLKit/gemma-metal`: `CARGO_TARGET_DIR=target GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0 cargo run --release --bin bench -- --dflash-31b`.
- Compare regenerated outputs against the exact artifacts below; do not overwrite the preserved results.

## Evidence quality

**Confidence: Low.** Synthetic mini gates are steered, real streams can mode-lock, acceptance is far below MLX, and historical false passes/NaN collapse invalidate stronger claims.

## Artifacts

- `Rust_MLKit/gemma-metal/bench/results/latest_dflash_parity_gates.json`
- `Rust_MLKit/gemma-metal/bench/results/run_dflash_parity_gates_1784001103.json`
- `Rust_MLKit/gemma-metal/docs/dflash_port.md`
- `Rust_MLKit/gemma-metal/docs/gates.md` — Phase 5 DFlash table
- `Rust_MLKit/gemma-metal/docs/audit_deep_2026-07-14.md`

## Why this experiment happened

MLX DFlash cleared the product gates, while custom Metal remained slower and lacked a validated speculative path. Porting block verification therefore became a joint correctness-and-speed problem: exactness, acceptance, stream diversity, and tok/s had to clear together. The preceding notebook context is [35-mlx-dflash-block-tuning](35-mlx-dflash-block-tuning.md).

## Experiment story

**Baseline.** MLX DFlash cleared the product gates, while custom Metal remained slower and lacked a validated speculative path. Porting block verification therefore became a joint correctness-and-speed problem: exactness, acceptance, stream diversity, and tok/s had to clear together. The preceding notebook context is [35-mlx-dflash-block-tuning](35-mlx-dflash-block-tuning.md).

**Hypothesis.** Porting MLX DFlash block-verify into gemma-metal (GPU draft + M>1 verify) would match greedy exactness and approach MLX mean_accept≈3 @ bs=5 while clearing product ≥15 / ≥25.

**Test contract.** Trainer / preset: `bench --dflash-31b` / parity gate harness writing `latest_dflash_parity_gates.json` Fixed knobs: hazard skip-auto on product 31B; always-on + MASK-steer **synthetic mini only**; draft FA `1/√d`, target FA `1.0`, embed `√H` Env flags: `GEMMA_METAL_31B_VERIFY_DIAG=1` for GEMM≡seq checks; dual act-scratch ×`VERIFY_MAX_M`

**Variant sequence.** The preserved comparison matrix was: Synthetic mini — Steered draft; M×GEMV verify (H=256); Real 31B Hot — Q4 GEMM + FA(Tq=M) when cols>256; MLX golden reference — exact verify PASS, mean_accept≈3.0.

**Measured turn.** The result board records Mini (latest) — DFlash best **764.8** tok/s — ≥ hazard greedy **739.4**; exact **PASS**; mean_accept@exact **2.0** — Steered — not HF-ready; Real 31B (latest gates) — best DFlash **1.91** @ bs=3 — greedy **4.57**; sweep mean_accept **0.77** (bs=3) / **1.0** (bs=5) — `exact_vs_greedy: true` but streams mode-lock (unique small); Honest pre dual-norm — DFlash **~1.17** @ bs=5 — mean_accept **≈0**; exact **FAIL** — `run_dflash_parity_gates_1784001103.json`; MLX reference — **~28–37** tok/s — mean_accept **≈3.0**; exact **PASS** — Product bar; Product gates — ≥15 / ≥25 — **false** / **false** — in `latest_dflash_parity_gates.json`.

**Turning point and readout.** Port progress (`dflash_port.md`): GEMM dual-norm + `layer_scalar` ×M **landing/landed**; device capture; conditioner Q8 g64. Remaining: drafts still ≪ MLX proposals (anchor **929≠531** on short ids); DFlash still ~2× slower than greedy under AO-while-capture tax. **Interpretation boundary.** Synthetic throughput and steered acceptance are not product evidence. Real 31B acceptance, stream diversity, exactness, and tok/s must all pass together before promotion.

**Failures and surprises.** Prior mean_accept≈3.8 claim invalid (NaN-collapsed target) — discard `run_dflash_parity_gates_final.json` Vacuous AO exact PASS when unique≤4 — do not claim product readiness Hazard exactness historically FAIL near-ties; mid-verify host sync under hazard not used for honest score Product 31B decode ≥15 / MTP ≥25 unmet on native

## Decision and aftermath

**Kept:** Mini steered gates are green; honest 31B native DFlash still far from MLX accept≈3 and ≥15 tok/s — treat latest exact=true with mode-locked streams as partial, not ship clearance. **Boundary:** Incomplete or failed confirmation prevents promotion beyond the explicitly successful stages. The notebook continues with [37-golden-token-parity](37-golden-token-parity.md); that link records the downstream experiment, not proof that this suite alone caused it.

## Detailed observations

- The result artifact reports: Mini (latest) — DFlash best **764.8** tok/s — ≥ hazard greedy **739.4**; exact **PASS**; mean_accept@exact **2.0** — Steered — not HF-ready.
- The result artifact reports: Real 31B (latest gates) — best DFlash **1.91** @ bs=3 — greedy **4.57**; sweep mean_accept **0.77** (bs=3) / **1.0** (bs=5) — `exact_vs_greedy: true` but streams mode-lock (unique small).
- The result artifact reports: Honest pre dual-norm — DFlash **~1.17** @ bs=5 — mean_accept **≈0**; exact **FAIL** — `run_dflash_parity_gates_1784001103.json`.
- The result artifact reports: MLX reference — **~28–37** tok/s — mean_accept **≈3.0**; exact **PASS** — Product bar.
- Failure/operational record: Prior mean_accept≈3.8 claim invalid (NaN-collapsed target) — discard `run_dflash_parity_gates_final.json`
- Failure/operational record: Vacuous AO exact PASS when unique≤4 — do not claim product readiness

## What this does not prove

**Confidence: Low.** Synthetic mini gates are steered, real streams can mode-lock, acceptance is far below MLX, and historical false passes/NaN collapse invalidate stronger claims. Incomplete or failed confirmation prevents promotion beyond the explicitly successful stages. The recorded association does not by itself establish transfer to other models, datasets, hardware, seeds, prompts, or budgets. Where the story above connects suite order to motivation, that chronology is explicitly a reconstruction from notebook ordering and linked artifacts, not a quoted contemporaneous rationale.

## See also

- Related suites: [`37-golden-token-parity`](37-golden-token-parity.md), [`34-mlx-dflash-product`](34-mlx-dflash-product.md), [`41-audit-deep-2026-07-14`](41-audit-deep-2026-07-14.md)

---

[Previous](35-mlx-dflash-block-tuning.md) · [Index](../00-INDEX.md) · [Next](37-golden-token-parity.md)
