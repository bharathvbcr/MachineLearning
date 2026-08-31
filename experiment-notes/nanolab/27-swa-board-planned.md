# 27: Sliding-window attention board (E12 / E15 / E16)

## Executive summary

- **Question:** Is a *local-attention* stack competitive with a *linear-attention* stack when both are pretrained identically — on held-out CE at two context lengths, and on in-context recall at sequence lengths that straddle the window?
- **Result:** **No result exists.** Code is complete and tested; no run has been executed. Any number quoted for these arms today is fabricated.
- **Implication:** Nothing changes until the board runs. The specification, cost basis, and the limits of what it can conclude are recorded in [`docs/SWA_BOARD_2026-08-31.md`](../../docs/SWA_BOARD_2026-08-31.md).
- **Status:** `planned`; evidence confidence **Low** (planned stage, no end-to-end outcome).

## Meta

| Field | Value |
|-------|-------|
| Suite id | `27-swa-board-planned` |
| Dates | code 2026-08-31; runs not started |
| Hardware | Target: Lambda GH200, 2 jobs/GPU. Code verified on Apple M-series (MPS/CPU) only. |
| Status | `planned` |

## Hypothesis

Prompted by arXiv:2608.28444 (*Sliding-window beats linear attention*), which reports that
training-free SWA(64, 4) on a **pretrained** model matches or beats a **post-trained**
linear-attention retrofit, and beats it 2–10× on NIAH/BABILong.

The from-scratch analogue this board can test: at matched tokens and recipe, SWA arms
should sit with the attention-containing arms rather than with GDN and minGRU — and if the
paper's mechanism transfers, `swa_w64` should hold up on recall where a pure recurrent
stack does not, with `swa_w64_nosink` showing the sinks are load-bearing.

## Setup

Three stages, one command: `python -m nanolab.crossover_replicate swaboard`
(`probe → swa32 → swa2k → mqar-calibrate → mqar-grid`; each phase resumable and idempotent).

| stage | recipe | arms | runs |
|---|---|---|---|
| E12 `swa32` | identical to suite 26 (bs32, ctx512, 50M, `eval_iters=20`) | `swa_w64`, `swa_w128`, `swa_w256`, `swa_w64_nosink`, `hybrid_mingru10_swa2` | 25 |
| E15 `swa2k` | identical to E9 (bs8, ctx2048, 50M) | `swa_w64`, `swa_w256`, `swa_w512`, `swa_w64_nosink` | 20 |
| E16 MQAR | E8's probe at pairs 16/64/128 → seq 63/255/511 | `attention`, `gdn`, `mingru`, `swa_w64`, `swa_w64_nosink` | 225 |

Estimated 33–90 GPU-h, $152–414, from measured anchors (`cxwc_attention` 11.4 min/run;
`cx2k_attention` 17.3 min/run; `mqar_e8` 159 s/run). E16's multipliers are extrapolation
from a 15-token cell — the calibration phase exists to replace them with measurements.

## Variants

`swa` is the `attention` mixer plus a mask and nothing else: same GQA, RoPE, QK-norm,
gating, value residual, and **identical parameter count** (123,699,612, asserted in
`nanolab.tests`). SWA(w, s) attends to the `s` first positions (StreamingLLM sinks —
ordinary positions, not learned logits) plus the `w − s` most recent.

## Results

**None.** No run has been executed.

## Failures

None yet — but three defects were found and fixed while building this, each of which would
have silently corrupted the board had it run first:

- `e8_config` ignored `Arm.overrides`, so every `swa_*` arm in E16 would have trained at the
  default window: one architecture under four names.
- `run_one` recorded `cfg.mixer` (`"swa"`), which would have collapsed `swa_w64` and
  `swa_w64_nosink` into one ledger row.
- `_mfu_from_toks` carried a second copy of the FLOPs formula and charged any unrecognised
  mixer **zero** attention FLOPs, so SWA's MFU would have been wrong in the probe and right
  in the model with nothing to surface the disagreement. `model.mixer_flops_per_token` is
  now the sole owner.

## Lesson

**No lesson is available from an unrun suite.** The one measured statement that exists is a
negative expectation-setter: at 12L/768d ctx512, SWA(64,4) is only 6% cheaper in counted
FLOPs than full attention (798.8M → 749.3M per token) and equal in wall clock within noise
on MPS. The paper's speed and memory claims live at long context; E15 is the axis on which
they could even in principle appear here.

## Reproduction

```
python -m nanolab.crossover_replicate swaboard
```

Run the `probe` phase first and read its `sdpa` lines. On CUDA an explicit `attn_mask`
cannot use the flash kernel; if only MATH can serve the window, the B·H·T·T score matrix is
materialised per layer and `swa2k` would OOM hours in. `assert_swa_backend_is_viable`
refuses in that case (override with `SWA_ALLOW_MATH=1`). This is the single property of the
SWA path that could not be verified off the target hardware.

## Evidence quality

**Confidence: Low.** Planned stage; no end-to-end outcome. What *is* verified is the code:
118/118 in `nanolab.tests`, of which 18 are new and all 18 were confirmed to fail against
the pre-change source. Cached windowed decode reproduces the uncached windowed forward to
7e-7 with the KV cache bounded at `window − 1` entries.

## What this does not prove

This board pretrains from scratch. The paper masks a **pretrained** model at inference and
compares against a **post-trained** retrofit on NIAH and BABILong. E12/E15/E16 answer the
from-scratch analogue only; no row from them may be reported as a reproduction of that
result, and E16's MQAR numbers are not NIAH numbers. `docs/SWA_BOARD_2026-08-31.md` scopes
what closing the remainder would take and recommends against starting it as part of this
paper.

In E16, sequence length is also task difficulty (more pairs to store), so the SWA-vs-GDN
comparison is valid **within** a cell and a trend read **across** cells cannot separate
distance from difficulty.

## See also

- [`docs/SWA_BOARD_2026-08-31.md`](../../docs/SWA_BOARD_2026-08-31.md) — full spec, cost basis, gap scoping
- [`docs/EXPERIMENT_BACKLOG_2026-08-26.md`](../../docs/EXPERIMENT_BACKLOG_2026-08-26.md) — E12's original entry
- Related suites: [`26-matched32-hybrids`](26-matched32-hybrids.md) (E12's board), [`24-matched20-prefix`](24-matched20-prefix.md)

---

[Previous](26-matched32-hybrids.md) · [Index](../00-INDEX.md)
