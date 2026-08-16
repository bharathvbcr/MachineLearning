# 53: Loop research tick — Soft FA_TILED frontier (2026-07-23)

## Executive summary

- **Question:** With no live SOTA/gemma trains holding the GPU, what additive Soft+FA_TILED trials move the Metal quality frontier, and which open claims survive honest measurement?
- **Result (tick1–tick3 harvest):** BLOCKSOFT noise-tied. Polar@0.025 reverses at Soft 3k. Mingru Soft 20k sota EMA **1.9933** (still +0.096 vs Soft attn golden **1.8969**). **T13 16M Soft FA_TILED 20k FINAL EMA 1.7591** (live 1.7601@20k; ~8k tok/s) — new 16M Soft quality lock; **do not** compare to sota 1.9944.
- **Implication:** Continue the **16M Soft FA_TILED + NS5** lane (seed42 confirm). Soft attn remains sota Soft crown; mingru is throughput/probe only. Skip Polar Soft 20k / BLOCKSOFT / gemma.
- **Status:** `partial` (16M seed42 confirm next); confidence **Medium–High** for T13 one-seed 16M Soft 20k.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `53-loop-research-soft-fatiled-2026-07-23` |
| Dates | `2026-07-23` |
| Hardware | Apple M5 Pro, 20-core GPU, 64 GB unified |
| Status | `partial` |
| Artifacts | `Rust_MLKit/arch_02_value_resid/metal-native/out/loop_research_2026-07-23/` |

## Hypothesis

1. `METAL_NATIVE_FA_BLOCKSOFT=1` would improve Soft FA_TILED EMA beyond PhaseG’s Δ≈−0.002 noise margin.
2. Polar Muon (128M champion) would transfer to sota Soft at matrix LR 0.05.
3. Note-52 mixer crossover throughput (~238k tok/s mamba/mingru) was stub-inflated and must be re-baselined on FineWeb.

## Setup

- Binary: `metal-native/target/release/train` (mtime 2026-07-21)
- Data: FineWeb SP1024 + `burn-port/token_bytes.json`, `METAL_NATIVE_DATA_SEED=0`
- Fixed: `--f32 --clip-soft`, `METAL_NATIVE_FA_TILED=1`, `METAL_NATIVE_GEMM_ACCUM` unset
- Soft quality trials: `--golden-init --seed 1337` (sota attention)
- Mixer honesty: seeded init (no `--golden-init`) — golden banks lack `mamba_*` / `mingru_*`

## Variants

| ID | Change | Steps |
|----|--------|------:|
| T1 | Soft FA_TILED NS5 ctrl | 1000 |
| T2 | + `METAL_NATIVE_FA_BLOCKSOFT=1` | 1000 |
| T3 | Polar `--matrix-lr 0.05` | 1000 |
| T4 | Mixer golden-init (failed for SSM) | 40 |
| T4b | Mixer seeded attn / mamba2 / mingru | 40 |
| T5 | Soft FA_TILED ctrl vs BLOCKSOFT | 3000 |
| T6 | Polar default matrix LR (0.025) | 1000 |
| T7 | Polar@0.025 Soft FA_TILED seed1337 | 3000 |
| T8 | NS5 Soft FA_TILED seed42 | 3000 |
| T9 | Polar@0.025 Soft FA_TILED seed42 | 3000 |
| T10 | Mixer attn / mamba2 / mingru seeded | 3000 |
| T11 | mingru 3k seed42 confirm | 3000 |
| T12 | mingru Soft FA_TILED 20k warmdown3500 | 20000 |
| T13 | 16M Soft FA_TILED 20k NS5 seeded | 20000 |

## Results

### Locked prior baselines (not re-run this tick)

| Artifact | FINAL EMA BPB | Notes |
|----------|--------------:|-------|
| Soft 100k WSD seed1337 | **1.882767** | Best Soft quality lock |
| Soft 20k warmdown seed1337 | 1.89688 | |
| PhaseG ctrl 3k | 2.106312 | Prior BLOCKSOFT A/B |
| PhaseG blocksoft 3k | 2.104363 | Δ −0.00195 |
| Polar 128M audit8 2k | 2.010659 | |
| Polar 128M long20k | 1.815495 | |
| 100k last-10% warmdown | STOPPED ~53.7k | gnorm explode |

### This tick Soft FA_TILED board

| Rank | Run | FINAL EMA BPB | mean tok/s | max gnorm | Notes |
|-----:|-----|--------------:|-----------:|----------:|-------|
| 1 | T6 Polar default-lr 1k | **2.247580** | 66919 | 0.95 | Beats NS5 1k |
| 2 | T2 BLOCKSOFT 1k | 2.250316 | 62743 | 1.03 | ≈ T1 |
| 3 | T1 NS5 ctrl 1k | 2.250343 | 60954 | 1.03 | |
| 4 | T3 Polar lr0.05 1k | 2.252484 | 63970 | 0.71 | 128M LR hurts sota |
| 5 | T5 BLOCKSOFT 3k | **2.121670** | 66117 | 1.20 | Δ −0.00071 vs T5 ctrl |
| 6 | T5 NS5 ctrl 3k | 2.122381 | 65788 | 1.03 | Higher than PhaseG 2.106 (Audit8/9 defaults) |

### Tick2 — Polar@0.025 vs NS5 Soft FA_TILED 3k (golden, two seeds)

| Seed | NS5 FINAL EMA | Polar@0.025 FINAL EMA | Δ (Polar−NS5) |
|-----:|--------------:|----------------------:|--------------:|
| 1337 | **2.122381** (T5) | 2.127708 (T7) | +0.005327 |
| 42 | **2.127647** (T8) | 2.131005 (T9) | +0.003358 |
| **mean** | **2.125014** | 2.129357 | **+0.004342** |

Throughput ~68k tok/s both arms. Polar max gnorm higher on s1337 (2.62 vs ~1.0). **Verdict: NS5 wins; do not promote Polar Soft 20k from the 1k hint.**

### Tick2 — honest mixer quality 3k (seeded, seed1337, FA_TILED env on)

| Mixer | FINAL EMA BPB | mean tok/s | max gnorm | Notes |
|-------|--------------:|-----------:|----------:|-------|
| mingru | **2.085016** | 151707 | 1.29 | Best 3k mixer EMA this campaign |
| attention | 2.123010 | 67724 | 1.01 | ≈ Soft NS5 golden board |
| mamba2 | 2.226744 | 35037 | 1.16 | Quality + speed lag at T=256 |

Caveat: mixer arms use seeded init (not golden); param counts differ (attn 0.78M / mingru 0.98M / mamba2 1.05M). Promote mingru only after seed42 + longer horizon.

### Tick3 — mingru two-seed 3k confirm + Soft 20k

| Run | FINAL EMA BPB | mean tok/s | max gnorm | Notes |
|-----|--------------:|-----------:|----------:|-------|
| T10 mingru 3k s1337 | 2.085016 | 151707 | 1.29 | tick2 |
| T11 mingru 3k s42 | **2.076374** | 155031 | 1.84 | confirms lead |
| **mean 3k** | **2.080695** | — | — | vs attn 3k 2.1230 |
| **T12 mingru Soft 20k s1337** | **1.993295** | **157243** | 5285 @12.2k | warmdown 3500 Soft recipe |

Comparisons for T12:
| Reference | BPB | Δ (mingru−ref) |
|-----------|----:|---------------:|
| CUDA sota-ladder ref | 1.9944 | **−0.0011** (ties ref) |
| Soft attn golden 20k | **1.89688** | +0.0964 (attn still wins Soft) |
| Soft attn 100k WSD | 1.882767 | +0.1105 |

Gnorm: first >100 @5400, >1000 @11400, peak 5285 @12200; live BPB still descended through warmdown to 1.9936@19999.

### Tick3 — 16M Soft 20k NS5 (**HARVESTED**)

| Metric | Value |
|--------|------:|
| FINAL EMA BPB | **1.759141** |
| live @19999 | **1.7601** |
| mean tok/s | ~7968 |
| params | 16.412M (12L×384d) |
| max gnorm | 2913 @ mid-run; last 340 |
| best live pre-warmdown | 1.9823 @8999 |
| warmdown window | steps 16500–20000 |

Live trajectory (eval every 1k): 2.256→2.177→2.109→2.110→2.073→2.018→2.004→2.044→**1.982**→1.995→2.027→2.000→2.004→**1.904**→1.849→1.797→1.797→1.783→1.762→**1.760**.

**Same-scale:** prior 16M Soft 1k EMA **2.1462** → 20k Soft **1.7591** (Δ **−0.387**).  
**Cross-scale context only (not ranking):** Soft attn sota golden 20k 1.8969 · mingru Soft 20k 1.9933 · Polar 128M long20k 1.8155. Log explicitly forbids comparing to sota **1.9944**.

First launch with `--golden-init` failed; successful run is **seeded** Soft FA_TILED + `--warmdown 3500`.

### Mixer honesty (40 FineWeb steps, seeded, FA_TILED env on)

| Mixer | params | loss 0→39 | mean tok/s | ms/step | disp | Verdict |
|-------|-------:|-----------|-----------:|--------:|-----:|---------|
| attention | 0.780M | 6.932→5.114 | **66382** | 61.7 | 250 | Real FA path |
| mamba2 | 1.053M | 6.932→5.154 | **33321** | 117.5 | 307 | Real; **slower** than attn at T=256 |
| mingru | 0.977M | 6.931→5.262 | **144367** | 27.2 | 195 | Real; fast but not note-52’s ~239k |

T4 golden-init failed: `mamba_in_proj missing` / `mingru_to_z missing`.

## Failures

- Golden-init Soft recipe is attention-only; SSM mixer screens must use seeded init.
- This tick’s 3k Soft EMA (~2.122) is not numerically identical to PhaseG (~2.106) — treat as current-binary reconfirm, not bit-replay.
- No 20k Soft, no 16M Soft, no CUDA 128M parity this tick (intentionally additive micro-board).

## Lesson

**16M Soft FA_TILED 20k locks EMA 1.7591 (seed1337) — continue that lane. Soft attn still owns sota Soft quality; mingru Soft 20k is fast but not Soft-crown. Polar Soft promotion remains closed.**

## Reproduction

```bash
cd Rust_MLKit/arch_02_value_resid/metal-native
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export METAL_NATIVE_FA_TILED=1 METAL_NATIVE_DATA_SEED=0
unset METAL_NATIVE_GEMM_ACCUM
DATA=../../../parameter-golf/data/datasets/fineweb10B_sp1024
TOK=../burn-port/token_bytes.json
BIN=./target/release/train

# T1 ctrl 1k
"$BIN" --data-dir "$DATA" --token-bytes "$TOK" --out out/loop_research_2026-07-23/T1_soft_fatiled_ctrl_1k \
  --iters 1000 --seed 1337 --golden-init --f32 --clip-soft --log-every 100 --eval-every 500

# T2 blocksoft 1k
METAL_NATIVE_FA_BLOCKSOFT=1 "$BIN" ... --out .../T2_soft_fatiled_blocksoft_1k --iters 1000 ...

# T6 Polar default LR
"$BIN" ... --optimizer muon_polar_adamw --iters 1000 ...

# Mixer honesty (seeded)
"$BIN" ... --mixer mamba2 --iters 40 --seed 1337 --f32 --clip-soft   # no --golden-init
```

Machine-readable board: `out/loop_research_2026-07-23/RESULTS.json`.

## Evidence quality

**Confidence: Medium–High** for Polar vs NS5 Soft 3k (two seeds), mingru 3k two-seed, and T13 16M Soft 20k one-seed. Mingru Soft 20k one-seed Medium. No same-shape CUDA 16M ref yet.

## Next highest-EV trials (next loop tick)

1. **16M Soft FA_TILED 20k seed42** confirm (same seeded Soft+warmdown3500 NS5 recipe) — primary.
2. Optional later: Soft attn **seeded** sota 20k for fair mingru compare; same-shape CUDA 16M ref.
3. Do **not** run Polar Soft 20k, BLOCKSOFT promo, mingru Soft 20k seed42 (attn still leads Soft quality), or gemma fusion.
