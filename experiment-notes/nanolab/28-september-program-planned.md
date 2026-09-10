# 28: The September program — E27 in flight, E28–E35 registered

## Executive summary

- **Question:** Do the two comparator defects the September 4 review found (the GDN
  delta-rule variant; the top-1 MoE router with no task gradient) change the boards
  that depend on them, and does the 8+4 minGRU/attention hybrid's no-regret curve
  survive parameter parity, a within-suite control, the recall grid and a copy-loss
  reading of the crossing?
- **Result:** **No result exists** for E28–E35. E27 (the width-1536 rung of the ladder)
  is half complete: its learning-rate probe resolved (minGRU lr40, interior), its minGRU
  arm is running, and four of its five attention jobs need one relaunch after an OOM
  collision. Any number quoted for E28–E35 today is fabricated.
- **Implication:** Nothing changes until the boards run. The code is committed
  (`1b6c5d0`), every arm has been timed on the GH200, and the readouts are
  pre-registered in the stage scripts and the handoff.
- **Status:** `planned` (E28–E35) / `running` (E27); evidence confidence **Low**.

## Meta

| Field | Value |
|-------|-------|
| Suite id | `28-september-program-planned` |
| Dates | review 2026-09-04; code and handoff 2026-09-05; E27 running 2026-09-05 |
| Hardware | Lambda GH200 480GB (94.5 GiB usable), torch 2.7.0 / CUDA 12.8, $2.29/h |
| Status | `planned` / `running` |

## Where the specification lives

- Operating document (box state, rules, launch lines, pre-registered readouts, prices,
  open decisions): [`docs/architecture-review-2026-09-04/LAMBDA_HANDOFF_2026-09-05.md`](../../docs/architecture-review-2026-09-04/LAMBDA_HANDOFF_2026-09-05.md).
- The review the program answers: [`docs/architecture-review-2026-09-04/README.md`](../../docs/architecture-review-2026-09-04/README.md)
  (its "Status on September 5, 2026" section is the short version).
- The measured cost basis: `docs/GPU_TUNING_2026-09-05.md` on branch
  `claude/lambda-gh200-sept-2026-f54de2` (unmerged at the time of writing).
- Backlog entry: [`docs/EXPERIMENT_BACKLOG_2026-08-26.md`](../../docs/EXPERIMENT_BACKLOG_2026-08-26.md), "Tier 2b".

## Setup

| board | question | arms | recipe | runs |
|---|---|---|---|---|
| E27 | width 1536 on the E21 ladder | `w1536_attention_lr80`, `w1536_mingru_lr40` | 50M, bs32, ctx512, each arm at its probe argmin; attention at workers 2, minGRU at workers 1 (two directories) | 10 |
| E28 | published vs repo delta rule on recall | `gdn`, `gdn_pub`, `hybrid_gdn_periodic`, `hybrid_gdn_periodic_pub` | E8's MQAR grid (p=4,8 × 3k,9k steps, batch 256, 15 seeds) + `gdn_pub` at seq 255 | 120 + 15 |
| E29 | MoE with a task gradient at the router | `attention`, `moe_e1k1_raw`, `moe_e4k1_raw`, `moe_e8k1_raw` | suite-26 recipe, `crossover50m_moe32d`, workers 2 | 20 |
| E30 | the 8+4 hybrid within-suite and near parity | `attention` into `ratioplace32`; `hybrid_mingru8_attn4_x1`, `hybrid_mingru_periodic_x1`, `attention` in `crossover50m_parity32` | suite-26 recipe | 5 + 15 |
| E31 | tied/untied × value residual | `attention`, `attention_untied`, `attention_novr`, `attention_untied_novr` | suite-26 recipe, `crossover50m_tie32` | 20 |
| E32 | 8+4 on the recall grid | `hybrid_mingru8_attn4` | E8's grid | 60 |
| E33 | depth for width at ~21M non-embedding | `attention`, `attn6_w512`, `attn6_w576`, `w384_attention_lr10` (+ `attn3` read from `loop32`) | suite-26 recipe, `crossover50m_shape32` | 20 |
| E34 | copy-loss drop vs the crossing token | `attention`, `mingru`, `hybrid_mingru8_attn4` with `copy_probe` | suite-26 recipe + `CROSSOVER_COPY_PROBE=1`, `crossover50m_copy32` | 15 |
| E35 | the token ladder at width 384 | `w384_attention_lr80`, `w384_mingru_lr40`, `w384_hybrid_mingru8_attn4_lr40/lr80` | 200M tokens (0.4 epochs), `crossover200m_w384`; 800M is a decision | 20 (+20) |

Estimated 17 GPU-h (~$39) with the 3000-step recall cells only, 34 GPU-h (~$78) with the
9000-step cells, from the sprint's measured per-job minutes at tenancy 1 (attention 6.4,
the x1 hybrids 6.3–6.4, `gdn` 30.0, `moe_e8k1_raw` 18.9; recall cells 229 s per run at
four workers, tripled for 9000 steps — the last figure is inferred). E35's second rung
is ~23 GPU-h more.

## What was verified before any board ran

- `python3 -m nanolab.tests`: 172/172 pass, 3 SKIP (no CUDA), on the laptop.
- Every new arm built and ran training steps on the GH200 during the tuning sprint
  (table A of its report); parameter counts match the CPU build.
- The published GDN rule costs the same as the repo rule (27.8K tok/s both); the
  expansion-1 hybrids train at attention's rate (131K tok/s) where the expansion-2
  hybrid runs at 0.85x.
- The board's determinism floor: two runs of one configuration differ by 0.0014 nats
  (max 0.0031); the effect E30 tests is about twelve times that.

## What this note will hold when the boards finish

The paired readouts against their pre-registrations (handoff §5.2), one dated doc per
board under `docs/` in the style of `LADDER_BOARD_2026-09-04.md`, and, for E28/E29, a new
row in the review's corrections table.
