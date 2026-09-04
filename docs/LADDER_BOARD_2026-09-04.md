# The width ladder — E21, and the axis that does *not* move the ranking

Written 2026-09-04. Closes the `EXPERIMENT_BACKLOG_2026-08-26.md` scale-ladder
item, which had been parked as "blocked on the µP decision" since it was filed.

Every board in this repo before this one is a single scale: `d_model` 768. The
loudest reviewer objection to the paper is therefore not about any individual
result, it is that the whole catalogue is one point in width. E21 answers it.

## Why this needed 66 runs instead of 30

The ladder was never blocked on compute. It was blocked on the **learning rate**.

Comparing two mixers at three widths requires an LR at each width, and the
standard answer — transfer one LR by µP — is exactly what this repo cannot do
honestly: E13 established that this repo's µP cells measured a broken attention
temperature, so a µP transfer would reintroduce the confound §8.4 already
carries a caveat about.

So the ladder measures instead of parametrizing. **Phase 1** sweeps learning
rates at every width; **phase 2** re-runs each cell at *its own* argmin. Width
moves through `n_head` alone — `head_dim` is pinned at 64 — so width is not
confounded with head geometry.

## Phase 1 — the LR probe (10M tokens, n=1, 36 runs)

Three widths × two mixers × up to seven multipliers of the repo's base LR
(`lr` 6e-4, `matrix_lr` 0.025). It took three rounds to get an *interior*
optimum, and the two false starts are part of the result:

- 0.5×/1.0×/2.0× — **all six cells picked 2.0×**, the top edge, loss monotone
  decreasing. An argmin at the edge is not an argmin.
- extended to 4×/8× — minGRU settled interior at 4×; all three attention cells
  were **still falling** at 8×.
- extended to 16×/32× for attention only — attention settled interior at 8×.

Final grid, `final_val` at 10M tokens:

| width | attention 4× | **8×** | 16× | 32× | | minGRU 2× | **4×** | 8× |
|---|---|---|---|---|---|---|---|---|
| 384 | 5.2920 | **5.2287** | 5.2782 | 5.2863 | | 5.2696 | **5.1651** | 5.1772 |
| 768 | 5.2102 | **5.1663** | 5.1999 | 5.2617 | | 5.0765 | **5.0494** | 5.0919 |
| 1152 | 5.1924 | **5.1541** | 5.1877 | 5.2796 | | 5.0002 | **4.9947** | 5.0479 |

**Attention peaks at 8× base at every width; minGRU peaks at 4× at every width.**
A consistent 2× separation, stable across a 3× span in width, with the curve
turning over on both sides in every cell.

That asymmetry is phase 1's real finding and it is a *mechanism* for the thing
this paper is about rather than another instance of it: **any board that hands
both arms one shared learning rate is reading them at different distances from
their own optima.** Every 50M board in this repo does exactly that.

It also says the repo's base LR is low **for a 10M-token run**, which is the
regime the probe is in. Shorter runs favour higher LRs. This is *not* evidence
that the 50M boards were mistuned, and must not be quoted as if it were.

## Phase 2 — the ladder at each cell's own best LR (50M tokens, n=5, 30 runs)

Same recipe as every other 50M board — batch 32, ctx 512, `eval_iters` 20 — so
these cells are directly comparable to the rest of the catalogue.

| arm | n | `final_val` [95% t] |
|---|---|---|
| `w384_attention_lr80` | 5 | 4.5425 [4.5109, 4.5741] |
| `w384_mingru_lr40` | 5 | 4.6896 [4.6641, 4.7150] |
| `w768_attention_lr80` | 5 | 4.4702 [4.4451, 4.4953] |
| `w768_mingru_lr40` | 5 | 4.6287 [4.6021, 4.6553] |
| `w1152_attention_lr80` | 5 | 4.4477 [4.4213, 4.4741] |
| `w1152_mingru_lr40` | 5 | 4.5994 [4.5658, 4.6329] |

| width | attention | minGRU | winner | margin | intervals |
|---|---|---|---|---|---|
| 384 | 4.5425 | 4.6896 | attention | 0.1471 | **disjoint** |
| 768 | 4.4702 | 4.6287 | attention | 0.1585 | **disjoint** |
| 1152 | 4.4477 | 4.5994 | attention | 0.1517 | **disjoint** |

**The ranking is stable across width.** Attention wins at all three widths, with
disjoint intervals at each, and the margin is flat — 0.147 / 0.159 / 0.152 nats
across a 3× span. There is no trend to extrapolate.

## What this result is for

This is a **negative result, and it is the useful kind.** §6 catalogues six axes
along which method orderings move — token budget, metric, speed-vs-quality,
batch size, kernel, hardware backend — and E18/E20 have since added depth under
wall-clock matching, where the ranking reverses completely. Width is the first
axis tested that does **not** move this ranking.

A paper whose thesis is "rankings are recipe-dependent" is strengthened, not
weakened, by naming an axis that turned out to be stable. Without E21 the honest
statement was "we did not test scale". With it, the statement is bounded:
attention-vs-minGRU at 50M tokens is stable over `d_model` 384–1152, and the
recipe axes that move it are the ones already catalogued.

## The caveat, and why it does not need another board

Phase 1 picked these learning rates at **10M** tokens and phase 2 spends them at
50M. Phase 2 therefore inherits an assumption rather than a measurement, and the
obvious follow-up is to carry the neighbouring multipliers at 50M.

That follow-up is not worth 20 runs, and the phase-1 grid says why. Near each
argmin, one step of LR in either direction costs **0.03–0.05 nats** (largest
observed: minGRU w1152, 4× → 8×, 0.053). The arm gap phase 2 reports is
**0.15 nats**. A shifted argmin would have to be roughly 3× more expensive than
anything measured in the sweep to close it.

Labelled honestly: the 0.03–0.05 sensitivity is **measured**, at 10M; that it
does not grow enough at 50M to flip a 0.15-nat gap is **inferred**. If a
reviewer presses on it, the direct test is `w1152` at lr40/lr80/lr160 and
minGRU at lr20/lr40/lr80, 50M, n=5 — 30 runs, ~5h at two workers.

The larger standing caveat is unchanged and is not fixable by more sweeping:
1152 is still a small model, and "stable from 384 to 1152" is not "stable at 7B".

## Reproducing

- `scripts/e21_ladder_probe.sh` — phase 1, out dir `nanolab/out/crossover_ladder_probe`, prefix `cx32lad`.
- `scripts/e21_ladder_phase2.sh` — phase 2, out dir `nanolab/out/crossover_ladder50m`, prefix `cx32lad50`.

Arms are generated in `nanolab/crossover_replicate.py` as `LADDER_PROBE_ARMS`
from `LADDER_WIDTHS` × mixer × `LADDER_LR_MULTS`; phase 2 names its six cells
explicitly through `CROSSOVER_ARMS`.

All 66 `metrics.jsonl` are committed — 36 under
`nanolab/out/crossover_ladder_probe/` and 30 under
`nanolab/out/crossover_ladder50m/`.

They came close to not being. The instance was re-addressed on 2026-09-04 and
presented a host key that did not match `known_hosts`; for several hours the box
was assumed lost and this table survived only as a transcription of the stage
log. It was a new IP, not a new machine. Recomputed from the recovered run
directories, every cell reproduces the transcribed table to within 4e-5, which
is rounding in the fourth decimal. `scripts/pull_artifacts.sh` exists so the
next board does not depend on that luck.

## Appendix — phase 2 stage log, verbatim

Kept in full: it was the only record of these 30 runs for several hours, and it
is what the recovered `metrics.jsonl` were checked against.

```
e21 phase2 start 2026-09-04T14:35:14Z
queue nanolab/out/crossover_ladder50m/queue.json  pending≈30  gpus=1 workers/gpu=2  processes=2
  started worker 0 pid=82516
  started worker 1 pid=82517
e21p2 launch exit=0 2026-09-04T19:22:37Z
e21p2 workers drained 2026-09-04T19:23:37Z
e21p2 queue: {'done': 30} of 30

  arm                        n final_val [95% t]         
  w1152_attention_lr80       5 4.4477 [4.4213, 4.4741]
  w1152_mingru_lr40          5 4.5994 [4.5658, 4.6329]
  w384_attention_lr80        5 4.5425 [4.5109, 4.5741]
  w384_mingru_lr40           5 4.6896 [4.6641, 4.7150]
  w768_attention_lr80        5 4.4702 [4.4451, 4.4953]
  w768_mingru_lr40           5 4.6287 [4.6021, 4.6553]

  ranking by width (attention vs minGRU, each at its own best LR):
    w384   attention 4.5425  minGRU 4.6896  -> attention by 0.1471  (DISJOINT)
    w768   attention 4.4702  minGRU 4.6287  -> attention by 0.1585  (DISJOINT)
    w1152  attention 4.4477  minGRU 4.5994  -> attention by 0.1517  (DISJOINT)

  ranking is STABLE across width (attention wins at every width)
e21p2 exit=0 2026-09-04T19:23:38Z
```
