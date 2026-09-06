# Two harness changes that live only on the rented box

`~/MLSystemsLab` on the GH200 is a **synced directory, not a git clone** — there
is no `.git` in it. Both changes below were made there, tested there, and would
be destroyed with the instance. They are captured here as unified diffs rather
than as edits to this branch, because this branch's copy of
`nanolab/crossover_replicate.py` is ~247 lines behind the box's: the box carries
the uncommitted §4.3 work (the whole E28–E35 arm registry), and applying these
on top of the older file would conflict or silently drop arms.

Apply against the working tree that has the §4.3 work, not against this branch.

## `crossover-fused-ce-gate.patch`

Adds `cluster_fused_ce()` and records `fused_ce` in `current_recipe()`.

`crossover50m` sets `fused_ce=True`; the preset comment dates that to "the
3070 Ti (8 GB)", where 16 chunks "frees the VRAM ... that lets bs32 fit". VRAM
is not the binding constraint on 94.5 GiB, and once Inductor compiles the loss
it fuses the CE reduction itself, so the hand-written chunking blocks the fusion
it exists to provide.

Measured on a board (5 seeds x 20M, compile on both sides, `fused_ce` the only
variable): **1.14x** steady state, 1.06x whole-board wall clock, and
**0.0037 nats** mean `final_val` shift (max 0.0088) — *above* the 0.0031 rerun
floor. So it is worth taking but is a board-wide recipe change, which is why the
field is now recorded. Default is unchanged (`True`); `CROSSOVER_FUSED_CE=0`
opts out.

A step-loop probe had predicted 1.55x for this. It over-promised by 36% because
it timed training steps only, while a run also evaluates 61 times, loads data
and checkpoints.

## `train-compile-gate.patch`

Widens the `torch.compile` gate to every arm.

The gate has now been wrong twice, both times from a bad measurement. It was
"attention only", from an Inductor stall on aarch64 that torch 2.7.0 does not
reproduce. It was then narrowed to "one mixer kind throughout", because a probe
reported hybrids 1.00x, gdn 0.97x, moe 1.08x — and that probe built every arm in
ONE process. Dynamo's recompile counter is per code object, and `forward`
(`model.py:410`) is the same object for every arm, so the arms measured first
spent the budget of 8 and the ones measured later were never compiled at all
(their "compile" took 0.7 s and 0.3 s; exactly one warning appears in the log,
because Dynamo warns once per frame).

Re-measured with `dynamo.reset()` between arms, then confirmed on a real board
(3 arms x 5 seeds x 10M, eager vs compiled):

| arm | board | steady state | step loop |
|---|---:|---:|---:|
| `attention` | 2.03x | 2.23x | 1.94x |
| `hybrid_mingru8_attn4` | 1.84x | 2.18x | 1.95x |
| `gdn` | 1.68x | 2.82x | 3.17x |

Every arm gains, so there is no condition left to gate on. The board figure is
below steady state because one seed pays Inductor's cold compile — 635 s against
145 s for gdn — and that amortises 5x better at the program's real 50M budget
(projected ~2.5x for gdn, from measured components).

`gdn`'s numerics cost is the outlier: **0.0065 nats** mean, 0.0105 max, over 3x
the rerun floor, against attention's 0.0021 and the hybrid's 0.0018. Compile a
whole GDN board or none of it.

`compile` stays OFF by default and is a recorded recipe field, so widening this
changes no existing run and cannot pool compiled with eager.
