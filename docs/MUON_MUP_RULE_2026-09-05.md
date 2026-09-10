# The µP learning-rate rule for Muon — measured, and it is the wrong one

Written 2026-09-05. Closes the "did the transfer land" question §8.4 leaves
dangling, and identifies a second arm-asymmetric defect in the µP path.

## What the code does

`nanolab/optim.py:750` scales the matrix-group learning rate by width:

```python
matrix_lr = cfg.matrix_lr / width_mult      # µP's rule for ADAM
```

That is the µP prescription for **Adam**, applied to a **Muon** group. Muon
orthogonalizes its update (Newton–Schulz), so the update's scale is already
independent of the gradient magnitude — the reason Adam's `1/fan_in` scaling
exists does not apply. Whether that matters is an empirical question, and the
basin answers it.

## The measurement

`e1_mup_basin_spattn` (42 jobs) swept a factor-2 grid; `e1_mup_basin_fine_spattn`
(18 jobs, 2026-09-05) added 1.5×/3×/6× to interleave with 2×/4×/8×, giving
factor-1.5 resolution where both optima sit. Multipliers are on the *transferred*
value, so **1× is what `optim.py` predicts** and **3× is what no width divisor
at all predicts** (`width_mult` = 768/256 = 3).

`final_val`, s24 recipe, µP with SP's attention temperature:

| ×transferred | attention | minGRU |
|---|---|---|
| 0.25 | 5.3152 | 5.1451 |
| 0.5 | 5.0835 | 5.0865 |
| **1 — `optim.py`** | **4.8753** | **5.0169** |
| 1.5 | 4.7625 | 4.9994 |
| 2 | 4.7118 | **4.9933** ← min |
| **3 — no divisor** | **4.6832** ← min | **4.9939** |
| 4 | 4.6853 | 5.0020 |
| 6 | 4.7003 | 5.0241 |
| 8 | 4.7253 | 5.0468 |
| 16 | 4.8035 | 5.1224 |
| 32 | 4.9003 | 5.2252 |

## What it says

**The no-divisor rule lands on the optimum.** 3× *is* attention's argmin exactly,
and is 0.0006 nats off minGRU's — indistinguishable at n=3.

**The Adam divisor does not.** 1× costs **+0.1921** nats on attention and
**+0.0236** on minGRU against each arm's own best.

**And it costs them unequally — 8.1× more on one arm.**

That last line is the point. This is not merely a suboptimal learning rate; it is
an *arm-asymmetric* one, in a repository whose central comparison is attention
versus minGRU. It is the §8.1 confound — "one arm was closer to its own optimum
than the other" — introduced by the harness itself, and it is the **second**
defect of exactly that shape found in the µP path. The first was the attention
temperature (E13), which also hit attention and not minGRU, because minGRU has no
attention logits. Here the asymmetry has a different cause: the two arms simply
have different LR sensitivities (the E21 ladder measured the same thing
independently — attention peaks at 8× base LR, minGRU at 4×, at every width), so
one shared wrong rule displaces them by different amounts.

> **Note 2026-09-09, and it cuts closer than the first version of this note admitted.**
>
> The parenthetical is a **10M-token** result. Re-probed at **50M** (G8a/G8b/G8d, an
> interior minimum in every cell), the E21 ladder says something different, and something
> this doc has a direct stake in:
>
> | width | attention | minGRU | best `lr` | width x lr |
> |---|---|---|---|---|
> | 384 | 2x | 2x | 0.0012 | 0.4608 |
> | 768 | 1x | 1x | 0.0006 | 0.4608 |
> | 1152 | 0.667x | 0.667x [dagger] | 0.0004 | 0.4608 |
> | 1536 | 0.5x | 0.5x | 0.0003 | 0.4608 |
>
>
> **Every argmin now carries n=3, and none of them moved (G11b, 2026-09-10).** That was
> the pre-registered test: *any argmin that shifts once its own point carries three seeds
> was never located*. Each width's argmin and its tight-side neighbour were run at seeds
> 1337/42/100, and every cell returned the same multiplier it had at n=1. Margins are
> paired on the three seeds each pair shares, so seed-to-seed spread — which on these
> boards reaches 0.0574, eighteen times the rerun floor — is differenced out rather than
> averaged in:
>
> | width | mixer | argmin | runner-up | paired margin (n=3) | verdict |
> |---|---|---|---|---|---|
> | 384 | attention | 2x | 1x | +0.0144, 3/3 | resolved |
> | 384 | minGRU | 2x | 1x | +0.0155, 3/3 | resolved |
> | 768 | attention | 1x | 0.5x | +0.0391, 3/3 | resolved |
> | 768 | minGRU | 1x | 0.5x | +0.0529, 3/3 | resolved |
> | 1152 | attention | 0.667x | 1x | +0.0053, 3/3 | resolved |
> | 1152 | minGRU | 0.667x | 1x | +0.0024, 3/3 | **flat — under the 0.0031 floor** |
> | 1536 | attention | 0.5x | 1x | +0.0125, 3/3 | resolved |
> | 1536 | minGRU | 0.5x | 1x | +0.0067, 3/3 | resolved |
>
> **Seven of the eight cells are resolved; `width x lr = 0.4608` holds at all eight.** The
> exception is d1152 minGRU, where 0.667x leads 1x by 0.0024 — the right sign on 3 of 3
> seeds, but under the floor, so it is reported as consistent with the law rather than
> evidence for it.
>
> [dagger] d1152 was re-probed at the law's own point by G11b (2026-09-10), paired on 3
> seeds. Attention's 0.667x beats **both** neighbours above the 0.0031 rerun floor --
> 0.0100 over 0.5x (3/3) and 0.0053 over 1x (3/3) -- so the argmin there is **located**.
> minGRU's 0.667x beats 0.5x by 0.0167 (3/3) but leads 1x by only +0.0024, *inside* the
> floor: right sign, unresolved magnitude. Read minGRU's row as consistent with 0.667x,
> not as evidence for it.
>
> Two things change. **The arm asymmetry is gone at 50M** -- attention and minGRU pick the
> same rate at every width -- so the E21 corroboration cited above is 10M-only. The
> argument here does not depend on it: the asymmetry it needs is measured on this doc's
> own grid, where 1x costs +0.1921 nats on attention against +0.0236 on minGRU.
>
> **The second change is the one to take seriously.** All four widths put the
> optimum at exactly `lr` proportional to `1/width` -- which is the Adam muP rule
> `optim.py:750` implements and this doc concludes is *wrong*. The two are not directly
> comparable and this note does not claim a contradiction: this doc's multipliers are on
> the *transferred* value while E21's scale raw `lr` and `matrix_lr` together, and the E21
> runs have `mup: False` so no divisor was applied to them at all. But both are asking
> whether hidden LR should fall with width, they answer oppositely, and **the variable
> that moved between them is the horizon** -- E21 itself found width-invariance at 10M and
> 1/width at 50M.
>
> **So this doc's verdict should be read as horizon-local until re-measured.** What would
> settle it: this exact grid re-run at 50M. Until then the no-divisor recommendation
> stands on its own measurement and carries this caveat. Labelled **inferred** -- E21's
> 50M probe is n=1 per cell on a 2x-spaced grid, and d1152 does not resolve.

## What is *not* resolved, and does not need to be

Both argmins are **not sign-consistent**: the curves are flat between 2× and 4×,
so the argmin wins on the mean and loses a seed to a neighbour. The exact optimum
— 3× vs 4× on attention, 2× vs 3× on minGRU — is not resolvable here, and more
seeds will not fix a basin this flat (the same situation `e1_sp_basin` hit, and
the reason `analyse_sp_basin` prices across a tied set rather than picking).

It does not need to be. The claim does not rest on the argmin:

- the **no-divisor prediction sits inside the flat region on both arms**;
- the **Adam prediction sits clearly outside it on attention**, by 0.19 nats —
  eight times the width of the flat region it should be in.

## Recommendation, and the decision that is not mine

The evidence supports removing the width divisor from the Muon group. Doing so
changes the learning rate of **every µP run in this repository**, so it is a
load-bearing default and the change should be deliberate rather than incidental.

The suggested shape:

1. Add an explicit flag (e.g. `mup_muon_no_divisor`) rather than silently
   changing `optim.py:750`, so every existing run stays reproducible and the
   recipe fingerprint records which rule a job used.
2. Default it **off** until the decision is made; flip the default in its own
   commit, with this document cited, so the change is greppable.
3. Re-run the one cell that matters at the new rule — µP at the corrected
   transferred value, n=5 — and check the target-width optimum lands at 1×.
   That is the transfer *landing*, which no cell in this repo has yet shown.

Until then, every µP number in this repository was measured through a learning
rate that was wrong by 3× and wrong **unequally between the arms**, and
`e1_mup_basin`'s "TRANSFER MISSED" verdict should be read as a statement about
the rule rather than about µP.
