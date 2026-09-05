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
