# Documentation integration checks — September 4, 2026

The material was integrated into the repository after the original review.
Existing changes in the shared checkout were preserved; only five existing
documentation files received navigation/qualification additions. Model code,
training configurations, manuscript tables, and experiment outputs were not edited.

## Checks run

- `memory_geometry_probe.py`: passed 10,000 randomized scalar feasibility/bisection
  comparisons, 1,000 projected-update fixtures, and the stated boundary cases.
  [Recorded output](math-check-results.json).
- `verify_curves.py`: passed all 61 shared periodic markers for each of five
  comparisons, with five seeds per comparison; saved deltas and pointwise intervals
  match the local records. These are repeated measurements, not additional seeds.
- `router_gradient_probe.py`: passed, reproducing near-zero top-1 task-router
  gradient and nonzero expert/auxiliary gradients.
- `gdn_rule_probe.py`: passed, reproducing the implemented/published-rule difference
  and agreement in the alpha=1 and beta=0 controls.
- `recompute.py`: ran against the current local records in a fresh temporary output
  directory; generated JSON parsed successfully. A second invocation using the same
  directory was refused, checking overwrite protection.
- `python3 paper/derive_figures.py --check`: passed all 60 derived figures and D7
  manifest consistency. It reports coverage of 61/706 figures (9%); the other 645
  are outside this check, not silently validated.
- Documentation links, supplied-input hashes, JSON parsing, Python syntax, and
  `git diff --check` were checked before delivery.

The fresh inventory reported 3,485 result files, 3,162 distinct hashes, 1,263
metrics files, 456 crossover metrics files, 251 finite-final-CE runs, and 591 MQAR
records. It still identified one historical nonfinite/parse-problem metric file.
These changed counts reflect additional local records since the archived snapshot.
They do not replace the review's 21:48 UTC inventory or establish updated campaign
conclusions. Fresh inventory outputs were temporary; the original evidence was kept.

## Unverified

No training, physical GPU benchmarking, or whole-manuscript verification was run.
The supplied 124/200 trajectory, its 124/124 alternatives, and percentage/norm
summaries remain report-backed. BINN's unavailable raw cells remain unavailable.
The independent geometry fixtures do not reproduce either dataset.

GitPulse scanned three worktrees and reported no overlapping files or unscanned
worktrees. Its code-intelligence facet was unavailable because its reader did not
support the database's future schema version 12; navigation used the repository
map and live files instead. That unavailable facet was not treated as a passed check.

## Additions later on September 4, 2026

- Added `inputs/04-efficiency-reconciliation-memo.md` (version 3.1 of the first memo; SHA-256 of the
  supplied text recorded in `input-provenance.json`). It adds the efficiency and
  serving-cost analysis, the reconciliation with the evidence review, experiments
  9–17, and the first memo's own corrections. It is a historical input, not the verdict.
- Corrected two statements in `README.md` and `EVIDENCE_REVIEW.md` by dated
  supersession notes, not by rewriting: the wall-clock loop board's six-block seed
  reached its full token budget and ran at tenancy 1 (finish order from file times;
  `budget_by_arm` in the suite's `recipe.json`), and nanolab's block-decoding
  algorithms exist for the attention mixer. Also corrected a misattribution in the
  corrections table (the first memo retired APRDH as a unit) and recorded the live
  `moe32c` queue state at the time of the check.
- Reran the four probes on this checkout: all passed with the recorded outputs.
  Rechecked every relative link and heading anchor in this folder and in the five
  linked documents: 0 broken. The earlier "58 links" figure and this check's count
  differ in what they enumerate (unique targets versus link occurrences), not in outcome.
- No model code, training configuration, experiment output, or manuscript file
  was changed.

## Additions on September 5, 2026

- Added `LAMBDA_HANDOFF_2026-09-05.md` (operating document for E27–E35) and revised it
  the same day after (a) the code it specifies was committed (`1b6c5d0`), (b) the E27
  repair was run once on the box and its attention half failed again, and (c) a GH200
  tuning sprint measured per-arm cost, tenancy, MPS, compile and sampler behaviour
  (`docs/GPU_TUNING_2026-09-05.md`, tuning branch). Its board prices were re-derived from
  the sprint's measured per-job minutes; the first version's elapsed ÷ tenancy arithmetic
  is retired in a dated note.
- Added a "Status on September 5, 2026" section and four dated rows to the corrections
  table in `README.md`; dated supersession notes in `EVIDENCE_REVIEW.md` (BINN row, the
  wave-26 coverage sentence, the QK-normalization sentence) and `evidence/README.md`.
  No earlier text was rewritten.
- Verified for this pass: `python3 -m nanolab.tests` on the working tree that includes
  the tuning branch's `nanolab/` port — 172/172 pass, 3 SKIP (no CUDA); the box's
  `nanolab/*.py` md5-identical to that working tree (15:10 UTC); the E27 queue states,
  probe `final_val`s and the running job read from the box at 15:05–15:11 UTC; the
  BINN v3 cell corpus tracked in git (555 files); BINN's `check_every_number.py` and
  `check_verdicts_transcribed.py` run at 15:30 UTC, both reporting waves 26–27 as
  uncovered.
- Not verified here: any number in the tuning report (read from the branch, not rerun);
  the recall-cell cost used to re-price E28/E32 (the sprint's 229 s per run at four
  workers, applied to every cell and tripled for 9000 steps, is arithmetic).
- No model code, training configuration, experiment output, or manuscript numbers were
  changed. `PAPER_2026-08_Recipe_Dependent_Rankings.md` §7.2 gained two dated limitation
  notes (GDN operator variant; compile).
