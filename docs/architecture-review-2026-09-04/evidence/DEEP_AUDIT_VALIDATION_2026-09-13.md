# September 13 deeper audit — validation record

This records checks performed for [the report](../DEEP_AUDIT_2026-09-13.md).
An audit fixture confirming a defect is a successful characterization, not a
repaired implementation. Existing training and analysis source was left unchanged.

## Current-corpus diagnostic

```bash
python3 docs/architecture-review-2026-09-04/evidence/deep_audit_probe.py --binn-root ../BINN
```

**PASS / completed.** The final JSON is preserved as
[deep-audit-results-2026-09-13.json](deep-audit-results-2026-09-13.json).
The probe writes only stdout plus temporary fixture files. It performs no network
request or campaign launch. Its actual training fixture is six optimizer steps
per complete trajectory, CPU FP32, AdamW, one layer, synthetic tokens, no dropout.

Observed environment: Python 3.14.7, PyTorch 2.13.0, NumPy 2.4.6, SciPy 1.18.0.
No dependencies were installed. These are the audit environment versions, not a
claim that the historical GH200 campaigns used them.

The output contains:

- Twenty paired LM comparisons with full per-seed deltas and config differences.
- Three scoped recall ledgers, with completed/planned counts and missing seeds.
- All 1,512 current NanoLab JSONL files inventoried, plus 1,501 sibling configs
  hashed; 974 metric files contain exactly one finite terminal loss.
- Eight reader-contract diagnostics covering sample-size calibration, omitted
  recipe fields, marker mismatch, directional crossing, duplicate selection,
  incomplete runs, and LR identity collision. The crossing CLI output is saved.
- Actual-trainer resume divergence: maximum parameter error 0.004305274225771427;
  restoring both sampler RNG states externally yields 0.0 on the same fixture.
- Actual train-evaluation sampler-state advancement.
- Protected-rank saturation and the count-only thinning counterexample.
- All 48 W29 cell measurements, twelve paired differences, and cell hashes.

The final source fingerprints and all 3,013 NanoLab metric/config hashes were
checked against the local files after the run. All 48 external W29 hashes were
also checked. This is a dated local snapshot, not a live remote-run status.

## Existing focused checks

These checks passed during this audit before documentation edits. They do not
exercise every new counterexample or replace historical training replay.

| Check | Observed result | Coverage boundary |
|---|---|---|
| `python3 docs/architecture-review-2026-09-04/evidence/verify_curves.py` | PASS: five comparisons, 61 markers, five seeds | Archived curve comparisons; not all new trajectories or terminal draws. |
| `python3 docs/architecture-review-2026-09-04/evidence/memory_geometry_probe.py` | PASS: 10,000 scalar and 1,000 projected fixtures | Mathematical special cases; not the supplied 124/200 trajectory. |
| `python3 docs/architecture-review-2026-09-04/evidence/second_audit_probe.py` | PASS: three probe groups | MQAR information timing, recurrence branches, and order/label controls. |
| Nine selected functions from `nanolab.tests` | PASS: 9/9 | Selection listed below; full suite was not run for this documentation/audit change. |
| `python3 paper/derive_figures.py --check` | PASS: 60 checked derived figures; coverage report 61/708, about 9% | The existing older-paper derivation coverage leaves 647 figures uncovered; not whole-manuscript verification or validation of the September draft's new claims. |
| `python3 scripts/verify_published_numbers.py`, run in BINN | `125/125 published numbers reproduce from the cells.` | Older covered campaign figures. This verifier does not establish W29; W29 was recomputed separately above. |
| `python3 scripts/lr_argmin.py crossover_probe50m` | Completed: MLA selected 0.5×, Mamba2/GDN 1× | One-seed probe, not confirmation of a scaling law. |

Selected tests, invoked directly from the repository's existing assert harness:

```text
gdn_chunked_matches_sequential
gdn_rules_differ_as_documented
mqar_supervises_only_the_query_positions
mqar_is_solvable_by_lookup_and_its_labels_agree_with_its_context
a_resumed_run_continues_the_token_axis_instead_of_restarting_it
a_resume_starts_at_the_step_that_has_not_run_and_replays_nothing
a_board_run_at_each_arms_own_argmin_can_still_be_paired
an_lr_grid_that_bottoms_out_at_its_edge_is_not_reported_as_an_argmin
an_lr_curve_is_read_paired_by_seed_not_by_whichever_seeds_a_point_happens_to_have
```

The existing counter-continuity resume tests pass because they use constant
batches. That success is compatible with the actual-batcher failure reproduced
by this audit; it is not evidence that the failure was fixed.

## Failures and unavailable coverage

The older broad inventory command was attempted with a fresh temporary output
directory and **failed**, rather than regenerating a complete current report:

```text
python3 docs/architecture-review-2026-09-04/evidence/recompute.py --output-dir /private/tmp/mlsystems-deep-audit-20260913-snapshot
RuntimeError: duplicate MQAR cell seed .../mqar_e28_p8/('gdn*3,attention,gdn*3,attention,gdn*3,attention', 8, 3000, 256, 31)
```

The error excerpt abbreviates the absolute path only. Repository and published
GDN rules share a layout string; the older reader fails to distinguish them. Its
partial temporary output was not delivered as a successful regenerated snapshot.
The archived September 4 evidence was not overwritten.

GitPulse's linked-worktree inspection returned `REPOSITORY_TRUST_REQUIRED` for
two worktrees; the collision result reported one scanned and two failed. The main
checkout was inspectable and clean at the start. No trust setting was changed,
and absence of reported overlap was not treated as complete collision clearance.

One ScholarLM cloud discovery search encountered an arXiv-provider HTTP 429 and
limited relevance. A later targeted optimizer search returned useful results.
Primary arXiv/publisher pages were inspected directly for the seven close papers
listed in the report. This is targeted literature coverage, not a full novelty
certification. No obsolete WisDev fallback or external manuscript job was used.

Not run: full repository test suites, GPU numerical equivalence, speed or memory
benchmarks, historical large-checkpoint replay, new scientific training runs,
fleet-platform replication, and missing recall/width-confirmation seeds. They
are unnecessary for verifying the documentation edits and remain explicit gates
for the broader scientific claims. No production/model source fix is included.

## Documentation and provenance checks

- Python syntax for the new probe and strict JSON parsing passed.
- New/added local Markdown targets and heading anchors were checked. Line links
  to existing files were checked for valid line numbers; manuscript/doc citations
  account for the newly added qualification notices.
- Source and input fingerprints matched on the final readback.
- `git diff --check` passed, including separate whitespace checks on the new
  untracked audit artifacts. BINN remained clean.
- The main diff contains documentation notices, navigation updates, the report,
  the executable evidence probe, this log, and its saved JSON. It does not modify
  the model/trainer/readers being audited or remove historical measurements.

No claim of repaired behavior, complete experiment reproduction, or an
attention-scale architectural discovery follows from these checks.
