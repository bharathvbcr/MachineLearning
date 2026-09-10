# September 8 second-audit validation

Scope: [the revised BINN/NanoLab plan](../SECOND_AUDIT_PLAN_2026-09-08.md), its
navigation notes, and a small executable audit probe. Existing training code and
scientific records were not edited. The BINN checkout was read only.

## Executed checks

The [audit probe](second_audit_probe.py) passed all three groups. Its
[saved output](second-audit-results.json) contains runtime versions and source hashes:

1. Current MQAR sampler: four checked earlier-answer input tokens exactly equal
   their gold targets, across two three-query episodes. The source also samples
   query keys without replacement. This is a task-contract check, not evidence of
   invalid benchmark labels.
2. Both actual GDN operators: suppressing beta after a write still decays outputs
   from 1 to 0.5 to 0.25; suppressing decay and beta preserves 1. The distinct
   repository/published two-token outputs also match their stated equations.
3. Exact synthetic enumeration: all 24 temporal permutations for each of two
   order labels yield identical observed-input distributions and Bayes accuracy
   0.5. This is not an SHD accuracy measurement.

Six existing collected functions from `nanolab.tests` were called directly on CPU,
after checking their definitions and the repository's test entry point:

```text
PASS gdn_chunked_matches_sequential
PASS gdn_rules_differ_as_documented
PASS gdn_chunked_survives_odd_lengths_and_tiny_gates
PASS cpu_batcher_windows_are_contiguous_and_shifted
PASS mqar_supervises_only_the_query_positions
PASS mqar_is_solvable_by_lookup_and_its_labels_agree_with_its_context
6/6 selected checks passed; full suite not run
```

The selected GDN checks cover output/input-gradient comparison with the existing
sequential implementation, both recurrence rules, and odd lengths/tiny gates.
They do not validate the proposed new state-snapshot or selective-patching API,
which has not been implemented. The sampler tests confirm the existing task's
supervision and lookup solvability; they do not test the proposed query-only task.

Python AST parsing and strict JSON serialization/parsing passed. Local Markdown
targets were checked in the revised plan, three updated navigation/protocol files,
and this log. A first link pass found only the two references to this log before it
was created; the final pass verifies those references too. Scoped whitespace checks
cover tracked edits and each added text artifact. Source fingerprints in the saved
probe output are compared against the files at delivery.

Final check: **62 local link occurrences, zero broken files or heading anchors;
four source hashes match; Python syntax and JSON pass; tracked diff checks and all
four added-file whitespace checks pass.** BINN remains clean. Three existing
documentation files gained 17 navigation lines; the four new audit artifacts carry
the plan, executable checks, saved outputs, and this validation record.

## Coverage limits

The full NanoLab suite, Rust test suite, CUDA kernels, training, scientific
replication, checkpoint replay, and new intervention code were not run: this change
is an audit and plan revision, with bounded CPU checks of the relevant existing
operators and sampler. No remote queue or artifact store was inspected. The local
checkpoint filename inventory is limited to `nanolab/out`; it cannot establish
absence of suitable weights elsewhere.

Primary literature was checked for targeted methodological context, not for an
exhaustive novelty claim. Historical input memos and registered BINN hypotheses
were preserved. The revised document explicitly supersedes the earlier proposed
execution sequence and distinguishes operator sanity controls from scientific
evidence.
