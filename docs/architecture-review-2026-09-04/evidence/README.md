# Evidence and reproduction

The saved experiment snapshot is dated **2026-09-04 21:48 UTC**. It is not a live
campaign dashboard. These files support the [current assessment](../README.md)
and [protocol](../MEMORY_UPDATE_PROTOCOL.md).

| Artifact | Coverage |
|---|---|
| [evidence-tables.md](evidence-tables.md) | Archived final-CE and MQAR cells. |
| [recomputed-evidence.json](recomputed-evidence.json) | Archived configurations, per-seed summaries, source lines, queues, and parse/nonfinite findings; compact JSON formatting only. |
| [artifact-manifest.json](artifact-manifest.json) | 3,463 original result paths and SHA-256 hashes; not 3,463 independent experiments. |
| [memo-curve-check.json](memo-curve-check.json) | Periodic-evaluation paired curves and their scope, distinct from final CE. |
| [source-fingerprints.json](source-fingerprints.json) | Source fingerprints at the original review; paths are repository-relative. |
| [router-gradient-results.json](router-gradient-results.json) | Actual-module CPU router-gradient diagnostic. |
| [gdn-rule-results.json](gdn-rule-results.json) | Actual chunked recurrence versus published update diagnostic. |
| [math-check-results.json](math-check-results.json) | Independent scalar and projected-update checks, not the supplied 124/200 trajectory. |
| [verification-notes.txt](verification-notes.txt) | Original review's command outputs and missing-data boundaries; absolute paths are historical provenance. |
| [input-provenance.json](input-provenance.json) | Hashes of supplied texts and description of import transformations. |
| [DOCS_VALIDATION.md](DOCS_VALIDATION.md) | Checks run when integrating the documentation, with current-versus-archived count boundaries. |

JSON formatting and repository path portability do not change the recorded
measurements. A fresh run against changing local artifacts will have a different
timestamp and potentially different counts. Preserve the archived snapshot when
recording new results. Missing source files on another checkout are a coverage
limitation, not evidence of successful reproduction.

## Commands

Run from the repository root. The math and curve checks use Python's standard
library. The two actual-model probes require the existing PyTorch environment;
they install nothing, use CPU, and do not train or modify model source.

```bash
python3 docs/architecture-review-2026-09-04/evidence/memory_geometry_probe.py
python3 docs/architecture-review-2026-09-04/evidence/verify_curves.py
python3 docs/architecture-review-2026-09-04/evidence/router_gradient_probe.py
python3 docs/architecture-review-2026-09-04/evidence/gdn_rule_probe.py
```

The broad inventory script writes only to a **new** output directory and refuses
to overwrite an existing one. The example path must not already exist:

```bash
python3 docs/architecture-review-2026-09-04/evidence/recompute.py --output-dir /tmp/mlsystems-fresh-architecture-review
```

This regenerates a manifest, full recomputed JSON, and evidence tables from the
current local records. It does not deserialize checkpoints or rerun training.

The paper's existing verification command is:

```bash
python3 paper/derive_figures.py --check
```

The original review recorded 60 checked derived figures, with 61/706 figures
derived or documented (9%). Its success is not whole-manuscript validation.
The research docs added here do not alter the manuscript or its derived figures.

## Interpretation limits

- The mathematical probes validate formulas on stated fixtures. They do not
  establish trained-model quality, BF16 behavior, or hardware efficiency.
- The independent projection fixtures use eight orthonormal witnesses in twelve
  key dimensions and six value dimensions, including separately checked boundary
  cases. They are not the missing original trajectory.
- The curve check compares all five seed trajectories at shared periodic markers.
  Pointwise t intervals are exploratory, not simultaneous bands or confirmation
  after independent architecture selection.
- The GDN and router probes characterize the archived source behavior. If source
  is intentionally corrected, preserve these outputs and introduce a separately
  identified comparator; do not rewrite historical results to make a check pass.
- BINN raw W26 cells and the claimed 124/200 audit bundle were unavailable in the
  provided material. No result file here pretends to reconstruct them.
- *2026-09-05:* the W26 cells (and W27's) are tracked in BINN's own repository under
  `results/shd_attention_campaign_v3/`; they were outside this review's snapshot, not
  missing from the record. Nothing here was recomputed from them.
