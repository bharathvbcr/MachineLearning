"""Small counterexamples for the September 8 experiment-plan audit.

Calls the current MQAR sampler and GDN operator; does not train a model or
change archived results. The order-label example is an exhaustive synthetic
calculation, not an SHD measurement. Run from any directory with the existing
NanoLab Python environment. JSON is written to stdout.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from nanolab.mixers import gdn_chunked
from nanolab.mqar import MQARBatcher


def check_mqar_input() -> dict:
    cfg = SimpleNamespace(
        mqar_n_pairs=3, mqar_n_queries=3, mqar_n_keys=8,
        mqar_n_values=8, batch_size=2, block_size=11, seed=7,
    )
    batcher = MQARBatcher(cfg, "cpu")
    x, y = batcher.batch()
    query_positions = list(range(6, 11, 2))
    prior_answer_positions = [position + 1 for position in query_positions[:-1]]
    exposed = x[:, prior_answer_positions]
    gold = y[:, query_positions[:-1]]
    assert torch.equal(exposed, gold)
    query_keys = x[:, query_positions]
    unique = all(len(set(row)) == cfg.mqar_n_queries for row in query_keys.tolist())
    assert unique
    return {
        "batch_shape": list(x.shape),
        "query_positions": query_positions,
        "earlier_gold_answers_in_input_per_episode": len(prior_answer_positions),
        "checked_answer_tokens": exposed.numel(),
        "all_checked_tokens_equal_previous_gold_targets": True,
        "queries_without_replacement_in_this_fixture": unique,
        "interpretation": "Current MQAR is teacher-forced and is not a query-only task.",
    }


def check_gdn_controls() -> dict:
    q = torch.ones(1, 1, 3, 1)
    k, v = q.clone(), q.clone()
    # First token stores one; the following two tokens have no delta writes.
    beta = torch.tensor([[[1.0, 0.0, 0.0]]])
    decay = torch.tensor([[[1.0, 0.5, 0.5]]])
    identity = torch.ones_like(decay)
    rows = {}
    for rule in ("repo", "published"):
        beta_only = gdn_chunked(q, k, v, decay, beta, chunk=3, rule=rule)
        full_freeze = gdn_chunked(q, k, v, identity, beta, chunk=3, rule=rule)
        expected_decay = torch.tensor([1.0, 0.5, 0.25])
        expected_identity = torch.ones(3)
        torch.testing.assert_close(beta_only.flatten(), expected_decay, rtol=0, atol=1e-7)
        torch.testing.assert_close(full_freeze.flatten(), expected_identity, rtol=0, atol=1e-7)
        rows[rule] = {
            "disable_beta_only": beta_only.flatten().tolist(),
            "disable_decay_and_beta": full_freeze.flatten().tolist(),
        }
    two = q[:, :, :2].contiguous()
    gate = torch.full((1, 1, 2), 0.5)
    repo = gdn_chunked(two, two, two, gate, gate, chunk=2, rule="repo")
    published = gdn_chunked(two, two, two, gate, gate, chunk=2, rule="published")
    torch.testing.assert_close(repo.flatten(), torch.tensor([0.5, 0.5]), rtol=0, atol=1e-7)
    torch.testing.assert_close(published.flatten(), torch.tensor([0.5, 0.625]), rtol=0, atol=1e-7)
    return {
        "controls": rows,
        "rule_check": {"repo": repo.flatten().tolist(), "published": published.flatten().tolist()},
        "interpretation": "Disabling the delta write does not disable decay. Both rules exist and differ.",
    }


def check_destroyed_order_label() -> dict:
    # A before B is one label; B before A is the other. Counts are identical.
    examples = [(1, 2, 0, 0), (2, 1, 0, 0)]
    orders = list(permutations(range(4)))
    histograms = [Counter(tuple(example[i] for i in order) for order in orders)
                  for example in examples]
    assert histograms[0] == histograms[1]
    total = 2 * len(orders)
    optimum_correct = sum(max(histograms[0][x], histograms[1][x])
                          for x in histograms[0].keys() | histograms[1].keys())
    assert optimum_correct / total == 0.5
    assert all(tuple(reversed(tuple(reversed(x)))) == x for x in examples)
    return {
        "permutations_per_label": len(orders),
        "distinct_observed_sequences": len(histograms[0]),
        "conditional_input_distributions_identical": True,
        "balanced_bayes_accuracy_after_unobserved_random_permutation": optimum_correct / total,
        "reversal_is_invertible": True,
        "interpretation": "On this synthetic task, unrecoverable labels alone force chance accuracy after full random shuffling; no architecture defect is required.",
    }


def main() -> None:
    torch.set_num_threads(1)
    watched = ["nanolab/mqar.py", "nanolab/mixers.py", "nanolab/mqar_suite.py", "nanolab/model.py"]
    result = {
        "status": "PASS",
        "scope": "sampler/operator counterexamples and exact finite enumeration; no training",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in watched},
        "mqar": check_mqar_input(),
        "gdn": check_gdn_controls(),
        "synthetic_order": check_destroyed_order_label(),
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
