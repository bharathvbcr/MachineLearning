"""Compare the live chunked recurrence to the published decayed-read rule.

Reference: Gated Delta Networks, arXiv:2412.06464, Table 1 / section 3.1.
No source model or experiment artifact is changed.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanolab.mixers import gdn_chunked

torch.set_num_threads(1)
results = []
for alpha_value, beta_value in [(0.5, 0.5), (1.0, 0.5), (0.5, 0.0)]:
    q = torch.ones(1, 1, 2, 1)
    k, v = q.clone(), q.clone()
    alpha = torch.full((1, 1, 2), alpha_value)
    beta = torch.full((1, 1, 2), beta_value)
    actual = gdn_chunked(q, k, v, alpha, beta, chunk=2).flatten().tolist()
    state, reference = 0.0, []
    for _ in range(2):
        state *= alpha_value
        state += beta_value * (1.0 - state)
        reference.append(state)
    delta = max(abs(a - b) for a, b in zip(actual, reference))
    if alpha_value == 0.5 and beta_value == 0.5:
        assert actual == [0.5, 0.5]
        assert reference == [0.5, 0.625]
        assert delta == 0.125
    else:
        assert delta < 1e-7
    results.append({'alpha': alpha_value, 'beta': beta_value,
                    'repository_chunked_outputs': actual,
                    'published_decayed_read_outputs': reference,
                    'max_difference': delta})
print(json.dumps(results, indent=2))
