"""Exercise the repository's existing MoE on CPU; change no research code."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nanolab.model import MoE

torch.set_num_threads(1)
results = []
for top_k in (1, 2):
    torch.manual_seed(17)
    model = MoE(SimpleNamespace(d_model=16, moe_experts=4, moe_top_k=top_k)).double()
    inputs = torch.randn(2, 9, 16, dtype=torch.float64)
    targets = torch.randint(0, 16, (18,))
    outputs = model(inputs)
    task_loss = F.cross_entropy(outputs.reshape(-1, 16), targets)
    task_loss.backward()
    task_gradient = model.gate.weight.grad.norm().item()
    expert_gradient = sum(p.grad.norm().item() for e in model.experts
                          for p in e.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    model(inputs)
    model.aux.backward()
    aux_gradient = model.gate.weight.grad.norm().item()
    assert expert_gradient > 1e-5
    assert aux_gradient > 1e-5
    if top_k == 1:
        assert task_gradient < 1e-12
    else:
        assert task_gradient > 1e-5
    results.append({'top_k': top_k, 'task_router_grad_norm': task_gradient,
                    'task_expert_grad_norm_sum': expert_gradient,
                    'aux_router_grad_norm': aux_gradient})
print(json.dumps(results, indent=2))
