"""
nanolab.optim — the optimizer spectrum (guide §4).

In order of "how much they do for you" (§4.2):
  SGD+momentum < AdamW < Lion < Schedule-Free AdamW < Muon

The whole point is the *bake-off* (§4.5, §6.3): identical seed/data/tokens,
swap only the optimizer, and read the curves. ``build_optimizers`` returns a
list so the Muon-hybrid (Muon on 2D hidden weights, AdamW on everything else,
§4.4) is just another entry in that list.

Param grouping follows the standard nanoGPT pattern (§4.3): decay 2D weights,
do NOT decay biases / norms / 1D params; embeddings, the LM head, norms, biases
and scalars stay on AdamW even in the Muon path.
"""

from __future__ import annotations

import math

import torch


POLAR_EXPRESS_COEFFICIENTS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization + Muon (guide §4.4)
# ---------------------------------------------------------------------------
@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Quintic Newton-Schulz iteration that orthogonalizes G (its singular
    values are pushed toward 1). Coefficients from Keller Jordan's Muon.

    Batch-aware: accepts a 2D matrix OR a stack ``(B, M, N)`` of same-shape
    matrices, so Muon can orthogonalize all same-shape layer weights in ONE set
    of batched matmuls instead of one launch per parameter (guide §7.1: the win
    on a single GPU is killing kernel-launch overhead)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


@torch.no_grad()
def zeropower_via_polar_express(G: torch.Tensor, eps: float = 1e-6):
    """Five-matmul-stage Polar Express reference used by the research oracle.

    The coefficients are iteration-specific, unlike Muon's repeated NS5
    polynomial.  Keeping this independent is important: mapping this arm to
    NS5 would invalidate an equal-data optimizer comparison.
    """
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + eps)
    for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


def _orthogonalize(update: torch.Tensor, method: str, ns_steps: int):
    if method == "ns":
        return zeropower_via_newtonschulz5(update, steps=ns_steps)
    if method == "polar":
        return zeropower_via_polar_express(update)
    raise ValueError(f"unknown Muon orthogonalizer {method}")


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-Schulz — ~2x more compute-efficient
    than AdamW at scale (guide §4.4). Single-GPU reference implementation.

    Apply ONLY to 2D hidden-layer weights; keep embeddings/head/norms/biases on
    AdamW (handled by build_optimizers)."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0, orthogonalizer="ns"):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        orthogonalizer=orthogonalizer)
        super().__init__(params, defaults)

    def _shape_buckets(self, params):
        """Group params by 2D shape so all same-shape weights can be
        orthogonalized in one batched Newton-Schulz call. Cached across steps."""
        buckets = {}
        for p in params:
            if p.grad is None:
                continue
            shape = (p.size(0), p.numel() // p.size(0))
            buckets.setdefault(shape, []).append(p)
        return buckets

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            nesterov, ns_steps = group["nesterov"], group["ns_steps"]
            orthogonalizer = group["orthogonalizer"]
            wd = group["weight_decay"]
            for (m, n), plist in self._shape_buckets(group["params"]).items():
                # accumulate momentum + build the (B, m, n) update stack
                updates = []
                for p in plist:
                    g = p.grad.view(m, n)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(mom).add_(g)
                    updates.append(g.add(buf, alpha=mom) if nesterov else buf)
                stack = torch.stack(updates, dim=0)               # (B, m, n)
                stack = _orthogonalize(stack, orthogonalizer, ns_steps)
                scale = max(1.0, m / n) ** 0.5                     # RMS match
                for p, upd in zip(plist, stack.unbind(0)):
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(upd.view_as(p), alpha=-lr * scale)
        return loss


class NorMuon(torch.optim.Optimizer):
    """Single-device, same-shape-batched NorMuon reference.

    State per matrix is one full momentum plus one row-wise second moment,
    rather than Adam's two full-sized tensors.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, beta2=0.95,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, beta, beta2 = group["lr"], group["momentum"], group["beta2"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.reshape(p.size(0), -1)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["second_moment"] = torch.zeros_like(g[..., :1])
                mom, second = state["momentum_buffer"], state["second_moment"]
                mom.lerp_(g, 1 - beta)
                update = g.lerp(mom, beta)
                update = zeropower_via_newtonschulz5(update, group["ns_steps"]).to(g.dtype)
                norm_before = update.norm()
                row_ms = update.square().mean(dim=-1, keepdim=True)
                second.lerp_(row_ms, 1 - beta2)
                update.mul_(second.sqrt().add_(1e-10).reciprocal())
                update.mul_(norm_before / update.norm().add_(1e-10))
                update.mul_(max(1.0, g.size(0) / g.size(1)) ** 0.5)
                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                p.add_(update.reshape_as(p), alpha=-lr)
        return loss


class MONA(torch.optim.Optimizer):
    """Muon with the paper's EMA-of-gradient-differences acceleration."""

    def __init__(self, params, lr=0.02, momentum=0.95, beta_a=0.99,
                 alpha=None, ns_steps=5, weight_decay=0.0):
        alpha = -1.0 / (2.0 * (1.0 - beta_a)) if alpha is None else alpha
        defaults = dict(lr=lr, momentum=momentum, beta_a=beta_a, alpha=alpha,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.reshape(p.size(0), -1)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["acceleration"] = torch.zeros_like(g)
                    state["prev_grad"] = g.clone()
                mom = state["momentum_buffer"]
                acc = state["acceleration"]
                diff = g - state["prev_grad"]
                acc.mul_(group["beta_a"]).add_(diff, alpha=1 - group["beta_a"])
                transformed = g + group["alpha"] * acc
                mom.mul_(group["momentum"]).add_(transformed)
                update = transformed + group["momentum"] * mom
                update = zeropower_via_newtonschulz5(update, group["ns_steps"])
                update.mul_(max(1.0, g.size(0) / g.size(1)) ** 0.5)
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape_as(p), alpha=-group["lr"])
                state["prev_grad"].copy_(g)
        return loss


class MiMuon(torch.optim.Optimizer):
    """Research oracle for singular-gap routed Muon/SGD mixing.

    Exact SVD is intentional here: this path establishes semantics.  A native
    approximation is not eligible until it matches this deterministic oracle.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, singular_gap=1e-3,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, singular_gap=singular_gap,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.reshape(p.size(0), -1)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                mom = state["momentum_buffer"]
                mom.mul_(group["momentum"]).add_(g)
                s = torch.linalg.svdvals(mom.float())
                gap = torch.inf if s.numel() < 2 else (s[:-1] - s[1:]).abs().min()
                if gap >= group["singular_gap"]:
                    update = zeropower_via_newtonschulz5(mom, group["ns_steps"])
                    update.mul_(max(1.0, g.size(0) / g.size(1)) ** 0.5)
                    state["used_muon"] = state.get("used_muon", 0) + 1
                else:
                    update = mom
                    state["used_sgd"] = state.get("used_sgd", 0) + 1
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape_as(p), alpha=-group["lr"])
        return loss


class Muown(torch.optim.Optimizer):
    """Row-magnitude/direction Muown reference (paper Algorithm 1)."""

    def __init__(self, params, lr=0.02, momentum=0.95, betas=(0.9, 0.95),
                 eps=1e-8, direction_scale=0.2, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, betas=betas, eps=eps,
                        direction_scale=direction_scale, ns_steps=ns_steps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                W = p.reshape(p.size(0), -1)
                grad = p.grad.reshape_as(W)
                state = self.state[p]
                if "direction" not in state:
                    state["direction"] = W.clone()
                    state["m_direction"] = torch.zeros_like(W)
                    state["m_mag"] = torch.zeros_like(W[..., :1])
                    state["v_mag"] = torch.zeros_like(W[..., :1])
                    state["step"] = 0
                R = state["direction"]
                r = R.norm(dim=-1, keepdim=True).clamp_min_(group["eps"])
                D = R / r
                gmag = W.norm(dim=-1, keepdim=True)
                radial = (grad * D).sum(dim=-1, keepdim=True)
                tangent = grad - D * radial
                grad_R = (gmag / r) * tangent
                md = state["m_direction"]
                md.mul_(group["momentum"]).add_(grad_R)
                nesterov = grad_R + group["momentum"] * md
                ortho = zeropower_via_newtonschulz5(nesterov, group["ns_steps"])
                scale = group["direction_scale"] * max(W.shape) ** 0.5
                R.add_(ortho, alpha=-group["lr"] * scale)
                state["step"] += 1
                mm, vm = state["m_mag"], state["v_mag"]
                mm.mul_(b1).add_(radial, alpha=1 - b1)
                vm.mul_(b2).addcmul_(radial, radial, value=1 - b2)
                mm_hat = mm / (1 - b1 ** state["step"])
                vm_hat = vm / (1 - b2 ** state["step"])
                gmag.addcdiv_(mm_hat, vm_hat.sqrt().add_(group["eps"]),
                              value=-group["lr"])
                new_r = R.norm(dim=-1, keepdim=True).clamp_min_(group["eps"])
                W.copy_((gmag / new_r) * R)
                if group["weight_decay"]:
                    W.mul_(1 - group["lr"] * group["weight_decay"])
                    gmag.copy_(W.norm(dim=-1, keepdim=True))
        return loss


# ---------------------------------------------------------------------------
# Lion — sign(momentum) update, memory-light (guide §4.2)
# ---------------------------------------------------------------------------
class Lion(torch.optim.Optimizer):
    """EvoLved Sign Momentum. Use ~3-10x smaller LR and higher WD than AdamW."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (b1, b2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                if wd:
                    p.mul_(1 - lr * wd)
                update = m.mul(b1).add(p.grad, alpha=1 - b1).sign_()
                p.add_(update, alpha=-lr)
                m.mul_(b2).add_(p.grad, alpha=1 - b2)
        return loss


def _cautious_mask(update: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Official C-Optim mask, normalized to preserve expected step magnitude."""
    mask = (update * grad > 0).to(update.dtype)
    return mask / mask.mean().clamp_min(1e-3)


class CautiousLion(Lion):
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (b1, b2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                if wd:
                    p.mul_(1 - lr * wd)
                update = m.mul(b1).add(p.grad, alpha=1 - b1).sign_()
                p.add_(update * _cautious_mask(update, p.grad), alpha=-lr)
                m.mul_(b2).add_(p.grad, alpha=1 - b2)
        return loss


class CautiousAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                bc1 = 1 - b1 ** state["step"]
                bc2 = 1 - b2 ** state["step"]
                denom = v.sqrt().div_(bc2 ** 0.5).add_(group["eps"])
                update = m * _cautious_mask(m, p.grad)
                p.addcdiv_(update, denom, value=-group["lr"] / bc1)
        return loss


class SOAP(torch.optim.Optimizer):
    """Small, independent SOAP oracle following the official implementation.

    It stores full per-axis covariance/eigenvector matrices only when an axis is
    below ``max_precond_dim``.  ``estimated_state_bytes`` lets the funnel apply
    its memory gate before constructing an exact-scale optimizer.
    """

    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), eps=1e-8,
                 weight_decay=0.0, precondition_frequency=10,
                 max_precond_dim=2048):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        precondition_frequency=precondition_frequency,
                        max_precond_dim=max_precond_dim)
        super().__init__(params, defaults)

    @staticmethod
    def estimated_state_bytes(params, max_precond_dim=2048, bytes_per_value=4):
        total = 0
        for p in params:
            total += 2 * p.numel() * bytes_per_value
            for d in p.shape:
                if d <= max_precond_dim:
                    total += 2 * d * d * bytes_per_value
        return total

    def _init_state(self, p, state, max_dim):
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p)
        state["exp_avg_sq"] = torch.zeros_like(p)
        state["cov"] = [
            torch.zeros((d, d), device=p.device, dtype=torch.float32)
            if d <= max_dim else None for d in p.shape
        ]
        state["basis"] = None

    @staticmethod
    def _project(x, bases, back=False):
        out = x
        for basis in bases:
            if basis is not None:
                out = torch.tensordot(out, basis, dims=([0], [1 if back else 0]))
            else:
                out = out.permute(*range(1, out.ndim), 0)
        return out

    @staticmethod
    def _update_covariances(grad, cov, beta):
        for axis, matrix in enumerate(cov):
            if matrix is None:
                continue
            dims = [i for i in range(grad.ndim) if i != axis]
            outer = torch.tensordot(grad.float(), grad.float(), dims=(dims, dims))
            matrix.lerp_(outer, 1 - beta)

    @staticmethod
    def _eigenbases(cov):
        out = []
        for matrix in cov:
            if matrix is None:
                out.append(None)
            else:
                eye = torch.eye(matrix.shape[0], device=matrix.device)
                _, q = torch.linalg.eigh(matrix + 1e-30 * eye)
                out.append(torch.flip(q, dims=[1]))
        return out

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    self._init_state(p, state, group["max_precond_dim"])
                    self._update_covariances(p.grad, state["cov"], b2)
                    state["basis"] = self._eigenbases(state["cov"])
                    continue  # official SOAP deliberately skips its first step
                projected = self._project(p.grad, state["basis"])
                state["step"] += 1
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(b1).add_(projected, alpha=1 - b1)
                v.mul_(b2).addcmul_(projected, projected, value=1 - b2)
                bc1, bc2 = 1 - b1 ** state["step"], 1 - b2 ** state["step"]
                pre = m / v.sqrt().add_(group["eps"])
                update = self._project(pre, state["basis"], back=True)
                p.add_(update, alpha=-group["lr"] * bc2 ** 0.5 / bc1)
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                self._update_covariances(p.grad, state["cov"], b2)
                if state["step"] % group["precondition_frequency"] == 0:
                    # Reproject the first moment into the refreshed eigenbasis.
                    original_m = self._project(m, state["basis"], back=True)
                    state["basis"] = self._eigenbases(state["cov"])
                    m.copy_(self._project(original_m, state["basis"]))
        return loss


# ---------------------------------------------------------------------------
# Sophia — Adam + a cheap, clipped diagonal curvature estimate (§4.2). A taste
# of second-order. The update ``clip(m / max(rho*h, eps), 1)`` preconditions
# momentum by a clipped diagonal estimate, refreshed every k steps.
# ---------------------------------------------------------------------------
class Sophia(torch.optim.Optimizer):
    """Sophia. The update is ``clip(m / max(rho * h, eps), 1)`` — momentum
    preconditioned by a clipped curvature estimate, so flat directions move at
    full step and sharp ones are damped (guide §4.2). Between Hessian refreshes
    it behaves like clipped momentum (≈ a damped Lion).

    ``update_hessian()`` here uses the **empirical-Fisher** diagonal (E[g²] of the
    minibatch gradient) — a cheap, label-free approximation of the Gauss-Newton
    diagonal. (The paper's Sophia-G resamples labels from the model for an
    unbiased GNB estimate; we trade that bias for not needing an extra forward.)"""

    def __init__(self, params, lr=1e-3, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=0.1, eps=1e-12):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self):
        """EMA of the squared minibatch gradient (empirical-Fisher diagonal).
        Call right after a backward, before the gradients are cleared."""
        for group in self.param_groups:
            b2 = group["betas"][1]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p)
                state["hessian"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, _ = group["betas"]
            lr, rho, wd, eps = group["lr"], group["rho"], group["weight_decay"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)
                m, h = state["m"], state["hessian"]
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                if wd:
                    p.mul_(1 - lr * wd)
                denom = (rho * h).clamp_min(eps)
                update = (m / denom).clamp_(-1.0, 1.0)        # per-coordinate clip
                p.add_(update, alpha=-lr)
        return loss


# ---------------------------------------------------------------------------
# Prodigy — learning-rate-free (D-Adaptation successor). Estimates the step size
# internally, so you can drop the schedule entirely (guide §4.5).
# ---------------------------------------------------------------------------
class Prodigy(torch.optim.Optimizer):
    """Prodigy (Mishchenko & Defazio). ``lr`` is a fixed multiplier (keep =1);
    the effective step size ``d`` is estimated online from the optimization
    trajectory. Run it head-to-head against tuned AdamW+cosine to *see* what the
    schedule was doing (guide §4.5)."""

    def __init__(self, params, lr=1.0, betas=(0.9, 0.99), beta3=None, eps=1e-8,
                 weight_decay=0.0, d0=1e-6):
        beta3 = beta3 if beta3 is not None else betas[1] ** 0.5
        defaults = dict(lr=lr, betas=betas, beta3=beta3, eps=eps,
                        weight_decay=weight_decay, d=d0, d0=d0, d_max=d0,
                        d_numerator=0.0, k=0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        g0 = self.param_groups[0]
        b1, b2 = g0["betas"]
        beta3, eps, lr = g0["beta3"], g0["eps"], g0["lr"]
        d, d0, d_max, k = g0["d"], g0["d0"], g0["d_max"], g0["k"]
        # bias correction (as in the official Prodigy)
        bc = (1 - b2 ** (k + 1)) ** 0.5 / (1 - b1 ** (k + 1))
        dlr = d * lr * bc
        # accumulate the d numerator/denominator on-device and sync ONCE at the
        # end (a per-parameter .item() here would force ~100 GPU syncs/step).
        dev = g0["params"][0].device
        d_numer_t = torch.zeros((), device=dev)
        d_denom_t = torch.zeros((), device=dev)

        for g in self.param_groups:
            wd = g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["s"] = torch.zeros_like(p)
                    state["p0"] = p.detach().clone()
                m, v, s, p0 = state["m"], state["v"], state["s"], state["p0"]
                # d numerator/denominator are scaled by (d/d0) — the growth of
                # the estimate relative to the fixed initial d0 (the Prodigy fix).
                d_numer_t += (d / d0) * dlr * torch.dot(
                    grad.flatten(), (p0 - p).flatten())
                m.mul_(b1).add_(grad, alpha=d * (1 - b1))
                v.mul_(b2).addcmul_(grad, grad, value=d * d * (1 - b2))
                s.mul_(beta3).add_(grad, alpha=(d / d0) * dlr)
                d_denom_t += s.abs().sum()
                if wd:
                    p.mul_(1 - dlr * wd)

        d_numer = g0["d_numerator"] * beta3 + d_numer_t.item()   # single sync
        d_denom = d_denom_t.item()
        if d_denom > 0:
            d_hat = d_numer / ((1 - beta3) * d_denom)
            d = max(d, d_hat)
            d_max = max(d_max, d_hat)
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                denom = state["v"].sqrt().add_(d * eps)
                p.addcdiv_(state["m"], denom, value=-dlr)
        for g in self.param_groups:
            g.update(d=d, d_max=d_max, d_numerator=d_numer, k=k + 1)
        return loss


# ---------------------------------------------------------------------------
# Schedule-Free AdamW — estimates the step size internally; drop the schedule
# entirely and watch what the cosine was doing for you (guide §4.5).
# Faithful single-file port of Defazio et al. "The Road Less Scheduled".
# ---------------------------------------------------------------------------
class ScheduleFreeAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=2.5e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0, warmup_steps=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        warmup_steps=warmup_steps, k=0, weight_sum=0.0)
        super().__init__(params, defaults)
        self._train = True

    @torch.no_grad()
    def _swap(self, to_eval: bool):
        # Schedule-Free evaluates at x (the averaged iterate), trains at y.
        for group in self.param_groups:
            b1 = group["betas"][0]
            for p in group["params"]:
                state = self.state[p]
                if "z" not in state:
                    continue
                # y = (1-b1) z + b1 x   <->   x = (y - (1-b1) z) / b1
                if to_eval:
                    p.lerp_(state["z"], weight=1 - 1 / b1)  # -> x
                else:
                    p.lerp_(state["z"], weight=1 - b1)      # -> y

    def train(self):
        if not self._train:
            self._swap(to_eval=False)
            self._train = True

    def eval(self):
        if self._train:
            self._swap(to_eval=True)
            self._train = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            k = group["k"]
            warmup = group["warmup_steps"]
            sched = min(1.0, (k + 1) / warmup) if warmup > 0 else 1.0
            lr_eff = lr * sched
            # weight for the x average (Defazio polynomial weighting)
            weight = (k + 1) ** 2
            group["weight_sum"] += weight
            ckp1 = weight / group["weight_sum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "z" not in state:
                    state["z"] = p.detach().clone()
                    state["v"] = torch.zeros_like(p)
                z, v = state["z"], state["v"]
                grad = p.grad
                if wd:
                    grad = grad.add(p, alpha=wd)
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                denom = v.sqrt().add_(eps)
                # x lives implicitly in p (we keep p = y); update z then re-form y
                z.addcdiv_(grad, denom, value=-lr_eff)
                # y <- (1-b1) z + b1 x ; with x updated by the running average:
                x_new = p.mul(1 - ckp1).add_(z, alpha=ckp1)  # p becomes x
                p.copy_(x_new.mul(b1).add_(z, alpha=1 - b1))  # p becomes new y
            group["k"] = k + 1
        return loss


# ---------------------------------------------------------------------------
# Param grouping + optimizer construction
# ---------------------------------------------------------------------------
def _split_params(model):
    """Return (matrix_2d_hidden, embed_head, scalars_1d). Embeddings and the LM
    head are kept separate so they never go to Muon (guide §4.4)."""
    matrix, embed_head, scalar = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "tok_emb" in n or "pos_emb" in n or "lm_head" in n:
            embed_head.append(p)
        elif p.ndim >= 2:
            matrix.append(p)
        else:
            scalar.append(p)
    return matrix, embed_head, scalar


def build_optimizers(model, cfg):
    """Build the optimizer(s) for ``cfg.optimizer``. Returns a list because the
    Muon path is a hybrid (Muon + AdamW). Train loop steps every entry."""
    matrix, embed_head, scalar = _split_params(model)
    opt = cfg.optimizer

    # μP (§10): hidden-layer LR scales as 1/width_mult so the LR tuned at the base
    # width transfers to wider models. Embeddings/scalars keep the base LR.
    width_mult = getattr(model, "width_mult", 1.0)
    hidden_lr = cfg.lr / width_mult
    matrix_lr = cfg.matrix_lr / width_mult

    # ---- per-layer standard parametrization (Everett et al., arXiv:2407.05872) ----
    # Table 1, "Adam LR" column for the Standard row: embedding LR is width-constant,
    # hidden and readout LRs scale as 1/sqrt(n). This is a *standard* parametrization
    # prescription, so it uses the width ratio directly rather than model.width_mult,
    # which is 1.0 whenever cfg.mup is False.
    #
    # Two caveats, both real and neither resolvable here:
    #
    #   1. The prescription is derived for pure Adam. Our Muon-family runs are a
    #      hybrid: 2-D hidden matrices go to Muon (whose update is normalized by
    #      construction) and only embeddings/scalars go to AdamW. Everett et al. give
    #      no exponent for that combination, so applying the Adam hidden exponent to
    #      the Muon group is an extrapolation, not their result. Prefer
    #      ``optimizer="adamw"`` when the point is to reproduce their prescription.
    #   2. ``_split_params`` groups tok_emb, pos_emb and lm_head together, and with
    #      cfg.tie_embeddings the embedding and readout are the SAME tensor. Their
    #      table gives these different exponents (constant vs 1/sqrt(n)); we cannot
    #      honour both. We apply the embedding rule (constant) to that group, because
    #      under tying the input embedding dominates the parameter count.
    if cfg.per_layer_sp:
        if cfg.mup:
            raise ValueError(
                "per_layer_sp and mup are different parametrizations; enable at most one")
        width_ratio = (cfg.d_model / cfg.mup_base_width) if cfg.mup_base_width else 1.0
        sp_scale = math.sqrt(width_ratio) if width_ratio > 0 else 1.0
        hidden_lr = cfg.lr / sp_scale
        matrix_lr = cfg.matrix_lr / sp_scale

    # ---- embedding-LR-only ablation (Kalra & Barkeshli, arXiv:2605.21486) ----
    # Raise ONLY the embedding/head LR and change nothing else, to test whether that
    # single layer carries the advantage attributed to the parametrization. Applies
    # on top of whichever parametrization is active; the default 1.0 is a no-op.
    embed_lr = cfg.lr * cfg.embed_lr_mult

    muon_family = {
        "muon_ns5_adamw", "muon_ns3_adamw", "muon_polar_adamw",
        "normuon_adamw", "muown_adamw", "mona_adamw", "mimuon_adamw",
    }
    if opt in muon_family:
        if opt == "muon_ns5_adamw":
            opt_muon = Muon(matrix, lr=matrix_lr, momentum=cfg.muon_momentum,
                            ns_steps=5, weight_decay=cfg.weight_decay)
        elif opt == "muon_ns3_adamw":
            opt_muon = Muon(matrix, lr=matrix_lr, momentum=cfg.muon_momentum,
                            ns_steps=3, weight_decay=cfg.weight_decay)
        elif opt == "muon_polar_adamw":
            opt_muon = Muon(matrix, lr=matrix_lr, momentum=cfg.muon_momentum,
                            orthogonalizer="polar", weight_decay=cfg.weight_decay)
        elif opt == "normuon_adamw":
            opt_muon = NorMuon(matrix, lr=matrix_lr, momentum=cfg.muon_momentum,
                               beta2=cfg.normuon_beta2, ns_steps=cfg.muon_ns_steps,
                               weight_decay=cfg.weight_decay)
        elif opt == "muown_adamw":
            opt_muon = Muown(matrix, lr=matrix_lr, momentum=cfg.muon_momentum,
                             betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
                             direction_scale=cfg.muown_direction_scale,
                             ns_steps=cfg.muon_ns_steps, weight_decay=cfg.weight_decay)
        elif opt == "mona_adamw":
            alpha = None if cfg.mona_alpha == 0.0 else cfg.mona_alpha
            opt_muon = MONA(matrix, lr=matrix_lr, momentum=cfg.muon_momentum,
                            beta_a=cfg.mona_beta_a, alpha=alpha,
                            ns_steps=cfg.muon_ns_steps, weight_decay=cfg.weight_decay)
        else:
            opt_muon = MiMuon(matrix, lr=matrix_lr, momentum=cfg.muon_momentum,
                              singular_gap=cfg.mimuon_singular_gap,
                              ns_steps=cfg.muon_ns_steps, weight_decay=cfg.weight_decay)
        opt_adam = torch.optim.AdamW(
            [{"params": embed_head, "weight_decay": 0.0, "lr": embed_lr},
             {"params": scalar, "weight_decay": 0.0, "lr": cfg.lr}],
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
        return [opt_muon, opt_adam]

    # non-Muon: standard decay / no-decay split (guide §4.3). Under μP the hidden
    # matrices use hidden_lr (= lr/width); embeddings/head/scalars keep base lr.
    decay = matrix + embed_head
    no_decay = scalar
    if opt == "adamw":
        return [torch.optim.AdamW(
            [{"params": matrix, "weight_decay": cfg.weight_decay, "lr": hidden_lr},
             {"params": embed_head, "weight_decay": cfg.weight_decay, "lr": embed_lr},
             {"params": no_decay, "weight_decay": 0.0, "lr": cfg.lr}],
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)]
    if opt == "sgd_momentum":
        return [torch.optim.SGD(decay + no_decay, lr=cfg.lr, momentum=0.9)]
    if opt == "lion":
        return [Lion(
            [{"params": decay, "weight_decay": cfg.weight_decay * 3},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr, betas=(0.9, 0.99))]
    if opt == "cautious_adamw":
        return [CautiousAdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)]
    if opt == "cautious_lion":
        return [CautiousLion(
            [{"params": decay, "weight_decay": cfg.weight_decay * 3},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr, betas=(0.9, 0.99))]
    if opt == "schedule_free_adamw":
        return [ScheduleFreeAdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), warmup_steps=cfg.warmup_steps)]
    if opt == "sophia":
        return [Sophia(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr, betas=(0.965, cfg.beta2), rho=0.04)]
    if opt == "prodigy":
        # LR-free: a single group; lr is a fixed multiplier (=1), d is learned.
        return [Prodigy(decay + no_decay, lr=1.0,
                        betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)]
    if opt == "soap_adamw":
        params = decay + no_decay
        estimate = SOAP.estimated_state_bytes(params, cfg.soap_max_precond_dim)
        limit = int(cfg.optimizer_state_limit_gib * 1024 ** 3)
        if estimate > limit:
            raise MemoryError(
                f"SOAP optimizer state estimate {estimate / 1024**3:.2f} GiB "
                f"exceeds gate {cfg.optimizer_state_limit_gib:.2f} GiB")
        return [SOAP(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
            precondition_frequency=cfg.soap_precondition_frequency,
            max_precond_dim=cfg.soap_max_precond_dim)]
    raise ValueError(f"unknown optimizer {opt}")


def is_schedule_free(cfg) -> bool:
    return cfg.optimizer == "schedule_free_adamw"


def is_lr_free(cfg) -> bool:
    """LR-free optimizers ignore the external schedule (they set their own step
    size): Schedule-Free and Prodigy (guide §4.5)."""
    return cfg.optimizer in ("schedule_free_adamw", "prodigy")
