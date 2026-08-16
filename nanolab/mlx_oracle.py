"""Independent MLX oracle for target-shape attention and matrix optimizers.

This module is deliberately not imported by the PyTorch lab.  It gives the
native Metal engine a second autograd implementation on Apple silicon and a
manual backward written only as matrix multiplications plus softmax reduction.

Run: ``python -m nanolab.mlx_oracle``
"""

from __future__ import annotations

import math

import mlx.core as mx


POLAR_EXPRESS_COEFFICIENTS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def _expand_gqa(x, query_heads: int):
    repeats = query_heads // x.shape[2]
    if repeats == 1:
        return x
    b, t, h, d = x.shape
    return mx.broadcast_to(x[:, :, :, None, :], (b, t, h, repeats, d)).reshape(
        b, t, query_heads, d)


def causal_attention(q, k, v):
    """Reference GQA attention; tensors are [B,T,H,D] and [B,T,Hkv,D]."""
    kq, vq = _expand_gqa(k, q.shape[2]), _expand_gqa(v, q.shape[2])
    qt = q.transpose(0, 2, 1, 3)
    kt = kq.transpose(0, 2, 1, 3)
    vt = vq.transpose(0, 2, 1, 3)
    scores = (qt @ kt.transpose(0, 1, 3, 2)) / math.sqrt(q.shape[-1])
    t = q.shape[1]
    mask = mx.triu(mx.full((t, t), -mx.inf), k=1)
    probs = mx.softmax(scores + mask, axis=-1)
    out = probs @ vt
    return out.transpose(0, 2, 1, 3), probs


def attention_backward_matmul(q, k, v, dout):
    """Manual causal-attention backward using GEMMs and one row reduction."""
    h, hkv, d = q.shape[2], k.shape[2], q.shape[3]
    repeats = h // hkv
    kq, vq = _expand_gqa(k, h), _expand_gqa(v, h)
    qt = q.transpose(0, 2, 1, 3)
    kt = kq.transpose(0, 2, 1, 3)
    vt = vq.transpose(0, 2, 1, 3)
    dot = dout.transpose(0, 2, 1, 3)
    scores = (qt @ kt.transpose(0, 1, 3, 2)) / math.sqrt(d)
    t = q.shape[1]
    probs = mx.softmax(scores + mx.triu(mx.full((t, t), -mx.inf), k=1), axis=-1)

    dv_heads = probs.transpose(0, 1, 3, 2) @ dot
    dp = dot @ vt.transpose(0, 1, 3, 2)
    ds = probs * (dp - mx.sum(dp * probs, axis=-1, keepdims=True))
    dq = (ds @ kt) / math.sqrt(d)
    dk_heads = (ds.transpose(0, 1, 3, 2) @ qt) / math.sqrt(d)

    dq = dq.transpose(0, 2, 1, 3)
    b, _, t, _ = dk_heads.shape
    dk = dk_heads.reshape(b, hkv, repeats, t, d).sum(axis=2).transpose(0, 2, 1, 3)
    dv = dv_heads.reshape(b, hkv, repeats, t, d).sum(axis=2).transpose(0, 2, 1, 3)
    return dq, dk, dv


def matrix_sign(x, method="ns5"):
    """MLX matrix-sign reference for NS3, NS5, and Polar Express."""
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.swapaxes(-2, -1)
    safety = 1.02 if method == "polar" else 1.0
    x = x / (mx.sqrt(mx.sum(x * x, axis=(-2, -1), keepdims=True)) * safety + 1e-6)
    if method == "ns3":
        coeffs = ((3.4445, -4.7750, 2.0315),) * 3
    elif method == "ns5":
        coeffs = ((3.4445, -4.7750, 2.0315),) * 5
    elif method == "polar":
        coeffs = POLAR_EXPRESS_COEFFICIENTS
    else:
        raise ValueError(method)
    for a, b, c in coeffs:
        gram = x @ x.swapaxes(-2, -1)
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.swapaxes(-2, -1) if transposed else x


def _attention_loss(q, k, v, dout):
    out, _ = causal_attention(q, k, v)
    return mx.sum(out * dout)


def self_test():
    mx.random.seed(1337)
    # Exact target channel geometry: C=24*32=768, KV=12*32=384.  T=7 keeps
    # this parity gate fast while still exercising an irregular causal tail.
    shape_q, shape_kv = (1, 7, 24, 32), (1, 7, 12, 32)
    q = mx.random.normal(shape_q) * 0.1
    k = mx.random.normal(shape_kv) * 0.1
    v = mx.random.normal(shape_kv) * 0.1
    dout = mx.random.normal(shape_q) * 0.1
    auto = mx.grad(_attention_loss, argnums=(0, 1, 2))(q, k, v, dout)
    manual = attention_backward_matmul(q, k, v, dout)
    mx.eval(*auto, *manual)
    errors = [float(mx.max(mx.abs(a - b)).item()) for a, b in zip(auto, manual)]
    assert max(errors) < 2e-5, f"attention backward mismatch: {errors}"

    probe = mx.random.normal((3, 16, 8))
    ns3, ns5, polar = (matrix_sign(probe, name) for name in ("ns3", "ns5", "polar"))
    mx.eval(ns3, ns5, polar)
    assert float(mx.max(mx.abs(ns3 - ns5)).item()) > 1e-4
    assert float(mx.max(mx.abs(polar - ns5)).item()) > 1e-4
    print(f"MLX oracle PASS | attention dq/dk/dv max errors={errors}")


if __name__ == "__main__":
    self_test()
