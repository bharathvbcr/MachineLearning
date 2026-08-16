// Head: softcap + fused CE mean loss. Logits GEMM is separate (tied embed).
#include <metal_stdlib>
using namespace metal;

/// logits_post = softcap * tanh(logits / softcap). In-place or out.
kernel void softcap_f32(
    device const float *pre [[buffer(0)]],
    device float *post [[buffer(1)]],
    constant float &softcap [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float z = pre[gid] / softcap;
    post[gid] = softcap * tanh(z);
}

/// Per-row CE contribution: -log_softmax(logits)[target]. Writes [rows] then host/device reduces.
kernel void ce_row_f32(
    device const float *logits [[buffer(0)]], // [rows, V]
    device const int *targets [[buffer(1)]],  // [rows]
    device float *row_loss [[buffer(2)]],     // [rows]
    constant uint &rows [[buffer(3)]],
    constant uint &V [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const device float *row = logits + gid * V;
    // max for stability
    float m = -INFINITY;
    for (uint v = 0; v < V; ++v) m = max(m, row[v]);
    float sum = 0.0f;
    for (uint v = 0; v < V; ++v) sum += exp(row[v] - m);
    const int tgt = targets[gid];
    const float log_prob = (row[(uint)tgt] - m) - log(sum);
    row_loss[gid] = -log_prob;
}

/// Mean of row_loss → single scalar (simd tree reduce).
kernel void mean_reduce_f32(
    device const float *row_loss [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    threadgroup float shared[256];
    float local = 0.0f;
    for (uint i = gid; i < n; i += tpg) {
        local += row_loss[i];
    }
    shared[lid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < tpg; stride <<= 1) {
        uint idx = lid * (stride << 1);
        if (idx + stride < tpg) {
            shared[idx] += shared[idx + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        out[0] = shared[0] / (float)n;
    }
}

/// Fused softcap + per-row CE into row_loss and optional post logits.
kernel void softcap_ce_row_f32(
    device const float *pre [[buffer(0)]],   // [rows, V]
    device const int *targets [[buffer(1)]],
    device float *post [[buffer(2)]],        // [rows, V]
    device float *row_loss [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &V [[buffer(5)]],
    constant float &softcap [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const device float *prow = pre + gid * V;
    device float *orow = post + gid * V;
    float m = -INFINITY;
    for (uint v = 0; v < V; ++v) {
        float z = prow[v] / softcap;
        float p = softcap * tanh(z);
        orow[v] = p;
        m = max(m, p);
    }
    float sum = 0.0f;
    for (uint v = 0; v < V; ++v) sum += exp(orow[v] - m);
    const int tgt = targets[gid];
    const float log_prob = (orow[(uint)tgt] - m) - log(sum);
    row_loss[gid] = -log_prob;
}
