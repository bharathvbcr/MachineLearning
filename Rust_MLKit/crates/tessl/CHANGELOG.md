# Changelog

All notable changes to `tessl` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Quantized int8 GEMM with fused dequantization**: `nn::gemm_i8_dequant`.
  `int8 x int8` accumulates into `int32` natively on TensorOps, and every
  product fits, so the integer result carries **no rounding at all** — tested by
  exact equality against an integer reference, not a tolerance. The per-column
  dequantization is applied in registers between the accumulate and the store.
  `k` above 131072 is refused, past which a full-range accumulation could wrap
  the int32 silently.
- **Corrected a misdiagnosis this crate had been repeating.** Quantized
  TensorOps was documented as blocked because `MTLTensorDataType::Int4` is
  unbound in objc2-metal 0.3. That binding gates host-created `MTLTensor`
  descriptors, and every kernel here builds tensors from raw device pointers
  instead, so it never applied. TensorOps supports
  `uint8_t/int8_t/uint4b_format/int4b_format` per the header's own diagnostic;
  what actually blocks Int4 is the shader-side tensor constructor for a
  sub-byte element type.
- **Strided batched GEMM**: `gemm_batched` with `BatchedGemm` and
  `BatchStrides`. The batch is the grid's second dimension, so it costs a
  pointer offset per threadgroup and nothing else — every element is
  bit-identical to the `gemm` that would have produced it. Per-operand strides,
  because a **zero** stride is the useful case: batched activations against one
  shared weight matrix needs no copies of B. Dimensions are passed explicitly
  rather than read from tensor shapes, since a rank-2 shape cannot distinguish
  `[batch * m, k]` from `[m, k]` and a broadcast B genuinely is `[k, n]`.
- **IEEE binary16 (`DType::F16`)**: `alloc_tensor_f16`, `cast_f32_to_f16` /
  `cast_f16_to_f32`, host `f32_to_f16_bits` / `f16_bits_to_f32` /
  `f32_slice_to_f16`, `GpuBuffer::write_f16_bits`, and f16 GEMM kernels
  (`matmul2d_tensorops_f16_f32`, the 64x64 variant, and the epilogue variant).
  f16 and bf16 are both two bytes and both accumulate in f32, but their bit
  layouts differ, so `nn_coop_kernel` now selects on a three-way `CoopElem`
  rather than a boolean, and `ensure_bf16` refuses f16 operands instead of
  converting them — that conversion would lose three mantissa bits and change
  the exponent range to buy a path the caller did not ask for.
- **Row-wise reductions**: `nn::softmax_rows_f32`, `nn::row_sum_f32`,
  `nn::row_max_f32`. One threadgroup per row, striding, so `cols` is unbounded.
  Softmax subtracts the row maximum before exponentiating — without it a single
  logit above about 88 overflows `exp` in f32 and takes the row to NaN, which is
  an ordinary attention input rather than a pathological one. A fully masked row
  (`-inf` everywhere) yields a uniform distribution rather than NaN.
- **`gemm_epilogue` — fused GEMM epilogue.**
  `C = activation(alpha * A@B + beta * C_prev + bias)` in one dispatch, applied
  while the accumulator is still in registers, so `C` is written once and read
  at most once. Measured 1.57x to 2.43x cheaper than a single elementwise sweep
  over `C` — which is itself strictly less work than any real unfused epilogue.
- `Activation` (`None`, `Relu`, `GeluTanh`, `Silu`) and `Epilogue`, with
  `Epilogue::default()` as the identity, which dispatches to plain `gemm`.
- Per-column bias broadcasts through a row-stride-0 tensor view, reusing the
  cooperative `load` path that fetches `C_prev`.
- `matmul2d_tensorops_bf16_f32_epi` and `matmul2d_tensorops_f32_relaxed_epi`.
  Separate entry points rather than extra parameters on the existing kernels:
  Metal faults on a declared-but-unbound buffer, so widening those signatures
  would force every current caller to bind four operands it does not use. The
  epilogue is a template parameter, so both share one source and the plain path
  compiles to exactly what it did before.
- `examples/epilogue_cost.rs`.
- **`nn` module — 44 kernels promoted out of `gemma-metal`.** RMSNorm (f32,
  bf16, fused residual-add with layer scale), gated MLP activations (SiLU,
  `gelu_pytorch_tanh`), sliding-window and global flash attention, fused
  RMSNorm+QKV+RoPE, MLX-format Q4 GEMV and GEMM, Q8 GEMV, KV-cache timestep
  stores and ring densify, quantized embedding lookup, and softcap/argmax
  sampling. These had lived in one model's crate, reachable only as raw strings
  through an overlay metallib.
- Every `nn` entry point validates operand extents on the host before encoding.
  The kernels guard `gid >= n` and nothing else, so an undersized buffer was
  previously an unchecked out-of-bounds *device* read.
- A `_with_scalars` variant of each entry point, taking a closure that binds the
  scalar operands. Callers that need stable GPU addresses across encodes — an
  Indirect Command Buffer that froze its binds — supply their own persistent
  scalar pool without reimplementing the dispatch.
- `GpuRuntime::max_threadgroup_memory`, so callers can check a kernel's
  threadgroup-memory request before a dispatch-time failure that names neither
  the kernel nor the dimension.
- `examples/gemm.rs` and `examples/nn_layer.rs`, both run by CI. The README's
  snippets are these files.

- **Attention tuning knobs, all measured rather than chosen.** `RowsGroups`
  (simdgroups per threadgroup in the row-parallel kernel, compiled per
  instantiation), `DecodeHeadBlock` (which query heads share a threadgroup in
  the KV-split partial pass), and a dispatch-parameter width for the KV-split
  reduce pass. [`bench/attn_tune.py`](bench/attn_tune.py) sweeps all six
  attention knobs in interleaved rounds and refuses to name a winner from a
  partial sweep; `--knob batched` decomposes wall clock into submit and kernel
  instead of tuning anything.
- **`nn::attn_kernel_for`** — the attention routing rule as a pure function, so
  a test can pin it. Both kernels compute the same thing, so a routing
  regression is invisible to every correctness test and shows up only as a
  slower clock.
- Four attention benchmark configs: two large-batch decodes that probe the
  routing rule out to `B*H = 2048`, one with GQA switched off (`Hkv = H`) to
  separate issued K/V traffic from unique, and one at `Hkv = 1` where two
  candidate cache layouts are byte-identical — which is what calibrates the
  run-to-run noise floor at ~3%.

### Fixed

- Removed a duplicate `cast_f32_to_bf16` that arrived with the promoted kernels.
  `GpuRuntime::pipeline` resolves the primary metallib before any overlay, so
  the copy in `rms_norm.metal` could never have been the one dispatched.
- `docs.rs` builds. The crate is Apple-silicon only and `build.rs` drives the
  Metal toolchain, neither of which exists on docs.rs's x86_64-linux builder;
  the build script now detects `DOCS_RS` and skips the shader compile, and
  `[package.metadata.docs.rs]` targets `aarch64-apple-darwin`.
- 18 rustdoc warnings: matrix-shape notation such as `C[M,N] = A[M,K] @ B[K,N]`
  was being parsed as intra-doc links, and one public doc linked a private item.
- README: the `add_metallib` / `from_metallib_path` snippets passed `&str` to
  functions that take `&Path` and would not have compiled.

- **The attention routing threshold was wrong.** `ATTN_SPLIT_KV_BELOW_TG` sent
  `B*H >= 128` decode dispatches to the row-parallel kernel on the evidence that
  the KV-split path lost 0.96 ms to 0.65 there — measured at one launch per
  submit, where ~88% of a decode call is the host round trip and the split path
  pays two submits to the row kernel's one. Measured kernel-only the split wins
  at every batch reachable, 1.4x to 9.9x, with no crossover. The threshold is
  gone.
- **The `tessl-decode` benchmark lane defaulted to chunk 256 for every head
  dim** while the library shipped 128, so a lane labelled "the decode kernel"
  was a kernel no caller reaches. Nothing timing-side could see it — both
  kernels are correct — and the parity dump caught it by showing the routed and
  forced lanes disagreeing where at `Tq == 1` they must be bit-identical.
- **`--dump-parity` wrote whichever implementation ran last** under
  `o_tessl.npy` while the manifest named a different kernel, so the scorer
  reported the row kernel's error under the routed path's name. Each
  implementation is now dumped from its own dispatch, and the dump refuses to
  run under a tuning override at all: a parity artifact describes the shipping
  configuration or it is not written.
- `attn_paired.py` recorded ratios only, so when a geomean moved there was no
  way to say which lane had moved. Both lanes' absolute medians now travel with
  every ratio.

### Changed

- `gemma-metal` now delegates seven wrappers to `tessl::nn` rather than
  reimplementing the dispatch, removing 52 lines of duplicate binding code.
- Documented the crate's `unsafe`. It splits into two classes: `objc2` message
  sends, where every site discharges the same obligation and which are covered
  once in the crate docs, and raw pointer/slice construction, where the
  obligation is site-specific. Every block in the second class now carries a
  `// SAFETY:` comment naming the invariant and where it is established —
  8 comments before, 22 now.
- `icb_smoke::verify_copy` bounds-checks its length against the buffer instead
  of relying on both call sites happening to pass the length the buffer was
  allocated from.

- **Attention throughput: 1.2x to 1.7x on every configuration.** K/V/Q reads in
  the row-parallel and KV-split kernels are `float4`, so a lane owns four
  consecutive head dims per step rather than issuing one scalar load each —
  at D=512 that was 32 memory instructions against 32 multiply-adds.
  Simdgroups per threadgroup is compiled per head dim instead of a fixed 8. The
  KV-split reduce pass keeps no accumulator array and takes its width as a
  dispatch parameter, rather than folding every chunk on one simdgroup per
  (batch, head). And the query heads that share a KV head now share a
  threadgroup, so they touch each K/V line while it is still in L1 instead of
  pulling it `H/Hkv` separate times — no threadgroup memory and no barrier.
  Against MLX the shipping path is now 0.90-0.97x on prefill and 0.91-0.97x on
  decode kernel-only, from 11.1x and 37.9x.
- Retuned `DECODE_CHUNK_*`, `DECODE_LANES_D128`, `ROWS_GROUPS_*` and
  `DECODE_HEAD_BLOCK_*` against the changed kernels. Every published tuning
  table is generated from its artifact JSON now rather than transcribed.

### Removed

- `ATTN_SPLIT_KV_BELOW_TG`. See *Fixed*: it was reading dispatch cost as kernel
  cost, and there is no batch at which the row kernel is the right choice for a
  single-query dispatch.
- `mtl_tensor::try_quant_tensorops_prefill_gemm`. Its entire body was
  `Err("not wired yet")` with every parameter underscored, no caller and no
  test. The fact it encoded — that quantized TensorOps prefill GEMM does not
  exist, because `MTLTensorDataType::Int4` is unbound in objc2-metal 0.3 — is
  now the single `QUANT_PREFILL_GEMM_WIRED` constant that
  `nax_verify_readiness` reports, replacing a second copy of the same sentinel
  that existed alongside it.

[Unreleased]: https://github.com/bharathvbcr/tessl
