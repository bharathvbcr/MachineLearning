//! From-scratch Metal training stack for arch_02.
//!
//! Phase 0–3: runtime, fwd/bwd, on-device optim.
//! Phase 4: bf16/async command path, FineWeb train harness, BPB eval.

#![allow(dead_code)]

pub mod ab_flags;
pub mod bpb;
pub mod checkpoint;
pub mod data;
pub mod dispatch;
pub mod engine;
pub mod ffi;
pub mod gemm;
pub mod init;
pub mod log;
pub mod mixers;
pub mod model_bwd;
pub mod model_fwd;
pub mod npy;
pub mod optim;
pub mod optimizer_registry;
pub mod parity;
pub mod runtime;
pub mod research;
pub mod signpost;
pub mod ssm_glue;
pub mod tape;
pub mod tensor;
pub mod weights;

pub use checkpoint::{
    collect_divergence_norms, collect_divergence_norms_device,
    load_ema_state_python_npy, load_muon_momentum_python_npy,
    load_optim_state_python_npy, read_training_checkpoint_meta,
    save_ema_state_python_npy, save_optim_state_python_npy,
    save_training_checkpoint, save_weights_python_npy, DivergenceNorms,
    TrainingCheckpointMeta, CHECKPOINT_VERSION,
};
pub use gemm::{gemm, gemm_f32, GemmBackend};
pub use engine::{EngineCreateConfig, EngineStepMetrics, PrecisionModeConfig, TrainingEngine};
pub use optim::{
    clip_grad_norm_device, copy_ema_into_weights, lr_mul_at, optim_step, ClipMode, LrSchedule,
    OptimHyperparams, OptimState,
};
pub use optimizer_registry::OptimizerKind;
pub use runtime::{BufferKind, DeviceMemoryInfo, GpuRuntime, PrecisionMode};
pub use tensor::{DType, GpuBuffer, Tensor};
pub use weights::{ModelConfig, Weights};

/// Metallib produced by `build.rs` (absolute path baked at compile time).
pub fn metallib_path() -> &'static str {
    env!("METAL_NATIVE_METALLIB")
}
