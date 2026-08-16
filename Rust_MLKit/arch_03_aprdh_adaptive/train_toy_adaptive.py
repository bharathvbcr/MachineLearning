"""
Single-GPU raw-byte adaptive toy trainer.

Designed for RTX 3070 Ti 8GB:
- MODEL_DIM=128
- NUM_HEADS=4
- TRAIN_SEQ_LEN=256
- TRAIN_BATCH_TOKENS=8192
- one shared adaptive block with recurrent reuse
"""

from __future__ import annotations

import json
import math
import os
import random
import copy
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path

import glob
import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from train_gpt import load_data_shard
from train_hypercascade import (
    dequantize_state_dict_int6_zstd,
    dequantize_state_dict_int8,
    quantize_state_dict_int6_zstd,
    quantize_state_dict_int8,
)
from train_rada import CastedLinear, DeepSeekMLA, Muon, RMSNorm, apply_rotary_emb


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
    )
    run_id = os.environ.get("RUN_ID", f"toy_aprdh_{uuid.uuid4()}")
    seed = int(os.environ.get("SEED", "1337"))

    model_dim = int(os.environ.get("MODEL_DIM", "128"))
    num_heads = int(os.environ.get("NUM_HEADS", "4"))
    num_layers = int(os.environ.get("NUM_LAYERS", "1"))
    mlp_mult = int(os.environ.get("MLP_MULT", "2"))
    state_dim = int(os.environ.get("STATE_DIM", "16"))
    slow_state_dim = int(os.environ.get("SLOW_STATE_DIM", "8"))
    pgdn_chunk_size = int(os.environ.get("PGDN_CHUNK_SIZE", "32"))
    pgdn_backend = os.environ.get("PGDN_BACKEND", "rect_exact")
    pgdn_compile = bool(int(os.environ.get("PGDN_COMPILE", "0")))
    pgdn_compile_mode = os.environ.get("PGDN_COMPILE_MODE", "reduce-overhead")
    pgdn_compile_fullgraph = bool(int(os.environ.get("PGDN_COMPILE_FULLGRAPH", "0")))
    recur_min = int(os.environ.get("RECUR_MIN", "2"))
    recur_max = int(os.environ.get("RECUR_MAX", "5"))
    recur_max_active_default = int(os.environ.get("RECUR_MAX_ACTIVE_DEFAULT", "0"))
    mla_d_latent = int(os.environ.get("MLA_D_LATENT", "64"))
    mla_d_rope = int(os.environ.get("MLA_D_ROPE", "16"))
    mla_topk = float(os.environ.get("MLA_TOPK", "0.25"))
    mla_dense_fallback_frac = float(os.environ.get("MLA_DENSE_FALLBACK_FRAC", "0.50"))
    mla_target = float(os.environ.get("MLA_TARGET", "0.18"))
    continue_target = float(os.environ.get("CONTINUE_TARGET", "0.45"))
    router_budget_lambda = float(os.environ.get("ROUTER_BUDGET_LAMBDA", "0.02"))
    patch_diversity_lambda = float(os.environ.get("PATCH_DIVERSITY_LAMBDA", "0.01"))
    halt_margin = float(os.environ.get("HALT_MARGIN", "0.02"))
    rope_base = float(os.environ.get("ROPE_BASE", "10000.0"))
    engram_enable = bool(int(os.environ.get("ENGRAM_ENABLE", "0")))
    engram_ngrams = tuple(
        int(x.strip())
        for x in os.environ.get("ENGRAM_NGRAMS", "2,3").split(",")
        if x.strip()
    )
    engram_hash_buckets = tuple(
        int(x.strip())
        for x in os.environ.get("ENGRAM_HASH_BUCKETS", "4096,8192").split(",")
        if x.strip()
    )
    engram_dim = int(os.environ.get("ENGRAM_DIM", "48"))
    engram_heads = int(os.environ.get("ENGRAM_HEADS", "4"))
    engram_passes = tuple(
        int(x.strip())
        for x in os.environ.get("ENGRAM_PASSES", "0,1").split(",")
        if x.strip()
    )
    engram_conv_kernel = int(os.environ.get("ENGRAM_CONV_KERNEL", "4"))
    engram_init_scale = float(os.environ.get("ENGRAM_INIT_SCALE", "0.05"))
    controller_mode = os.environ.get("CONTROLLER_MODE", "marginal_gain")
    controller_cost_lambda = float(os.environ.get("CONTROLLER_COST_LAMBDA", "0.02"))
    controller_huber_delta = float(os.environ.get("CONTROLLER_HUBER_DELTA", "0.05"))
    patch_spans = tuple(
        int(x.strip())
        for x in os.environ.get("PATCH_SPANS", "1,2,4,8").split(",")
        if x.strip()
    )
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", "30.0"))

    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", "256"))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", "8192"))
    iterations = int(os.environ.get("ITERATIONS", "200"))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", "50"))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", "10"))
    val_max_bytes = int(os.environ.get("VAL_MAX_BYTES", str(256 * 1024)))
    corpus_bytes = int(os.environ.get("CORPUS_BYTES", str(256 * 1024)))

    matrix_lr = float(os.environ.get("MATRIX_LR", "0.03"))
    scalar_lr = float(os.environ.get("SCALAR_LR", "0.02"))
    embed_lr = float(os.environ.get("EMBED_LR", "0.05"))
    beta1 = float(os.environ.get("BETA1", "0.9"))
    beta2 = float(os.environ.get("BETA2", "0.95"))
    adam_eps = float(os.environ.get("ADAM_EPS", "1e-8"))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", "1.0"))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", "0.95"))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", "3"))
    muon_wd = float(os.environ.get("MUON_WD", "0.04"))
    max_grad_skip_norm = float(os.environ.get("MAX_GRAD_SKIP_NORM", "100.0"))
    skip_on_grad_spike = bool(int(os.environ.get("SKIP_ON_GRAD_SPIKE", "1")))
    grad_skip_warmup_steps = int(os.environ.get("GRAD_SKIP_WARMUP_STEPS", "20"))
    grad_recovery_shrink = float(os.environ.get("GRAD_RECOVERY_SHRINK", "0.5"))
    grad_recovery_tau_boost = float(os.environ.get("GRAD_RECOVERY_TAU_BOOST", "1.08"))
    grad_recovery_tau_cap = float(os.environ.get("GRAD_RECOVERY_TAU_CAP", "1.20"))
    grad_spike_patience = int(os.environ.get("GRAD_SPIKE_PATIENCE", "3"))
    min_lr_scale = float(os.environ.get("MIN_LR_SCALE", "0.2"))
    router_hard_after_step = int(os.environ.get("ROUTER_HARD_AFTER_STEP", "80"))
    rollback_spike_patience = int(os.environ.get("ROLLBACK_SPIKE_PATIENCE", "8"))
    soft_routing_cooldown_steps = int(os.environ.get("SOFT_ROUTING_COOLDOWN_STEPS", "16"))
    tau_floor_before_step = int(os.environ.get("TAU_FLOOR_BEFORE_STEP", "140"))
    tau_floor_early = float(os.environ.get("TAU_FLOOR_EARLY", "0.55"))

    debug_level = os.environ.get("DEBUG_LEVEL", "trace")
    fail_on_nonfinite = bool(int(os.environ.get("FAIL_ON_NONFINITE", "1")))
    fail_bundle_dir = os.environ.get("FAIL_BUNDLE_DIR", "debug/failures")
    trace_dir = os.environ.get("TRACE_DIR", "debug/traces")
    trace_every = int(os.environ.get("TRACE_EVERY", "1"))
    trace_max_tensors = int(os.environ.get("TRACE_MAX_TENSORS", "12"))

    artifact_format = os.environ.get("ARTIFACT_FORMAT", "int6_zstd")
    arch_version = os.environ.get("ARCH_VERSION", "toy_aprdh_v1")
    enable_ttt = bool(int(os.environ.get("ENABLE_TTT", "0")))
    ttt_rank = int(os.environ.get("TTT_RANK", "4"))
    ttt_decay = float(os.environ.get("TTT_DECAY", "0.96"))
    ttt_eta = float(os.environ.get("TTT_ETA", "0.05"))


@dataclass
class ForwardStats:
    patch_hist: list[float]
    avg_continue: float
    mla_fraction: float
    mla_fraction_raw: float
    avg_router_entropy: float
    patch_diversity_loss: float
    continue_budget_loss: float
    mla_budget_loss: float
    continue_controller_loss: float
    mla_controller_loss: float
    avg_fast_mix: float
    avg_slow_mix: float
    engram_gate_mean: float
    avg_continue_gain_pred: float
    avg_mla_gain_pred: float


@dataclass
class RecoveryState:
    consecutive_grad_spikes: int = 0
    lr_scale: float = 1.0
    total_grad_spikes: int = 0
    spikes_since_rollback: int = 0
    soft_routing_cooldown: int = 0


@dataclass
class AdaptiveRunState:
    recur_limit: int
    mla_topk_scale: float = 1.0
    stable_steps: int = 0


@dataclass
class EvalPolicy:
    recur_limit: int
    mla_topk_scale: float
    hard_routing: bool


def log0(msg: str) -> None:
    print(msg, flush=True)


def format_train_line(
    step: int,
    total_steps: int,
    loss: float,
    bpb: float,
    tau: float,
    stats: ForwardStats,
    step_ms: float,
    mem: dict[str, float],
    recur_limit: int | None = None,
    mla_topk_scale: float | None = None,
) -> str:
    core = (
        f"[train {step:>4d}/{total_steps}] "
        f"loss {loss:>7.4f} | bpb {bpb:>6.4f} | tau {tau:>4.2f} | "
        f"cont {stats.avg_continue:>5.3f} | "
        f"mla {stats.mla_fraction:>5.3f}/{stats.mla_fraction_raw:>5.3f} | "
        f"ent {stats.avg_router_entropy:>5.3f} | "
        f"eng {stats.engram_gate_mean:>5.3f} | "
        f"gain {stats.avg_continue_gain_pred:>5.3f}/{stats.avg_mla_gain_pred:>5.3f} | "
        f"mix {stats.avg_fast_mix:>4.2f}/{stats.avg_slow_mix:>4.2f} | "
        f"ms {step_ms:>7.1f} | "
        f"mem {mem['cuda_alloc_mb']:>4.0f}/{mem['cuda_peak_mb']:>4.0f}MB"
    )
    if recur_limit is not None and mla_topk_scale is not None:
        core += f" | adapt r{recur_limit}/m{mla_topk_scale:>4.2f}"
    return core


def format_val_line(step: int, total_steps: int, raw_byte_bpb: float, ttt_bpb: float | None = None) -> str:
    msg = f"[val   {step:>4d}/{total_steps}] raw_bpb {raw_byte_bpb:>7.6f}"
    if ttt_bpb is not None:
        msg += f" | ttt_bpb {ttt_bpb:>7.6f}"
    return msg


def scale_optimizer_lrs(optimizers, lr_scale: float) -> None:
    for opt in optimizers.values():
        for group in opt.param_groups:
            base_lr = group.get("base_lr", group["lr"])
            group["base_lr"] = base_lr
            group["lr"] = base_lr * lr_scale


def reset_optimizer_state(opt) -> None:
    opt.state.clear()


def grad_norm_by_group(model: nn.Module) -> dict[str, float]:
    out = {"embed": 0.0, "matrix": 0.0, "scalar": 0.0}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        gn = float(torch.linalg.vector_norm(g).item())
        if name.startswith("byte_emb"):
            out["embed"] += gn * gn
        elif p.ndim == 2:
            out["matrix"] += gn * gn
        else:
            out["scalar"] += gn * gn
    for key, value in out.items():
        out[key] = math.sqrt(value) if value > 0.0 else 0.0
    return out


def top_grad_params(model: nn.Module, topk: int = 5) -> list[dict[str, float | str]]:
    grads = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        gn = float(torch.linalg.vector_norm(p.grad.detach().float()).item())
        grads.append({"name": name, "grad_norm": gn})
    grads.sort(key=lambda x: x["grad_norm"], reverse=True)
    return grads[:topk]


def recover_from_grad_spike(
    args: Hyperparameters,
    optimizers,
    recovery: RecoveryState,
    tau: float,
) -> float:
    recovery.consecutive_grad_spikes += 1
    recovery.total_grad_spikes += 1
    recovery.spikes_since_rollback += 1
    recovery.soft_routing_cooldown = max(
        recovery.soft_routing_cooldown, args.soft_routing_cooldown_steps
    )
    if recovery.consecutive_grad_spikes >= args.grad_spike_patience:
        recovery.lr_scale = max(args.min_lr_scale, recovery.lr_scale * args.grad_recovery_shrink)
        scale_optimizer_lrs(optimizers, recovery.lr_scale)
        reset_optimizer_state(optimizers["muon"])
        reset_optimizer_state(optimizers["scalar"])
        tau = min(args.grad_recovery_tau_cap, tau * args.grad_recovery_tau_boost)
        recovery.consecutive_grad_spikes = 0
    return tau


def adapt_runtime_on_spike(args: Hyperparameters, runtime: AdaptiveRunState) -> None:
    runtime.stable_steps = 0
    if runtime.recur_limit > args.recur_min + 1:
        runtime.recur_limit -= 1
    runtime.mla_topk_scale = max(0.50, runtime.mla_topk_scale * 0.85)


def adapt_runtime_on_stable_step(args: Hyperparameters, runtime: AdaptiveRunState) -> None:
    runtime.stable_steps += 1
    if runtime.stable_steps >= 20:
        if runtime.recur_limit < args.recur_max:
            runtime.recur_limit += 1
        runtime.mla_topk_scale = min(1.0, runtime.mla_topk_scale * 1.05)
        runtime.stable_steps = 0


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda_backends() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    try:
        from torch.backends.cuda import (
            enable_cudnn_sdp,
            enable_flash_sdp,
            enable_math_sdp,
            enable_mem_efficient_sdp,
        )

        enable_cudnn_sdp(False)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(True)
        enable_math_sdp(True)
    except Exception:
        pass


def autocast_context(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


def build_token_byte_lut(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> list[np.ndarray]:
    table_size = max(int(sp.vocab_size()), vocab_size)
    lut: list[np.ndarray] = []
    for token_id in range(table_size):
        if token_id >= int(sp.vocab_size()):
            lut.append(np.zeros((0,), dtype=np.uint8))
            continue
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            lut.append(np.zeros((0,), dtype=np.uint8))
            continue
        if sp.is_byte(token_id):
            piece = sp.id_to_piece(token_id)
            arr = np.frombuffer(piece.encode("utf-8"), dtype=np.uint8)
            lut.append(arr.copy())
            continue
        piece = sp.id_to_piece(token_id)
        prefix = b""
        if piece.startswith("▁"):
            prefix = b" "
            piece = piece[1:]
        arr = np.frombuffer(prefix + piece.encode("utf-8"), dtype=np.uint8)
        lut.append(arr.copy())
    return lut


class RawByteStream:
    def __init__(self, pattern: str, token_byte_lut: list[np.ndarray]):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.token_byte_lut = token_byte_lut
        self.file_idx = 0
        self.byte_buffer = np.zeros((0,), dtype=np.uint8)

    def _tokens_to_bytes(self, tokens: Tensor) -> np.ndarray:
        pieces: list[np.ndarray] = []
        total = 0
        for tok in tokens.tolist():
            arr = self.token_byte_lut[int(tok)]
            if arr.size:
                pieces.append(arr)
                total += int(arr.size)
        if total == 0:
            return np.zeros((0,), dtype=np.uint8)
        out = np.empty((total,), dtype=np.uint8)
        off = 0
        for arr in pieces:
            nxt = off + arr.size
            out[off:nxt] = arr
            off = nxt
        return out

    def _fill(self, n: int) -> None:
        while self.byte_buffer.size < n:
            tokens = load_data_shard(self.files[self.file_idx])
            self.file_idx = (self.file_idx + 1) % len(self.files)
            decoded = self._tokens_to_bytes(tokens)
            if decoded.size == 0:
                continue
            if self.byte_buffer.size == 0:
                self.byte_buffer = decoded
            else:
                self.byte_buffer = np.concatenate([self.byte_buffer, decoded], axis=0)

    def take(self, n: int) -> Tensor:
        self._fill(n)
        out = self.byte_buffer[:n].copy()
        self.byte_buffer = self.byte_buffer[n:]
        return torch.from_numpy(out)


class RawByteLoader:
    def __init__(self, stream: RawByteStream, device: torch.device):
        self.stream = stream
        self.device = device

    def next_batch(self, total_bytes: int, seq_len: int) -> tuple[Tensor, Tensor]:
        need = total_bytes + max(total_bytes // seq_len, 1)
        local = self.stream.take(need).to(dtype=torch.int64)
        usable = (local.numel() - 1) // seq_len * seq_len
        local = local[: usable + 1]
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def load_validation_bytes(
    pattern: str, token_byte_lut: list[np.ndarray], max_bytes: int
) -> Tensor:
    stream = RawByteStream(pattern, token_byte_lut)
    return stream.take(max_bytes + 1).contiguous()


def write_compile_corpus(
    out_path: Path, pattern: str, token_byte_lut: list[np.ndarray], max_bytes: int
) -> None:
    stream = RawByteStream(pattern, token_byte_lut)
    raw = stream.take(max_bytes).numpy().tobytes()
    out_path.write_text(raw.decode("utf-8", errors="replace"), encoding="utf-8")


def hard_gumbel_softmax(logits: Tensor, tau: float, dim: int = -1) -> Tensor:
    return F.gumbel_softmax(logits, tau=tau, hard=True, dim=dim)


def hard_gumbel_sigmoid(logits: Tensor, tau: float) -> Tensor:
    stacked = torch.stack([-logits, logits], dim=-1)
    return F.gumbel_softmax(stacked, tau=tau, hard=True, dim=-1)[..., 1]


def soft_or_hard_gumbel_softmax(logits: Tensor, tau: float, dim: int = -1, hard: bool = True) -> Tensor:
    if hard:
        return F.gumbel_softmax(logits, tau=tau, hard=True, dim=dim)
    return F.softmax(logits / max(tau, 1e-3), dim=dim)


def soft_or_hard_gumbel_sigmoid(logits: Tensor, tau: float, hard: bool = True) -> Tensor:
    if hard:
        stacked = torch.stack([-logits, logits], dim=-1)
        return F.gumbel_softmax(stacked, tau=tau, hard=True, dim=-1)[..., 1]
    return torch.sigmoid(logits / max(tau, 1e-3))


def finite_or_zero(x: Tensor) -> Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def inv_sigmoid(x: float) -> float:
    x = min(max(x, 1e-4), 1.0 - 1e-4)
    return math.log(x / (1.0 - x))


class FastWeightAdapter(nn.Module):
    def __init__(self, dim: int, rank: int, decay: float, eta: float):
        super().__init__()
        self.decay = decay
        self.eta = eta
        self.in_proj = CastedLinear(dim, rank, bias=False)
        self.out_proj = CastedLinear(rank, dim, bias=False)
        self.u_proj = CastedLinear(dim + 257, rank, bias=False)
        self.v_proj = CastedLinear(dim + 257, rank, bias=False)
        self.register_buffer("state", torch.zeros(rank, rank), persistent=False)

    def reset_state(self) -> None:
        self.state.zero_()

    def forward(self, x: Tensor) -> Tensor:
        z = self.in_proj(x.float())
        delta = torch.einsum("blr,rs->bls", z, self.state).to(x.dtype)
        return self.out_proj(delta)

    @torch.no_grad()
    def update(self, hidden: Tensor, logits: Tensor, targets: Tensor) -> None:
        probs = logits.float().softmax(dim=-1)
        one_hot = F.one_hot(targets, num_classes=256).float()
        err = one_hot - probs
        hidden_mean = hidden.float().mean(dim=(0, 1))
        err_mean = err.mean(dim=(0, 1))
        features = torch.cat([hidden_mean, err_mean, err_mean.new_zeros(1)], dim=0)
        u = self.u_proj(features).float()
        v = self.v_proj(features).float()
        self.state.mul_(self.decay).add_(torch.outer(u, v), alpha=self.eta)


class SpanMixer(nn.Module):
    def __init__(self, dim: int, spans: tuple[int, ...]):
        super().__init__()
        self.spans = spans
        self.mixers = nn.ModuleList()
        for span in spans:
            if span == 1:
                self.mixers.append(nn.Identity())
            else:
                self.mixers.append(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size=span,
                        groups=dim,
                        padding=span - 1,
                        bias=False,
                    )
                )
        self.proj = nn.ModuleList(CastedLinear(dim, dim, bias=False) for _ in spans)
        self.router = CastedLinear(dim, len(spans), bias=False)

    def forward(self, x: Tensor, tau: float, hard: bool = True) -> tuple[Tensor, Tensor, Tensor]:
        bsz, seqlen, _ = x.shape
        x_t = x.transpose(1, 2)
        feats = []
        for mixer, proj, span in zip(self.mixers, self.proj, self.spans):
            if span == 1:
                feat = x
            else:
                feat = mixer(x_t)[..., :seqlen].transpose(1, 2)
            feats.append(proj(feat))
        logits = self.router(x)
        weights = soft_or_hard_gumbel_softmax(logits, tau=tau, dim=-1, hard=hard)
        mixed = (torch.stack(feats, dim=-2) * weights.unsqueeze(-1)).sum(dim=-2)
        return mixed, logits, weights


class TinyEngramMemory(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        if len(args.engram_ngrams) != len(args.engram_hash_buckets):
            raise ValueError("ENGRAM_NGRAMS and ENGRAM_HASH_BUCKETS must have matching lengths")
        if args.model_dim % args.engram_heads != 0:
            raise ValueError("MODEL_DIM must be divisible by ENGRAM_HEADS")
        self.ngrams = args.engram_ngrams
        self.hash_buckets = args.engram_hash_buckets
        self.num_heads = args.engram_heads
        self.head_dim = args.model_dim // args.engram_heads
        self.passes = set(args.engram_passes)
        self.embeddings = nn.ModuleList(
            nn.Embedding(bucket, args.engram_dim) for bucket in self.hash_buckets
        )
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=args.engram_init_scale)
        total_dim = len(self.ngrams) * args.engram_dim
        self.query_proj = CastedLinear(args.model_dim, args.model_dim, bias=False)
        self.key_proj = CastedLinear(total_dim, args.model_dim, bias=False)
        self.value_proj = CastedLinear(total_dim, args.model_dim, bias=False)
        self.conv = nn.Conv1d(
            args.model_dim,
            args.model_dim,
            kernel_size=args.engram_conv_kernel,
            groups=args.model_dim,
            padding=args.engram_conv_kernel - 1,
            bias=False,
        )
        self.out_proj = CastedLinear(args.model_dim, args.model_dim, bias=False)
        self.register_buffer(
            "hash_primes",
            torch.tensor([10007, 10009, 10037, 10039, 10061, 10067], dtype=torch.int64),
            persistent=False,
        )

    def _shift(self, x: Tensor, amount: int) -> Tensor:
        if amount == 0:
            return x
        pad_val = 257 + amount
        return F.pad(x, (amount, 0), value=pad_val)[:, : x.size(1)]

    def _hash_ids(self, x: Tensor) -> list[Tensor]:
        x64 = x.to(torch.int64)
        out = []
        for idx, (ngram, bucket) in enumerate(zip(self.ngrams, self.hash_buckets)):
            shifted = [self._shift(x64, k) for k in range(ngram)]
            mix = shifted[0] * int(self.hash_primes[idx])
            for k in range(1, ngram):
                mix = torch.bitwise_xor(mix, shifted[k] * int(self.hash_primes[idx + k]))
            out.append(torch.remainder(mix, bucket).long())
        return out

    def forward(self, hidden: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        ids = self._hash_ids(x)
        embeds = [emb(idx) for emb, idx in zip(self.embeddings, ids)]
        feats = torch.cat(embeds, dim=-1)
        bsz, seqlen, _ = hidden.shape
        query = self.query_proj(hidden.float()).view(bsz, seqlen, self.num_heads, self.head_dim)
        key = self.key_proj(feats.float()).view(bsz, seqlen, self.num_heads, self.head_dim)
        query = F.normalize(query, dim=-1, eps=1e-6)
        key = F.normalize(key, dim=-1, eps=1e-6)
        gate = torch.sigmoid(
            (query * key).sum(dim=-1).mean(dim=-1, keepdim=True) / math.sqrt(self.head_dim)
        )
        value = self.value_proj(feats.float())
        conv = self.conv(value.transpose(1, 2))[..., :seqlen].transpose(1, 2)
        out = self.out_proj((value + conv) * gate)
        return finite_or_zero(out.to(hidden.dtype)), gate


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int):
        super().__init__()
        hidden = dim * mult
        self.w1 = CastedLinear(dim, hidden, bias=False)
        self.w2 = CastedLinear(dim, hidden, bias=False)
        self.out = CastedLinear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(F.silu(self.w1(x)) * self.w2(x))


def _pgdn_dual_chunk_scan(
    mem_fast: Tensor,
    mem_slow: Tensor,
    q_fast: Tensor,
    k_fast: Tensor,
    q_slow: Tensor,
    k_slow: Tensor,
    v_chunk: Tensor,
    alpha_fast: Tensor,
    beta_fast: Tensor,
    alpha_slow: Tensor,
    beta_slow: Tensor,
    mix_chunk: Tensor,
    inv_err_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    _, _, chunk_size, head_dim = v_chunk.shape
    out_chunk = torch.empty_like(v_chunk)
    for t in range(chunk_size):
        z_k_fast = k_fast[:, :, t]
        z_q_fast = q_fast[:, :, t]
        z_k_slow = k_slow[:, :, t]
        z_q_slow = q_slow[:, :, t]
        v_t = v_chunk[:, :, t]

        pred_fast = torch.matmul(mem_fast, z_k_fast.unsqueeze(-1)).squeeze(-1)
        pred_slow = torch.matmul(mem_slow, z_k_slow.unsqueeze(-1)).squeeze(-1)
        err_fast = torch.tanh((v_t - pred_fast) * inv_err_scale) / inv_err_scale
        err_slow = torch.tanh((v_t - pred_slow) * inv_err_scale) / inv_err_scale
        err_fast = torch.nan_to_num(err_fast, nan=0.0, posinf=0.0, neginf=0.0)
        err_slow = torch.nan_to_num(err_slow, nan=0.0, posinf=0.0, neginf=0.0)

        mem_fast = alpha_fast[:, :, t, None, None] * mem_fast + beta_fast[:, :, t, None, None] * (
            err_fast.unsqueeze(-1) * z_k_fast.unsqueeze(-2)
        )
        mem_slow = alpha_slow[:, :, t, None, None] * mem_slow + beta_slow[:, :, t, None, None] * (
            err_slow.unsqueeze(-1) * z_k_slow.unsqueeze(-2)
        )
        mem_fast = torch.nan_to_num(mem_fast, nan=0.0, posinf=0.0, neginf=0.0)
        mem_slow = torch.nan_to_num(mem_slow, nan=0.0, posinf=0.0, neginf=0.0)

        fast_out = torch.matmul(mem_fast, z_q_fast.unsqueeze(-1)).squeeze(-1)
        slow_out = torch.matmul(mem_slow, z_q_slow.unsqueeze(-1)).squeeze(-1)
        mixed = mix_chunk[:, :, t, None] * fast_out + (1.0 - mix_chunk[:, :, t, None]) * slow_out
        out_chunk[:, :, t] = torch.nan_to_num(mixed, nan=0.0, posinf=0.0, neginf=0.0)

    return out_chunk, mem_fast, mem_slow


def _pgdn_dual_intra_chunk_exact(
    q_fast: Tensor,
    k_fast: Tensor,
    q_slow: Tensor,
    k_slow: Tensor,
    v_chunk: Tensor,
    alpha_fast: Tensor,
    beta_fast: Tensor,
    alpha_slow: Tensor,
    beta_slow: Tensor,
    mix_chunk: Tensor,
    mem_fast_in: Tensor,
    mem_slow_in: Tensor,
    inv_err_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    chunk_size = v_chunk.shape[2]
    mem_fast = mem_fast_in
    mem_slow = mem_slow_in
    outputs = []
    for t in range(chunk_size):
        z_k_fast = k_fast[:, :, t]
        z_q_fast = q_fast[:, :, t]
        z_k_slow = k_slow[:, :, t]
        z_q_slow = q_slow[:, :, t]
        v_t = v_chunk[:, :, t]

        pred_fast = torch.matmul(mem_fast, z_k_fast.unsqueeze(-1)).squeeze(-1)
        pred_slow = torch.matmul(mem_slow, z_k_slow.unsqueeze(-1)).squeeze(-1)
        err_fast = torch.tanh((v_t - pred_fast) * inv_err_scale) / inv_err_scale
        err_slow = torch.tanh((v_t - pred_slow) * inv_err_scale) / inv_err_scale

        mem_fast = alpha_fast[:, :, t, None, None] * mem_fast + beta_fast[:, :, t, None, None] * (
            err_fast.unsqueeze(-1) * z_k_fast.unsqueeze(-2)
        )
        mem_slow = alpha_slow[:, :, t, None, None] * mem_slow + beta_slow[:, :, t, None, None] * (
            err_slow.unsqueeze(-1) * z_k_slow.unsqueeze(-2)
        )

        fast_out = torch.matmul(mem_fast, z_q_fast.unsqueeze(-1)).squeeze(-1)
        slow_out = torch.matmul(mem_slow, z_q_slow.unsqueeze(-1)).squeeze(-1)
        outputs.append(mix_chunk[:, :, t, None] * fast_out + (1.0 - mix_chunk[:, :, t, None]) * slow_out)

    return torch.stack(outputs, dim=2), mem_fast, mem_slow


class PGDNDualExactFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_fast: Tensor,
        k_fast: Tensor,
        q_slow: Tensor,
        k_slow: Tensor,
        v: Tensor,
        alpha_fast: Tensor,
        beta_fast: Tensor,
        alpha_slow: Tensor,
        beta_slow: Tensor,
        mix: Tensor,
        chunk_size: int,
        inv_err_scale: float,
    ) -> Tensor:
        bsz, num_heads, seqlen, head_dim = v.shape
        pad_len = (-seqlen) % chunk_size
        if pad_len > 0:
            q_fast = F.pad(q_fast, (0, 0, 0, pad_len))
            k_fast = F.pad(k_fast, (0, 0, 0, pad_len))
            q_slow = F.pad(q_slow, (0, 0, 0, pad_len))
            k_slow = F.pad(k_slow, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            alpha_fast = F.pad(alpha_fast, (0, pad_len), value=1.0)
            alpha_slow = F.pad(alpha_slow, (0, pad_len), value=1.0)
            beta_fast = F.pad(beta_fast, (0, pad_len))
            beta_slow = F.pad(beta_slow, (0, pad_len))
            mix = F.pad(mix, (0, pad_len), value=0.5)

        padded_len = seqlen + pad_len
        num_chunks = padded_len // chunk_size
        state_dim = q_fast.shape[-1]
        slow_state_dim = q_slow.shape[-1]

        q_fast = q_fast.contiguous()
        k_fast = k_fast.contiguous()
        q_slow = q_slow.contiguous()
        k_slow = k_slow.contiguous()
        v = v.contiguous()
        alpha_fast = alpha_fast.contiguous()
        beta_fast = beta_fast.contiguous()
        alpha_slow = alpha_slow.contiguous()
        beta_slow = beta_slow.contiguous()
        mix = mix.contiguous()

        qf_c = q_fast.view(bsz, num_heads, num_chunks, chunk_size, state_dim)
        kf_c = k_fast.view(bsz, num_heads, num_chunks, chunk_size, state_dim)
        qs_c = q_slow.view(bsz, num_heads, num_chunks, chunk_size, slow_state_dim)
        ks_c = k_slow.view(bsz, num_heads, num_chunks, chunk_size, slow_state_dim)
        v_c = v.view(bsz, num_heads, num_chunks, chunk_size, head_dim)
        af_c = alpha_fast.view(bsz, num_heads, num_chunks, chunk_size)
        bf_c = beta_fast.view(bsz, num_heads, num_chunks, chunk_size)
        as_c = alpha_slow.view(bsz, num_heads, num_chunks, chunk_size)
        bs_c = beta_slow.view(bsz, num_heads, num_chunks, chunk_size)
        mix_c = mix.view(bsz, num_heads, num_chunks, chunk_size)

        mem_fast = v.new_zeros((bsz, num_heads, head_dim, state_dim), dtype=torch.float32)
        mem_slow = v.new_zeros((bsz, num_heads, head_dim, slow_state_dim), dtype=torch.float32)
        outputs = []
        fast_carries = [mem_fast]
        slow_carries = [mem_slow]

        for chunk_idx in range(num_chunks):
            y_chunk, mem_fast, mem_slow = _pgdn_dual_intra_chunk_exact(
                qf_c[:, :, chunk_idx],
                kf_c[:, :, chunk_idx],
                qs_c[:, :, chunk_idx],
                ks_c[:, :, chunk_idx],
                v_c[:, :, chunk_idx],
                af_c[:, :, chunk_idx],
                bf_c[:, :, chunk_idx],
                as_c[:, :, chunk_idx],
                bs_c[:, :, chunk_idx],
                mix_c[:, :, chunk_idx],
                mem_fast,
                mem_slow,
                inv_err_scale,
            )
            outputs.append(y_chunk)
            fast_carries.append(mem_fast)
            slow_carries.append(mem_slow)

        ctx.save_for_backward(
            q_fast,
            k_fast,
            q_slow,
            k_slow,
            v,
            alpha_fast,
            beta_fast,
            alpha_slow,
            beta_slow,
            mix,
        )
        ctx.fast_carries = [s.detach() for s in fast_carries]
        ctx.slow_carries = [s.detach() for s in slow_carries]
        ctx.chunk_size = chunk_size
        ctx.pad_len = pad_len
        ctx.inv_err_scale = inv_err_scale

        y = torch.cat(outputs, dim=2)
        if pad_len > 0:
            y = y[:, :, :seqlen]
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        (
            q_fast,
            k_fast,
            q_slow,
            k_slow,
            v,
            alpha_fast,
            beta_fast,
            alpha_slow,
            beta_slow,
            mix,
        ) = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        pad_len = ctx.pad_len
        inv_err_scale = ctx.inv_err_scale
        fast_carries = ctx.fast_carries
        slow_carries = ctx.slow_carries

        bsz, num_heads, padded_len, head_dim = v.shape
        num_chunks = padded_len // chunk_size
        state_dim = q_fast.shape[-1]
        slow_state_dim = q_slow.shape[-1]

        if pad_len > 0:
            grad_y = F.pad(grad_y, (0, 0, 0, pad_len))

        qf_c = q_fast.view(bsz, num_heads, num_chunks, chunk_size, state_dim)
        kf_c = k_fast.view(bsz, num_heads, num_chunks, chunk_size, state_dim)
        qs_c = q_slow.view(bsz, num_heads, num_chunks, chunk_size, slow_state_dim)
        ks_c = k_slow.view(bsz, num_heads, num_chunks, chunk_size, slow_state_dim)
        v_c = v.view(bsz, num_heads, num_chunks, chunk_size, head_dim)
        af_c = alpha_fast.view(bsz, num_heads, num_chunks, chunk_size)
        bf_c = beta_fast.view(bsz, num_heads, num_chunks, chunk_size)
        as_c = alpha_slow.view(bsz, num_heads, num_chunks, chunk_size)
        bs_c = beta_slow.view(bsz, num_heads, num_chunks, chunk_size)
        mix_c = mix.view(bsz, num_heads, num_chunks, chunk_size)
        grad_y_c = grad_y.view(bsz, num_heads, num_chunks, chunk_size, head_dim)

        grad_q_fast = torch.zeros_like(qf_c)
        grad_k_fast = torch.zeros_like(kf_c)
        grad_q_slow = torch.zeros_like(qs_c)
        grad_k_slow = torch.zeros_like(ks_c)
        grad_v = torch.zeros_like(v_c)
        grad_alpha_fast = torch.zeros_like(af_c)
        grad_beta_fast = torch.zeros_like(bf_c)
        grad_alpha_slow = torch.zeros_like(as_c)
        grad_beta_slow = torch.zeros_like(bs_c)
        grad_mix = torch.zeros_like(mix_c)

        grad_mem_fast = torch.zeros_like(fast_carries[0])
        grad_mem_slow = torch.zeros_like(slow_carries[0])

        for chunk_idx in reversed(range(num_chunks)):
            with torch.enable_grad():
                qf = qf_c[:, :, chunk_idx].detach().requires_grad_(True)
                kf = kf_c[:, :, chunk_idx].detach().requires_grad_(True)
                qs = qs_c[:, :, chunk_idx].detach().requires_grad_(True)
                ks = ks_c[:, :, chunk_idx].detach().requires_grad_(True)
                vv = v_c[:, :, chunk_idx].detach().requires_grad_(True)
                af = af_c[:, :, chunk_idx].detach().requires_grad_(True)
                bf = bf_c[:, :, chunk_idx].detach().requires_grad_(True)
                a_s = as_c[:, :, chunk_idx].detach().requires_grad_(True)
                b_s = bs_c[:, :, chunk_idx].detach().requires_grad_(True)
                mm = mix_c[:, :, chunk_idx].detach().requires_grad_(True)
                mem_f = fast_carries[chunk_idx].detach().requires_grad_(True)
                mem_s = slow_carries[chunk_idx].detach().requires_grad_(True)

                y_chunk, mem_f_out, mem_s_out = _pgdn_dual_intra_chunk_exact(
                    qf, kf, qs, ks, vv, af, bf, a_s, b_s, mm, mem_f, mem_s, inv_err_scale
                )
                grads = torch.autograd.grad(
                    outputs=(y_chunk, mem_f_out, mem_s_out),
                    inputs=(qf, kf, qs, ks, vv, af, bf, a_s, b_s, mm, mem_f, mem_s),
                    grad_outputs=(grad_y_c[:, :, chunk_idx], grad_mem_fast, grad_mem_slow),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )

            grad_q_fast[:, :, chunk_idx] = grads[0]
            grad_k_fast[:, :, chunk_idx] = grads[1]
            grad_q_slow[:, :, chunk_idx] = grads[2]
            grad_k_slow[:, :, chunk_idx] = grads[3]
            grad_v[:, :, chunk_idx] = grads[4]
            grad_alpha_fast[:, :, chunk_idx] = grads[5]
            grad_beta_fast[:, :, chunk_idx] = grads[6]
            grad_alpha_slow[:, :, chunk_idx] = grads[7]
            grad_beta_slow[:, :, chunk_idx] = grads[8]
            grad_mix[:, :, chunk_idx] = grads[9]
            grad_mem_fast = grads[10]
            grad_mem_slow = grads[11]

        grad_q_fast = grad_q_fast.view(bsz, num_heads, padded_len, state_dim)
        grad_k_fast = grad_k_fast.view(bsz, num_heads, padded_len, state_dim)
        grad_q_slow = grad_q_slow.view(bsz, num_heads, padded_len, slow_state_dim)
        grad_k_slow = grad_k_slow.view(bsz, num_heads, padded_len, slow_state_dim)
        grad_v = grad_v.view(bsz, num_heads, padded_len, head_dim)
        grad_alpha_fast = grad_alpha_fast.view(bsz, num_heads, padded_len)
        grad_beta_fast = grad_beta_fast.view(bsz, num_heads, padded_len)
        grad_alpha_slow = grad_alpha_slow.view(bsz, num_heads, padded_len)
        grad_beta_slow = grad_beta_slow.view(bsz, num_heads, padded_len)
        grad_mix = grad_mix.view(bsz, num_heads, padded_len)

        if pad_len > 0:
            grad_q_fast = grad_q_fast[:, :, :-pad_len]
            grad_k_fast = grad_k_fast[:, :, :-pad_len]
            grad_q_slow = grad_q_slow[:, :, :-pad_len]
            grad_k_slow = grad_k_slow[:, :, :-pad_len]
            grad_v = grad_v[:, :, :-pad_len]
            grad_alpha_fast = grad_alpha_fast[:, :, :-pad_len]
            grad_beta_fast = grad_beta_fast[:, :, :-pad_len]
            grad_alpha_slow = grad_alpha_slow[:, :, :-pad_len]
            grad_beta_slow = grad_beta_slow[:, :, :-pad_len]
            grad_mix = grad_mix[:, :, :-pad_len]

        return (
            grad_q_fast,
            grad_k_fast,
            grad_q_slow,
            grad_k_slow,
            grad_v,
            grad_alpha_fast,
            grad_beta_fast,
            grad_alpha_slow,
            grad_beta_slow,
            grad_mix,
            None,
            None,
        )


def pgdn_dual_chunked_exact(
    q_fast: Tensor,
    k_fast: Tensor,
    q_slow: Tensor,
    k_slow: Tensor,
    v: Tensor,
    alpha_fast: Tensor,
    beta_fast: Tensor,
    alpha_slow: Tensor,
    beta_slow: Tensor,
    mix: Tensor,
    chunk_size: int,
    inv_err_scale: float,
) -> Tensor:
    return PGDNDualExactFunction.apply(
        q_fast,
        k_fast,
        q_slow,
        k_slow,
        v,
        alpha_fast,
        beta_fast,
        alpha_slow,
        beta_slow,
        mix,
        chunk_size,
        inv_err_scale,
    )


class ProjectedGDN(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        state_dim: int,
        slow_state_dim: int,
        ttt_rank: int,
        ttt_decay: float,
        ttt_eta: float,
        chunk_size: int,
        backend: str,
        compile_scan: bool,
        compile_mode: str,
        compile_fullgraph: bool,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"MODEL_DIM={dim} must divide NUM_HEADS={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.state_dim = state_dim
        self.slow_state_dim = slow_state_dim
        self.chunk_size = max(1, chunk_size)
        self.backend = backend
        if self.backend not in {"rect_exact", "compiled_loop"}:
            raise ValueError(f"Unknown PGDN_BACKEND={backend!r}")
        self.compile_scan = compile_scan and hasattr(torch, "compile")
        self.compile_mode = compile_mode
        self.compile_fullgraph = compile_fullgraph
        self._compiled_chunk_scan = None
        self._chunk_scan_compile_failed = False
        self._chunk_scan_warning_emitted = False
        self.in_proj = CastedLinear(dim, 3 * dim + 3 * num_heads, bias=False)
        self.k_state = nn.Parameter(
            torch.randn(num_heads, self.head_dim, state_dim) * 0.02
        )
        self.q_state = nn.Parameter(
            torch.randn(num_heads, self.head_dim, state_dim) * 0.02
        )
        self.k_state_slow = nn.Parameter(
            torch.randn(num_heads, self.head_dim, slow_state_dim) * 0.02
        )
        self.q_state_slow = nn.Parameter(
            torch.randn(num_heads, self.head_dim, slow_state_dim) * 0.02
        )
        self.decay_bias = nn.Parameter(torch.linspace(-2.0, 2.0, num_heads))
        self.slow_decay_bias = nn.Parameter(torch.linspace(1.0, 3.0, num_heads))
        self.update_bias = nn.Parameter(torch.linspace(-4.0, -2.0, num_heads))
        self.mix_bias = nn.Parameter(torch.zeros(num_heads))
        self.out_norm = RMSNorm()
        self.out_proj = CastedLinear(dim, dim, bias=False)
        self.adapter = FastWeightAdapter(dim, ttt_rank, ttt_decay, ttt_eta)

    def reset_ttt(self) -> None:
        self.adapter.reset_state()

    def _chunk_scan_impl(self):
        if self._compiled_chunk_scan is not None:
            return self._compiled_chunk_scan
        if self.compile_scan and self.k_state.is_cuda and not self._chunk_scan_compile_failed:
            try:
                self._compiled_chunk_scan = torch.compile(
                    _pgdn_dual_chunk_scan,
                    mode=self.compile_mode,
                    fullgraph=self.compile_fullgraph,
                    dynamic=False,
                )
                return self._compiled_chunk_scan
            except Exception:
                self._chunk_scan_compile_failed = True
        self._compiled_chunk_scan = _pgdn_dual_chunk_scan
        return self._compiled_chunk_scan

    def forward(self, x: Tensor, use_ttt: bool = False) -> tuple[Tensor, dict[str, Tensor]]:
        bsz, seqlen, _ = x.shape
        proj = self.in_proj(x)
        q, k, v, gates = proj.split(
            [self.dim, self.dim, self.dim, 3 * self.num_heads], dim=-1
        )
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim).permute(0, 2, 1, 3).float()
        k = k.view(bsz, seqlen, self.num_heads, self.head_dim).permute(0, 2, 1, 3).float()
        v = v.view(bsz, seqlen, self.num_heads, self.head_dim).permute(0, 2, 1, 3).float()
        gates = gates.view(bsz, seqlen, self.num_heads, 3).permute(0, 2, 1, 3).float()
        alpha = torch.sigmoid(gates[..., 0] + self.decay_bias[:, None]).clamp(0.05, 0.995)
        alpha_slow = torch.sigmoid(gates[..., 0] + self.slow_decay_bias[:, None]).clamp(0.70, 0.999)
        beta = 0.35 * torch.sigmoid(gates[..., 1] + self.update_bias[:, None])
        beta_slow = 0.15 * torch.sigmoid(gates[..., 1] + self.update_bias[:, None])
        mix = torch.sigmoid(gates[..., 2] + self.mix_bias[:, None]).clamp(0.05, 0.95)

        q_s = F.normalize(torch.einsum("bhld,hdn->bhln", q, self.q_state), dim=-1, eps=1e-6)
        k_s = F.normalize(torch.einsum("bhld,hdn->bhln", k, self.k_state), dim=-1, eps=1e-6)
        q_s_slow = F.normalize(
            torch.einsum("bhld,hdn->bhln", q, self.q_state_slow), dim=-1, eps=1e-6
        )
        k_s_slow = F.normalize(
            torch.einsum("bhld,hdn->bhln", k, self.k_state_slow), dim=-1, eps=1e-6
        )
        q_s = finite_or_zero(q_s)
        k_s = finite_or_zero(k_s)
        q_s_slow = finite_or_zero(q_s_slow)
        k_s_slow = finite_or_zero(k_s_slow)
        if self.backend == "rect_exact":
            inv_err_scale = 1.0 / math.sqrt(self.head_dim)
            y = pgdn_dual_chunked_exact(
                q_s,
                k_s,
                q_s_slow,
                k_s_slow,
                v,
                alpha,
                beta,
                alpha_slow,
                beta_slow,
                mix,
                self.chunk_size,
                inv_err_scale,
            )
            y = y.permute(0, 2, 1, 3).reshape(bsz, seqlen, self.dim)
            y = self.out_proj(self.out_norm(y).to(x.dtype))
            if use_ttt:
                y = y + self.adapter(x)
            y = finite_or_zero(y)
            return y, {
                "alpha_mean": alpha.mean().detach(),
                "alpha_slow_mean": alpha_slow.mean().detach(),
                "beta_mean": beta.mean().detach(),
                "mix_mean": mix.mean().detach(),
            }
        pad_len = (-seqlen) % self.chunk_size
        if pad_len > 0:
            q_s = F.pad(q_s, (0, 0, 0, pad_len))
            k_s = F.pad(k_s, (0, 0, 0, pad_len))
            q_s_slow = F.pad(q_s_slow, (0, 0, 0, pad_len))
            k_s_slow = F.pad(k_s_slow, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            alpha = F.pad(alpha, (0, pad_len), value=1.0)
            alpha_slow = F.pad(alpha_slow, (0, pad_len), value=1.0)
            beta = F.pad(beta, (0, pad_len))
            beta_slow = F.pad(beta_slow, (0, pad_len))
            mix = F.pad(mix, (0, pad_len), value=0.5)
        padded_len = seqlen + pad_len
        num_chunks = padded_len // self.chunk_size
        q_s = q_s.contiguous().view(bsz, self.num_heads, num_chunks, self.chunk_size, self.state_dim)
        k_s = k_s.contiguous().view(bsz, self.num_heads, num_chunks, self.chunk_size, self.state_dim)
        q_s_slow = q_s_slow.contiguous().view(
            bsz, self.num_heads, num_chunks, self.chunk_size, self.slow_state_dim
        )
        k_s_slow = k_s_slow.contiguous().view(
            bsz, self.num_heads, num_chunks, self.chunk_size, self.slow_state_dim
        )
        v = v.contiguous().view(bsz, self.num_heads, num_chunks, self.chunk_size, self.head_dim)
        alpha = alpha.contiguous().view(bsz, self.num_heads, num_chunks, self.chunk_size)
        alpha_slow = alpha_slow.contiguous().view(bsz, self.num_heads, num_chunks, self.chunk_size)
        beta = beta.contiguous().view(bsz, self.num_heads, num_chunks, self.chunk_size)
        beta_slow = beta_slow.contiguous().view(bsz, self.num_heads, num_chunks, self.chunk_size)
        mix = mix.contiguous().view(bsz, self.num_heads, num_chunks, self.chunk_size)
        mem = torch.zeros(
            bsz, self.num_heads, self.head_dim, self.state_dim, device=x.device, dtype=torch.float32
        )
        mem_slow = torch.zeros(
            bsz,
            self.num_heads,
            self.head_dim,
            self.slow_state_dim,
            device=x.device,
            dtype=torch.float32,
        )
        outs = []
        chunk_scan = self._chunk_scan_impl()
        inv_err_scale = 1.0 / math.sqrt(self.head_dim)
        for chunk_idx in range(num_chunks):
            chunk_args = (
                mem,
                mem_slow,
                q_s[:, :, chunk_idx],
                k_s[:, :, chunk_idx],
                q_s_slow[:, :, chunk_idx],
                k_s_slow[:, :, chunk_idx],
                v[:, :, chunk_idx],
                alpha[:, :, chunk_idx],
                beta[:, :, chunk_idx],
                alpha_slow[:, :, chunk_idx],
                beta_slow[:, :, chunk_idx],
                mix[:, :, chunk_idx],
                inv_err_scale,
            )
            try:
                y_chunk, mem, mem_slow = chunk_scan(*chunk_args)
            except Exception:
                if chunk_scan is not _pgdn_dual_chunk_scan and not self._chunk_scan_warning_emitted:
                    self._chunk_scan_warning_emitted = True
                    self._chunk_scan_compile_failed = True
                    self._compiled_chunk_scan = _pgdn_dual_chunk_scan
                    y_chunk, mem, mem_slow = self._compiled_chunk_scan(*chunk_args)
                else:
                    raise
            outs.append(y_chunk)
        y = torch.cat(outs, dim=2)[:, :, :seqlen]
        y = y.permute(0, 2, 1, 3).reshape(bsz, seqlen, self.dim)
        y = self.out_proj(self.out_norm(y).to(x.dtype))
        if use_ttt:
            y = y + self.adapter(x)
        y = finite_or_zero(y)
        return y, {
            "alpha_mean": alpha.mean().detach(),
            "alpha_slow_mean": alpha_slow.mean().detach(),
            "beta_mean": beta.mean().detach(),
            "mix_mean": mix.mean().detach(),
        }


class AdaptiveToyModel(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        if args.controller_mode != "marginal_gain":
            raise ValueError("Only CONTROLLER_MODE=marginal_gain is supported in the toy trainer")
        self.byte_emb = nn.Embedding(256, args.model_dim)
        self.patch_mixer = SpanMixer(args.model_dim, args.patch_spans)
        self.input_norm = RMSNorm()
        self.pass_embed = nn.Embedding(args.recur_max, args.model_dim)
        self.engram = TinyEngramMemory(args) if args.engram_enable else None
        self.gdn = ProjectedGDN(
            args.model_dim,
            args.num_heads,
            args.state_dim,
            args.slow_state_dim,
            args.ttt_rank,
            args.ttt_decay,
            args.ttt_eta,
            args.pgdn_chunk_size,
            args.pgdn_backend,
            args.pgdn_compile,
            args.pgdn_compile_mode,
            args.pgdn_compile_fullgraph,
        )
        self.ffn = FeedForward(args.model_dim, args.mlp_mult)
        self.mla = DeepSeekMLA(
            args.model_dim,
            args.num_heads,
            args.mla_d_latent,
            args.mla_d_rope,
            args.rope_base,
        )
        self.mla_adapter = FastWeightAdapter(
            args.model_dim, args.ttt_rank, args.ttt_decay, args.ttt_eta
        )
        self.continue_gain_head = nn.Sequential(
            CastedLinear(args.model_dim, 32, bias=False),
            nn.SiLU(),
            CastedLinear(32, 1, bias=False),
        )
        self.mla_gain_head = nn.Sequential(
            CastedLinear(4, 16, bias=False),
            nn.SiLU(),
            CastedLinear(16, 1, bias=False),
        )
        self.final_norm = RMSNorm()
        self.mtp_head = CastedLinear(args.model_dim, 256, bias=False)
        self.head = None if args.tie_embeddings else CastedLinear(args.model_dim, 256, bias=False)
        self.gdn_scale = nn.Parameter(torch.tensor(inv_sigmoid(0.25), dtype=torch.float32))
        self.ffn_scale = nn.Parameter(torch.tensor(inv_sigmoid(0.25), dtype=torch.float32))
        self.mla_scale = nn.Parameter(torch.tensor(inv_sigmoid(0.10), dtype=torch.float32))
        engram_init = args.engram_init_scale if args.engram_enable else 0.0
        self.engram_scale = nn.Parameter(torch.tensor(inv_sigmoid(max(engram_init, 1e-4)), dtype=torch.float32))

    def reset_ttt(self) -> None:
        self.gdn.reset_ttt()
        self.mla_adapter.reset_state()

    def _head_logits(self, x: Tensor) -> Tensor:
        x = self.final_norm(x)
        logits = F.linear(x, self.byte_emb.weight) if self.head is None else self.head(x)
        if self.args.logit_softcap > 0:
            logits = torch.tanh(logits / self.args.logit_softcap) * self.args.logit_softcap
        return logits

    def _mla_sparse(self, x: Tensor, top_idx: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape
        k_queries = top_idx.size(1)
        if k_queries >= seqlen or k_queries >= max(1, int(seqlen * self.args.mla_dense_fallback_frac)):
            return self.mla(x)
        top_idx_sorted, _ = torch.sort(top_idx, dim=1)
        gather_idx = top_idx_sorted.unsqueeze(-1).expand(-1, -1, x.size(-1))
        q_input = x.gather(1, gather_idx)

        c_kv = self.mla.kv_down(x)
        kv = self.mla.kv_up(c_kv).view(
            bsz, seqlen, self.mla.n_heads, self.mla.head_dim + self.mla.d_rope
        )
        k_content, k_rope = kv.split([self.mla.head_dim, self.mla.d_rope], dim=-1)

        q = self.mla.q_proj(q_input).view(
            bsz, k_queries, self.mla.n_heads, self.mla.head_dim + self.mla.d_rope
        )
        q_content, q_rope = q.split([self.mla.head_dim, self.mla.d_rope], dim=-1)

        cos_full, sin_full = self.mla.rotary(seqlen, x.device, q.dtype)
        cos_full = cos_full.squeeze(0).squeeze(0)
        sin_full = sin_full.squeeze(0).squeeze(0)
        cos_q = torch.gather(
            cos_full.unsqueeze(0).expand(bsz, -1, -1),
            1,
            top_idx_sorted.unsqueeze(-1).expand(-1, -1, cos_full.size(-1)),
        )
        sin_q = torch.gather(
            sin_full.unsqueeze(0).expand(bsz, -1, -1),
            1,
            top_idx_sorted.unsqueeze(-1).expand(-1, -1, sin_full.size(-1)),
        )
        q_rope = apply_rotary_emb(
            q_rope.transpose(1, 2),
            cos_q.unsqueeze(1),
            sin_q.unsqueeze(1),
        ).transpose(1, 2)
        k_rope = apply_rotary_emb(
            k_rope.transpose(1, 2),
            cos_full.unsqueeze(0).unsqueeze(0).to(dtype=q.dtype),
            sin_full.unsqueeze(0).unsqueeze(0).to(dtype=q.dtype),
        ).transpose(1, 2)

        q_final = torch.cat([q_content, q_rope], dim=-1).transpose(1, 2)
        k_final = torch.cat([k_content, k_rope], dim=-1).transpose(1, 2)
        v_final = k_content.transpose(1, 2)

        positions = torch.arange(seqlen, device=x.device)
        disallow = positions.view(1, 1, 1, seqlen) > top_idx_sorted.view(bsz, 1, k_queries, 1)
        attn_bias = torch.zeros(
            bsz, 1, k_queries, seqlen, device=x.device, dtype=q_final.dtype
        )
        attn_bias = attn_bias.masked_fill(disallow, torch.finfo(q_final.dtype).min)
        y_sel = F.scaled_dot_product_attention(
            q_final, k_final, v_final, attn_mask=attn_bias, is_causal=False
        )
        y_sel = y_sel.transpose(1, 2).reshape(bsz, k_queries, self.args.model_dim)
        y_sel = self.mla.out_proj(y_sel)
        out = x.new_zeros(x.shape)
        out.scatter_(1, gather_idx, y_sel.to(dtype=out.dtype))
        return out

    def forward(
        self,
        x: Tensor,
        y: Tensor | None = None,
        tau: float = 1.0,
        use_ttt: bool = False,
        recur_limit: int | None = None,
        mla_topk_scale: float = 1.0,
        hard_routing: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor], ForwardStats]:
        base_hidden = self.input_norm(self.byte_emb(x))
        h, patch_logits, patch_weights = self.patch_mixer(base_hidden, tau=tau, hard=hard_routing)
        patch_hist_t = patch_weights.float().mean(dim=(0, 1))
        patch_hist = patch_hist_t.detach().cpu().tolist()
        adaptive_router_entropies = []
        adaptive_continue_rates = []
        continue_losses = []
        mix_means = []
        engram_gate_means = []
        continue_gain_preds = []
        mla_gain_pred = None
        last_token_ce = None
        pending_continue_pred = None
        last_engram_gate = None
        gdn_scale = torch.sigmoid(self.gdn_scale)
        ffn_scale = torch.sigmoid(self.ffn_scale)
        mla_scale = torch.sigmoid(self.mla_scale)
        engram_scale = torch.sigmoid(self.engram_scale)

        active_recur_max = self.args.recur_max if recur_limit is None else max(self.args.recur_min, min(recur_limit, self.args.recur_max))
        if y is not None:
            base_logits = self._head_logits(h.detach())
            last_token_ce = F.cross_entropy(
                base_logits.reshape(-1, 256).float(), y.reshape(-1), reduction="none"
            ).reshape(y.shape)
        for pass_idx in range(active_recur_max):
            h_in = h + self.pass_embed.weight[pass_idx][None, None, :]
            gdn_out, gdn_stats = self.gdn(h_in, use_ttt=use_ttt)
            mix_means.append(gdn_stats["mix_mean"])
            ffn_out = self.ffn(h_in)
            engram_out = h.new_zeros(h.shape)
            engram_gate = h.new_zeros(h.shape[:2] + (1,))
            if self.engram is not None and pass_idx in self.engram.passes:
                engram_out, engram_gate = self.engram(h_in, x)
            updated = h + gdn_scale * gdn_out + ffn_scale * ffn_out + engram_scale * engram_out
            engram_gate_means.append(float(engram_gate.mean().item()))
            if y is not None:
                probe_logits = self._head_logits(updated.detach())
                probe_ce = F.cross_entropy(
                    probe_logits.reshape(-1, 256).float(), y.reshape(-1), reduction="none"
                ).reshape(y.shape)
                if pending_continue_pred is not None and last_token_ce is not None:
                    continue_losses.append(
                        F.huber_loss(
                            pending_continue_pred,
                            (last_token_ce - probe_ce).detach().clamp(-1.0, 1.0),
                            delta=self.args.controller_huber_delta,
                        )
                    )
                last_token_ce = probe_ce
            continue_gain_pred = self.continue_gain_head(updated.float()).squeeze(-1)
            continue_gain_preds.append(float(continue_gain_pred.detach().mean().item()))
            step_cost = self.args.controller_cost_lambda * float(pass_idx + 1)
            cont_score = continue_gain_pred - step_cost
            cont_prob = torch.sigmoid(cont_score / max(tau, 1e-3))
            if pass_idx < self.args.recur_min:
                cont_mask = torch.ones_like(cont_score)
                h = updated
            elif pass_idx == active_recur_max - 1:
                cont_mask = torch.zeros_like(cont_score)
                h = updated
            else:
                adaptive_router_entropies.append(
                    -(
                        cont_prob * cont_prob.clamp_min(1e-8).log()
                        + (1.0 - cont_prob) * (1.0 - cont_prob).clamp_min(1e-8).log()
                    ).mean()
                )
                cont_mask = soft_or_hard_gumbel_sigmoid(cont_score, tau=tau, hard=hard_routing)
                h = cont_mask.unsqueeze(-1) * updated + (1.0 - cont_mask.unsqueeze(-1)) * h
                adaptive_continue_rates.append(cont_mask.mean())
            pending_continue_pred = continue_gain_pred
            last_engram_gate = engram_gate

        provisional_logits = self._head_logits(h)
        probs = finite_or_zero(provisional_logits.float()).softmax(dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        top2 = torch.topk(probs, k=2, dim=-1).values
        margin = (top2[..., :1] - top2[..., 1:2]).abs()
        delta_norm = (h - base_hidden).pow(2).mean(dim=-1, keepdim=True).sqrt()
        engram_feature = (
            last_engram_gate
            if last_engram_gate is not None
            else delta_norm.new_zeros(delta_norm.shape)
        )
        uncertainty_features = torch.cat(
            [
                entropy.clamp(0.0, 8.0),
                (1.0 - margin).clamp(0.0, 1.0),
                delta_norm.clamp(0.0, 8.0),
                engram_feature.clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        mla_gain_pred = self.mla_gain_head(uncertainty_features.float()).squeeze(-1)
        mla_gate = soft_or_hard_gumbel_sigmoid(
            mla_gain_pred - self.args.controller_cost_lambda, tau=tau, hard=hard_routing
        )
        mla_fraction_raw = float(mla_gate.mean().item())
        topk_frac = max(0.05, min(1.0, self.args.mla_topk * mla_topk_scale))
        topk = max(1, int(self.args.train_seq_len * topk_frac))
        _, top_idx = torch.topk(mla_gain_pred, k=topk, dim=1)
        topk_mask = torch.zeros_like(mla_gain_pred)
        topk_mask.scatter_(1, top_idx, 1.0)
        mla_mask = mla_gate * topk_mask
        mla_out = self._mla_sparse(h, top_idx)
        if use_ttt:
            mla_out = mla_out + self.mla_adapter(h)
        h = h + mla_scale * mla_mask.unsqueeze(-1) * mla_out
        logits = self._head_logits(h)

        uniform = torch.full_like(patch_hist_t, 1.0 / max(patch_hist_t.numel(), 1))
        patch_diversity_loss = ((patch_hist_t - uniform) ** 2).sum()
        continue_budget_loss = h.new_zeros(())
        if adaptive_continue_rates:
            continue_budget_loss = (
                torch.stack(adaptive_continue_rates).mean() - self.args.continue_target
            ) ** 2
        mla_budget_loss = (mla_mask.mean() - self.args.mla_target) ** 2
        continue_controller_loss = (
            torch.stack(continue_losses).mean() if continue_losses else h.new_zeros(())
        )
        mla_controller_loss = h.new_zeros(())
        if y is not None:
            pre_mla_ce = F.cross_entropy(
                provisional_logits.reshape(-1, 256).float(), y.reshape(-1), reduction="none"
            ).reshape(y.shape)
            post_mla_ce = F.cross_entropy(
                logits.reshape(-1, 256).float(), y.reshape(-1), reduction="none"
            ).reshape(y.shape)
            mla_controller_loss = F.huber_loss(
                mla_gain_pred,
                (pre_mla_ce - post_mla_ce).detach().clamp(-1.0, 1.0),
                delta=self.args.controller_huber_delta,
            )
        avg_router_entropy = (
            float(torch.stack(adaptive_router_entropies).mean().item())
            if adaptive_router_entropies
            else 0.0
        )

        aux = {
            "patch_logits": patch_logits.detach(),
            "patch_weights": patch_weights.detach(),
            "mla_mask": mla_mask.detach(),
        }
        stats = ForwardStats(
            patch_hist=patch_hist,
            avg_continue=(
                float(torch.stack(adaptive_continue_rates).mean().item())
                if adaptive_continue_rates
                else 0.0
            ),
            mla_fraction=float(mla_mask.mean().item()),
            mla_fraction_raw=mla_fraction_raw,
            avg_router_entropy=avg_router_entropy,
            patch_diversity_loss=float(patch_diversity_loss.item()),
            continue_budget_loss=float(continue_budget_loss.item()),
            mla_budget_loss=float(mla_budget_loss.item()),
            continue_controller_loss=float(continue_controller_loss.item()),
            mla_controller_loss=float(mla_controller_loss.item()),
            avg_fast_mix=float(torch.stack(mix_means).mean().item()),
            avg_slow_mix=float(1.0 - torch.stack(mix_means).mean().item()),
            engram_gate_mean=float(sum(engram_gate_means) / max(len(engram_gate_means), 1)),
            avg_continue_gain_pred=float(sum(continue_gain_preds) / max(len(continue_gain_preds), 1)),
            avg_mla_gain_pred=float(mla_gain_pred.detach().mean().item()),
        )
        if y is None:
            return logits, aux, stats
        loss = F.cross_entropy(logits.reshape(-1, 256).float(), y.reshape(-1))
        if y.size(1) > 1:
            aux_logits = self.mtp_head(self.final_norm(h[:, :-1]))
            loss = loss + 0.1 * F.cross_entropy(
                aux_logits.reshape(-1, 256).float(), y[:, 1:].reshape(-1)
            )
        loss = loss + self.args.router_budget_lambda * (
            continue_budget_loss + mla_budget_loss
        )
        loss = loss + self.args.patch_diversity_lambda * patch_diversity_loss
        loss = loss + continue_controller_loss + mla_controller_loss
        return loss, aux, stats

    @torch.no_grad()
    def ttt_update(self, hidden: Tensor, logits: Tensor, targets: Tensor) -> None:
        self.gdn.adapter.update(hidden, logits, targets)
        self.mla_adapter.update(hidden, logits, targets)


def write_trace(trace_file: Path, payload: dict[str, object]) -> None:
    with trace_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def cuda_mem_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"cuda_alloc_mb": 0.0, "cuda_reserved_mb": 0.0, "cuda_peak_mb": 0.0}
    return {
        "cuda_alloc_mb": torch.cuda.memory_allocated(device) / (1024**2),
        "cuda_reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        "cuda_peak_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
    }


def tensor_summary(t: Tensor) -> dict[str, object]:
    td = t.detach().float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": float(td.min().item()) if t.numel() else 0.0,
        "max": float(td.max().item()) if t.numel() else 0.0,
        "mean": float(td.mean().item()) if t.numel() else 0.0,
        "std": float(td.std().item()) if t.numel() > 1 else 0.0,
    }


def write_failure_bundle(
    args: Hyperparameters,
    step: int,
    event: str,
    batch_x: Tensor,
    batch_y: Tensor,
    aux: dict[str, Tensor] | None,
    extra: dict[str, object] | None = None,
) -> None:
    out_dir = ensure_dir(Path(args.fail_bundle_dir) / args.run_id / f"step_{step:05d}_{event}")
    summary = {
        "event": event,
        "step": step,
        "run_id": args.run_id,
        "seed": args.seed,
        "arch_version": args.arch_version,
        "batch_x": tensor_summary(batch_x),
        "batch_y": tensor_summary(batch_y),
    }
    if extra:
        summary.update(extra)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    torch.save(batch_x.detach().cpu(), out_dir / "batch_x.pt")
    torch.save(batch_y.detach().cpu(), out_dir / "batch_y.pt")
    if aux:
        for idx, (name, tensor) in enumerate(aux.items()):
            if idx >= args.trace_max_tensors:
                break
            torch.save(tensor.detach().cpu(), out_dir / f"{name}.pt")


def load_validation_seq(val_bytes: Tensor, seq_len: int) -> Tensor:
    usable = (val_bytes.numel() - 1) // seq_len * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for TRAIN_SEQ_LEN={seq_len}")
    return val_bytes[: usable + 1]


@torch.no_grad()
def eval_raw_bytes(
    args: Hyperparameters,
    model: AdaptiveToyModel,
    val_bytes: Tensor,
    device: torch.device,
    use_ttt: bool = False,
    recur_limit: int | None = None,
    mla_topk_scale: float = 1.0,
    hard_routing: bool = True,
) -> float:
    seq_len = args.train_seq_len
    val_seq = load_validation_seq(val_bytes, seq_len)
    total_seqs = (val_seq.numel() - 1) // seq_len
    batch_tokens = args.train_batch_tokens
    if device.type == "cuda":
        total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        if total_mem_gb <= 8.5:
            batch_tokens = min(batch_tokens, 4096)
    batch_seqs = max(1, batch_tokens // seq_len)
    eval_recur_limit = recur_limit
    if eval_recur_limit is None:
        eval_recur_limit = args.recur_max
        if args.recur_max_active_default > 0:
            eval_recur_limit = min(args.recur_max_active_default, args.recur_max)
    loss_sum = 0.0
    token_count = 0
    model.eval()
    if use_ttt:
        model.reset_ttt()
    for start in range(0, total_seqs, batch_seqs):
        end = min(start + batch_seqs, total_seqs)
        raw = val_seq[start * seq_len : end * seq_len + 1].to(device=device, dtype=torch.int64)
        x = raw[:-1].reshape(-1, seq_len)
        y = raw[1:].reshape(-1, seq_len)
        with autocast_context(device):
            logits, _, _ = model(
                x,
                None,
                tau=max(0.5, args.tau_floor_early),
                use_ttt=use_ttt,
                recur_limit=eval_recur_limit,
                mla_topk_scale=mla_topk_scale,
                hard_routing=hard_routing,
            )
            batch_loss = F.cross_entropy(logits.reshape(-1, 256).float(), y.reshape(-1))
        loss_sum += float(batch_loss.item()) * y.numel()
        token_count += int(y.numel())
        if use_ttt:
            hidden = model.byte_emb(x)
            model.ttt_update(hidden, logits, y)
    model.train()
    return (loss_sum / max(token_count, 1)) / math.log(2.0)


def build_optimizers(args: Hyperparameters, model: AdaptiveToyModel):
    matrix_params = []
    scalar_params = []
    embed_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("byte_emb"):
            embed_params.append(p)
        elif p.ndim == 2:
            matrix_params.append(p)
        else:
            scalar_params.append(p)
    opt_embed = torch.optim.Adam(
        [{"params": embed_params, "lr": args.embed_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=torch.cuda.is_available(),
    )
    opt_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    opt_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=torch.cuda.is_available(),
    )
    return {"embed": opt_embed, "muon": opt_muon, "scalar": opt_scalar}


def zero_grads(optimizers) -> None:
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)


def step_optimizers(optimizers) -> None:
    for opt in optimizers.values():
        opt.step()


def export_and_roundtrip(
    args: Hyperparameters,
    model: AdaptiveToyModel,
    val_bytes: Tensor,
    device: torch.device,
    code: str,
    eval_policy: EvalPolicy | None = None,
) -> None:
    torch.save(model.state_dict(), "toy_adaptive_raw.pt")
    raw_bytes = Path("toy_adaptive_raw.pt").stat().st_size
    code_bytes = len(code.encode("utf-8"))
    log0("")
    log0("+" + "-" * 58 + "+")
    log0("|  TOY MODEL EXPORT                                        |")
    log0("+" + "-" * 58 + "+")
    log0(f"|  Raw model:              {raw_bytes:>12,} bytes" + " " * 13 + "|")
    log0(f"|  Code size:              {code_bytes:>12,} bytes" + " " * 13 + "|")

    quant_used = "int8"
    roundtrip_loader = None
    blob = b""
    if args.artifact_format == "int6_zstd":
        try:
            blob, ratio = quantize_state_dict_int6_zstd(model.state_dict())
            with open("toy_adaptive.int6.ptz", "wb") as f:
                f.write(blob)
            qbytes = Path("toy_adaptive.int6.ptz").stat().st_size
            total = qbytes + code_bytes
            fits = "YES" if total <= 16_000_000 else "NO"
            log0(
                f"|  INT6+zstd model:       {qbytes:>12,} bytes ({ratio:.1f}x)"
                + " " * max(0, 18 - len(f"({ratio:.1f}x)"))
                + "|"
            )
            log0(
                f"|  INT6 total:            {total:>12,} bytes ({total / 16_000_000 * 100:.1f}% of 16MB)|"
            )
            log0(f"|  INT6 fits 16MB?        {fits}" + " " * max(0, 27 - len(fits)) + "|")
            quant_used = "int6"
            roundtrip_loader = dequantize_state_dict_int6_zstd
        except Exception as exc:
            log0(f"|  INT6+zstd skipped:     {type(exc).__name__}: {exc}" + " " * 3 + "|")
    if quant_used == "int8":
        import io
        import zlib

        quant_obj, _ = quantize_state_dict_int8(model.state_dict())
        buf = io.BytesIO()
        torch.save(quant_obj, buf)
        blob = zlib.compress(buf.getvalue(), level=9)
        with open("toy_adaptive.int8.ptz", "wb") as f:
            f.write(blob)
        qbytes = Path("toy_adaptive.int8.ptz").stat().st_size
        total = qbytes + code_bytes
        fits = "YES" if total <= 16_000_000 else "NO"
        log0(f"|  INT8+zlib model:       {qbytes:>12,} bytes" + " " * 18 + "|")
        log0(f"|  INT8 total:            {total:>12,} bytes" + " " * 18 + "|")
        log0(f"|  INT8 fits 16MB?        {fits}" + " " * max(0, 27 - len(fits)) + "|")
        roundtrip_loader = lambda b: dequantize_state_dict_int8(  # noqa: E731
            torch.load(io.BytesIO(zlib.decompress(b)), map_location="cpu", weights_only=False)
        )
    log0("+" + "-" * 58 + "+")

    if roundtrip_loader is None:
        raise RuntimeError("No roundtrip loader selected for export")
    model.load_state_dict(roundtrip_loader(blob), strict=True)
    eval_policy = eval_policy or EvalPolicy(
        recur_limit=(
            min(args.recur_max_active_default, args.recur_max)
            if args.recur_max_active_default > 0
            else args.recur_max
        ),
        mla_topk_scale=1.0,
        hard_routing=True,
    )
    q_bpb = eval_raw_bytes(
        args,
        model,
        val_bytes,
        device,
        use_ttt=args.enable_ttt,
        recur_limit=eval_policy.recur_limit,
        mla_topk_scale=eval_policy.mla_topk_scale,
        hard_routing=eval_policy.hard_routing,
    )
    log0(f"final_{quant_used}_roundtrip_raw_byte_bpb:{q_bpb:.8f}")


def adaptive_grad_skip_threshold(
    args: Hyperparameters,
    step: int,
    grad_ema: float | None,
) -> float:
    if step <= args.grad_skip_warmup_steps or grad_ema is None:
        return args.max_grad_skip_norm
    return max(args.max_grad_skip_norm, 6.0 * grad_ema)


def should_use_hard_routing(args: Hyperparameters, step: int, recovery: RecoveryState) -> bool:
    return (
        step >= args.router_hard_after_step
        and recovery.lr_scale >= 0.7
        and recovery.soft_routing_cooldown <= 0
    )


def main() -> None:
    args = Hyperparameters()
    if args.num_layers != 1:
        raise ValueError("Set NUM_LAYERS=1 for the toy trainer; recurrence supplies depth.")
    if args.recur_min > args.recur_max:
        raise ValueError("RECUR_MIN must be <= RECUR_MAX")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_cuda_backends()
    set_seed(args.seed)
    ensure_dir(args.fail_bundle_dir)
    trace_dir = ensure_dir(args.trace_dir)
    trace_file = trace_dir / f"{args.run_id}.jsonl"
    run_summary_path = trace_dir / f"{args.run_id}_summary.json"
    code = Path(__file__).read_text(encoding="utf-8")

    effective_batch_tokens = args.train_batch_tokens
    effective_val_bytes = args.val_max_bytes
    if device.type == "cuda":
        total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        if total_mem_gb <= 8.5:
            effective_batch_tokens = min(effective_batch_tokens, 4096)
            effective_val_bytes = min(effective_val_bytes, 128 * 1024)

    log0("=" * 72)
    log0(f"TOY ADAPTIVE RUN: {args.run_id}")
    log0(f"device:{device} seed:{args.seed} arch_version:{args.arch_version}")
    log0(
        f"model_dim:{args.model_dim} heads:{args.num_heads} seq_len:{args.train_seq_len} "
        f"batch_tokens:{effective_batch_tokens} recur:{args.recur_min}->{args.recur_max}"
    )
    log0("RTX 3070 Ti target: single-GPU, raw-byte toy, no DDP")
    log0(
        f"pgdn_backend:{args.pgdn_backend}/chunk{args.pgdn_chunk_size}"
        f"{'+compile' if args.pgdn_backend == 'compiled_loop' and args.pgdn_compile and device.type == 'cuda' else '+eager'} "
        f"| mla_dense_fallback:{args.mla_dense_fallback_frac:.2f}"
    )
    log0("=" * 72)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    token_byte_lut = build_token_byte_lut(sp, max(int(sp.vocab_size()), 1024))
    train_loader = RawByteLoader(RawByteStream(args.train_files, token_byte_lut), device)
    val_bytes = load_validation_bytes(args.val_files, token_byte_lut, effective_val_bytes)
    corpus_path = Path(f"toy_compile_corpus_{args.run_id}.txt")
    if not corpus_path.exists():
        write_compile_corpus(corpus_path, args.train_files, token_byte_lut, args.corpus_bytes)

    model = AdaptiveToyModel(args).to(device)
    optimizers = build_optimizers(args, model)
    zero_grads(optimizers)
    default_recur_limit = args.recur_max
    if args.recur_max_active_default > 0:
        default_recur_limit = min(args.recur_max_active_default, args.recur_max)
    best_bpb = float("inf")
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    best_eval_policy = EvalPolicy(
        recur_limit=default_recur_limit,
        mla_topk_scale=1.0,
        hard_routing=False,
    )
    tau = 1.5
    recovery = RecoveryState()
    runtime = AdaptiveRunState(recur_limit=default_recur_limit)
    grad_norm_ema = None

    for step in range(1, args.iterations + 1):
        x, y = train_loader.next_batch(effective_batch_tokens, args.train_seq_len)
        t0 = time.perf_counter()
        aux = None
        hard_routing = should_use_hard_routing(args, step, recovery)
        try:
            with autocast_context(device):
                loss, aux, stats = model(
                    x,
                    y,
                    tau=tau,
                    use_ttt=False,
                    recur_limit=runtime.recur_limit,
                    mla_topk_scale=runtime.mla_topk_scale,
                    hard_routing=hard_routing,
                )
            if args.fail_on_nonfinite and not torch.isfinite(loss):
                write_failure_bundle(
                    args, step, "nonfinite_loss", x, y, aux, {"loss": float(loss.detach().item())}
                )
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.backward()
            group_grad_norms = grad_norm_by_group(model)
            top_grads = top_grad_params(model)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            grad_norm_t = torch.as_tensor(grad_norm)
            if torch.isfinite(grad_norm_t):
                curr_gn = float(grad_norm_t.item())
                grad_norm_ema = curr_gn if grad_norm_ema is None else (0.95 * grad_norm_ema + 0.05 * curr_gn)
            skip_threshold = adaptive_grad_skip_threshold(args, step, grad_norm_ema)
            if (
                args.skip_on_grad_spike
                and step > args.grad_skip_warmup_steps
                and torch.isfinite(grad_norm_t)
                and grad_norm_t.item() > skip_threshold
            ):
                tau = recover_from_grad_spike(args, optimizers, recovery, tau)
                adapt_runtime_on_spike(args, runtime)
                zero_grads(optimizers)
                if args.debug_level == "trace":
                    write_trace(
                        trace_file,
                        {
                            "step": step,
                            "event": "grad_spike_skip",
                            "grad_norm": float(grad_norm_t.item()),
                            "grad_norm_ema": grad_norm_ema,
                            "skip_threshold": skip_threshold,
                            "max_grad_skip_norm": args.max_grad_skip_norm,
                            "grad_skip_warmup_steps": args.grad_skip_warmup_steps,
                            "lr_scale": recovery.lr_scale,
                            "consecutive_grad_spikes": recovery.consecutive_grad_spikes,
                            "runtime_recur_limit": runtime.recur_limit,
                            "runtime_mla_topk_scale": runtime.mla_topk_scale,
                            "hard_routing": hard_routing,
                            "group_grad_norms": group_grad_norms,
                            "top_grad_params": top_grads,
                            **cuda_mem_stats(device),
                        },
                    )
                log0(
                    f"[train {step:>4d}/{args.iterations}] "
                    f"grad_spike {grad_norm_t.item():>7.3f} > {skip_threshold:>7.3f} | "
                    f"lr_scale {recovery.lr_scale:>4.2f} | tau {tau:>4.2f} | "
                    f"recur {runtime.recur_limit:>1d} | mla_scale {runtime.mla_topk_scale:>4.2f} | "
                    f"route {'hard' if hard_routing else 'soft'} | skip_update"
                )
                if best_step and recovery.spikes_since_rollback >= args.rollback_spike_patience:
                    model.load_state_dict(best_state, strict=True)
                    scale_optimizer_lrs(optimizers, 1.0)
                    recovery.lr_scale = 1.0
                    recovery.consecutive_grad_spikes = 0
                    recovery.spikes_since_rollback = 0
                    recovery.soft_routing_cooldown = max(
                        recovery.soft_routing_cooldown, args.soft_routing_cooldown_steps
                    )
                    runtime.recur_limit = max(args.recur_min + 1, runtime.recur_limit)
                    runtime.mla_topk_scale = max(runtime.mla_topk_scale, 0.7)
                    reset_optimizer_state(optimizers["muon"])
                    reset_optimizer_state(optimizers["scalar"])
                    reset_optimizer_state(optimizers["embed"])
                    tau = max(tau, 0.9)
                    log0(
                        f"[train {step:>4d}/{args.iterations}] "
                        f"rollback_to_best step {best_step} | tau {tau:>4.2f} | route soft"
                    )
                continue
            if args.fail_on_nonfinite and not torch.isfinite(torch.as_tensor(grad_norm)):
                write_failure_bundle(
                    args, step, "nonfinite_grad", x, y, aux, {"grad_norm": float(grad_norm)}
                )
                raise RuntimeError(f"Non-finite grad norm at step {step}")
            step_optimizers(optimizers)
            recovery.consecutive_grad_spikes = 0
            recovery.spikes_since_rollback = 0
            if recovery.soft_routing_cooldown > 0:
                recovery.soft_routing_cooldown -= 1
            adapt_runtime_on_stable_step(args, runtime)
            zero_grads(optimizers)
        except Exception as exc:
            write_failure_bundle(
                args,
                step,
                "exception",
                x,
                y,
                aux,
                {"error": repr(exc), "traceback": traceback.format_exc()},
            )
            raise

        step_ms = 1000.0 * (time.perf_counter() - t0)
        mem = cuda_mem_stats(device)
        if args.debug_level == "trace" and step % args.trace_every == 0:
            write_trace(
                trace_file,
                {
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "grad_norm": float(grad_norm),
                    "ms": step_ms,
                    "tau": tau,
                    "patch_hist": stats.patch_hist,
                    "avg_continue": stats.avg_continue,
                    "mla_fraction": stats.mla_fraction,
                    "mla_fraction_raw": stats.mla_fraction_raw,
                    "avg_router_entropy": stats.avg_router_entropy,
                    "patch_diversity_loss": stats.patch_diversity_loss,
                    "continue_budget_loss": stats.continue_budget_loss,
                    "mla_budget_loss": stats.mla_budget_loss,
                    "continue_controller_loss": stats.continue_controller_loss,
                    "mla_controller_loss": stats.mla_controller_loss,
                    "avg_fast_mix": stats.avg_fast_mix,
                    "avg_slow_mix": stats.avg_slow_mix,
                    "engram_gate_mean": stats.engram_gate_mean,
                    "avg_continue_gain_pred": stats.avg_continue_gain_pred,
                    "avg_mla_gain_pred": stats.avg_mla_gain_pred,
                    "grad_norm_ema": grad_norm_ema,
                    "lr_scale": recovery.lr_scale,
                    "runtime_recur_limit": runtime.recur_limit,
                    "runtime_mla_topk_scale": runtime.mla_topk_scale,
                    "hard_routing": hard_routing,
                    "group_grad_embed": group_grad_norms["embed"],
                    "group_grad_matrix": group_grad_norms["matrix"],
                    "group_grad_scalar": group_grad_norms["scalar"],
                    **mem,
                },
            )
        if step % args.train_log_every == 0 or step == 1:
            log0(
                format_train_line(
                    step=step,
                    total_steps=args.iterations,
                    loss=float(loss.item()),
                    bpb=float(loss.item() / math.log(2.0)),
                    tau=float(tau),
                    stats=stats,
                    step_ms=float(step_ms),
                    mem=mem,
                    recur_limit=runtime.recur_limit,
                    mla_topk_scale=runtime.mla_topk_scale,
                )
            )
        if step % args.val_loss_every == 0 or step == args.iterations:
            val_bpb = eval_raw_bytes(
                args,
                model,
                val_bytes,
                device,
                use_ttt=False,
                recur_limit=runtime.recur_limit,
                mla_topk_scale=runtime.mla_topk_scale,
                hard_routing=hard_routing,
            )
            ttt_bpb = None
            if args.enable_ttt:
                ttt_bpb = eval_raw_bytes(
                    args,
                    model,
                    val_bytes,
                    device,
                    use_ttt=True,
                    recur_limit=runtime.recur_limit,
                    mla_topk_scale=runtime.mla_topk_scale,
                    hard_routing=hard_routing,
                )
            log0(format_val_line(step, args.iterations, val_bpb, ttt_bpb))
            if val_bpb < best_bpb:
                best_bpb = val_bpb
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
                best_eval_policy = EvalPolicy(
                    recur_limit=runtime.recur_limit,
                    mla_topk_scale=runtime.mla_topk_scale,
                    hard_routing=hard_routing,
                )
            run_summary_path.write_text(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "step": step,
                        "best_raw_byte_bpb": best_bpb,
                        "best_step": best_step,
                        "last_raw_byte_bpb": val_bpb,
                        "tau": tau,
                        "lr_scale": recovery.lr_scale,
                        "runtime_recur_limit": runtime.recur_limit,
                        "runtime_mla_topk_scale": runtime.mla_topk_scale,
                        "avg_continue": stats.avg_continue,
                        "mla_fraction": stats.mla_fraction,
                        "mla_fraction_raw": stats.mla_fraction_raw,
                        "avg_router_entropy": stats.avg_router_entropy,
                        "continue_controller_loss": stats.continue_controller_loss,
                        "mla_controller_loss": stats.mla_controller_loss,
                        "avg_fast_mix": stats.avg_fast_mix,
                        "avg_slow_mix": stats.avg_slow_mix,
                        "engram_gate_mean": stats.engram_gate_mean,
                        "avg_continue_gain_pred": stats.avg_continue_gain_pred,
                        "avg_mla_gain_pred": stats.avg_mla_gain_pred,
                        **mem,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        tau_floor = args.tau_floor_early if step < args.tau_floor_before_step else 0.45
        tau = max(tau_floor, tau * 0.992)

    log0(f"best_raw_byte_bpb:{best_bpb:.8f}")
    if best_step:
        model.load_state_dict(best_state, strict=True)
        log0(f"exporting_best_checkpoint_from_step:{best_step}")
    export_and_roundtrip(args, model, val_bytes, device, code, eval_policy=best_eval_policy)


if __name__ == "__main__":
    main()
