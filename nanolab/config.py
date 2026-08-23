"""
nanolab.config — single source of truth for model/training configuration.

This is the companion package referenced throughout
``modern-small-lm-training-guide.md``. Every architectural and training lever
the guide discusses is one field here, and every "run" is one ``Config``.

Three principles drive the design (guide §0):
  1. Isolate variables  — change ONE field per run.
  2. Instrument everything — see ``train.py`` logging.
  3. Short runs, one variable at a time — see the phase2 preset + experiments.py.

The recommended 128M config (guide §2.2) is the ``phase1`` preset below. The
defaults bake in the *empirically winning* choices from this repo's own old
runs (``logs/ablations/.../champion.json``): Muon matrix LR 0.025, momentum
0.99, grad-clip ~0.3, gated-attention + value-residual on attention.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, fields


# Registries — the pluggable axes the guide calls out (§2.5, §4.2, §5.3).
MIXERS = ("attention", "mingru", "mamba2", "gdn", "mla")
# Compact LAYER_TYPES aliases used by the hypercascade trainers.
MIXER_ALIASES = {"attn": "attention", "mamba": "mamba2"}
OPTIMIZERS = (
    "muon_ns5_adamw", "muon_ns3_adamw", "muon_polar_adamw",
    "normuon_adamw", "muown_adamw", "mona_adamw", "mimuon_adamw",
    "adamw", "lion", "cautious_adamw", "cautious_lion", "sgd_momentum",
    "sophia", "schedule_free_adamw", "prodigy", "soap_adamw",
)
OPTIMIZER_ALIASES = {
    "muon": "muon_ns5_adamw", "neomuon": "muon_ns3_adamw",
    "polar": "muon_polar_adamw", "normuon": "normuon_adamw",
    "muown": "muown_adamw", "mona": "mona_adamw", "mimuon": "mimuon_adamw",
    "sgd": "sgd_momentum", "schedulefree": "schedule_free_adamw",
    "soap": "soap_adamw",
}
SCHEDULES = ("cosine", "constant", "wsd", "plateau")
DIFFUSION_MODES = ("none", "full", "block")


@dataclass
class Config:
    # ---- identity / bookkeeping (guide §6.4: name runs by the variable changed) ----
    run_name: str = "run"
    seed: int = 1337
    out_dir: str = "nanolab/out"

    # ---- model: the modern stack vs GPT-2 defaults (guide §2) ----
    n_layer: int = 12
    d_model: int = 768
    n_head: int = 12
    n_kv_head: int = 0           # 0 -> = n_head (MHA). <n_head enables GQA (§2.1).
    head_dim: int = 64           # 0 -> d_model // n_head
    block_size: int = 1024       # context length
    vocab_size: int = 50304      # padded GPT-2 BPE (50257 -> mult of 64 for speed)
    dropout: float = 0.0

    pos: str = "rope"            # rope | none (RoPE is strictly better, §2.1)
    norm: str = "rmsnorm"        # rmsnorm | layernorm  (pre-norm always)
    qk_norm: bool = True         # RMSNorm on Q,K — stabilizes training (§2.1)
    ffn: str = "swiglu"          # swiglu | relu2 | gelu | moe (§2.1, §2.3)
    # MoE (§2.1): sparse experts — big capacity at low active compute.
    moe_experts: int = 8         # number of FFN experts
    moe_top_k: int = 2           # experts active per token
    moe_aux_weight: float = 0.01 # load-balancing auxiliary-loss weight
    # MLA (§2.1, DeepSeek): low-rank KV compression + decoupled RoPE.
    kv_lora_rank: int = 0        # 0 -> d_model//4 ; KV latent dim
    q_lora_rank: int = 0         # 0 -> d_model//2 ; Q latent dim (0-len disables)
    rope_head_dim: int = 0       # 0 -> head_dim//2 ; decoupled-RoPE sub-dimension
    tie_embeddings: bool = True  # share input/output embeddings (saves params)
    zero_init_proj: bool = True  # zero-init output projections — cheap stabilizer
    rope_base: float = 10000.0
    mup: bool = False            # muP-style 1/d attention + LR scaling (§10)
    mup_base_width: int = 256    # d_model at which HPs were tuned; transfer target
    # Per-layer standard-parametrization LR prescription (Everett et al.,
    # arXiv:2407.05872, Table 1, Adam column): hidden and readout learning rates
    # scale as 1/sqrt(width) while the embedding LR stays constant. Mutually
    # exclusive with ``mup`` -- they are different parametrizations. See the caveats
    # in ``optim.build_optimizers``: the prescription is stated for pure Adam, and
    # tied embeddings prevent giving the readout its own rate.
    per_layer_sp: bool = False
    # Multiply ONLY the embedding/head LR by this factor, under standard
    # parametrization. The Kalra & Barkeshli probe (arXiv:2605.21486) sets it to the
    # width ratio to test whether the embedding LR alone carries muP's advantage.
    embed_lr_mult: float = 1.0

    # ---- sequence mixer (guide §2.5) — the one A/B flag ----
    mixer: str = "attention"
    # Per-layer mixer stack. Empty = every layer uses ``mixer``. Compact syntax
    # matches the hypercascade LAYER_TYPES recipe, e.g. ``gdn*10,attention*2``
    # or ``gdn*3,attention,gdn*3,attention,gdn*3,attention``. Length must equal
    # ``n_layer``. Unknown names and length mismatches fail closed.
    layer_mixers: str = ""
    # attention-only knobs (champion variant from old runs):
    gated_attention: bool = True   # per-head sigmoid gate on attn output
    value_residual: bool = True    # blend layer-0 values into later layers
    # recurrent-mixer knobs (mamba2 / gdn):
    d_state: int = 64
    mixer_chunk: int = 64

    # ---- optimizer (guide §4) ----
    optimizer: str = "muon_ns5_adamw"  # see OPTIMIZERS
    lr: float = 6e-4               # AdamW/SGD/Lion peak LR (scalar/embed groups)
    matrix_lr: float = 0.025       # Muon LR for 2D hidden weights (old-run winner)
    muon_momentum: float = 0.99    # old-run winner (0.95 in guide §4.4 is fine too)
    beta1: float = 0.9
    beta2: float = 0.95            # NB: 0.95 for LM pretraining, not 0.999 (§4.3)
    eps: float = 1e-8
    weight_decay: float = 0.1      # decoupled, 2D params only (§4.3)
    grad_clip: float = 1.0         # global-norm clip (§5.4). old runs used ~0.3.
    sophia_hess_interval: int = 10 # Sophia: refresh diagonal Hessian every k steps (§4.2)
    muon_ns_steps: int = 5
    normuon_beta2: float = 0.95
    mona_beta_a: float = 0.99
    mona_alpha: float = 0.0         # 0 -> paper rule -1/(2*(1-beta_a))
    mimuon_singular_gap: float = 1e-3
    muown_direction_scale: float = 0.2
    soap_precondition_frequency: int = 10
    soap_max_precond_dim: int = 2048
    optimizer_state_limit_gib: float = 52.0

    # ---- schedule (guide §5.3) ----
    schedule: str = "cosine"       # see SCHEDULES
    warmup_steps: int = 256        # ~1-5% of total (§5.2)
    lr_floor_frac: float = 0.1     # cosine/wsd floor as fraction of peak
    wsd_decay_frac: float = 0.2    # WSD: fraction of run spent decaying
    plateau_patience: int = 5      # ReduceLROnPlateau (reactive schedule)
    plateau_factor: float = 0.5

    # ---- data (guide §3) ----
    dataset: str = "tinystories"   # tinystories | shakespeare | hf | fineweb_bin
    hf_dataset: str = ""           # e.g. roneneldan/TinyStories, Skylion007/openwebtext
    hf_config: str = ""            # subset config, e.g. sample-10BT for fineweb-edu
    data_dir: str = "nanolab/data"
    tokenizer: str = "gpt2"        # gpt2 (tiktoken) | char
    fineweb_pattern: str = ""      # glob for pre-tokenized .bin shards (reuse repo data)
    # curriculum (§3): "none" | "seqlen" (grow context short->full, ~3x faster
    # time-to-accuracy) | "difficulty" (easy->hard data frontier; needs data
    # prepared with `prep_fineweb --sort_difficulty`).
    curriculum: str = "none"
    curriculum_start_len: int = 0  # 0 -> block_size // 4
    curriculum_frac: float = 0.5   # fraction of run to reach full length/frontier

    # ---- training loop (guide §4.1) ----
    batch_size: int = 16           # micro-batch (sequences per fwd)
    grad_accum: int = 4            # gradient accumulation -> large effective batch
    max_steps: int = 5000
    lr_max_steps: int = 0  # if >0, schedule decays over this many steps instead of max_steps
    eval_interval: int = 250
    eval_iters: int = 100
    eval_train: bool = True        # also eval the train split (doubles eval cost)
    log_interval: int = 10
    ckpt_interval: int = 1000

    # ---- systems / precision (guide §7) ----
    dtype: str = "bf16"            # bf16 | fp16 | fp32 (bf16 default on Ampere+)
    compile: bool = True           # torch.compile (auto-off for recurrent mixers)
    grad_checkpoint: bool = False  # activation checkpointing for long sequences
    device: str = "auto"           # auto | cuda | cpu
    # fused linear cross-entropy (guide §7.1, Liger-style): never materialize the
    # full B*T*vocab logits — chunk over tokens. Big VRAM win -> bigger batch.
    fused_ce: bool = False
    # 16 chunks is the measured throughput optimum at ctx1024/vocab-50304 on the
    # 3070 Ti: fewer chunks blow up the fp32 logit intermediates (chunks=2 → 14 GB
    # → thrash), more chunks add launch overhead. 16 is fastest AND frees ~1.4 GB
    # vs 8 (4.2 vs 5.6 GB), which is what lets bs32 fit. See probe_perf / sweep_gpu.
    fused_ce_chunks: int = 16
    tf32: bool = True              # TF32 matmul/cudnn on Ampere+ (free throughput)
    # VRAM cap (guide §7): on an 8 GB Windows/WDDM card, an over-budget step
    # silently spills to host RAM over PCIe (~25x slower) instead of OOMing —
    # 100% util, ~60 W, multi-second steps that look like a hang. Capping the
    # allocator makes that a clean OutOfMemoryError instead. 0 disables the cap.
    mem_fraction: float = 0.92

    # ---- weight initialization from a pre-trained checkpoint ----
    init_ckpt: str = ""  # weights-only load; skipped when RESUME=1 is active

    # ---- diffusion / tri-mode (NVIDIA Nemotron-style AR+diffusion+self-spec) ----
    diffusion_mode: str = "none"          # none | full | block
    diffusion_init_ckpt: str = ""         # AR checkpoint to adapt from (required if mode != none)
    diffusion_anneal_steps: int = 400     # causal→bidirectional annealing steps (full mode)
    diffusion_block_len: int = 32         # block length for block-causal mode
    diffusion_complementary: bool = True  # LLaDA-2.0 complementary masking

    def __post_init__(self):
        self.optimizer = OPTIMIZER_ALIASES.get(self.optimizer, self.optimizer)
        if self.head_dim == 0:
            assert self.d_model % self.n_head == 0
            self.head_dim = self.d_model // self.n_head
        if self.n_kv_head == 0:
            self.n_kv_head = self.n_head
        assert self.mixer in MIXERS, f"mixer must be one of {MIXERS}"
        assert self.optimizer in OPTIMIZERS, f"optimizer must be one of {OPTIMIZERS}"
        assert self.schedule in SCHEDULES, f"schedule must be one of {SCHEDULES}"
        assert self.diffusion_mode in DIFFUSION_MODES, f"diffusion_mode must be one of {DIFFUSION_MODES}"
        parse_layer_mixers(self)  # fail closed on bad hybrid specs

    # -- convenience --------------------------------------------------------
    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum * self.block_size

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def estimate_params(self) -> int:
        """Rough non-embedding-aware parameter count for the attention path."""
        d, L, V = self.d_model, self.n_layer, self.vocab_size
        ffn_hidden = _swiglu_hidden(d) if self.ffn == "swiglu" else 4 * d
        per_layer = (
            4 * d * d                 # attn q,k,v,o (approx, ignores GQA shrink)
            + (3 if self.ffn == "swiglu" else 2) * d * ffn_hidden  # FFN
        )
        emb = V * d * (1 if self.tie_embeddings else 2)
        return L * per_layer + emb


def _resolve_mixer_name(name: str) -> str:
    key = name.strip().lower()
    key = MIXER_ALIASES.get(key, key)
    if key not in MIXERS:
        raise ValueError(f"unknown mixer {name!r}; known: {MIXERS} (aliases: {MIXER_ALIASES})")
    return key


def parse_layer_mixers(cfg: Config) -> tuple[str, ...]:
    """Expand ``cfg.layer_mixers`` into one mixer name per layer.

    Empty/blank → every layer is ``cfg.mixer``. ``name*N`` repeats. Length must
    equal ``cfg.n_layer``; mismatches and unknown names raise ``ValueError``.
    """
    spec = (cfg.layer_mixers or "").strip()
    if not spec:
        return tuple(cfg.mixer for _ in range(cfg.n_layer))
    kinds: list[str] = []
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            raise ValueError(f"empty entry in layer_mixers {spec!r}")
        if "*" in part:
            name, _, count_s = part.partition("*")
            name = name.strip()
            try:
                count = int(count_s.strip())
            except ValueError as e:
                raise ValueError(f"bad repeat count in {part!r}") from e
            if count <= 0:
                raise ValueError(f"repeat count must be >0 in {part!r}")
        else:
            name, count = part, 1
        kinds.extend([_resolve_mixer_name(name)] * count)
    if len(kinds) != cfg.n_layer:
        raise ValueError(
            f"layer_mixers specifies {len(kinds)} layers, n_layer={cfg.n_layer} "
            f"(spec={spec!r})"
        )
    return tuple(kinds)


def stack_is_attention_only(cfg: Config) -> bool:
    return all(k == "attention" for k in parse_layer_mixers(cfg))


def _swiglu_hidden(d_model: int) -> int:
    """SwiGLU hidden width ~= 2/3 * 4 * d_model, rounded to a multiple of 64
    so the gated FFN matches a plain 4x GELU MLP in parameter count (§2.1)."""
    h = int(2 / 3 * 4 * d_model)
    return (h + 63) // 64 * 64


# ---------------------------------------------------------------------------
# Presets — the phased experiment plan (guide §8).
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    # CPU-friendly sanity check: tiny model, char data, a handful of steps.
    # "Loss flat from step 0 = bug" — this preset is how you catch that fast.
    "cpu_smoke": dict(
        run_name="cpu_smoke", n_layer=2, d_model=128, n_head=4, head_dim=32,
        block_size=128, vocab_size=0, dataset="shakespeare", tokenizer="char",
        batch_size=8, grad_accum=1, max_steps=40, eval_interval=20, eval_iters=20,
        log_interval=5, ckpt_interval=40, dtype="fp32", compile=False,
        warmup_steps=10, optimizer="adamw", lr=3e-3, schedule="cosine",
    ),
    # Phase 0 — "watch a small model actually learn" (guide §8, §3).
    "phase0": dict(
        run_name="phase0", n_layer=6, d_model=384, n_head=6, head_dim=64,
        block_size=256, vocab_size=50304, dataset="tinystories", tokenizer="gpt2",
        hf_dataset="roneneldan/TinyStories",
        batch_size=32, grad_accum=2, max_steps=4000, eval_interval=250,
        warmup_steps=100, optimizer="adamw", lr=1e-3, schedule="cosine",
    ),
    # Phase 1 — instrumented 128M base run (guide §2.2, §8). 1-2B tokens.
    "phase1": dict(
        run_name="phase1", n_layer=12, d_model=768, n_head=12, head_dim=64,
        block_size=1024, vocab_size=50304, dataset="hf", tokenizer="gpt2",
        hf_dataset="HuggingFaceFW/fineweb-edu", hf_config="sample-10BT",
        batch_size=24, grad_accum=20, max_steps=20000, eval_interval=500,
        warmup_steps=700, optimizer="muon", schedule="cosine",
    ),
    # GPU-max — the 124M model tuned to saturate an 8 GB RTX 3070 Ti (guide §7).
    # Combines all the levers found by nanolab/bench_gpu: fused linear
    # cross-entropy (no full-logits tensor), gradient checkpointing (big batch),
    # batched-Muon (cheap optimizer), TF32 + bf16 + flash-SDPA, and a
    # GPU-resident dataloader. Fills VRAM (~7.5 GB) at ~100% utilization.
    "gpu_max": dict(
        run_name="gpu_max", n_layer=12, d_model=768, n_head=12, head_dim=64,
        block_size=1024, vocab_size=50304, dataset="shakespeare", tokenizer="gpt2",
        # bs32 + fused_ce_chunks=16 is the measured throughput peak on the 3070 Ti
        # (8 GB): ~13.7K tok/s / 25% MFU at ~6.1 GB reserved. chunks=16 frees the
        # VRAM (vs chunks=8) that lets bs32 fit — at chunks=8 bs32 thrashed past
        # 8 GB and *dropped* to ~9K. See probe_perf / sweep_gpu for the curves.
        batch_size=32, grad_accum=1, max_steps=200, eval_interval=100,
        eval_iters=20, log_interval=10, ckpt_interval=200,
        optimizer="muon", schedule="cosine", warmup_steps=20,
        fused_ce=True, fused_ce_chunks=16, grad_checkpoint=True, tf32=True,
        mem_fraction=0.92, compile=False,
    ),
    # Phase 2 — optimizer & LR experiments: SHORT runs, one variable each (§8).
    "phase2": dict(
        run_name="phase2", n_layer=12, d_model=768, n_head=12, head_dim=64,
        block_size=1024, vocab_size=50304, dataset="hf", tokenizer="gpt2",
        hf_dataset="HuggingFaceFW/fineweb-edu", hf_config="sample-10BT",
        batch_size=24, grad_accum=20, max_steps=2000, eval_interval=200,
        warmup_steps=100, optimizer="muon", schedule="cosine",
    ),
    # Diffusion phase 0 — adapt a trained xs (30M) AR model to full diffusion.
    # Starts from phase0_tinystories AR checkpoint; anneals causal→bidirectional
    # over 400 steps, then trains pure masked diffusion with complementary masking.
    "diffusion_phase0": dict(
        run_name="diffusion_phase0", n_layer=6, d_model=384, n_head=6, head_dim=64,
        block_size=256, vocab_size=50304, dataset="tinystories", tokenizer="gpt2",
        hf_dataset="roneneldan/TinyStories",
        batch_size=24, grad_accum=1, max_steps=1500, eval_interval=300, eval_iters=30,
        warmup_steps=40, optimizer="muon", lr=1e-3, schedule="cosine",
        grad_checkpoint=True, compile=False, tf32=True,
        diffusion_mode="full",
        diffusion_init_ckpt="nanolab/out/phase0_tinystories/best.pt",
        diffusion_anneal_steps=400, diffusion_complementary=True,
    ),
    # Diffusion block phase 1 — adapt the 128M AR model to block-causal diffusion.
    # Starts from run128m_10k; block_len=32 gives semi-AR generation (KV-cached).
    "diffusion_block_phase1": dict(
        run_name="diffusion_block_phase1", n_layer=12, d_model=768, n_head=12, head_dim=64,
        block_size=256, vocab_size=50304, dataset="hf", tokenizer="gpt2",
        hf_dataset="HuggingFaceFW/fineweb-edu", hf_config="sample-10BT",
        batch_size=4, grad_accum=2, max_steps=700, eval_interval=175, eval_iters=20,
        warmup_steps=40, optimizer="adamw", lr=2e-4, schedule="cosine",
        grad_checkpoint=True, compile=False, tf32=True, mem_fraction=0.92,
        fused_ce=True, fused_ce_chunks=16,
        diffusion_mode="block",
        diffusion_init_ckpt="nanolab/out/run128m_10k/best.pt",
        diffusion_block_len=32, diffusion_complementary=True,
    ),
    # Token-budget crossover replication (suite 14 architecture, 50M-token horizon).
    # Batch is a cluster knob: GH200 uses a large microbatch (see
    # nanolab.crossover_replicate.scale_to_token_budget) so tensor cores see a
    # wide GEMM. Warmup/eval cadence is kept in *tokens* matching suite 14
    # (256 steps × 4096 tok, eval every 0.8192M). Hybrids set --layer_mixers.
    "crossover50m": dict(
        run_name="crossover50m", n_layer=12, d_model=768, n_head=12, head_dim=64,
        block_size=512, vocab_size=50304, dataset="hf", tokenizer="gpt2",
        hf_dataset="HuggingFaceFW/fineweb-edu", hf_config="sample-10BT",
        batch_size=96, grad_accum=1, max_steps=1017, eval_interval=16,
        eval_iters=4, eval_train=False, log_interval=1, ckpt_interval=166,
        warmup_steps=21, optimizer="muon", lr=6e-4, schedule="cosine",
        mixer_chunk=32, fused_ce=True, fused_ce_chunks=16, compile=False,
        tf32=True, grad_checkpoint=False, mem_fraction=0.0, dtype="bf16",
    ),
}


def build_config(preset: str | None = None, overrides: dict | None = None) -> Config:
    base: dict = {}
    if preset:
        if preset not in PRESETS:
            raise SystemExit(f"unknown preset '{preset}'. choices: {list(PRESETS)}")
        base.update(PRESETS[preset])
    if overrides:
        base.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**base)


# ---------------------------------------------------------------------------
# CLI plumbing — expose every Config field as a flag so any run is reproducible
# from the command line (the guide's whole workflow is "one flag per run").
# ---------------------------------------------------------------------------
def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", type=str, default=None, choices=list(PRESETS))
    for f in fields(Config):
        name = f"--{f.name}"
        if f.type == "bool" or isinstance(f.default, bool):
            # support both --flag / --no-flag without forcing a default here
            parser.add_argument(name, type=_str2bool, default=None,
                                metavar="{true,false}")
        else:
            typ = {int: int, float: float}.get(f.type, str)
            if f.type == "int":
                typ = int
            elif f.type == "float":
                typ = float
            parser.add_argument(name, type=typ, default=None)


def _str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def config_from_args(args: argparse.Namespace) -> Config:
    overrides = {f.name: getattr(args, f.name) for f in fields(Config)
                 if getattr(args, f.name, None) is not None}
    return build_config(args.preset, overrides)
