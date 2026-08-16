"""One-off: block-diffusion convert the 128M AR checkpoint (guide §9 / tri-mode).

Loads the checkpoint's OWN stored config (so the model shape matches exactly),
applies a short-conversion override set sized for the 8 GB 3070 Ti, and runs the
block-diffusion objective (block_len=32, block-causal attention) on the existing
FineWeb-edu .bin. Produces nanolab/out/diffusion128_block32/best.pt for re-bench.
"""
import torch

from nanolab.config import build_config
from nanolab.diffusion import train

CKPT = "nanolab/out/run128m_fineweb_2k/best.pt"

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = build_config(None, sd["cfg"])          # exact architecture of the checkpoint

# conversion overrides — small batch/ctx because diffusion_loss materializes full
# (B,T,vocab) logits (no fused-CE on the masked path), the 8 GB memory limiter.
cfg.run_name = "diffusion128_block32"
cfg.block_size = 256
cfg.batch_size = 4
cfg.grad_accum = 2          # keep ~2K tokens/step effective, lean activation memory
cfg.max_steps = 700
cfg.eval_interval = 175
cfg.eval_iters = 20
cfg.log_interval = 25
cfg.warmup_steps = 40
cfg.optimizer = "adamw"     # Muon's Newton-Schulz faulted on the large early grads
cfg.lr = 2e-4
cfg.schedule = "cosine"
cfg.grad_checkpoint = True
cfg.compile = False
cfg.dtype = "bf16"
cfg.device = "cuda"
cfg.mem_fraction = 0.92

print(f"converting {CKPT} -> {cfg.run_name} | block_len=32 "
      f"bs{cfg.batch_size}x ga{cfg.grad_accum}/ctx{cfg.block_size} steps{cfg.max_steps}",
      flush=True)
import traceback
try:
    best = train(cfg, init_ckpt=CKPT, anneal_steps=0, complementary=True, block_len=32)
    print(f"DONE best masked-val loss {best:.4f}", flush=True)
except Exception:
    print("CONVERT FAILED:\n" + traceback.format_exc(), flush=True)
    raise
