from nanolab.config import build_config
from nanolab.mixers import swa_sdpa_backends, swa_auto_chunk
print("Which SDPA backends can serve a WINDOWED mask on this GPU, at the")
print("stages' real shapes. This is the one property that could not be")
print("checked off the target hardware.\n")
for block, bs, w in ((512, 32, 64), (2048, 8, 64), (2048, 8, 512)):
    cfg = build_config(None, dict(
        vocab_size=50304, n_layer=12, d_model=768, n_head=12, head_dim=64,
        block_size=block, batch_size=bs, device="cuda", dtype="bf16",
        compile=False, mixer="swa", swa_window=w, swa_sinks=4))
    rep = swa_sdpa_backends(cfg, "cuda", bs)
    served = [(n, r["peak_gb"]) for n, r in rep.items() if r["ok"]]
    names = ", ".join("{}({:.2f}GB)".format(n, g) for n, g in served)
    fast = [n for n, _ in served if n != "MATH"]
    verdict = "OK" if fast else "MATH ONLY -- would materialise B*H*T*T"
    print("  ctx{:<5} bs{:<3} w{:<4} chunk={:<4} {}".format(
        block, bs, w, swa_auto_chunk(block, w), verdict))
    print("      {}".format(names or "NONE"))
