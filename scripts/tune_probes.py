"""Four targeted probes behind the GH200 tuning sprint (2026-09-05).

Each answers a question that decides whether a speed-up is adoptable, not just
whether it is fast:

  datapath  — Batcher picks its sampler by FREE VRAM at construction time
              (data.should_gpu_resident). The GPU-resident path seeds a CUDA
              generator, the memmap fallback a CPU one, so the two paths train on
              DIFFERENT tokens from the same seed. If raising tenancy can push a
              late-starting worker under the threshold, tenancy is not the
              recipe-neutral knob every board assumes it is. Proves or refutes it
              by holding VRAM and reading back the first batch.
  gdnchunk  — GDN is the program's slowest arm (4.7x attention). `mixer_chunk`
              sets the WY chunk width; sweep it for speed AND report agreement
              against the O(T) reference, so the cost of adopting it is stated in
              nats-equivalent terms rather than assumed to be zero.
  evalsync  — evaluate() does one `.item()` per eval iteration (1220 of them per
              50M run, ~15% of the run is eval). Times the loop as written against
              one that transfers all `eval_iters` losses in a single copy, and
              checks the two means are bit-identical.
  moereal   — bench_gpu.synth_batch returns y==x, a trivially learnable task that
              collapses the MoE router and makes `_raw` rows look slower than
              `renorm` ones. Re-times the MoE arms on non-degenerate targets.

    python scripts/tune_probes.py datapath gdnchunk evalsync moereal
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

OUT = REPO / "nanolab/out/_tune"


def _cfg(arm_name, **extra):
    import os
    from nanolab.crossover_replicate import ARMS, job_config
    arm = next(a for a in ARMS if a.name == arm_name)
    os.environ["CROSSOVER_BATCH"] = "32"
    os.environ["CROSSOVER_BLOCK"] = "512"
    os.environ["CROSSOVER_TOKEN_BUDGET"] = "50000000"
    cfg = job_config({"id": "probe", "arm": arm.name, "mixer": arm.mixer,
                      "layer_mixers": arm.layer_mixers, "seed": 1337}, OUT)
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------- datapath
def probe_datapath() -> dict:
    from nanolab.data import Batcher, get_dataset, should_gpu_resident
    cfg = _cfg("attention")
    data_dir, _, _ = get_dataset(cfg)
    n_tok = Path(data_dir, "train.bin").stat().st_size // 2
    need_free = n_tok * 4 / 0.2
    free, total = torch.cuda.mem_get_info()
    res = {"n_tokens": n_tok, "corpus_gib": n_tok * 4 / 2**30,
           "free_needed_gib": need_free / 2**30,
           "free_now_gib": free / 2**30, "total_gib": total / 2**30}

    b_gpu = Batcher(Path(data_dir), "train", cfg, "cuda")
    x_gpu, _ = b_gpu.batch()
    res["path_when_empty"] = "gpu_resident" if b_gpu.gpu_resident else "memmap"

    # Hold VRAM so the NEXT Batcher sees less than the threshold -- the exact
    # thing a late-starting worker at high tenancy sees.
    hold, held = [], 0
    free, _ = torch.cuda.mem_get_info()
    target = free - int(need_free * 0.8)
    while held < target:
        chunk = min(2 * 1024 ** 3, target - held)
        hold.append(torch.empty(chunk, dtype=torch.uint8, device="cuda"))
        held += chunk
    free2, _ = torch.cuda.mem_get_info()
    res["free_after_hold_gib"] = free2 / 2**30
    res["should_gpu_resident_after_hold"] = should_gpu_resident(n_tok, "cuda")

    b_mm = Batcher(Path(data_dir), "train", cfg, "cuda")
    x_mm, _ = b_mm.batch()
    res["path_when_squeezed"] = "gpu_resident" if b_mm.gpu_resident else "memmap"
    del hold
    torch.cuda.empty_cache()

    same = bool(torch.equal(x_gpu.cpu(), x_mm.cpu()))
    res["same_first_batch"] = same
    res["first_row_gpu"] = x_gpu[0, :8].tolist()
    res["first_row_memmap"] = x_mm[0, :8].tolist()
    res["VERDICT"] = ("SAFE: both paths agree" if same else
                      "HAZARD: same seed, different tokens -- the sampler that "
                      "runs depends on free VRAM at construction")
    return res


# ---------------------------------------------------------------- gdnchunk
def probe_gdnchunk() -> dict:
    from nanolab.bench_gpu import bench
    out = {"speed": {}, "agreement": {}}
    for c in (16, 32, 64, 128, 256):
        cfg = _cfg("gdn", mixer_chunk=c)
        try:
            r = bench(cfg, 752.8e12, iters=10)
            out["speed"][c] = {k: r[k] for k in
                               ("tok_s", "ms_per_step", "peak_mem_gb", "fwd_ms", "bwd_ms")}
            print(f"  chunk {c:>4}: {r['tok_s']/1e3:7.1f}K tok/s  "
                  f"{r['ms_per_step']:7.1f} ms  {r['peak_mem_gb']:5.2f}GB")
        except torch.cuda.OutOfMemoryError:
            out["speed"][c] = {"err": "OOM"}
            print(f"  chunk {c:>4}: OOM")
        torch.cuda.empty_cache()

    # Does chunk width change the MATH or only the fp association order? The
    # O(T) `_sequential` reference is chunk-independent, so it is the invariant
    # to measure every chunk against (same construction as the regression test
    # `gdn_chunked_matches_sequential`, at the board's real width and context).
    from nanolab.model import build_model
    for c in (16, 32, 64, 128, 256):
        cfg = _cfg("gdn", mixer_chunk=c)
        model = build_model(cfg).to("cuda")
        gdn = model.blocks[0].mixer
        gdn.chunk = c                      # built from cfg, pinned here to be explicit
        torch.manual_seed(3)
        x = torch.randn(2, cfg.block_size, cfg.d_model, device="cuda")
        with torch.no_grad():
            y_chunk, _ = gdn(x)
            y_seq, _ = gdn._sequential(x)
        out["agreement"][c] = {
            "chunk_attr": gdn.chunk,
            "max_abs_vs_sequential": float((y_chunk - y_seq).abs().max()),
            "rel_rms_vs_sequential": float((y_chunk - y_seq).pow(2).mean().sqrt()
                                           / y_seq.pow(2).mean().sqrt()),
        }
        print(f"  chunk {c:>4}: max|y-seq| {out['agreement'][c]['max_abs_vs_sequential']:.3e}  "
              f"relRMS {out['agreement'][c]['rel_rms_vs_sequential']:.3e}")
        del model, gdn
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- evalsync
def probe_evalsync() -> dict:
    from nanolab.data import Batcher, get_dataset
    from nanolab.model import build_model
    cfg = _cfg("attention")
    data_dir, _, _ = get_dataset(cfg)
    val = Batcher(Path(data_dir), "val", cfg, "cuda")
    model = build_model(cfg).to("cuda").eval()
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    N = cfg.eval_iters

    def as_written():
        losses = torch.zeros(N)
        with torch.no_grad():
            for i in range(N):
                x, y = val.batch()
                with ac:
                    _, loss = model(x, y)
                losses[i] = loss.item()          # one host sync per iteration
        return losses.mean().item()

    def deferred():
        buf = []
        with torch.no_grad():
            for _ in range(N):
                x, y = val.batch()
                with ac:
                    _, loss = model(x, y)
                buf.append(loss.detach())        # stays on device
        losses = torch.stack(buf).float().cpu()  # ONE copy
        return losses.mean().item()

    res = {}
    for name, fn in (("as_written", as_written), ("deferred", deferred)):
        fn()                                     # warm
        torch.cuda.synchronize()
        t = time.time()
        vals = [fn() for _ in range(3)]
        torch.cuda.synchronize()
        res[name] = {"ms": (time.time() - t) / 3 * 1000, "val": vals[0]}
        print(f"  {name:<11} {res[name]['ms']:7.1f} ms/eval   val={vals[0]:.10f}")
    res["identical"] = res["as_written"]["val"] == res["deferred"]["val"]
    res["speedup"] = res["as_written"]["ms"] / res["deferred"]["ms"]
    # 61 evals per 50M run; a 50M attention job is ~6.4 min of train compute
    res["saved_s_per_50M_run"] = 61 * (res["as_written"]["ms"] - res["deferred"]["ms"]) / 1000
    print(f"  bit-identical={res['identical']}  speedup={res['speedup']:.2f}x  "
          f"saves {res['saved_s_per_50M_run']:.0f}s per 50M run")
    return res


# ---------------------------------------------------------------- moereal
def probe_moereal() -> dict:
    """Re-time MoE arms on next-token targets instead of synth_batch's y==x."""
    from nanolab.data import Batcher, get_dataset
    from nanolab.model import build_model
    from nanolab.optim import build_optimizers
    out = {}
    for arm in ("attention", "moe_e1k1", "moe_e1k1_raw", "moe_e4k1", "moe_e4k1_raw",
                "moe_e8k1", "moe_e8k1_raw"):
        cfg = _cfg(arm)
        data_dir, _, _ = get_dataset(cfg)
        b = Batcher(Path(data_dir), "train", cfg, "cuda")
        model = build_model(cfg).to("cuda")
        opts = build_optimizers(model, cfg)
        ac = torch.autocast("cuda", dtype=torch.bfloat16)

        def step():
            x, y = b.batch()
            with ac:
                _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            for o in opts:
                o.step()
                o.zero_grad(set_to_none=True)

        for _ in range(5):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        for _ in range(15):
            step()
        torch.cuda.synchronize()
        ms = (time.time() - t) / 15 * 1000
        out[arm] = {"ms_per_step": ms,
                    "tok_s": cfg.batch_size * cfg.block_size / (ms / 1000),
                    "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
        print(f"  {arm:<16} {out[arm]['tok_s']/1e3:7.1f}K tok/s  {ms:7.1f} ms  "
              f"{out[arm]['peak_gb']:5.2f}GB   (real corpus batches)")
        del model, opts, b
        torch.cuda.empty_cache()
    return out


PROBES = {"datapath": probe_datapath, "gdnchunk": probe_gdnchunk,
          "evalsync": probe_evalsync, "moereal": probe_moereal}

if __name__ == "__main__":
    names = sys.argv[1:] or list(PROBES)
    OUT.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for nm in names:
        print(f"\n===== probe: {nm} =====", flush=True)
        try:
            all_res[nm] = PROBES[nm]()
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_res[nm] = {"ERROR": f"{type(e).__name__}: {e}"}
        torch.cuda.empty_cache()
    (OUT / "probes.json").write_text(json.dumps(all_res, indent=2, default=str))
    print(f"\nwrote {OUT/'probes.json'}")
