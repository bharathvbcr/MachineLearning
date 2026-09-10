import json
from pathlib import Path
from nanolab.crossover_replicate import mean_ci
for suite, pre, label in (("crossover50m_swa32","cx32swa_","E12  ctx512  bs32  50M"),
                          ("crossover50m_swa2k","cx2kswa_","E15  ctx2048 bs8   50M")):
    root = Path("nanolab/out")/suite
    by, tps = {}, {}
    for d in sorted(root.glob(pre + "*")):
        m = d/"metrics.jsonl"
        if not m.exists(): continue
        arm = d.name[len(pre):].rsplit("_s",1)[0]
        fin = tok = None
        for line in m.read_text().splitlines():
            try: r = json.loads(line)
            except Exception: continue
            if r.get("event") == "done":
                fin, tok = r.get("final_val"), r.get("mean_tok_s")
        if fin is not None:
            by.setdefault(arm, []).append(fin)
            tps.setdefault(arm, []).append(tok or 0)
    print(f"\n=== {label} ===")
    rows = sorted((mean_ci(v)[0], a, *mean_ci(v)[1:], len(v)) for a, v in by.items())
    for mu, arm, lo, hi, n in rows:
        rate = sum(tps[arm])/max(1,len(tps[arm]))
        print(f"  {arm:24} {mu:.4f}  [{lo:.4f}, {hi:.4f}]  n={n}  {rate/1000:5.1f}K tok/s")
