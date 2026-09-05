"""Assemble the GH200 tuning-sprint tables from the measured JSON.

Generated rather than transcribed so the doc cannot drift from the run records
in nanolab/out/_tune/.

    python scripts/tune_report.py > docs/GPU_TUNING_2026-09-05.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
TUNE = REPO / "nanolab/out/_tune"


def jload(name):
    p = TUNE / name
    return json.loads(p.read_text()) if p.exists() else None


def arm_table() -> str:
    d = (jload("arm_cost_t1.json") or {}).get("arm", {})
    w = (jload("arm_cost_w1536.json") or {}).get("arm", {})
    d = {**d, **w}
    rows = sorted(((k, v) for k, v in d.items() if v.get("ok")),
                  key=lambda kv: -kv[1]["tok_s"])
    out = ["| arm | tok/s | ms/step | peak alloc | peak resv | params | 50M job |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in rows:
        out.append(f"| `{k}` | {v['tok_s']/1e3:.1f}K | {v['ms_per_step']:.0f} | "
                   f"{v['peak_mem_gb']:.1f} GB | {v['reserved_gb']:.1f} GB | "
                   f"{v['params_total']/1e6:.1f}M | {50e6/v['tok_s']/60:.1f} min |")
    bad = [k for k, v in d.items() if not v.get("ok")]
    if bad:
        out.append(f"\nFailed rows: {bad}")
    return "\n".join(out)


def tenancy_table() -> str:
    out = ["| arm | MPS | t=1 | t=2 | t=3 | t=4 | t=6 | best |",
           "|---|---|---:|---:|---:|---:|---:|---|"]
    for tag, label in (("nomps", "off"), ("mps", "on")):
        for arm in ("attention", "hybrid_mingru8_attn4", "gdn"):
            d = jload(f"tenancy_{tag}_{arm}.json")
            if not d:
                continue
            cells = []
            for t in ("1", "2", "3", "4", "6"):
                r = d["rows"].get(t)
                if r is None:
                    cells.append("—")
                elif not r.get("ok"):
                    cells.append("**OOM**")
                else:
                    cells.append(f"{r['agg_tok_s']/1e3:.1f}K")
            out.append(f"| `{arm}` | {label} | " + " | ".join(cells) +
                       f" | **{d.get('best_tenancy','?')}** |")
    rev = jload("tenancy_reversal_attention.json")
    if rev:
        cells = []
        for t in ("1", "2", "3", "4", "6"):
            r = rev["rows"].get(t)
            cells.append("**OOM**" if r and not r.get("ok") else
                         (f"{r['agg_tok_s']/1e3:.1f}K" if r else "—"))
        out.append(f"| `attention` (descending order) | off | " + " | ".join(cells) +
                   f" | **{rev.get('best_tenancy','?')}** |")
    return "\n".join(out)


def compile_table() -> str:
    d = jload("compile.json")
    if not d:
        return "_(not measured)_"
    out = ["| arm | eager | compiled | speed-up | compile time | max abs logit diff |",
           "|---|---:|---:|---:|---:|---:|"]
    for k, v in d.items():
        if "ERROR" in v:
            out.append(f"| `{k}` | — | — | — | — | error: {v['ERROR'][:70]} |")
            continue
        c = v.get("compiled", {})
        if "err" in c:
            out.append(f"| `{k}` | {v['eager']['tok_s']/1e3:.1f}K | **failed** | — | "
                       f"{v.get('compile_s',0):.0f} s | {c['err'][:70]} |")
        else:
            out.append(f"| `{k}` | {v['eager']['tok_s']/1e3:.1f}K | "
                       f"{c['tok_s']/1e3:.1f}K | **{v['speedup']:.2f}x** | "
                       f"{v['compile_s']:.0f} s | {v.get('max_abs_logit_diff')} |")
    return "\n".join(out)


def probes_text() -> str:
    d = jload("probes.json")
    if not d:
        return "_(not measured)_"
    out = []
    dp = d.get("datapath", {})
    if dp and "ERROR" not in dp:
        out.append("**Sampler path (data.should_gpu_resident)**\n")
        out.append(f"- corpus {dp['n_tokens']:,} tokens = {dp['corpus_gib']:.2f} GiB as int32; "
                   f"the GPU-resident path needs **> {dp['free_needed_gib']:.2f} GiB free** "
                   f"at Batcher construction")
        out.append(f"- with an empty GPU: `{dp['path_when_empty']}`; "
                   f"squeezed to {dp['free_after_hold_gib']:.2f} GiB free: `{dp['path_when_squeezed']}`")
        out.append(f"- same seed, same split, first batch identical: **{dp['same_first_batch']}**")
        out.append(f"- {dp['VERDICT']}\n")
    gc = d.get("gdnchunk", {})
    if gc and "ERROR" not in gc:
        out.append("**GDN chunk width (`mixer_chunk`, default 32)**\n")
        out.append("| chunk | tok/s | ms/step | peak | max abs vs O(T) reference |")
        out.append("|---|---:|---:|---:|---:|")
        for c in sorted(gc.get("speed", {}), key=int):
            s = gc["speed"][c]
            a = gc.get("agreement", {}).get(c, {})
            if "err" in s:
                out.append(f"| {c} | {s['err']} | | | |")
            else:
                out.append(f"| {c} | {s['tok_s']/1e3:.1f}K | {s['ms_per_step']:.0f} | "
                           f"{s['peak_mem_gb']:.1f} GB | "
                           f"{a.get('max_abs_vs_sequential', float('nan')):.2e} |")
        out.append("")
    ev = d.get("evalsync", {})
    if ev and "ERROR" not in ev:
        out.append("**Eval loop host sync**\n")
        out.append(f"- as written: {ev['as_written']['ms']:.0f} ms/eval; "
                   f"one deferred transfer: {ev['deferred']['ms']:.0f} ms/eval "
                   f"(**{ev['speedup']:.2f}x**)")
        out.append(f"- means bit-identical: **{ev['identical']}**; "
                   f"saves {ev['saved_s_per_50M_run']:.0f} s per 50M run "
                   f"(61 evals)\n")
    mr = d.get("moereal", {})
    if mr and "ERROR" not in mr:
        out.append("**MoE arms re-timed on real corpus batches**\n")
        out.append("| arm | tok/s (real) | ms/step | peak |")
        out.append("|---|---:|---:|---:|")
        for k, v in mr.items():
            out.append(f"| `{k}` | {v['tok_s']/1e3:.1f}K | {v['ms_per_step']:.0f} | "
                       f"{v['peak_gb']:.1f} GB |")
        out.append("")
    return "\n".join(out)



def tenancy_recommendation() -> str:
    """Per-arm tenancy, derived from the measured rows rather than chosen."""
    rows = []
    for f in sorted(TUNE.glob("tenancy_nomps_*.json")):
        arm = f.name[len("tenancy_nomps_"):-len(".json")]
        d = json.loads(f.read_text())
        ok = {int(k): v for k, v in d["rows"].items() if v.get("ok")}
        if not ok:
            continue
        t1 = ok.get(1, {}).get("agg_tok_s")
        best_t = max(ok, key=lambda t: ok[t]["agg_tok_s"])
        best = ok[best_t]["agg_tok_s"]
        m = jload(f"tenancy_mps_{arm}.json")
        mps_best = mps_t = None
        if m:
            mok = {int(k): v for k, v in m["rows"].items() if v.get("ok")}
            if mok:
                mps_t = max(mok, key=lambda t: mok[t]["agg_tok_s"])
                mps_best = mok[mps_t]["agg_tok_s"]
        cost = (jload("arm_cost_t1.json") or {}).get("arm", {}).get(arm, {})
        rows.append((arm, cost.get("mfu"), t1, best_t, best, mps_t, mps_best))
    if not rows:
        return "_(not measured)_"
    out = ["| arm | MFU @ t=1 | t=1 | best without MPS | best with MPS | recommend |",
           "|---|---:|---:|---|---|---|"]
    for arm, mfu, t1, bt, b, mt, mb in sorted(rows, key=lambda r: -(r[1] or 0)):
        cand = [("1", t1)]
        if b:
            cand.append((f"{bt} (no MPS)", b))
        if mb:
            cand.append((f"{mt} (MPS)", mb))
        pick = max(cand, key=lambda c: c[1])
        out.append(
            f"| `{arm}` | {mfu*100:.1f}% | {t1/1e3:.1f}K | "
            f"{bt} @ {b/1e3:.1f}K ({b/t1:.2f}x) | "
            + (f"{mt} @ {mb/1e3:.1f}K ({mb/t1:.2f}x)" if mb else "—")
            + f" | **{pick[0]}** ({pick[1]/t1:.2f}x) |")
    return "\n".join(out)


def fusedce_table() -> str:
    d = jload("fusedce.json")
    if not d:
        return "_(not measured)_"
    out = ["| arm | setting | tok/s | ms/step | peak resv | loss on fixed input |",
           "|---|---|---:|---:|---:|---:|"]
    for arm, settings in d.items():
        for k, v in settings.items():
            if "err" in v:
                out.append(f"| `{arm}` | {k} | {v['err']} | | | |")
            else:
                out.append(f"| `{arm}` | {k} | {v['tok_s']/1e3:.1f}K | "
                           f"{v['ms_per_step']:.0f} | {v['reserved_gb']:.1f} GB | "
                           f"{v['loss']:.6f} |")
    return "\n".join(out)


def evaliters_text() -> str:
    d = jload("evaliters.json")
    if not d:
        return "_(not measured)_"
    out = [f"Two trained arms at seed {d['seed']} (`{d['a']}` vs `{d['b']}`), "
           f"scored on the same {d['batches']} val batches.\n",
           f"- per-batch SD: {d['sd_a']:.4f} and {d['sd_b']:.4f} nats; "
           f"correlation between the arms **{d['corr']:.4f}**",
           f"- paired difference {d['mean_diff']:+.4f}, SD {d['sd_diff']:.4f} "
           f"(the pairing removes most of the batch noise)\n",
           "| eval_iters | SE of one arm's mean | SE of the paired difference | eval cost |",
           "|---|---:|---:|---:|"]
    for n, v in sorted(d["by_n"].items(), key=lambda kv: int(kv[0])):
        out.append(f"| {n} | {v['se_arm']:.4f} | {v['se_diff']:.4f} | "
                   f"{int(n)/20*100:.0f}% |")
    return "\n".join(out)

if __name__ == "__main__":
    print("## A. Per-arm cost at the board shape (batch 32, ctx 512, tenancy 1)\n")
    print(arm_table())
    print("\n## B. Aggregate throughput vs tenancy\n")
    print(tenancy_table())
    print("\n## C. torch.compile\n")
    print(compile_table())
    print("\n## D. Probes\n")
    print(probes_text())
    print("\n## E. Per-arm tenancy, derived from the rows above\n")
    print(tenancy_recommendation())
    print("\n## F. Fused cross-entropy chunking\n")
    print(fusedce_table())
    print("\n## G. What eval_iters buys\n")
    print(evaliters_text())
