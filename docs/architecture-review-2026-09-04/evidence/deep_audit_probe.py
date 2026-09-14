"""Read-only September 13 audit; stdout is JSON, all fixtures live in temporary dirs.

Run from any directory with the repository's existing Python/PyTorch/SciPy stack.
Optional --binn-root independently recomputes the 48 Wave-29 cell measurements.
No campaign training, network calls, or existing artifact writes. The resume probe
uses six tiny CPU optimizer steps to characterize the actual trainer contract.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

import numpy as np
import scipy
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from nanolab.crossover_replicate import _arm_from_run_dir
from nanolab.config import build_config
from nanolab.data import Batcher
from nanolab.mqar_suite import arm_of
from nanolab import train as trainer


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def estimate(values):
    n = len(values)
    if n < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("at least two finite values required")
    mean = statistics.fmean(values)
    half = stats.t.ppf(0.975, n - 1) * statistics.stdev(values) / math.sqrt(n)
    return {"n": n, "mean": mean, "ci95": [mean-half, mean+half], "values": values}


def rows(path):
    result = []
    for no, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{no}: non-object")
            result.append(value)
    return result


def read_arm(suite, arm, expected_seeds):
    result = {}
    for config in sorted((ROOT / "nanolab/out" / suite).glob("*/config.json")):
        if _arm_from_run_dir(config.parent) != arm:
            continue
        cfg = json.loads(config.read_text())
        seed = cfg["seed"]
        if seed not in expected_seeds:
            continue
        if seed in result:
            raise ValueError(f"duplicate arm/seed: {suite}/{arm}/{seed}")
        events = rows(config.with_name("metrics.jsonl"))
        done = [r for r in events if r.get("event") == "done"]
        if len(done) != 1 or not math.isfinite(done[0].get("final_val", float("nan"))):
            raise ValueError(f"not one finite terminal result: {config}")
        curve = [(r["tokens"], r["val_loss"]) for r in events if r.get("event") == "eval"]
        if not curve or any(b[0] <= a[0] for a, b in zip(curve, curve[1:])):
            raise ValueError(f"invalid curve: {config}")
        if not all(math.isfinite(v) for _, v in curve):
            raise ValueError(f"nonfinite curve: {config}")
        starts = [r for r in events if r.get("event") == "start"]
        result[seed] = {"config": cfg, "curve": dict(curve), "final": done[0]["final_val"],
                        "params": starts[0].get("params"), "starts": starts,
                        "path": str(config.parent.relative_to(ROOT))}
    if set(result) != set(expected_seeds):
        raise ValueError(f"missing seeds: {suite}/{arm}: {sorted(set(expected_seeds)-set(result))}")
    return result


def compare(left, right, seeds):
    a, b = read_arm(*left, seeds), read_arm(*right, seeds)
    differences = collections.defaultdict(list)
    for s in seeds:
        for k in sorted(set(a[s]["config"]) | set(b[s]["config"])):
            x, y = a[s]["config"].get(k), b[s]["config"].get(k)
            if x != y:
                differences[k].append({"seed": s, "arm": x, "ref": y})
    marks = sorted(set.intersection(*(set(v["curve"]) for group in (a,b) for v in group.values())))
    deltas = [a[s]["final"] - b[s]["final"] for s in seeds]
    mean_curve = [(t,statistics.fmean(a[s]["curve"][t]-b[s]["curve"][t] for s in seeds)) for t in marks]
    crossings = {}
    directions = {}
    for s in seeds:
        flips = []
        for lo, hi in zip(marks, marks[1:]):
            d0 = a[s]["curve"][lo]-b[s]["curve"][lo]
            d1 = a[s]["curve"][hi]-b[s]["curve"][hi]
            if d0 < 0 <= d1:
                flips.append(lo+(hi-lo)*(-d0)/(d1-d0))
        crossings[s] = flips
        signs = [int(np.sign(a[s]["curve"][t]-b[s]["curve"][t])) for t in marks]
        signs = [sign for sign in signs if sign]
        directions[s] = [(lo, hi) for lo, hi in zip(signs, signs[1:]) if lo != hi]
    # Estimate the late recovery after the arm's initial learning advantage.
    # An earlier positive-to-negative warmup transient is allowed and recorded;
    # a reverse transition after recovery is not. The independent terminal draw
    # must also favor the reference before calling the trajectory "overtaken".
    valid_crossing = all(len(crossings[s]) == 1 and directions[s][-1:] == [(-1, 1)]
                         and a[s]["final"] > b[s]["final"] for s in seeds)
    return {"arm": left, "ref": right, "seeds": seeds,
            "final_difference": estimate(deltas),
            "arm_mean": statistics.fmean(a[s]["final"] for s in seeds),
            "ref_mean": statistics.fmean(b[s]["final"] for s in seeds),
            "negative_signs": sum(x < 0 for x in deltas),
            "params": {"arm": sorted({a[s]["params"] for s in seeds}),
                       "ref": sorted({b[s]["params"] for s in seeds})},
            "shared_markers": len(marks),
            "mean_curve_negative_markers":sum(d<0 for _,d in mean_curve),
            "worst_mean_curve_difference_after_first":max(mean_curve[1:],key=lambda x:x[1]) if len(mean_curve)>1 else None,
            "all_seed_negative_markers": sum(all(a[s]["curve"][t]<b[s]["curve"][t] for s in seeds) for t in marks),
            "crossings": crossings,
            "sign_transitions": directions,
            "crossing_estimate": estimate([v[0] for v in crossings.values()]) if valid_crossing else None,
            "config_differences": dict(differences),
            "paths": {"arm": [a[s]["path"] for s in seeds], "ref": [b[s]["path"] for s in seeds]}}


def recall_results():
    output = {}
    for suite, pairs, batch, planned in [("mqar_e16",64,64,15),("mqar_e16_seq511",128,128,10),
                                        ("mqar_e28_p8",8,256,15)]:
        groups = collections.defaultdict(dict)
        for r in rows(ROOT / "nanolab/out" / suite / "runs.jsonl"):
            if r.get("n_pairs") != pairs or r.get("batch_size") != batch or r.get("steps") != 3000:
                continue
            arm, seed = arm_of(r), r["seed"]
            if seed in groups[arm]:
                raise ValueError(f"duplicate recall result {suite}/{arm}/{seed}")
            if not math.isfinite(r["recall"]) or r["solved"] != (r["recall"] >= .8):
                raise ValueError(f"invalid recall {r}")
            groups[arm][seed] = r
        ref = "attention" if "attention" in groups else "gdn"
        out = {}
        for arm, g in groups.items():
            seeds = sorted(set(g) & set(groups[ref]))
            wins = sum(g[s]["solved"] and not groups[ref][s]["solved"] for s in seeds)
            losses = sum(groups[ref][s]["solved"] and not g[s]["solved"] for s in seeds)
            p = stats.binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.
            out[arm] = {"n":len(g),"planned":planned,"missing_seeds":sorted(set(range(1,planned+1))-set(g)),
                        "solved":sum(r["solved"] for r in g.values()), "recall_by_seed":{s:r["recall"] for s,r in g.items()},
                        "paired_ref":ref,"paired_n":len(seeds),"wins":wins,"losses":losses,"exact_p":p}
        output[suite] = out
    return output


def reader_contracts():
    pb = module("audit_paired_board",ROOT/"scripts/paired_board.py")
    ct = module("audit_crossing",ROOT/"scripts/crossing_token.py")
    la = module("audit_argmin",ROOT/"scripts/lr_argmin.py")
    result = {"paired_interval_n3": {"actual":pb.interval([0.,1.,2.])[1]["95%"],
                                     "correct":estimate([0.,1.,2.])["ci95"]},
              "unchecked_recipe_fields":sorted(set(["compile","fused_ce","grad_accum","dataset","dtype","eval_train","weight_decay"])-set(pb.RECIPE_KEYS)),
              "nearest_marker_mismatch":{"arm_value":pb.at([(100,1.)],200),"ref_value":pb.at([(200,2.)],200)},
              "recross_fixture":ct.seed_crossing([(1,-1.),(2,1.),(3,-1.)],[(1,0.),(2,0.),(3,0.)],[1,2,3])}
    old_argv = sys.argv
    try:
        def fixture_arm(suite, arm):
            is_left = arm == "left"
            return {s:{"curve": [(1,1.),(2,3.),(3,1.)] if is_left else [(1,2.),(2,2.),(3,2.)],
                       "final":1. if is_left else 2., "recipe":{"batch_size":32 if is_left else 8}}
                    for s in (1,2,3)}
        ct.paired_board.load_arm=fixture_arm
        sys.argv=["crossing_token.py","left@fixture","--ref","right@fixture"]
        text=io.StringIO()
        with contextlib.redirect_stdout(text):ct.main()
        result["crossing_cli_accepts_recipe_mismatch_and_reverse_recross"]=text.getvalue()
    finally:
        sys.argv=old_argv
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp); suite = base/"nanolab/out/fixture"; suite.mkdir(parents=True)
        def write(name, seed, final, compile_on=False, done=True):
            d=suite/name;d.mkdir()
            cfg={"seed":seed,"d_model":8,"mixer":"attention","layer_mixers":"","lr":.0006,
                 "compile":compile_on,"max_steps":2}
            (d/"config.json").write_text(json.dumps(cfg))
            events=[{"event":"eval","tokens":100,"val_loss":2.}, {"event":"eval","tokens":200,"val_loss":1.}]
            if done:events.append({"event":"done","final_val":final})
            (d/"metrics.jsonl").write_text("\n".join(json.dumps(x) for x in events))
        write("a",42,1.)
        write("z",42,9.,True)
        write("partial",100,None,done=False)
        pb.ROOT=base;pb._arm_from_run_dir=lambda d:"attention"
        la.ROOT=base
        got=pb.load_arm("fixture","attention")
        result["duplicate_seed_selected_final"]=got[42]["final"]
        result["incomplete_seed_is_loaded"]=100 in got and got[100]["final"] is None
        result["lr_reader_merges_compile_variants"]=la.cells("fixture")[(8,"attention")][1.0][42]
    assert result["duplicate_seed_selected_final"] == 9.
    assert result["incomplete_seed_is_loaded"]
    assert result["lr_reader_merges_compile_variants"] == 9.
    assert result["recross_fixture"] == (1.5,1)
    return result


def resume_contract():
    torch.set_num_threads(1)
    class Interrupted(Exception):
        pass
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp);data=base/"data";data.mkdir()
        rng=np.random.default_rng(913)
        for name in ("train","val"):
            rng.integers(0,32,size=4096,dtype=np.uint16).tofile(data/f"{name}.bin")
        def config(name):
            return build_config(None,dict(out_dir=tmp,run_name=name,device="cpu",dtype="fp32",compile=False,
                vocab_size=32,n_layer=1,d_model=16,n_head=2,head_dim=8,block_size=8,batch_size=2,
                grad_accum=1,max_steps=6,lr_max_steps=6,warmup_steps=0,optimizer="adamw",lr=.003,
                eval_train=False,eval_interval=99,eval_iters=1,ckpt_interval=2,log_interval=1))
        def batchers(cfg):
            return Batcher(data,"train",cfg,"cpu"),Batcher(data,"val",cfg,"cpu")
        save=trainer._save
        previous=os.environ.pop("RESUME",None)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                full=config("full");trainer.train(full,batchers=batchers(full))
                cut=config("cut");saved_states=[]
                def interrupt(path, *args, **kwargs):
                    save(path,*args,**kwargs)
                    if path.name=="ckpt.pt":
                        saved_states.extend(x.gen.get_state().clone() for x in live)
                        raise Interrupted()
                trainer._save=interrupt;live=batchers(cut)
                try:trainer.train(cut,batchers=live)
                except Interrupted:pass
                else:raise AssertionError("fixture never interrupted")
                trainer._save=save
                checkpoint=torch.load(base/"cut/ckpt.pt",weights_only=False)
                os.environ["RESUME"]="1"
                # First characterize the real fresh-process sampler behavior.
                trainer.train(cut,batchers=batchers(cut))
                failed=torch.load(base/"cut/final.pt",weights_only=False)["model"]
                # Restore the exact same pre-resume checkpoint and sampler states
                # as a positive diagnostic control, without changing trainer code.
                torch.save(checkpoint,base/"cut/ckpt.pt")
                corrected=batchers(cut)
                for b,state in zip(corrected,saved_states):b.gen.set_state(state)
                trainer.train(cut,batchers=corrected)
                restored=torch.load(base/"cut/final.pt",weights_only=False)["model"]
            expected=torch.load(base/"full/final.pt",weights_only=False)["model"]
            delta=lambda model:max(float((model[k]-expected[k]).abs().max()) for k in expected)
            result={"checkpoint_keys":sorted(checkpoint),"next_step":checkpoint["next_step"],
                    "fresh_sampler_max_parameter_error":delta(failed),
                    "externally_restored_sampler_max_parameter_error":delta(restored),
                    "scope":"six steps, one-layer CPU AdamW model, dropout zero; not a historical training replay"}
            assert result["fresh_sampler_max_parameter_error"]>1e-6
            assert result["externally_restored_sampler_max_parameter_error"]<1e-7
            # The actual evaluator advances the training batcher's private RNG.
            cfg=config("eval");train_b, _=batchers(cfg)
            before=train_b.gen.get_state().clone()
            model=trainer.build_model(cfg)
            trainer.evaluate(model,train_b,cfg,contextlib.nullcontext())
            result["train_evaluation_advances_training_rng"]=not torch.equal(before,train_b.gen.get_state())
            assert result["train_evaluation_advances_training_rng"]
            return result
        finally:
            trainer._save=save
            if previous is None:os.environ.pop("RESUME",None)
            else:os.environ["RESUME"]=previous


def geometry():
    # Protecting a spanning set freezes every linear prediction, even when the
    # protected outputs are wrong. Thinning preserves count information.
    rng=np.random.default_rng(913)
    q,_=np.linalg.qr(rng.normal(size=(12,12)))
    k=rng.normal(size=12);k/=np.linalg.norm(k)
    residuals=[float(np.linalg.norm(k-q[:,:r]@(q[:,:r].T@k))) for r in range(13)]
    assert residuals[-1]<1e-12
    # Exact count-only classifier after 90% independent event deletion:
    # class 0 originally has 100 spikes; class 1 has 1000. Threshold 40.
    accuracy=.5*(stats.binom.cdf(40,100,.1)+stats.binom.sf(40,1000,.1))
    assert accuracy>.999
    return {"projected_key_norm_by_protected_rank":residuals,
            "count_only_accuracy_after_90pct_thinning":accuracy,
            "count_fixture":"equal-prior Binomial(100,.1) vs Binomial(1000,.1), threshold 40; not SHD"}


def binn(root):
    p=root/"results/shd_attention_wave29_local"
    data={}
    fingerprints={}
    for path in sorted(p.glob("w29asy__*.json")):
        r=json.loads(path.read_text());seed=int(re.search(r"__s(\d+)\.json$",path.name).group(1))
        key=(r["arm"],r.get("temporal_condition") or "intact",seed)
        if key in data:raise ValueError(f"BINN collision {key}")
        data[key]=r["accuracy"];fingerprints[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(data)==48
    groups={}
    for arm in ("ff+fixed","ff+fixed+attn"):
        for condition in ("intact","spike-dropout-p90"):
            groups[f"{arm}/{condition}"]=estimate([data[(arm,condition,s)] for s in range(5290001,5290013)])
    did=[(data[("ff+fixed+attn","intact",s)]-data[("ff+fixed+attn","spike-dropout-p90",s)])-
         (data[("ff+fixed","intact",s)]-data[("ff+fixed","spike-dropout-p90",s)]) for s in range(5290001,5290013)]
    return {"head":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),
            "groups":groups,"did":estimate(did),"below_minus_003":sum(x<-.03 for x in did),"hashes":fingerprints}


def inventory():
    output={"jsonl_files":0,"metrics":0,"done_records":0,"finite_final_runs":0,"multiple_done_files":[],
            "multiple_start_files":[],"malformed":[],"nonfinite":[],"mqar_rows":0,
            "mqar_unique_names":0,"mqar_conflicting_names":[],"hashes":{}}
    recall=collections.defaultdict(list)
    for p in sorted((ROOT/"nanolab/out").rglob("*.jsonl")):
        output["jsonl_files"]+=1
        rel=str(p.relative_to(ROOT));output["hashes"][rel]=hashlib.sha256(p.read_bytes()).hexdigest()
        config=p.with_name("config.json")
        if p.name=="metrics.jsonl" and config.exists():
            output["hashes"][str(config.relative_to(ROOT))]=hashlib.sha256(config.read_bytes()).hexdigest()
        try:events=rows(p)
        except (ValueError,UnicodeError) as exc:
            output["malformed"].append({"path":rel,"error":str(exc)});continue
        for i,r in enumerate(events):
            for k,v in r.items():
                if isinstance(v,float) and not math.isfinite(v):output["nonfinite"].append({"path":rel,"record":i,"field":k})
        if p.name=="metrics.jsonl":
            output["metrics"]+=1
            done=[r for r in events if r.get("event")=="done"]
            starts=sum(r.get("event")=="start" for r in events)
            output["done_records"]+=len(done)
            if len(done)>1:output["multiple_done_files"].append(rel)
            if starts>1:output["multiple_start_files"].append(rel)
            if len(done)==1 and math.isfinite(done[0].get("final_val",float("nan"))):output["finite_final_runs"]+=1
        if p.name=="runs.jsonl" and "mqar" in rel:
            output["mqar_rows"]+=len(events)
            for r in events:recall[r["run"]].append((rel,r))
    output["mqar_unique_names"]=len(recall)
    for name,copies in recall.items():
        if len({r["recall"] for _,r in copies})>1:
            output["mqar_conflicting_names"].append({"run":name,"copies":[{"path":p,"recall":r["recall"]} for p,r in copies]})
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binn-root",type=Path)
    args=parser.parse_args()
    five=[42,100,777,1337,2026];three=[42,100,1337]
    pairs=[]
    for suffix in ("","_x1"):
        suite="crossover50m_parity32" if suffix else "crossover50m_ratioplace32"
        for arm in ("hybrid_mingru8_attn4","hybrid_mingru_periodic"):
            pairs.append(compare((suite,arm+suffix),(suite,"attention"),five))
    for arm in ("moe_e1k1_raw","moe_e4k1_raw","moe_e8k1_raw"):
        pairs.append(compare(("crossover50m_moe32d",arm),("crossover50m_moe32d","attention"),five))
    for width,mult,suite in [(384,"20","crossover_ladder50m"),(768,"10","crossover_ladder50m"),
                             (1152,"10","crossover_ladder50m"),(1536,"05","crossover_ladder1536m_argmin")]:
        pairs.append(compare((suite,f"w{width}_mingru_lr{mult}"),(suite,f"w{width}_attention_lr{mult}"),five))
    pairs.append(compare(("crossover_ladder50m","w1152_mingru_lr0667"),("crossover_ladder50m","w1152_attention_lr0667"),three))
    # The only long-horizon board: inspect ALL configuration differences.
    for arm in ("w384_attention_lr80","w384_mingru_lr40"):
        pairs.append(compare(("crossover200m_w384",arm),("crossover_ladder50m",arm),five))
    for family in ("attention","mingru"):
        pairs.append(compare(("crossover_ladder50m",f"w1152_{family}_lr0667"),
                             ("crossover_ladder50m",f"w1152_{family}_lr10"),three))
    for mult in ("40","80"):
        for family in ("hybrid_mingru8_attn4","hybrid_mingru_periodic"):
            pairs.append(compare(("crossover50m_ratioplace32",f"w768_{family}_lr{mult}"),
                                 ("crossover50m_ratioplace32",f"w768_attention_lr{mult}"),five))
    output={"head":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
            "versions":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"scipy":scipy.__version__},
            "comparisons":pairs,"recall":recall_results(),"reader_contracts":reader_contracts(),
            "resume":resume_contract(),"geometry":geometry(),"inventory":inventory()}
    source_paths=["scripts/paired_board.py","scripts/lr_argmin.py","scripts/crossing_token.py",
                  "scripts/g12_sampler_check.sh","scripts/e35_w384_ladder.sh","scripts/g4_hybrid_seq511.sh",
                  "nanolab/train.py","nanolab/data.py","nanolab/mqar.py","nanolab/mqar_suite.py",
                  "nanolab/model.py","nanolab/mixers.py","nanolab/optim.py","nanolab/config.py",
                  "nanolab/crossover_replicate.py","nanolab/tests.py",str(Path(__file__).relative_to(ROOT))]
    output["source_sha256"]={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}
    if args.binn_root:output["binn_wave29"]=binn(args.binn_root)
    print(json.dumps(output,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
