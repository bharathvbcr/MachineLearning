# Evidence — GH200 tuning sprint, 2026-09-05

Raw measurement output for `docs/GPU_TUNING_2026-09-05.md`. Kept here rather than
under `nanolab/out/` because `.gitignore` publishes only the suite run records
(`metrics.jsonl`, `config.json`, `queue.json`, `recipe.json`, `ledger.json`) from
there, and these are neither.

| file | produced by |
|---|---|
| `arm_cost_t1.json` | `python -m nanolab.sweep_gpu arm --arms <29 arms> --batch_size 32 --block_size 512 --peak_flops 752.8e12` |
| `arm_cost_w1536.json` | the same, for the width-1536 ladder arms E27 still needs |
| `tenancy_nomps_*.json` | `scripts/tune_tenancy.py --arm <arm> --tenancies 1,2,3,4,6` |
| `tenancy_mps_*.json` | the same, with an MPS daemon serving (`scripts/tune_mps.sh`) |
| `tenancy_reversal_attention.json` | tenancies descending, to exclude ordering effects |
| `probes.json` | `scripts/tune_probes.py datapath gdnchunk evalsync moereal` |
| `compile.json` | `scripts/tune_compile.py` |
| `fusedce.json` | `scripts/tune_fusedce.py` |
| `evaliters.json` | `scripts/tune_evaliters.py` against two trained `crossover50m_loop32` checkpoints |
| `*.log` | the console output of each stage, including the failures |

Regenerate the tables from these with `python scripts/tune_report.py`.
