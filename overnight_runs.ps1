# overnight_runs.ps1
# Three sequential GPU training runs with 30-min cooldowns between each.
#
# Run 1: xs_attention  (30M params, 2000 steps, ~16M tokens)  crossover-vs-model-size experiment
# Run 2: xs_mingru     (30M params, 2000 steps, ~16M tokens)  same
# Run 3: run128m_10k   (124M params, 10000 steps, ~327M tokens, ~6.5 hours)  big overnight
#
# Usage (keep terminal open overnight):
#   powershell -ExecutionPolicy Bypass -File overnight_runs.ps1

$PYTHON = "C:\conda-data\envs\cuda_torch_env\python.exe"
$WDIR   = "C:\Users\bhara\Downloads\Code\parameter_golf"
Set-Location $WDIR

function Log($msg) {
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Write-Host "[$ts] $msg"
}
function GpuInfo {
    $info = (& nvidia-smi --query-gpu=temperature.gpu,memory.used,power.draw --format=csv,noheader 2>$null)
    Write-Host "  GPU: $($info.Trim())"
}
function Cooldown([int]$mins) {
    Log "Cooldown: waiting $mins min..."
    GpuInfo
    Start-Sleep -Seconds ($mins * 60)
    GpuInfo
    Log "Cooldown done."
}

# ---- Clean stale xs_ output dirs (Logger appends; stale entries would corrupt curves) ----
Remove-Item -Recurse -Force nanolab\out\xs_attention -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force nanolab\out\xs_mingru    -ErrorAction SilentlyContinue

# ========================================================
# Run 1/3: xs_attention  (30M, 2000 steps, ~16M tokens)
# Crossover-vs-model-size experiment — small (30M) vs 124M baseline from ext_*
# ========================================================
Log "=== Run 1/3: xs_attention (30M, 2000 steps, ~16M tok) ==="
GpuInfo
& $PYTHON -u -m nanolab.train `
    --preset phase1 `
    --mixer attention `
    --n_layer 6 --d_model 384 --n_head 6 --head_dim 64 `
    --batch_size 16 --block_size 512 --grad_accum 1 `
    --max_steps 2000 `
    --fused_ce true --fused_ce_chunks 16 --grad_checkpoint true `
    --warmup_steps 100 --eval_interval 200 --eval_iters 40 --ckpt_interval 250 `
    --seed 1337 --optimizer muon --run_name xs_attention
if ($LASTEXITCODE -ne 0) { Log "FAILED: xs_attention (exit $LASTEXITCODE)"; exit 1 }
Log "xs_attention done."

Cooldown 30

# ========================================================
# Run 2/3: xs_mingru  (30M, 2000 steps, ~16M tokens)
# ========================================================
Log "=== Run 2/3: xs_mingru (30M, 2000 steps, ~16M tok) ==="
GpuInfo
& $PYTHON -u -m nanolab.train `
    --preset phase1 `
    --mixer mingru `
    --n_layer 6 --d_model 384 --n_head 6 --head_dim 64 `
    --batch_size 16 --block_size 512 --grad_accum 1 `
    --max_steps 2000 `
    --fused_ce true --fused_ce_chunks 16 --grad_checkpoint true `
    --warmup_steps 100 --eval_interval 200 --eval_iters 40 --ckpt_interval 250 `
    --seed 1337 --optimizer muon --run_name xs_mingru
if ($LASTEXITCODE -ne 0) { Log "FAILED: xs_mingru (exit $LASTEXITCODE)"; exit 1 }
Log "xs_mingru done."

Cooldown 30

# ========================================================
# Run 3/3: 128M overnight  (124M params, 10000 steps, ~327M tokens, ~6.5h)
# Uses existing 50M-token fineweb-edu bin (data cycles ~6.5x; fine — model is underfitting)
# ========================================================
Log "=== Run 3/3: run128m_10k (124M, 10000 steps, ~327M tok, ~6.5h) ==="
GpuInfo
& $PYTHON -u -m nanolab.train `
    --preset phase1 `
    --batch_size 32 --grad_accum 1 `
    --max_steps 10000 `
    --fused_ce true --fused_ce_chunks 16 --grad_checkpoint true `
    --warmup_steps 500 --eval_interval 500 --eval_iters 50 --ckpt_interval 500 `
    --seed 1337 --optimizer muon --run_name run128m_10k
Log "run128m_10k done (exit $LASTEXITCODE)."

Log "=== ALL RUNS COMPLETE ==="
GpuInfo
