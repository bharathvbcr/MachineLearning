$PYTHON = "C:\conda-data\envs\cuda_torch_env\python.exe"
$COOLDOWN_MINS = 15

$COMMON = @(
    "-u", "-m", "nanolab.train",
    "--preset", "phase1",
    "--batch_size", "32", "--grad_accum", "1",
    "--fused_ce", "true", "--fused_ce_chunks", "16", "--grad_checkpoint", "true",
    "--warmup_steps", "100", "--eval_interval", "500", "--eval_iters", "50", "--ckpt_interval", "500",
    "--seed", "1337", "--optimizer", "muon",
    "--matrix_lr", "0.005", "--lr", "1.2e-4",
    "--schedule", "wsd",
    "--lr_max_steps", "20000",
    "--run_name", "run128m_20k"
)

function Cooldown {
    param($mins)
    Write-Host ""
    Write-Host "====== COOLING DOWN for $mins minutes ======"
    for ($i = $mins * 60; $i -gt 0; $i -= 30) {
        $remaining = [math]::Ceiling($i / 60)
        Write-Host "  $remaining min remaining..."
        Start-Sleep -Seconds ([math]::Min(30, $i))
    }
    Write-Host "====== COOLDOWN DONE -- resuming training ======"
    Write-Host ""
}

# All targets: cooldown after every 2500 steps
$targets = @(2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000)
$seg = 1

foreach ($target in $targets) {
    Write-Host "====== SEGMENT ${seg}: to step $target ======"

    if ($seg -eq 1) {
        # First segment: load weights from pretrained checkpoint
        & $PYTHON @COMMON --max_steps $target --init_ckpt "nanolab/out/run128m_10k/best.pt"
    } else {
        $env:RESUME = "1"
        & $PYTHON @COMMON --max_steps $target
        $env:RESUME = "0"
    }

    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: segment $seg failed (exit $LASTEXITCODE)"; exit 1 }

    if ($target -lt 20000) {
        Cooldown $COOLDOWN_MINS
    }
    $seg++
}

Write-Host ""
Write-Host "====== ALL DONE -- run128m_20k complete ======"
