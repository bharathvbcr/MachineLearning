#!/usr/bin/env python3
import subprocess
import re
import os

def run_experiment():
    presets = ["sota", "16m", "arch02-128m"]
    mixers = ["attention", "mamba2", "mingru"]
    
    max_steps = 30
    
    print(f"{'Preset':<15} | {'Mixer':<12} | {'Params (M)':<12} | {'Tok/s':<10} | {'Loss@30':<10}", flush=True)
    print("-" * 70, flush=True)
    
    cwd = os.path.join(os.path.dirname(__file__), "..", "Rust_MLKit", "arch_02_value_resid", "metal-native")
    if not os.path.exists(cwd):
        cwd = os.path.join("Rust_MLKit", "arch_02_value_resid", "metal-native")

    print("Compiling engine...", flush=True)
    subprocess.run(["cargo", "build", "--release", "--bin", "train"], cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Compilation finished. Starting 9-arm crossover experiment...\n", flush=True)

    results = []

    for preset in presets:
        for mixer in mixers:
            cmd = [
                "cargo", "run", "--release", "--bin", "train", "--",
                "--preset", preset,
                "--mixer", mixer,
                "--iters", str(max_steps),
                "--bench-steps", str(max_steps)
            ]
            
            try:
                result = subprocess.run(
                    cmd, 
                    cwd=cwd, 
                    capture_output=True, 
                    text=True, 
                    timeout=90
                )
                
                output = result.stdout + result.stderr
                
                params_match = re.search(r"params=([\d\.]+M)", output)
                params = params_match.group(1) if params_match else "N/A"
                
                toks_matches = re.findall(r"\|\s+([\d]+)\s+tok/s", output)
                loss_matches = re.findall(r"loss\s+([\d\.]+)", output)
                
                tok_s = toks_matches[-1] if toks_matches else "N/A"
                final_loss = loss_matches[-1] if loss_matches else "N/A"
                
                print(f"{preset:<15} | {mixer:<12} | {params:<12} | {tok_s:<10} | {final_loss:<10}", flush=True)
                results.append((preset, mixer, params, tok_s, final_loss))
                
            except subprocess.TimeoutExpired:
                print(f"{preset:<15} | {mixer:<12} | {'TIMEOUT':<12} | {'N/A':<10} | {'N/A':<10}", flush=True)
            except Exception as e:
                print(f"{preset:<15} | {mixer:<12} | {'ERROR':<12} | {'N/A':<10} | {'N/A':<10}", flush=True)
                print(str(e), flush=True)

    print("\nExperiment Complete.", flush=True)
    
    # Save the results to an artifact
    with open("/Users/bharath/.gemini/antigravity/brain/9ee32a8b-09bb-4b08-b10f-7d55ecae0ff2/crossover_results.md", "w") as f:
        f.write("# 9-Arm Crossover Results\n\n")
        f.write("| Preset | Mixer | Params | Tok/s | Loss@30 |\n")
        f.write("|--------|-------|--------|-------|---------|\n")
        for res in results:
            f.write(f"| {res[0]} | {res[1]} | {res[2]} | {res[3]} | {res[4]} |\n")

if __name__ == "__main__":
    run_experiment()
