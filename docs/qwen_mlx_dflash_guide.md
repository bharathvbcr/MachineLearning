# Qwen 27B Speculative Decoding Guide (Apple Silicon)

Setup, inference pipelines, and benchmarks for **Qwen 27B speculative decoding** on Apple Silicon
(Apple M5 Pro, 64 GB unified memory).

Status as of **2026-08-17**: the stack has moved from Qwen3.6 to **Qwen3.8-27B**, and sections 4b
and 4b-iii are measured. Headline: the **official MTP drafter remains the default** (34.56 tok/s vs
15.26 AR, 91.2% acceptance, 2.26× on 2026-08-16). The cross-applied 3.6 DFlash drafter manages only
1.17× — acceptance falls from 80.5% on 3.6 to 53.5% on 3.8 — so `dflash` is not the path for 3.8.

**3.8-native drafters now exist.** z-lab has still published no Qwen3.8 DFlash checkpoint, but the
DSpark family (DFlash + target auxiliary features + a confidence head) is trained against 3.8
itself and runs on Apple Silicon via `mlx-dspark`. Measured 2026-08-17 (§4b-iii): DSpark is real and
lossless, but MTP beats it on the 4-bit target by 18% (36.91 vs 31.20 tok/s, 18 sigma), and the
8-bit pair did not earn its 32 GB. DSpark accepts *more* per round and is still slower — its 1.36B
drafter costs more per round than MTP's 239 MB head saves.

---

## 1. Model roles

### Qwen3.8 (current target)

| Role | Identifier | Type | Size | Engine |
| :--- | :--- | :--- | :--- | :--- |
| **Target** | `mlx-community/Qwen3.8-27B-4bit` | affine 4-bit, gs=64 | ~16.1 GB | `mlx-lm` / `mlx-vlm` / `dflash-mlx` |
| **MTP drafter** | `mlx-community/Qwen3.8-27B-MTP-4bit` | native MTP head, block size 3 | 239 MB | `mlx-vlm` |
| **DFlash drafter** | `z-lab/Qwen3.6-27B-DFlash` | block diffusion, 16 tok/pass | ~1.5 GB | `dflash-mlx` |
| **DSpark drafter (4-bit)** | `DimInfer/Qwen3.8-27B-Dspark-v1` | 1.36B, block 15, 3.8-native | ~3.5 GB | `mlx-dspark` |
| **DSpark drafter (8-bit)** | `RadixArk/Qwen3.8-27B-DSpark` | 1.36B, block 7, 3.8-native | ~2.7 GB | `mlx-dspark` |

The MTP drafter is split from the **same** `Qwen/Qwen3.8-27B` checkpoint as the target — it holds
only the MTP weights and borrows the target's embeddings and LM head at runtime. Same training run,
so there is no cross-model acceptance risk.

The DSpark drafters are **3.8-native** — trained against 3.8's own hidden states, so they carry
none of the cross-application risk below. Each is matched to a precision: DimInfer to the 4-bit
class, RadixArk to the FP8/8-bit verifier. On this machine that pairing rule did not hold (§4b-iii).

The DFlash drafter is a **cross-application**. There is still no `z-lab/Qwen3.8-27B-DFlash`, and
none is announced — verified against z-lab's published model list on 2026-08-17, whose newest
DFlash checkpoints are Alpamayo (2026-07-04) and gemma4-12B (2026-06-28). 3.6 and 3.8 match on every field DFlash depends on (`hidden_size=5120`,
`num_hidden_layers=64`, identical `layer_types`, `vocab_size=248320`, mask id `248070`), and the
drafter's tap layers `[1, 16, 31, 46, 61]` all exist on 3.8 — so it will **load**. Whether it
**accepts** is the open question, because DFlash is trained against a specific target's hidden
states. That is what the benchmark exists to settle.

### Max-throughput alternatives (different quantization)

| Identifier | Notes |
| :--- | :--- |
| `mlx-community/Qwen3.8-27B-nvfp4` + `-MTP-nvfp4` | NVFP4 pair, stays inside the MLX stack |
| `ollama run qwen3.8:27b-mlx` | ~18 GB, modelopt mixed-precision, vision + tools |

These are **not** quality-neutral. A faster number here is a different model, not a free win.

---

## 2. Prerequisites

```bash
# Official MTP drafter path
pip install -U mlx-vlm            # needs >= 0.6.13 for --draft-model / --draft-kind

# DFlash path — v0.1.9 and v0.1.10 are GitHub-only; PyPI still serves 0.1.8
pip install "git+https://github.com/bstnxbt/dflash-mlx@v0.1.10"

# DSpark path — 3.8-native drafters, Apple Silicon
pip install mlx-dspark            # 0.12.2 measured; needs mlx >= 0.32, mlx-vlm >= 0.6.12
```

**The dflash upgrade is not optional if you plan to trust the DFlash arm.** v0.1.9 fixes Qwen
target-wrapper selection for checkpoints that expose both `.model` and `.language_model`.
Qwen3.8 is exactly that shape (`Qwen3_5ForConditionalGeneration`), so on 0.1.8 the verifier can
take an outer wrapper's tied-embedding path instead of the real untied `lm_head`. v0.1.10 also
adds full-context draft features worth +17–32% tok/s on long-context generation.

**If `mlx` came from Homebrew, upgrade it with brew first.** mlx-vlm 0.6.13 requires
`mlx>=0.32.0`, and pip cannot uninstall a brew-installed mlx (no RECORD file), so the install
aborts partway — after having already written pip's `mlx-metal` into `site-packages/mlx/`. That
real directory then blocks brew from symlinking its own `core.cpython-*.so`, and `import
mlx.core` fails. Order that avoids it:

```bash
brew upgrade mlx                                  # 0.32.0; also bumps ollama, which depends on it
brew services restart ollama                      # server keeps the old binary otherwise
pip install --break-system-packages -U mlx-vlm
```

If you already hit the broken state: `pip uninstall mlx-metal`, delete the leftover
`site-packages/mlx/` and `mlx-*.dist-info` (verify they hold only `.pyc` files first), then
`brew link mlx`.

Verified on this machine: mlx 0.32.0, mlx-lm 0.31.3, mlx-vlm 0.6.13, dflash-mlx 0.1.10,
ollama 0.32.13.

### Download

```bash
# 3.8 target + its own MTP drafter + the already-cached 3.6 DFlash drafter
python3 scripts/download_qwen.py --model all-recommended

# everything the benchmark needs, including the NVFP4 pair
python3 scripts/download_qwen.py --model bench-arms
```

---

## 3. CLI quickstart

### A. MTP speculative generation (default)

```bash
python3 scripts/run_qwen_inference.py --prompt "Explain quantum computing in 3 sentences."
```

Or directly:

```bash
mlx_vlm.generate \
  --model mlx-community/Qwen3.8-27B-4bit \
  --draft-model mlx-community/Qwen3.8-27B-MTP-4bit \
  --draft-kind mtp \
  --prompt "Explain quantum computing in 3 sentences." \
  --max-tokens 256 --temperature 0
```

### B. DFlash speculative generation (cross-applied 3.6 drafter)

```bash
python3 scripts/run_qwen_inference.py --mode dflash-speculative --prompt "..."
```

`--draft` is always passed explicitly — the dflash registry has no Qwen3.8 entry and will reject
the target rather than auto-resolving.

### C. AR baseline

```bash
python3 scripts/run_qwen_inference.py --mode mlx-direct --prompt "..."
```

### D. OpenAI-compatible server

```bash
python3 scripts/serve_qwen.py --backend mlx-vlm --port 8000   # MTP — default, fastest (2.18x)
python3 scripts/serve_qwen.py --backend dspark  --port 8000   # 3.8-native DSpark (1.84x, lossless)
python3 scripts/serve_qwen.py --backend dflash  --port 8000   # 3.6 cross-apply (1.17x, superseded)
python3 scripts/serve_qwen.py --backend mlx-lm  --port 8000   # plain AR
```

---

## 4. Benchmarks

### 4a. Qwen3.6 — measured, M5 Pro 64 GB

| Engine | Model | Throughput | Acceptance | Peak memory |
| :--- | :--- | :--- | :--- | :--- |
| MLX baseline (AR) | `Qwen3.6-27B-4bit` | **17.63 tok/s** | — | 15.34 GB |
| DFlash speculative | `Qwen3.6-27B-4bit` + z-lab 3.6 drafter | **52.35 tok/s** | **80.5%** | 16.57 GB |

**2.97× speedup**, lossless. Note this is a fresh-GPU number; Apple Silicon throttles sustained 27B
decode after ~2–3 minutes, so a long session will sit below it.

### 4b. Qwen3.8 — measured, M5 Pro 64 GB, 2026-08-16

`--arms all --repeat 3 --max-tokens 256`, dflash-mlx 0.1.10 / mlx-vlm 0.6.13 / mlx 0.32.0.
Artifacts: `.artifacts/dflash/qwen38_bench/20260816-130645/`.

**Lossless** — all verify against `Qwen3.8-27B-4bit`, so output is interchangeable:

| Arm | tok/s (median) | vs AR | Spread | Acceptance | Peak | Verdict |
| :-- | --: | --: | --: | --: | --: | :-- |
| `ar` | 15.26 | 1.00× | 2.2% | — | 15.72 GB | baseline |
| `mtp` (official drafter) | **34.56** | **2.26×** | 5.3% | 91.2% of drafted | 17.46 GB | **SHIP AS DEFAULT** |
| `dflash` (3.6 cross-applied) | 17.90 | 1.17× | 7.3% | 53.5% | — | keep available, default to AR |

**Max-throughput** — different quantization, quality NOT held constant:

| Arm | tok/s (median) | vs AR | Spread | Note |
| :-- | --: | --: | --: | :-- |
| `nvfp4-mtp` | 34.82 | 2.28× | **34.3%** ⚠ | unstable; ~zero gain over 4-bit `mtp` |
| `ollama` | 29.91 | 1.96× | 21.5% ⚠ | ignored `--max-tokens`; ~2287 tokens/run |

**The cross-apply gamble lost.** On 3.6-to-3.6 the DFlash drafter hit 80.5% acceptance for 2.97×.
Pointed at 3.8 it drops to 53.5% and 1.17× — the run line reads `copyspec 1 blocks / 3 tokens`,
i.e. roughly 3 tokens landing per 16-token block. The drafter does not model 3.8's distribution.

**The safe arm won outright.** The official MTP drafter is the fastest lossless option *and* the
most stable (5.3% spread), despite block size 3 versus DFlash's 16 — acceptance dominates block
size here.

**NVFP4 buys nothing.** 34.82 vs 34.56 tok/s is inside the noise, and its 34.3% spread makes even
that unreliable. There is no throughput case for taking the quantization quality hit.

Every lossless arm shows a monotonic tok/s decline across the three runs even with 60s cooldowns —
these remain fresh-GPU numbers. See §4c.

### 4b-iii. Qwen3.8 DSpark — measured, M5 Pro 64 GB, 2026-08-17

DSpark extends DFlash with target auxiliary features and a confidence head. Unlike the 3.6 DFlash
drafter of §4b, these checkpoints are trained against **3.8 itself**, so this is not a cross-apply.

`--arms ar,mtp,dspark,dspark-8bit --repeat 3 --max-tokens 256`, mlx-dspark 0.12.2 / mlx-vlm 0.6.13 /
mlx 0.32.0. Artifacts: `.artifacts/dflash/qwen38_bench/20260817-213108/` (clean run; the earlier
`20260817-203145/` was taken under load and its verdict section predates the arm-derivation fix).

| Arm | tok/s (median) | vs AR | Spread | Acceptance | Peak | Verdict |
| :-- | --: | --: | --: | --: | --: | :-- |
| `ar` | 16.92 | 1.00x | 1.6% | — | 15.72 GB | baseline |
| `mtp` | **36.91** | **2.18x** | 2.5% | 91.2% of drafted (2.81/round) | 17.46 GB | **SHIP AS DEFAULT** |
| `dspark` (4-bit, DimInfer) | 31.20 | 1.84x | 0.3% | 3.79 accepted/round | — | lossless, second |
| `dspark-8bit` (RadixArk) | 23.50 | 1.39x | 0.4% | 3.51 accepted/round | — | does not earn its 32 GB |

No stability flags and no parse flags on this run; the AR baseline came in at 16.92 tok/s, above the
15.26 of 2026-08-16, so these absolutes are quotable.

**MTP wins outright.** The gap over `dspark` is 5.37 tok/s against a pooled standard deviation of
0.30 across n=3 — **18 sigma**. An earlier contended run put the same comparison at 1.04 sigma and
could not separate them; this one can.

**Higher acceptance, lower throughput.** DSpark accepts 3.79 tokens/round to MTP's 2.81 and is still
18% slower. The reason is drafter cost, not draft quality: MTP's drafter is a 239 MB head split from
the target checkpoint, DSpark's is a full 1.36B model, and that per-round cost outweighs the extra
accepted token. Acceptance is not the figure of merit — tok/s is. (The two engines' per-round
figures are indicative rather than strictly comparable: mlx-dspark documents its count as including
the target's bonus token, and mlx-vlm's convention is not stated.)

**The 8-bit pair did not justify its download.** Slowest speculative arm at 23.50 tok/s, and its
acceptance came in at 3.51/round against the **4.05** its card publishes — while the 4-bit pair
came in at 3.79 against its published **3.28**. Upstream's precision-matching rule (RadixArk trained
against the FP8 verifier, therefore best at 8-bit) inverts on this machine. 32 GB buys 8-bit weights
at ~64% of MTP's throughput; it does not buy speed.

**Losslessness spot-checked, not assumed.** Greedy `--temperature 0` on the same prompt, AR versus
`dspark`: **351/351 characters identical**. DSpark ran 3 tokens past AR's budget (` 7 ×`) because
block granularity overshoots `--max-new-tokens`, which is a budget artefact, not a divergence.

**Qwen3.8 is hybrid-GDN and mlx-dspark handles it.** The target's `layer_types` is 48
`linear_attention` to 16 `full_attention`. §7 previously recorded that mlx-dspark could not run a
hybrid-GDN 3.8 target on this machine; that is refuted — it ran on both the 4-bit and 8-bit targets
and produced verified-identical output.

**The calibrated cap differs from upstream's.** mlx-dspark measured this machine's cost curves and
derived **cap 4** for the DimInfer pair, where upstream's M4 Pro derived 7. The benchmark leaves
`--max-draft` unpinned deliberately, so the table reports what this machine chooses by default.

### 4b-ii. Reproducing

Run:

```bash
python3 scripts/bench_qwen38.py --arms all --repeat 3 --max-tokens 256
```

Arms:

| Arm | Class | What it tests |
| :-- | :-- | :-- |
| `ar` | lossless | `Qwen3.8-27B-4bit` autoregressive — the anchor for every speedup |
| `mtp` | lossless | + official MTP drafter (same checkpoint, block size 3) |
| `dflash` | lossless | + 3.6 DFlash drafter cross-applied (block size 16) |
| `dspark` | lossless | + 3.8-native DSpark drafter on the 4-bit target (mlx-dspark) |
| `dspark-8bit` | max-throughput | 8-bit target + RadixArk DSpark — different quant, quality NOT held constant |
| `nvfp4-mtp` | max-throughput | NVFP4 target + NVFP4 MTP drafter |
| `ollama` | max-throughput | `qwen3.8:27b-mlx` |

Artifacts land in `.artifacts/dflash/qwen38_bench/<timestamp>/` as `results.json` (with raw stdout
per run, so nothing is lost if a metric fails to parse) and `REPORT.md`.

The table above is the measured result; the `\`.artifacts\`` JSON holds raw stdout per run.

### 4b-i. The losslessness gate

Every entry point that shells out to `dflash` checks the installed version first, via
`scripts/dflash_guard.py` (the single owner of this rule). Below **v0.1.9** the DFlash arm
still runs and still reports throughput, but its output is **not** verified against the
target, so it is not a lossless speedup:

`QwenGdnTargetOps.text_wrapper` returns the *outer* wrapper for any checkpoint exposing both
`.model` and `.language_model`. Qwen3.8 is exactly that shape (`Qwen3_5ForConditionalGeneration`).
The outer wrapper has no `args`, so `tie_word_embeddings` falls through to its `True` default
and the verifier computes logits from `embed_tokens.as_linear(...)` instead of the text model's
`lm_head` — a different head, and in practice a different token. Confirmed by executing it
against the installed 0.1.8, and fixed upstream in v0.1.9.

When the gate trips:

- the benchmark marks that arm **INVALID** and refuses to recommend it, however fast it was;
- `REPORT.md` drops its "outputs are interchangeable" claim and prints the blocker instead;
- `results.json` carries `validity.lossless_claim_valid: false`;
- `run_qwen_inference.py` / `serve_qwen.py` print a loud banner before generating or serving;
- `bench_qwen38.py --strict` exits `3` without running at all — use this in CI.

An *undetermined* version is treated as blocked, not as fine. Note `dflash --version` does not
exist (argparse exits 2), so the version comes from package metadata.

Regression suite, no models or network required:

```bash
python3 scripts/test_bench_qwen38.py
```

### 4c. Sustained throughput (separate question)

`bench_qwen38.py` at `--max-tokens 256` decodes in seconds, so every arm it reports is a
fresh-GPU number — the benchmark now says so in its preflight. Steady state is lower, and
speculative arms lose more than AR because the drafter competes for the same GPU. To get a
sustained figure, use dflash's own harness (requires **v0.1.10** — the flag does not exist on
0.1.8):

```bash
dflash benchmark --model mlx-community/Qwen3.8-27B-4bit \
  --draft z-lab/Qwen3.6-27B-DFlash --sustained-minutes 10
```

Treat peak and sustained as two different numbers. If they disagree about which arm wins,
sustained is the one that matches how you actually use the model.

### Decision rule (lossless arms only)

Applied automatically by the benchmark's verdict block:

- Acceptance **≥ 60%** and **> 1.8×** AR → make it the default engine.
- Acceptance **30–60%** → keep the mode available, default serving to AR.
- Acceptance **< 30%**, or slower than AR → do not ship that pairing.

A reasonable prior, not a prediction: the MTP drafter is the safe arm (same checkpoint, but block
size 3 caps its ceiling), while DFlash has the higher ceiling (block size 16) and the real risk of
acceptance collapse. Block-16 drafting also only converts to wall-clock when acceptance is high,
because on Apple Silicon verify cost grows with the number of tokens verified. Let the numbers
decide.

---

## 5. API query example

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.8-27B-4bit",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 100
  }'
```

---

## 6. Coding agent against the local server

A terminal coding agent needs three things from the endpoint: `/v1/models` for discovery,
`/v1/chat/completions` for the loop, and **tool calling**. mlx-vlm's server provides all three —
it registers `/v1/models`, `/v1/chat/completions`, and `/v1/responses`, and ships a `qwen3_coder`
tool parser that recognises Qwen's `<tool_call>\n<function=` emission.

**Use [kon](https://github.com/0xku/kon).** Five harnesses were driven end to end against this
server and scored by `scripts/bench_agents.py`. kon is the only one that was both correct and fast
— see §6a.

### Launching it

Two shell functions in `~/.zshrc` do everything — `qq` starts the server if it is not already up,
waits for it, then hands off to kon:

```bash
qq                      # interactive
qq -p "your task"       # one-shot
qwen-stop               # unload the model, frees ~15 GB
```

Measured cold (server down, page cache warm): **36 s** from `qq` to answer. A genuinely cold first
run after boot is slower, since the 15 GB has to come off disk.

kon expects noise on stderr at exit — `RuntimeError: generator didn't stop after athrow()` — from
its own async teardown. The run still exits 0 with correct output; ignore it.

### Doing it by hand

```bash
uv tool install kon-coding-agent

APC_ENABLED=1 python3 scripts/serve_qwen.py --backend mlx-vlm --port 8000   # the 2.26x MTP arm
kon -p "<task>" \
  --provider openai \
  --model mlx-community/Qwen3.8-27B-4bit \
  --base-url http://127.0.0.1:8000/v1 \
  --openai-compat-auth none
```

`--openai-compat-auth none` is what lets it skip the API key mlx-vlm never checks. The `--model`
string must match what `GET /v1/models` reports, which is whatever you passed to
`serve_qwen.py --model`.

Those four flags can move into `~/.config/kon/config.toml` instead, which is what makes bare `kon`
(and therefore `qq`) work:

```toml
[llm]
default_provider = "openai"
default_model    = "mlx-community/Qwen3.8-27B-4bit"
default_base_url = "http://127.0.0.1:8000/v1"

[llm.auth]
openai_compat = "none"
```

### What kon is built on

**No agent framework.** kon is ~22,800 lines of Python calling the official provider SDKs directly:

| Layer | Dependency |
| :-- | :-- |
| Providers | `openai>=2.21`, `anthropic>=0.79` (official SDKs, no LangChain / LlamaIndex / AutoGen / CrewAI) |
| Tool schemas | `pydantic` |
| TUI | `textual`, `rich` |
| I/O | `aiohttp`, `aiofiles`, `curl-cffi` |
| Web tools | `ddgs`, `readability-lxml`, `html-to-markdown` |

Eight built-in tools — `read`, `write`, `edit`, `bash`, `find`, `grep`, `web_fetch`, `web_search`
— and a hand-rolled agent loop (`kon/loop.py`, `kon/turn.py`) over a thin provider abstraction
(`kon/llm/`). The absence of a framework is the point: nothing injects scaffolding into the prompt.

Its base system prompt lives in `kon/defaults/config.toml` and measures **125 tokens** under the
Qwen3.8 tokenizer (583 chars) — verified, not quoted. The ~3,030 tokens the benchmark records is
that base plus tool schemas, file contents, and conversation growth.

### 6a. Measured: prompt size decides everything

Five harnesses, same server, same task, scored by `scripts/bench_agents.py` on 2026-08-16.
Correctness is decided by parsing the resulting file's AST — never by what the agent said it did.

| Harness | Result | Wall | Max prompt | Disk |
| :-- | :-- | --: | --: | --: |
| **kon** | **PASS** | **52.8 s** | **3,030** | 130 M |
| zero | PASS | 109.3 s | 9,341 | 137 M |
| qwen-code | PASS | 367.6 s | 33,533 | 119 M |
| pi | **FAIL** | 30.1 s | 2,096 | 139 M |
| oh-my-pi (`omp`) | 2 of 3 correct | ~802 s | ~39,000 | 990 M |

Wall clock tracks prompt size almost linearly, because **cold prefill runs at ~74–101 tok/s no
matter which harness is driving**. Nothing a harness does changes the server's speed; the only
variable is how many tokens must pass through it — 3k costs ~20 s, 33k costs ~340 s.

kon wins by being correct *and* fastest among the correct. pi is quicker still with the smallest
prompt, but it got the task wrong.

Two harnesses needed non-obvious configuration before they would run at all, now encoded in
`scripts/bench_agents.py`:

- **qwen-code** aborts a stream after 240 s without a chunk. Its ~33.5k prompt needs ~340 s of
  prefill, so at defaults it reliably kills the request it just issued and completes nothing —
  set `QWEN_STREAM_IDLE_TIMEOUT_MS=0`. It also refuses every edit/write/shell tool headlessly
  without `-y`, reasoning correctly and then changing nothing.
- **kon** takes the endpoint entirely on the command line and needs `--openai-compat-auth none`.
  No config file anywhere.

`APC_ENABLED=1` (off by default; env var, no CLI flag) is worth setting and is **not** the fix.
It works exactly as advertised when it hits — omp's cache-hit requests clock 7,400–12,100 tok/s and
return in 7–9 s. But 2 of every 5 requests miss, because omp's prompt varies near the front and the
shared prefix is short. omp turn 2 against a cache reporting `token_hit_rate: 1.0` took **829 s vs
802 s cold — no improvement.** A 100% hit-rate reading is not evidence of a fast run; check
per-request `prefill=` in the server log instead.

### 6b. Never trust an agent's self-report

**Two of five harnesses stated they had verified work they had not done.** This is why
`bench_agents.py` decides correctness by AST and treats agent output as narration only.

- **omp** was told to delete two named unused imports and deleted **three**, including `re`, which
  was used on three lines. It then reported: *"Verified: the file still parses cleanly
  (ast.parse) ... the file retains the subprocess/re/os imports it actually uses."* Both claims
  false. The file *did* still parse — which is exactly why its own check passed. Removing a used
  import is a runtime `NameError`, not a syntax error, so `ast.parse` and `py_compile` both sail
  through. Only an explicit set comparison catches it.
- **pi** left `sys` in place and reported *"Kept `subprocess`, `re`, `sys`, and `os` since they're
  all actually referenced."* The file contains zero references to `sys`.

Practical consequences: **snapshot any file before pointing a local agent at it**, and never accept
"I verified it" as verification. A syntax check is not a correctness check.

Verdict: kon at ~53 s per single-file edit is a usable loop on this hardware. qwen-code at ~6 min
is tolerable for one-shots. omp at ~13 minutes is not.

## 7. Out of scope

- Training a 3.8-specific DFlash drafter (z-lab's recipe is not public). Moot as of 2026-08-17:
  the DSpark family covers 3.8 natively and is measured in §4b-iii, so there is nothing left for a
  hand-trained drafter to unlock here.
- A Rust/Metal GatedDeltaNet port. `Rust_MLKit/gemma-metal` holds Gemma-4 kernels (SWA, GELU MLP,
  PLE); Qwen3.8 needs GDN + Qwen MLP/RoPE/MTP. New architecture, multi-week project, not a wiring
  task.
