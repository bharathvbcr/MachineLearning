# Architectural opportunities in MLSystemsLab and BINN

> **Dated evidence snapshot, not live campaign status.** Imported from the September 4 review (inventory timestamp 21:48 UTC). Later discussion refined the proposed operator: prediction drift is not necessarily harm, and blocked scalar writes do not establish capacity exhaustion; the under-budget reading of the wall-clock board below was withdrawn (every seed reached its token budget; one ran at lower tenancy). The [current assessment](README.md) owns the consolidated verdict; the [memory protocol](MEMORY_UPDATE_PROTOCOL.md) supersedes the operator recommendation below. Training was not rerun for this document.

September 4, 2026. Assessment of the local research snapshot, with live primary-literature checks.

**My conclusion:** the available evidence does not establish a new architecture with attention-level impact. It does justify a focused search for a better memory operation. The best supported opportunity is to learn when compressed memory can safely absorb information, when it must preserve an explicit association, and when a query needs additional computation. A generic recurrent/attention hybrid, value residual, recurrent depth, or surprise-selected cache would substantially overlap existing work. The contribution must be narrower, measurable, and transferable.

I would organize the next research program around **memory interference and retrieval**, keep **value-anchored recurrent depth** as the cheaper secondary experiment, and treat **event timing and synchrony** as a distinct exploratory branch. These are hypotheses proposed here, not discoveries demonstrated by your current runs. No probability of a field-changing breakthrough can responsibly be assigned from these results.

**What I reviewed and what the verification means**

The primary scope is MLSystemsLab: its architecture sources, experiment-note index, paper, ablation records, research manifests, recent crossover campaigns, and recall results. BINN supplies a related source of temporal-mechanism hypotheses. Other research repositories, unsynced remote experiments, and private datasets were not comprehensively reviewed.

The analysis [script](evidence/recompute.py) inventoried **3,463 JSON/JSONL/CSV result artifacts**, representing **3,140 distinct file hashes**, and parsed **1,252 metrics files** across seven result roots. These are file counts, not independent experiments. Copies, summaries, restarts, and failed attempts remain identifiable. It examined **448 crossover metrics files across 18 directories**, and **591 recall records**: 406 in E8, including one older calibration record, and 185 in the newer recall boards. The full inventory and source hashes are in [artifact-manifest.json](evidence/artifact-manifest.json).

The new final-loss tables use **241 runs with explicit finite `final_val`**. Older completed runs without that field are not silently substituted with `best_val`; their evaluation curves remain covered by the repository's existing figure checker where applicable. Inventorying the systems benchmarks is not equivalent to independently validating every throughput claim. I did not rerun full training, physical GPU kernels, inference products, or cloud campaigns.

Throughout this report:

- **Verified:** recomputed from accessible records, established by inspected source, or reproduced by a named check. Archived results establish what was recorded, not independent reproducibility of the training itself.
- **Inferred:** a plausible explanation or research recommendation supported indirectly by those measurements.
- **Unverified / report-backed:** a result stated in a research report whose underlying cells were unavailable here.

The repository's `python3 paper/derive_figures.py --check` passed, but explicitly reported **61/706 stated figures derived or documented (9%)**; it checked 60 derived figures. It is not whole-manuscript validation. The parser also recorded one historical nonfinite metric in `out/funnel/lr16m/muon_ns5_adamw_lr0.00625.attempt-1784077549/metrics.jsonl`; this failed attempt is not a successful finite observation. [Verification record](evidence/verification-notes.txt).

**What the experiments actually suggest**

All CE differences below are nats per token, with lower better. Small-sample Student-t intervals describe seed variation; they are not a correction for searching many designs. Five wins on five seeds alone is not a two-sided sign test at p < .05. Overlapping intervals are not proof of equivalence. The [recomputed tables](evidence/evidence-tables.md) and [per-seed evidence](evidence/recomputed-evidence.json) contain the underlying values and source lines.

| Research family | Current evidence | Architectural interpretation |
|---|---|---|
| Value residual and gated attention | **Verified:** long-stage value residual 1.987491 BPB; gated+value 1.984742; gating alone 2.088671, two seeds each. The added-gate effect changes sign between seeds. | Preserve early value information as a useful baseline. The extra gate is not established as a reliable gain. These six long-stage records contain no ungated/no-value control, so they alone cannot establish that gating hurts that control or isolate a full factorial interaction. |
| Compact versus deeper models; lean auxiliary heads | Earlier notes record compact depth/width tradeoffs and auxiliary-head gains; those do not isolate a new sequence operator. | Useful efficiency controls. Their selected settings must be retuned at the scale and horizon being compared. |
| Attention, minGRU, GDN, Mamba-2, MLA | **Verified within the figure check's coverage:** crossover location changes with batch and LR horizon. The older 7M attention/minGRU boundary is not universal. | There is no support for a fixed training-token threshold at which one architecture intrinsically becomes superior. |
| Hybrid ratio and placement | **Verified:** periodic 9 minGRU + 3 attention gives 4.193875 versus 4.231331 for 10+2; paired difference −0.037455 [−0.042565, −0.032346], n=5. Last-two attention beats first-and-last placement by 0.028290 with the same attention count. | Placement matters. Ratio and parameter count move together, so “three attention layers are universally optimal” is not established. The placement comparison is the cleaner intervention. |
| Local versus global attention, context 512 | **Verified:** full 4.217875; window256+sinks 4.222840; window128+sinks 4.254328; window64+sinks 4.316473. All n=5. | Much of this short-context LM objective is available locally. A very small window still costs quality. |
| Local versus global attention, context 2048 | **Verified:** full 4.199478; window512+sinks 4.197756. Paired difference −0.001722 [−0.014054, +0.010609], n=5. Window256 loses 0.020653 [0.010735, 0.030570]. | The 512-window arm has no detected CE deficit at this recipe. This is neither a proof of equivalence nor evidence that long-range retrieval is unnecessary. |
| Attention sinks | **Verified:** at window64, adding four sinks worsens LM CE by 0.008257 at context512 and 0.011237 at context2048, each 5/5 seeds. On the harder recall board it raises low mean recall without producing a solved seed. | Keeping a few prefix tokens is not a demonstrated substitute for task-relevant memory. Sinks may have different numerical or streaming roles elsewhere. |
| E8 associative recall | **Verified:** at four pairs/9000 steps, pure GDN reaches the ≥80% recall threshold in 13/15 seeds, minGRU 1/15, attention 15/15. At eight pairs, GDN 6/15, minGRU 0/15, attention 12/15, both studied hybrids 13/15. | The failure is not shared by all recurrent models. Matrix state, key/value addressing, update rule, optimization, and state capacity are candidates to isolate. Recall thresholds do not demonstrate that an induction-head circuit formed. |
| E16 harder recall | **Verified:** 64 pairs/sequence255/3000 steps: attention 11/15 solved; GDN, minGRU, window64±sinks all 0/15. At 128 pairs/sequence511: attention 8/10; window256 6/10; window512 8/10; GDN and minGRU 0/10. | The small-pair GDN result does not establish scalable exact retrieval. Pair count, distance, vocabulary size, and sometimes batch change across cells; there is no identified distance-only or capacity-only effect. |
| Shared depth at equal tokens | **Verified:** six blocks used twice have 81.17M parameters and CE 4.232161; twelve distinct blocks have 123.70M and CE 4.218434. Difference +0.013727 [0.008992, 0.018463], n=5. Six blocks once give 4.295998; reuse improves them by 0.063837. | A useful parameter/quality tradeoff: 34.38% fewer total parameters with a small, measured loss. It is not “shared depth fails.” Extra passes help the same small model at equal tokens. |
| Shared depth at nominal equal time | **Verified:** the six-block unlooped arm averaged 582.7s against a roughly 674s target; other arms averaged about 671–674s. | The complete board fails the existing 5% matching tolerance. Do not rank all five as equal-time measurements. The three-block and looped-three-block pair is close in time and favors the shallow model, warning that more tokens can beat more passes at this budget. **Superseded later on 2026-09-04:** every seed, including the 221 s one, reached its pre-registered token budget; that seed ran alone (tenancy 1) after the other jobs finished. The loss board is matched by construction; the correction is to record tenancy per run. See the [corrections table](README.md#corrections-that-govern-future-claims). |
| Width ladder | **Verified:** attention beats minGRU at widths 384, 768, 1152 by 0.1471, 0.1585, 0.1517 CE, 5/5 seeds each. Thirty individual metrics files are present locally. | The sign survives this width ladder. LRs were selected at 10M tokens and transferred to 50M, so the ladder is not a comparison at independently established 50M optima. Width-matched arms are not parameter-matched. |
| MoE | **Verified:** old 20-run board includes balancing loss in reported values. Corrected local snapshot: 10/20 complete, 2 marked running, 8 pending; all completed runs are dense and one-expert controls. | The corrected controls approximately agree. Multi-expert conclusions remain unavailable, and the current top-1 router has the task-gradient issue below. Queue labels do not verify a live remote process. |
| µP and attention temperature | The current source uses a particular initialization/scaling combination; saved ablations and the paper attribute a large arm-asymmetric loss penalty to it. | This is a parametrization diagnosis, not evidence against µP generally, and not an architecture discovery. QK scaling and optimizer-specific transfer must be controlled before comparing new operators. |
| Muon/Polar/NorMuon optimizer funnel | **Verified through the repository checker:** D7 manifest agrees with the ledger. The selected champion is retired after LR-conditioned rank changes. | Keep optimizer engineering; stop treating an inherited-LR winner as a universal optimizer. A new architecture needs a fair tuning budget. |
| APRDH | **Verified source design:** shared GDN passes, byte processing, patch routing, engram lookup, sparse MLA queries, gain predictors and optional fast-weight adaptation. Archived notes report an interrupted/unranked comparison. | A collection of plausible ingredients, not evidence for their combination. The main recurrence computes GDN and FFN before masking updates; masking does not establish reduced executed compute. Full-context MLA keys also prevent a blanket constant-memory claim for the composite model. |
| Diffusion and AR conversion | Preserved runs establish adaptation experiments; the notes distinguish a short/restarted FineWeb arm from TinyStories adaptation and report an earlier target-construction defect. | Promising implementation surface, but masked-denoising CE/exponentiated loss is not directly interchangeable with AR likelihood/perplexity. No equal-quality speed superiority is established by those logs alone. |
| Quantization, QAT, packing, SFT and STaR | Code and artifacts support separate engineering or objective choices; the inspected architecture comparisons do not establish new operators from them. | Retain as downstream efficiency/capability tests. They should not all be varied while identifying a memory mechanism. |
| Tessl, native Metal training, chunked GDN/Mamba, Gemma/DFlash/MTP | Kernel and deployment research provides implementation capacity and measurement lessons. This review inventoried those outputs, rather than rerunning every numerical/performance gate. | A strong substrate for co-design. Custom kernels, speculative acceptance, artifact size, and language-model quality answer different questions. None by itself demonstrates a novel learning architecture. |
| BINN temporal work | **Report-backed:** W9's attention readout gain depends strongly on intact temporal structure; W25 has an unevaluable recurrent comparison; W26 finds a structural positional-code effect and a residual gap versus shuffling. | Supplies controlled temporal hypotheses, not evidence that spikes or local credit rules replace attention/backpropagation. W26 raw cells were unavailable for local re-analysis. *2026-09-05 note:* wave 27 landed after this snapshot (τ½ = 261 ms; the QK-norm arm is NOT MET at −0.0692 accuracy; dropout at p30 is an insensitive instrument). The wave 26 and 27 cells are tracked in BINN's repository (`results/shd_attention_campaign_v3/`). |

Several older narrative claims are therefore unsafe: “gating alone hurts” without its matched baseline; “all recurrence fails recall”; “looping destroys useful depth”; “we have no scale ladder”; “the MoE board measures trained expert specialization”; and “a green figure check verifies the whole paper.” Each needs the narrower evidence statement above. There is also a substantive distinction between the repository's GDN variant and the published operator.

**A concrete GDN comparator difference, reproduced**

The live [chunked operator](../../nanolab/mixers.py#L684) and its sequential reference compute, with M written in value-by-key orientation:

$$
M_t=\alpha_t M_{t-1}+\beta_t(v_t-M_{t-1}k_t)k_t^\top.
$$

The [published Gated DeltaNet rule, Table 1](https://arxiv.org/html/2412.06464v1) instead computes:

$$
M_t=\alpha_t M_{t-1}+\beta_t(v_t-\alpha_t M_{t-1}k_t)k_t^\top.
$$

For a unit key, the old-state transition along that key is **α−β** in the repository, versus **α(1−β)** in the paper. The first can change sign when β exceeds α; the second is nonnegative with the stated gate ranges. With unit keys and α,β in [0,1], the repository transition nevertheless has spectral norm at most one. A sign flip alone does not establish unstable state propagation or an unstable full-network Jacobian. This is a different memory operation, not merely a tensor-layout choice.

The [CPU probe](evidence/gdn_rule_probe.py) calls the actual chunked function on two unit key/value/query tokens, zero initial state, α=β=0.5. It returns **[0.5, 0.5]**; the published update gives **[0.5, 0.625]**. Controls with α=1 or β=0 agree. The difference is therefore reproduced, and tests against the repository's own sequential reference cannot detect it because that reference implements the same variant.

This does **not** establish that this difference caused the recall failures, or that the implemented variant is intrinsically inferior. The module also differs in projections/local convolution/output gating from the paper's full block. The appropriate next experiment is an explicit implemented-rule versus published-rule ablation, followed by a faithful-block comparator. Preserve the old run identities. A potentially interesting signed-memory behavior can be studied deliberately, but a different update is not automatically a novel or better architecture; signed transition extensions already occur in the newer GDN literature.

Every GDN result above should therefore be read as a result about **this repository's implemented GDN variant**. This comparator check precedes attributing its hard-recall failure to finite memory capacity or interference.

**A concrete issue to resolve before interpreting the MoE campaign**

The current [MoE implementation](../../nanolab/model.py#L89) selects top-k probabilities and normalizes by their sum. At k=1 the task multiplier is p/p=1. The selected integer index supplies no ordinary autograd derivative, leaving the router without a useful task-loss gradient. Load balancing still trains it; experts still receive the task signal.

The [CPU probe](evidence/router_gradient_probe.py) exercised the actual module, four experts, width16, float64:

| Route | Task router gradient norm | Auxiliary router gradient norm |
|---|---:|---:|
| Top-1 | 5.15e-18 | 0.08526 |
| Top-2 | 0.04880 | 0.08526 |

This verifies the local gradient mechanism, not the eventual training-quality penalty. The existing one-expert control cannot detect this problem because a one-expert model needs no meaningful choice of expert. Before drawing a general MoE conclusion, establish a task-trained routing mechanism and a controlled synthetic specialization test. A changed router requires a separate experiment identity and fresh matched runs; it should not be pooled with this campaign. No source was changed in this review.

**Where prior work already occupies the obvious designs**

| Proposed ingredient | Closest primary prior work checked | Consequence for a novelty claim |
|---|---|---|
| Carry the first layer's values through depth | [ResFormer / Value Residual Learning](https://arxiv.org/abs/2410.17897) | Your value-residual result is an implementation/empirical result, not priority for that operation. |
| Reuse blocks and choose depth | [Universal Transformers](https://arxiv.org/abs/1807.03819), [recurrent-depth latent reasoning](https://arxiv.org/abs/2502.05171), [Universal Transformers Need Memory](https://arxiv.org/abs/2604.21999) | Reuse, adaptive halting, and adding memory to recurrent depth already have close precedents. |
| Allocate compute to selected tokens | [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) | “Use more compute on difficult tokens” is too broad to be the contribution. |
| Addressable recurrent memory with better updates | [DeltaNet](https://arxiv.org/abs/2406.06484), [Gated DeltaNet](https://arxiv.org/abs/2412.06464), [Longhorn](https://arxiv.org/abs/2407.14207), [MIRAS](https://arxiv.org/abs/2504.13173), [ATLAS](https://arxiv.org/abs/2505.23735) | Online regression, retention objectives, and better memory optimization are established directions. |
| Separate erase/write controls | [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) | This specific extension is already explicit, including a chunked training algorithm. |
| Cache what a recurrent model cannot predict | [Hybrid Associative Memories](https://arxiv.org/abs/2603.22325) | Very close to the first obvious synthesis of your LM/recall results. Its full text includes learned routers as well as prediction-error routing. |
| Make recurrent memory grow | [Memory Caching](https://arxiv.org/abs/2602.24281) | Caching recurrent-state checkpoints is also occupied. |
| Learn memory at inference | [Titans](https://arxiv.org/abs/2501.00663) | Generic surprise-driven test-time memory is not new. |
| Dynamic byte patches, including diffusion acceleration | [BLT](https://arxiv.org/abs/2412.09871), [H-Net](https://arxiv.org/abs/2507.07955), [Fast BLT](https://arxiv.org/abs/2605.08044) | Even combining byte patches and diffusion has a direct current comparator. |

This is a targeted prior-art screen, not an exhaustive novelty or patent search. The cited authors' empirical claims were not independently reproduced. The comparisons are enough to reject overly broad priority claims; they do not prove that the narrower proposals below are original.

**Primary proposal: memory updates selected by their damage to existing associations**

**Status: inferred opportunity; novelty and performance unverified.**

The useful question is: *Can the model estimate the damage a proposed memory write would cause, then use that estimate to allocate a fixed memory budget better than prediction-error routing?*

This follows from two verified observations: GDN can retrieve where minGRU struggles at small load, but also fails the harder recall cells; short-window LM quality can remain near full attention while retrieval deteriorates. Together they motivate measuring compressed-memory failure directly. They do not yet identify interference as its cause.

Begin with a standard simplified delta-rule memory M of shape d_value × d_key:

$$
\hat v_t=M_{t-1}k_t,\qquad e_t=v_t-\hat v_t,\qquad
\Delta M_t=\beta_t e_t k_t^\top.
$$

For an old key k_i, the write changes its retrieved value by

$$
\Delta M_t k_i=\beta_t e_t(k_t^\top k_i).
$$

This familiar identity gives a concrete falsification: equal new-item prediction error can cause very different old-item damage depending on key overlap. It is not a new theorem, and this simplified update is not asserted to be identical to every decay convention in the current GDN implementation.

**Superseded interpretation:** the following score measures prediction drift, not actual loss increase. Retain it as a comparator; use the residual cross-term and direction-aware protocol in [MEMORY_UPDATE_PROTOCOL.md](MEMORY_UPDATE_PROTOCOL.md).

Test a small protected set of previous keys P with a prediction-drift score

$$
I_t=\sum_{i\in P}w_i\|\Delta M_t k_i\|^2.
$$

The proposed layer has three operations:

1. A local attention or recurrent path handles ordinary nearby computation.
2. A compressed associative memory accepts a write when its estimated retention cost is low.
3. A strictly budgeted explicit store preserves selected associations when the compressed write is damaging; a query can retrieve from that store.

Initially use a fixed policy and actual counterfactual damage. Only after that works should a small controller learn to predict the benefit of writing, preserving, or retrieving. Train it from controlled comparisons of the actions during training, without future answers at inference. Use a fixed anchor/storage byte budget, and compare random anchors and simple FIFO policies to learned protection. Otherwise a gain may simply come from adding memory.

**What might distinguish it:** the explicit measurement and control of harm to previously retained associations, evaluated against HAM-style surprise routing and against the same architecture without that signal. Generic selective caching, learned routing, delta updates, and regularized regression are already known. A covariance-preconditioned or recursive-least-squares update is a baseline here, not a novelty claim. If the proposed mechanism reduces to one of those baselines, credit it accordingly.

**The decisive experiment:** independently vary key similarity, number of stored associations, distractor distance, and overwrite frequency. Hold state bytes fixed. First show that the proposed interference score predicts which prior facts are lost better than incoming prediction error. Then intervene on the score/policy and show that reducing the measured interference recovers those facts. Failure of either step defeats the proposed explanation.

**The hardest engineering issue:** a policy depending on the evolving recurrent state can destroy the parallel scan/WY execution structure. Do not claim to inherit GDN's throughput automatically. Prototype the causal sequential operator, then evaluate an explicitly specified lagged or chunk-boundary policy against it. Include routing, anchor maintenance, memory traffic, and synchronization in the timing. If a parallel approximation removes the benefit, this may remain a mechanism paper rather than an efficient backbone.

**The unavoidable limit:** finite-precision fixed-size state cannot retain arbitrarily many independent random associations exactly. Measure recall as a function of bytes, including the explicit store and metadata. Growth of that store must be charged; a fixed cap must expose eviction losses. The attainable target is a better accuracy/memory/latency tradeoff on useful distributions.

**Secondary proposal: recurrent depth with an immutable value memory and real selective execution**

**Status: an evidence-supported parameter-efficiency experiment; broad novelty is low.**

The six-block/two-pass result is more encouraging than the older “recursion fails” narrative. It supports asking whether a shared computational core can refine queries while retaining an anchored representation of token values. The current loop already keeps the first exposed v0 across passes, so this baseline should be extended and ablated rather than reimplemented. [Forward loop](../../nanolab/model.py#L371).

A testable form is

$$
h^{(r+1)}=h^{(r)}+F_\theta(h^{(r)},V_0,M,r),\qquad
a^{(r)}=\mathbb{1}[\widehat{\Delta\mathcal L}^{(r)}>\lambda\,\widehat{c}^{(r)}].
$$

V0 is fixed within the example; h carries computation; M carries retrieved associations. The controller predicts the benefit of another pass relative to its measured cost. A small pass embedding or low-rank adapter is an optional comparator, not an assumption that specialization is necessary.

Run a clean factorial: fixed versus refreshed/absent value anchor; one versus two/four shared passes; fixed-depth versus random-budget versus learned-budget policy. Compare both equal parameters and equal executed compute. Keep a twelve-block reference. Establish task-gradient flow to the controller and a genuine reduction in dispatched work. APRDH-style computing the full update and multiplying by a stop mask is not that reduction.

Use compositional retrieval and reasoning probes in addition to CE. The interesting result would be that extra passes improve multi-step composition because they revisit a stable memory, with interventions localizing that role. Mere additional iterations, improved small-model CE, or a learned halting gate would remain close to the existing recurrent-depth literature.

Do not begin by adding byte patching, MoE, diffusion, fast weights, and adaptive depth simultaneously. The present code already has the components for a much smaller isolating experiment.

**Exploratory proposal: separate content, relative order, and synchrony in event memory**

**Status: hypothesis motivated by report-backed BINN results; weakest basis for an immediate general-purpose LM claim.**

Today's Wave 26 report (`BINN/results/RESULT_2026-09-04_W26_SATURATION_IS_REAL_AND_NOT_SPECIFIC.md`, external sibling repository; not bundled) states that removing positional coding reduces readout advantage from 0.1258 to 0.0506, while bin shuffling leaves 0.0050. It also reports that saturation occurs in both the collapsing readout and its control, defeating the registered collapse-specific explanation. Its 264/264 coverage claim and exact numbers were not rederived: the local analyser returned `NOTHING TO ANALYSE`, exit 2. *(2026-09-05: the cells exist in BINN's git, so the numbers are rederivable there; on that date BINN's own every-number and verdict sweeps did not yet include waves 26–27, which BINN's audit record states.)*

Removing positional features and shuffling raw input are not equivalent interventions. A recurrent or filtered feature extractor can already encode temporal history; shuffling can alter those features and task information. Thus the residual gain is an unanswered mechanism question, not proof of a new temporal code.

The experiment to earn an architecture is a factorial decomposition of the readout inputs: count/rate content, relative temporal structure, and cross-channel coincidence. Use several controlled timescales and compare a simple multiscale linear/conv/state-space baseline before a complex attention module. Test time translation, dilation, reversal, bin-width changes, channel-wise desynchronization, and position removal separately. Preserve the quantities each intervention claims to hold fixed and inspect the actual intermediate representations.

Only if a stable benefit survives those controls should you build a content-addressable event memory whose keys include learned relative-time or synchrony features. Evaluate transfer to a second event/audio dataset. The proposed contribution would be a specific invariance or sufficient representation supported by interventions, not “add time constants” or “use spikes.”

Keep BINN's local-credit and recurrent claims separate: Wave 25's main recurrent mechanism hypothesis was not evaluable after numerical failures. Those failures are not evidence that a new learning rule succeeds or that recurrence is impossible. Saturation also does not logically rule out a useful QK-normalization intervention; it only defeats the specific causal explanation tested in Wave 26. *(Superseded 2026-09-05: Wave 27's H27-4 measured that intervention — NOT MET on both halves; QK-norm is a different read-out, worse at the task and consuming order differently.)*

**How I would overhaul the research program**

The central change is from ranking named architectures to explaining which operation fails under which demand. Preserve the current measurement paper as one output. Run the architecture-discovery program separately, so a result can contradict the original explanation without requiring a narrative rescue.

| Stage | Concrete work | Advance condition | Stop or redirect condition |
|---|---|---|---|
| 1. Repair the comparison contract | Compare the implemented GDN rule with the published rule; resolve the MoE task-gradient issue before a specialization claim; verify task-only CE; record tenancy per run on the wall-clock board (the six-block cell is superseded above); reconcile stale experiment statuses; retain immutable per-run configurations and source fingerprints. | Numerical equivalence controls, gradients, causal masking, replay, and actual executed work are measured. | A comparator still differs in unintended training or reporting behavior. |
| 2. Build a retrieval diagnostic | Fix vocabulary and query count while independently varying pair load, distance, similarity, overwrites, and distractor statistics. Include both seen-symbol unseen-pair and held-out-symbol tests. | A model's failure is localized to addressing, retention, update interference, or optimization. | More training/tuning alone removes the apparent mechanism effect; report that result before adding an operator. |
| 3. Test the smallest operator | Compare diagonal/vector recurrence, matrix associative memory, delta updates, retention controls, and the proposed damage-aware policy at equal memory bytes. Reuse the existing registry and parity references. | An intervention changes the predicted mechanism and the held-out behavior in the predicted direction. | Benefit is explained by added state, different vocabulary, a tuned LR, or a selective cache alone. |
| 4. Compare with actual prior art | Prioritize the closest rivals: GDN-2, HAM-style routing and an online-regression/retention baseline. Include dense attention, SWA, and a strong periodic hybrid. | The proposal improves a prespecified recall/CE/memory frontier at matched resources. | The nearest existing method explains or dominates the result. |
| 5. Scale the surviving idea | At least three sizes, two datasets, multiple context lengths, multiple training-token budgets, equal tuning budgets, and independent confirmation seeds. Use fixed held-out evaluation. | The mechanism survives beyond the calibrated synthetic task and one short LM recipe. | It only changes early convergence or one operating point, without an interesting scoped use case. |
| 6. Establish usable execution | Measure training, prefill, decode, recurrent state/KV bytes, routing overhead, and quality on the same configurations. Compare both fixed tokens and fixed wall-clock/FLOPs. | The gain survives actual execution on a competitive implementation. | Sparse bookkeeping, serial state dependence, or dispatch erases the apparent advantage. |

Suggested initial decision margins are **proposals to preregister**, not thresholds retrospectively applied to current runs: require a practically useful recall gain, such as ten percentage points at the same state budget, while constraining CE regression to at most 0.02 nats on the selected LM validation; alternatively, require materially lower state bytes at a fixed recall target. Choose the exact margins from the intended use case and estimate seed counts from a pilot. Use independent confirmation after selection, rather than counting exploratory cells as confirmatory evidence.

A small first mechanism screen could compare six operators over four independently defined stress conditions and five exploration seeds: 120 runs. That count is a proposed design, not a cost estimate or authorization to spend. Expand the recall seed count for finalists because outcomes are bimodal. Set budget and training termination from calibrated baselines; do not repeat the earlier assumption that 3000 steps means the same thing for every architecture.

The 50M-token boards train roughly 124M-parameter attention models on only about 0.40 tokens per parameter. This does not make their measurements invalid, but it makes them evidence about a very short training regime. A field-scale architecture claim needs longer learning curves and independently tuned horizons, not only more seeds at 50M. The existing width ladder should be kept as a useful sign check, with its 10M-to-50M LR-transfer limitation explicit.

**Research allocation I recommend, as judgment rather than a measured optimum:** spend most new experiment effort on the memory diagnostic and smallest operator; a smaller share on the anchored-depth factorial; keep the BINN temporal branch exploratory until its cell artifacts and mechanism tests are available. Kernel work should unblock a selected experiment or validate its real cost. Routine extra ablations of already-understood rankings should have a lower priority.

The result worth pursuing is a simple operation with a causal account, a strong nearest-neighbor comparison, and a reproducible scaling advantage. Your current experiments provide several useful constraints for that search. They do not yet determine the winning architecture.

**Delivered artifacts**

- [Full per-suite tables](evidence/evidence-tables.md)
- [Recomputed values, seed deltas, configurations and source lines](evidence/recomputed-evidence.json)
- [Artifact paths and SHA-256 hashes](evidence/artifact-manifest.json)
- [Read-only recomputation script](evidence/recompute.py)
- [MoE gradient probe](evidence/router_gradient_probe.py) and [measured output](evidence/router-gradient-results.json)
- [GDN rule probe](evidence/gdn_rule_probe.py) and [measured output](evidence/gdn-rule-results.json)
- [Checks, coverage and missing-data disclosure](evidence/verification-notes.txt)
