# Historical input: 02-retention-memo.md

Supplied in the September 4, 2026 architecture discussion. Preserved verbatim below; this is not the current verdict. **Scalar retention proposal; blocked writes do not establish exhausted memory capacity.** See the [current assessment](../README.md) and [memory protocol](../MEMORY_UPDATE_PROTOCOL.md). Embedded labels such as “verified” belong to the supplied memo. Missing attachment glyphs are not working artifact links.

---

Bharath, there is a real research opportunity here—but I would not build the composite architecture in the second memo yet. It combines several individually plausible improvements without establishing which failure each one fixes.

My strongest recommendation is:

Build a memory operator that controls how much retrieval error a write is allowed to introduce—not one that simply avoids surprising inputs or preserves every previous representation.

That is a refinement of the interference-aware direction, but it leads to a concrete operation, a testable guarantee, and some revealing failure cases.

I would also revise my earlier protected-subspace suggestion: keep exact subspace protection as an analytical baseline, rather than making it the default architecture. Protecting old outputs perfectly can prevent useful learning. The more interesting question is how to permit change without losing the associations that still matter.

I am assessing your pasted records, not claiming to have rerun the local repositories. I checked additional primary literature and ran a small mathematical prototype described below.

1. My assessment of the two memos

The first memo is the better scientific foundation. The second contains useful experiments, but its explanations run ahead of its evidence.

The reported hybrid curves are worth investigating. So are the token ladder, tying/value-residual factorial, and repeated-span copy probe. But I would not accept these stronger conclusions yet:

“The problem is formation speed, not capacity.” Improvement with more training at four pairs does not resolve why a model fails at 64 or 128 pairs. Binding, conditioning, memory interference, representation capacity, and optimization can all contribute—and can dominate at different loads.

“The crossover is induction-head formation.” This is a hypothesis, not an interpretation you can assign from the curve alone. Even the original induction-head work distinguishes causal evidence in small attention-only models from more limited evidence in larger models. Successful associative recall is not a circuit identification.  

“The data selects the full composite architecture.” Results supporting value residuals, hybrid placement, and weight sharing do not establish that their combination with untied embeddings and adaptive halting is optimal. You need interaction tests.

“The corrected MoE board establishes that additional parameters do not help.” The two memos contradict each other here. Under the first memo’s account, completed corrected runs are controls, not a finished multi-expert comparison. That cannot support a general conclusion about useful expert capacity.

There is also a useful mathematical nuance in the GDN discrepancy. For the reported repository transition,

A_t=\alpha_t I-\beta_t k_tk_t^\top,

a unit key and \alpha_t,\beta_t\in[0,1] give

\|A_t\|_2=\max\{\alpha_t,|\alpha_t-\beta_t|\}\le 1.

A negative transition along the key is not automatically an unstable transition. This does not bound the entire trained network’s Jacobian or establish quality, but it prevents “sign flip” from becoming another premature failure explanation. The published GDN rule is nevertheless different, so the comparator ablation remains essential.  

My reading is therefore:

Your evidence motivates a search over binding, memory editing, and retrieval. It does not yet justify selecting one of them as the sole explanation.

2. The prior-art comparison needs another update

Several recent preprints are especially relevant. These are descriptions of their proposed mechanisms—not independently reproduced performance endorsements.

Work	Why it changes the research target
Preconditioned DeltaNet — April 2026	Curvature-aware writes and efficient diagonal approximations are already explicit. A covariance-aware delta update needs this comparison.  
Memory by Design — May 2026	Propagates memory uncertainty through a Bayesian formulation, preserving confident associations and changing writes according to uncertainty. “Know which directions are safe to update” is already a developed direction.  
HOLA — July 2026	Adds a bounded exact KV cache to a delta-rule state and selects entries using committed residual magnitude. A compressed-memory-plus-exceptions design has a very close comparator.  
Sparse Delta Memory and Raven — July 2026	Explore larger sparsely accessed state and selective slot updates. More memory capacity or less broadly destructive writing must be separated from the benefit of a new controller.  
Query-derived Erase Direction — August 2026	Introduces a query-derived correction to GDN-2’s erase direction. Measuring interference only at stored keys may miss what actual queries experience.  

This does not mean the space is exhausted. It means the contribution needs to be sharper than “use memory more intelligently.”

A plausible distinguishing claim would be:

A causal, budgeted write policy that controls measured retrieval degradation, permits beneficial transfer, and improves over surprise routing, uncertainty-aware updates, and extra storage at matched resources.

That is a research claim you can actually try to establish.

3. The first experiment I would add: separate binding from storage capacity

Before inventing another update rule, determine whether the memory receives usable associations.

Start with oracle binding

Compare normal end-to-end processing with a diagnostic condition where the data generator supplies the intended key–value pair directly:

\text{ordinary tokens}
\longrightarrow
\text{learned binding}
\longrightarrow
\text{memory},

versus

\text{correct }(k_t,v_t)
\longrightarrow
\text{same memory}.

Use the same memory dimensions, precision, update rule, and retrieval objective.

If oracle binding largely rescues the model, the next architecture should improve which key is paired with which value. A complicated interference controller would be treating the downstream symptom.

If oracle binding does not rescue it, storage and retrieval deserve more attention.

Temporal alignment is not an unoccupied design space either: recent fast-weight work explicitly studies previous-key/current-value associations and related alignment choices. Your previous-token-shift arm should be treated as a binding comparator, not presumed novelty.  

Then vary compressibility, not just pair count

Here is a more revealing synthetic family than another “4 versus 64 pairs” board:

v_i=A_\star k_i+\sigma \epsilon_i.

Within each episode, A_\star is a shared linear map, while \epsilon_i is an independent value component. Keep total value variance controlled as \sigma changes.

This separates two demands:

At \sigma=0, many associations share a compact rule.

As \sigma increases, more information must be remembered as individual exceptions.

Now independently vary pair count, key geometry, distractor distance, overwrite frequency, and memory bytes.

The same number of pairs can impose radically different storage demands. A model failing on independent random values but succeeding on rule-generated values is telling you something different from a model failing both.

Add a function-class oracle

For a fixed-key, linear-readout diagnostic, collect keys and targets as

K=[k_1,\ldots,k_n],\qquad V=[v_1,\ldots,v_n].

The best unconstrained linear memory for reconstructing those targets is

M^\star=VK^\dagger,

with minimum squared reconstruction error

\min_M\|MK-V\|_F^2
=
\left\|V\left(I-K^\dagger K\right)\right\|_F^2.

This is ordinary least squares, not a new result. Its value here is diagnostic.

When the online memory performs far worse than this oracle, the gap is not explained by that linear function class’s representational limit alone. Update dynamics, optimization, or forgetting remain candidates.

When it approaches the oracle and substantial error remains, modifying the learning rate indefinitely will not solve that particular fixed-key linear reconstruction problem.

Important boundary: this is not an accuracy ceiling for an entire multilayer language model, a learned nonlinear decoder, or categorical recall. Keep the oracle attached to the exact function class and loss it evaluates.

This experiment would give you a much stronger foundation for saying “capacity,” “interference,” or “formation speed.”

4. Architecture candidate: retention-budgeted memory commits

Status: a proposed research operator, not an established novelty or performance result.

The idea is simple:

Propose a memory update, calculate how much of it can be applied without exceeding explicit error budgets on a small set of retained associations, and commit only that fraction.

This differs from both “always write” and “freeze the protected subspace.”

Why prediction drift is not enough

The first memo’s score is

I_t=\sum_i w_i\|\Delta M_t k_i\|^2.

That measures how much old predictions move. It does not generally measure whether they become worse.

For an old association with residual

r_i=Mk_i-v_i,

a proposed change z_i=\Delta M_tk_i changes its squared error by

\boxed{
\|r_i+z_i\|^2-\|r_i\|^2
=
2r_i^\top z_i+\|z_i\|^2.
}

The cross-term distinguishes harmful interference from repair.

For example, two writes can both move an old prediction by +1. If that prediction was correct, its squared error increases by one. If it was wrong by -1, the same movement fixes it completely.

A policy that treats both writes as equally damaging throws away useful information.

The operation

Let D_t denote a proposed change to the memory. Initially use the ordinary delta proposal:

D_t=(v_t-M_{t-1}k_t)k_t^\top.

Maintain a small, budgeted set of witness associations (q_i,u_i). In the first synthetic experiment, these can simply be previously observed keys and values.

For each witness, calculate

r_i=M_{t-1}q_i-u_i,\qquad z_i=D_tq_i.

Assign an absolute squared-error ceiling C_i. We seek the largest \lambda_t\in[0,1] satisfying

\|r_i+\lambda_t z_i\|^2\le C_i
\quad\text{for every active witness}.

Define

a_i=\|z_i\|^2,\qquad
b_i=r_i^\top z_i,\qquad
s_i=C_i-\|r_i\|^2.

Assuming the current state is feasible, s_i\ge0. For a_i>0, the maximum permitted step for witness i is

\lambda_i^{\max}
=
\frac{-b_i+\sqrt{b_i^2+a_i s_i}}{a_i}.

Therefore,

\boxed{
\lambda_t=
\min\left\{1,\min_i\lambda_i^{\max}\right\},
\qquad
M_t=M_{t-1}+\lambda_tD_t.
}

A witness with a_i=0 places no restriction because the proposed write does not change its prediction.

This gives an exact property in exact arithmetic: the committed update respects the declared ceilings on the active witnesses. It does not certify all previous facts.

The same calculation can constrain a more general proposed edit, including decay and erasure, by defining D_t=M_t^{\mathrm{candidate}}-M_{t-1}. Any uncontrolled decay applied outside the check would invalidate the guarantee.

The budgets must not reset after every write

This is easy to get wrong.

A rule permitting “another 0.01 error” relative to the current state at every step can accumulate arbitrarily large forgetting.

Instead, define a ceiling relative to a fixed reference when the witness is admitted:

C_i=L_i^{\mathrm{reference}}+\varepsilon_i.

Subsequent writes consume or replenish the remaining slack under that same ceiling. Replacing, expiring, or revising a witness is a separate, recorded operation.

What I checked

I implemented the scalar-commit calculation and ran 10,000 randomized feasibility and boundary checks. The largest numerical ceiling overshoot was approximately 7.1\times10^{-15} in float64.

That verifies the calculation, not a learned architecture.

The more revealing result came from a small matrix-memory trajectory: with fixed protected associations and fixed ceilings, 124 of 200 proposed writes were nearly blocked.

That is not something to hide. It exposes the central tradeoff:

Perfect retention can become an inability to learn.

The architecture therefore needs an explicit response when the write budget is exhausted: preserve an exception, retire an obsolete witness, accept a measured tradeoff, or decline the write. It cannot get unlimited retention and unlimited plasticity from fixed storage.

The executable reference probe⁠￼ and its recorded results⁠￼ include both the passing checks and the blocked-write result.

Where originality might lie

The broad principle of constraining new learning to protect older examples is established. Gradient Episodic Memory already uses episodic examples and constrained updates to reduce forgetting while allowing beneficial transfer. Bayesian memory and preconditioned delta methods are also essential comparisons.  

Consequently, the claim cannot be “we invented interference-aware updates.”

The potentially distinguishing contribution is the combination of:

An exact, cheap scalar commit for recurrent fast weights; absolute retention budgets; a causal budget-reallocation policy; and a demonstrated advantage after charging the witnesses and storage.

That may turn out to be incremental. It may turn out to be useful. The current evidence does not decide.

5. The experiment that could immediately kill this architecture

There is a serious fairness problem:

The witnesses themselves contain memory.

Storing old queries and target values could improve retrieval even if the write controller does nothing.

A mandatory control is therefore the same backbone with the same witness storage exposed as a direct retrieval cache, but without retention-budgeted commits.

Evaluate protected and unprotected associations separately. A controller that protects only the exact pairs it explicitly stores has not demonstrated better compressed memory.

I would require the following comparisons:

Comparison	What it isolates
Same memory and cache, ordinary writes versus budgeted commits	Whether the commit mechanism contributes beyond storage.
Budgeted commits versus shuffled or matched-average commit strengths	Whether the damage-dependent decision matters, rather than merely writing less.
Actual error-change constraints versus prediction-drift constraints	Whether permitting beneficial transfer is useful.
Fixed ceilings versus repeatedly reset allowances	Whether the policy controls cumulative forgetting.
Current facts versus legitimate overwrites	Whether protection preserves useful knowledge rather than obsolete values.
Witness queries versus unseen/noisy/rephrased queries	Whether protection transfers beyond the monitored addresses.

The final comparison is especially important. Protecting Mk_i does not ensure that a later, differently represented query retrieves the association. Query-derived memory editing is already motivated by this read/write mismatch.  

Do not give the inference-time controller future queries or future answers. An all-history oracle is useful for measuring headroom, but it is not a deployable mechanism.

For an LM, a bound on a layer’s latent reconstruction error is also not a bound on final-token loss or factual correctness. That transfer must be measured.

The hardware problem remains real

Because \lambda_t depends on the evolving state, the exact operator is state-dependent. It does not automatically inherit GDN’s chunkwise execution.

I would test the exact sequential version first, then a separately identified boundary-only variant. A chunk-boundary decision can affect subsequent chunks without looking into their future, but it does not preserve the same per-token guarantee inside the completed chunk.

A learned causal approximation could be another variant. It would lose the exact guarantee unless separately checked.

Do not combine these implementations into one result. A useful sequential mechanism and a fast approximate mechanism are two different claims.

6. Two other experiments I would prioritize over another architecture sweep

A. Does asking a question damage the memory?

This connects your memory and recurrent-depth directions more cleanly than another shared-depth CE board.

Construct a synthetic store of immutable facts. After the facts are encoded, issue multiple independent queries that introduce no new facts.

Branch from the same stored-state snapshot and compare:

* ordinary query processing, including state updates;
* query processing with long-term memory writes disabled;
* multiple read-only refinement passes over the same state.

Use fixed diagnostic query representations initially so that changes in query encoding do not obscure changes in storage. Permute query order and examine which associations become harder to retrieve after each query.

If the task semantics contain no updates, later answers should not depend on whether an unrelated question was asked earlier. A dependence localized to state mutation would motivate separating retrieval-time computation from storage modification.

That is a sharper reason to use anchored recurrent depth:

\text{persistent memory remains fixed},
\qquad
h^{(r+1)}=F_\theta(h^{(r)},M).

Adaptive recursive depth and shared KV memory already have close precedents, so the contribution would be identifying and fixing destructive query-time writes—not merely adding more passes.  

This experiment could also fail usefully: perhaps writes during queries help compute the answer without materially damaging later retrieval. Then strict read-only processing is the wrong constraint.

B. Can you causally explain the learning-curve crossing?

The repeated-span copy-loss probe is worth running, but I would change the success criterion.

“Copy loss falls near the CE crossover” establishes temporal association. It does not establish that copying caused the crossover.

A stronger test varies the need for copying while holding other aspects of the data as constant as practical, then intervenes on the candidate mechanism. Compare targeted disruption with matched control disruption. Check whether the crossover responds in the predicted direction.

Also separate raw token count, optimizer steps, and learning-rate phase. Otherwise both phenomena may simply respond to the same schedule transition.

The valuable result would be a predictor that transfers to a held-out recipe—not a post hoc alignment of two curves.

7. What I would do with the hybrid result

Keep 8+4 and 9+3 as strong reference architectures. Do not turn either into a design law.

The reported 8+4 learning curve could be an interesting recipe-specific result even without a novel operator. But I would avoid “no-regret” until the claim specifies its domain: training-token interval, quality metrics, tuning procedure, memory, hardware, and execution budget.

The reported parameter counts also matter. A ratio sweep changing attention count can change both the operation and the number of parameters. The placement comparison with a fixed attention count is cleaner.

For the next scale experiment, I agree with extending training horizons. But call it a token-and-tuning ladder, not simply a token ladder. Reusing a short-horizon learning rate at a longer horizon leaves a different unresolved comparison.

I would also track capability-specific loss alongside aggregate CE. As a simple accounting example, a four-nat deterioration on a subset comprising 0.5% of evaluated tokens contributes only 0.02 nats to the overall mean:

0.005\times4=0.02.

Thus your proposed aggregate CE margin could coexist with a substantial deterioration on a rare retrieval demand. That is not a reason to abandon CE; it is a reason not to make it the sole guardrail.

On statistics, five paired wins are useful evidence, but not automatically a two-sided sign-test result below 0.05:

2(1/2)^5=0.0625.

A paired magnitude-based analysis can still be informative. However, checkpoint correlation, architecture selection, and cross-suite provenance remain relevant. Confirm the selected learning-curve claim on new seeds rather than treating many checkpoints as independent replications.

8. BINN: investigate temporal binding before claiming a new temporal memory

For BINN, my most useful addition would be a variable-delay binding benchmark that connects naturally to the oracle-binding experiment.

Generate events where a cue identifies an entity and a later event provides its value. Independently vary the cue–value delay, overlapping entities, distractor rate, temporal order, and cross-channel synchrony.

Then compare learned temporal binding against directly supplied correct associations using the same downstream memory.

This separates two possibilities that can otherwise look identical at the output:

The temporal encoder never constructed the right association.

The association was constructed correctly but was subsequently lost.

Next, create matched tasks whose labels depend separately on counts, order, and coincidence. Apply interventions that preserve the intended nuisance quantities, and inspect the resulting intermediate features—not just final accuracy.

A promising result would be a compact representation that preserves the task-relevant temporal relation across changed bin widths and timescales, then transfers to a second dataset.

That would justify an event-memory architecture. The present report-backed shuffle and positional-code effects do not yet determine which representation to build.

9. The research order I would choose

I would organize the work around a small number of decisions, rather than launch all the proposed experiments simultaneously.

First: repair and diagnose. Resolve the GDN comparator, isolate task-trained MoE routing before interpreting specialization, and run oracle binding plus the compressibility/function-class diagnostic.

Second: test the smallest new operation. Compare retention-budgeted commits against ordinary writes, drift-based control, and the same directly readable storage. Include changing facts and unprotected queries from the beginning.

Third: compare against the closest existing methods. Preconditioned/Bayesian memory, a bounded exception cache, and a sparse-state method answer different alternative explanations. A win over an older GDN alone is insufficient.  

Fourth: scale the survivor and measure execution. Use longer horizons, independently tuned recipes, multiple byte budgets, and confirmation seeds. Report training, prefill, decode, and routing/storage overhead separately.

I would keep the anchored-depth and BINN branches active, but make their next experiments diagnostic rather than architectural.

Bottom line

The useful shift is from “preserve memory” to “manage permitted changes in memory.”

Exact preservation is too rigid. Unconditional writing is too destructive. Surprise alone does not distinguish a harmful update from one that repairs several old associations. And no controller can evade the need to discard information when genuinely independent facts exceed its storage budget.

My strongest candidate is therefore retention-budgeted memory commits with explicit replacement or overflow, initially tested without a learned controller. My strongest experiment is oracle binding combined with a controllable-compressibility task and a function-class oracle.

Together, they can answer something the current boards cannot:

Is the model failing because it never formed the association, because its update destroyed an otherwise representable association, or because the chosen memory simply cannot represent all the demanded answers?

A simple operation that wins after those distinctions are controlled would be far more compelling than a larger hybrid assembled from individually successful ingredients.