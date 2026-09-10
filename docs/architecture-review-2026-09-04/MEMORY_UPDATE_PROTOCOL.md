# Binding, update direction, and retention: experimental protocol

**Status: proposed research protocol with checked mathematical special cases.**
This is the final synthesis of the September 4 discussion, superseding scalar
retention gating as the leading architecture. No new memory layer has been trained
or shown to improve language modeling. See the [current assessment](README.md) for
the evidence base and [historical inputs](README.md#reading-order-and-authority)
for the proposals and corrections that led here.

**Execution revision, September 8:** use the [second audit plan](SECOND_AUDIT_PLAN_2026-09-08.md)
for experiment ordering, task semantics, state-intervention controls, and decision
rules. In particular, complete-state restoration is a sanity control, and the
current teacher-forced MQAR task does not implement query-only evaluation.
This document remains the mathematical reference.

## Question and decomposition

Determine whether a failure arises because the model never constructs the right
association, chooses a destructive update, cannot represent the demanded answers
under the stated constraints, or cannot retrieve what was retained.

The experimental decomposition is:

1. Construct an association from the causal input.
2. Choose a candidate memory-update direction.
3. Evaluate retained-association errors and the update budget.
4. Commit, redirect, retire an obsolete constraint, store an exception, or decline.

These are separable operations. The scalar calculation below implements part of
step 3; it does not choose a good direction. A successful composition is not
automatically a novel architecture.

Use a value-by-key matrix memory $M\in\mathbb R^{d_v\times d_k}$, incoming key
$k$, target $v$, and residual $e=v-Mk$. Freeze feature maps in the first
mechanism test. State precision, normalization, byte budgets, and information
available to every method are part of the experimental contract.

## Diagnose binding and representability first

Compare ordinary token processing with an oracle that supplies the intended
key–value association to the same memory. Match dimensions, update rule, retrieval
objective, and the time at which information becomes available. Supplying future
values earlier, changing key geometry, or changing the effective token budget
would confound the binding intervention. Oracle rescue points toward association
construction; it does not by itself identify which encoder operation to change.

Use a compressibility family within each episode:

$$
v_i=A_\star k_i+\sigma\epsilon_i.
$$

Control total value variance as $\sigma$ changes. At zero noise the associations
share a compact linear rule; increasing independent components demands individual
memorization. Vary pair count, key geometry, distractor distance, overwrite rate,
and memory bytes independently. Test recall of observed noisy associations
separately from generalization to new keys: an unseen key's independent noise is
not inferable from the shared rule.

For fixed keys $K=[k_1,\ldots,k_n]$ and targets $V=[v_1,\ldots,v_n]$, the
minimum-norm least-squares solution and residual are

$$
M^\star=VK^\dagger,\qquad
\min_M\|MK-V\|_F^2=\|V(I-K^\dagger K)\|_F^2.
$$

This ordinary least-squares oracle is an offline diagnostic for exactly that
linear readout and squared loss. It is not a ceiling for categorical recall, a
multilayer model, learned keys, or a nonlinear decoder. A large online–oracle gap
leaves update dynamics, optimization, and forgetting as candidates. A small gap
with large residual identifies a limitation of the specified fixed-key function
class. The offline oracle is not a resource-matched deployable baseline.

## Prediction drift versus actual harm

For a retained query/target $(q_i,u_i)$, define
$r_i=Mq_i-u_i$ and $z_i=Dq_i$ for proposed update $D$. Then

$$
\|r_i+z_i\|^2-\|r_i\|^2=2r_i^\top z_i+\|z_i\|^2.
$$

The earlier score $\sum_i w_i\|Dq_i\|^2$ measures prediction movement. It ignores
whether that movement repairs an error. Keep drift-based control as a baseline,
and deliberately include repairable witnesses when testing actual-error control.

## Scalar retention check

Assign each active witness an absolute ceiling $C_i$, with current feasibility
$\|r_i\|^2\le C_i$. For $M^+=M+\lambda D$, define

$$
a_i=\|z_i\|^2,\quad b_i=r_i^\top z_i,\quad
s_i=C_i-\|r_i\|^2\ge0.
$$

For $a_i>0$, the largest feasible step for that witness is

$$
\lambda_i^{\max}=\frac{-b_i+\sqrt{b_i^2+a_i s_i}}{a_i},\qquad
\lambda=\min\{1,\min_i\lambda_i^{\max}\}.
$$

If $a_i=0$, the write does not change that witness and imposes no restriction
provided it is currently feasible. In exact arithmetic this enforces the declared
ceilings on the active witnesses only. It does not certify all previous facts,
latent-to-token-loss transfer, or factual correctness of the complete model.

For numerical evaluation with $b_i\ge0$, the equivalent expression
$s_i/(\sqrt{b_i^2+a_i s_i}+b_i)$ avoids subtractive cancellation when the
denominator is positive. Handle the all-zero case explicitly. Fail visibly on
infeasible inputs or nonfinite values; floating-point tolerance is not an unlimited
budget. Precision and conditioning must be reported.

Set $C_i=L_i^{\mathrm{reference}}+\varepsilon_i$ at admission. Do not permit
“another epsilon” relative to the current error at every write: that can accumulate
unbounded forgetting. Admission, expiry, target revision, and ceiling reallocation
are separately recorded causal actions. Legitimate overwrites retire or revise an
obsolete target. Changing feature maps requires a defined migration of stored keys,
values, and constraints.

If the candidate includes decay or erasure, use
$D=M^{\mathrm{candidate}}-M$. Applying additional unchecked decay afterwards
invalidates the guarantee. The candidate proposal itself must have sensible new-item
learning behavior; maximizing feasible step size is not universally the same as
minimizing incoming error, especially for unnormalized keys.

## Why blocked writes do not establish capacity exhaustion

A scalar rule searches only a segment in one direction. In a two-coordinate
memory, preserving $(1,0)\mapsto0$ exactly blocks every positive ordinary delta
step toward $(1,1)/\sqrt2\mapsto1$ from $M=(0,0)$. Yet
$M=(0,\sqrt2)$ represents both facts exactly at the same state size. An update
of the original delta proposal's norm can also make useful progress by changing
only the second coordinate. A positive ceiling of 0.01 still limits the scalar
step to about 0.1414 while an exact joint solution exists.

More generally, let $Q$ contain witness keys and $U$ be an orthonormal basis
for their span. Define

$$
p=(I-UU^\top)k.
$$

When $p\ne0$,

$$
D_\perp=\frac{e p^\top}{\|p\|^2},\qquad
D_\perp Q=0,\qquad D_\perp k=e,\qquad
\|D_\perp\|_F=\frac{\|e\|}{\|p\|}.
$$

Thus a redirected write can preserve current witness outputs and fit the new
association. Small $\|p\|$ can require a large norm and amplify numerical
sensitivity; preserving witness errors can also prevent beneficial transfer.

| Observation | Permitted interpretation |
|---|---|
| A blocked scalar step has a modest successful alternative | Update-direction failure at that state. |
| An exact alternative needs excessive norm or precision | Conditioning or practical-budget limitation. |
| Certified optimum cannot meet the old/new requirements | Incompatibility under the specified keys, targets, errors, update constraints, and budget. |

None of these statements alone establishes the general capacity of an entire
language model.

### Status of the reported 124/200 trajectory

The supplied memos report a 6×12 memory, eight fixed rank-eight witnesses, 124/200
nearly blocked scalar writes, and witness-preserving alternatives for all 124.
They further report 25.2% minimum / 79.9% median incoming-error reduction under the
original full proposal's Frobenius norm cap, with exact-fit norm ratios of about
1.81 median / 7.39 maximum. **These trajectory numbers are report-backed here.**
The pasted attachment contains no usable audit-bundle link or executable original
trajectory, so they were not independently replayed in this checkout.

The exact-fit existence is geometrically consistent with four unprotected
directions and nonzero projected incoming keys. It does not show coexistence of all
200 facts, protection of unmonitored queries, or competitive execution. The
independent checks bundled here use separately generated fixtures; they must not
be called reproductions of that trajectory.

## Constrained-update oracle

At a fixed state, define

$$
E_{\mathrm{all}}(B)=\min_X\|(M+X)k-v\|^2
$$

subject to

$$
\|(M+X)q_i-u_i\|^2\le C_i\quad\forall i,\qquad \|X\|_F\le B.
$$

For fixed keys/targets this is a convex constrained least-squares problem. Current
feasibility makes $X=0$ admissible; feasibility of the old constraints is distinct
from the ability to fit the new target. Use this offline on small problems.

Define $E_{\mathrm{ray}}(B)$ by restricting $X=\lambda D$, $\lambda\ge0$,
under the same constraints. Then

$$
E_{\mathrm{ray}}(B)-E_{\mathrm{all}}(B)\ge0
$$

measures the error attributable to the direction restriction at that state and
budget. If evaluating the original scalar gate, also impose $\lambda\le1$
and call it the segment-restricted oracle. Ray and segment coincide when the norm
budget already imposes that cap; do not silently interchange them.

A feasible solver output gives an upper bound on minimum error. A conclusion that
no better permitted update exists needs an optimality certificate or sufficiently
tight lower bound. Report solver tolerances, primal feasibility, and objective
bounds. Nonconvergence is not incompatibility. Run the oracle with the actual
witnesses and, separately, a fuller set of still-valid past associations. The latter
is diagnostic privileged information, not inference-time access for the controller.

### Closed form for exact output preservation

For the special constraints $XQ=0$, $\|X\|_F\le B$, the optimum is

$$
\boxed{E_{\mathrm{preserve}}(B)=
\left[\max(\|e\|-B\|p\|,0)\right]^2.}
$$

Proof: $Xk=Xp$ and $\|Xp\|\le B\|p\|$, so the reverse triangle inequality
gives this lower bound. A rank-one update along $ep^\top$, scaled to fit the
new target or saturate the norm budget, attains it. For nonzero $e,p$, its norm
is $\min(B,\|e\|/\|p\|)$. If $p=0$, the error remains $\|e\|^2$;
if $e=0$, no update is needed. Zero budget also leaves the incoming error unchanged.

This is the full oracle when the retention constraints enforce exact output
preservation, for example perfectly fitted witnesses with zero ceilings. With
nonzero error allowances it is a stricter comparator, not the general
$E_{\mathrm{all}}$. The norm-capped projected write is therefore optimal for
this special case; a general convex solver is unnecessary there.

## Comparison design

### Shared states

At selected states on a common reference trajectory compare:

| Candidate | Main question |
|---|---|
| Ordinary delta | What does the unconstrained proposal learn and damage? |
| Scalar-limited delta | Does shortening that direction help? |
| Norm-capped output-preserving projection | Does steering recover achievable learning? |
| Preconditioning, with/without the same finite-error check | Can cheaper geometry correction recover the gain, and does the check add value? |
| Loss-gradient projection | Does first-order protection differ from output or finite-error protection? |
| Full constrained oracle | What attainable improvement remains? |

Gradient projection and output projection are not interchangeable. For a perfectly
fitted squared-error witness, $\nabla_M L_i=0$, yet a finite update can produce
$L_i(M+X)=\|Xq_i\|^2>0$. Conversely, $XQ=0$ freezes predictions that may
already be wrong. Include deliberately repairable witnesses.

Apply common orthogonal rotations to fixed-key fixtures to test coordinate
dependence of diagonal methods. Rotate queries and the memory representation
consistently; preserve norms and pairwise geometry. This is a fixed-feature
diagnostic, not proof about an end-to-end model's ability to learn a basis.

### Independent trajectories and storage controls

Let each operator subsequently run its own trajectory. Compare a fixed protected
set with evolving still-valid associations. Measure acquisition of incoming facts,
protected and unprotected retention, repair, overwrites, admission/eviction losses,
query-order effects, and conditioning as the witness span grows.

Mandatory controls include identical witness storage exposed as a direct retrieval
cache; drift-based versus actual-error checks; fixed ceilings versus reset allowances;
and shuffled or matched-average commit strengths to separate selective decisions
from merely writing less. Test unseen/noisy queries and held-out associations.

Information-matched comparisons isolate mechanisms. Resource-matched comparisons
also charge state precision, witness keys/targets, projection bases, Gram summaries,
cached predictions, working memory, parameters, and execution. Neither comparison
substitutes for the other. Controllers must not receive future queries or answers.

### Query processing, binding, and secondary branches

For immutable facts, branch from an identical memory snapshot and compare ordinary
queries, disabled persistent writes, and read-only recurrent refinement
$h^{(r+1)}=F_\theta(h^{(r)},M)$. Retain temporary computation so disabling long-term
writes does not simply remove the model's working state. Permute independent query
order and test later retrieval. Beneficial query-time writes would falsify a blanket
read-only rule.

For BINN, independently vary cue–value delay, overlapping entities, distractors,
order, and synchrony before building a temporal memory. Compare oracle and learned
binding with the same downstream memory, and test count/order/coincidence labels,
bin widths, time dilation/translation/reversal, and a second dataset.

For the LM crossing, combine copy-loss logging with interventions on copying demand
and candidate circuitry, matched control disruptions, and held-out recipe prediction.
Temporal alignment alone is correlation. Keep learning-rate phase, steps, and tokens
separate. A token-and-tuning ladder follows the small mechanism study.

## Execution and precision

State-dependent direction/check policies do not automatically inherit GDN's WY or
parallel-scan implementation. Test the exact causal sequential operator first, then
separately identify chunk-boundary or learned approximations. A boundary decision
does not enforce per-token ceilings inside an already processed chunk. Approximate
policies lose the exact guarantee unless a separate check enforces it.

Report training, prefill, decode, synchronization, routing and witness maintenance,
and peak/resident state bytes. A scalar root formula can be cheap while evaluating
all witnesses is expensive. Measure quality and cost for the same configuration.

## Prior art and novelty boundary

These primary sources were checked during the discussion; performance claims were
not independently reproduced. This is a targeted comparison, not proof of novelty.

| Work | Comparator role |
|---|---|
| [Gradient Episodic Memory](https://arxiv.org/abs/1706.08840) | Constrained learning and beneficial backward transfer; first-order loss constraints differ from exact finite-error bounds. |
| [Orthogonal Gradient Descent](https://proceedings.mlr.press/v108/farajtabar20a.html) | Output-gradient subspaces; explicitly distinguishes output gradients from vanishing loss gradients. |
| [Scaled Gradient Projection](https://ojs.aaai.org/index.php/AAAI/article/view/26157) | Relaxes restrictive orthogonal updates with scaled steps in important subspaces. |
| [Preconditioned DeltaNet](https://arxiv.org/abs/2604.21100) | Curvature-aware writes and efficient diagonal approximations; not an arbitrary witness-preserving projector. |
| [Memory by Design](https://arxiv.org/abs/2605.31163) | Bayesian memory uncertainty and confidence-dependent updates. |
| [HOLA](https://arxiv.org/abs/2607.02303) | Bounded exact KV cache selected by committed residual magnitude. |
| [Hybrid Associative Memories](https://arxiv.org/abs/2603.22325) | Compressed state plus explicit associations, surprise and learned routing. |
| [Sparse Delta Memory](https://arxiv.org/abs/2607.07386) | Larger sparsely accessed state; distinguish capacity from update-controller benefit. |
| [Raven](https://arxiv.org/abs/2607.25357) | Input-dependent sparse slot updates and retention. |
| [Query-derived Erase Direction](https://arxiv.org/abs/2608.13668) | Query/key mismatch and an additional erase direction. |

See the [broader review](EVIDENCE_REVIEW.md) for GDN/GDN-2, Longhorn, Titans,
MIRAS/ATLAS, value residual, recurrent depth, and byte/diffusion precedents. Temporal
alignment and previous-token shifts are binding comparators, not presumed novelty.

## Advance conditions

1. The diagnostic identifies a reproducible failure after comparator and tuning
   problems are controlled.
2. A cheap candidate recovers meaningful oracle headroom on new fixtures and its
   own trajectories, without hiding rejection of new facts or unprotected damage.
3. The gain survives equal-storage retrieval controls and the closest prior methods.
4. It survives longer independently tuned horizons, multiple scales/datasets,
   capability-specific metrics, and independent confirmation seeds.
5. The gain remains after actual execution, memory traffic, and precision costs.

Preregister practical margins and estimate seed counts from a pilot. A proposed
ten-point recall gain at equal state bytes or a 0.02 aggregate-CE bound is not a
retrospective pass criterion; rare-capability loss and noninferiority intervals must
also be specified. Mathematical feasibility or a favorable protected-set result
alone does not satisfy these conditions.
