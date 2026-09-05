# Historical input: 03-direction-revision.md

Supplied in the September 4, 2026 architecture discussion. Preserved verbatim below; this is not the current verdict. **Direction-selection revision; its reported trajectory audit is not independently reproduced here.** See the [current assessment](../README.md) and [memory protocol](../MEMORY_UPDATE_PROTOCOL.md). Embedded labels such as “verified” belong to the supplied memo. Missing attachment glyphs are not working artifact links.

---

I agree with this critique. It identifies a substantive error in my interpretation—not in the scalar bound. A blocked update can mean “this direction is unsuitable,” not “the memory cannot accommodate the information.”

I would withdraw the capacity interpretation of the 124/200 blocked writes and stop treating scalar gating as the leading architecture candidate. It remains a potentially useful retention check applied after choosing an update direction.

More importantly, I checked the actual prototype. The critique holds for all 124 blocked writes in that trajectory—not just for the two-dimensional counterexample.

1. I reproduced the probe—and tested the missing comparison

The prototype files are available in this conversation, so that part no longer needs to remain unresolved. I reran the original script; its numerical outputs matched the supplied results, including the 124 nearly blocked updates. The script uses a 6\times12 memory and eight fixed witness keys; incoming associations are not added to the protected set.  

I then replayed the same scalar-controlled trajectory. Before each update, I evaluated an alternative direction that preserved the current predictions at all eight witnesses. These alternatives were evaluated counterfactually; I did not commit them or change the subsequent trajectory.

Check on the original trajectory	Result
Nearly blocked scalar writes	124 of 200
Blocked writes with an alternative that exactly fitted the incoming association while preserving witness predictions, within numerical precision	124 of 124
Blocked writes where a redirected update still improved the incoming association when capped to the original full delta proposal’s norm	124 of 124
Incoming squared-error reduction under that norm cap	25.2% minimum; 79.9% median

Here, “norm cap” means the Frobenius norm of the original, unscaled delta proposal, not the almost-zero committed scalar update.

The exact-fitting projected updates sometimes required larger norms: the median was approximately 1.81 times the original proposal’s norm, and the maximum was 7.39 times. That is why the norm-capped comparison matters: steering still helped without increasing that norm budget.

The correct interpretation is:

Every one of those 124 blocks was avoidable with respect to the probe’s witness constraints. None establishes an incompatibility between the incoming association and those eight witnesses in the unrestricted linear memory.

There are important limits. This does not establish that all 200 incoming associations can coexist, that unprotected associations remain intact, or that projection is computationally competitive. It establishes a direction-selection failure in this specific diagnostic.

The reproducible audit bundle⁠￼ contains the original files, the additional audit, and execution instructions. The per-write results⁠￼ record both exact-fitting and norm-capped alternatives.

2. The counterexample exposes a general geometric distinction

The scalar controller searches only a line segment:

M^{+}=M+\lambda D,\qquad 0\le\lambda\le1.

It can select a point along the proposed direction. It cannot discover another direction.

To see the alternative generally, let the witness keys be the columns of Q, and let U be an orthonormal basis for their span. For an incoming key k, define

p=(I-UU^\top)k,
\qquad
e=v-Mk.

When p\neq0, the update

\boxed{
D_{\perp}=\frac{e\,p^\top}{\|p\|^2}
}

satisfies

D_{\perp}Q=0,
\qquad
D_{\perp}k=e.

It therefore preserves the current witness predictions and fits the new association exactly.

In the supplied probe, the eight witness keys have rank eight in a twelve-dimensional key space. That leaves four directions unconstrained by exact witness preservation. Every incoming key in this trajectory had a nonzero component in that complement.

This is not a novelty claim. Protecting previous outputs by changing update directions has a clear antecedent in Orthogonal Gradient Descent; the linear-memory setting simply lets us state the preservation property exactly.  

But projection is not automatically the winner either:

\|D_{\perp}\|_F=\frac{\|e\|}{\|p\|}.

As \|p\| becomes small, exact fitting requires a large update. That can be unacceptable under a norm budget, finite precision, or evaluation on unprotected queries.

So I would distinguish three situations:

Situation	What it actually means
A scalar step is blocked, but a modest redirected step succeeds	Update-direction failure
A redirected exact fit exists, but requires excessive norm or numerical sensitivity	Conditioning or practical-budget limitation
No permitted update meets the old and new requirements	Incompatibility under the specified representation and constraints

Even the last row is narrower than “memory capacity is exhausted.” The fixed keys, permitted errors, overwrite semantics, and update budget all matter.

3. Add a constrained-update oracle—not just a least-squares oracle

The least-squares oracle remains useful, but it answers a different question: how well can a linear map fit a collection of fixed associations in aggregate?

The missing diagnostic is:

From this exact memory state, how well could any update learn the new association while respecting the same retention and update-size constraints?

For small diagnostic problems, solve

E_{\mathrm{all}}(B)=
\min_X
\left\|(M+X)k-v\right\|^2

subject to

\left\|(M+X)q_i-u_i\right\|^2\le C_i
\quad\text{for every witness }i,

and

\|X\|_F\le B.

With fixed keys and targets, this is a convex constrained least-squares problem. Use it offline as an instrument, not as a proposed production layer.

Then compute the same optimum restricted to the proposed direction,

X=\lambda D,\qquad \lambda\ge0,

with the same constraints and norm budget. Call that E_{\mathrm{ray}}(B).

The difference

\boxed{
E_{\mathrm{ray}}(B)-E_{\mathrm{all}}(B)
}

measures the loss caused by restricting the update direction at that budget.

This gives a much stronger diagnostic than counting blocked writes:

A large gap: the proposed direction leaves achievable learning unused.

Both errors small: the simple proposal already accesses a good solution; elaborate steering may be unnecessary.

Both errors large: investigate the retention constraints, key representation, norm budget, or genuinely incompatible targets.

A solver failing to converge would not establish incompatibility. That conclusion needs a reliable solution or a suitable bound.

Also, run this oracle against two sets separately: the controller’s actual witnesses, and a fuller diagnostic set of still-valid past associations. Otherwise you might successfully steer around the witnesses while destroying everything they fail to represent.

4. I would revise the comparison into two separate experiments

The proposed “magnitude versus direction” comparison is mandatory. I would make it more explicit.

First: compare candidate updates from identical states

At selected states on a shared reference trajectory, evaluate:

Candidate	Question
Ordinary delta write	What does unconstrained writing do?
Scalar-limited delta write	Does shortening the ordinary direction help?
Projected write, with a norm cap	Can steering preserve the witnesses while learning?
Preconditioned write, with and without the same retention check	Does a cheaper geometry correction recover the benefit?
Full constrained-update oracle	How much headroom remains under the declared constraints?

Preconditioned DeltaNet is directly relevant, but its efficient construction uses a diagonal approximation to key-Gram information. That is not equivalent to an arbitrary witness-preserving projection, so the two should not be treated as interchangeable controls.  

I would include randomly rotated versions of the same fixed-key problems. Applying a common orthogonal rotation preserves pairwise key geometry and linear representability, while changing coordinate alignment. This tests whether an apparent advantage depends on unusually favorable axes for a diagonal method.

Second: let each operator run its own trajectory

The shared-state comparison isolates the quality of a proposed edit. It does not establish the quality of the resulting memory over time.

In separate rollouts, measure whether successive writes accumulate damage, whether witnesses become unrepresentative, and whether early steering creates difficult later states.

Use both:

A fixed protected set, to isolate update geometry.

An evolving set of still-valid associations, to test actual retention, admission, overwrites, and eviction.

The second condition is crucial. The original probe protects eight initial associations forever; it does not test retaining an expanding stream of new facts.

Finally, separate information-matched comparisons from resource-matched comparisons. Giving all methods the same witness data isolates mechanism. A practical comparison must additionally charge for projection bases, Gram summaries, cached predictions, working memory, and execution. Those are different experiments, and neither should silently stand in for the other.

5. Two further refinements would prevent misleading conclusions

A. Gradient projection is not necessarily finite-error protection

The critique correctly cites Gradient Episodic Memory, but there is an important distinction in how to implement that comparator.

GEM uses gradient inner-product constraints motivated by a local linear approximation to loss. That is not the same as enforcing the exact post-update squared-error ceilings used here.  

For a perfectly fitted witness,

L_i(M)=\|Mq_i-u_i\|^2=0,

its loss gradient is zero:

\nabla_M L_i=2(Mq_i-u_i)q_i^\top=0.

A first-order loss-gradient check therefore imposes no useful restriction. Yet a finite update can produce

L_i(M+X)=\|Xq_i\|^2>0.

At an exactly fitted association, harmful drift can be entirely second-order in the update.

Consequently, compare output-preserving projection, gradient-based projection, and exact finite-error checking as distinct mechanisms. Do not assume that a generic “projected-gradient baseline” satisfies the same guarantee.

B. Exact projection can unnecessarily reject beneficial transfer

A projected write satisfying XQ=0 preserves all current witness predictions—including their mistakes.

When a witness is already wrong, a useful update may need to change its prediction. This is where the scalar rule’s exact error-change calculation still has value: it can permit a change that repairs old error.

The right comparison therefore includes deliberately repairable witnesses, not only initially perfect ones. Prior work on Scaled Gradient Projection already explores relaxing restrictive orthogonal updates to improve learning while controlling forgetting, so “mix protected and plastic directions” is also not an unoccupied claim.  

These two observations suggest keeping direction selection and retention checking separate, rather than choosing between projection and scalar control as though they were complete competing architectures.

6. What survives as a research direction

The revised structure would be

\boxed{
\text{construct association}
\;\rightarrow\;
\text{choose update direction}
\;\rightarrow\;
\text{check retained-association errors}
\;\rightarrow\;
\text{commit or use another action}.
}

The scalar calculation belongs in the third step. It does not solve the second.

A small prototype could generate an ordinary direction and a steered direction, apply the same finite-error check to each, and compare their attainable new-item improvement. But I would treat that as an experimental composition—not immediately name it a new architecture.

The practical questions become sharper:

Can an inexpensive direction generator recover much of the constrained oracle’s improvement?

Does checking actual error change help beyond preconditioning alone?

Does either mechanism beat spending those same bytes on directly retrievable associations?

The cache comparison remains essential because HOLA already combines a delta-rule state with a bounded exact KV cache selected using committed residual magnitude. An apparent memory-management gain must survive that close alternative.  

Witness semantics also need to be explicit. Freeze the feature maps in the first mechanism test. For legitimate fact updates, retire or revise the obsolete target rather than penalizing the correct replacement. If representations change while memory persists, specify how stored keys, targets, and constraints are kept consistent.

And yes, retain the “does asking a question damage memory?” experiment. It tests a different issue: whether persistent storage is being unnecessarily modified during temporary computation. Fixing write direction would not, by itself, answer that question.

Bottom line

This critique improves the proposal substantially, and the additional audit strengthens its central objection.

My earlier interpretation should be replaced with:

The prototype demonstrates that a scalar retention check can respect fixed witness-error ceilings. Its blocked-write count demonstrates failure of the chosen update directions under those ceilings—not exhausted memory capacity.

For this particular trajectory, we can now say more: all 124 blocked writes had witness-preserving alternatives, and all still allowed useful learning under the original full proposal’s norm budget.

The next research priority is therefore not a more elaborate gate or a larger overflow cache. It is to measure how much learning is being lost because of update direction, then determine whether that gap can be closed cheaply without harming unmonitored associations.

That is a stronger foundation for an architecture than the scalar commit rule alone.