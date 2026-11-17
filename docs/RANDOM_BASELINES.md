# Generic comparisons, random controls and transfer

The reward measures efficient agreement with a specified network on a specified input distribution.
A randomly generated network is still a deterministic computation once its weights are fixed, and
the researcher can inspect those weights. Architecture, algebraic identities, numerical degeneracies
and output imbalance can all make such a computation compressible. High fidelity on a random network
is therefore compatible with a correct evaluation.

To study structure associated with learning, keep three comparisons distinct.

## 1. A supplied generic program on the target network

Freeze a comparison program and evaluate it using the target checkpoint's weights. Match the candidate's
assigned input scope, samples, reward, prices and setup workload. This asks whether a candidate improves
on that known way of computing the same network.

For a candidate P, a comparison B and target weights W, the within-scope gap is

```math
\Delta_S = R_S(P;W)-R_S(B;W).
```

Report coverage, fidelity and cost alongside the gap. A positive gap is improvement over that supplied
program under the declared tradeoff; it does not prove optimality or measure a quantity of learned
structure. Even a program valid for arbitrary weights may inspect particular weight values and exploit
special cases when it executes.

The original baseline procedure searches on a matched random counterpart with an additional request
for validity across arbitrary weights. It validates candidates on other checkpoints, freezes one, and
then applies that program to target weights. This is chiefly a method for finding a reusable comparator.
Finite checkpoint tests provide empirical support, not a proof for every possible weight assignment.
The baseline's own claimed region and coverage must remain visible.

In multi-agent mode, `orchestration.baseline_entry` imports a frozen comparison. The controller remeasures
it separately on each investigated scope. Importing JSON alone does not establish its provenance or
arbitrary-weight validity. The comparison gap is reported; the agents still optimize the ordinary
reward, without subtracting that gap automatically.

## 2. Independent searches on trained and random checkpoints

This asks whether the same research procedure discovers different compression or regional structure
after training. Run independent searches on the target checkpoint and matched random checkpoints.
Keep the following fixed:

- Architecture, sizes, numerical representation, output definition and input distribution.
- Research instructions, region-selection policy, reward, prices and setup-workload policy.
- Total API allowance, tool calls, submissions, turns, time and entry-language limits.
- Development-sample and final-evaluation procedures, including whether a comparator is supplied.

Use the same ordinary discovery objective in both arms. Do not require only the random arm to find an
interpretation exact for all weights: that is a different and harder task. Keep each arm blind to the
target's training status and to the other arm's discoveries. A common frozen comparator can be supplied
to both arms if that assistance is part of the declared design.

Freeze each run's selected programs before evaluating fresh inputs or held-out checkpoints. Compare
those final outcomes over multiple checkpoint seeds and, where practical, repeated agent searches.
Different observed results can reflect both available structure and the search procedure's ability to
find it. A single winning run is not a reliable estimate of either.

Random checkpoints need not be hard to describe. They may produce nearly constant outputs or other
simple behavior. Such an outcome should be reported as part of the control distribution. Random controls
must be specified by an initialization or randomization procedure; "random" is not a universal null
model independent of architecture and scale.

The existing automatic baseline runner does not implement this matched multi-agent experiment. In
particular, its extra exactness instruction and its run-matching rules should not be reused as evidence
that search objectives and total spending were matched. An automatic matched-search runner and aggregate
analysis remain separate future work.

## 3. Transfer of one frozen interpretation

After a discovery run, evaluate the same fixed program on fresh inputs, other trained checkpoints and
matched random checkpoints. No agent search or program revision occurs during transfer. Weight-derived
preprocessing is rebuilt from the destination weights and charged by the same policy.

This asks whether the particular description transfers. It does not determine the best description
another search could find for the destination checkpoint. A general algebraic rewrite may transfer to
both trained and random checkpoints. A parameter-specific rule may fail on random checkpoints and
transfer to other trained checkpoints. These are useful observations, but neither alone proves a
causal account of learning or identifies the internal mechanism.

The current multi-agent audits cover fresh inputs and a configured second checkpoint in the same model
adapter. Random counterparts usually live in separate adapter folders. Cross-adapter transfer is not
automatically scheduled, and the final synthesis must not claim it occurred unless its evidence is
actually supplied. A separately specified destination model/checkpoint interface and aggregation would
be needed for an automatic cross-adapter control audit.

## Regional discoveries need comparable populations

A region selected after inspecting a checkpoint is itself a discovery. There are two useful designs:

1. **Common-domain evaluation.** Specify input-defined regions in advance, or discover them on separate
   checkpoints, then compare candidates on those same regions in both arms.
2. **Independent regional discovery.** Let each arm discover its own regions. Compare globally evaluated
   programs on the common original distribution and describe the regional portfolios separately.

The highest conditional reward on one discovered region should not be subtracted from the highest on
an unrelated region as though both described the same task. A tiny region can have high conditional
reward. Always retain outer probability, conditional coverage, fidelity, cost, support cardinality where
known, and the declared setup workload.

For a fixed-region comparison, the candidate and comparator must use the same workload. If a scope's
cardinality varies across checkpoints, or is unknown, explicitly fixing `verifier.workload_size` avoids
attributing a changed setup denominator to a changed interpretation.

A predicate may read weights. In the current implementation it is rebound to each checkpoint's weights,
so the same predicate program can select a different set of inputs after transfer. The receipt records
this binding. Input-only predicates are the simplest supported way to preserve exactly the same input
population. Freezing the predicate's discovery weights while changing the predictor's weights would
require an additional, explicit binding policy.

## Reading the final report

The final lead should identify the recorded global winner and explain regional findings separately.
For each material claim, ask:

- Was the candidate compared with a supplied program on this same checkpoint and scope?
- Was its fixed interpretation checked on another checkpoint, and which comparisons are absent?
- Was an independent matched random-control search actually performed?
- Is the explanation supported by measurements, a mathematical argument, or an independently checked
  certificate?
- Does the claimed regularity depend on parameters, on the selected input region, or on the architecture?

A scientific report should keep successful generic compression and evidence for parameter-specific
structure visible as separate findings. The multi-agent search can preserve both; the scalar reward
does not establish the distinction by itself.
