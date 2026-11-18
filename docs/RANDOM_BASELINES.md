# Generic comparisons, random controls and transfer

The reward measures efficient agreement with a specified network on a specified input distribution.
A randomly generated network is still a deterministic computation once its weights are fixed, and
the researcher can inspect those weights. Architecture, algebraic identities, numerical degeneracies
and output imbalance can all make such a computation compressible. High fidelity on a random network
is therefore compatible with a correct evaluation.

To study structure associated with learning, keep three comparisons distinct.

## 1. A frozen generic program on the target network

Freeze a comparison program and evaluate it using the target checkpoint's weights. Match the candidate's
assigned input scope, samples, reward, prices and setup workload. This asks whether a candidate improves
on that known way of computing the same network.

For a candidate P, a comparison B and target weights W, the within-scope gap is

```math
\Delta_S = R_S(P;W)-R_S(B;W).
```

Report coverage, fidelity and cost alongside the gap. A validated generic-comparison gap requires
the comparator to have full coverage and exact agreement on the actual evaluated target scope,
with matching samples or enumeration, prices and setup workload. If the comparator fails these
checks, retain its raw receipts and any raw arithmetic with an invalid-comparison label; do not
present that difference as established generic efficiency improvement. A valid positive gap is
improvement over this frozen program under the declared tradeoff. It does not prove optimality or
measure a quantity of learned structure. Even a program valid for arbitrary weights may inspect
particular weight values and exploit special cases when it executes.

The framework can ask one agent to find a global generic program, either alone (`--mode baseline`)
or before target research (`--mode both`). The existing `baseline_of` declaration identifies a counterpart with the same
architecture, shapes and input distribution and at least four parameter checkpoints. The agent
receives the additional requirement of validity across arbitrary weights, without being told the
target's task or its own checkpoint's control status. It has no regional baseline workers.

Only formally submitted programs are considered for generated baselines. New runs use validation
policy version 2, in which every eligible submitted program can compete. To pass, a program must have full coverage and
match every evaluated output on all declared control checkpoints and all checkpoints of all
declared compatible trained twins. This includes the discovery checkpoint, a separate control
selection checkpoint and the other parameter controls. A cheap rule that matches the control
networks but fails a required trained checkpoint is rejected before baseline freezing.

Candidates are checked in order of their reward on the control selection checkpoint until one
passes every acceptance gate; there is no old three-candidate cutoff. Trained outputs affect
pass/fail eligibility; trained rewards do not rank programs.
The search agent is not given these trained measurements as research feedback. The baseline is
frozen before trained-model research begins. Finite checkpoint tests provide empirical support,
not a proof for every possible weight assignment or unsampled input. The source, resource allocation,
validation policy version and individual receipts remain on record.

Version 1 records retain their original control-only acceptance policy and any original candidate
limit. Resume does not reinterpret those records as if version 2's trained acceptance checks had
occurred. Legacy imported entries with no parameter validation remain explicitly unvalidated.

The one frozen global program is then remeasured on the target's global domain and every regional
claim, with matching inputs and workload within each comparison. A regional comparison means applying
that same global program to the region; it does not mean another agent optimized a regional baseline.
The gap is reported while research agents continue optimizing the ordinary reward, without subtracting
the gap automatically. A comparator can still fail a later fresh-input or regional check despite
passing parameter validation. The validated gap is withheld for that check rather than overridden
by its earlier validation status.

The three modes use one shared allowance:

- `trained` runs trained-model research without a baseline agent or imported comparison. When
  final synthesis is enabled, its share is reserved and the remaining allowance goes to research.
- `baseline` gives its one global agent the full nominal run allowance, subject to the usual
  session cap. It runs the same control and trained verification gates, then writes a deterministic
  report. There are no trained-model research sessions and no paid lead summary.
- `both` reserves the configured `baseline_fraction`, by default 20 percent, for the global agent.
  With the usual 10 percent final-summary share, a $25 total assigns up to $5 to baseline search,
  $2.50 to synthesis and $17.50 to research. These are allocations within the $25, not extra budgets.

`baseline_submissions: 2` explicitly budgets the global agent's formal submissions. All API
attempts use the parent spending ledger. When mode is omitted, an enabled configuration with
`baseline_agent: true` selects `both`, preserving existing configuration inference.

`orchestration.baseline_entry` imports an existing frozen program and releases its search allocation.
In a new version 2 run the imported program must pass the same full-coverage exactness gates on all
required control and trained checkpoints before use. It remains labeled as supplied, with zero
search sessions. These local checks do not establish its search provenance or universal validity.
Explicit `trained` mode rejects a supplied baseline entry. Old saved modes and validation policies
remain unchanged on resume, and completed baseline phases are reused. Missing required counterpart
models or checkpoints fail preflight before API use. A baseline phase without a validated full-domain
program stops visibly before target research. `MULTI_AGENT.md` describes commands and saved records.

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
must be specified by a weight-sampling or parameter-randomization procedure; "random" is not a universal null
model independent of architecture and scale.

The automatic single-agent global baseline does not implement this matched multi-agent experiment.
Its additional exactness instruction, one-agent search and distinct resource allocation differ from
the target research procedure. Its trained-checkpoint acceptance gates also differ from an entirely
independent random-control research arm. A gain over its frozen program may be a generic optimization that
this baseline search missed. Neither the automatic phase nor the original single-agent baseline
runner establishes matched discovery objectives or total spending across research arms. An automatic
matched-search runner and aggregate analysis remain separate future work.

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

Version 2 baseline validation evaluates the comparator on control and trained parameters; it does not
evaluate the discovered target interpretations on those parameter controls. Its validation receipts therefore
do not fill this transfer experiment's missing evidence.

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

- Was the candidate compared with the frozen global baseline on this same checkpoint and scope?
- Did the comparator have full coverage and exact fidelity on that actual check, with matching inputs, prices and workload?
- Was that baseline generated in the recorded single-agent phase or imported, and what validation passed?
- Does the recorded policy include trained checkpoint acceptance, and were those outputs used only for pass/fail rather than reward ranking?
- Was its fixed interpretation checked on another checkpoint, and which comparisons are absent?
- Was an independent matched random-control search actually performed?
- Is the explanation supported by measurements, a mathematical argument, or an independently checked
  certificate?
- Does the claimed regularity depend on parameters, on the selected input region, or on the architecture?

A scientific report should keep successful generic compression and evidence for parameter-specific
structure visible as separate findings. The multi-agent search can preserve both; the scalar reward
does not establish the distinction by itself.
