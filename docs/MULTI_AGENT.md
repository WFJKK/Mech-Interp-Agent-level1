# Team research and global integration

The optional research framework uses the existing reward and entry language. It can run trained-model
research, one global generic baseline agent, or both. A lead decides, each cycle, what each researcher
works on, and uses their checked results to construct a complete program. Code owns baseline
validation, evaluation, the shared budget, the archive and the final lock. When research is enabled,
an optional final tool-free lead session explains the frozen global and regional findings.

The original run mode remains the default. No operation prices or entry-format identifiers change.
The framework is selected by `orchestration.enabled: true` or an explicit `run --mode` option.

## Run the experiment

From the repository root, create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m level1.sandbox info
```

Use a virtual environment inside this repository. The hygiene check inspects the Python path
visible to researchers and rejects paths that disclose withheld information.

The configured backend requires the Claude Code CLI (`claude`) to be installed and signed in.
The API-key option below uses the same CLI with separate authentication. Then start the experiment:

```bash
python -m level1.loop run --config config/experiments/lin-c-regional.yaml --mode both
```

Choose the stages for a new run with `--mode`:

| Mode | Agent sessions | Automatic validation and output |
| --- | --- | --- |
| `trained` | Planning lead, regional researchers, integrating lead, optional final lead synthesis | Trained-model research, frozen global and regional candidates, final audits and report; no baseline search or imported comparison |
| `baseline` | One global baseline agent, or no agent when a frozen entry is supplied | Validate the comparator on all required control and trained checkpoints, freeze it and write a deterministic report; no trained-model research or final lead synthesis |
| `both` | One global baseline agent followed by trained-model research and optional final lead synthesis | Validate and freeze the baseline first, then reuse it for global and regional comparisons |

The choice is saved in `orchestration.mode`. The CLI option also enables orchestration, so it can be
used with an existing model configuration. A trained-model check in `baseline` mode is local verifier
work, not another agent session. Without an explicit mode, existing configuration choices still
apply: an enabled framework with `baseline_agent: true` or a supplied `baseline_entry` selects `both`;
an enabled framework without either selects `trained`. A configuration with orchestration disabled
retains the original single-agent pipeline when no `--mode` is given.

The model-neutral profile is `config/experiments/research.yaml`. The example above selects one model
and its checkpoints but supplies no expected behavior or input-length split. The lead chooses regions
from its investigation. If it makes no assignments, two generic full-domain questions provide a
fallback: search for useful special explanations, and independently examine alternatives and exceptions.
A regional result does not need to improve the global score to be retained.

Regional researchers explicitly seek mechanistic explanations relating behavior to the supplied
network's weights and internal computations. The lead also seeks an overall mechanistic understanding:
shared mechanisms, connections between regional findings and internal reasons for differences in
behavior. These investigations take place within the existing research allowances and submission
checkpoints. Strong interpretations remain eligible when their mechanisms are unclear; mechanistic
understanding adds no acceptance gate, score term or archive-retention criterion. Closing notes
separate supported explanations, hypotheses and unresolved questions. The instructions remain
model-neutral and supply no expected behavior or preferred input partition.

The shared research profile defaults to `both`. Its single global baseline agent searches on
the model's declared parameter-control counterpart, freezes one validated full-domain program,
and reuses that program for comparisons on the target's global and regional domains. There are
no regional baseline agents. `--with-baseline` is also accepted when the saved research configuration
already includes a baseline phase; it does not add another run or change a trained-only mode.

To bill the experiment through the existing API-key backend with one total spending limit:

```bash
python -m level1.loop run --config config/experiments/lin-c-regional.yaml --mode both --auth api_key --budget-usd 60
```

The existing `CPL_ANTHROPIC_KEY` mechanism is used. All roles can use the same API account and model.
Each role gets its own session, prompt, saved programs and logs. Setting `orchestration.parallel` to
`false` runs the researcher sessions sequentially with the same roles and allocations.

A separate single-researcher control evaluates only one declared input domain:

```bash
python -m level1.loop run --config config/experiments/lin-c-length2.yaml
```

This control exhaustively checks its finite uniform domain under `verifier.check: auto`. It does
not infer or receive a training objective. Its explicitly selected domain is an optional diagnostic,
not an assumption in the shared profile or research prompts.

As usual, each execution needs a fresh run ID, or resumes an existing run:

```bash
python -m level1.loop resume --run /absolute/path/to/the/run
python -m level1.loop finish --run /absolute/path/to/the/run
```

Resume and finish honor the saved mode; they do not accept a new `--mode`. To change the stages,
start a new run ID. The normal sandbox checks and trusted-code fingerprint apply. Do not resume an
old scientific run under changed trusted code unless deliberately accepting that change with the existing flag.

## Applying the framework to another model

The controller and prompts do not assume a particular architecture, training objective or special
input length. Researchers receive the definition of the actual model being investigated, its weights,
the declared input distribution and their assigned scopes. They do not receive withheld training
objectives, training-input metadata or the target checkpoint's training status.

An experiment can extend `research.yaml` and supply its own `model`, `network`, `run_id`, optional
`seed_check.network`, and verifier settings. Those are experiment choices; no controller code or
research prompt needs to change when selecting another compatible adapter.

Baseline generation or version 2 validation of a supplied comparison also needs a declared counterpart
with the same architecture, shapes and input distribution. The existing `withheld/<control-model>/task.yaml` declaration
names the target in `baseline_of`; the counterpart supplies at least four parameter checkpoints.
Version 2 also checks every checkpoint of every compatible trained model declared as a twin of
that counterpart. This operator metadata is not disclosed to the agents. If a required counterpart,
checkpoint or compatible declaration is missing, preflight fails before any API call. A frozen
program supplied with `orchestration.baseline_entry` avoids agent search but receives the same local
validation. `--mode trained` does not need a control counterpart.

Model independence is relative to the existing model interface. An adapter supplies `reference(g,
lv1)`, configured weight arrays and a supported input distribution. The current reference returns one
integer output per input, has constant forward operation cost across those inputs, and is expressible
in the entry language. Existing adapters cover several architectures and both discrete and real
inputs. An arbitrary external model or natural-language corpus may need a new adapter, distribution
implementation and cost model. Unknown-cardinality domains need an explicit workload.

## Teams for the recurrent networks

Three configurations run a team on the recurrent networks on real numbers. The lead decides every
task; the configurations only set the team's size, the number of cycles and the allowances.

| Config | Network | Researchers | Cycles | At most |
| --- | --- | ---: | ---: | ---: |
| `config/experiments/rnn-b-team-smoke.yaml` | `rnn-b-s0` (4 units) | 2 | 1 | $15 |
| `config/experiments/rnn-b-team.yaml` | `rnn-b-s0` (4 units) | 4 | 5 | $120 |
| `config/experiments/rnn-a-team.yaml` | `rnn-a-s2` (2 units) | 2 | 3 | $60 |

Each file's comments give what every role gets. The two `rnn-b` files run no baseline agent
(`orchestration.mode: trained`); `rnn-b-team.yaml` is set up to be read against the single-agent
run `pilot-rnn-b`, with the same networks, five rounds as five cycles, patience, sample size,
reward and agent. Run the smoke configuration first:

```bash
python -m level1.loop run --config config/experiments/rnn-b-team-smoke.yaml --auth api_key
python -m level1.loop run --config config/experiments/rnn-b-team.yaml --auth api_key
```

A cycle's researchers run side by side, and each of their Python programs may use up to
`python.memory_limit_mb` of memory, so four researchers can need four times that at once. Set
`orchestration.parallel: false` to run them one after another. The baseline is one agent's
session, as in the rest of this framework, not a team.

## Roles and schedule

In `both` mode, one global baseline session runs first. Its validated program is frozen before the target's
research begins. An initial lead session then plans investigations. Each research cycle has
researcher sessions, an evaluation barrier and a lead session that integrates the findings and
plans further work. Only the frozen comparison program and its target-scope measurements enter
research context; the baseline's exploratory notes and unselected programs do not enter the
regional archive.

`trained` mode starts with the planning lead. `baseline` mode ends after baseline verification and
its deterministic report. It creates no regional workers, target-research lead or paid summary.
The `baseline` and `both` modes use the same baseline validation policy; running it alone
does not relax its acceptance checks.

The lead gives out tasks with `assign_task(task, scope=None, entry_name=None)`: at most
`orchestration.workers` in a cycle, one per researcher. The lead decides what a task is, for example
some of the network's units or another part of its computation, a group of inputs, making a saved
program cheaper, or testing a claim. A researcher's programs are ordinary entries scored by the same
reward. A program that computes part of the network its own way and the rest as the network does is
such an entry, so work on a part of the network is measured by its effect on the network's output.
`scope` only fixes where the researcher's submissions are scored; without it they are scored on all
inputs. Each assignment's scope is frozen before the researcher begins. Researchers can explore and
submit programs directly to the same verifier used by the lead. They cannot broaden their assigned
scope or change the reward, prices or workload policy. They can submit an entry within that
assignment or make a separately evaluated claim on a narrower domain. The controller retains each
actual evaluation domain.

A lead that gives out fewer tasks than `orchestration.workers` runs fewer researchers, and each gets
more of the cycle's allowance. With more than one task, the lead is encouraged to leave room for an
alternative explanation instead of making every researcher follow the same hypothesis. Researcher
contexts and archive snapshots are fixed before a parallel phase begins. Their results become
available to the integration session after evaluation. If the planning lead gives out no tasks,
`orchestration.initial_tasks` (tasks whose scope defaults to all inputs) or the older
`orchestration.initial_regions` are used instead.

A lead after a cycle can instead end the research with `finish_research(reason)`. The run then
goes through the same final locking, audits, closing summary and report as after the last cycle,
and the report gives the lead's reason as the stop reason. The tool needs a program for the whole
input distribution that has already been scored, by that lead or earlier in the run, and it cannot be
combined with tasks given out in the same session. The planning lead cannot finish, and the lead of
the last cycle needs no tool for it. An empty task list without `finish_research` still means "no
decision": the configured initial tasks are used, or, without them, the run stops unfinished.

Progress is recorded separately for the global incumbent and regional frontiers. Patience expires
only when neither improves. All submissions are retained, including programs that are useful for
shared computation but are not currently on a frontier.

### Saved programs and checked submissions

`save_entry` preserves an executable draft. `submit_entry` spends a submission allowance and produces
a verifier receipt. These are separate operations. Planning has no submission allowance, but its
saved programs must still be available to the researchers who receive its assignments.

After each completed session, the controller snapshots new saved programs as immutable artifacts.
Later sessions receive them through `entries`, with a stable `draft_...` name derived from the
program's content hash and its source-session provenance. An original name is also available when
it is unambiguous and does not conflict with another entry. Unchanged programs carried through
several sessions are stored once. Parallel workers use the snapshot fixed before their phase;
they cannot see a sibling's unfinished work.

Drafts remain explicitly unscored. A local measurement or a claim in a research note does not make
a draft eligible for selection. Only formally submitted, checked candidates can enter the global
selection or regional frontiers. A later researcher can inspect, improve and submit a carried draft
under its own assigned scope and allowance. Saved work can therefore survive a session limit
without being mistaken for a validated result.

On resume, completed sessions can contribute saved drafts that older code did not collect. Existing
phase snapshots remain fixed for reproducibility; recovered drafts enter newly created snapshots.
Applying a code repair does not retroactively change the inputs of a research session already planned.

### Required integration checkpoint

New research runs give each integrating lead a short initial exploration window, controlled by
`orchestration.checkpoint_python_calls` and set to two Python calls by default. After that window,
if a saved draft exists, `run_python` refuses further calls until the lead has a completed, scored
submission on its assigned global domain. The refusal does not spend a Python call. The initial
window permits inspecting carried drafts or constructing a first candidate when no draft is available.

If the initial window ends with no saved entry, the controller allows one additional construction
repair call within the original Python quota. Once that call saves an entry, the lead must submit
it before further exploration. If the repair also returns no draft, further Python calls remain
blocked and the session has no checkpoint. Reopening the session or retrying the backend does not
refresh this one-call exception. Its granted or consumed allowance remains visible as
`bootstrap_repair_calls` in checkpoint status. If the resulting draft is formally refused, the
separate one-call refusal repair still applies while a submission slot remains. Neither allowance
adds a formal submission or evaluates a fallback program automatically.

The checkpoint uses an ordinary `submit_entry` call and consumes an ordinary submission slot.
Saving an entry, measuring it locally or describing it in notes does not satisfy the requirement.
A narrower claim also does not satisfy it. Until the global checkpoint is scored, the lead's final
remaining submission slot is reserved for that global domain; a child claim cannot consume it.
Once the checkpoint is scored, the lead can continue improving programs or examining regional
claims with its remaining allowances.

If the verifier refuses a global submission, that attempt still consumes a submission slot. While
another slot remains, each completed refusal allows one additional repair Python call before
resubmission is required. A failed child claim does not grant this global repair allowance, and an
interrupted evaluation without a receipt does not count as a checkpoint. These rules survive a
backend retry and reopening the session.

The controller does not choose the latest draft, submit programs implicitly or add a reference
fallback. The researcher chooses which saved program to submit. A scored checkpoint establishes
that an artifact was evaluated, not that its explanation is good: a reference implementation or
weak predictor retains its ordinary measured cost and reward. The protocol cannot guarantee that
an agent discovers a useful mechanism or succeeds before a provider or resource limit ends it.

Tool responses expose checkpoint status and remaining calls in `remaining.checkpoint`. The
session's persisted policy is authoritative. Older saved sessions without that policy retain
their original behavior; setting the new option to `null` disables it for a new configuration.

### Independently evaluated narrower claims

Both workers and integrating leads can supply an optional `scope` to `submit_entry`. The controller
intersects it with the assigned domain, so requesting `{"kind": "all"}` cannot broaden a worker's
assignment. For a saved predicate entry, use the separate `scope_entry_name` argument:

```python
submit_entry(name="candidate", claimed_fidelity=0.99,
             scope={"kind": "predicate"}, scope_entry_name="identified_domain")
```

This is a formal tool call. Inside `run_python`, `measure(entry, scope=...)` and
`network_outputs(scope=...)` support the same conditioned investigation. An inline predicate
contains the saved entry itself, for example
`scope={"kind": "predicate", "entry": entries["identified_domain"]}`. Local measurements remain
unscored drafts until an ordinary submission is made.

Every narrower claim consumes its own submission slot. Its receipt freezes the actual scope, its
probability or estimated probability under the original distribution and the workload convention. Its candidate competes
only with other candidates evaluated on that same domain, and its regional frontier survives
even if no global program improves. This also applies to regional claims submitted by the lead.
The scope is reused for comparison programs and final audits. Overlapping regional results are
retained as separate claims; their rewards or probability masses are not added.

## The same reward on each evaluation domain

For a task domain S, evaluate the original distribution conditioned on S. If the candidate claims
only T within S through its entry's region test, its coverage is P(T given S). Narrowing that test
does not give it coverage 1 on S. An explicitly submitted child scope creates a different evaluation
domain with its own conditional coverage and reward; that score cannot replace the score on S.

```math
R_S = \mu_S [(1-c_S)-\alpha(1-q_S)]
```

The existing `r_alpha` implementation and the configured alpha are unchanged. Exact enumeration
uses exact fidelity. Sampled evaluation uses the existing lower confidence bound. The archive also
records the domain's mass under the original distribution, when available, and its support size.

Compare programs within the same actual evaluation domain and cost policy. Conditional rewards on unrelated
domains do not rank the importance of those domains. The final global winner is selected only among
submissions evaluated on the full domain; the code never promotes a conditional regional score into
a global score. A worker assigned the full domain can contribute a global candidate by submitting
on that full domain, without requiring the lead to resubmit the same program. Its narrower claims
remain regional candidates.

A composed program must implement its routing, prediction and shared preprocessing in the entry
language. It is evaluated as a whole. Regional rewards are not added to estimate its score.

## Scopes and workloads

The region machinery is independent of the model's architecture and its training objective:

| Scope | Definition |
| --- | --- |
| `{"kind": "all"}` | The original input distribution |
| `{"kind": "lengths", "lengths": [...]}` | An allowed subset of token lengths, retaining the conditional distribution |
| `{"kind": "predicate", "entry": ...}` | The region test of an ordinary entry |
| `{"kind": "intersection", "scopes": [...]}` | Inputs satisfying every component restriction |

For a predicate, the lead can save an ordinary entry with that test as its region and pass its
name through `entry_name`. No new executable language is introduced. Conditional sampling has a
bounded draw budget and time limit; an empty or infeasibly rare domain is refused clearly.
Narrower claims can combine length restrictions and predicates through intersection. The resulting
domain keeps the original distribution conditioned on all its restrictions, rather than treating
their individual probabilities as independent. Intersections flatten and deduplicate their
restrictions, have a bounded component count, and share the existing node and constant limits
across predicate components. When a predicate's mass must be sampled inside a length-restricted
domain, its estimate and confidence interval are scaled back to the original distribution.

Scopes with a known finite support default to a workload whose size equals that support. For a
uniform exhaustively checked domain, this is one evaluation per member. For a nonuniform domain,
runtime is averaged at the declared conditional distribution and the support size sets the number
of workload queries. This distinction is recorded; enumerating a nonuniform distribution as if
every distinct input had equal probability would change the task.

For continuous domains or predicates whose finite support cannot be determined within the allowed
enumeration, set `verifier.workload_size` explicitly. That is an operator choice for the experiment,
not a number a worker can increase to discount its own setup. Use the same choice for every method
being compared. Predicate-conditioning work is reported separately; global routing must still be
implemented and charged in the global entry.

Predicates may depend on weights. Their receipts identify that binding explicitly: transfer checks
evaluate the same predicate program using the new checkpoint's weights, so the selected subset can
change. An input-only predicate selects the same subset across checkpoints.

The number of verification samples does not determine setup amortization in scoped mode. Raw
operations, the verification sample size, the workload and the scope are kept in receipts. Setup
is the interpreter's observed lazy build; sampled evaluation is a measurement, not a proof that
every possible setup branch has been exercised.

An omitted `verifier.scope` retains the original measurement policy, including its original setup
amortization. Existing published results remain under that policy. Compare experimental methods
under matching workload conventions, rather than comparing raw scores from different policies.

## Shared resource limits

The original `rounds_max` now counts research cycles. API, Python-call, turn and time allowances
are shared across roles rather than copied to each agent. In `both` mode,
`orchestration.baseline_fraction: 0.2` reserves 20 percent of the original total allowance for
one global baseline session. `orchestration.final_summary: true` reserves
`summary_fraction: 0.1` of the original API, turn and time allowances for final synthesis.
The research roles divide the remaining allowance: by default, planning receives 10 percent,
regional work 60 percent and integration 30 percent of their research share.

With a $25 total API allowance and the default fractions, the nominal allocations are:

| Mode | Global baseline search | Research | Final lead synthesis |
| --- | --- | --- | --- |
| `trained` | $0 | $22.50 | $2.50 |
| `baseline` | $25 | $0 | $0 |
| `both` | $5 | $17.50 | $2.50 |

These are portions of the $25, not additional budgets. The configured per-session cap still
applies: for example, a standalone baseline with a $20 session cap can spend at most that nominal
$20 allocation, even when its parent run allowance is $25. A provider's final response may exceed
an attempt cap as described below. Baseline-only mode assigns the full shared Python, turn and time
allowance to its single search session, while retaining its explicit baseline submission quota.
It reserves nothing for synthesis, even when `final_summary: true` remains in the configuration.

In `both` mode, Python calls reserve 20 percent for the baseline and leave 80 percent for research
because synthesis has no tools. Turn and time allocations reserve both shares and leave 70 percent
for research. In `trained` mode the research roles receive the full Python allowance, and the
remaining turn and time allowance after synthesis. Configurations with other fractions use the
same accounting. Supplying a frozen `baseline_entry` releases the search allocation; its validation
does not call an LLM. Disabling synthesis releases its share to research.

The baseline has an explicit `orchestration.baseline_submissions: 2` allowance for its one
session, including a checkpoint and an improved candidate. This is separate from the existing
research submission quota and does not reduce a worker's or integrating lead's submission slots.
`orchestration.regional_submissions` sets each worker's formal submission allowance per cycle;
the integrating lead receives the remaining `session.submissions` after all workers' shares.
The configuration must leave at least one submission for integration. The planning session
receives no submissions, and the final summary has no tools.

The shared `research.yaml` profile now explicitly allows six submissions per cycle: two for each
of two workers and two for integration. This lets a worker retain both a broader candidate and
a special regional explanation without forcing them to share one score. It is an increase in the
declared verification allowance, not a set of uncounted evaluations. The profile does not raise the
API budget. Other configurations retain the default of one submission per worker unless they
override it; saved configurations without the new option keep that original allocation.

In `both` mode at five cycles, the shared research profile allows thirty research submissions plus the two
baseline submissions; a configuration with four research submissions per cycle allows twenty
research submissions plus the baseline's explicit quota. No synthesis submissions are added.
One shared API spending log covers the baseline, every research role, synthesis, retry and resume.
Session identifiers are unique.
Atomic reservations enforce the parent run, cycle and role caps when workers run simultaneously.
Unreported spending is conservatively counted at the reserved cap. As in the original backend,
the provider's cap may be exceeded by the final answer that reaches it; this is not a prepayment
or an exact billing guarantee.

Research prompts state their tool, turn, time and nominal API allowances. After the controller
reserves an API attempt, the request also states its actual dollar allowance, which may be smaller
because of earlier costs or outstanding reservations. A redacted copy of the complete task prompt
sent to the CLI is saved as `request_NN.md` beside its stream log. Before each CLI attempt, the
backend also records that attempt's wall-clock deadline. Tool responses include
`remaining.wall_seconds_remaining`, so a researcher can distinguish remaining time from unused
tool calls. A retry gets a new attempt deadline while retaining the session's tool, submission
and checkpoint counters. This is a deadline for the current backend attempt, not a live estimate
of dollar spend. The backend reconciles reported cost when the attempt ends.

Spending messages distinguish open reservations from completed attempts without a usable reported
cost. An open record may belong to a worker still running or to an interrupted attempt. Both kinds
of unreported cost retain their conservative accounting; the wording does not declare a running
worker to have failed.

Under login authentication, the API-dollar cap is not a billing mechanism. Compare research
procedures using the same declared tool, turn and time allowances, and use API-key accounting when
matching dollar spend matters.

### Deadline and execution failures

A backend wall-clock timeout is recorded as `timeout`, not `end_turn`. Turn-limit and backend-error
stops are also distinct from a completed response. A partial or failed final summary is reported as
incomplete and does not become a completed lead synthesis merely because text was returned.
Completed submission receipts remain valid evidence when a later session limit is reached.

An exception in sandboxed Python returns an error status as well as its traceback. Any entries
successfully saved before that exception are retained when the wrapper returns them. A hard process
termination may prevent the wrapper from returning, so an in-memory `save_entry` call is not itself
a guarantee that the draft reached disk. Existing completed draft and submission artifacts remain
available. Failed tool calls do not silently become successful measurements.

### Provider refusals

The existing `max_consecutive_refusals` limit also applies to multi-agent research. Completed
sessions are considered in their assigned order, so parallel completion order does not change the
decision. Once the configured consecutive-refusal threshold is reached, it stays reached on resume.
The controller retains work already produced by the current phase, but launches no later research
sessions or new synthesis call. Previously saved synthesis output can still be recovered.
Offline locking, checks and reports preserve the available scientific results.

Raw responses and provider metadata remain in the session records for diagnosis. Subsequent
research context includes a neutral status for a refused or failed session and any checked
submissions, rather than replaying provider error prose as research notes. The research request
identifies the supplied target network and asks for reproducible experiments and concise,
checkable explanations. It does not ask the researcher assistant to expose private reasoning.
These clarifications do not guarantee that a provider will accept a request. A refusal is not
handled by silently changing the model or retrying the same blocked work indefinitely.

## Automatic global baseline and supplied comparisons

The baseline options in the shared research profile are:

```yaml
orchestration:
  mode: null
  baseline_agent: true
  baseline_validation_version: 2
  baseline_fraction: 0.2
  baseline_submissions: 2
  baseline_entry: null
```

`mode: null` preserves configuration inference and selects `both` here because `baseline_agent`
is true. An explicit `trained`, `baseline` or `both` takes precedence over `baseline_agent`.

One agent works on the complete input distribution of the declared control model. It receives
the ordinary model definition, weights, entry language and reward, together with the neutral
requirement that its computation remain valid for arbitrary weights. It is not told what the
target was trained to do or that its own parameter checkpoint is a control. This session does
not assign regional workers.

Only formal submissions are eligible for the generated baseline. Under validation policy version 2,
the controller considers every eligible submission. Each must cover every evaluated input and
agree with every checked output on the session's control checkpoint, a separate control
selection checkpoint, every remaining declared control checkpoint and every checkpoint of the
declared compatible trained twins. A failure on any required checkpoint rejects that candidate.
The session's early checkpoint likewise requires full coverage and exact empirical fidelity.

Among the candidates passing all gates, ranking uses rewards from the control selection
checkpoint only. Trained checkpoint outputs therefore affect eligibility, but their rewards do
not select the winner. The validated program is frozen before any trained-model research session.
This policy catches, for example, a cheap rule that happens to work on low-variation control
outputs but fails after training. It remains empirical validation over finite checkpoints and
evaluated inputs, not a universal proof for every weight assignment or every unsampled input.

Validation records store their policy version. Legacy version 1 runs retain their original
control-only acceptance checks and any original candidate limit; resume does not upgrade them
or claim that trained-checkpoint validation happened. A new run uses version 2.

If no formally submitted candidate passes, the pipeline records a failed baseline and stops
visibly before target research. It does not substitute an unsubmitted draft or silently report
a completed comparison. A completed baseline phase and its frozen entry are reused on resume;
its session quotas and spending records are retained after interruption.

The same frozen global program is evaluated on the target's full domain and each regional claim.
For each comparison, both programs use the same target weights, input scope, sample, reward,
operation prices and setup workload. The baseline's tables, if any, are rebuilt from the target's
weights and charged under that policy. Regional scores describe this global baseline restricted
to the region; there is no separate search for an optimized regional baseline. Development
comparisons and final fresh-input or second-checkpoint comparisons remain separate.

Every displayed validated reward gap also requires the comparator to cover the complete actual
evaluation scope and match every checked output on that target checkpoint. The candidate and
comparator receipts must establish matching input samples or enumeration, prices, scope and setup
workload. If those conditions fail, the report marks the comparison invalid and withholds the
validated gap. The raw receipts and raw reward difference can remain available with that status.
Passing earlier parameter checks does not override a later incorrect or partial-scope comparator.

An existing frozen entry can be supplied without starting another model run:

```yaml
orchestration:
  baseline_entry: /absolute/path/to/baseline_entry.json
```

This overrides generation and its resource reservation. In a new version 2 run the supplied entry
is snapshotted and must pass the same full-coverage exactness checks on all required control
and trained checkpoints before use. No search session is claimed for it, and validation does not
establish how it was found or prove correctness for arbitrary weights. A failure stops the phase
visibly. Researchers see either source of frozen program under the neutral name `comparison_entry`.
Legacy supplied-entry records remain labeled with the validation that actually occurred; merely
resuming one does not add these new gates retroactively.

To reuse a baseline-only run, set a new run's `baseline_entry` to the old run's
`multi_agent/baseline_entry.json` and select `both`. The new run validates the imported program under
its saved policy without paying for another baseline agent. Explicit `trained` mode rejects a supplied
comparison entry, because that mode requests research without a baseline. With `mode: null`,
`baseline_agent: false` omits generation only when no entry is supplied. Old saved configurations
retain their original stages and validation policy on resume and do not acquire a new paid phase.
Normal `run` and `resume` perform the configured automatic phase without an extra command.
The original single-agent pipeline's baseline commands retain their existing behavior.

This single-agent baseline has a different objective and resource allocation from multi-agent
discovery. It is a known generic comparator, not a matched-budget multi-agent control search.
A positive reward gap can include generic optimizations that the baseline agent did not find.
Evaluating the discovered interpretation itself on random parameters is also a separate test.
See `RANDOM_BASELINES.md` for these distinctions.

## Development, locking and reports

Development uses fixed samples per task so resubmitting an identical program cannot obtain a new
lucky draw. This does not prevent adaptive overfitting. The global winner and regional frontier
portfolio are frozen before fresh-input evaluation or second-network evaluation. Final results never
enter research-session prompts and do not select the winners. A terminal synthesis session may read
them to explain the already frozen results; it cannot restart research or alter selection.

The run folder contains:

| Path | Contents |
| --- | --- |
| `multi_agent/state.json` | Persisted session plans, indices and cycle progress |
| `multi_agent/sessions/` | Separate prompts, tool records, programs and responses |
| `multi_agent/artifacts/` | Immutable saved drafts, addressed by content hash |
| `multi_agent/archive.json` | Submissions, per-domain selections, comparison scores and separate draft provenance |
| `multi_agent/baseline.json` | Baseline source, validation outcome, selection and provenance |
| `multi_agent/baseline_checks.json` | Cached candidate validation, checkpoint identities, seeds and receipts under the saved validation policy |
| `multi_agent/baseline_entry.json` | The frozen global comparison program, when available |
| `multi_agent/lock.json` | Frozen global and regional portfolio |
| `multi_agent/audits/` | Fresh-input and second-network measurements after locking |
| `locked_entry.json` | The selected global program, if one was submitted successfully |
| `lead_summary.md` | The completed lead explanation of global and regional findings |
| `multi_agent/summary/` | Frozen synthesis evidence, prompt, response, status and execution metadata |
| `report.md`, `report.json` | Global and regional results with their evaluation conditions |

Resume reuses persisted plans and completed session records. A completed lock remains fixed.
The report describes empirical agreement; an agent's explanation is not treated as a formal proof.
The baseline record identifies whether trained checkpoints were acceptance gates and whether their
rewards affected ranking. Baseline-only execution writes its validation report and frozen comparator
without constructing a research portfolio or a lead summary. Mode and policy version remain visible
in the saved records, including for older control-only runs.

## Final lead synthesis

In `trained` and `both` modes, when final synthesis is enabled, the controller locks candidates and
completes their final checks. The lead then receives a fixed
evidence packet containing the selected programs, their receipts, scopes and workload conventions,
the final audits, baseline provenance and comparison measurements, and earlier researchers' notes. Notes are explicitly
unverified evidence. The summary uses the same configured model and account as the research roles.

The requested synthesis explains:

- The best global program found under the declared reward, its computation and source of savings.
- The best supported overall mechanistic understanding, including shared mechanisms and internal
  reasons for differences between regions, with unresolved connections stated explicitly.
- Useful regional interpretations, including findings that did not improve global composition.
- The mechanistic explanation for each material region when supported, or a clear statement that
  the mechanism remains unclear without dismissing the measured interpretation.
- Each material region's scope probability, conditional coverage, fidelity, cost and workload.
- Supporting measurements, recorded failures, assumptions, unresolved hypotheses and proof status.
- Which arguments appear valid for general parameters and which depend on particular parameter values.
- Valid global and regional reward gaps against the same frozen global baseline, with matched conditions and explicit reasons when a comparison is invalid or unavailable.
- Available comparisons and missing control experiments, without inventing a control result.

"Best" means the recorded selection among evaluated candidates under the fixed policy. The summary
does not claim a global optimum or choose a new winner using final-audit results. Both generated and
supplied comparisons have only the empirical validation actually recorded under their policy version;
legacy control-only records do not become version 2 results. The summary explains that trained
checkpoint outputs can affect comparator eligibility without trained rewards determining its rank.
It withholds generic-improvement claims when the comparator fails its actual-scope checks. It
distinguishes the single-agent generic search from a matched
multi-agent control and from transferring the discovered program to other parameters. Agreement on
another checkpoint does not by itself identify a learned mechanism.

Comparison-search provenance and research-candidate control evidence are separate fields. A false
candidate-control flag means that no such candidate evaluation was supplied; it does not negate a
recorded independent comparison search. New evidence omits the ambiguous legacy control flag, and
the prompt projection ignores it in older evidence snapshots without changing the saved records.
Existing completed narratives are preserved and are not regenerated automatically.

The synthesis session exposes no tools or MCP servers. It cannot execute code, submit entries, assign
regions or change the archive. Its output is commentary with evidence references, not a certificate.
`report.md` links the narrative, and `report.json` records its status and completed text.

The summary plan, evidence and response checkpoints are persisted. A completed summary is reused on
resume; interrupted synthesis can continue without reopening research. A missing backend, exhausted
allowance, empty answer or incomplete response is reported explicitly, while the frozen scientific
results remain available. Old saved configurations without `final_summary` do not silently acquire
an additional model call on resume. No real model call is made by the fake-backend tests.

## Validate without model calls

The fake backend exercises the real sessions, sandbox, regional sampler, verifier, archive,
integration and resume behavior. Unit tests also cover competing regional/global scores, shrinking
claims, fixed setup workloads, duplicate submissions, parallel budget reservations and access to
the assignment tool.

```bash
python -m pytest tests/test_regions.py tests/test_multi_agent.py tests/test_multi_budget.py tests/test_research_session.py
python -m pytest tests/test_checkpoints.py tests/test_multi_agent_repairs.py tests/test_prompt_repair.py tests/test_backend.py
python -m pytest tests/test_global_baseline_pipeline.py tests/test_global_baseline_prompt.py tests/test_synthesis.py tests/test_research_modes_cli.py
```

To attribute gains to delegation, compare with a single researcher given the same regional
objectives, archive and total budget. Comparing the original pipeline against this entire mode
instead measures the combined effect of the new task formulation and orchestration.
