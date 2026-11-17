# Regional research and global integration

The optional research mode uses the existing reward and entry language. A lead researcher assigns
input domains to regional researchers, then uses their checked results to construct a complete
program. Code owns evaluation, the shared budget, the archive and the final lock. After the final
checks, a separate tool-free lead session explains the frozen results.

The original run mode remains the default. No operation prices or entry-format identifiers change.
The experiment is selected by `orchestration.enabled: true`.

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
python -m level1.loop run --config config/experiments/lin-c-regional.yaml
```

The model-neutral profile is `config/experiments/research.yaml`. The example above selects one model
and its checkpoints but supplies no expected behavior or input-length split. The lead chooses regions
from its investigation. If it makes no assignments, two generic full-domain questions provide a
fallback: search for useful special explanations, and independently examine alternatives and exceptions.
A regional result does not need to improve the global score to be retained.

To bill the experiment through the existing API-key backend with one total spending limit:

```bash
python -m level1.loop run --config config/experiments/lin-c-regional.yaml --auth api_key --budget-usd 60
```

The existing `CPL_ANTHROPIC_KEY` mechanism is used. All roles can use the same API account and model.
Each role gets its own session, prompt, saved programs and logs. Setting `orchestration.parallel` to
`false` runs the regional sessions sequentially with the same roles and allocations.

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

The normal sandbox checks and trusted-code fingerprint apply. Do not resume an old scientific run
under changed trusted code unless deliberately accepting that change with the existing flag.

## Applying the framework to another model

The controller and prompts do not assume a particular architecture, training objective or special
input length. Researchers receive the definition of the actual model being investigated, its weights,
the declared input distribution and their assigned scopes. They do not receive withheld training
objectives, training-input metadata or the target checkpoint's training status.

An experiment can extend `research.yaml` and supply its own `model`, `network`, `run_id`, optional
`seed_check.network`, and verifier settings. Those are experiment choices; no controller code or
research prompt needs to change when selecting another compatible adapter.

Model independence is relative to the existing model interface. An adapter supplies `reference(g,
lv1)`, configured weight arrays and a supported input distribution. The current reference returns one
integer output per input, has constant forward operation cost across those inputs, and is expressible
in the entry language. Existing adapters cover several architectures and both discrete and real
inputs. An arbitrary external model or natural-language corpus may need a new adapter, distribution
implementation and cost model. Unknown-cardinality domains need an explicit workload.

## Roles and schedule

An initial lead session plans investigations. Each research cycle then has regional sessions,
an evaluation barrier and a lead session that integrates the findings and plans further work.

The lead can call `assign_region(scope, task, entry_name=None)`. Each assignment contains an
immutable scope and a bounded question. Workers can explore and submit programs directly to the
same verifier used by the lead. They cannot broaden their assigned scope or change the reward,
prices or workload policy. They can submit an entry within that assignment or make a separately
evaluated claim on a narrower domain. The controller retains each actual evaluation domain.

With two workers, the lead is encouraged to use one for a directed investigation and the other for
an alternative hypothesis or underexplored domain. Worker contexts and archive snapshots are fixed
before a parallel phase begins. Their results become available to the integration session after
evaluation.

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

The original `rounds_max` now counts research cycles. Tool quotas are divided across roles from
the original overall allowance, rather than copied to each agent. Planning receives 10 percent of
exploration calls, regional work 60 percent, and integration the remaining 30 percent.
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

For new research runs, `orchestration.final_summary: true` reserves `summary_fraction: 0.1` of the
API, turn and time allowances for final synthesis. Planning, regional work and integration divide the
remaining 90 percent in the same 10/60/30 proportions. Thus a configured $60 API allowance reserves
up to $6 for synthesis and up to $54 for research; it does not add another $6. Disabling final synthesis
restores the original allocation. Python-call and submission totals are unchanged because synthesis
cannot use those tools.

At five cycles, the shared research profile therefore allows thirty submissions in total; a
configuration with four submissions per cycle still allows twenty.
One shared API spending log covers every role, retry and resume. Session identifiers are unique.
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

## Generic comparison program

An existing frozen entry can be supplied without starting another model run:

```yaml
orchestration:
  baseline_entry: /absolute/path/to/baseline_entry.json
```

The entry is snapshotted, then evaluated on each encountered scope using that task's weights,
inputs, reward and workload policy. Its development and final measurements are separate. The
researchers see it under the neutral name `comparison_entry`.

Automatic `--with-baseline` and baseline-run generation are not supported for the new archive
layout. Those commands fail with an explanation instead of silently comparing incompatible
single-agent and regional experiments. Existing baseline runs continue to work in the original
mode.

An imported program does not, by itself, establish that an independent search was run on randomly
generated parameter controls. These are separate experiments. See `RANDOM_BASELINES.md` for the
distinction between a same-target generic comparison, an independent matched control search, and
transfer of one frozen interpretation.

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
| `multi_agent/lock.json` | Frozen global and regional portfolio |
| `multi_agent/audits/` | Fresh-input and second-network measurements after locking |
| `locked_entry.json` | The selected global program, if one was submitted successfully |
| `lead_summary.md` | The completed lead explanation of global and regional findings |
| `multi_agent/summary/` | Frozen synthesis evidence, prompt, response, status and execution metadata |
| `report.md`, `report.json` | Global and regional results with their evaluation conditions |

Resume reuses persisted plans and completed session records. A completed lock remains fixed.
The report describes empirical agreement; an agent's explanation is not treated as a formal proof.

## Final lead synthesis

After the controller locks candidates and completes their final checks, the lead receives a fixed
evidence packet containing the selected programs, their receipts, scopes and workload conventions,
the final audits, supplied comparison measurements, and earlier researchers' notes. Notes are explicitly
unverified evidence. The summary uses the same configured model and account as the research roles.

The requested synthesis explains:

- The best global program found under the declared reward, its computation and source of savings.
- Useful regional interpretations, including findings that did not improve global composition.
- Each material region's scope probability, conditional coverage, fidelity, cost and workload.
- Supporting measurements, recorded failures, assumptions, unresolved hypotheses and proof status.
- Which arguments appear valid for general parameters and which depend on particular parameter values.
- Available comparisons and missing control experiments, without inventing a control result.

"Best" means the recorded selection among evaluated candidates under the fixed policy. The summary
does not claim a global optimum or choose a new winner using final-audit results. A supplied comparison
program is not automatically a verified arbitrary-weight method, and agreement on another checkpoint
does not by itself identify a learned mechanism.

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
```

To attribute gains to delegation, compare with a single researcher given the same regional
objectives, archive and total budget. Comparing the original pipeline against this entire mode
instead measures the combined effect of the new task formulation and orchestration.
