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
same verifier used by the lead. They cannot change their own assigned scope, the reward, prices or
workload size. They may still submit an entry claiming a subset of that scope.

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

## The same reward on each assigned domain

For a task domain S, evaluate the original distribution conditioned on S. If the candidate claims
only T within S, its coverage is P(T given S). It does not get coverage 1 by narrowing T.

```math
R_S = \mu_S [(1-c_S)-\alpha(1-q_S)]
```

The existing `r_alpha` implementation and the configured alpha are unchanged. Exact enumeration
uses exact fidelity. Sampled evaluation uses the existing lower confidence bound. The archive also
records the domain's mass under the original distribution, when available, and its support size.

Compare programs within the same assigned domain and cost policy. Conditional rewards on unrelated
domains do not rank the importance of those domains. The final global winner is selected only among
submissions evaluated on the full domain; the code never promotes a conditional regional score into
a global score. A worker explicitly assigned the full domain can contribute a global candidate
without requiring the lead to resubmit the same program.

A composed program must implement its routing, prediction and shared preprocessing in the entry
language. It is evaluated as a whole. Regional rewards are not added to estimate its score.

## Scopes and workloads

The region machinery is independent of the model's architecture and its training objective:

| Scope | Definition |
| --- | --- |
| `{"kind": "all"}` | The original input distribution |
| `{"kind": "lengths", "lengths": [...]}` | An allowed subset of token lengths, retaining the conditional distribution |
| `{"kind": "predicate", "entry": ...}` | The region test of an ordinary entry |

For a predicate, the lead can save an ordinary entry with that test as its region and pass its
name through `entry_name`. No new executable language is introduced. Conditional sampling has a
bounded draw budget and time limit; an empty or infeasibly rare domain is refused clearly.

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
exploration calls, regional work 60 percent, and integration the remaining 30 percent. Each worker
receives one submission per cycle; the lead receives the remaining submissions in that cycle.
The planning session receives no submissions, and the final summary has no tools.

For new research runs, `orchestration.final_summary: true` reserves `summary_fraction: 0.1` of the
API, turn and time allowances for final synthesis. Planning, regional work and integration divide the
remaining 90 percent in the same 10/60/30 proportions. Thus a configured $60 API allowance reserves
up to $6 for synthesis and up to $54 for research; it does not add another $6. Disabling final synthesis
restores the original allocation. Python-call and submission totals are unchanged because synthesis
cannot use those tools.

The default five cycles and four submissions per cycle therefore allow twenty submissions in total.
One shared API spending log covers every role, retry and resume. Session identifiers are unique.
Atomic reservations enforce the parent run, cycle and role caps when workers run simultaneously.
Unreported spending is conservatively counted at the reserved cap. As in the original backend,
the provider's cap may be exceeded by the final answer that reaches it; this is not a prepayment
or an exact billing guarantee.

Research prompts state their tool, turn, time and nominal API allowances. After the controller
reserves an API attempt, the request also states its actual dollar allowance, which may be smaller
because of earlier costs or outstanding reservations. A redacted copy of the complete request is
saved as `request_NN.md` beside its stream log. The backend cannot supply a live balance during an
attempt; it reconciles the reported cost when that attempt ends. Researchers are asked to save
usable work promptly and budget for a formal submission when their role permits one.

Spending messages distinguish open reservations from completed attempts without a usable reported
cost. An open record may belong to a worker still running or to an interrupted attempt. Both kinds
of unreported cost retain their conservative accounting; the wording does not declare a running
worker to have failed.

Under login authentication, the API-dollar cap is not a billing mechanism. Compare research
procedures using the same declared tool, turn and time allowances, and use API-key accounting when
matching dollar spend matters.

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
python -m pytest tests/test_multi_agent_repairs.py tests/test_prompt_repair.py tests/test_backend.py
```

To attribute gains to delegation, compare with a single researcher given the same regional
objectives, archive and total budget. Comparing the original pipeline against this entire mode
instead measures the combined effect of the new task formulation and orchestration.
