# Regional research and global integration

The optional research mode uses the existing reward and entry language. A lead researcher assigns
input domains to regional researchers, then uses their checked results to construct a complete
program. Code owns evaluation, the shared budget, the archive and the final lock.

The original run mode remains the default. No operation prices or entry-format identifiers change.
The experiment is selected by `orchestration.enabled: true`.

## Run the experiment

Use the repository's usual installation and Claude Code authentication. From the repository root:

```bash
python -m level1.loop run --config config/experiments/lin-c-regional.yaml
```

The configuration includes two initial input-defined investigations. The lead may propose other
investigations; the configured tasks supply a fallback if it makes no assignments. A regional result
does not need to improve the global score to be retained.

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
not infer or receive a training objective.

As usual, each execution needs a fresh run ID, or resumes an existing run:

```bash
python -m level1.loop resume --run /absolute/path/to/the/run
python -m level1.loop finish --run /absolute/path/to/the/run
```

The normal sandbox checks and trusted-code fingerprint apply. Do not resume an old scientific run
under changed trusted code unless deliberately accepting that change with the existing flag.

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
the original overall allowance, rather than copied to each agent. By default, planning receives
10 percent, regional work 60 percent, and integration the remaining 30 percent of exploration and
API allocations. Each worker receives one submission per cycle; the lead receives the remaining
submissions in that cycle. The planning session receives no submissions.

The default five cycles and four submissions per cycle therefore allow twenty submissions in total.
One shared API spending log covers every role, retry and resume. Session identifiers are unique.
Atomic reservations enforce the parent run, cycle and role caps when workers run simultaneously.
Unreported spending is conservatively counted at the reserved cap. As in the original backend,
the provider's cap may be exceeded by the final answer that reaches it; this is not a prepayment
or an exact billing guarantee.

Under login authentication, the API-dollar cap is not a billing mechanism. Compare research
procedures using the same declared tool, turn and time allowances, and use API-key accounting when
matching dollar spend matters.

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

## Development, locking and reports

Development uses fixed samples per task so resubmitting an identical program cannot obtain a new
lucky draw. This does not prevent adaptive overfitting. The global winner and regional frontier
portfolio are frozen before fresh-input evaluation or second-network evaluation. Final results
never enter researchers' prompts and do not select the winners.

The run folder contains:

| Path | Contents |
| --- | --- |
| `multi_agent/state.json` | Persisted session plans, indices and cycle progress |
| `multi_agent/sessions/` | Separate prompts, tool records, programs and responses |
| `multi_agent/archive.json` | All submissions, per-domain selections and comparison scores |
| `multi_agent/lock.json` | Frozen global and regional portfolio |
| `multi_agent/audits/` | Fresh-input and second-network measurements after locking |
| `locked_entry.json` | The selected global program, if one was submitted successfully |
| `report.md`, `report.json` | Global and regional results with their evaluation conditions |

Resume reuses persisted plans and completed session records. A completed lock remains fixed.
The report describes empirical agreement; an agent's explanation is not treated as a formal proof.

## Validate without model calls

The fake backend exercises the real sessions, sandbox, regional sampler, verifier, archive,
integration and resume behavior. Unit tests also cover competing regional/global scores, shrinking
claims, fixed setup workloads, duplicate submissions, parallel budget reservations and access to
the assignment tool.

```bash
python -m pytest tests/test_regions.py tests/test_multi_agent.py tests/test_multi_budget.py tests/test_research_session.py
```

To attribute gains to delegation, compare with a single researcher given the same regional
objectives, archive and total budget. Comparing the original pipeline against this entire mode
instead measures the combined effect of the new task formulation and orchestration.
