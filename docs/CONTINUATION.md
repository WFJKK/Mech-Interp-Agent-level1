# Going on from a finished run at a higher price

A run's reward prices a wrong answer at alpha forward passes (`docs/REWARD.md`). At alpha = 10 a fix
for the last few errors often costs more than it earns, so a run can end with an entry that is right
on almost every input and leaves the rest unexplained. A run with `continue_from: <run_id>` goes on
from such a run on the same network, at a price at which those errors are worth fixing
(`level1/continuation.py`). It works for any model: everything it uses is measured.

## The price

It comes from the entry the earlier run locked, as measured again on fresh inputs (or on every
input, when every input is checked):

- C, its cost over all inputs, an input outside its region counting a forward pass;
- e, the share of all inputs it answers wrongly.

Removing all of its errors raises the reward by alpha times e, and pays if that is more than the
cost it adds. So that it pays even if the cost rises to `c_max` (1: a forward pass on every input):

    alpha = margin * (c_max - C) / e

rounded up and never below the earlier price, with `margin` 2 so that an entry right on every input
wins clearly rather than ties. A number in `continuation.alpha` sets the price instead.

## At a lower price

`continuation.alpha` can also be set below the earlier price. A higher price pays for being right by any
means. On inputs that a program gets right only by doing much of the network's own computation, a
cheaper prediction takes the place of that computation only where the share of those inputs it gets
wrong stays below what it saves per input, in forward passes, divided by alpha. On ARC's challenge
network the team's program computes a third of all inputs at about 0.63 of a forward pass each, so at
alpha = 360 that bar is about 0.18%: an account of those inputs that is not exact earns nothing there,
and the cheapest way to be right is to compute them more cheaply, which is what the team did. At
alpha = 30 the bar is about 2.1%, so an account that is right on most of them counts. The note the
agents get then says so (`prompt.CONTINUATION_NOTE_LOWER`, `prompt.TEAM_CONTINUATION_NOTE_LOWER`).

Part of any gain at a lower price comes without new understanding, from computing fewer inputs and
accepting more errors. To tell the two apart, compare with the earlier locked program with only its
thresholds re-tuned to the new price.

## With less room

A run that goes on can give entries less room than the earlier one: fewer nodes (`entry.max_nodes`)
or fewer numbers in their constants (`entry.max_const_scalars`). Fewer numbers are the way to keep
an entry from storing what was observed of the network's outputs. On ARC's challenge network the
single agent's locked entry held 338 numbers and the team's 460, in both almost all of them tables
fitted to millions of sampled outputs, which say where a rule fails and not why. With room for 64
there is room for the exact computation of the network (41 numbers in the team's) and a few
thresholds, and for no such table. What an entry needs beyond its constants it computes from the
weights, in its build, where the work is counted.

The earlier entries that are over the new limits are still carried in `entries`, to read and to
take parts from; they cannot be measured, saved or submitted as they are, and the note says so
(`prompt.CONTINUATION_LIMITS_LOWER`, `prompt.TEAM_CONTINUATION_LIMITS_LOWER`). Their rewards stay in
the earlier rounds' table, so the best reward on record can be one that no entry within the limits
reaches: a single agent's run then needs a `patience` as long as its `rounds_max`, or it stops for
want of an improvement.

## The sample

On a sample, an entry right on every input drawn is still charged alpha times the gap below 1 of
the fidelity's lower bound, about alpha * 3.84 / n for n inputs. The sample therefore grows with the
price, to `inputs_per_alpha` (200) times alpha inputs, at most `max_sample_size`, which keeps that
charge near 0.02. When every input is checked, the sample stays as it was. With more inputs per
scoring, an entry's build is also spread over more inputs; builds are small next to a sample's
forward passes.

## What the agent sees

The earlier run is read and never written. This run's rounds are numbered after the earlier ones,
and each round's prompt carries:

- the earlier rounds' table, every reward recomputed at the new price from what was measured then;
- their notes, and their scored entries in `entries` (with `best`, the best at the new price);
- a note (`prompt.CONTINUATION_NOTE`): the earlier prices, the locked entry's measurements, that
  these rounds are for going into more detail on what the earlier rounds did not understand, the
  new price and how it was set, what the locked entry scores at it, and the new sample size.

With `continuation.memory: false` the run gets the same price and sample and nothing of the earlier
rounds: the control that shows what the memory is worth. A continuation can itself be continued;
the note then lists every earlier price with its rounds.

## A team going on from a team or from a single agent

A team run (`orchestration.enabled`, `docs/MULTI_AGENT.md`) goes on from a finished team run, or from
a finished single agent's run, in the same way: the price and the sample come from the program the
earlier run locked, measured again on fresh inputs. A single agent's run goes on only from a single
agent's, and a team that goes on does research only (`orchestration.mode: trained`). Its research
phases are planned, run and stopped as a new run's are: a planning lead gives out the tasks of the
next phase, and the run has its own `rounds_max` phases, patience and allowances. After a team, its
sessions are numbered after the earlier ones (team-rnn-b's seven sessions end at a0007, so the next
run starts at a0008); after a single agent, whose programs are named r1s1 to r9s4 and the like, they
start at a0001. Every session's prompt carries:

- the earlier work in the checked archive, under "Earlier research": the frontier of each domain,
  every reward recomputed at the new price beside the one recorded then, where they differ. A single
  agent's scored submissions are its archive, each scored on the whole input distribution;
- the earlier programs in `entries` under their names, and an earlier team's saved drafts under their
  aliases, read from the earlier run's folder;
- the earlier notes, marked as written at the earlier prices, before this run's: an earlier team's
  sessions, or a single agent's rounds, each headed with its round;
- a note (`prompt.TEAM_CONTINUATION_NOTE`) that says what the single agent's note says, of the
  earlier research phases or rounds, and adds that a domain given by a region test must hold clearly
  more than the sample size over 200,000 (`regions.DEFAULT_DRAWS`) of all inputs to be scored.

The earlier programs are shown and carried but never selected: only a program submitted in the new
run can be locked, as in a single agent's continuation, so an earlier program counts only if it is
submitted again. The control (`continuation.memory: false`) starts at a0001 with none of this.

## How much is understood

The report of a continued run gives the understood share of the entry it locked, if that entry is
right on every input it is measured on again:

    U = (1 - C_ex) / (1 - C)

with C_ex its cost over all inputs: the share of the earlier entry's saving that survives when every
answer must be right. U is 100% when the earlier errors cost nothing extra to get right and 0% when
getting them right takes the network on every input. It is for people reading the run; the agent
sees only the reward, which ranks entries right on every input by their cost, as U does.

## Running it

```bash
python -m level1.loop run --config config/pilot/rnn-b-detail-smoke.yaml --auth api_key
python -m level1.loop run --config config/pilot/rnn-b-detail.yaml --auth api_key
python -m level1.loop run --config config/pilot/rnn-b-detail-control.yaml --auth api_key
```

For the rnn-b team, two phases going on from `team-rnn-b` (alpha 364 and 72,800 inputs per scoring
for its lock), after a one-phase smoke run:

```bash
python -m level1.loop run --config config/experiments/rnn-b-team-detail-smoke.yaml --auth api_key
python -m level1.loop run --config config/experiments/rnn-b-team-detail.yaml --auth api_key
```

A team of four on ARC's 432-weight challenge network, three phases going on from the single agent's
nine rounds on it (`pilot-rnn-g-detail-2` and the two runs before it), at the single agent's price of
alpha 360 and 72,000 inputs per scoring:

```bash
python -m level1.loop run --config config/experiments/rnn-g-team-detail-smoke.yaml --auth api_key
python -m level1.loop run --config config/experiments/rnn-g-team-detail.yaml --auth api_key
```

The same team going on from that run at a lower price, alpha = 30 (see "At a lower price"), two phases
after a one-phase smoke run:

```bash
python -m level1.loop run --config config/experiments/rnn-g-team-alpha30-smoke.yaml --auth api_key
python -m level1.loop run --config config/experiments/rnn-g-team-alpha30.yaml --auth api_key
```

The single agent going on from its own nine rounds on that network with room for 64 numbers, at
alpha = 100, three rounds after a short smoke run:

```bash
python -m level1.loop run --config config/pilot/rnn-g-few-numbers-smoke.yaml --auth api_key
python -m level1.loop run --config config/pilot/rnn-g-few-numbers.yaml --auth api_key
```

The same start under the reward that prices stored numbers (`r_alpha_numbers`, `docs/REWARD.md`), at
three prices, with the room for numbers back at 512 so that the price decides:

```bash
python -m level1.loop run --config config/pilot/rnn-g-numbers-001.yaml --auth api_key
python -m level1.loop run --config config/pilot/rnn-g-numbers-002.yaml --auth api_key
python -m level1.loop run --config config/pilot/rnn-g-numbers-005.yaml --auth api_key
```

A single agent's run can go on from rounds scored with `r_alpha` under `r_alpha_numbers`; a team's
cannot yet.

The earlier run must be finished (it has a `lock.json`) and in the same `runs_dir`, on the same model
and network, under the same prices and reward.
