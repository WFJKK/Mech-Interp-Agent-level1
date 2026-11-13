# Mech-Interp-Agent-level1

## Big picture

In this repo we consider an automated, agentic hill-climbing approach to mechanistic understanding.
This is a setup in which an agent proposes an interpretation $I$ of a network, a verifier scores
it, and the agent climbs that score round after round. While this is formalised in the next
section, we may think of an interpretation as a description of the network that predicts the
network's outputs more cheaply than running it. This is in line with the bottom-up approach of ARC,
where explaining a network means estimating its behaviour mechanistically, by reasoning from its
weights rather than from samples of its outputs, and where such an estimate should compete with
random sampling ([ARC progress update: Competing with sampling](https://www.lesswrong.com/posts/XdQd9gELHakd5pzJA/arc-progress-update-competing-with-sampling)).
Hence the interpretation should be more accurate than any method of the same cost that only runs
the network as a black box, such as running it on a random share of the inputs.

## More formally

We have a model $M$ and a data set $D$, a distribution over the inputs of $M$. An interpretation
$I$ is a program that maps an input $x$ to a predicted output $I(x)$ without running $M$. It is
written in a small language of priced operations ([`agent_docs/ENTRY.md`](agent_docs/ENTRY.md)), in
which $M$ itself is written too, so that both are priced by the same list. We may think of an
interpretation claim as a triple $(I, S, p)$: the interpretation $I$; the region $S$, the inputs it
is claimed for, given by a test on the input in the same language; and the claimed fidelity $p$.
In the code the pair $(S, I)$ is called an entry, and $p$ is handed in with it. A claim is judged
by three metrics:

- **Coverage** $\mu = P(x \in S)$, for $x$ drawn from $D$: how much of the data set the claim
  covers.
- **Fidelity** $q = P(I(x) = M(x) \mid x \in S)$: how often, on the region, the interpretation
  gives the network's output.
- **Cost** $c$, the length of the computation: the operations the claim spends (the interpretation
  on $S$, the region test on every input, and whatever it computes once from the weights), per
  input of $S$, in forward passes of $M$. The network itself has $c = 1$.

The verifier measures all three on inputs it draws from $D$, or on all of them when they are few.
Nothing is proved: a claim is checked by measuring it.

## Reward function

## Agentic setup / Auto-research

![The agentic flow of level 1](docs/figures/level1-flow.png)

## More implementation details

### Coverage, fidelity and cost, as the verifier computes them

An entry is JSON data: a list of nodes, each one operation on earlier nodes, whose sources are the
input's tokens, the network's weights by name, constants and ranges. Two nodes are its results:
`rule`, the region test, one boolean per input (absent: every input is in the region), and
`predict`, the interpretation, one whole number per input. The agent writes entries in Python with a
builder (`lv1.Graph`), but nothing of an entry is ever run as code: the verifier interprets the JSON
(`level1/lang.py`, `Evaluator`) and counts every operation by the price list `prices-v1`. An
elementwise result costs 1, a matrix product 1 per multiply-add, a reduction over n entries n - 1,
a sort n * ceil(log2 n), reading a number by index 1, and moving data (reshape, slice, concat) 0.
`cond(c, x, y)` computes only the chosen side and costs 1 on top of it; `where(c, x, y)` computes
both. The network itself is written in the same language (`models/<name>/model.py`), and one forward
pass costs it F operations, the same on every input.

The verifier (`level1/verifier.py`) picks n inputs: every input of the data set when it can be
listed (lin-a's 4,096 pairs), otherwise a sample drawn from D (16,384 inputs in the pilots), afresh
for each of the agent's submissions, from a seed nobody chose. It runs the network on them for its
outputs M(x), then the entry, with `lang.measure`:

```python
# level1/lang.py, measure(): one pass over the n inputs
for x, truth in zip(inputs, model_answers):     # model_answers: the network's outputs M(x)
    inside, answer, c_rule, c_pred = ev.run(x)  # the region test, then the interpretation if inside
    rule_ops += c_rule                          # the region test is paid for on every input
    if inside:
        inside_n += 1
        predict_ops += c_pred                   # the interpretation only on the region
        agree += int(answer == int(truth))
coverage = inside_n / n
fidelity = agree / inside_n
cost = (predict_ops + rule_ops + ev.build_cost) / (inside_n * model_ops)
```

- **Coverage** is `n_in_region / n_inputs`, the share of the inputs scored on which the region test
  is true.
- **Fidelity** is `n_agree / n_in_region`, the share of the region on which the interpretation's
  output equals the network's output exactly: agreement of outputs, not of distributions. On a
  sample the verifier also records `fidelity_low`, the lower end of the 95% Wilson interval for
  `n_agree` successes in `n_in_region` draws (`lang.wilson_lower`, z = 1.96), and the reward uses it,
  so that a lucky sample does not pay. When every input is checked, `fidelity_low` is the fidelity.
- **Cost** is everything the entry computed, per input of its region, in forward passes. A node that
  does not depend on the input is computed once, the first time it is needed, and counted once, in
  `build_ops`. The rest is counted per input: the region test on every input, the interpretation on
  the inputs of the region, and a value the region test already computed for an input is not counted
  again. The receipt keeps three parts, `r = predict_ops / (n_in_region * F)`,
  `r_rule = rule_ops / (n_inputs * F)` and `b = build_ops / (n_inputs * F)`, so that
  `cost = r + (r_rule + b) / coverage`. The build is spread over the inputs scored.

Every receipt keeps these counts and parts, so a run can be scored again under another reward
without running anything. With the entry the agent also states a claimed fidelity; the receipt
records it and the overclaim, the claim minus the measured fidelity when positive. An entry may have
at most 400 nodes and 256 numbers in constants, so a table of the network's outputs cannot be
written into it; a table computed from the weights in the build is allowed, and paid for there. The
verifier gives an entry 120 seconds on its inputs.

### Checking what the agent did

Nothing the agent says is taken on trust: what it did is recorded, fenced in and checked.

- **Before any model is asked**, the hygiene check (`level1/hygiene.py`) searches everything the
  agent can reach (the prompt, the tools' descriptions, the files in its sandbox, names) for the
  words in `withheld/<model>/task.yaml` that would say what the network does, and stops the run on a
  hit.
- **Everything is recorded as it happens**: each round's prompt and the agent's answer, every tool
  call with its result, every program it ran with what it printed, the CLI's whole stream (the
  agent's text between its tool calls included) and every receipt (see "What a run writes"). Each
  round ends with notes under fixed headings: what the network does and on which inputs, how that
  was found (the outputs, the weights, or both), the best entry and where its saving comes from,
  the claims, and what to try next. The next round knows only these notes and the receipts.
- **The agent's code runs in a sandbox.** A guard inside Python refuses files outside the
  program's folder, sockets, other processes, signals and native code, and writes every refusal to
  the program's audit record. The guard shares a process with the code it watches, so for a run that
  counts the system's own sandbox is the wall: `seatbelt` on a Mac, a container or Landlock on
  Linux. Every run starts with a probe that tries real escapes, with and without the guard (reading
  the verifier, the second network's weights and the withheld task file in several ways, writing
  outside the folder, opening connections), and judges them by evidence outside the sandbox
  (`docs/SANDBOX.md`).
- **The report audits every round**: how many programs ran and for how long, every attempt the
  guard refused, every program whose text reaches for files or the system (`open(`, `os.`,
  `subprocess`, sockets, `eval`, the word withheld and others, `report.REACHES`), and every program
  whose audit record is incomplete.
- **Entries are data, not code.** The verifier interprets them, so an entry can do no work that is
  not counted and can read nothing but the input and the weights of the network it is scored on. The
  locked entry is the submission byte for byte; its hash is checked.
- **Claims are set against measurements.** With every entry the agent states the fidelity it
  expects on this network and on a second network; both are kept next to what is measured. Every
  submission is also scored on the second network after its round, on one fixed sample, and the
  agent never sees those scores (`seed_scores.jsonl`): they show round by round whether its entries
  drift towards the first network's quirks.

In the nine runs so far the guard refused nothing, the probe got nothing out of the seatbelt
sandbox, and no claimed fidelity was above the measured one by more than 0.0013. Two early pilots
ran in process mode, lin-b with the guard and lin-a-1to4 before the guard existed; lin-a-1to4's 52
programs were read by hand. The notes are the agent's own words: the measurements check the entries
built on them, not the words.

### The baseline

Much of what an entry saves can come from the algebra of the architecture alone: the forward pass
repeats the same products for the same token in the same slot, and tables computed once from the
weights remove that on any network of the design. The baseline measures that part, so that a run can
be read against it (`level1/baseline.py`, `docs/DESIGN.md` section 10).

- **Baseline models.** `lin-e`, `lin-f` and `att-e` have the design, sizes and data set of lin-c,
  lin-d and att-c (and so also serve lin-a-1to4, lin-b and att-d), and `rnn-d`, `rnn-e` and `rnn-f`
  those of rnn-a, rnn-b and rnn-c. Their networks have nothing in them for an agent to find; what
  they are is in `withheld/<name>/`. The first network is for the rounds, the second for the seed
  check, the others for freezing.
- **A baseline run** is a usual run on a baseline model, under the same conditions as the runs it
  serves (prompt, rounds, budgets, entry limits, verifier, reward and prices: `baseline.CONDITIONS`),
  with one more paragraph in its prompt (`prompt.EXACT_NOTE`, run config `exact_on_any_weights`):
  its entries will also be run on other networks of the design, and count only if they are exact on
  all of them. A test checks that its first prompt and a twin run's are the same, byte for byte, but
  for that paragraph.
- **Freezing.** When the baseline run ends, its submissions that gave the second network's output on
  every input of their region are taken in the order of their reward there, one per different entry,
  and each is scored on every other network of the baseline model, on one fresh sample. The first
  that gives each of them the network's output on every input of its region is frozen
  (`baseline.json`, `baseline_entry.json`), and the baseline is established. If none of the first
  three does, it is not established, and no run is read against it.
- **Reading a run against it.** The frozen entry is scored on the run's network and on its second
  network, on the very inputs of the run's own scorings of its locked entry (the fresh sample and
  the seed check). With R0 and R1 the locked entry's rewards there and B0 and B1 the frozen
  entry's, the report splits R0 into three parts:

      R0 = B0 + (R1 - B1) + [(R0 - B0) - (R1 - B1)]

  the baseline on the run's network, the excess kept on the held-out network, and the gap between
  the two networks. Negative values are kept. It is accounting, not a causal split: the excess can
  hold generic tricks the baseline run did not find, and the gap can be sampling noise.
- **Why exactness.** The first baseline runs (pilot-att-e and pilot-lin-e), made without that
  paragraph, fitted their entries to their own networks: pilot-lin-e's were right on 74% to 81% of
  the other networks' inputs, and pilot-att-e's on about a third of a trained network's. A run's
  score read against them said nothing, so they serve no run.

## Results

The six pilots run so far (11 and 12 November 2026), all on trained networks, and the three
baseline runs they are read against (13 November). Every run used `claude-fable-5-1` at effort
high, at most 5 rounds of at most 4 submissions, 16,384 inputs per scoring, and the reward r_alpha
with alpha = 10.

**The reward scale.** 0 is the network itself, and 1 an entry that is always right and costs
nothing; on a sample of 16,384 inputs the most an entry can score is about 0.998, because the reward
uses the lower end of the fidelity's interval. Below 0, an entry's wrong answers cost more than it
saves. With the network run wherever an entry does not apply, and each wrong answer counted as ten
forward passes, an entry of reward R costs 1 - R of running the network on everything: 0.83 is about
a sixfold saving, 0.99 about a hundredfold.

### The runs

| Run | Network | Inputs | F | Locked | Fresh sample | Second seed | Generic reference |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| lin-a-1to4 | linear, width 8 | 1 to 4 tokens | 671 | 0.8314 | 0.8310 | 0.8298 | 0.519 |
| lin-c | linear, width 8 | 1 to 4 tokens | 671 | 0.8898 | 0.8900 | 0.8893 | 0.519 |
| lin-b | linear, width 8 | 1 to 8 tokens | 703 | 0.9180 | 0.9129 | 0.8027 | 0.176 |
| lin-d | linear, width 8 | 1 to 8 tokens | 703 | 0.9257 | 0.9128 | 0.8301 | 0.176 |
| att-c | transformer, 1 layer, 1 head, width 32 | 1 to 4 tokens | 12,987 | 0.9889 | 0.9887 | 0.9905 | 0.955 |
| att-d | transformer, 1 layer, 1 head, width 32 | 1 to 4 tokens | 12,987 | 0.9884 | 0.9884 | 0.9871 | 0.955 |

- **F**: the operations of one forward pass, by the price list.
- **Locked**: the reward of the run's best submission on its own network, as scored when it was
  submitted. It is the best of several scorings, so it leans high.
- **Fresh sample**: the locked entry scored again on the same network, on a fresh sample.
- **Second seed**: the locked entry, unchanged, on a network of the same design and training with
  another seed, which the agent never sees.
- **Generic reference**: an entry written by hand from the algebra of the design alone, exact for any
  weights (`contraction` in `withheld/lin-e/` and `withheld/lin-f/`, `tables` in `withheld/att-e/`),
  scored on the run's networks on 4 fixed samples of 16,384 inputs and averaged. It scores the same
  on both seeds. The baseline runs below found more than these on every design.

### Against their baselines

Each baseline run froze an entry that gave the network's output on every checked input of all four
of its model's other networks, so all three baselines are established:

| Baseline run | Serves | Frozen entry | On its second network | On its other four networks |
| --- | --- | --- | ---: | --- |
| pilot-lin-e-exact | lin-a-1to4, lin-c | r5s3 (ts4) | 0.8388 | 0.838 to 0.842, exact on each |
| pilot-lin-f-exact | lin-b, lin-d | r3s1 (tri_v3) | 0.6888 | 0.687, exact on each |
| pilot-att-e-exact | att-c, att-d | r3s2 (v3g) | 0.9735 | 0.977 to 0.990, exact on each |

Each run read against its baseline, every number on the run's own inputs: R0 is the locked entry on
its fresh sample, R1 in the seed check, and B0 and B1 the frozen entry on the same inputs. The three
parts add up to R0. In brackets, the range when each fidelity moves across its 95% interval, as the
report gives it.

| Run | R0 | Baseline B0 | Kept on the held-out network, R1 - B1 | Gap, (R0 - B0) - (R1 - B1) |
| --- | ---: | ---: | --- | --- |
| lin-a-1to4 | 0.8310 | 0.8565 | -0.026 (-0.028 to -0.023) | +0.001 (-0.005 to +0.005) |
| lin-c | 0.8900 | 0.8601 | +0.029 (+0.027 to +0.031) | +0.001 (-0.004 to +0.006) |
| lin-b | 0.9129 | 0.6894 | +0.113 (+0.110 to +0.144) | +0.111 (+0.077 to +0.122) |
| lin-d | 0.9128 | 0.6861 | +0.139 (+0.136 to +0.168) | +0.088 (+0.056 to +0.103) |
| att-c | 0.9887 | 0.9573 | +0.033 (+0.031 to +0.036) | -0.002 (-0.006 to +0.003) |
| att-d | 0.9884 | 0.9575 | +0.030 (+0.028 to +0.032) | +0.001 (-0.004 to +0.006) |

The kept part is the run's lead over the baseline on a network its agent never saw. The gap is the
part of the lead that holds on the run's own network and not on the held-out one: whatever was
fitted to the first network, or sampling noise.

### The Pareto frontier

The report also marks each run's Pareto frontier: the submissions that no other submission of the
run dominates, that is, none is at least as good on all three of right share (up), wrong share
(down) and cost (down) and better on one. It needs no alpha. Every submission is scored on its own
sample, so an entry submitted twice can be on it twice. The last column is each submission on the
second network, all of a run's on one fixed sample; the seed check above uses a sample of its own,
so for the locked entry the two differ a little.

| Run | Submission (entry) | Right share | Wrong share | Cost | Reward | On the second network |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| lin-a-1to4 | r4s2 (tri_v2) | 0.99994 | 0.00006 | 0.1660 | 0.8305 | 0.8310 |
| lin-a-1to4 | r5s3 (tri_v3), locked | 1.00000 | 0.00000 | 0.1663 | 0.8314 | 0.8314 |
| lin-c | r5s4 (r5_hedge60), locked | 1.00000 | 0.00000 | 0.1079 | 0.8898 | 0.8861 |
| lin-b | r4s4 (r4d), locked | 0.99951 | 0.00049 | 0.0723 | 0.9180 | 0.8118 |
| lin-b | r5s1 (r5b) | 0.99915 | 0.00085 | 0.0714 | 0.9143 | 0.8126 |
| lin-b | r5s3 (r5b) | 0.99939 | 0.00061 | 0.0722 | 0.9166 | 0.8126 |
| lin-b | r5s4 (r5c) | 0.99866 | 0.00134 | 0.0693 | 0.9103 | 0.8113 |
| lin-d | r3s1 (v3b) | 0.99866 | 0.00134 | 0.0624 | 0.9173 | 0.8404 |
| lin-d | r4s3 (v4a) | 0.99927 | 0.00073 | 0.0641 | 0.9231 | 0.8434 |
| lin-d | r4s4 (v4a), locked | 0.99951 | 0.00049 | 0.0646 | 0.9257 | 0.8434 |
| lin-d | r5s1 (best) | 0.99933 | 0.00067 | 0.0642 | 0.9238 | 0.8434 |
| lin-d | r5s2 (best) | 0.99908 | 0.00092 | 0.0635 | 0.9214 | 0.8434 |
| att-c | r5s1 (v14), locked | 1.00000 | 0.00000 | 0.0088 | 0.9889 | 0.9906 |
| att-d | r3s3 (v4_perm), locked | 1.00000 | 0.00000 | 0.0092 | 0.9884 | 0.9870 |

On att-c, att-d and lin-c one exact entry is also the cheapest of the run, so it alone is the
frontier and alpha decides nothing; on lin-a-1to4 one more entry is a hair cheaper with one wrong
answer in 16,384. On lin-b and lin-d the frontier is a trade between cost and wrong answers:
cheaper entries make a few more mistakes, and alpha = 10 picks among them. There the spread between
two scorings of one entry is as large as the differences between entries. Each baseline run's
frontier is a single exact entry, its locked one.

### What they show so far

- Every locked entry beats the network by a wide margin, and the runs on 4 slots keep it on the
  second seed: there they are within 0.002 of the fresh sample.
- On the 4-slot linear design almost all of the score is generic. An agent that never saw a trained
  network found an exact entry worth 0.857 on lin-a-1to4's own network, more than that run's 0.831:
  a search that bounds blocks of outputs and computes exact logits only where one can still win,
  where the run used tables of pair sums. lin-c's run keeps +0.03 over it.
- On the transformers the generic part is 0.957, close to the hand-written tables (0.955), and the
  runs keep +0.03 over it on networks their agents never saw, with no gap.
- On the 8-slot linear design the runs keep +0.11 and +0.14 over a weaker baseline (0.69; its agent
  found much less than the 4-slot one did), and lose 0.11 and 0.09 between their two networks.
  Their entries compute exact logits only for the outputs that can still win on long
  inputs, and which outputs those are was read off the first network.
- A frozen entry can score more on a trained network than on its own model's (lin-e's: 0.857 on
  lin-a-1to4, 0.839 on its second network): it is exact for any weights, but what it costs depends
  on them.
- The agents resubmit the same entry to draw a luckier sample, so the fresh sample is the number to
  compare.

Not run yet: the pilots on pairs (lin-a, att-a, att-b) and on the recurrent networks.

## Getting started

### Install

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest                      # about a minute; no model is called
    python -m level1.sandbox info         # which sandbox modes work on this machine

Use a virtual environment of this repository's own. Outside a container, the hygiene check also
reads the path of the Python that runs the agent's code, because a traceback shows that path to the
agent: an environment that lives in a folder named after the proof pipeline is refused.

Rounds run under your own Claude Code login, so the CLI (`claude`) has to be installed and signed
in. To bill an API key instead, see "On an API key" below.

### Run a pilot

Each model has a run config, `config/pilot/<name>.yaml`, and a small smoke config next to it,
`config/pilot/<name>-smoke.yaml`. First a dry run, with scripted rounds in place of a model (two
minutes):

    python -m level1.loop run --config config/pilot/lin-a-1to4.yaml --fake tests/fake_run_1to4 --run-id dry-run

Then the smoke run, one short round, to see that the login, the tools, the sandbox and the verifier
work together on this machine:

    python -m level1.loop run --config config/pilot/lin-a-1to4-smoke.yaml

Then the pilot proper, at most 5 rounds of up to 90 minutes each:

    python -m level1.loop run --config config/pilot/lin-a-1to4.yaml

If it stops (a usage limit, a dropped connection, a closed laptop), continue where it stopped:

    python -m level1.loop resume --run ~/mech-interp-level1-runs/pilot-lin-a-1to4

The pilot on pairs runs the same way with `lin-a-smoke.yaml`, `lin-a.yaml` and `tests/fake_run`, and
`tests/fake_run_any` is one scripted round that fits every model. Run folders live outside the
repository, in `~/mech-interp-level1-runs/<run_id>/` by default. On a Mac, run a long pilot under
`caffeinate -is`, with the lid open and on power: a session whose machine sleeps loses its prompt
cache and pays to write it again.

### With its baseline

A run is read against the baseline of its shape when one is frozen (`docs/DESIGN.md`, section 10).
Three ways to start:

    python -m level1.loop run --config config/pilot/att-c.yaml                   # the run only
    python -m level1.loop baseline --config config/pilot/att-c.yaml              # its baseline only
    python -m level1.loop run --config config/pilot/att-c.yaml --with-baseline   # both, side by side

A baseline is made once for a shape and its conditions and serves every run that matches; a
finished run gets one with `python -m level1.loop baseline --run <run folder>`. Each command takes
the same `--sandbox`, `--auth` and `--budget-usd` as `run`. The baseline models' own pilot configs
make runs named `pilot-<name>-exact`; baseline runs made before their prompt asked for an exact
entry serve no run.

### On an API key

A run can be billed to an API key instead of your login (`auth: api_key` in the run config, or
`--auth api_key` with any config). Put the key in `CPL_ANTHROPIC_KEY`, never in `ANTHROPIC_API_KEY`,
for example by loading the file you keep it in; `CPL_ANTHROPIC_WORKSPACE` takes a workspace ID if
the key is not scoped to one workspace. Then

    python -m level1.loop run --config config/pilot/lin-b-api.yaml --sandbox seatbelt

Before the first round the key is checked against the API's list of models, which costs nothing.
The run has a budget, by Claude Code's own count: `budget.run_usd` for the run (60 dollars) and
`budget.round_usd` for one round (20). Every session starts with what is left as its cap. A round
that reaches its cap ends with what it submitted; the rounds end when the run's budget is spent.
Claude Code checks its spending after every answer, so a session can go over its cap by one answer.
The run folder's `spend.jsonl` has every session's cap and cost. A session that reported no cost
counts at its cap; one the loop never saw end (the loop was killed) counts in the run's total only,
so its round gets its whole budget again when resumed. To go on with more:

    python -m level1.loop resume --run ~/mech-interp-level1-runs/pilot-lin-b-api --budget-usd 90

The CLI's `--bare` mode, which this uses, reads no login, so an API run cannot draw on a
subscription, and a login run cannot bill a key: the CLI's first event says what it billed, and a
session that does not fit stops the run.

### The sandbox

In every mode a guard inside Python refuses the agent's code files outside its folder, the network
and other programs, and records every attempt; the report lists them. The guard shares a process
with the code it watches, so for a run that is meant to count add `--sandbox hardened` (Docker or
Linux) or, on a Mac, `--sandbox seatbelt`. Try a mode first with the probe, which tries to escape
it and reports (`docs/SANDBOX.md`):

    python -m level1.sandbox probe --model models/lin-a --mode seatbelt

### Other commands

    python -m level1.loop finish --run <run folder>              # lock, seed check and report again; scores
                                                                 # any submission not yet on the second network
    python -m level1.loop rescore --run <run folder> --alpha 1   # the records under another reward
    python -m level1.report --model models/lin-a --alpha 1       # the reference entries under a reward
    python -m level1.report --model models/lin-a-1to4 --sample-size 16384
    python -m level1.verifier --model models/lin-a --network lin-a-s0 --entry entry.json
    python -m level1.verifier --model models/lin-a --network lin-a-s0 --entry entry.json --check sample
    python -m level1.sandbox build                               # the container image, if Docker runs
    python -m level1.sandbox probe --model models/lin-a          # try to escape the sandbox, and report

### What a run writes

Everything is written as it happens, so a run can stop and resume, and can be read afterwards:

- every prompt (`round_k/prompt.md`) and the agent's answer (`response.md`);
- every tool call with its result (`session/tools.jsonl`);
- every program the agent ran, in full, with what it printed (`session/python/NNN.py`, `.txt`),
  and what the guard refused it (`NNN.audit.json`);
- everything the CLI emitted, the agent's text between its tool calls included
  (`round_k/stream_NN.jsonl`);
- every submission's receipt, with the parts measured;
- every scored submission on the second network (`seed_scores.jsonl`);
- billed to an API key, every session's cap and cost (`spend.jsonl`);
- the report: `report.md` and `report.json`.

### Repository layout

    level1/            the pipeline
      lang.py          the entry language: operations, prices, the interpreter, the builder, data sets
      verifier.py      measures an entry; the only judge
      rewards.py       reward functions
      networks.py      model folders: the networks, the network written as an entry, what the agent is given
      prompt.py        a round's prompt
      session.py       one agent session: the two tools, budgets, log
      mcp_server.py    the session's tools as an MCP server, for the Claude Code CLI
      loop.py          rounds, lock, seed check; the command line
      baseline.py      the baseline of a run's shape: found, run, frozen, and a run read against it
      report.py        the report; the reference entries under any reward
      hygiene.py       the check that the agent is not told what the model does
      backends/        who answers a round: claude_code, fake
      budget.py        what a run billed to an API key may spend, and what it has spent
      run_config.py    run configs: defaults, then what a config extends, then the config
      sandbox.py, runner.py, harness.py    where the agent's code runs
      util.py          small helpers
    models/lin-a/           a linear network on pairs of tokens, two seeds
    models/lin-a-1to4/      the same two networks, asked about inputs of 1 to 4 tokens
    models/lin-b/           a network of the same design trained afresh, asked about inputs of 1 to 8 tokens
    models/lin-c/, lin-d/   the designs of lin-a and lin-b trained afresh on another task
    models/att-a/ .. att-d/   one-layer, one-head transformers: two with 2 slots, two with 4
    models/lin-e/, lin-f/, att-e/   baseline models of the shapes of lin-c, lin-d and att-c
    models/rnn-a/ .. rnn-c/   one-layer ReLU recurrent networks on 2, 3 and 10 real numbers, five seeds each
    models/rnn-d/ .. rnn-f/   their baseline models
    withheld/<model>/       what it was trained to do, and reference entries (never shown to the agent)
    withheld/rnn/           where the recurrent networks come from: the original files and the converter
    agent_docs/        the entry guide the agent reads
    prompts/           the round's prompt template
    config/            defaults, the pilots' run configs and their small smoke runs
    sandbox/           the container image (Dockerfile)
    tests/             the tests, with each known way to game the score as a test
    docs/              DESIGN.md, REWARD.md, SANDBOX.md; figures/: the flow diagram and its source

`CLAUDE.md` has the working rules for changing this repository.
