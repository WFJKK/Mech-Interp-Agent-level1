# Mech-Interp-Agent-level1

An agent is given a neural network, its definition and its weights, and is not told what the
network was trained to do. It writes an **interpretation**: a program that predicts the network's
output from the input without running the network. It hands it in as an **entry**: the
interpretation together with the **region** of inputs it applies to. A **verifier** then measures
the entry: how much of the data set it covers, how often it gives the network's output there, and
what it costs.

This is level 1 of the three levels for loosening compact proofs: the claim is checked by
measuring it on inputs the verifier picks, and nothing is proved. Levels 2 (an adversary with a
budget) and 3 (an argument that convinces) are not in this repository.

Status, 12 November 2026: nine pilots are set up (see "The pilots"). The pilot on 1 to 4 tokens
has been run once, with a smoke run before it; lin-b has had its smoke run, and its pilot is under
way. Rounds run under a Claude Code login or billed to an API key ("On an API key").

## How a run goes

1. **The network.** `models/<name>/` holds the weights, `model.py`, which writes the network in the
   entry language, and the data set the network is asked about. That fixes what one forward pass
   costs (655, 671 and 703 operations for the three pilots).
2. **The rounds.** The agent explores the network with `run_python` (a sandbox with the weights and
   the entry language) and hands entries to the verifier with `submit_entry`. Each round is a fresh
   session; its closing notes and its scored entries go to the next round.
3. **The lock.** When the rounds end, the submission with the highest reward is locked as it was
   submitted. If the verifier samples, the locked entry is scored once more on a fresh sample.
4. **The seed check.** The locked entry is run, unchanged, on a second network of the same sizes
   and training with another seed. No agent takes part. Every other submission is also scored
   there after its round, for the record only: the agent never sees those scores and they choose
   nothing (`seed_scores.jsonl`, and a column of the report).
5. **The report.** `report.md` and `report.json`: the locked entry on both networks, every
   submission, the withheld reference entries for comparison, what the agent ran, and its notes.

## What is measured, and the reward

| Word | What it is |
| --- | --- |
| coverage | the share of the data set in the entry's region |
| fidelity | the share of the region on which the interpretation gives the network's output |
| cost | everything the entry computes, per input of its region, in forward passes |

The cost has three parts, kept apart in every receipt: the interpretation on the region (`r`), the
region test on every input (`r_rule`) and whatever is computed once ahead (the build, `b`).

The default reward, with alpha = 10:

    reward = coverage * [ (1 - cost) - alpha * (1 - fidelity) ]

Read it for one input of the region: the entry saves (1 - cost) of a forward pass, and a wrong
answer is priced at alpha forward passes. The network itself scores 0, and nothing scores above 1.
Rewards are functions in `level1/rewards.py`, picked by the run config; the parts are kept, so a
finished run can be scored again under another reward. `docs/REWARD.md` has the details and the
known ways to game a reward.

## The data set, and how the verifier checks

A model's `config.yaml` says what its data set is (`data_set`), and the agent is told the same in
its prompt. The run config says how the verifier checks an entry (`verifier.check`):

- `all`: on every input of the data set;
- `sample`: on a fresh sample for every scoring (`verifier.sample_size` inputs). The reward then
  uses the lower end of the fidelity's 95% confidence interval;
- `auto`, the default: `all` when the data set can be listed and is small enough, else `sample`.

## The pilots

| Run config | Network | Data set | Checked on |
| --- | --- | --- | --- |
| `config/pilot/lin-a.yaml` | `models/lin-a` | every pair of tokens: 4,096 inputs | every input |
| `config/pilot/lin-a-1to4.yaml` | `models/lin-a-1to4` | inputs of 1 to 4 tokens, each length a quarter: about 17 million inputs | 16,384 sampled per scoring |
| `config/pilot/lin-b.yaml` | `models/lin-b` | inputs of 1 to 8 tokens, each length an eighth: about 2.8 x 10^14 inputs | 16,384 sampled per scoring |
| `config/pilot/lin-c.yaml` | `models/lin-c` | as lin-a-1to4 | 16,384 sampled per scoring |
| `config/pilot/lin-d.yaml` | `models/lin-d` | as lin-b | 16,384 sampled per scoring |
| `config/pilot/att-a.yaml` | `models/att-a` | inputs of 1 or 2 tokens, each half: 4,160 inputs | 16,384 sampled per scoring |
| `config/pilot/att-b.yaml` | `models/att-b` | as att-a | 16,384 sampled per scoring |
| `config/pilot/att-c.yaml` | `models/att-c` | inputs of 1 to 4 tokens, each length a quarter: about 17 million inputs | 16,384 sampled per scoring |
| `config/pilot/att-d.yaml` | `models/att-d` | as att-c | 16,384 sampled per scoring |
| `config/pilot/lin-e.yaml` | `models/lin-e` | as lin-c (baseline of lin-c's shape) | 16,384 sampled per scoring |
| `config/pilot/lin-f.yaml` | `models/lin-f` | as lin-d (baseline of lin-d's shape) | 16,384 sampled per scoring |
| `config/pilot/att-e.yaml` | `models/att-e` | as att-c (baseline of att-c's shape) | 16,384 sampled per scoring |

Every model up to att-d has two trained networks (two seeds): the rounds work on s0, the seed check runs the
locked entry on s1. The first two pilots use the same two networks. `lin-b` is a network of the
same design trained afresh (`withheld/lin-b/train.py`, resumable, a minute or two per 10,000 steps
on a laptop's CPU), and `lin-c` and `lin-d` are the designs of lin-a and lin-b trained afresh on
another task (`withheld/lin-c/train.py`, `withheld/lin-d/train.py`).

`att-a` to `att-d` are transformers with one layer, one attention head and nothing else (no MLP, no
layer norm, no biases), 64 tokens, width 32. The network is read at the slot of the last token
given, attending over the tokens given, so it has an output for inputs that leave slots empty.
Three come from the proof pipeline's zoo; `att-d` was trained here by the same recipe
(`withheld/att-d/train.py`, numpy, half a minute per network). A forward pass costs 8,621
operations at 2 slots and 12,987 at 4, against about 700 for the linear networks.

In all but the pilot on pairs the networks are asked about more inputs than they were trained on,
and the agent is not told which ones those were: finding where the network's behaviour can be
described cheaply is part of the work. On lin-b and lin-d there are too many inputs for the agent,
or anyone, to check them all. On att-a and att-b a short rule gives the network's output on every
input, so their pilots check that a run finds it rather than pose a problem (a reference entry
scores 0.997). Each pilot has a small smoke config next to it (`<name>-smoke.yaml`), and
`lin-b-api.yaml` is the lin-b pilot billed to an API key; `--auth api_key` bills any of them.

`lin-e`, `lin-f` and `att-e` are baseline models. Each has the design, sizes and data set of the
models whose shape it shares (lin-e: lin-c and lin-a-1to4; lin-f: lin-d and lin-b; att-e: att-c
and att-d), and six networks with nothing in them for an agent to find; what they are is in
`withheld/<name>/`. A run on one gives the agent exactly what a run on its twin gives it, the same
prompt, rounds, budgets and reward, so what it achieves there is generic: the baseline that runs of
that shape are read against. The rounds use s0 and the seed check s1; s2 to s5 are for checking an
entry again later. A hand-written reference, `tables` on att-e, scores 0.955 on every network of
att-c's shape, trained or not.

## Quick start

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest                      # about a minute; no model is called
    python -m level1.sandbox info         # which sandbox modes work on this machine

Use a virtual environment of this repository's own. Outside a container, the hygiene check also
reads the path of the Python that runs the agent's code, because a traceback shows that path to the
agent: an environment that lives in a folder named after the proof pipeline is refused.

A dry run with scripted rounds in place of a model:

    python -m level1.loop run --config config/pilot/lin-a-1to4.yaml --fake tests/fake_run_1to4 --run-id dry-run   # two minutes

A first contact with a real model, kept small: one short round, to see that the login, the tools,
the sandbox and the verifier work together on this machine. It needs the Claude Code CLI (`claude`)
installed and signed in; rounds run under your own login, and no API key is used.

    python -m level1.loop run --config config/pilot/lin-a-1to4-smoke.yaml

Then the pilot proper, at most 5 rounds of up to 90 minutes each:

    python -m level1.loop run --config config/pilot/lin-a-1to4.yaml

If it stops (a usage limit, a dropped connection, a closed laptop), continue with

    python -m level1.loop resume --run ~/mech-interp-level1-runs/pilot-lin-a-1to4

The pilot on pairs runs the same way with `lin-a-smoke.yaml`, `lin-a.yaml` and `tests/fake_run`. A
dry run of any pilot can use `tests/fake_run_any`, one scripted round that fits every model.

### On an API key

A run can be billed to an API key instead of your login (`auth: api_key`). Put the key in
`CPL_ANTHROPIC_KEY`, never in `ANTHROPIC_API_KEY`, for example by loading the file you keep it in;
`CPL_ANTHROPIC_WORKSPACE` takes a workspace ID if the key is not scoped to one workspace. Then

    python -m level1.loop run --config config/pilot/lin-b-api.yaml --sandbox seatbelt

Before the first round the key is checked against the API's list of models, which costs nothing.
The run has a budget, by Claude Code's own count: `budget.run_usd` for the run (60 dollars) and
`budget.round_usd` for one round (20). Every session starts with what is left as its cap. A round
that reaches its cap ends with what it submitted; the rounds end when the run's budget is spent.
Claude Code checks its spending after every answer, so a session can go over its cap by one answer.
The run folder's `spend.jsonl` has every session's cap and cost; a session that reported no cost
(stopped at its time limit, or the loop was killed) counts at its cap. To go on with more:

    python -m level1.loop resume --run ~/mech-interp-level1-runs/pilot-lin-b-api --budget-usd 90

`--auth api_key` and `--budget-usd` work on `run` with any config. The CLI's `--bare` mode, which
this uses, reads no login, so an API run cannot draw on a subscription, and a login run cannot bill
a key: the CLI's first event says what it billed, and a session that does not fit stops the run.

In every mode a guard inside Python refuses the agent's code files outside its folder, the network
and other programs, and records every attempt; the report lists them. The guard shares a process
with the code it watches, so for a run that is meant to count add `--sandbox hardened` (Docker or
Linux) or, on a Mac, `--sandbox seatbelt`, which still needs its first test on a Mac:
`python -m level1.sandbox probe --model models/lin-a --mode seatbelt` (`docs/SANDBOX.md`).

Other commands:

    python -m level1.loop finish --run <run folder>              # lock, seed check and report again; scores
                                                                 # any submission not yet on the second network
    python -m level1.loop rescore --run <run folder> --alpha 1   # the records under another reward
    python -m level1.report --model models/lin-a --alpha 1       # the reference entries under a reward
    python -m level1.report --model models/lin-a-1to4 --sample-size 16384
    python -m level1.verifier --model models/lin-a --network lin-a-s0 --entry entry.json
    python -m level1.verifier --model models/lin-a --network lin-a-s0 --entry entry.json --check sample
    python -m level1.sandbox build                               # the container image, if Docker runs
    python -m level1.sandbox probe --model models/lin-a          # try to escape the sandbox, and report

Run folders live outside the repository, in `~/mech-interp-level1-runs/<run_id>/` by default.

## What a run keeps

Everything is written as it happens, so a run can stop and resume, and can be read afterwards:

- every prompt (`round_k/prompt.md`) and the agent's answer (`response.md`);
- every tool call with its result (`session/tools.jsonl`);
- every program the agent ran, in full, with what it printed (`session/python/NNN.py`, `.txt`),
  and what the guard refused it (`NNN.audit.json`);
- everything the CLI emitted, the agent's text between its tool calls included
  (`round_k/stream_NN.jsonl`);
- every submission's receipt, with the parts measured;
- billed to an API key, every session's cap and cost (`spend.jsonl`);
- every scored submission on the second network (`seed_scores.jsonl`).

The report counts the programs and the sandbox time per round, lists every attempt the guard
refused, and lists the programs whose text reaches for files or the system.

## The rule the pipeline is built around

The agent is never told what the model does. Nothing it can reach says so: the prompt, the tools'
descriptions, the files in its sandbox, names, the entry guide. What a network was trained to do
is in `withheld/<model>/`, which only the verifier's side reads. A check searches everything the
agent can reach for the forbidden strings before any model is asked, and stops the run on a hit.
The sandbox enforces the rest; see `docs/SANDBOX.md` for when it can and when it cannot.

## Layout

    level1/          the pipeline
      lang.py        the entry language: operations, prices, the interpreter, the builder, data sets
      verifier.py    measures an entry; the only judge
      rewards.py     reward functions
      session.py     one agent session: the two tools, budgets, log
      loop.py        rounds, lock, seed check; the command line
      report.py      the report; the reference entries under any reward
      hygiene.py     the check that the agent is not told what the model does
      backends/      who answers a round: claude_code, fake
      budget.py      what a run billed to an API key may spend, and what it has spent
      sandbox.py, runner.py, harness.py    where the agent's code runs
    models/lin-a/         a linear network on pairs of tokens, two seeds
    models/lin-a-1to4/    the same two networks, asked about inputs of 1 to 4 tokens
    models/lin-b/         a network of the same design trained afresh, asked about inputs of 1 to 8 tokens
    models/lin-c/, lin-d/ the designs of lin-a and lin-b trained afresh on another task
    models/att-a/ .. att-d/   one-layer, one-head transformers: two with 2 slots, two with 4
    models/lin-e/, lin-f/, att-e/   baseline models of the shapes of lin-c, lin-d and att-c
    withheld/<model>/     what it was trained to do, and reference entries (never shown to the agent)
    agent_docs/      the entry guide the agent reads
    prompts/         the round's prompt template
    config/          defaults, the pilots' run configs and their small smoke runs
    tests/           the tests, with each known way to game the score as a test
    docs/            DESIGN.md, REWARD.md, SANDBOX.md

`CLAUDE.md` has the working rules for changing this repository.
