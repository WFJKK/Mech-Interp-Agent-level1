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

## Results

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
