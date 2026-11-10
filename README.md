# Mech-Interp-Agent-level1

An agent is given a small neural network, its definition and its weights, and is not told what the
network was trained to do. It writes an **entry**: a small program that predicts the network's
output with fewer operations than the network itself needs, on a region of the inputs that it
chooses. A **verifier** then measures the entry: how much of the data set it covers, how often it
gives the network's output there, and what it costs.

This is level 1 of the three levels for loosening compact proofs: the claim is checked by
measuring it on inputs the verifier picks, and nothing is proved. Levels 2 (an adversary with a
budget) and 3 (an argument that convinces) are not in this repository.

Status, 10 November 2026: built and tested with scripted rounds only. No run with a model has been
made yet.

## How a run goes

1. **The network.** `models/<name>/` holds the weights and `model.py`, which writes the network in
   the entry language. That fixes what one forward pass costs (655 operations for the pilot).
2. **The rounds.** The agent explores the network with `run_python` (a sandbox with the weights and
   the entry language) and hands entries to the verifier with `submit_entry`. Each round is a fresh
   session; its closing notes and its scored entries go to the next round.
3. **The lock.** When the rounds end, the submission with the highest reward is locked as it was
   submitted.
4. **The seed check.** The locked entry is run, unchanged, on a second network of the same sizes
   and training with another seed. No agent takes part.
5. **The report.** `report.md` and `report.json`: the locked entry on both networks, every
   submission, the withheld reference entries for comparison, and the agent's notes.

## What is measured, and the reward

| Symbol | What it is |
| --- | --- |
| coverage | the share of the data set in the entry's region |
| fidelity | the share of the region on which the entry gives the network's output |
| r | operations of the prediction per input of the region, over one forward pass |
| r_rule | operations of the rule that decides the region, per input, over one forward pass |
| b | operations computed once ahead (the build), over a forward pass on every input |

The default reward, with alpha = 10:

    reward = coverage * (1 - r) - r_rule - b
             - alpha * coverage * (1 - fidelity)

The first line is the share of the work saved if the entry is used on its region and the network
elsewhere. The second prices a wrong answer at alpha forward passes. The network itself scores
exactly 0, and nothing scores above 1. Rewards are functions in `level1/rewards.py`, picked by the
run config; the parts are kept, so a finished run can be scored again under another reward.
`docs/REWARD.md` has the details and the known ways to game a reward.

## Quick start

    pip install -r requirements.txt
    python -m pytest                      # about a minute; no model is called
    python -m level1.sandbox info         # which sandbox modes work on this machine

A dry run with scripted rounds in place of a model:

    python -m level1.loop run --config config/pilot/lin-a.yaml --fake tests/fake_run --run-id dry-run

A first contact with a real model, kept small: one short round, to see that the login, the tools
and the sandbox work together on this machine. It needs the Claude Code CLI (`claude`) installed
and signed in; rounds run under your own login, and no API key is used.

    python -m level1.loop run --config config/pilot/lin-a-smoke.yaml

Then the pilot proper, at most 5 rounds of up to 90 minutes each:

    python -m level1.loop run --config config/pilot/lin-a.yaml

If it stops (a usage limit, a dropped connection, a closed laptop), continue with

    python -m level1.loop resume --run ~/mech-interp-level1-runs/pilot-lin-a

Add `--sandbox hardened` to a run that is meant to count: it then refuses to start unless the
agent's code can be kept away from the withheld folder (`docs/SANDBOX.md`).

Other commands:

    python -m level1.loop finish --run <run folder>              # lock, seed check and report again
    python -m level1.loop rescore --run <run folder> --alpha 1   # the records under another reward
    python -m level1.report --model models/lin-a --alpha 1       # the reference entries under a reward
    python -m level1.verifier --model models/lin-a --network lin-a-s0 --entry entry.json
    python -m level1.sandbox build                               # the container image, if Docker runs
    python -m level1.sandbox probe --model models/lin-a          # try to escape the sandbox, and report

Run folders live outside the repository, in `~/mech-interp-level1-runs/<run_id>/` by default.

## The rule the pipeline is built around

The agent is never told what the model does. Nothing it can reach says so: the prompt, the tools'
descriptions, the files in its sandbox, names, the entry guide. What a network was trained to do
is in `withheld/<model>/`, which only the verifier's side reads. A check searches everything the
agent can reach for the forbidden strings before any model is asked, and stops the run on a hit.
The sandbox enforces the rest; see `docs/SANDBOX.md` for when it can and when it cannot.

## Layout

    level1/          the pipeline
      lang.py        the entry language: operations, prices, the interpreter, the builder
      verifier.py    measures an entry; the only judge
      rewards.py     reward functions
      session.py     one agent session: the two tools, budgets, log
      loop.py        rounds, lock, seed check; the command line
      report.py      the report; the reference entries under any reward
      hygiene.py     the check that the agent is not told what the model does
      backends/      who answers a round: claude_code, fake
      sandbox.py, runner.py, harness.py    where the agent's code runs
    models/lin-a/    the pilot: a linear network on pairs of tokens, two seeds
    withheld/lin-a/  what it was trained to do, and reference entries (never shown to the agent)
    agent_docs/      the entry guide the agent reads
    prompts/         the round's prompt template
    config/          defaults, the pilot's run config and a small smoke run
    tests/           the tests, with each known way to game the score as a test
    docs/            DESIGN.md, REWARD.md, SANDBOX.md

`CLAUDE.md` has the working rules for changing this repository.
