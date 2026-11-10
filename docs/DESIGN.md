# Design

What the pipeline does and why it is built this way. `docs/REWARD.md` covers the reward and how it
could be gamed; `docs/SANDBOX.md` covers where the agent's code runs.

## 1. Level 1 in a few lines

A compact proof bounds a network's accuracy with a proof that is cheaper than running the network
on every input. Level 1 keeps the claim and drops the proof: the claim is measured.

- **Data set** D: the inputs. **Network** M.
- **Interpreter** (the agent): hands in an entry.
- **Entry**: a region S of D, given by a rule, and an interpretation I, a program that predicts
  M's output on S.
- **Verifier**: runs the entry on inputs it picks and measures three things.

The three things:

    coverage   mu = |S| / |D|
    fidelity   p  = share of S where I(x) = M(x)
    cost       operations of the entry, in forward passes

Fidelity is agreement of outputs (accuracy), not a KL divergence. That was decided on 10 November
2026: it is the simpler measure, and an entry then only has to produce an output, not a
distribution.

The interpretation need not be a circuit. Anything that predicts the output in counted operations
is an entry, the task the network was trained on included. On the pilot network that entry scores
about 0.996. This is accepted for level 1: the level asks whether a cheap, faithful description
exists and holds up, not how it relates to the weights.

## 2. The entry

An entry is a small program stored as JSON data, with two results per input:

- `rule`: a boolean, is this input in the region. Absent means every input.
- `predict`: a whole number, the network's output as the entry predicts it. Only computed for
  inputs in the region.

It is a graph of operations on tensors. The sources are the input's tokens, the network's weights
by name, constants and ranges; every other node applies one operation to earlier nodes. The agent
writes entries in Python with a builder (`lv1.Graph`), and the verifier interprets the JSON. Nothing
of an entry is ever executed as code, so an entry can do no work that is not counted. That is the
reason for a language of its own in place of plain Python.

**Prices** (`prices-v1`, the table at the top of `level1/lang.py`, repeated for the agent in
`agent_docs/ENTRY.md`): each operation on numbers costs 1. Elementwise arithmetic and comparisons
cost 1 per result entry, a matrix product 1 per multiply-add, a reduction over n entries n - 1,
reading 1 per number read, a sort n * ceil(log2 n). Moving data (reshape, slice, concat) is free.
The network itself is written in the same language (`models/<name>/model.py`), so a forward pass and
an entry are priced by one list. For the pilot a forward pass is 655 operations, the number the
proof pipeline's cost model gives for the same network.

**Where the cost is charged.** The interpreter splits an entry's work three ways:

- **build**: what does not depend on the input, computed once when first needed;
- **rule**: what the rule computes per input, charged for every input of the data set;
- **prediction**: what the prediction computes per input, charged for the inputs of the region. A
  value the rule already computed for an input is not charged twice.

`cond(c, x, y)` computes only the chosen side, so an entry can run something expensive on a few
inputs and pay for it only there.

**Limits.** A run limits an entry to 400 nodes and 256 numbers in constants (config: `entry`). The
constant limit is what stands in for "the region must be too large to enumerate": a table of the
network's outputs cannot be pasted in. A table can still be computed from the weights in the build,
and is then paid for there.

## 3. The verifier

`level1/verifier.py` is the only judge. Given an entry, it

1. checks that it is an entry (shape, operations, limits) and refuses it with the reason if not;
2. picks the inputs: every input when the data set has at most `verifier.enumerate_up_to` (200,000)
   of them, otherwise a fresh uniform sample of `verifier.sample_size` (4,000) from a seed nobody
   chose;
3. gets the network's outputs on them by running the network, written as an entry, through the same
   interpreter;
4. runs the entry on them and counts;
5. returns a receipt: the parts measured, and the reward under the run's reward function.

On a sample, the receipt also has the 95% lower bound of the fidelity (Wilson), and the reward uses
that bound, so a lucky sample does not pay. The pilot's data set has 4,096 inputs, so every input
is checked and nothing is sampled.

The agent states a fidelity with every submission. The default reward ignores the claim and uses
what was measured; the receipt records the overclaim. `r_alpha_claimed` is the variant in which the
claim is credited when the measurement confirms it and an overclaim is penalised.

## 4. The agent's session

A round is one session with two tools (`level1/session.py`):

- `run_python(code)`: runs the agent's code in a sandbox. In scope: the weights, the entry language
  (`lv1`), the network's definition (`model`), `measure(entry)` (the verifier's own measuring code on
  the same kind of inputs), `save_entry(name, entry)` and the entries saved so far. Exploring costs
  nothing; only submitted entries are scored.
- `submit_entry(name, claimed_fidelity, expected_fidelity_other)`: hands a saved entry to the
  verifier and returns the receipt.

Both have budgets per round (config: `session`). Every call is appended to the session's
`tools.jsonl`.

**The agent is not told what the model does.** This holds for every route: the prompt, the tools'
descriptions, the files placed in the sandbox, names, the entry guide, what the code can import.
Three things carry it:

- The model folder says nothing about the task. Weights are named for what they are (`embed`,
  `hidden`, `unembed`), networks by a neutral id (`lin-a-s0`). The task and the reference entries
  are in `withheld/<model>/`.
- `level1/hygiene.py` searches everything the pipeline puts within the agent's reach for the
  forbidden strings (the run config's list and the model's own in `withheld/<model>/task.yaml`),
  before the run starts and before every prompt. A hit stops the run before a model is asked. The
  agent's own notes, which come back to it in later rounds, are its words and are not searched.
- The sandbox keeps the agent's code away from the repository, the withheld folder, the second
  network's weights and the run folder. Only the container and Landlock modes enforce this; see
  `docs/SANDBOX.md`.

## 5. The rounds, the lock, the seed check

`level1/loop.py` drives a run.

- **Before the first round**: the network as an entry must score exactly 0, the hygiene check must
  pass, and the sandbox is probed with real escape attempts.
- **A round**: the prompt (`prompts/round_v1.md`) holds the network's definition, the entry guide,
  how entries are scored, the budgets, and the earlier rounds: every submission's receipt and the
  notes the agent left. Scored entries of earlier rounds are in the session's workspace under their
  names, the best one also as `best`.
- **The rounds end** at `rounds_max` (5), after `patience` (2) rounds in a row that did not raise
  the best reward, or after two refusals in a row.
- **The lock**: the scored submission with the highest reward, the earliest on a tie, byte for byte
  as it was submitted (its hash is checked).
- **The seed check**: the locked entry on the second network. What the entry computes from the
  weights is computed afresh there; a number read off the first network and written in as a
  constant stays what it was. This is the check on hard-coded exceptions.
- **The report**: the locked entry on both networks, every submission with its parts, the withheld
  reference entries on both networks, and the agent's closing notes. It also marks which
  submissions are on the frontier of coverage, fidelity and whole cost, which needs no reward
  function at all.

## 6. Who answers a round

`backend: claude_code` runs the Claude Code CLI headless, once per round, under your own login:
no built-in tools, no files, no project settings, one tool server with the session's two tools. The
CLI's environment has the API key variables removed, so it cannot bill a key, and model fallback is
switched off; an answer that still shows another model stops the run rather than record that
model's work under the run's model. Usage limits and overloads are waited out with growing pauses.
A login problem stops the run cleanly.

`backend: fake` replays scripted tool calls from JSON files against the real session, sandbox and
verifier. The tests and dry runs use it; no model is called.

## 7. Stopping and resuming

Everything a run knows is in its folder, written as it happens:

    run.json            config, commit, hash of the trusted files, sandbox and its probe
    log.jsonl           what happened, appended
    round_001/          prompt.md, session/, response.md, meta.json, round.json
    locked_entry.json, lock.json, seed_check.json, report.md, report.json

A round is done when its `round.json` is written. `resume` starts at the first round that is not.
A session that was cut off keeps its budgets, its saved entries and its submissions; when it starts
again, the prompt ends with a note that says what is left and what was already submitted. If the
code, the prompt or the model folder changed since the run began, `resume` refuses unless told
`--accept-changes`, and the log records it. `finish` redoes the lock, the seed check and the report
from the records alone, and `rescore` prints the records under another reward.

## 8. Adding a model

1. `models/<name>/config.yaml`: sizes, the input space, the weights' names and shapes, the networks
   (id, file, sha256). At least two networks if the run is to have a seed check.
2. `models/<name>/model.py`: `reference(g, lv1)`, the network's computation in the entry language.
   Its cost must be the same on every input.
3. `models/<name>/zoo/*.npz`: the weights, saved without pickle.
4. `withheld/<name>/task.yaml`: what the network was trained to do, and the strings that would give
   it away (`hygiene`). Optionally `references.py` with reference entries for the report.
5. A run config next to `config/pilot/lin-a.yaml`, and a small one like `lin-a-smoke.yaml` to try
   first.

Names are part of what the agent can reach: a model folder, a network id or a weight's name must
not say what the network does. The hygiene check reads them. If the network needs an operation the
language lacks, add it to `level1/lang.py` with a price, and say so in `agent_docs/ENTRY.md`.

## 9. Defaults, and where to change them

| What | Default | Where |
| --- | --- | --- |
| Reward | `r_alpha`, alpha = 10 | `config/defaults.yaml`, `reward` |
| Fidelity | agreement of outputs | `level1/lang.py`, `measure` |
| Entry limits | 400 nodes, 256 numbers in constants | `entry` |
| Rounds | at most 5, patience 2 | `rounds_max`, `patience` |
| Session budgets | 40 `run_python` calls, 4 submissions, 150 turns, 90 minutes | `session` |
| One `run_python` call | 300 s, 4 GB | `python` |
| Verifier | every input up to 200,000; else a sample of 4,000; 120 s per entry | `verifier` |
| Agent | `claude-fable-5-1`, effort high (the proof pipeline's pilots used this model) | `agent_model`, `effort` |
| Sandbox | `auto`: the strongest mode that works | `sandbox` |

Numbers live in the config files and never in code. Nothing specific to one model is in `level1/`.

## 10. What is not here

- Levels 2 and 3.
- KL as the measure of fidelity.
- A run scored without a scalar: the frontier is reported, but the lock needs one number.
- Networks beyond the pilot. Everything runs in numpy on a laptop; no GPU is involved.
