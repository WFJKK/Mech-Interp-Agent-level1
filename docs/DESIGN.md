# Design

What the pipeline does and why it is built this way. `docs/REWARD.md` covers the reward and how it
could be gamed; `docs/SANDBOX.md` covers where the agent's code runs.

## 1. Level 1 in a few lines

A compact proof bounds a network's accuracy with a proof that is cheaper than running the network
on every input. Level 1 keeps the claim and drops the proof: the claim is measured.

- **Data set** D: the inputs the network is asked about. **Network** M, with F operations per
  forward pass.
- **Interpreter** (the agent): hands in an entry.
- **Interpretation** I: a program that predicts M's output from the input without running M.
- **Entry**: the interpretation together with its **region** S, the inputs of D it applies to. The
  region is given by a test on the input. The agent also states a claimed fidelity p.
- **Verifier**: runs the entry on inputs it picks and measures three things.

The three things:

    coverage   mu = |S| / |D|
    fidelity   q  = share of S where I(x) = M(x)
    cost       c  = operations of the entry / (|S| * F)

The cost counts everything the entry computes: the interpretation on S, the region test on every
input, and what is computed once ahead. It is one number; section 2 says how it splits.

Fidelity is agreement of outputs (accuracy), not a KL divergence. That was decided on 10 November
2026: it is the simpler measure, and an entry then only has to produce an output, not a
distribution.

The interpretation need not be a circuit. Anything that predicts the output in counted operations
is an entry, the task the network was trained on included. On the pilot on pairs that entry scores
about 0.996. This is accepted for level 1: the level asks whether a cheap, faithful description
exists and holds up, not how it relates to the weights.

## 2. The entry

An entry is a small program stored as JSON data, with two results per input:

- the **region** (stored as `rule`): a boolean, is this input in the region. Absent means every
  input.
- the **interpretation** (stored as `predict`): a whole number, the network's output as the entry
  predicts it. Only computed for inputs in the region.

The stored names are older than the words and stay, because changing them would change the entry
format. The builder takes both: `g.entry(interpretation=..., region=...)`.

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
an entry are priced by one list. For the pilot on pairs a forward pass is 655 operations, the number
the proof pipeline's cost model gives for the same network.

**Where the cost is charged.** The interpreter splits an entry's work three ways:

- **build**: what does not depend on the input, computed once when first needed;
- **region test**: what the region computes per input, charged for every input the verifier runs;
- **interpretation**: what the interpretation computes per input, charged for the inputs of the
  region. A value the region test already computed for an input is not charged twice.

The entry's cost is the three together, divided by the inputs in the region and by F. A receipt
keeps the three apart as well (`r`, `r_rule`, `b`), so that

    cost = r + (r_rule + b) / coverage

`cond(c, x, y)` computes only the chosen side, so an entry can run something expensive on a few
inputs and pay for it only there.

**Limits.** A run limits an entry to 400 nodes and 256 numbers in constants (config: `entry`). The
constant limit is what stands in for "the region must be too large to enumerate": a table of the
network's outputs cannot be pasted in. A table can still be computed from the weights in the build,
and is then paid for there.

## 3. The data set and the verifier

**The data set** is part of the model: `data_set` in `models/<name>/config.yaml`. Two kinds exist.

- `all`, the default: every input of the input space, each counted once.
- `lengths`: an input has a number of tokens drawn from a list, each equally likely; the tokens
  are uniform and fill the first slots, and the remaining slots hold a number that stands for
  "empty". This is a data set by its distribution: it cannot be listed, only drawn from.

The config's `data_set.says` is the sentence the agent's prompt gives about the data set. What the
agent is told is the data set, in full. What it is not told is anything about the training.

`level1/verifier.py` is the only judge. Given an entry, it

1. checks that it is an entry (shape, operations, limits) and refuses it with the reason if not;
2. picks the inputs, as `verifier.check` says:
   - `all`: every input of the data set;
   - `sample`: `verifier.sample_size` inputs drawn afresh for each scoring, from a seed nobody chose;
   - `auto`, the default: `all` when the data set can be listed and has at most
     `verifier.enumerate_up_to` (200,000) inputs, else `sample`;
3. gets the network's outputs on them by running the network, written as an entry, through the same
   interpreter;
4. runs the entry on them and counts;
5. returns a receipt: the parts measured, and the reward under the run's reward function.

On a sample, the receipt also has the 95% lower bound of the fidelity (Wilson), and the reward uses
that bound, so a lucky sample does not pay. A build is spread over the inputs scored, not over the
size of the data set (`docs/REWARD.md`, section 1). The pilot on pairs has 4,096 inputs, so every
one is checked; the pilot on 1 to 4 tokens has about 17 million and is sampled, 16,384 per scoring.

The agent states a fidelity with every submission. The default reward ignores the claim and uses
what was measured; the receipt records the overclaim. `r_alpha_claimed` is the variant in which the
claim is credited when the measurement confirms it and an overclaim is penalised.

## 4. The agent's session

A round is one session with two tools (`level1/session.py`):

- `run_python(code)`: runs the agent's code in a sandbox. In scope: the weights, the entry language
  (`lv1`), the network's definition (`model`), `info` (sizes, the data set, how it is checked),
  `network_outputs()` (the data set's inputs, or a fixed sample of them, with the network's
  outputs), `measure(entry)` (the verifier's own measuring code on those inputs),
  `save_entry(name, entry)` and the entries saved so far. Exploring costs nothing; only submitted
  entries are scored.
- `submit_entry(name, claimed_fidelity, expected_fidelity_other)`: hands a saved entry to the
  verifier and returns the receipt.

Both have budgets per round (config: `session`). Every call is appended to the session's
`tools.jsonl`, and every program the agent ran is kept in full with what it printed
(`session/python/NNN.py` and `.txt`). The Claude Code backend also keeps everything the CLI emitted
(`round_k/stream_NN.jsonl`), which holds what the agent wrote between its tool calls. What the agent
does while exploring is recorded and neither rewarded nor charged.

**The agent is not told what the model does.** This holds for every route: the prompt, the tools'
descriptions, the files placed in the sandbox, names, the entry guide, what the code can import.
Three things carry it:

- The model folder says nothing about the task. Weights are named for what they are (`embed`,
  `hidden`, `unembed`), networks by a neutral id (`lin-a-s0`). The task and the reference entries
  are in `withheld/<model>/`. Where the data set is wider than the training, which of its inputs
  the network was trained on is withheld in the same way.
- `level1/hygiene.py` searches everything the pipeline puts within the agent's reach for the
  forbidden strings (the run config's list and the model's own in `withheld/<model>/task.yaml`),
  before the run starts and before every prompt. A hit stops the run before a model is asked. The
  agent's own notes, which come back to it in later rounds, are its words and are not searched.
- The sandbox keeps the agent's code away from the repository, the withheld folder, the second
  network's weights and the run folder. A guard inside Python refuses and records every attempt in
  every mode; the container, Landlock and seatbelt modes also enforce it at the system level. See
  `docs/SANDBOX.md`.

## 5. The rounds, the lock, the seed check

`level1/loop.py` drives a run.

- **Before the first round**: the network as an entry must have fidelity 1 and cost 1 (a reward of
  0; on a sample the lower bound puts it just under 0), the hygiene check must pass, and the
  sandbox is probed with real escape attempts.
- **A round**: the prompt (`prompts/round_v1.md`) holds the network's definition, the data set, the
  entry guide, how entries are scored, the budgets, and the earlier rounds: every submission's
  receipt and the notes the agent left. The template names no model: everything about one comes
  from its folder and the run config. Scored entries of earlier rounds are in the session's workspace under their
  names, the best one also as `best`.
- **The rounds end** at `rounds_max` (5), after `patience` (2) rounds in a row that did not raise
  the best reward, or after two refusals in a row.
- **The lock**: the scored submission with the highest reward, the earliest on a tie, byte for byte
  as it was submitted (its hash is checked). When the verifier samples, the locked entry is the
  best of several scorings and its recorded reward leans high, so it is scored once more on a fresh
  sample, and the report gives both numbers.
- **The seed check**: the locked entry on the second network. What the entry computes from the
  weights is computed afresh there; a number read off the first network and written in as a
  constant stays what it was. This is the check on hard-coded exceptions.
- **The report**: the locked entry on both networks, every submission with its parts, the withheld
  reference entries on both networks (on a sampled data set, the mean of several fixed samples),
  what the agent ran, and its closing notes. It also marks
  which submissions are on the frontier, which needs no reward function at all. The frontier's
  three axes are the share of the data set answered rightly (up), the share answered wrongly
  (down) and the cost (down). Coverage and fidelity are not the axes because an entry should not
  count as better for covering inputs it gets wrong.

## 6. Who answers a round

`backend: claude_code` runs the Claude Code CLI headless, once per round: no built-in tools, no
files, no project settings, one tool server with the session's two tools. Model fallback is switched
off; an answer that still shows another model stops the run rather than record that model's work
under the run's model. Usage limits and overloads are waited out with growing pauses. A login
problem stops the run cleanly.

Who pays is the run config's `auth`:

- `login`, the default: your own Claude Code login. The CLI's environment has the API key variables
  removed, so it cannot bill a key.
- `api_key`: the key in the variable `api_key_env` names (`CPL_ANTHROPIC_KEY`), billed per token.
  The CLI runs in its `--bare` mode, which reads no login, and gets the key from a helper that
  prints that variable, so the key's value is on no command line and in no file. Before the first
  round the key is checked against the API's model list, which costs nothing. The run has a budget
  (`budget.run_usd`, and `budget.round_usd` for one round), counted in `spend.jsonl` by the CLI's
  own count; every session is started with what is left as its cap. A round that reaches its cap
  ends with what it submitted, and the rounds end when the run's budget is spent.

Every session's first event says what the CLI billed; one that does not fit `auth` stops the run.
An account that cannot pay (no credit, a spend limit) stops it too, rather than wait.

The two modes differ a little in what the agent reads besides the run's own text. With a stand-in
for the API, the CLI under `--bare` sent one line of its own ("You are a Claude agent, built on
Anthropic's Claude Agent SDK.") and the run's system prompt and prompt, nothing else; under a login
it also adds a short note on its environment (working folder, platform, date). Neither says
anything about the network.

`backend: fake` replays scripted tool calls from JSON files against the real session, sandbox and
verifier. The tests and dry runs use it; no model is called.

## 7. Stopping and resuming

Everything a run knows is in its folder, written as it happens:

    run.json            config, commit, hash of the trusted files, sandbox and its probe
    log.jsonl           what happened, appended
    spend.jsonl         billed to an API key: every session's cap and cost, appended
    round_001/          prompt.md, session/, response.md, stream_01.jsonl, meta.json, round.json
    locked_entry.json, lock.json, seed_check.json, report.md, report.json

A round is done when its `round.json` is written. `resume` starts at the first round that is not.
A session that was cut off keeps its budgets, its saved entries and its submissions; when it starts
again, the prompt ends with a note that says what is left and what was already submitted. If the
code, the prompt or the model folder changed since the run began, `resume` refuses unless told
`--accept-changes`, and the log records it. `finish` redoes the lock, the seed check and the report
from the records alone, and `rescore` prints the records under another reward.

## 8. Adding a model

1. `models/<name>/config.yaml`: sizes, the input space, the data set (and the sentence the agent
   is told about it), the weights' names and shapes, the networks (id, file, sha256). At least two
   networks if the run is to have a seed check.
2. `models/<name>/model.py`: `reference(g, lv1)`, the network's computation in the entry language.
   Its cost must be the same on every input.
3. `models/<name>/zoo/*.npz`: the weights, saved without pickle.
4. `withheld/<name>/task.yaml`: what the network was trained to do, and the strings that would give
   it away (`hygiene`). Optionally `references.py` with reference entries for the report.
5. A run config next to `config/pilot/lin-a.yaml`, and a small one like `lin-a-smoke.yaml` to try
   first.

`models/lin-a-1to4` shows a second model made from the first: the same weights with one all-zero
column added to `embed` for an empty slot, four slots in place of two, and a `lengths` data set.
`models/lin-b` shows a model with networks trained here: the training script lives in
`withheld/lin-b/train.py`, since it has to say what the networks are trained to do.

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
| Verifier | `check: auto`: every input up to 200,000, else a sample of 4,000; 120 s per entry; references on 8 fixed samples | `verifier` |
| Agent | `claude-fable-5-1`, effort high (the proof pipeline's pilots used this model) | `agent_model`, `effort` |
| Sandbox | `auto`: the strongest mode that works | `sandbox` |

Numbers live in the config files and never in code. Nothing specific to one model is in `level1/`.

## 10. What is not here

- Levels 2 and 3.
- KL as the measure of fidelity.
- A run scored without a scalar: the frontier is reported, but the lock needs one number.
- Networks beyond the pilots' two designs of size. Everything runs in numpy on a laptop; no GPU is
  involved, training included.
- A count of the forward passes the agent's own code makes while exploring. Its programs are kept,
  so the count can be made afterwards.
