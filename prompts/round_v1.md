# The task

You are given a small neural network: its definition and its weights. You are not told what it was
trained to do. Your goal is an **entry**: a program that predicts the network's output from the
input with fewer operations than the network itself needs, on as many inputs as it can.

This is round {{ROUND}} of at most {{ROUNDS_MAX}}. Each round is a fresh session: you keep nothing
from earlier rounds except what is written under "Earlier rounds" below and the saved entries.

## The network

Its definition is this file. In your sandbox it is the module `model`, and the same network is
written there as an entry by `model.reference(g, lv1)`.

```python
{{MODEL_SOURCE}}
```

Sizes: {{SIZES}}. The data set is every input: {{N_TOTAL}} inputs. One forward pass of the network
costs {{MODEL_OPS}} operations under the prices below.

## Entries

{{ENTRY_DOC}}

## How an entry is scored

The verifier measures an entry on inputs of its own choosing ({{INPUTS_NOTE}}):

- **coverage**: the share of the data set in the entry's region;
- **fidelity**: the share of the region on which the prediction is the network's output;
- **r**: the prediction's operations per input of the region, over one forward pass;
- **r_rule**: the rule's operations per input, over one forward pass;
- **b**: the build's operations, over one forward pass on every input of the data set.

The reward is

    {{REWARD_FORMULA}}

The first part is the share of the work saved if the entry is used on its region and the network
elsewhere. The second part prices a wrong answer at {{ALPHA}} forward passes. The network itself, as
an entry, scores exactly 0; an entry that is right everywhere at no cost would score 1.

When the rounds are over, the entry with the highest reward is locked and run, unchanged, on a
second network: the same definition, sizes and training, with another random seed. You never see
that network. What an entry computes from the weights is computed afresh there; a number you read
off this network and wrote into the entry as a constant stays what it was. With every submission you
state the fidelity you claim on this network and the fidelity you expect on the second one; both are
compared with what is measured.

## Tools, budgets and limits

- `run_python(code)`: up to {{PYTHON_CALLS}} calls this round, {{PYTHON_SECONDS}} s each. Exploring
  costs nothing: only a submitted entry is scored.
- `submit_entry(name, claimed_fidelity, expected_fidelity_other)`: up to {{SUBMISSIONS}} submissions
  this round.
- An entry may have up to {{MAX_NODES}} nodes, and its constants up to {{MAX_CONST}} numbers in all.

## Earlier rounds

{{HISTORY}}

## What to send back at the end

End your last message with these notes, with exactly these headings. They are all the next round
will know of this one.

    ## What the network does
    (in words: what you found, and how sure you are)

    ## The best entry
    (its name, what it does, and where its saving comes from)

    ## Claims
    (the fidelity you claim on this network and the one you expect on the second network, and why)

    ## Next
    (what you would try next)
