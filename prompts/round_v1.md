# The task

You are given a neural network: its definition and its weights. You are not told what it was
trained to do.

Your goal is to interpret it. An **interpretation** here is a program that predicts the network's
output from the input without running the network. You hand it in as an **entry**: the
interpretation together with the **region** of inputs it applies to.

The verifier measures three things about an entry: its **coverage** (how much of the data set the
region holds), its **fidelity** (how often the interpretation gives the network's output there) and
its **cost** (its operations, compared with a forward pass). They are defined exactly under "How an
entry is scored", where one formula combines them into the reward. Your aim is the highest reward.

This is round {{ROUND}} of at most {{ROUNDS_MAX}}. Each round is a fresh session: you keep nothing
from earlier rounds except what is written under "Earlier rounds" below and the saved entries.

## The network

Its definition is this file. In your sandbox it is the module `model`, and the same network is
written there as an entry by `model.reference(g, lv1)`.

```python
{{MODEL_SOURCE}}
```

Sizes: {{SIZES}}. One forward pass of the network costs {{MODEL_OPS}} operations under the prices
below.

## The data set

{{DATA_SET}}

## Entries

{{ENTRY_DOC}}

## How an entry is scored

The verifier runs the entry on {{INPUTS_NOTE}} and measures:

- **coverage**: the share of those inputs that are in the entry's region;
- **fidelity**: the share of the inputs in the region on which the interpretation gives the
  network's output;
- **cost**: all the operations the entry spends on those inputs (its build once, its region test on
  every input, its interpretation on the inputs in the region), divided by the number of inputs in
  the region and by the operations of one forward pass.

The reward is

    {{REWARD_FORMULA}}

{{REWARD_EXPLANATION}}

The network itself, as an entry, has fidelity 1 and cost 1, which makes 0; an entry that is right
everywhere at no cost would score 1. An entry whose region is empty scores 0, less what its region
test and its build cost.
{{SAMPLE_NOTE}}
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
  The verifier gives an entry {{VERIFIER_SECONDS}} s on its inputs.

## Earlier rounds

{{HISTORY}}

## What to send back at the end

End your last message with these notes, with exactly these headings. They are all the next round
will know of this one.

    ## What the network does
    (in words: what you found, on which inputs, and how sure you are)

    ## How you found it
    (what you looked at: the network's outputs, its weights, or both, and what you ran)

    ## The best entry
    (its name, its region, what its interpretation does, and where its saving comes from)

    ## Claims
    (the fidelity you claim on this network and the one you expect on the second network, and why)

    ## Next
    (what you would try next)
