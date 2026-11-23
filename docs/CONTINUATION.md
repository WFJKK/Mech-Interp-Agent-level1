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

The earlier run must be finished (it has a `lock.json`) and in the same `runs_dir`, on the same model
and network, under the same prices and reward. Team runs cannot be continued yet.
