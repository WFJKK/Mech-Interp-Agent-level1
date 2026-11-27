# Mech-Interp-Agent-level1: experiments

This branch is the pipeline of [`main`](https://github.com/WFJKK/Mech-Interp-Agent-level1) with rewards being
tried out. Main's README describes the pipeline, the models and the main runs; its section on the 3-number
network, under "Further experiments: a budget per input", also has the results below.

## What is different from main

- Two rewards in `level1/rewards.py`, written up in `docs/REWARD.md`: `r_budget` and `r_budget_nodes`
  (below).
- The count of stored numbers (`level1/lang.py`). A value the entry makes from its input or its weights
  counts as a stored number only if it comes out the same under 32 other sets of weights, each tried on 16
  inputs of its own. Before, a test that rarely holds could pass for a stored truth value.
- The verifier keeps what each single input cost (`input_ops` in the parts), which a budget per input
  needs.
- Configs: `config/pilot/rnn-g-budget.yaml` (ARC's challenge network, budget 0.1),
  `config/pilot/rnn-b-budget.yaml` and `config/pilot/rnn-b-nodes.yaml` (the 3-number network), each with a
  smoke run.
- A reference entry for the 3-number network, `target_with_pieces_near_ties` in
  `withheld/rnn-b/references.py`, built by hand (below).

## The rewards

Under main's default reward, `coverage × [(1 - cost) - α (1 - fidelity)]`, running the network on the
inputs near a tie pays: one forward pass there costs less than the α forward passes a wrong answer costs.
The best entries then compute the network's departures instead of predicting them.

`r_budget` gives every input a budget:

```
reward = (saved + α · predicted - price · numbers) / (1 + α)
```

- **saved** is what the entry saves of the forward passes, 1 - cost on an entry that covers every input.
- **predicted** is the share of inputs on which the entry gives the network's answer and spends at most
  `budget` of a forward pass on that input (its region test and its interpretation), taken at its 95% lower
  bound on a sample.
- **numbers** are the numbers the entry stores beyond those of the network's own definition.

An input over the budget counts as not predicted, whatever the entry answers there, so running the network
there earns nothing. The network itself scores 0; an entry that predicts every input at no cost scores 1.

`r_budget_nodes` charges length too: `node_price` for every node beyond the network's own definition.

```
reward = (saved + α · predicted - price · numbers - node_price · nodes) / (1 + α)
```

A budget per input cannot tell a short explanation from a long table of the network's answers looked up
in a few operations; their lengths can. A node counts whether or not it runs on a given input, so a table
or a deep tree of cases pays for every step of it.

The runs below use α = 10, budget 0.25 (19.5 of the 78 operations of a forward pass), price 0.01 and
node_price 0.0005. At that price about 25 nodes cost as much as one operation on every input.

## Results on the 3-number network

`models/rnn-b`: one layer of 4 ReLU units without biases, reading three standard-normal numbers, trained
to output the position of the second-largest; rounds on `rnn-b-s0`, which does so on 98.46% of inputs.

| Run | Config | Reward | Agent | Spent | Time |
| --- | --- | --- | --- | ---: | ---: |
| `pilot-rnn-b-budget-gpt-max` | `rnn-b-budget.yaml` | `r_budget` | `gpt-6.2-astra`, effort max | $34.66 | 3 h 35 min |
| `pilot-rnn-b-nodes-gpt-max` | `rnn-b-nodes.yaml` | `r_budget_nodes` | `gpt-6.2-astra`, effort max | $21.44 after 3 of 5 rounds | still running |

Both through OpenAI's API, 5 rounds, $20 a round and $60 a run, started afresh:
`python -m level1.loop run --config config/pilot/rnn-b-nodes.yaml --backend openai_api --agent-model gpt-6.2-astra --effort max --run-id <id>`,
with the key in `CPL_OPENAI_KEY` (`docs/OPENAI.md`).

### Scores

Each entry on three fresh samples of 16,384 inputs of `rnn-b-s0`; the second network is `rnn-b-s4`, the
seed check's, on one sample.

| Entry | Nodes | Coverage | Fidelity | Operations per input | Main's reward ($\alpha = 10$) | `r_budget` | `r_budget_nodes` | Second network: fidelity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure second argmax | 3 | 1 | 98.53% | 6.0 | 0.757 | 0.978 | 0.978 | 94.7% |
| Main's main run ($\alpha = 10$) | 18 | 0.972 | 99.54% | 8.3 | 0.811 | 0.955 | 0.955 | 95.5% |
| Budget run, locked | 381 | 1 | 99.85% | 8.3 | 0.868 | **0.988** | 0.972 | 89.7% |
| Node price, best so far | 61 | 1 | 99.34% | 6.5 | 0.836 | 0.985 | **0.983** | 71.4% |
| Node price, most faithful so far | 210 | 1 | 99.75% | 9.8 | 0.840 | 0.986 | 0.977 | 64.9% |
| Short entry by hand (reference) | 81 | 1 | 99.42% | 7.9 | 0.828 | 0.983 | 0.980 | 94.8% |

### What each entry predicts

On `rnn-b-s0`, from $10^6$ drawn inputs: the share of each part of the network's behaviour on which the
entry gives the network's answer within the budget.

| Part of the behaviour | Share of inputs | Main's main run | Budget run | Node price, best so far | Node price, most faithful so far |
| --- | ---: | ---: | ---: | ---: | ---: |
| Answers the second-largest | 98.46% | 98.3% | 99.99% | 99.99% | 99.93% |
| Answers position 0 where the second-largest is at position 1 or 2 | 1.19% | 0% | 95.7% | 74.5% | 90.4% |
| Answers position 2 where the second-largest is at position 0 | 0.21% | 0% | 94% | 0% | 80% |
| Answers position 1 where the second-largest is at position 0 | 0.12% | 0% | 32% | 17% | 83% |
| Answers 1 where the second-largest is at 2, or 2 where it is at 1 | 0.03% | 0% | 95% | 14% | 79% |
| All inputs | 100% | 96.76% | 99.84% | 99.36% | 99.75% |

Main's main run leaves out the 2.8% of inputs with $\lvert x_1 - x_2 \rvert \le 0.05$, which hold 91% of
the second row and 1.7% of the first.

### What the entries do

- **Budget run.** A table, worked out from the weights in advance: 128 rows of cut points on $x_1 / x_2$,
  one row per bin of $x_0 / x_1$, read in a few operations. It predicts the band of the second row and the
  shifted switch of the third almost everywhere, with no forward pass, but it is 381 nodes long.
- **Node price.** About 60 nodes. The second argmax written on one ratio of differences,
  $(x_0 - x_2) / (x_0 - x_1)$ or its inverse, with three cut points read off the weights. Near the tie
  $x_1 = x_2$ the entry computes the network itself as the linear map it is there: which units are on is
  read off the weights, and so is the map. The node price favours the cheapest of these: the best so far
  drops the shifted switch at $x_0 = x_2$ (an entry of round 2 predicted it for two thirds). The most
  faithful so far also predicts most of the sliver at $x_0 = x_1$, at 210 nodes.
- **Short entry by hand.** Built by Claude from `rnn-b-s0`'s weights, to see what a short explanation
  scores. The network has no biases, so only the ratios $x_0 / x_2$ and $x_1 / x_2$ matter. Away from
  ties: the second argmax on them. Near a tie: the network's linear map in the piece named by a simple
  direction lying in it, $(\pm 1, 0, 0)$ for the tie $x_1 = x_2$ and $(\pm 1, 0, \pm 1)$ for the tie
  $x_0 = x_2$, with which units are on read off the network's states there. In that map output 0 lies
  between outputs 1 and 2, which makes the band. It writes two numbers, the margins 0.92 and 1.1.

What none of them predicts: where all three numbers are close in ratio, 54 different patterns of units on
and off share 0.7% of the inputs, and the short entries miss most of the band there.

## Open issues

- Some cut points are numbers from the weights raised to a power chosen to fit (1.5 or 2). The count of
  stored numbers charges nothing for them.
- A value made from constants alone is counted only by its constants: one written 1.0 plus arithmetic in
  the build gives any number of thresholds for one stored number. It needs a test in
  `tests/test_gaming.py`, then a fix.
- The `openai_api` backend counts no tokens for an answer whose stream breaks off, which OpenAI may still
  bill (about $0.10 in the node-price run).
- The margins are small: the best entry under the node price beats the pure second argmax by 0.005, and
  operations, nodes and written numbers take most of what predicting the departures earns.
