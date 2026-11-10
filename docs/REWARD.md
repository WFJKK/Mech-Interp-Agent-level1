# The reward

The verifier measures an entry; a reward function turns what it measured into one number. The
measuring is fixed. The reward is meant to be changed: it is a function in `level1/rewards.py`,
picked by the run config, and every receipt keeps the parts, so a finished run can be scored again.

## 1. The parts

With N inputs in the data set, n of them used by the verifier (n = N when every input is checked)
and F operations for one forward pass:

| Part | Computed as |
| --- | --- |
| coverage | inputs in the region / n |
| fidelity | inputs in the region where the entry gives the network's output / inputs in the region |
| r | operations of the prediction on the region / (inputs in the region * F) |
| r_rule | operations of the rule on all n inputs / (n * F) |
| b | operations of the build / (N * F) |

The build is paid once for the whole data set, so it is divided by N forward passes even when only
a sample is run. An empty region has no fidelity and no r.

## 2. `r_alpha`, the default

    saved  = coverage * (1 - r) - r_rule - b
    errors = coverage * (1 - fidelity)
    reward = saved - alpha * errors

How to read it: use the entry on its region and the network everywhere else, and compare with
running the network everywhere. `saved` is the share of the work that goes away. `errors` is the
share of inputs now answered wrongly. alpha is how many forward passes one wrong answer is worth.

What follows from the formula:

- The network itself, as an entry, has r = 1 and fidelity 1, so it scores exactly 0.
- Nothing scores above 1. An entry that is right everywhere at no cost would score 1.
- An empty region scores 0 at best.
- On its region, an entry beats the network only if `1 - r > alpha * (1 - fidelity)`. With alpha =
  10, even a free entry needs a fidelity above 0.9.
- There is no threshold, floor or budget anywhere. The reward is a sum over inputs, so there is no
  boundary to place an entry just inside of.

On a sampled data set the fidelity used is its 95% lower bound (`lower_bound: false` switches that
off).

**Choosing alpha.** alpha must be above 1. At 1, a wrong answer costs exactly the forward pass it
saves, so a free guess that is sometimes right scores above 0 and padding a region with wrong
inputs costs nothing; below 1, padding pays. The run's log and `python -m level1.report` warn when
alpha is 1 or less. Above 1, alpha sets how much an exception is worth chasing. On the pilot, one
wrong input out of 4,096 costs alpha / 4096 of reward; at alpha = 10 that is the price of 1.6
operations per input, so fixing a single exception pays only if the fix costs less than that.

## 3. `r_alpha_claimed`

The same formula with a credited fidelity in place of the measured one:

    credited = claimed                          if measured >= claimed
    credited = measured - penalty * overclaim   otherwise

where overclaim = claimed - measured, and penalty = 2 by default. Claiming exactly what will be
measured is best: a lower claim gives credit away, and a higher one loses more than it could have
gained. This is the claim-and-penalty form the levels were first described with. It is not the
default because, with the verifier measuring every submission anyway, the claim adds nothing to the
score that the measurement does not already give. The claims are recorded in every run either way.

## 4. The pilot's numbers

Reference entries on the pilot's two networks, at alpha = 10. They are withheld from the agent;
what each one is, is in `withheld/lin-a/references.py`. Reproduce with
`python -m level1.report --model models/lin-a`.

| Entry | lin-a-s0 | lin-a-s1 | In short |
| --- | --- | --- | --- |
| the network itself | 0.0000 | 0.0000 | r = 1 |
| contracted | 0.6947 | 0.6947 | the weights multiplied together once, then a cheaper pass per input |
| answer_table | 0.7908 | 0.7908 | every output computed ahead in the build, then looked up |
| two_candidates | 0.9716 | 0.9643 | few numbers compared per input |
| target_with_network_in_corner | 0.9932 | 0.9932 | the task, with the network run on a few inputs |
| target_outside_corner | 0.9947 | 0.9947 | the task, on a region that leaves a few inputs out |
| target | 0.9960 | 0.9887 | the task the network was trained on, as a one-operation program |

Two things to take from the table. First, the task itself is nearly the best entry there is: it
costs one operation and is wrong on 1 input of 4,096 on the first network and on 4 on the second.
So this network leaves the agent little to climb once it has seen what the outputs are. Second,
the seed check does its job: the task with the first network's one deviating input written in as
a constant scores 0.9969 on the first network and 0.9872 on the second, below the plain task's
0.9887 there.

## 5. Ways to game a reward, and what stops each

Each has a test in `tests/test_gaming.py`, unless another test is named or it holds by
construction. A hole found later gets its test there first.

| Route | What stops it |
| --- | --- |
| Run the network and call it an entry | r = 1: it saves nothing, on any region |
| Hide work in the rule, for example a region made of the inputs where a cheap guess is right, found by running the network | the rule is charged on every input of the data set |
| Hide work in the build, for example every output computed ahead | the build is charged in full; a table is worth what it saves |
| Paste the network's outputs in as constants | the limit on numbers in constants (256) |
| Copy one network's exceptions into the entry | the seed check: constants stay, weights change |
| Pad the region with inputs the entry gets wrong | alpha above 1 |
| Weight the region, or pick the distribution | the verifier counts every input once; an entry names a subset |
| Do work outside the count: code, files, other weights | an entry is data; only the listed operations and the network's own weights exist |
| Charge an expensive branch as a cheap one | a lazy branch is charged on every input that takes it |
| Fit to the verifier's sample | a fresh sample per scoring, and the fidelity's lower bound |
| Convert for free what takes a computation | casts are charged (`tests/test_lang.py`) |
| Claim more than is measured | the default reward ignores the claim; `r_alpha_claimed` penalises it |
| Sit just inside a threshold or a budget | there is none |

The reward functions also say when a setting invites one of these (`rewards.warnings`): alpha at 1
or less, the lower bound switched off, a penalty of 0.

## 6. What is known and left open

- **The trained task is a top entry.** Accepted for level 1 on 10 November 2026: the level asks
  for a cheap and faithful description, and the task is one. It does mean the pilot network has
  little room between the first good idea and the ceiling.
- **Fitting the outputs is allowed.** An entry may come from looking at outputs alone, with no
  reading of the weights. Level 1 cannot tell the two apart; the seed check only catches what does
  not carry over to another seed.
- **A number is 64 bits, whatever it holds.** Several small numbers packed into one and operated
  on together count as one operation. Nothing on the pilot gains from this; an entry built around
  it would.
- **The pilot's data set can be enumerated.** The limit on constants is what keeps a pasted table
  out; a table computed from the weights is allowed and is charged to the build (0.7908 above).
- **Choosing the best of many submissions** on a sampled data set favours lucky samples a little,
  even with the lower bound. On the pilot nothing is sampled.
- **A weak sandbox enforces nothing.** In process mode the agent's code can read the withheld
  folder. See `docs/SANDBOX.md`.
- **One number or a frontier.** The lock needs one number. The report also marks the submissions
  on the frontier of coverage, fidelity and whole cost, which involves no alpha.

## 7. Changing the reward

- **A parameter**: in the run config, `reward: {alpha: 3}`.
- **Another reward**: `reward: {name: r_alpha_claimed, penalty: 2}`. A block that names another
  reward replaces the default block whole.
- **A new reward**: a function in `level1/rewards.py` that takes the parts (and `claimed`) and its
  own parameters by name and returns at least `{"reward": ...}`; add it to `REWARDS`, and its
  formula and one-paragraph explanation for the agent's prompt to `FORMULAS` and `EXPLANATIONS`.
  The tests check that it gives the network itself 0 and that its texts pass the hygiene check
  (the file is also placed in the agent's sandbox). A parameter the function does not take is
  refused, so a typo is not ignored in silence.
- **See what a reward does before running anything**:
  `python -m level1.report --model models/lin-a --name r_alpha --alpha 3` scores the reference
  entries on both networks.
- **Score a finished run again**:
  `python -m level1.loop rescore --run <run folder> --alpha 3` (and `--param KEY=VALUE` for any
  other parameter). The records are not changed; note that the agent was steering by the reward
  the run was made with.
