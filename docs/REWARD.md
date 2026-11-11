# The reward

The verifier measures an entry; a reward function turns what it measured into one number. The
measuring is fixed. The reward is meant to be changed: it is a function in `level1/rewards.py`,
picked by the run config, and every receipt keeps the parts, so a finished run can be scored again.

## 1. The parts

With n inputs scored by the verifier (every input of the data set, or a sample), m of them in the
entry's region, and F operations for one forward pass:

| Part | Computed as |
| --- | --- |
| coverage | m / n |
| fidelity | inputs in the region where the entry gives the network's output / m |
| cost | all the entry's operations on the n inputs / (m * F) |

The cost is one number. A receipt also keeps its three parts:

| Part | Computed as |
| --- | --- |
| r | operations of the interpretation on the region / (m * F) |
| r_rule | operations of the region test on all n inputs / (n * F) |
| b | operations of the build / (n * F) |

    cost = r + (r_rule + b) / coverage

An empty region has no fidelity, no r and no cost.

**The build is spread over the inputs scored.** Until 11 November 2026 it was divided by the size
of the data set. When every input is checked the two are the same, so no receipt on the pilot on
pairs changes. They differ on a sample: with 17 million possible inputs anything computed once
would be nearly free, and an entry could run the network ahead on a great many inputs and look the
answers up. Spread over the inputs scored, a build is worth what it saves within the check. The
price is that a build looks cheaper under a bigger sample, so `verifier.sample_size` belongs to the
conditions of a run, like alpha.

## 2. `r_alpha`, the default

    reward = coverage * [ (1 - cost) - alpha * (1 - fidelity) ]

How to read it, for one input of the region: using the entry there in place of the network saves
(1 - cost) of a forward pass, and with probability (1 - fidelity) its answer is wrong, which is
charged at alpha forward passes. Coverage turns that into a share of the data set.

The same number, written as a difference:

    saved  = coverage * (1 - cost)
    errors = coverage * (1 - fidelity)
    reward = saved - alpha * errors

`saved` is the share of the work that goes away if the entry is used on its region and the network
elsewhere. `errors` is the share of inputs now answered wrongly.

What follows from the formula:

- The network itself, as an entry, has cost 1 and fidelity 1, so it scores 0.
- Nothing scores above 1. An entry that is right everywhere at no cost would score 1.
- An empty region scores 0 at best.
- On its region, an entry beats the network only if `1 - cost > alpha * (1 - fidelity)`. With
  alpha = 10, even a free entry needs a fidelity above 0.9.
- There is no threshold, floor or budget anywhere. The reward is a sum over inputs, so there is no
  boundary to place an entry just inside of.

On a sampled data set the fidelity used is its 95% lower bound (`lower_bound: false` switches that
off). An entry that is right on all 16,384 sampled inputs is then credited with a fidelity of
0.99977, which costs it 0.0023 of reward at alpha = 10. The network itself is charged the same, so
on a sample it scores -0.0023 and not 0.

**Read as a cost of finishing.** The proof pipeline scores a proof by the cost of the brute-force
proof over the cost of finishing with this one. The same ratio here: answering every input with
the network costs F per input; with the entry it costs the entry's work, a forward pass for each
input left uncovered, and alpha forward passes for each wrong answer. That ratio is

    1 / (1 - reward)

so the reward is the same ordering on another scale: 0 is a ratio of 1, and 0.996 is a ratio of 250.

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

## 4. The pilots' numbers

### On pairs (`models/lin-a`, every input checked)

Reference entries on the pilot's two networks, at alpha = 10. They are withheld from the agent;
what each one is, is in `withheld/lin-a/references.py`. Reproduce with
`python -m level1.report --model models/lin-a`.

| Entry | lin-a-s0 | lin-a-s1 | In short |
| --- | --- | --- | --- |
| the network itself | 0.0000 | 0.0000 | cost 1 |
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

### On 1 to 4 tokens (`models/lin-a-1to4`, sampled)

The same two networks, asked about more inputs than they were trained on. Which inputs those were
is in `withheld/lin-a-1to4/task.yaml`, with the reference entries next to it. Rewards at alpha =
10, each the mean of 8 scorings on fixed samples of 16,384 inputs; reproduce with
`python -m level1.report --model models/lin-a-1to4 --sample-size 16384` (about two minutes).

| Entry | s0 | s1 | Coverage | Cost | In short |
| --- | --- | --- | --- | --- | --- |
| the network itself | -0.0023 | -0.0023 | 1.00 | 1.000 | the lower bound's charge |
| target_everywhere | -0.6336 | -0.8269 | 1.00 | 0.010 | the task on every input: wrong on about one input in six |
| target_on_two | 0.2399 | 0.2367 | 0.25 | 0.020 | the task on the inputs of the training |
| target_up_to_two | 0.4491 | 0.4042 | 0.50 | 0.006 | the task on those and on shorter inputs |
| contraction | 0.5188 | 0.5188 | 1.00 | 0.479 | the weights multiplied together once, then a cheaper pass per input |
| target_then_contraction | 0.7062 | 0.6612 | 1.00 | 0.245 | the task where it holds, the contraction elsewhere |

Here the task is no longer the answer. Claimed everywhere it loses badly; claimed where it holds it
covers half the data set at most. The best reference pairs it with an exact but dearer program on
the other half and reaches about 0.71, so there is room above it for an agent that finds something
cheaper than the contraction on the inputs the training never saw. Nothing here says such a thing
exists.

One scoring is noisy: for an entry near 0.70, the reward of a single sample of 16,384 inputs has a
standard deviation of about 0.005, and the eight samples above lie up to 0.02 apart. That is why
the report scores each reference on `verifier.reference_samples` fixed samples and gives the mean.

## 5. Ways to game a reward, and what stops each

Each has a test in `tests/test_gaming.py`, unless another test is named or it holds by
construction. A hole found later gets its test there first.

| Route | What stops it |
| --- | --- |
| Run the network and call it an entry | cost 1: it saves nothing, on any region |
| Hide work in the region test, for example a region made of the inputs where a cheap guess is right, found by running the network | the region test is charged on every input scored |
| Hide work in the build, for example every output computed ahead | the build is charged in full; a table is worth what it saves |
| Make a build nearly free by spreading it over a huge data set | the build is spread over the inputs scored |
| Paste the network's outputs in as constants | the limit on numbers in constants (256) |
| Copy one network's exceptions into the entry | the seed check: constants stay, weights change |
| Pad the region with inputs the entry gets wrong | alpha above 1 |
| Weight the region, or pick the distribution | the verifier counts every input once; an entry names a subset |
| Do work outside the count: code, files, other weights | an entry is data; only the listed operations and the network's own weights exist |
| Charge an expensive branch as a cheap one | a lazy branch is charged on every input that takes it |
| Fit to the verifier's sample | a fresh sample per scoring, and the fidelity's lower bound |
| Send one entry again and again and keep the luckiest scoring | the locked entry is scored once more on a fresh sample, and the report gives that number |
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
- **The data set of the pilot on pairs can be enumerated.** The limit on constants is what keeps a
  pasted table out; a table computed from the weights is allowed and is charged to the build
  (0.7908 above). The pilot on 1 to 4 tokens cannot be enumerated within a run.
- **Choosing the best of many submissions** on a sampled data set favours lucky samples a little,
  even with the lower bound. The lock still picks by the recorded rewards; the second scoring of
  the locked entry is the number to quote. Two scorings of one entry on 16,384 inputs typically
  differ by a little under 0.01.
- **A sampled reward depends on the sample size**, through the lower bound and through the build.
  Compare runs made with the same `verifier.sample_size`.
- **The agent learns the data set, not the training.** On the pilot on 1 to 4 tokens it is told how
  an input is drawn and nothing about which inputs the network saw in training. Whether it should
  also have to find the data set itself is open.
- **A weak sandbox enforces nothing.** In process mode the agent's code can read the withheld
  folder. See `docs/SANDBOX.md`.
- **One number or a frontier.** The lock needs one number. The report also marks the submissions
  on the frontier of right share, wrong share and cost, which involves no alpha.

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
