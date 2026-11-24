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

## 3b. `r_alpha_numbers`

`r_alpha` less a price for what an entry stores:

    reward = saved - alpha * errors - price * numbers

Reading a number from a table costs one operation however long the table is, so `r_alpha` cannot
tell an entry that stores what was observed of the network's outputs from one that computes its
answers; only the limit on constants stood in the way. On ARC's challenge network the gains at
alpha = 360 and at alpha = 30 came from fitted tables of more and more numbers, up to 491 of the
512 allowed.
Here every stored number has to earn its price.

- `numbers` is what the entry stores beyond what the network's own definition stores: every number
  in its constants and the length of every `arange`, each counted once per 4 significant decimal
  digits it is written with and at least once (`lang.stored_numbers`). 0.25 counts once, 0.123456
  twice, a full float five times. The verifier counts them for the entry and for the network's
  definition and keeps both in the parts (`numbers`, `numbers_network`) when the reward asks.
- What an entry computes from the weights is not counted here: it is charged as computation, in
  the cost. So a table derived from the weights in the build is paid for by its operations, and a
  table fitted outside the entry and written into it by its numbers.
- The network itself stores nothing beyond its definition, so it scores what `r_alpha` gives it.
- `price` (0.002 by default) is a dial like alpha: how much reward a stored number must earn. With
  the single agent's entries on that network at alpha = 100: at 0.001 a table of 300 numbers no
  longer pays and a fit of 50 does; at 0.005 a fit of 50 loses to a rule on two thresholds.

The count as it stands can be sidestepped: it counts what is written in constants and in `arange`
lengths, not what an entry holds. Two of the first three runs under this reward did so within two
rounds (the table of ways to game below, and the two tests expected to fail in
`tests/test_gaming.py`). Until the count is of the values an entry computes whatever the input and
the weights, a reward under `r_alpha_numbers` says what the entry was charged, and the entry has to
be read to see what it stores.

A run can go on from rounds scored with `r_alpha` under this reward (`docs/CONTINUATION.md`): the
earlier entries are counted from their files and their rewards recomputed, and the note says what
changed. `python -m level1.loop rescore --run <run> --name r_alpha_numbers --alpha 100 --param
price=0.002` reads a finished run under it without running anything.

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
the other half and reaches about 0.71.

The first full pilot (11 November 2026, 5 rounds) went well above that: 0.832 on both networks,
measured exactly on every input. Its entry takes each token's own output from a table computed
from the weights on one or two tokens, and on three or four adds two columns of a precomputed table
of sums for every two tokens. Every gain after the first round came from computing the network's
own arithmetic more cheaply, not from a new description of what it does.

One scoring is noisy: for an entry near 0.70, the reward of a single sample of 16,384 inputs has a
standard deviation of about 0.005, and the eight samples above lie up to 0.02 apart. That is why
the report scores each reference on `verifier.reference_samples` fixed samples and gives the mean.

### On 1 to 8 tokens (`models/lin-b`, sampled)

A network of the same design trained afresh (`withheld/lin-b/train.py`), asked about inputs of 1 to 8
tokens: about 2.8 x 10^14 inputs, far too many to check. What it was trained on, and how often each
network follows the task by number of tokens, is in `withheld/lin-b/task.yaml`. Rewards at alpha =
10, each the mean of 8 scorings on fixed samples of 16,384 inputs; reproduce with
`python -m level1.report --model models/lin-b --sample-size 16384` (about two minutes).

| Entry | s0 | s1 | Coverage | Cost | In short |
| --- | --- | --- | --- | --- | --- |
| the network itself | -0.0023 | -0.0023 | 1.00 | 1.000 | the lower bound's charge |
| contraction | 0.1765 | 0.1765 | 1.00 | 0.821 | the weights multiplied together once, all 8 slots read |
| target_on_four | 0.1134 | 0.1130 | 0.12 | 0.039 | the task on the inputs of the training |
| target_on_three_to_five | 0.3313 | 0.3289 | 0.37 | 0.033 | the task where it holds best |
| target_everywhere | 0.3839 | 0.3796 | 1.00 | 0.021 | the task on every input: wrong on about one in eighteen |
| per_length | 0.6669 | 0.6710 | 1.00 | 0.128 | one token from a table, the task on 2 to 7, the contraction on 8 |

Unlike the network of the first two pilots, this one does not follow its task exactly even on its own
inputs: it follows it on 99.8% of the inputs of the training, and on 81% to 99% of the others,
depending on their length. Where it departs from the task, and how cheaply that can be said, is the
work.

### The six models of 12 November (sampled)

`lin-c` and `lin-d` are the designs of lin-a and lin-b trained afresh on another task, asked about 1
to 4 and 1 to 8 tokens; `att-a` to `att-d` are one-layer, one-head transformers asked about inputs
that fill one slot or more (2 slots for att-a and att-b, 4 for att-c and att-d). What each was trained
on is in `withheld/<name>/task.yaml`. Rewards at alpha = 10, each the mean of 8 scorings on fixed
samples of 16,384 inputs; reproduce with `python -m level1.report --model models/<name> --sample-size
16384` (a few minutes for the linear models, about ten for a transformer, whose forward pass is longer).

| Model | Entry | s0 | s1 | In short |
| --- | --- | --- | --- | --- |
| lin-c | target_then_contraction | 0.7031 | 0.7015 | the task on 1 or 2 tokens, the contraction on 3 or 4 |
| lin-c | contraction | 0.5188 | 0.5188 | the weights multiplied together once |
| lin-c | target_everywhere | -0.6959 | -0.5241 | the task on every input |
| lin-d | per_length | 0.6914 | 0.7082 | one token from a table, the task on 2 to 7, the contraction on 8 |
| lin-d | target_everywhere | 0.4024 | 0.4789 | the task on every input |
| lin-d | contraction | 0.1765 | 0.1765 | the weights multiplied together once |
| att-a | target_everywhere | 0.9973 | 0.9973 | the task on every input, right on all of them |
| att-b | target_everywhere | 0.9975 | 0.9975 | the same |
| att-c | per_length | 0.4985 | 0.4985 | one token from a table, the network itself on 2 or 3, the task on 4 |
| att-c | target_on_full | 0.2480 | 0.2480 | the task on the inputs of the training only |
| att-c | target_everywhere | -1.4089 | 0.1098 | the task on every input |
| att-d | per_length | 0.4985 | 0.4982 | as for att-c |
| att-d | target_everywhere | -0.3880 | -3.0919 | the task on every input |

lin-c and lin-d behave much like their counterparts (on lin-a-1to4 the same kind of entry as lin-c's
best scores 0.715 and 0.674; on lin-b, per_length 0.667 and 0.671). On att-a and att-b the task
itself is right on every input: a run there checks that the pipeline finds it, no more. att-c and
att-d are right on 99.99% of the inputs of their training and far from the task on fewer tokens, very
differently per seed: att-d's s1 follows it on 22% of its two-token inputs and 27% of its three-token
ones, att-c's s0 on 17% of its three-token ones. An entry tuned to s0 can do badly on s1 (the task on
every input: -1.41 on att-c's s0, 0.11 on its s1), so on these two the seed check matters most.

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
| Send one entry again and again and keep the luckiest scoring | an entry submitted again is scored on all of its samples together and stands as that one submission (`level1/pooling.py`); the locked entry is also scored once more on a fresh sample, and the report gives that number |
| Send one entry again and again to narrow the confidence bound | a pooled entry keeps the bound of one sample |
| Send trivially different copies of one entry, each scored on its own sample | not closed: they count as different entries; the fresh scoring of the locked entry in the report shows what luck added |
| Store what was observed of the network in a table, read at one operation | the limit on numbers in constants; under `r_alpha_numbers` every stored number is charged |
| Pack several numbers into the digits of one long number (`r_alpha_numbers`) | a number counts once per 4 significant digits |
| Get a number without a constant, as the length of an `arange` (`r_alpha_numbers`) | the length of an `arange` is counted like a stored number |
| Hold numbers as places in one long `arange`, a slice for each (`r_alpha_numbers`) | not closed: the count sees one number. The agent of `pilot-rnn-g-numbers-002` held about 290 fitted numbers this way from its second round on, so its rewards from then on are those of an entry that pays nothing for them. One node for each number: the limit on nodes does not bound it |
| Make numbers by arithmetic that gives the same value on every input, such as a token minus itself and e to that power (`r_alpha_numbers`) | not closed: the count sees none. The agent of `pilot-rnn-g-numbers-005` made about a dozen thresholds this way |
| The same through the wiring: a `stack` of up to 64 nodes that each hold a digit (`r_alpha_numbers`) | not closed, and not seen in a run yet. A count of what is written cannot close it; a count of the values an entry computes whatever the input and the weights closes the three, but for values tied to one weight's value |
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
  even with the lower bound. Pooling (`level1/pooling.py`) takes out the luck of submitting one
  entry again, not that of choosing between entries. The lock still picks by the recorded rewards;
  the second scoring of the locked entry is the number to quote. Two scorings of one entry on 16,384
  inputs typically differ by a little under 0.01. Read as pooled, the run on rnn-c locks an entry
  scored once at 0.821 over one whose six scorings together give 0.813; on 16 more samples of the
  first network the two average 0.814 and 0.816.
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
