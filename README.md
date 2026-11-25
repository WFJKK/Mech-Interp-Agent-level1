# Mech-Interp-Agent-level1

## Big picture

In this repo we consider an automated, agentic hill-climbing approach to mechanistic understanding.
This is a setup in which an agent proposes an interpretation $I$ of a network, a verifier scores
it, and the agent climbs that score round after round. We may think of an interpretation as a
description of the network that predicts the network's outputs (with some fidelity) more cheaply
than running it for some part of the input space. We will formalise this in more detail in the next
section. Our goal is to estimate the model's behaviour "mechanistically" instead of brute-force
running the network. Let us be clear that in this repo we are not letting the agent predict the
model's accuracy but only the model's output. In the next section we show how that can be related
to accuracy.

## More formally

We have a model $M$ and a data set $D$, a distribution over the inputs of $M$. An interpretation
$I$ is a program that maps an input $x$ to a predicted output $I(x)$ without running $M$. It is
written in a small language of priced operations ([`agent_docs/ENTRY.md`](agent_docs/ENTRY.md)), in
which $M$ itself is written too, so that both are priced by the same list. We may think of an
interpretation claim as a triple $(I, S, p)$: the interpretation $I$; the region $S$, the inputs it
is claimed for; and the claimed fidelity $p$.
In the code the pair $(S, I)$ is called an entry, and $p$ is handed in with it. A claim is judged
by three metrics:

- **Coverage** $\mu = P(x \in S)$, for $x$ drawn from $D$: how much of the data set the claim
  covers.
- **Fidelity** $q = P(I(x) = M(x) \mid x \in S)$: how often, on the region, the interpretation
  gives the network's output.
- **Cost** $c$, the length of the computation: the operations the claim spends (the interpretation
  on $S$, the region test on every input, and whatever it computes once from the weights), per
  input of $S$, in forward passes of $M$. The network itself has $c = 1$.

The verifier measures all three on inputs it draws from $D$, or on all of them when they are few.
Nothing is proved: a claim is checked by measuring it.

### Relationship to compact proofs

### Relationship to ARC's approach and to accuracy

ARC has set itself the goal to deductively estimate properties of networks instead of merely
sampling by running the network. This is usually (perhaps not conclusively) formulated via the
matching sampling principle, which states that such an estimate should be at least as accurate as
random sampling at the same compute ([ARC progress update: Competing with
sampling](https://www.lesswrong.com/posts/XdQd9gELHakd5pzJA/arc-progress-update-competing-with-sampling)).
Take as the property the accuracy of a model $M$ on a task with right answers $y(x)$, for inputs
$x$ drawn from $D$:

```math
A = \mathbb{E}_x \big[ \mathbf{1}\{M(x) = y(x)\} \big]
```

The goal is to construct a mechanistic estimate $\hat{A}$, which we would judge by its squared
error $(\hat{A}(B) - A)^2$ at a compute budget of $B$ forward passes. It is set against what
sampling achieves at that budget: running $M$ on $B$ inputs drawn from $D$ and taking the share it
gets right is off by $A (1 - A) / B$ in mean squared error.

Now in our setup we do not estimate accuracy but fidelity to the model. In the notation of this
section fidelity would be defined as

```math
q = \mathbb{E}_x \big[ \mathbf{1}\{I(x) = M(x)\} \mid x \in S \big]
```

Hence, in short, the average for our setup here is part of the grading, not part of the answer.

We can of course calculate our interpretation's accuracy $A_I$. As interpretation and model differ
on at most $1 - q$ of the inputs of the region, we have there:

```math
| A - A_I | \le 1 - q
```

## Reward function

### What we use

Each claim gets one number from what the verifier measured. With $\mu$ the coverage, $c$ the cost
and $q_{\text{low}}$ the lower end of the fidelity's 95% interval on the sample (the fidelity itself
when every input is checked):

```math
R = \mu \, \big[ (1 - c) - \alpha \, (1 - q_{\text{low}}) \big], \qquad \alpha = 10
```

Read it for one input of the region: answering it with the interpretation instead of the network
saves $1 - c$ of a forward pass, and a wrong answer is charged $\alpha$ forward passes. Multiplied
out over the $n$ inputs scored, with $F$ the operations of one forward pass and the measured
fidelity in place of its lower bound:

```math
R = \mu - \frac{\text{operations spent}}{n F} - \alpha \, \frac{\text{wrong answers}}{n}
```

In units of $1/n$: every input covered earns 1, every operation costs $1/F$, and every wrong answer
costs $\alpha$.

- **The network itself scores 0**: $\mu = q = c = 1$. An input on which the network itself is run
  earns nothing, so a positive score has to come from computing less than the network does.
- **1 is the most**: right on every input at no cost. On a sample of 16,384 inputs the lower bound
  caps it at about 0.998.
- **Errors are expensive**: a claim gains only where $q > 1 - (1 - c)/\alpha$, so at $\alpha = 10$
  it must be right more than 90% of the time even if it is free. $\alpha$ must exceed 1: at
  $\alpha = 1$ the reward is $q - c$, and a free constant guess, right on 1 input in 64, would beat
  the network.
- **It adds up over the data set**: an input left out of the region earns 0, the same as an input on
  which the network itself is run, so a claim's score is the sum of what its parts earn.
- **As a speed-up**: run the network wherever the claim does not apply and count each wrong answer as
  $\alpha$ forward passes. Then $R$ is 1 less the compute with the claim over the compute without it,
  and $1/(1 - R)$ is the speed-up: 0.99 is about a hundredfold.
- **The frontier needs no $\alpha$**: the reports also mark the Pareto frontier of right share, wrong
  share and cost; $R$ picks one point on it.

Rewards are functions in `level1/rewards.py`, chosen by the run config. Every receipt keeps the
measured parts, so a finished run can be scored again under another reward
(`python -m level1.loop rescore`), and a new reward has to give the network itself 0.
`docs/REWARD.md` has the details and the ways a reward can be gamed.

### Louis's suggestion

```math
\text{score} = \begin{cases} \mu \, p & \text{if } \hat q \ge p \\ \mu \, \big(\hat q - \lambda \, (p - \hat q)\big) & \text{otherwise} \end{cases}
```

## Agentic setup / Auto-research

![The agentic flow of level 1](docs/figures/level1-flow.png)

## Results

### Overall

In the following experiments we will now use the agentic setup from above to analyse different
models. While our goal is "heuristic understanding", naturally the "reward function" merely evaluates
how well the agent can "predict" the output. Hence, we expect that part of the hill-climbing will
amount to inference optimisation in a sense. For that reason we always compare to a baseline of the
agent working on an untrained model of the same architecture. The baseline agent gets the same prompt
and rounds, with one more paragraph that asks for an interpretation exact for any weights. For a
baseline entry to be valid, we demand that it is exact, that is, that it has 100% fidelity, on every
other network of the architecture, both trained and untrained; the best valid entry is then run
unchanged on the trained network, on the
same inputs as the trained run's entry, and the two rewards are compared. (How an untrained network
is initialised depends on the model, and we say it in each experiment's section.)

### Experiment 1: linear + attention models on trained input only

#### Linear max of 4

##### Architecture and training

- Sizes: 64 tokens, 4 slots per input, width 8.
- Weights: three matrices and no biases: an embedding (8 × 64), a hidden matrix (8 × 8) and an
  unembedding (64 × 8).
- Task: output the largest of the four tokens.
- Inputs: four tokens, each uniformly random from 0 to 63, drawn fresh at every step.
- Loss and optimiser: cross-entropy with AdamW (learning rate 0.001, betas 0.9 and 0.999, weight decay
  0.01), batch 1,024, float32.
- Starting weights: uniform in ±1/√k, where k is the number of inputs to the matrix (64 for the
  embedding, 8 for the other two).
- Stopping: training ran in blocks of 10,000 steps until accuracy over all 64⁴ inputs gained less than
  0.05 percentage points over 20,000 steps.
- Networks: two, from seeds 0 and 1 (80,000 and 70,000 steps). They give the largest token on 99.83%
  and 99.79% of all inputs.

##### Price, runtime, agent, internet, inference

- Price: $4.31 (agent on trained), $34.89 (agent on baseline).
- Runtime: 14 min (agent on trained), 2 h 22 min (agent on baseline).
- Agent: Claude Fable 5.1 (`claude-fable-5-1`), thinking effort high (Claude Code's `--effort high`).
- Agent tools: `run_python` (runs Python in a sandbox, with the network's weights) and `submit_entry`
  (hands an entry to the verifier, which scores it).
- Hyperparameters: money cap $20 a round and $60 a run; 5 rounds; entry limits 400 nodes and 256 numbers in constants.
- Internet: not used (no web tools, and the agent's Python runs with the network blocked).
- Inference: about 9.6 million forward passes (agent on trained), more than 11 million (agent on baseline;
  103 of its 159 programs counted).

##### Scores

The baseline's untrained networks are the trainer's starting weights, kept as drawn: uniform in ±1/√k
as above. There are six of them, from six other seeds (2000 to 2005); the baseline agent works on one,
and its entry must be exact on all six and on the two trained networks.

Each score below is measured on 16,384 inputs drawn at random from the data set (all 64⁴ inputs of
four tokens), a separate draw for each network. One forward pass costs 671 operations.

| Entry | Network | Coverage | Fidelity | Cost | Reward |
| --- | --- | ---: | ---: | ---: | ---: |
| Brute force (the network itself) | both seeds | 1 | 100% | 1 | -0.0023<sup>1</sup> |
| Pure max of 4 | first seed | 1 | 99.82% | 0.0045 | 0.970 |
| Pure max of 4 | second seed | 1 | 99.85% | 0.0045 | 0.973 |
| Agent on trained | first seed | 1 | 99.87% | 0.0064 | 0.974 |
| Agent on trained | second seed | 1 | 99.92% | 0.0064 | 0.980 |
| Agent on baseline | first seed | 1 | 100% | 0.174 | 0.824 |
| Agent on baseline | second seed | 1 | 100% | 0.171 | 0.827 |

<sup>1</sup> Not 0, because the reward uses the 95% lower bound of the fidelity and not the measured
fidelity. 16,384 right answers out of 16,384 give a lower bound of 99.977%, and the missing 0.023% is
charged like wrong answers, at 10 forward passes each. Every entry measured on a sample pays this same
0.0023.

##### Evaluation

Agent on trained against agent on baseline: in reward the difference looks small, 0.974 against 0.824
on the first seed, because both are already close to 1. In cost it is large. The agent on trained
needs 0.0064 of a forward pass per input and the agent on baseline 0.174, which is 27 times as much.
The baseline's entry is never wrong and the agent's is wrong on about 0.1% of inputs, and the agent's
still comes out far ahead.

Agent on trained against pure max of 4: the two are close, 0.974 against 0.970. But the agent's entry
is not just the max. Its cost is 0.0064 against 0.0045, about 1.4 times as much, so it does slightly
more: one more step per input, a floor that it works out from the weights. That buys a
little fidelity, 99.87% against 99.82%.

What follows is read from the run's report, which the pipeline writes at the end of every run. It
prints the entry the agent submitted, which is the program the verifier scored. It submitted a program
that takes 4 operations per input. Per input it computes `answer = max(t₁, t₂, t₃, t₄, floor)`.

The floor is one number per network. The entry finds it by asking the network what it answers to each
single token from 0 to 7, and taking the smallest of those eight answers: 10 on the first seed, 11 on
the second. This takes 5,119 operations. They are done once per network and not per input, so they are
not part of the 4 operations above; in the cost they are spread over the 16,384 scored inputs, which
adds 0.3 operations per input.

The floor is needed because the network never answers below it: when all four tokens are below the
floor, the network answers the floor and not the largest token. That is 0.06% of inputs, about a third
of the network's mistakes (inputs where the network does not give the largest token; it gives the
largest token on 99.83% of inputs). With the floor the entry predicts them.

It misses the other two thirds, 0.11% of inputs. There the tokens at the top are close together, most
often with two tokens one below the largest, and the network answers one below the largest token (57%
of the missed inputs), two below (15%) or up to four above (20%). The entry answers the largest token.

The agent on baseline found no rule: on the untrained network the largest token, the smallest, sums
and others each agree with it on about 1 input in 64. So its entry is the network's own calculation,
exact for any weights, done with less work in three steps.

One table. It multiplies the three matrices together once, `M = U · H · E` (64 × 64). The score of an
output o is then

    s(o) = M[o,t₁] + M[o,t₂] + M[o,t₃] + M[o,t₄]

and the network answers the o with the highest score.

Pair sums. For every pair of tokens it stores once `P[a,b][o] = M[o,a] + M[o,b]`, sorted from highest
to lowest, so that

    s(o) = P[t₁,t₂][o] + P[t₃,t₄][o]

Early stop. With A₁ ≥ A₂ ≥ … the sorted values of the first pair and B₁ ≥ B₂ ≥ … those of the second,
it computes s(o) only for the K highest outputs of each pair. Every other output has

    s(o) ≤ A[K+1] + B[K+1]

so if the best score found is above this bound, it is the network's answer. Otherwise K grows, and
rarely all 64 scores are added up.

This takes about 55 operations per input, plus 61 for the tables (1.0 million operations once per
network, spread over the 16,384 scored inputs): 0.174 of a forward pass. None of it uses anything a
network has learned.

##### Is the network explained?

On the first seed, over all 64⁴ inputs:

| Part of the behaviour | Share of inputs | Short description |
| --- | ---: | --- |
| Answers the largest token | 99.83% | yes: "max of 4" |
| Answers the floor when all tokens are small | 0.06% | yes: one number from the weights |
| Close-token mistakes | 0.11% | none found; they depend on the exact numbers in M |

So we have explained the network up to 0.11% of inputs. For those we know where they are, among the
inputs whose top tokens are close together, but we have no explanation for them apart from running the
network.

#### Attention max of 4

##### Architecture and training

- Sizes: 64 tokens, 4 slots per input, width 32.
- Architecture: a transformer with one layer and one attention head, and nothing else: no MLP, no
  layer norm, no biases. Each slot has a learned position embedding, and the output is read at the last
  token. This is the network of Gross et al.
- Weights: a token embedding (64 × 32), a position embedding (4 × 32), the query, key, value and output
  matrices of the head (32 × 32 each) and an unembedding (32 × 64).
- Task: output the largest of the four tokens.
- Inputs: four tokens, each uniformly random from 0 to 63, drawn fresh at every step.
- Loss and optimiser: cross-entropy at the last position with AdamW (learning rate 0.001, weight decay
  0.01), batch 512, float32.
- Starting weights: uniform in ±1/√k, where k is the first dimension of the matrix (64 for the token
  embedding, 4 for the position embedding, 32 for the others).
- Stopping: after 4,000 steps.
- Networks: two, from seeds 0 and 1. They give the largest token on 99.992% and 99.996% of all inputs.

##### Scores

##### Evaluation

##### Is the network explained?

### Experiment 2: linear + attention models on inputs beyond the trained ones

We have seen that for a network evaluated on its training input distribution, the agent could
discover a compressed interpretation that predicts the model's behaviour with high fidelity.
Interestingly, with linear max of 4 we saw that the agent does not merely approximate the basic task,
max_4, but also includes a cheap modification that mirrors some of the "easy" errors the network
makes. In the following experiments we will ask for interpretations of the network on more inputs
than it was trained on, so for linear max of 4 we will actually input up to 8 tokens. It is a priori
not clear how the agent will think about this. Clearly, in the standard input space (i.e. 4 tokens)
the model should have an interpretation close to max_4, whereas it will deviate away from that outside
that space. We therefore expect that for 4 tokens we should have the cleanest explanation, whereas
outside that space things will be less clear. One metric to evaluate an interpretation on is its
coverage. Will it be beneficial with regard to the reward function for the agent to use max_4 on its
input space and a different interpretation outside? As we will see below, this is what the agent does.
It does not restrict itself to the 4-token inputs, which are only one eighth of the data set. It
submits one interpretation that covers every input and treats each number of tokens separately: on 4
tokens it is the max_4 with the floor from Experiment 1, and outside it uses other, more expensive
interpretations.

#### Linear max of 4

##### Architecture and training

The same as in Experiment 1: the same two networks, with the same architecture and training. Only the
inputs differ. Here the networks have 8 slots and are asked about inputs of 1 to 8 tokens, each number
of tokens equally likely, while they were trained on exactly 4.

##### Price, runtime, agent, internet, inference

- Price: $39.83 at API prices, run under a login and not charged (agent on trained), $19.91 (agent on
  baseline).
- Runtime: 2 h 4 min (agent on trained), 1 h 9 min (agent on baseline).
- Agent: Claude Fable 5.1 (`claude-fable-5-1`), thinking effort high (Claude Code's `--effort high`).
- Agent tools: `run_python` (runs Python in a sandbox, with the network's weights) and `submit_entry`
  (hands an entry to the verifier, which scores it).
- Hyperparameters: money cap $20 a round and $60 a run (agent on baseline), none (agent on trained, run
  under a login); 5 rounds; entry limits 400 nodes and 256 numbers in constants.
- Internet: not used (no web tools, and the agent's Python runs with the network blocked).
- Inference: about 500 million forward passes (agent on trained), about 3.2 million (agent on baseline).

##### Scores

| Entry | Network | Coverage | Fidelity | Cost | Reward |
| --- | --- | ---: | ---: | ---: | ---: |
| Brute force (the network itself) | both seeds | 1 | 100% | 1 | -0.0023<sup>1</sup> |
| Pure max | first seed | 1 | 94.42% | 0.021 | 0.384 |
| Pure max | second seed | 1 | 94.37% | 0.021 | 0.380 |
| Agent on trained | first seed | 1 | 99.92% | 0.073 | 0.913 |
| Agent on trained | second seed | 1 | 98.92% | 0.072 | 0.803 |
| Agent on baseline | first seed | 1 | 100% | 0.308 | 0.689 |
| Agent on baseline | second seed | 1 | 100% | 0.308 | 0.690 |

<sup>1</sup> Not 0, because the reward uses the 95% lower bound of the fidelity and not the measured
fidelity. 16,384 right answers out of 16,384 give a lower bound of 99.977%, and the missing 0.023% is
charged like wrong answers, at 10 forward passes each. Every entry measured on a sample pays this same
0.0023.

##### Evaluation

Agent on trained against agent on baseline: the difference is quite large, 0.913 against 0.689 on the
first seed. In cost, the agent on trained needs 0.073 of a forward pass per input and the agent on
baseline 0.308, which is 4.2 times as much.

The agent on baseline again found no rule, and its entry is again the network's own calculation, exact
for any weights. It uses the table `M = U · H · E` and the pair sums `P[a,b][o] = M[o,a] + M[o,b]` of
Experiment 1. An input of 1 or 2 tokens is answered from a table. For a longer input the entry pairs
up the tokens, adds the pair sums,

    s(o) = P[t₁,t₂][o] + P[t₃,t₄][o] + …

and takes the highest of all 64 scores. That is 5 operations for 1 or 2 tokens, 197 for 3 or 4, 262 for
5 or 6 and 327 for 7 or 8, plus 20 per input for the tables.

It does not use the early stop of the baseline in Experiment 1. We checked whether that would have
been better here, and it would not. The early stop needs every pair sum sorted, which costs about 50
operations per input, and it only saves work on short inputs: about 135 operations on 3 or 4 tokens,
some on 5 or 6, and nothing on 7 or 8, where it costs more than adding up all 64 scores. By our
estimate an entry with the early stop needs between 211 and 227 operations per input, against 217 for
the baseline agent's. So we found no better exact entry and take 0.69 as the baseline. It is lower
than the 0.82 of Experiment 1 because the inputs are longer.

Agent on trained against pure max: the gap is now much larger than in Experiment 1, 0.913 against
0.384, where it was 0.974 against 0.970. We are outside the training distribution. Off 4 tokens the
network departs from the largest token, on 5.6% of all inputs, so the task alone no longer predicts it.

What the agent on trained submitted: one program of 400 steps. 105 of the steps are branch points
("if this holds, take that step's result, otherwise the other one's"), and for an input only the steps
on its path are run and counted. With `M = U · H · E` as before, the score of an output is
`s(o) = M[o,t₁] + … + M[o,tₙ]`, and m is the largest token.

- Counting the tokens: three checks "is this slot empty?", each followed by a branch point. 6
  operations.
- 1 token: the answer is read from a table of the network's answer to each single token,
  `T[t] = argmax over o of M[o,t]`, built once. 1 operation.
- 4 tokens: `answer = max(m, floor)`, the entry of Experiment 1, read from a table. 3 operations for m
  and 1 for the table.
- 2 or 3 tokens: the same. For some values of m the entry also holds a second candidate; then it
  computes s(o) for both candidates and answers the higher.
- 5 to 8 tokens: 5 or 6 more comparisons, on m and on the ratio `(t₁ + … + tₙ) / (m + 1)`, lead to one
  of 98 short lists C of candidate outputs, 2 to about 9 of the 64. The entry computes s(o) only for
  those and answers the highest, `answer = argmax over o in C of s(o)`. With 8 candidates and 8 tokens
  that is 64 operations for the scores.

On the first seed, with 20,000 inputs per number of tokens:

| Tokens | What the entry does | Operations per input | Fidelity |
| ---: | --- | ---: | ---: |
| 1 | reads the network's answer to that single token from a table | 7 | 100% |
| 2 | the largest token with the floor of Experiment 1; for some values of the largest token it also holds a second candidate, computes the exact score of both and takes the higher | 11 | 100% |
| 3 | the same | 13 | 100% |
| 4 | the largest token with the floor, exactly the entry of Experiment 1 | 10 | 99.88% |
| 5 | no rule: from the largest token and the sum of the tokens it picks a short list of outputs that can win, about 9 of the 64, computes their exact scores and takes the highest | 59 | 99.93% |
| 6 | the same | 70 | 99.82% |
| 7 | the same, with other lists | 94 | 99.89% |
| 8 | the same | 109 | 99.77% |

Which outputs are on a list, and the second candidates for 2 and 3 tokens, were read off the first
network.

##### Is the network explained?

On the first seed:

| Part of the behaviour | Share of inputs | Short description |
| --- | ---: | --- |
| Answers the largest token, with the floor | 96.4% | yes: one rule |
| Answers to a single token that depart from it | 0.8% | yes: a table of 64 answers, from the weights |
| Departures on 2 and 3 tokens | 0.06% | not explained, only stored: the entry saves these cases as seen on this network (for some values of the largest token, a second possible answer) and computes the exact score of both. That gets all 18 departures on 2 tokens right, and 163 of the 181 on 3 tokens |
| Departures on 5 to 8 tokens that the entry catches | 2.65% | none. With many tokens the smaller ones can outvote the largest, and the entry has no rule for when. From the largest token and the sum of the tokens it picks one of 98 lists of outputs that were seen to win in such a case on this network, computes the network's exact score for each output on the list and answers the highest. So it predicts these inputs by redoing part of the network's own calculation |
| Not predicted | 0.09% | none |

So we have explained the network up to about 2.8% of inputs. For 2.7% the entry predicts the network,
but only from cases it saved or by computing part of it, both read off this one network. For 0.09% we
have no prediction at all. On the second seed the saved cases and lists catch much less: of the
departures on 2 and 3 tokens the entry gets 0 of 18 and 24 of 130, and 1.1% of all inputs are not
predicted.

#### Attention max of 4

##### Architecture and training

The same as in Experiment 1: the same two networks, with the same architecture and training. Here they
are asked about inputs of 1 to 4 tokens, each number of tokens equally likely, while they were trained
on exactly 4.

##### Price, runtime, agent, internet, inference

- Price: $29.42 (agent on trained), $41.63 (agent on baseline).
- Runtime: 1 h 43 min (agent on trained), 2 h 13 min (agent on baseline).
- Agent: Claude Fable 5.1 (`claude-fable-5-1`), thinking effort high (Claude Code's `--effort high`).
- Agent tools: `run_python` (runs Python in a sandbox, with the network's weights) and `submit_entry`
  (hands an entry to the verifier, which scores it).
- Hyperparameters: money cap $20 a round and $60 a run; 5 rounds; entry limits 400 nodes and 256 numbers in constants.
- Internet: not used (no web tools, and the agent's Python runs with the network blocked).
- Inference: about 36 million forward passes (agent on trained), about 4.4 million (agent on baseline).

##### Scores

| Entry | Network | Coverage | Fidelity | Cost | Reward |
| --- | --- | ---: | ---: | ---: | ---: |
| Brute force (the network itself) | both seeds | 1 | 100% | 1 | -0.0023 |
| Pure max | first seed | 1 | 76.57% | 0.0005 | -1.4089 |
| Pure max | second seed | 1 | 91.54% | 0.0005 | 0.1098 |
| Agent on trained | first seed | 1 | 100% | 0.0089 | 0.9887 |
| Agent on trained | second seed | 1 | 100% | 0.0071 | 0.9905 |
| Agent on baseline | first seed | 1 | 100% | 0.0404 | 0.9573 |
| Agent on baseline | second seed | 1 | 100% | 0.0404 | 0.9573 |

##### Evaluation

Agent on trained against agent on baseline: in reward the two are close here, 0.989 against 0.957,
where for linear max of 4 they were far apart, 0.913 against 0.689. Counted in work, the picture is
the same for both networks. Here the agent on trained needs 116 operations per input and the agent on
baseline 525, which is 4.5 times as much. For the linear network it was 51 against 217, which is 4.2
times as much. So the agent gains about as much from the trained network in both cases. The rewards
differ because a forward pass costs 12,987 operations here and 703 there, and most of the
transformer's forward pass can be stored in tables for any weights.

### Experiment 3: AlgZoo

The inputs of these networks are real numbers, so there are infinitely many of them. Fidelity is then
the share of inputs drawn from the data set on which the entry gives the network's output. As in the
other experiments, each score is measured on 16,384 inputs drawn at random, here each number from a
standard normal, and the reward uses the 95% lower bound of that share.

#### RNN second argmax of 3 numbers

##### Architecture and training

- Sizes: 3 numbers per input, 4 hidden units, 3 outputs (the positions 0, 1 and 2).
- Architecture: a recurrent network with one layer, a ReLU and no biases. It reads the numbers one at a
  time, `h = relu(w · xₜ + R · h)`, starting from h = 0, and its output is the position of the highest
  of `readout · h`. One forward pass is 78 operations.
- Weights: input (4 × 1), recurrent (4 × 4) and readout (3 × 4), 32 numbers.
- Task: output the position of the second-largest of the three numbers.
- Inputs: three numbers, each from the standard normal distribution, drawn fresh at every step.
- Loss and optimiser: cross-entropy with AdamW (learning rate 0.02, warmed up over 2¹⁴ steps and then
  decayed to 1% of it, weight decay 0.001), batch 1,024, float32.
- Starting weights: uniform in ±0.5.
- Stopping: after 2²⁰ steps.
- Networks: trained and published by ARC in AlgZoo, five seeds. We use the best one (it gives the
  second-largest on 98.4% of inputs) and the next best as second seed (94.5%). The baseline networks
  are ARC's saved starting weights of the same seeds.

##### Price, runtime, agent, internet, inference

- Price: $11.48 (agent on trained), $49.11 (agent on baseline).
- Runtime: 38 min (agent on trained), 2 h 30 min (agent on baseline).
- Agent: Claude Fable 5.1 (`claude-fable-5-1`), thinking effort high (Claude Code's `--effort high`).
- Agent tools: `run_python` (runs Python in a sandbox, with the network's weights) and `submit_entry`
  (hands an entry to the verifier, which scores it).
- Hyperparameters: money cap $20 a round and $60 a run; 5 rounds; entry limits 400 nodes and 256 numbers in constants.
- Internet: not used (no web tools, and the agent's Python runs with the network blocked).
- Inference: about 120 million forward passes (agent on trained), about 16 million (agent on baseline).

##### Scores

| Entry | Network | Coverage | Fidelity | Cost | Reward |
| --- | --- | ---: | ---: | ---: | ---: |
| Brute force (the network itself) | both seeds | 1 | 100% | 1 | -0.0023 |
| Pure second argmax | first seed | 1 | 98.41% | 0.077 | 0.744 |
| Pure second argmax | second seed | 1 | 94.45% | 0.077 | 0.332 |
| Agent on trained | first seed | 0.973 | 99.60% | 0.108 | 0.818 |
| Agent on trained | second seed | 0.972 | 95.62% | 0.108 | 0.409 |
| Agent on baseline | first seed | 1 | 100% | 0.504 | 0.494 |
| Agent on baseline | second seed | 1 | 100% | 0.515 | 0.483 |

#### RNN second argmax of 10 numbers

##### Architecture and training

- Sizes: 10 numbers per input, 16 hidden units, 10 outputs (the positions 0 to 9).
- Architecture: as for 3 numbers. One forward pass is 2,937 operations.
- Weights: input (16 × 1), recurrent (16 × 16) and readout (10 × 16), 432 numbers.
- Task: output the position of the second-largest of the ten numbers.
- Inputs: ten numbers, each from the standard normal distribution, drawn fresh at every step.
- Loss and optimiser: cross-entropy with AdamW (learning rate 0.01, warmed up over 2¹⁴ steps and then
  decayed to 1% of it, weight decay 0.001), batch 1,024, float32.
- Starting weights: uniform in ±0.25.
- Stopping: after 2²⁰ steps.
- Networks: trained and published by ARC in AlgZoo, five seeds. We use the best one (it gives the
  second-largest on 94.9% of inputs) and the next best as second seed (93.2%). The baseline networks
  are ARC's saved starting weights of the same seeds.

##### Price, runtime, agent, internet, inference

- Price: $22.47 (agent on trained), $33.76 plus one interrupted session of unreported cost, at most $20
  (agent on baseline).
- Runtime: 1 h 22 min (agent on trained), 2 h 1 min without the interrupted session (agent on baseline).
- Agent: Claude Fable 5.1 (`claude-fable-5-1`), thinking effort high (Claude Code's `--effort high`).
- Agent tools: `run_python` (runs Python in a sandbox, with the network's weights) and `submit_entry`
  (hands an entry to the verifier, which scores it).
- Hyperparameters: money cap $20 a round and $80 a run; 5 rounds; entry limits 400 nodes and 256 numbers in constants.
- Internet: not used (no web tools, and the agent's Python runs with the network blocked).
- Inference: about 225 million forward passes (agent on trained), about 0.5 million (agent on baseline).

##### Scores

| Entry | Network | Coverage | Fidelity | Cost | Reward |
| --- | --- | ---: | ---: | ---: | ---: |
| Brute force (the network itself) | both seeds | 1 | 100% | 1 | -0.0023 |
| Pure second argmax | first seed | 1 | 94.84% | 0.014 | 0.436 |
| Pure second argmax | second seed | 1 | 93.24% | 0.014 | 0.271 |
| Agent on trained | first seed | 1 | 99.58% | 0.130 | 0.818 |
| Agent on trained | second seed | 1 | 97.66% | 0.142 | 0.600 |
| Agent on baseline | first seed | 1 | 100% | 0.743 | 0.255 |
| Agent on baseline | second seed | 1 | 100% | 0.804 | 0.193 |

#### RNN second argmax of 10 numbers, ARC's challenge network

##### Architecture and training

- Sizes, architecture, weights, task and inputs: as in the previous section.
- Networks: the first is the 432-weight network that ARC's AlgZoo write-up analyses and sets as its
  challenge. It is published next to the zoo, and its training recipe is not published with it. It
  gives the second-largest on 95.3% of inputs. As the second network we use the zoo's best network of
  this size, the first seed of the previous section (94.9%). The baseline is the one of the previous
  section.

##### Price, runtime, agent, internet, inference

- Price: $27.10 (agent on trained); the baseline is the run of the previous section.
- Runtime: 1 h 33 min (agent on trained).
- Agent: Claude Fable 5.1 (`claude-fable-5-1`), thinking effort high (Claude Code's `--effort high`).
- Agent tools: `run_python` (runs Python in a sandbox, with the network's weights) and `submit_entry`
  (hands an entry to the verifier, which scores it).
- Hyperparameters: money cap $20 a round and $60 a run; 5 rounds; entry limits 400 nodes and 256 numbers in constants.
- Internet: not used (no web tools, and the agent's Python runs with the network blocked).
- Inference: about 90 million forward passes (agent on trained).

##### Scores

| Entry | Network | Coverage | Fidelity | Cost | Reward |
| --- | --- | ---: | ---: | ---: | ---: |
| Brute force (the network itself) | both networks | 1 | 100% | 1 | -0.0023 |
| Pure second argmax | first network | 1 | 95.27% | 0.014 | 0.480 |
| Pure second argmax | second network | 1 | 94.84% | 0.014 | 0.436 |
| Agent on trained | first network | 1 | 99.52% | 0.134 | 0.806 |
| Agent on trained | second network | 1 | 98.86% | 0.145 | 0.724 |
| Agent on baseline | first network | 1 | 100% | 0.728 | 0.270 |
| Agent on baseline | second network | 1 | 100% | 0.743 | 0.255 |

## More implementation details

### The prompt

The system prompt of every session:

```text
You are working on a research task with two tools (run_python and submit_entry) and no others; you have no files of your own. Use the tools as the message says, submit what you find, and end with the notes the message asks for, in exactly that format.
```

Every round's prompt is one template, `prompts/round_v1.md`, the same for every model. Its slots, written
`{{...}}`, are filled in for the run:

````text
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
compared with what is measured.{{EXACT_NOTE}}

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
````

#### Slots filled the same way in every run

- `{{ROUND}}`, `{{ROUNDS_MAX}}`: the round's number, and 5.
- `{{ENTRY_DOC}}`: the entry guide, [`agent_docs/ENTRY.md`](agent_docs/ENTRY.md): the entry language, what
  each operation costs, and an entry's limits.
- `{{INPUTS_NOTE}}`: "16,384 inputs drawn afresh from the data set for each submission".
- `{{REWARD_FORMULA}}`: `reward = coverage * [ (1 - cost) - alpha * (1 - fidelity) ],   alpha = 10`
- `{{REWARD_EXPLANATION}}`:

  > Read it for one input of the region: using the entry there in place of the network saves
  > (1 - cost) of a forward pass, and with probability (1 - fidelity) its answer is wrong, which is
  > charged at 10 forward passes. Coverage turns that into a share of the data set; the inputs
  > outside the region are left to the network and earn nothing.

- `{{SAMPLE_NOTE}}`:

  > Because the verifier scores a sample, two things follow. The fidelity the reward uses is the lower end
  > of its 95% confidence interval on that sample: a lucky sample does not pay, and even an entry that is
  > right on every sampled input is charged a little (the network itself then scores just under 0). And
  > the same entry submitted twice gets two slightly different rewards.

- `{{PYTHON_CALLS}}`, `{{PYTHON_SECONDS}}`, `{{SUBMISSIONS}}`: 40, 300 and 4.
- `{{MAX_NODES}}`, `{{MAX_CONST}}`, `{{VERIFIER_SECONDS}}`: 400, 256 and 120.
- `{{HISTORY}}`: in round 1, "None: this is the first round." In a later round, every earlier submission with its
  receipt, and the notes each earlier round left.
- `{{EXACT_NOTE}}`: nothing. In a baseline run, this paragraph:

  > This run asks one more thing of an entry: that it be exact whatever the weights. When the rounds are
  > over, the entries are also run, unchanged, on several other networks of the same definition and sizes,
  > with other weights, and an entry counts only if it gives each network's output on every input of its
  > region that is checked there. So whatever depends on the weights has to be computed from the weights
  > inside the entry, and every shortcut has to hold for any weights: a rule that holds only for this
  > network makes the entry fail on the others.

#### Slots each model fills in

A baseline run fills them exactly as the run it is compared with.

| Model | `{{SIZES}}` | `{{MODEL_OPS}}` |
| --- | --- | ---: |
| Linear max of 4, Experiment 1 | n_ctx = 4, d_vocab = 64, d_model = 8, outputs: 64 | 671 |
| Linear max of 4, Experiment 2 | n_ctx = 8, d_vocab = 64, d_model = 8, outputs: 64 | 703 |
| Attention max of 4 | n_ctx = 4, d_vocab = 64, d_model = 32, d_head = 32, outputs: 64 | 12,987 |
| RNN second argmax of 3 numbers | n_ctx = 3, d_hidden = 4, outputs: 3 | 78 |
| RNN second argmax of 10 numbers (both sections) | n_ctx = 10, d_hidden = 16, outputs: 10 | 2,937 |

`{{DATA_SET}}`:

- Linear max of 4, Experiment 1:

  > An input is drawn like this. Four tokens are picked, each a whole number from 0 to 63 and equally likely, and they fill the four slots, so no slot is empty. The data set is every input that can be drawn this way, each as likely as any other: 64^4 = 16,777,216 different inputs.

- Linear max of 4, Experiment 2:

  > An input is drawn like this. First a number of tokens is picked from 1 to 8, each equally likely. Then that many tokens are picked, each a whole number from 0 to 63 and equally likely. The tokens fill the first slots and the remaining slots are empty (64). The data set is every input that can be drawn this way, each counted as often as it is drawn: about 2.8 x 10^14 different inputs, far more with eight tokens than with one, yet every number of tokens makes up an eighth of the data set.

- Attention max of 4:

  > An input is drawn like this. First a number of tokens is picked from 1, 2, 3 and 4, each equally likely. Then that many tokens are picked, each a whole number from 0 to 63 and equally likely. The tokens fill the first slots and the remaining slots are empty (64). The data set is every input that can be drawn this way, each counted as often as it is drawn: about 17 million different inputs, most of them filling every slot, yet every number of tokens makes up a quarter of the data set.

- RNN second argmax of 3 numbers:

  > An input is three real numbers, x_0, x_1 and x_2, each drawn on its own from the standard normal distribution (mean 0, standard deviation 1). The data set is every input that can be drawn this way, each weighted by how likely it is to be drawn. There are infinitely many, so the verifier scores a sample.

- RNN second argmax of 10 numbers (both sections):

  > An input is ten real numbers, x_0 to x_9, each drawn on its own from the standard normal distribution (mean 0, standard deviation 1). The data set is every input that can be drawn this way, each weighted by how likely it is to be drawn. There are infinitely many, so the verifier scores a sample.

`{{MODEL_SOURCE}}`, the linear network (Experiments 1 and 2):

```python
"""The network, written in the entry language.

An input is n_ctx slots. A slot holds a token, a whole number from 0 to d_vocab - 1, or it is empty,
which is written as the number d_vocab. An input has at least one token; its tokens fill the first
slots and the empty slots come after them.

The network has three weight matrices and no biases:

    embed     d_model x (d_vocab + 1)    its last column, the empty slot's, is all zeros
    hidden    d_model x d_model
    unembed   d_vocab x d_model

It adds up the columns of embed at the input's slots (an empty slot adds nothing), applies hidden,
applies unembed, and its output is the position of the highest of the d_vocab resulting numbers
(the first one on a tie).
"""


def reference(g, lv1):
    """The network's output on the graph's input. g is an lv1.Graph, lv1 the entry language."""
    h = lv1.take_sum(g.weight("embed"), g.tokens(), axis=1)
    logits = g.weight("unembed") @ (g.weight("hidden") @ h)
    return logits.argmax()
```

`{{MODEL_SOURCE}}`, the attention network:

```python
"""The network, written in the entry language.

An input is n_ctx slots. A slot holds a token, a whole number from 0 to d_vocab - 1, or it is empty,
which is written as the number d_vocab. An input has at least one token; its tokens fill the first
slots and the empty slots come after them.

The network is a transformer with one layer, one attention head and no MLP, no layer norm and no biases.
Its weights:

    embed     (d_vocab + 1) x d_model   a row per token; the last row, an empty slot's, is all zeros
    pos       n_ctx x d_model            a row per slot
    query     d_model x d_head
    key       d_model x d_head
    value     d_model x d_head
    out       d_head x d_model
    unembed   d_model x d_vocab

Each slot's vector is its token's row of embed plus its own row of pos. The network is read at the slot of
the last token given: that slot's vector times query, divided by the square root of d_head, is scored
against every given slot's vector times key (a dot product per slot; empty slots take no part); the softmax
of the scores weights the given slots' vectors times value; the sum, times out, is added to the read slot's
vector; that times unembed gives d_vocab numbers, and the output is the position of the highest (the first
one on a tie).
"""

import math


def reference(g, lv1):
    """The network's output on the graph's input. g is an lv1.Graph, lv1 the entry language."""
    t = g.tokens()
    embed, pos = g.weight("embed"), g.weight("pos")
    n_ctx, empty = pos.shape[0], embed.shape[0] - 1
    x = lv1.take(embed, t, axis=0) + pos                     # a vector per slot
    n = (t != empty).sum()                                    # the number of tokens given
    read = x[n - 1]                                           # the slot of the last token given
    q = (read @ g.weight("query")) * (1 / math.sqrt(g.weight("query").shape[1]))
    scores = (x @ g.weight("key")) @ q
    scores = lv1.where(g.arange(n_ctx) < n, scores, -1e30)    # empty slots take no part
    head = lv1.softmax(scores) @ (x @ g.weight("value"))
    logits = (read + head @ g.weight("out")) @ g.weight("unembed")
    return logits.argmax()
```

`{{MODEL_SOURCE}}`, the RNNs (all three):

```python
"""The network, written in the entry language.

An input is n_ctx real numbers, x_0 to x_(n_ctx - 1). The network reads them one at a time, in that order.
In an entry, g.token(t) is x_t, as a float, and g.tokens() is all of them; the output is still a whole number.

The network is a recurrent network with one layer of d_hidden units, a ReLU and no biases. Its weights:

    input       d_hidden x 1
    recurrent   d_hidden x d_hidden
    readout     n_outputs x d_hidden

Its state h, a vector of d_hidden numbers, starts at zero. At each position t, in order, it becomes

    h = relu(input x_t + recurrent h)

where relu sets every negative number to zero. After the last position, readout h gives n_outputs numbers,
and the output is the position of the highest (the first one on a tie).
"""


def reference(g, lv1):
    """The network's output on the graph's input. g is an lv1.Graph, lv1 the entry language."""
    w, rec = g.weight("input")[:, 0], g.weight("recurrent")
    h = lv1.maximum(w * g.token(0), 0.0)  # the first position: the state before it is zero, and so is recurrent h
    for t in range(1, g.n_ctx):
        h = lv1.maximum(w * g.token(t) + rec @ h, 0.0)
    return (g.weight("readout") @ h).argmax()
```

### Coverage, fidelity and cost, as the verifier computes them

An entry is JSON data: a list of nodes, each one operation on earlier nodes, whose sources are the
input's tokens, the network's weights by name, constants and ranges. Two nodes are its results:
`rule`, the region test, one boolean per input (absent: every input is in the region), and
`predict`, the interpretation, one whole number per input. The agent writes entries in Python with a
builder (`lv1.Graph`), but nothing of an entry is ever run as code: the verifier interprets the JSON
(`level1/lang.py`, `Evaluator`) and counts every operation by the price list `prices-v1`. An
elementwise result costs 1, a matrix product 1 per multiply-add, a reduction over n entries n - 1,
a sort n * ceil(log2 n), reading a number by index 1, and moving data (reshape, slice, concat) 0.
`cond(c, x, y)` computes only the chosen side and costs 1 on top of it; `where(c, x, y)` computes
both. The network itself is written in the same language (`models/<name>/model.py`), and one forward
pass costs it F operations, the same on every input.

The verifier (`level1/verifier.py`) picks n inputs: every input of the data set when it can be
listed (linear max2 asked only about pairs: 4,096 inputs), otherwise a sample drawn from D (16,384 inputs in the pilots), afresh
for each of the agent's submissions, from a seed nobody chose. It runs the network on them for its
outputs M(x), then the entry, with `lang.measure`:

```python
# level1/lang.py, measure(): one pass over the n inputs
for x, truth in zip(inputs, model_answers):     # model_answers: the network's outputs M(x)
    inside, answer, c_rule, c_pred = ev.run(x)  # the region test, then the interpretation if inside
    rule_ops += c_rule                          # the region test is paid for on every input
    if inside:
        inside_n += 1
        predict_ops += c_pred                   # the interpretation only on the region
        agree += int(answer == int(truth))
coverage = inside_n / n
fidelity = agree / inside_n
cost = (predict_ops + rule_ops + ev.build_cost) / (inside_n * model_ops)
```

- **Coverage** is `n_in_region / n_inputs`, the share of the inputs scored on which the region test
  is true.
- **Fidelity** is `n_agree / n_in_region`, the share of the region on which the interpretation's
  output equals the network's output exactly: agreement of outputs, not of distributions. On a
  sample the verifier also records `fidelity_low`, the lower end of the 95% Wilson interval for
  `n_agree` successes in `n_in_region` draws (`lang.wilson_lower`, z = 1.96), and the reward uses it,
  so that a lucky sample does not pay. When every input is checked, `fidelity_low` is the fidelity.
- **Cost** is everything the entry computed, per input of its region, in forward passes. A node that
  does not depend on the input is computed once, the first time it is needed, and counted once, in
  `build_ops`. The rest is counted per input: the region test on every input, the interpretation on
  the inputs of the region, and a value the region test already computed for an input is not counted
  again. The receipt keeps three parts, `r = predict_ops / (n_in_region * F)`,
  `r_rule = rule_ops / (n_inputs * F)` and `b = build_ops / (n_inputs * F)`, so that
  `cost = r + (r_rule + b) / coverage`. The build is spread over the inputs scored.

Every receipt keeps these counts and parts, so a run can be scored again under another reward
without running anything. With the entry the agent also states a claimed fidelity; the receipt
records it and the overclaim, the claim minus the measured fidelity when positive. An entry may have
at most 400 nodes and 256 numbers in constants, so a table of the network's outputs cannot be
written into it; a table computed from the weights in the build is allowed, and paid for there. The
verifier gives an entry 120 seconds on its inputs.

### Checking what the agent did

Nothing the agent says is taken on trust: what it did is recorded, fenced in and checked.

- **Before any model is asked**, the hygiene check (`level1/hygiene.py`) searches everything the
  agent can reach (the prompt, the tools' descriptions, the files in its sandbox, names) for the
  words in `withheld/<model>/task.yaml` that would say what the network does, and stops the run on a
  hit.
- **Everything is recorded as it happens**: each round's prompt and the agent's answer, every tool
  call with its result, every program it ran with what it printed, the CLI's whole stream (the
  agent's text between its tool calls included) and every receipt (see "What a run writes"). Each
  round ends with notes under fixed headings: what the network does and on which inputs, how that
  was found (the outputs, the weights, or both), the best entry and where its saving comes from,
  the claims, and what to try next. The next round knows only these notes and the receipts.
- **The agent's code runs in a sandbox.** A guard inside Python refuses files outside the
  program's folder, sockets, other processes, signals and native code, and writes every refusal to
  the program's audit record. The guard shares a process with the code it watches, so for a run that
  counts the system's own sandbox is the wall: `seatbelt` on a Mac, a container or Landlock on
  Linux. Every run starts with a probe that tries real escapes, with and without the guard (reading
  the verifier, the second network's weights and the withheld task file in several ways, writing
  outside the folder, opening connections), and judges them by evidence outside the sandbox
  (`docs/SANDBOX.md`).
- **The report audits every round**: how many programs ran and for how long, every attempt the
  guard refused, every program whose text reaches for files or the system (`open(`, `os.`,
  `subprocess`, sockets, `eval`, the word withheld and others, `report.REACHES`), and every program
  whose audit record is incomplete.
- **Entries are data, not code.** The verifier interprets them, so an entry can do no work that is
  not counted and can read nothing but the input and the weights of the network it is scored on. The
  locked entry is the submission byte for byte; its hash is checked.
- **Claims are set against measurements.** With every entry the agent states the fidelity it
  expects on this network and on a second network; both are kept next to what is measured. Every
  submission is also scored on the second network after its round, on one fixed sample, and the
  agent never sees those scores (`seed_scores.jsonl`): they show round by round whether its entries
  drift towards the first network's quirks.

In the nine runs so far the guard refused nothing, the probe got nothing out of the seatbelt
sandbox, and no claimed fidelity was above the measured one by more than 0.0013. Two early pilots
ran in process mode, linear max4 with the guard and linear max2 before the guard existed; linear max2's 52
programs were read by hand. The notes are the agent's own words: the measurements check the entries
built on them, not the words.

## Getting started

### Install

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest                      # about a minute; no model is called
    python -m level1.sandbox info         # which sandbox modes work on this machine

Use a virtual environment of this repository's own. Outside a container, the hygiene check also
reads the path of the Python that runs the agent's code, because a traceback shows that path to the
agent: an environment that lives in a folder named after the proof pipeline is refused.

Rounds run under your own Claude Code login, so the CLI (`claude`) has to be installed and signed
in. To bill an API key instead, see "On an API key" below. To put an OpenAI model in the agent's
place, through the Codex CLI under a ChatGPT login or through OpenAI's API on a key, see
`docs/OPENAI.md`.

### Run a pilot

Each model has a run config, `config/pilot/<name>.yaml`, and a small smoke config next to it,
`config/pilot/<name>-smoke.yaml`. First a dry run, with scripted rounds in place of a model (two
minutes):

    python -m level1.loop run --config config/pilot/lin-a-1to4.yaml --fake tests/fake_run_1to4 --run-id dry-run

Then the smoke run, one short round, to see that the login, the tools, the sandbox and the verifier
work together on this machine:

    python -m level1.loop run --config config/pilot/lin-a-1to4-smoke.yaml

Then the pilot proper, at most 5 rounds of up to 90 minutes each:

    python -m level1.loop run --config config/pilot/lin-a-1to4.yaml

If it stops (a usage limit, a dropped connection, a closed laptop), continue where it stopped:

    python -m level1.loop resume --run ~/mech-interp-level1-runs/pilot-lin-a-1to4

The pilot on pairs runs the same way with `lin-a-smoke.yaml`, `lin-a.yaml` and `tests/fake_run`, and
`tests/fake_run_any` is one scripted round that fits every model. Run folders live outside the
repository, in `~/mech-interp-level1-runs/<run_id>/` by default. On a Mac, run a long pilot under
`caffeinate -is`, with the lid open and on power: a session whose machine sleeps loses its prompt
cache and pays to write it again.

### With its baseline

A run is read against the baseline of its shape when one is frozen (`docs/DESIGN.md`, section 10).
Three ways to start:

    python -m level1.loop run --config config/pilot/att-c.yaml                   # the run only
    python -m level1.loop baseline --config config/pilot/att-c.yaml              # its baseline only
    python -m level1.loop run --config config/pilot/att-c.yaml --with-baseline   # both, side by side

A baseline is made once for a shape and its conditions and serves every run that matches; a
finished run gets one with `python -m level1.loop baseline --run <run folder>`. Each command takes
the same `--sandbox`, `--auth` and `--budget-usd` as `run`. The baseline models' own pilot configs
make runs named `pilot-<name>-exact`; baseline runs made before their prompt asked for an exact
entry serve no run.

### On an API key

A run can be billed to an API key instead of your login (`auth: api_key` in the run config, or
`--auth api_key` with any config). Put the key in `CPL_ANTHROPIC_KEY`, never in `ANTHROPIC_API_KEY`,
for example by loading the file you keep it in; `CPL_ANTHROPIC_WORKSPACE` takes a workspace ID if
the key is not scoped to one workspace. Then

    python -m level1.loop run --config config/pilot/lin-b-api.yaml --sandbox seatbelt

Before the first round the key is checked against the API's list of models, which costs nothing.
The run has a budget, by Claude Code's own count: `budget.run_usd` for the run (60 dollars) and
`budget.round_usd` for one round (20). Every session starts with what is left as its cap. A round
that reaches its cap ends with what it submitted; the rounds end when the run's budget is spent.
Claude Code checks its spending after every answer, so a session can go over its cap by one answer.
The run folder's `spend.jsonl` has every session's cap and cost. A session that reported no cost
counts at its cap; one the loop never saw end (the loop was killed) counts in the run's total only,
so its round gets its whole budget again when resumed. To go on with more:

    python -m level1.loop resume --run ~/mech-interp-level1-runs/pilot-lin-b-api --budget-usd 90

The CLI's `--bare` mode, which this uses, reads no login, so an API run cannot draw on a
subscription, and a login run cannot bill a key: the CLI's first event says what it billed, and a
session that does not fit stops the run.

### The sandbox

In every mode a guard inside Python refuses the agent's code files outside its folder, the network
and other programs, and records every attempt; the report lists them. The guard shares a process
with the code it watches, so for a run that is meant to count add `--sandbox hardened` (Docker or
Linux) or, on a Mac, `--sandbox seatbelt`. Try a mode first with the probe, which tries to escape
it and reports (`docs/SANDBOX.md`):

    python -m level1.sandbox probe --model models/lin-a --mode seatbelt

### Other commands

    python -m level1.loop finish --run <run folder>              # lock, seed check and report again; scores
                                                                 # any submission not yet on the second network
    python -m level1.loop rescore --run <run folder> --alpha 1   # the records under another reward
    python -m level1.report --model models/lin-a --alpha 1       # the reference entries under a reward
    python -m level1.report --model models/lin-a-1to4 --sample-size 16384
    python -m level1.verifier --model models/lin-a --network lin-a-s0 --entry entry.json
    python -m level1.verifier --model models/lin-a --network lin-a-s0 --entry entry.json --check sample
    python -m level1.sandbox build                               # the container image, if Docker runs
    python -m level1.sandbox probe --model models/lin-a          # try to escape the sandbox, and report

### What a run writes

Everything is written as it happens, so a run can stop and resume, and can be read afterwards:

- every prompt (`round_k/prompt.md`) and the agent's answer (`response.md`);
- every tool call with its result (`session/tools.jsonl`);
- every program the agent ran, in full, with what it printed (`session/python/NNN.py`, `.txt`),
  and what the guard refused it (`NNN.audit.json`);
- everything the CLI emitted, the agent's text between its tool calls included
  (`round_k/stream_NN.jsonl`);
- every submission's receipt, with the parts measured;
- every scored submission on the second network (`seed_scores.jsonl`);
- billed to an API key, every session's cap and cost (`spend.jsonl`);
- the report: `report.md` and `report.json`.

The complete records of the 11 runs behind the Results section are published, one archive per run, in
the release [Run records for the README's experiments](https://github.com/WFJKK/Mech-Interp-Agent-level1/releases/tag/run-records-2026-11-19).

### Repository layout

    level1/            the pipeline
      lang.py          the entry language: operations, prices, the interpreter, the builder, data sets
      verifier.py      measures an entry; the only judge
      rewards.py       reward functions
      networks.py      model folders: the networks, the network written as an entry, what the agent is given
      prompt.py        a round's prompt
      session.py       one agent session: the two tools, budgets, log
      mcp_server.py    the session's tools as an MCP server, for the Claude Code and Codex CLIs
      loop.py          rounds, lock, seed check; the command line
      baseline.py      the baseline of a run's shape: found, run, frozen, and a run read against it
      report.py        the report; the reference entries under any reward
      hygiene.py       the check that the agent is not told what the model does
      backends/        who answers a round: claude_code, codex, openai_api (docs/OPENAI.md), fake
      budget.py        what a run billed to an API key may spend, and what it has spent
      run_config.py    run configs: defaults, then what a config extends, then the config
      sandbox.py, runner.py, harness.py    where the agent's code runs
      util.py          small helpers
    models/lin-a/           linear max2, asked only about pairs of tokens
    models/lin-a-1to4/      linear max2: the same two networks, asked about inputs of 1 to 4 tokens
    models/lin-b/           linear max4, asked about inputs of 1 to 8 tokens
    models/lin-c/, lin-d/   linear min2 and linear min4
    models/att-a/ .. att-d/   att max2, att min2 (2 slots), att max4, att min4 (4 slots)
    models/lin-e/, lin-f/, att-e/   baseline models of the shapes of linear min2, linear min4 and att max4
    models/rnn-a/ .. rnn-c/   one-layer ReLU recurrent networks on 2, 3 and 10 real numbers, five seeds each
    models/rnn-d/ .. rnn-f/   their baseline models
    withheld/<model>/       what it was trained to do, and reference entries (never shown to the agent)
    withheld/rnn/           where the recurrent networks come from: the original files and the converter
    agent_docs/        the entry guide the agent reads
    prompts/           the round's prompt template
    config/            defaults, the pilots' run configs and their small smoke runs
    sandbox/           the container image (Dockerfile)
    tests/             the tests, with each known way to game the score as a test
    docs/              DESIGN.md, REWARD.md, SANDBOX.md; figures/: the flow diagram, the frontier figure and their sources

`CLAUDE.md` has the working rules for changing this repository.
