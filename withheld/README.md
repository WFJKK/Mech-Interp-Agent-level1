# What the networks were trained to do

Withheld from the agent: no tool reaches this folder, and only this folder and the tests may say
what is written here, so that nothing the agent can reach says it (`level1/hygiene.py` and the tests
check it). Each model's `task.yaml` has the full record: the task, where its networks come from, how
they were trained, how often they follow the task, and the words the hygiene check looks for.

| Model | Network | Trained to output | Trained on | Asked about |
| --- | --- | --- | --- | --- |
| lin-a | linear, width 8 | the larger of its tokens (max of 2) | exactly 2 tokens | every pair of tokens: its training inputs |
| lin-a-1to4 | lin-a's two networks | max of 2 | exactly 2 tokens | 1 to 4 tokens |
| lin-c | linear, width 8 | the smaller of its tokens (min of 2) | exactly 2 tokens | 1 to 4 tokens |
| lin-b | linear, width 8 | the largest of its tokens (max of 4) | exactly 4 tokens | 1 to 8 tokens |
| lin-d | linear, width 8 | the smallest of its tokens (min of 4) | exactly 4 tokens | 1 to 8 tokens |
| lin-b-4 | lin-b's two networks | max of 4 | exactly 4 tokens | exactly 4 tokens, in 4 slots: its training inputs |
| att-a | transformer, width 32 | max of 2 | exactly 2 tokens | 1 or 2 tokens |
| att-b | transformer, width 32 | min of 2 | exactly 2 tokens | 1 or 2 tokens |
| att-c | transformer, width 32 | max of 4 | exactly 4 tokens | 1 to 4 tokens |
| att-d | transformer, width 32 | min of 4 | exactly 4 tokens | 1 to 4 tokens |
| rnn-a, rnn-b, rnn-c | ReLU RNN, 2, 4 and 16 units | the position of the second-largest of 2, 3 and 10 numbers (2nd argmax) | standard normal numbers | the same: its training inputs |
| lin-e, lin-f, att-e | the shapes of lin-c, lin-d and att-c | nothing: never trained | | as their twins |
| lin-f-4 | lin-f's networks, in lin-b-4's design and data set | nothing: never trained | | as lin-b-4 |
| rnn-d, rnn-e, rnn-f | the shapes of rnn-a, rnn-b and rnn-c | nothing: never trained | | as their twins |

- **Linear**: 64 tokens, each with an embedding column; the columns of an input's tokens are added up,
  passed through a hidden layer of width 8 and an unembedding, and the output is the argmax. Trained
  by the proof pipeline's recipe: lin-a's two networks come from the proof pipeline, lin-b, lin-c and
  lin-d were trained here (`train.py` in their folders).
- **Transformer**: Gross et al.'s one-layer, one-head, attention-only transformer, 64 tokens, width
  32, read at the last token given. att-a, att-b and att-c come from the proof pipeline's zoo; att-d
  was trained here by the same recipe.
- **Recurrent**: one layer of ReLU units without biases, from ARC's AlgZoo, which publishes such
  networks uninterpreted as a challenge to explain them (`rnn/convert.py`).
- **Baseline models**: each network is the state its twin's trainer starts from, kept as drawn
  (for the recurrent ones, AlgZoo's own saved starting states of the same seeds). Nothing learned is
  in them, so what an agent achieves on them is generic.

The trained networks follow their task on nearly all of their training inputs (99.7% to 99.99%; the
recurrent networks of the pilots 93% to 100%). Off them, they follow it far less, and differently per
seed: att-c's first
network gives the largest token on 17% of 3-token inputs, its second on 98%. Finding where a
network's behaviour can be described cheaply is the agent's work, and it is told none of this.
