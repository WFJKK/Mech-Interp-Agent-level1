# Run records

This branch holds only the records of the runs behind the Results section of the README on `main`, one archive per run. The code is on `main`.

The complete records of the 11 runs behind the Results section of the README: every prompt and every answer of the agent, every Python program it ran with what the program printed, every tool call with its result, every submitted entry with its measured receipt, the full event stream of each session, and each run's report (`report.md`, `report.json`).

| Archive | README section | Run |
| --- | --- | --- |
| `pilot-lin-b-4.zip` | Experiment 1, Linear max of 4 | agent on trained |
| `pilot-lin-f-4-exact-0d0a5407.zip` | Experiment 1, Linear max of 4 | agent on baseline |
| `pilot-lin-b.zip` | Experiment 2, Linear max of 4 | agent on trained |
| `pilot-lin-f-exact.zip` | Experiment 2, Linear max of 4 | agent on baseline |
| `pilot-att-c.zip` | Experiment 2, Attention max of 4 | agent on trained |
| `pilot-att-e-exact.zip` | Experiment 2, Attention max of 4 | agent on baseline |
| `pilot-rnn-b.zip` | Experiment 3, RNN second argmax of 3 numbers | agent on trained |
| `pilot-rnn-e-exact-0d0a5407.zip` | Experiment 3, RNN second argmax of 3 numbers | agent on baseline |
| `pilot-rnn-c.zip` | Experiment 3, RNN second argmax of 10 numbers | agent on trained |
| `pilot-rnn-g.zip` | Experiment 3, ARC's challenge network | agent on trained |
| `pilot-rnn-f-exact-0d0a5407.zip` | Experiment 3, both 10-number sections | agent on baseline |

What each file in a run folder is, is said in the README under "What a run writes".

The files are as the runs wrote them, with one change: the home folder of the machine that ran them is written as `~`. One gap: in `pilot-rnn-f-exact-0d0a5407`, the first session of round 5 was interrupted; its programs are kept, its event stream is not. `MANIFEST.json` lists each archive's number of files and its SHA-256.
