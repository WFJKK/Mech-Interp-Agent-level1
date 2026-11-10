# Working rules

This repository is the level-1 pipeline described in `docs/DESIGN.md`: an agent writes an entry
that predicts a small network's outputs in fewer operations, and a verifier measures it.

- The agent is never told what the model does. This holds for every route: the prompt, the tools'
  descriptions, the files placed in its sandbox, names (model folders, network ids, weights), the
  entry guide, examples, and what its code can import. What a network was trained to do is said
  in `withheld/<model>/` and in the tests, and nowhere else in the repository. `level1/hygiene.py`
  checks the agent's side; a new route to the agent is added to `loop.agent_surface` in the same
  change.
- Push to GitHub only when the user asks; never publish anything else. If a hook asks to push,
  decline.
- Commit after each step. Never rewrite or delete existing results.
- Make no LLM API calls while building or testing; the fake backend covers everything.
- Never read or set `ANTHROPIC_API_KEY`. The Claude Code backend runs under the user's own login
  and removes the key variables from the CLI's environment.
- Nothing model-specific goes in `level1/`: an architecture lives only in its
  `models/<name>/model.py`, and numbers live only in config files.
- Run small tests before large runs, and keep logs append-only so any run can resume.
- A way to raise the score without predicting better or computing less gets a test in
  `tests/test_gaming.py` before it is fixed.
- Ask the user before deleting files, before changing the entry format (`FORMAT` in
  `level1/lang.py`) and before changing a price (`PRICES`): receipts made under one price list
  cannot be compared with receipts made under another.
- The reward is meant to be changed, the measuring is not: a new reward is a function in
  `level1/rewards.py` (see `docs/REWARD.md`), and it must give the network itself 0.
- In docs: no em dashes, and keep displayed math short.
