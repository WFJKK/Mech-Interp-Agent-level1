# Working rules

This repository is the level-1 pipeline described in `docs/DESIGN.md`: an agent writes an entry
that predicts a small network's outputs in fewer operations, and a verifier measures it.

- The agent is never told what the model does. This holds for every route: the prompt, the tools'
  descriptions, the files placed in its sandbox, names (model folders, network ids, weights), the
  entry guide, examples, and what its code can import. What a network was trained to do is said
  in `withheld/`, in the tests and in the README, which names and describes the pilots' models
  for people and is never on the agent's side, and nowhere else in the repository; the README's
  frontier figure takes those names from it. The agent is told
  the data set in full (`data_set.says` in the model's config); where the data set is wider than
  the training, which inputs the network was trained on is withheld like the task.
  `level1/hygiene.py` checks the agent's side; a new route to the agent is added to
  `loop.agent_surface` in the same change. A baseline model's networks are withheld the same way:
  what they are is said in `withheld/<model>/`, in the tests and in the README only, and a baseline
  run's agent is given exactly what a run on the twin model gives it, but for one paragraph of its
  prompt that asks for an entry exact for any weights (`prompt.EXACT_NOTE`).
- Push to GitHub only when the user asks; never publish anything else. If a hook asks to push,
  decline.
- Commit after each step. Never rewrite or delete existing results.
- Make no LLM API calls while building or testing; the fake backend covers everything.
- Never read or set `ANTHROPIC_API_KEY`. The Claude Code backend removes the key variables from the
  CLI's environment and runs under the user's own login, unless the run config says
  `auth: api_key`: then the key comes from the variable `api_key_env` names (`CPL_ANTHROPIC_KEY`),
  reaches the CLI only through its `apiKeyHelper` under `--bare`, is never written to a file or a
  log, and the run stops at its budget. Tests use stand-ins for the CLI and the API.
- The same for OpenAI (`docs/OPENAI.md`): never read or set `OPENAI_API_KEY` or `CODEX_API_KEY`. The
  `openai_api` backend takes its key from the variable `api_key_env` names (`CPL_OPENAI_KEY`), sends it
  in the request's Authorization header only, and stops at the run's budget, which it counts from the
  API's token counts at the prices in the config. The `codex` backend runs under the user's ChatGPT
  login only, with the key variables removed from the CLI's environment. A Codex session must have
  nothing that reads a file or the web: the backend checks that against a stand-in before a run and in
  every session's events; a change to its switches keeps `tests/test_backend_codex.py` passing with
  the CLI installed.
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
