# OpenAI models as the agent

A run's agent can be an OpenAI model in two ways. Both give the model what a Claude Code run gives:
the run's system prompt, the round's prompt and the session's tools. Everything after the agent
(the sandbox, the verifier, the reward, the lock, the report) is the same code.

| | `backend: codex` | `backend: openai_api` |
|---|---|---|
| What runs | OpenAI's Codex CLI, headless | this pipeline, talking to OpenAI's API |
| Who pays | your ChatGPT login (`auth: login`) | an API key (`auth: api_key`), per token |
| Spending cap | none: a plan is not billed per call | the run's budget, counted here |
| What the model has besides the run's | a few inert tools of Codex's own (below) | nothing |
| Turns | `session.max_turns` bounds tool calls | one answer is one turn |

Neither was run against a real model while it was built: the repository's rule is that building and
testing call no model. Every path was run against stand-ins, the Codex CLI itself included (version
0.172.0, against a stand-in for the API on the same machine). So start with the two smoke configs
below, and read their records before anything longer.

## Through the API (`openai_api`)

```bash
export CPL_OPENAI_KEY=...            # the key, in this terminal only; never OPENAI_API_KEY
python -m level1.loop run --config config/pilot/lin-a-openai-smoke.yaml
```

Any other config runs the same way with the switches on the command line, under a run id of its
own:

```bash
python -m level1.loop run --config config/pilot/lin-b.yaml --backend openai_api --auth api_key \
    --api-key-env CPL_OPENAI_KEY --agent-model gpt-6.2-astra --run-id pilot-lin-b-gpt --budget-usd 40
```

What happens (`level1/backends/openai_api.py`):

- Each turn is one `POST /responses` with the conversation so far: the system prompt as a developer
  message, the round's prompt, and every item the model answered with, its tool calls followed by
  their results. Nothing is stored at OpenAI (`store: false`); the model's reasoning comes back
  encrypted and is sent along again, as OpenAI's guide asks for a conversation kept by the client.
- The session's tools are the only tools, and they run in this process. There is no file and no web.
- `effort` is sent as `reasoning.effort`. The API takes none, minimal, low, medium, high, xhigh and
  max; which of them a model supports is the model's.
- The key is read from the variable `api_key_env` names. Its value goes into the request's
  Authorization header and nowhere else: not into a file, a log or an error message. A name that
  reads like an Anthropic key's variable is refused, so that no run sends one company's key to the
  other. Before the first round the key is checked against the model list, which costs nothing.

### The budget

`budget.run_usd` and `budget.round_usd` hold as they do under Claude Code (`level1/budget.py`,
`spend.jsonl`): every try starts with what is left as its cap, a round at its cap ends with what it
submitted, and the rounds end when the run's budget is spent. The difference is who counts. Claude
Code reports what a session cost. Here the backend counts it, from the tokens every answer reports:

    cost = input x p_in + cached x p_cached + output x p_out

per million tokens, at `openai.prices.<agent_model>` in `config/defaults.yaml`, with the
long-context prices for a request over that line. The prices there are OpenAI's pricing page as of
25 November 2026, for five models. Two consequences:

- A model without an entry cannot be run through the API: add its prices first.
- The count is this pipeline's, not the invoice. After the first run, compare `spend.jsonl` with
  OpenAI's usage page; if they differ, a price in the config is out of date.

The cap is checked after every answer, so a try can go over it by one answer, as with the CLI. The
tools that last answer called still run, since a submission in it should stand.

### Long sessions

The whole conversation is sent with every turn, so its cost per turn grows with the session; the
prompt cache takes most of that (cached input is a twentieth of the price for the gpt-6.2 models).
A conversation that outgrows the model's context ends the round with what it submitted
(`stop_reason: context_limit`). To have the API compact the conversation instead, add its field to
every request:

```yaml
openai:
  request:
    context_management: [{type: compaction, compact_threshold: 200000}]
```

`openai.request` passes any other field of the API the same way (`service_tier`,
`max_output_tokens`). It was not tried against the real API.

### When something fails

A rate limit, an overload or a dropped connection is waited out and the same turn asked again; the
conversation is held here, so nothing of the session is lost. A refused key, an account that cannot
pay, a model the key cannot use or a request the API does not take stops the run cleanly, and
`resume` goes on later with a new try that is told what the session already used. An answer from
another model than the run's stops the run. A refusal is recorded as one.

## Through the Codex CLI (`codex`)

```bash
npm install -g @openai/codex         # once
codex login                          # once, with ChatGPT
python -m level1.loop run --config config/pilot/lin-a-codex-smoke.yaml
```

or, for any config:

```bash
python -m level1.loop run --config config/pilot/lin-b.yaml --backend codex \
    --agent-model gpt-6.2-astra --run-id pilot-lin-b-codex
```

Codex is a coding agent. Left alone it gives the model a shell, file edits, the web, images,
sub-agents, skills, the apps connected to your ChatGPT account and long instructions of its own. A
run needs the opposite, and one thing above all: the model must not read files, because this
repository says what its networks were made for. So `level1/backends/codex.py` starts the CLI with

- an empty temporary folder as its working directory and a read-only sandbox;
- your config and rules not loaded (`--ignore-user-config`, `--ignore-rules`), nothing saved
  (`--ephemeral`);
- its shell, image, web, app, plugin, browser, computer, skill, memory and sub-agent features
  switched off;
- its own instructions replaced by the run's system prompt, and its notes on the environment,
  permissions, apps and skills left out;
- the session's tool server attached, its two tools given to the model directly.

### What is left, and how a run makes sure

What remains of Codex's own depends on the model and is the CLI's decision. With version 0.172.0:

| Model family | Besides the session's tools |
|---|---|
| gpt-5.6, gpt-6 | a script tool (JavaScript that can call other tools; no file system, no network), a tool to wait on it, a tool to message the user, and a short note about sub-agents |
| gpt-5.5 | a file-patch tool, which the read-only sandbox refuses, and three tools that list a tool server's resources, of which there are none |

None of these reads a file or reaches the web. Since another version or model may differ, a run
does not take that on trust:

1. **Before the first round** the CLI is run once against a stand-in for the API on this machine,
   with the run's own switches. No model is asked and nothing leaves the machine. What the CLI
   sends is read: every tool it offers, directly or through its script tool, and every text besides
   the run's. Anything outside a short list of known inert tools stops the run there, with the
   reason; what was seen is kept as `codex_check.json` in the round's folder.
2. **After every session** its events are read. A shell command, a file change, a web search, a
   sub-agent or a call to another tool server stops the run.

The first check sees what the CLI sends to a custom address. Under your login the CLI talks to
OpenAI's own address, where more could in principle be switched on from OpenAI's side; the second
check is what covers that case.

### Your personal instructions

The CLI gives the file `AGENTS.md` of its home folder (`~/.codex`) to every model it runs, and no
switch leaves it out. A run refuses a home that has one. Give the runs a home of their own:

```yaml
codex_home: ~/.codex-level1
```

```bash
CODEX_HOME=~/.codex-level1 codex login
```

### Limits of this route

- **Login only.** The CLI has no spending cap, so `auth: api_key` is refused: a key is billed
  through `openai_api`. The CLI's environment has `OPENAI_API_KEY` and `CODEX_API_KEY` removed and
  it is told to use the ChatGPT login only; a login that is itself an API key is refused.
- **Turns.** The CLI has no turn limit. `session.max_turns` bounds the tool calls of one try, and
  the session's own budgets (Python calls, submissions) and its wall clock hold as always.
- **The model's name is not confirmed.** The CLI's events do not say which model answered, so the
  records carry the name the run asked for.
- **Effort.** Codex's models take low, medium, high, xhigh, max and, for some, ultra.
- **Usage limits** of the plan are waited out with growing pauses, as under Claude Code.

## Comparing with a Claude run

The prompt, the tools, the session's budgets and everything measured are the same, so rewards are
comparable. What differs is the harness around the model: Claude Code's own turn handling, Codex's
leftover tools, or none at all through the API. The baseline of a run's shape is kept per backend
and model (`level1/baseline.py`), so a run under another backend gets a baseline of its own.
