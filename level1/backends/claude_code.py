"""The Claude Code backend: one headless run of the official Claude Code CLI per round.

Each round runs

    claude -p --tools "" --output-format stream-json --verbose --no-session-persistence
           --setting-sources "" --disable-slash-commands --strict-mcp-config
           --mcp-config <round>/mcp.json --allowedTools mcp__lv1__run_python,mcp__lv1__submit_entry
           --max-turns <n> --system-prompt <short> --model <agent_model> --effort <effort>

with the prompt on stdin and an empty temporary folder as the working directory: no built-in tools, no
files, no project settings, and one MCP server (mcp_server.py) that serves the session's two tools. The
tools run in that server's process; their budgets and log are the session's.

Who pays (`auth` in the run config):

- login, the default: the CLI's own login, so a run can draw on a Claude subscription (Pro or Max). The
  CLI's environment has ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN removed, so it cannot fall back to
  billing a key. Whether a subscription may be used this way is governed by Anthropic's terms for Claude
  Code, which have changed over time: check the current policy for your account before relying on it.
- api_key: an API key, billed per token. The key stays in the variable named by api_key_env
  (CPL_ANTHROPIC_KEY), never ANTHROPIC_API_KEY, which is removed here too. The CLI runs with --bare, which
  never reads a login (no OAuth, no keychain), and gets the key from an apiKeyHelper that prints that
  variable: the key's value is never on a command line or in a file, and the tool server drops it from its
  environment. Before the first round the key is checked against the API's model list (free), so a
  refused key stops the run at once; the CLI itself retries a key its helper gave for about four minutes.
  Every try gets --max-budget-usd, the smaller of what is left of the round's budget and of the run's;
  budget.py keeps the count in the run folder, so it holds across tries, rounds and resumes. A round whose
  budget is spent ends there and what it submitted stands; the loop ends the rounds when the run's is
  spent. The CLI checks its spending after every answer, so a try can go over its cap by one answer.

Every try's init event says what the CLI billed (apiKeySource): a try whose source does not fit the mode
stops the run. CLAUDE_CODE_NO_MODEL_FALLBACK is set, so the CLI cannot answer with another model when the
requested one declines. An answer that still shows another model stops the run; a refusal is recorded as
one, and the loop counts refusals. Usage-limit and overload errors are retried with growing waits; a login
or key problem, or an account that cannot pay, stops the run, which `resume` continues later. The session's
wall clock is session.seconds: when it runs out, what the session submitted stands. Each try's whole event
stream is kept in the round's folder (stream_NN.jsonl, numbered on from the last one there, so a resumed
round adds its tries rather than writing over them). A try that finds the session already used (a retry
after a dropped connection, or a resumed run) gets the same prompt with a note at its end: the budgets as
they stand, the entries saved and the submissions made before the cut (prompt.restart_note).

Taken from the proof pipeline (core/agent/backends/claude_code.py and core/open_lane/backends.py of
Compact-Proof-Agent), merged into one class.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .. import budget, prompt
from ..mcp_server import SERVER_NAME
from ..session import TOOLS
from ..util import REPO_ROOT, atomic_write
from . import BackendError, Response, StopRun

SYSTEM_PROMPT = (
    "You are working on a research task with two tools (run_python and submit_entry) and no others; you have no "
    "files of your own. Use the tools as the message says, submit what you find, and end with the notes the "
    "message asks for, in exactly that format."
)
KEY_VARIABLES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
AUTH_MODES = ("login", "api_key")
# What a try's init event may say the CLI billed (apiKeySource), by auth mode. Under a login: no key (a
# subscription), or the key a Console login manages, which is billed per token. With api_key: the helper.
KEY_SOURCES = {"login": ("none", "/login managed key"), "api_key": ("apiKeyHelper",)}
BILLED_SOURCES = ("/login managed key", "apiKeyHelper", "ANTHROPIC_API_KEY")
VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
WORKSPACE = re.compile(r"wrkspc_[A-Za-z0-9_-]+")
API_URL = "https://api.anthropic.com"
# Unless this is set, the CLI answers with another model when the requested one declines or is unavailable,
# and says so only in its events and the result's modelUsage: the records would credit the run's model with
# another model's work. A run checks every answer for a swap anyway.
NO_MODEL_FALLBACK = {"CLAUDE_CODE_NO_MODEL_FALLBACK": "1"}
SWAP_EVENTS = ("model_fallback", "model_refusal_fallback", "model_consent_fallback")
REFUSAL_EVENT = "model_refusal_no_fallback"
LIMIT_WORDS = ("usage limit", "rate limit", "rate_limit", "overloaded", "too many requests", "capacity", "529", "429")
LOGIN_WORDS = ("not logged in", "login", "authenticat", "unauthorized", "invalid api key", "401", "403")
# An API account that cannot pay for more: no credit left, or a spend limit reached. Waiting does not help.
BILLING_WORDS = ("credit balance", "billing", "regain access", "api usage limits", "spend limit")
# The run's model cannot be used: it does not exist, or the login or key has no access to it.
MODEL_WORDS = ("issue with the selected model", "may not have access to it", "not_found_error")
STDERR_TAIL = 2000


def parse_output(out: str) -> tuple[dict | None, list[dict]]:
    """The CLI's result event and every assistant message's text, in order (stream-json: one object per line)."""
    result, messages = None, []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        if ev.get("type") == "assistant":
            msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
            content = msg.get("content")
            parts = []
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append(block["text"])
            if parts:
                messages.append({"text": "".join(parts), "stop_reason": msg.get("stop_reason")})
        elif ev.get("type") == "result":
            result = ev
    return result, messages


def model_events(out: str) -> list[dict]:
    """The CLI's events about the model itself, in order: a swap to another model, or a refusal."""
    events = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        if ev.get("type") == "system" and ev.get("subtype") in (*SWAP_EVENTS, REFUSAL_EVENT):
            keep = ("subtype", "original_model", "fallback_model", "api_refusal_category", "api_refusal_explanation", "scope")
            events.append({k: ev[k] for k in keep if k in ev})
        elif ev.get("type") == "assistant" and isinstance(ev.get("message"), dict) and ev["message"].get("stop_reason") == "refusal":
            events.append({"subtype": "refusal"})
    return events


def models_used(data) -> dict:
    """Output tokens per model, from the result's modelUsage (empty when the CLI gave none)."""
    usage = data.get("modelUsage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return {}
    return {m: int((u or {}).get("outputTokens") or 0) for m, u in usage.items() if isinstance(u, dict) or u is None}


def token_counts(data) -> dict:
    """Tokens of the session, summed over models from the result's modelUsage; from its usage when there is none
    (a session stopped at its spending cap reports zeros in usage and the real counts in modelUsage)."""
    names = {"input_tokens": "inputTokens", "output_tokens": "outputTokens",
             "cache_creation_input_tokens": "cacheCreationInputTokens", "cache_read_input_tokens": "cacheReadInputTokens"}
    per_model = data.get("modelUsage") if isinstance(data, dict) else None
    if isinstance(per_model, dict) and any(isinstance(u, dict) for u in per_model.values()):
        return {k: sum(int(u.get(v) or 0) for u in per_model.values() if isinstance(u, dict)) for k, v in names.items()}
    usage = (data.get("usage") if isinstance(data, dict) else None) or {}
    return {k: int(usage.get(k) or 0) for k in names}


def swapped(events: list[dict], data, model: str) -> str | None:
    """Why the answer is not (only) the run's model's, or None."""
    for e in events:
        if e["subtype"] in SWAP_EVENTS:
            return f"Claude Code answered with {e.get('fallback_model')} instead of {e.get('original_model') or model} ({e['subtype']})"
    usage = data.get("modelUsage") if isinstance(data, dict) else None
    others = sorted(m for m, u in (usage or {}).items() if isinstance(u, dict) and (u.get("outputTokens") or 0) > 0
                    and model not in (m, u.get("canonicalModel")))
    if others:
        return f"Claude Code's answer has output from {', '.join(others)}, not only {model}"
    return None


def refused(events: list[dict]) -> str | None:
    """The refusal category ('' when none came with it) if the model declined, else None."""
    found = [e for e in events if e["subtype"] in (REFUSAL_EVENT, "refusal")]
    if not found:
        return None
    return next((str(e["api_refusal_category"]) for e in found if e.get("api_refusal_category")), "")


def join(messages: list[dict]) -> str:
    """Assistant messages in order, as one text. The CLI splits an answer when its stream is cut off (no stop
    reason, or max_tokens) and continues at the exact character, so nothing goes between such pieces;
    messages that ended on their own are separated by a newline."""
    text = ""
    for i, m in enumerate(messages):
        if i and messages[i - 1].get("stop_reason") not in (None, "max_tokens") and not text.endswith("\n"):
            text += "\n"
        text += m["text"]
    return text


def init_event(out: str) -> dict:
    """The CLI's init event (what it loaded, the model, the key it bills, its version), or {}."""
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict) and ev.get("type") == "system" and ev.get("subtype") == "init":
            return ev
    return {}


def as_text(output) -> str:
    """What a stopped process had printed: subprocess hands it over as bytes, or None."""
    if output is None:
        return ""
    return output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)


def next_stream(where: Path) -> int:
    """The number of the next stream_NN.jsonl in a round's folder: one past the highest there."""
    numbers = [int(m.group(1)) for p in Path(where).glob("stream_*.jsonl") if (m := re.fullmatch(r"stream_(\d+)\.jsonl", p.name))]
    return max(numbers, default=0) + 1


def _run_cli(cmd, prompt_text, env, cwd, timeout_s):
    """subprocess.run, kept separate so tests can replace it."""
    return subprocess.run(cmd, input=prompt_text, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout_s)


def _http_get(url: str, headers: dict, timeout_s: float = 30) -> tuple[int, str]:
    """(status, the start of the body) of a GET; kept separate so tests can replace it."""
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.status, r.read(1 << 16).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(1 << 16).decode("utf-8", errors="replace")


class ClaudeCodeBackend:
    name = "claude_code"

    def __init__(self, cfg: dict, log=print, runner=None, sleep=time.sleep, environ=None, check_login: bool = True, http=None):
        self.cfg = cfg
        self.log = log
        self.run = runner or _run_cli
        self.sleep = sleep
        self.retries = cfg["api_retries"]
        self.command = cfg["claude_code_command"]
        self.auth = cfg.get("auth") or "login"
        if self.auth not in AUTH_MODES:
            raise BackendError(f"auth must be login or api_key, not {self.auth!r}")
        self.key_var = str(cfg.get("api_key_env") or "")
        self.workspace_var = str(cfg.get("api_workspace_env") or "")
        for var, what in ((self.key_var, "api_key_env"), (self.workspace_var, "api_workspace_env")):
            if var and (not VARIABLE.fullmatch(var) or var in KEY_VARIABLES):
                raise BackendError(f"{what} must name an environment variable other than {' and '.join(KEY_VARIABLES)}, not {var!r}")
        env = dict(os.environ if environ is None else environ)
        for var in KEY_VARIABLES:
            env.pop(var, None)  # never a key that happens to be in the environment
        self.workspace = None
        if self.auth == "api_key":
            if not self.key_var or not env.get(self.key_var):
                raise BackendError(f"auth is api_key, but {self.key_var or 'api_key_env'} is not set: put your API key in it first")
            errors = budget.check(cfg.get("budget"))
            if errors:
                raise BackendError("; ".join(errors))
            self.workspace = env.get(self.workspace_var) if self.workspace_var else None
            if self.workspace:
                if not WORKSPACE.fullmatch(self.workspace):
                    raise BackendError(f"{self.workspace_var} must hold a workspace ID (wrkspc_...)")
                env["ANTHROPIC_CUSTOM_HEADERS"] = f"anthropic-workspace-id: {self.workspace}"
        else:
            for var in (self.key_var, self.workspace_var):
                if var:
                    env.pop(var, None)  # a login run has no use for the key; keep it out of the CLI and the tools
        env.update(NO_MODEL_FALLBACK)
        self.env = env
        self.http = http or _http_get
        if runner is None:
            if shutil.which(self.command) is None:
                raise BackendError(f"the Claude Code CLI ({self.command!r}) is not installed or not on PATH")
            if self.auth == "api_key":
                self._check_flags()
                self._check_key()
            elif check_login:
                self._check_login()
        elif http is not None and self.auth == "api_key":
            self._check_key()

    def _check_login(self) -> None:
        try:
            r = subprocess.run([self.command, "auth", "status"], capture_output=True, text=True, env=self.env, timeout=60)
            status = json.loads(r.stdout) if r.stdout.strip().startswith("{") else {}
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            raise BackendError(f"could not check the Claude Code login: {exc}") from exc
        if not status.get("loggedIn"):
            raise BackendError("Claude Code is not logged in; run `claude` once and sign in with your subscription")

    def _check_flags(self) -> None:
        """The api_key mode needs --bare and --max-budget-usd, which older versions of the CLI do not have."""
        try:
            r = subprocess.run([self.command, "--help"], capture_output=True, text=True, env=self.env, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BackendError(f"could not run `{self.command} --help`: {exc}") from exc
        missing = [f for f in ("--bare", "--max-budget-usd", "--settings") if f not in (r.stdout or "") + (r.stderr or "")]
        if missing:
            raise BackendError(f"this Claude Code has no {', '.join(missing)}; update it (`claude update`) to bill an API key")

    def _check_key(self) -> None:
        """Ask the API whether it takes the key and lets it use the run's model (GET /v1/models/<model>, free of
        charge). The CLI retries a refused key from its helper for about four minutes; this finds it at once."""
        base = (self.env.get("ANTHROPIC_BASE_URL") or API_URL).rstrip("/")
        model = self.cfg["agent_model"]
        headers = {"x-api-key": self.env[self.key_var], "anthropic-version": "2023-06-01"}
        if self.workspace:
            headers["anthropic-workspace-id"] = self.workspace
        try:
            status, body = self.http(f"{base}/v1/models/{urllib.parse.quote(model, safe='')}", headers)
        except (OSError, ValueError) as exc:  # no network, no certificates: the first round will tell
            self.log(f"WARNING: could not check the API key before the run ({type(exc).__name__}: {str(exc)[:200]}); going on")
            return
        said = " ".join(str(body).split())[:300]
        if status in (401, 403):
            raise BackendError(f"the API refused the key in {self.key_var} (HTTP {status}): {said}")
        if status == 404:
            raise BackendError(f"the key in {self.key_var} cannot use the model {model} (HTTP 404): {said}")
        if status == 400 and "workspace" in said.lower():
            raise BackendError(f"the key in {self.key_var} is not scoped to one workspace: set {self.workspace_var or 'api_workspace_env'} "
                               f"to the workspace ID (HTTP 400): {said}")
        if status != 200:
            self.log(f"WARNING: checking the API key before the run gave HTTP {status}: {said}; going on")
            return
        try:
            lifecycle = json.loads(body).get("lifecycle")
        except (ValueError, AttributeError):
            lifecycle = None
        if lifecycle == "retired":
            raise BackendError(f"the model {model} is retired: the API no longer answers it")
        if lifecycle == "deprecated":
            self.log(f"WARNING: the model {model} is deprecated: the API still answers it, but not for long")

    def available(self, index: int) -> bool:
        return True

    def run_path(self, session) -> Path:
        return Path(session.state["run_path"])

    def exhausted(self, run_path: Path) -> str | None:
        """Why no round can start, when the run's budget is spent; None otherwise (and always under a login)."""
        if self.auth != "api_key":
            return None
        st = budget.state(run_path, self.cfg["budget"])
        return budget.spent_why(st) if budget.left(st) < budget.SMALLEST_CAP_USD else None

    def command_line(self, mcp_config: Path) -> list[str]:
        cmd = [self.command, "-p"]
        if self.auth == "api_key":
            # --bare reads no login at all (no OAuth, no keychain), so the key is the only way it can pay; the
            # helper prints the key from its variable, so the value is on no command line and in no file.
            cmd += ["--bare", "--settings", json.dumps({"apiKeyHelper": f"printenv {self.key_var}"})]
        # Under a login, not --bare: that mode reads no login, so a subscription would look logged out. The
        # flags below give the same isolation either way.
        cmd += ["--tools", "", "--output-format", "stream-json", "--verbose", "--no-session-persistence"]
        cmd += ["--setting-sources", "", "--disable-slash-commands", "--strict-mcp-config"]
        cmd += ["--mcp-config", str(mcp_config), "--allowedTools", ",".join(f"mcp__{SERVER_NAME}__{t}" for t in TOOLS)]
        cmd += ["--max-turns", str(int(self.cfg["session"]["max_turns"]))]
        cmd += ["--system-prompt", SYSTEM_PROMPT, "--model", self.cfg["agent_model"]]
        if self.cfg.get("effort") is not None:
            cmd += ["--effort", self.cfg["effort"]]
        return cmd

    def mcp_config(self, session, where: Path) -> Path:
        # The CLI hands its own environment to the tool server; the server needs no key, so it drops them first.
        forget = sorted({*KEY_VARIABLES, "ANTHROPIC_CUSTOM_HEADERS", *(v for v in (self.key_var, self.workspace_var) if v)})
        code = (f"import os, sys; [os.environ.pop(v, None) for v in {forget!r}]; sys.path.insert(0, {str(REPO_ROOT)!r}); "
                f"from level1.mcp_server import main; sys.exit(main(['--session', {str(session.path)!r}]))")
        env = {"PYTHONDONTWRITEBYTECODE": "1", "PATH": os.environ.get("PATH", os.defpath)}
        cfg = {"mcpServers": {SERVER_NAME: {"command": self.cfg.get("python_executable") or sys.executable,
                                            "args": ["-c", code], "env": env}}}
        path = Path(where) / "mcp.json"
        atomic_write(path, json.dumps(cfg, indent=1))
        return path

    def _wait(self, attempt: int) -> float:
        wait = min(self.retries["max_wait_s"], self.retries["first_wait_s"] * 2**attempt)
        return wait * (1 + 0.1 * random.random())

    def _classify(self, text: str) -> tuple[bool, str]:
        """(retry?, description) for an error the CLI reported."""
        low = text.lower()
        if any(w in low for w in BILLING_WORDS):
            return False, f"the API account cannot pay for more: {text[:300]}"
        if any(w in low for w in MODEL_WORDS):
            return False, f"the model {self.cfg['agent_model']} cannot be used here: {text[:300]}"
        if any(w in low for w in LOGIN_WORDS) and not any(w in low for w in LIMIT_WORDS):
            what = f"the API key in {self.key_var} was refused" if self.auth == "api_key" else "Claude Code login problem"
            return False, f"{what}: {text[:300]}"
        if any(w in low for w in LIMIT_WORDS):
            return True, f"usage or rate limit: {text[:300]}"
        return True, f"Claude Code error: {text[:300]}"

    def respond(self, prompt_text: str, index: int, where, session) -> Response:
        where = Path(where)
        where.mkdir(parents=True, exist_ok=True)
        cmd = self.command_line(self.mcp_config(session, where))
        k, run_path = index + 1, self.run_path(session)
        t0 = time.monotonic()
        failures = []
        tries = int(self.retries["tries"])
        timeout = float(self.cfg["session"]["seconds"])
        data, messages, text, events, declined = {}, [], "", [], None
        restarts, caps, costs, sources, version, ended = [], [], [], [], None, None
        for attempt in range(tries):
            session.reload()  # the tools run in the tool server's process; an earlier try may have used some
            note = prompt.restart_note(session)
            if note:
                restarts.append(dict(session.state["used"]))
            n = next_stream(where)
            this_try = list(cmd)
            if self.auth == "api_key":
                st = budget.state(run_path, self.cfg["budget"])
                cap = budget.cap(st, k)
                if cap <= 0:  # what the session submitted stands; the loop ends the rounds if the run's is spent
                    ended = budget.spent_why(st, k)
                    self.log(f"  {ended}; the round ends with what it submitted")
                    data, messages, text = {}, [], f"({ended})"
                    break
                budget.start(run_path, k, n, cap)
                caps.append(cap)
                this_try += ["--max-budget-usd", f"{cap:.2f}"]
            r, timed_out = None, False
            with tempfile.TemporaryDirectory(prefix="lv1-claude-code-") as cwd:
                try:
                    r = self.run(this_try, prompt_text + note, self.env, cwd, timeout)
                    stdout = r.stdout or ""
                except subprocess.TimeoutExpired as exc:
                    # the session's wall clock ran out: what it submitted stands; there is no closing text
                    timed_out, stdout = True, as_text(exc.stdout)
                except OSError as exc:
                    if self.auth == "api_key":
                        budget.end(run_path, k, n, 0.0, "not started")
                    raise StopRun(f"could not run the Claude Code CLI: {exc}") from exc
            # Everything the CLI emitted on this try, as it came: the messages, the tool calls and as much of the
            # model's reasoning as the CLI shows (a try stopped at the time limit: what it emitted until then).
            # response.md keeps only the text; this keeps all.
            atomic_write(where / f"stream_{n:02d}.jsonl", stdout)
            out = stdout.strip()
            data, messages = parse_output(out)
            init = init_event(out)
            cost = data.get("total_cost_usd") if isinstance(data, dict) else None
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                costs.append(float(cost))
            if self.auth == "api_key":
                budget.end(run_path, k, n, cost, "timed out" if timed_out else (data or {}).get("subtype"))
            if init:
                version = init.get("claude_code_version") or version
                sources.append(init.get("apiKeySource"))
                if init.get("apiKeySource") not in KEY_SOURCES[self.auth]:
                    raise StopRun(f"Claude Code billed {init.get('apiKeySource')!r} for this session, where auth {self.auth} "
                                  f"allows {' or '.join(map(repr, KEY_SOURCES[self.auth]))}; the run stops")
            events = model_events(out)
            swap = swapped(events, data, self.cfg["agent_model"])
            if swap:
                raise StopRun(f"{swap}; the run stops rather than record another model's work as {self.cfg['agent_model']}'s")
            if timed_out:
                failures.append(f"the session did not finish within {timeout:g} s")
                data, messages, text = {"timed_out": True}, [], "(the session ran out of time)"
                break
            text = join(messages) if messages else (data or {}).get("result")
            declined = refused(events)
            if declined is not None:  # what the session submitted before it stands; the loop counts refusals
                data, text = data if isinstance(data, dict) else {}, join(messages) if messages else ""
                self.log(f"  Claude Code: the model declined to go on ({declined or 'no category given'})")
                break
            if r.returncode == 0 and isinstance(data, dict) and not data.get("is_error") and text is not None:
                break
            message = ""
            if isinstance(data, dict):
                message = str(data.get("result") or data.get("error") or data.get("subtype") or "")
            message = message or (r.stderr or "")[-STDERR_TAIL:] or out[-STDERR_TAIL:] or f"exit code {r.returncode}"
            if isinstance(data, dict) and data.get("subtype") == "error_max_turns":
                text = join(messages) if messages else "(the session reached its turn limit)"
                break
            if isinstance(data, dict) and data.get("subtype") == "error_max_budget_usd":
                ended = f"the session reached its spending cap (${caps[-1]:.2f})" if caps else "the session reached its spending cap"
                self.log(f"  {ended}; the round ends with what it submitted")
                text = join(messages) if messages else f"({ended})"
                break
            retry, what = self._classify(message)
            failures.append(what)
            if not retry:
                raise StopRun(f"Claude Code stopped the run: {what}")
            if attempt + 1 >= tries:
                raise StopRun(f"Claude Code kept failing after {tries} tries; last: {what}. Resume later with `python -m level1.loop resume`.")
            wait = self._wait(attempt)
            self.log(f"  Claude Code: {what}; trying again in {wait:.0f} s")
            self.sleep(wait)

        data = data if isinstance(data, dict) else {}
        atomic_write(where / "response_raw.json", json.dumps({**data, "assistant_messages": messages}))
        last_stop = messages[-1].get("stop_reason") if messages else None
        session.reload()  # the tools ran in the tool server's process; the counters are in the saved session
        billed = self.auth == "api_key" or any(src in BILLED_SOURCES for src in sources)
        meta = {
            "backend": self.name,
            "source": data.get("session_id"),
            "model": self.cfg["agent_model"],
            "stop_reason": "refusal" if declined is not None else "budget" if ended else
                           (last_stop or data.get("stop_reason") or ("end_turn" if not data.get("is_error") else "error")),
            "stop_details": {"category": declined} if declined is not None else {"budget": ended} if ended else None,
            "models_used": models_used(data),
            "safety_stops": data.get("safety_stops"),
            "model_events": events,
            **token_counts(data),
            "auth": self.auth,
            "billed_to": sources[-1] if sources else None,  # the CLI's apiKeySource: none means a subscription
            "cost_usd": round(sum(costs), 6) if billed else 0.0,  # what the tries cost by the CLI's count; a subscription is not billed per call
            "api_equivalent_usd": data.get("total_cost_usd"),
            "claude_code_version": version,
            "num_turns": data.get("num_turns"),
            "assistant_messages": len(messages),
            "tools_used": dict(session.state["used"]),
            "submissions": len(session.receipts()),
            "request": {"command": cmd[:1] + cmd[1:10]},
            "failed_tries": failures,
            "restarted_with_used": restarts,  # one item per try that began with a used session: what was used
            "seconds": round(time.monotonic() - t0, 1),
        }
        if self.auth == "api_key":
            st = budget.state(run_path, self.cfg["budget"])
            meta["budget"] = {"caps_usd": caps, "round_counted_usd": round(st["rounds"].get(k, 0.0), 6),
                              "run_counted_usd": round(st["spent"], 6), "round_usd": st["round_usd"], "run_usd": st["run_usd"],
                              "unreported_tries": st["unreported"]}
            self.log(f"  API: round {k} ${st['rounds'].get(k, 0.0):.2f}; the run ${st['spent']:.2f} of ${st['run_usd']:.2f}"
                     + (f" ({st['unreported']} tries counted at their caps: the CLI reported no cost)" if st["unreported"] else ""))
        return Response(text=str(text or ""), meta=meta)
