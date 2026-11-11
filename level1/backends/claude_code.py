"""The Claude Code backend: one headless run of the official Claude Code CLI per round.

This lets a run draw on a Claude subscription (Pro or Max) through the user's own Claude Code login instead
of billing an API key. Each round runs

    claude -p --tools "" --output-format stream-json --verbose --no-session-persistence
           --setting-sources "" --disable-slash-commands --strict-mcp-config
           --mcp-config <round>/mcp.json --allowedTools mcp__lv1__run_python,mcp__lv1__submit_entry
           --max-turns <n> --system-prompt <short> --model <agent_model> --effort <effort>

with the prompt on stdin and an empty temporary folder as the working directory: no built-in tools, no
files, no project settings, and one MCP server (mcp_server.py) that serves the session's two tools. The
tools run in that server's process; their budgets and log are the session's.

The CLI's environment has ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN removed, so it cannot fall back to
billing an API key, and has CLAUDE_CODE_NO_MODEL_FALLBACK set, so it cannot answer with another model when
the requested one declines. An answer that still shows another model stops the run; a refusal is recorded
as one, and the loop counts refusals. Usage-limit and overload errors are retried with growing waits; a
login problem stops the run, which `resume` continues later. The session's wall clock is session.seconds:
when it runs out, what the session submitted stands. Each try's whole event stream is kept in the round's
folder (stream_NN.jsonl). A try that finds the session already used (a retry
after a dropped connection, or a resumed run) gets the same prompt with a note at its end: the budgets as
they stand, the entries saved and the submissions made before the cut (prompt.restart_note).

Whether a subscription may be used this way is governed by Anthropic's terms for Claude Code, which have
changed over time: check the current policy for your account before relying on it.

Taken from the proof pipeline (core/agent/backends/claude_code.py and core/open_lane/backends.py of
Compact-Proof-Agent), merged into one class.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .. import prompt
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
# Unless this is set, the CLI answers with another model when the requested one declines or is unavailable,
# and says so only in its events and the result's modelUsage: the records would credit the run's model with
# another model's work. A run checks every answer for a swap anyway.
NO_MODEL_FALLBACK = {"CLAUDE_CODE_NO_MODEL_FALLBACK": "1"}
SWAP_EVENTS = ("model_fallback", "model_refusal_fallback", "model_consent_fallback")
REFUSAL_EVENT = "model_refusal_no_fallback"
LIMIT_WORDS = ("usage limit", "rate limit", "rate_limit", "overloaded", "too many requests", "capacity", "529", "429")
LOGIN_WORDS = ("not logged in", "login", "authenticat", "unauthorized", "invalid api key", "401", "403")
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


def _run_cli(cmd, prompt_text, env, cwd, timeout_s):
    """subprocess.run, kept separate so tests can replace it."""
    return subprocess.run(cmd, input=prompt_text, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout_s)


class ClaudeCodeBackend:
    name = "claude_code"

    def __init__(self, cfg: dict, log=print, runner=None, sleep=time.sleep, environ=None, check_login: bool = True):
        self.cfg = cfg
        self.log = log
        self.run = runner or _run_cli
        self.sleep = sleep
        self.retries = cfg["api_retries"]
        self.command = cfg["claude_code_command"]
        env = dict(os.environ if environ is None else environ)
        for var in KEY_VARIABLES:
            env.pop(var, None)  # the CLI's own login, never a key from the environment
        env.update(NO_MODEL_FALLBACK)
        self.env = env
        if runner is None:
            if shutil.which(self.command) is None:
                raise BackendError(f"the Claude Code CLI ({self.command!r}) is not installed or not on PATH")
            if check_login:
                self._check_login()

    def _check_login(self) -> None:
        try:
            r = subprocess.run([self.command, "auth", "status"], capture_output=True, text=True, env=self.env, timeout=60)
            status = json.loads(r.stdout) if r.stdout.strip().startswith("{") else {}
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            raise BackendError(f"could not check the Claude Code login: {exc}") from exc
        if not status.get("loggedIn"):
            raise BackendError("Claude Code is not logged in; run `claude` once and sign in with your subscription")

    def available(self, index: int) -> bool:
        return True

    def command_line(self, mcp_config: Path) -> list[str]:
        # Not --bare: that mode skips keychain reads and accepts only an API key, so a subscription login
        # would look logged out. These flags give the same isolation without touching auth.
        cmd = [self.command, "-p", "--tools", "", "--output-format", "stream-json", "--verbose", "--no-session-persistence"]
        cmd += ["--setting-sources", "", "--disable-slash-commands", "--strict-mcp-config"]
        cmd += ["--mcp-config", str(mcp_config), "--allowedTools", ",".join(f"mcp__{SERVER_NAME}__{t}" for t in TOOLS)]
        cmd += ["--max-turns", str(int(self.cfg["session"]["max_turns"]))]
        cmd += ["--system-prompt", SYSTEM_PROMPT, "--model", self.cfg["agent_model"]]
        if self.cfg.get("effort") is not None:
            cmd += ["--effort", self.cfg["effort"]]
        return cmd

    def mcp_config(self, session, where: Path) -> Path:
        code = (f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); from level1.mcp_server import main; "
                f"sys.exit(main(['--session', {str(session.path)!r}]))")
        env = {"PYTHONDONTWRITEBYTECODE": "1", "PATH": os.environ.get("PATH", os.defpath)}
        cfg = {"mcpServers": {SERVER_NAME: {"command": self.cfg.get("python_executable") or sys.executable,
                                            "args": ["-c", code], "env": env}}}
        path = Path(where) / "mcp.json"
        atomic_write(path, json.dumps(cfg, indent=1))
        return path

    def _wait(self, attempt: int) -> float:
        wait = min(self.retries["max_wait_s"], self.retries["first_wait_s"] * 2**attempt)
        return wait * (1 + 0.1 * random.random())

    @staticmethod
    def _classify(text: str) -> tuple[bool, str]:
        """(retry?, description) for an error the CLI reported."""
        low = text.lower()
        if any(w in low for w in LOGIN_WORDS) and not any(w in low for w in LIMIT_WORDS):
            return False, f"Claude Code login problem: {text[:300]}"
        if any(w in low for w in LIMIT_WORDS):
            return True, f"usage or rate limit: {text[:300]}"
        return True, f"Claude Code error: {text[:300]}"

    def respond(self, prompt_text: str, index: int, where, session) -> Response:
        where = Path(where)
        where.mkdir(parents=True, exist_ok=True)
        cmd = self.command_line(self.mcp_config(session, where))
        t0 = time.monotonic()
        failures = []
        tries = int(self.retries["tries"])
        timeout = float(self.cfg["session"]["seconds"])
        data, messages, text, events, declined = {}, [], "", [], None
        restarts = []
        for attempt in range(tries):
            session.reload()  # the tools run in the tool server's process; an earlier try may have used some
            note = prompt.restart_note(session)
            if note:
                restarts.append(dict(session.state["used"]))
            with tempfile.TemporaryDirectory(prefix="lv1-claude-code-") as cwd:
                try:
                    r = self.run(cmd, prompt_text + note, self.env, cwd, timeout)
                except subprocess.TimeoutExpired:
                    # the session's wall clock ran out: what it submitted stands; there is no closing text
                    r = None
                    retry, what = False, f"the session did not finish within {timeout:g} s"
                except OSError as exc:
                    raise StopRun(f"could not run the Claude Code CLI: {exc}") from exc
            if r is not None:
                out = (r.stdout or "").strip()
                # Everything the CLI emitted on this try, as it came: the messages, the tool calls and as much
                # of the model's reasoning as the CLI shows. response.md keeps only the text; this keeps all.
                atomic_write(where / f"stream_{attempt + 1:02d}.jsonl", (r.stdout or ""))
                data, messages = parse_output(out)
                events = model_events(out)
                swap = swapped(events, data, self.cfg["agent_model"])
                if swap:
                    raise StopRun(f"{swap}; the run stops rather than record another model's work as {self.cfg['agent_model']}'s")
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
                retry, what = self._classify(message)
            failures.append(what)
            if not retry:
                if r is None:
                    data, messages, text = {"timed_out": True}, [], "(the session ran out of time)"
                    break
                raise StopRun(f"Claude Code stopped the run: {what}")
            if attempt + 1 >= tries:
                raise StopRun(f"Claude Code kept failing after {tries} tries; last: {what}. Resume later with `python -m level1.loop resume`.")
            wait = self._wait(attempt)
            self.log(f"  Claude Code: {what}; trying again in {wait:.0f} s")
            self.sleep(wait)

        data = data if isinstance(data, dict) else {}
        atomic_write(where / "response_raw.json", json.dumps({**data, "assistant_messages": messages}))
        usage = data.get("usage") or {}
        last_stop = messages[-1].get("stop_reason") if messages else None
        session.reload()  # the tools ran in the tool server's process; the counters are in the saved session
        meta = {
            "backend": self.name,
            "source": data.get("session_id"),
            "model": self.cfg["agent_model"],
            "stop_reason": "refusal" if declined is not None else
                           (last_stop or data.get("stop_reason") or ("end_turn" if not data.get("is_error") else "error")),
            "stop_details": {"category": declined} if declined is not None else None,
            "models_used": models_used(data),
            "safety_stops": data.get("safety_stops"),
            "model_events": events,
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
            "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
            "cost_usd": 0.0,  # a subscription is not billed per call
            "api_equivalent_usd": data.get("total_cost_usd"),
            "num_turns": data.get("num_turns"),
            "assistant_messages": len(messages),
            "tools_used": dict(session.state["used"]),
            "submissions": len(session.receipts()),
            "request": {"command": cmd[:1] + cmd[1:10]},
            "failed_tries": failures,
            "restarted_with_used": restarts,  # one item per try that began with a used session: what was used
            "seconds": round(time.monotonic() - t0, 1),
        }
        return Response(text=str(text or ""), meta=meta)
