"""The Codex backend: one headless run of OpenAI's Codex CLI per round, under the user's ChatGPT login.

Each round runs

    codex exec --json --ephemeral --skip-git-repo-check --ignore-user-config --ignore-rules -s read-only
               -m <agent_model> -c model_reasoning_effort=<effort> <the switches of ISOLATION> <the tool server> -

with the prompt on stdin and an empty temporary folder as the working directory. Codex is a coding agent: left
alone it gives the model a shell, file edits, the web, images, sub-agents, skills, the user's connected apps and
its own long instructions. A run here needs the opposite: the run's system prompt, the round's prompt, the
session's tools, and nothing that reads a file or the web (the repository holds what the networks were made
for). So every such feature is switched off, the instructions are replaced by the run's, the notes on the
environment, permissions, apps and skills are left out, the user's config and rules are not loaded, and the one
tool server (mcp_server.py) is given to the model directly.

What is left of Codex's own depends on the model, and is decided by the CLI, not here. With the CLI this was
written against (0.172.0) a model of the gpt-5.6 or gpt-6 families still gets Codex's script tool (JavaScript
that can call other tools: no file system, no network), a tool to wait on it and one to message the user; an
earlier model gets a file-patch tool, which the read-only sandbox refuses, and tools that list a tool server's
resources, of which there are none. None of them reads a file or reaches the web. Because a later version of
the CLI or another model may add something, the backend does not take this on trust:

- Before the first round the CLI is run once against a stand-in for the API on this machine (standin.py: no
  model, no network), with the run's own switches, and what it sends is read: every tool it offers the model,
  directly or through its script tool, and every text besides the run's. Anything outside ALLOWED_TOOLS,
  ALLOWED_NESTED and ALLOWED_NOTES stops the run before any model is asked. What was seen is kept in the round's
  folder (codex_check.json).
- After every session its events are read: a shell command, a file change, a web search, a sub-agent or a call
  to another tool server stops the run.

The check sees what the CLI sends a custom address; under the login the CLI talks to OpenAI's own, where OpenAI
could in principle enable more. The second check is what covers that.

Who pays: the ChatGPT login only (`auth: login`). The CLI has no spending cap, so a run billed to an API key goes
through the openai_api backend, which has one. Here OPENAI_API_KEY, CODEX_API_KEY and the run's own key variable
are removed from the CLI's environment, the CLI is told to use the ChatGPT login only, and a login that is an API
key is refused before the run.

The personal instructions file of the Codex home (AGENTS.md) is given to every model the CLI runs, and no switch
leaves it out. A run refuses a home that has one: `codex_home` in the run config names another, which
`CODEX_HOME=<that folder> codex login` signs in once.

Usage limits are waited out with growing pauses; a login problem or a model the plan cannot use stops the run,
which `resume` continues later. The session's wall clock is session.seconds; session.max_turns bounds the tool
calls of one try (the CLI has no turn limit of its own). Each try's whole event stream is kept in the round's
folder (stream_NN.jsonl), and a try that finds the session already used is told so (prompt.restart_note).
"""

from __future__ import annotations

import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from .. import prompt, run_config
from ..mcp_server import SERVER_NAME
from ..util import REPO_ROOT, atomic_write
from . import BackendError, Response, StopRun
from .claude_code import STDERR_TAIL, SUMMARY_SYSTEM_PROMPT, SYSTEM_PROMPT, next_stream
from .standin import StandIn, message, script_call

KEY_VARIABLES = run_config.KEY_VARIABLES + run_config.OPENAI_KEY_VARIABLES
NAMESPACE = f"mcp__{SERVER_NAME}"
# Features of the CLI that give the model something of Codex's own: each is switched off if this CLI has it.
FEATURES_OFF = ("shell_tool", "unified_exec", "view_image", "multi_agent", "goals", "apps", "plugins", "image_generation",
                "browser_use", "computer_use", "tool_suggest", "sleep_tool", "hooks", "skill_search", "memories")
ISOLATION = (
    'web_search="disabled"', 'forced_login_method="chatgpt"', 'approval_policy="never"',
    "include_environment_context=false", "include_permissions_instructions=false", "include_apps_instructions=false",
    "include_collaboration_mode_instructions=false", "skills.include_instructions=false", "skills.bundled.enabled=false",
    "project_doc_max_bytes=0", "tools.experimental_request_user_input.enabled=false", 'history.persistence="none"',
    "check_for_update_on_startup=false", "analytics.enabled=false",
    # the session's tools as direct tools, not behind the script tool or a search; no sub-agent tools
    f'features.code_mode={{direct_only_tool_namespaces=["{NAMESPACE}"]}}', "features.multi_agent_v2={disable_native_tools=true}",
)
# What may be left of the CLI's own, by name: none reads a file or reaches the web (see the module's text).
ALLOWED_TOOLS = ("exec", "wait", "send_message_to_user_async", "apply_patch", "list_mcp_resources",
                 "list_mcp_resource_templates", "read_mcp_resource")
ALLOWED_NESTED = ("apply_patch", "clock__curr_time", "list_mcp_resources", "list_mcp_resource_templates", "read_mcp_resource")
ALLOWED_NOTES = ("<multi_agent_mode>",)  # developer texts of the CLI's own, by how they begin
# Items in a session's events that mean a tool outside the session's was used.
FORBIDDEN_ITEMS = ("command_execution", "file_change", "web_search", "collab_tool_call")
LIMIT_WORDS = ("usage limit", "rate limit", "rate_limit", "too many requests", "overloaded", "capacity", "429", "try again")
LOGIN_WORDS = ("not logged in", "log in", "login", "unauthorized", "401", "403", "authentication", "sign in")
MODEL_WORDS = ("model is not supported", "not supported when using codex", "model_not_found", "does not exist", "do not have access")
STANDIN_KEY = "LV1_STANDIN_KEY"
_CHECKED: dict = {}  # isolation checks that passed in this process, by what was checked


def toml(value) -> str:
    """A string, a list of strings or a mapping of strings as TOML (a JSON string is a TOML basic string)."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}={toml(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml(v) for v in value) + "]"
    return json.dumps(str(value))


def parse_events(out: str) -> dict:
    """What a try's event stream says: the agent's messages, the usage, failures, the tool calls and items by type."""
    got = {"messages": [], "usage": None, "errors": [], "failed": None, "tool_calls": 0, "items": {}, "foreign": [],
           "thread": None, "completed": False}
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
        kind = ev.get("type")
        if kind == "thread.started":
            got["thread"] = ev.get("thread_id")
        elif kind == "turn.completed":
            got["completed"], got["usage"] = True, ev.get("usage") if isinstance(ev.get("usage"), dict) else {}
        elif kind == "turn.failed":
            got["failed"] = str((ev.get("error") or {}).get("message") or "the turn failed")
        elif kind == "error":
            got["errors"].append(str(ev.get("message") or ""))
        elif kind == "item.completed" and isinstance(ev.get("item"), dict):
            item = ev["item"]
            t = str(item.get("type"))
            got["items"][t] = got["items"].get(t, 0) + 1
            if t == "agent_message" and isinstance(item.get("text"), str):
                got["messages"].append(item["text"])
            elif t == "mcp_tool_call":
                got["tool_calls"] += 1
                if item.get("server") != SERVER_NAME:
                    got["foreign"].append(f"a call to the tool server {item.get('server')!r}")
            elif t in FORBIDDEN_ITEMS:
                got["foreign"].append(t.replace("_", " "))
    return got


def tool_names(body: dict) -> list[str]:
    """Every tool a request offers the model, namespaced ones as namespace.name: the tools field and, for the
    models the CLI sends them to that way, the additional_tools items of the input."""
    def walk(tools, prefix=""):
        out = []
        for t in tools or []:
            if not isinstance(t, dict):
                continue
            if t.get("type") == "namespace":
                out += walk(t.get("tools"), f"{t.get('name')}.")
            else:
                out.append(prefix + str(t.get("name") or t.get("type")))
        return out

    names = walk(body.get("tools"))
    for item in body.get("input") or []:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            names += walk(item.get("tools"))
    return names


def texts(body: dict) -> list[tuple[str, str]]:
    """(role, text) of every message of a request's input, and of its instructions."""
    out = [("instructions", body["instructions"])] if isinstance(body.get("instructions"), str) and body["instructions"] else []
    for item in body.get("input") or []:
        if not isinstance(item, dict) or item.get("type") not in (None, "message"):
            continue
        content = item.get("content")
        text = content if isinstance(content, str) else "".join(str(p.get("text") or "") for p in content or [] if isinstance(p, dict))
        out.append((str(item.get("role")), text))
    return out


def _run_cli(cmd, prompt_text, env, cwd, timeout_s, max_tool_calls=None):
    """Run the CLI with the prompt on stdin; stop it at the time limit, or once it has made more tool calls than
    the session has turns. Returns its output so far either way (stopped: None, "timeout" or "turn_limit").
    Kept separate so tests can replace it."""
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, env=env, cwd=cwd, text=True,
                                start_new_session=True)
        lines, calls = [], [0]

        def read():
            for line in proc.stdout:
                lines.append(line)
                if '"mcp_tool_call"' in line and '"item.completed"' in line:
                    calls[0] += 1

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        try:
            proc.stdin.write(prompt_text)
            proc.stdin.close()
        except OSError:
            pass
        deadline, stopped = time.monotonic() + timeout_s, None
        try:
            while proc.poll() is None:
                if time.monotonic() >= deadline:
                    stopped = "timeout"
                elif max_tool_calls is not None and calls[0] > max_tool_calls:
                    stopped = "turn_limit"
                if stopped:
                    break
                time.sleep(0.2)
        finally:
            if proc.poll() is None:  # the CLI and the tool server it started
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(proc.pid, sig)
                    except OSError:
                        break
                    try:
                        proc.wait(timeout=10)
                        break
                    except subprocess.TimeoutExpired:
                        continue
        reader.join(timeout=10)
        err.seek(0)
        return SimpleNamespace(stdout="".join(lines), stderr=err.read().decode("utf-8", errors="replace"),
                               returncode=proc.returncode, stopped=stopped)


class CodexBackend:
    name = "codex"

    def __init__(self, cfg: dict, log=print, runner=None, sleep=time.sleep, environ=None, check_login: bool = True,
                 check_isolation: bool | None = None, provider: list[str] | None = None):
        """provider: switches that point the CLI at another address than OpenAI's, for tests against a stand-in."""
        self.cfg = cfg
        self.log = log
        self.run = runner or _run_cli
        self.sleep = sleep
        self.retries = cfg["api_retries"]
        self.command = cfg["codex_command"]
        self.auth = cfg.get("auth") or "login"
        if self.auth != "login":
            raise BackendError("the codex backend runs under a ChatGPT login only (auth: login); a key is billed through the "
                               "openai_api backend, which has a spending cap")
        env = dict(os.environ if environ is None else environ)
        for var in (*KEY_VARIABLES, str(cfg.get("api_key_env") or ""), str(cfg.get("api_workspace_env") or "")):
            env.pop(var, None)  # never a key that happens to be in the environment
        if cfg.get("codex_home"):
            env["CODEX_HOME"] = str(Path(str(cfg["codex_home"])).expanduser())
        self.env = env
        self.home = Path(env.get("CODEX_HOME") or Path(env.get("HOME") or Path.home()) / ".codex")
        self.features = FEATURES_OFF
        self.version = None
        self.provider = list(provider or [])
        self.check_isolation = runner is None if check_isolation is None else check_isolation
        if runner is None:
            if shutil.which(self.command, path=env.get("PATH")) is None:
                raise BackendError(f"the Codex CLI ({self.command!r}) is not installed or not on PATH")
            self._check_cli()
            self._check_home()
            if check_login:
                self._check_login()

    # before a run ------------------------------------------------------------------------------
    def _output(self, *args) -> str:
        try:
            r = subprocess.run([self.command, *args], capture_output=True, text=True, env=self.env, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BackendError(f"could not run `{self.command} {' '.join(args)}`: {exc}") from exc
        return (r.stdout or "") + (r.stderr or "")

    def _check_cli(self) -> None:
        """The flags a run needs, the CLI's version, and which of FEATURES_OFF this CLI knows."""
        missing = [f for f in ("--json", "--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check")
                   if f not in self._output("exec", "--help")]
        if missing:
            raise BackendError(f"this Codex CLI has no {', '.join(missing)}; update it (`codex update`)")
        self.version = next((line.strip() for line in self._output("--version").splitlines() if line.strip().startswith("codex")), None)
        known = {line.split()[0] for line in self._output("features", "list").splitlines() if line.split()}
        self.features = tuple(f for f in FEATURES_OFF if f in known) if known else FEATURES_OFF

    def _check_home(self) -> None:
        """No switch keeps the personal instructions of the Codex home from the model, so a home that has them is refused."""
        found = [name for name in ("AGENTS.md", "AGENTS.override.md") if (self.home / name).exists()]
        if found:
            raise BackendError(
                f"{self.home} has {' and '.join(found)}: the Codex CLI gives that file to every model it runs, and nothing "
                "switches it off. Give the runs a Codex home of their own: set codex_home in the run config to an empty folder "
                "and sign in there once with `CODEX_HOME=<that folder> codex login`")

    def _check_login(self) -> None:
        status = " ".join(self._output("login", "status").split())
        if "Logged in using ChatGPT" in status:
            return
        if "Logged in" in status:
            raise BackendError(f"the Codex CLI is signed in another way ({status[:120]}); the codex backend takes a ChatGPT login "
                               "only. To bill an API key within a budget, use backend: openai_api")
        raise BackendError(f"the Codex CLI is not logged in ({status[:120] or 'no status'}); run `codex login` once and sign in "
                           "with ChatGPT" + (f" (with CODEX_HOME={self.home})" if self.cfg.get("codex_home") else ""))

    # the command -------------------------------------------------------------------------------
    def tool_server(self, session) -> list[str]:
        """The -c switches that attach the session's tool server, and nothing when the session has no tools."""
        if not session.TOOLS:
            return []
        code = (f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
                f"from level1.mcp_server import main; sys.exit(main(['--session', {str(session.path)!r}]))")
        seconds = float(self.cfg["session"]["seconds"])
        server = {
            "command": toml(self.cfg.get("python_executable") or sys.executable),
            "args": toml(["-c", code]),
            "env": toml({"PYTHONDONTWRITEBYTECODE": "1", "PATH": self.env.get("PATH", os.defpath)}),
            "enabled_tools": toml(list(session.TOOLS)),
            "default_tools_approval_mode": toml("approve"),  # there is nobody to ask; the session's own budgets hold
            "required": "true",
            "startup_timeout_sec": "120",
            "tool_timeout_sec": f"{seconds:g}",  # a tool call ends by the session's rules, not by the CLI's patience
        }
        return [arg for key, value in server.items() for arg in ("-c", f"mcp_servers.{SERVER_NAME}.{key}={value}")]

    def command_line(self, session, instructions: Path, provider: list[str] | None = None) -> list[str]:
        cmd = [self.command, "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules",
               "--color", "never", "-s", "read-only", "-m", self.cfg["agent_model"]]
        for feature in self.features:
            cmd += ["--disable", feature]
        for setting in ISOLATION:
            cmd += ["-c", setting]
        cmd += ["-c", f"model_instructions_file={toml(instructions)}"]
        if self.cfg.get("effort") is not None:
            cmd += ["-c", f"model_reasoning_effort={toml(self.cfg['effort'])}"]
        return cmd + self.tool_server(session) + (self.provider if provider is None else provider) + ["-"]

    # the isolation check -----------------------------------------------------------------------
    def isolation(self, session, where: Path, instructions: Path, system: str) -> dict:
        """Run the CLI once against a stand-in for the API and read what it sends: the tools it offers the model,
        directly and through its script tool, and every text besides the run's. Raises BackendError when there is
        anything a run must not have; returns what was seen."""
        marker = "LV1-ISOLATION-CHECK"
        wanted = [f"{NAMESPACE}.{name}" for name in session.TOOLS]

        def answer(body, k):
            if k == 0 and any(n.rsplit(".", 1)[-1] == "exec" for n in tool_names(body)):
                return [script_call("text(JSON.stringify(ALL_TOOLS.map(t => t.name)))")]
            return [message("ok")]

        with StandIn(answer) as server, tempfile.TemporaryDirectory(prefix="lv1-codex-check-") as cwd:
            provider = ["-c", 'model_provider="lv1standin"', "-c", 'model_providers.lv1standin.name="lv1standin"',
                        "-c", f"model_providers.lv1standin.base_url={toml(server.url)}",
                        "-c", f'model_providers.lv1standin.env_key="{STANDIN_KEY}"', "-c", 'model_providers.lv1standin.wire_api="responses"']
            r = _run_cli(self.command_line(session, instructions, provider), marker + "\n", {**self.env, STANDIN_KEY: "stand-in"}, cwd, 120)
            requests = list(server.requests)
        events = parse_events(r.stdout)
        if not requests or not events["completed"]:
            raise BackendError("the isolation check could not run the Codex CLI against the stand-in: "
                               + (events["failed"] or "; ".join(events["errors"]) or r.stderr[-STDERR_TAIL:] or f"exit code {r.returncode}")[-600:])
        direct = sorted({n for body in requests for n in tool_names(body)})
        nested = []
        for body in requests[1:]:
            for item in body.get("input") or []:
                if isinstance(item, dict) and item.get("type") == "custom_tool_call_output":
                    out = item.get("output")
                    parts = [out] if isinstance(out, str) else [str(p.get("text") or "") for p in out or [] if isinstance(p, dict)]
                    for part in parts:
                        start = part.find("[")
                        if start >= 0:
                            try:
                                nested = sorted({str(n) for n in json.loads(part[start:])})
                            except ValueError:
                                pass
        notes = []
        for role, text in texts(requests[0]):
            text = text.strip()
            if text and text != system.strip() and text != marker:
                notes.append({"role": role, "text": text})
        seen = {"codex": self.version, "model": self.cfg["agent_model"], "tools": direct, "script_tool_reaches": nested,
                "other_texts": notes, "features_switched_off": list(self.features)}
        atomic_write(Path(where) / "codex_check.json", json.dumps(seen, indent=1))
        problems = []
        extra = [n for n in direct if n not in wanted and n.rsplit(".", 1)[-1] not in ALLOWED_TOOLS]
        if extra:
            problems.append(f"it offers the model tools of its own that a run does not allow: {', '.join(extra)}")
        missing = [n for n in wanted if n not in direct]
        if missing:
            problems.append(f"it does not offer the session's tools directly: {', '.join(missing)}")
        reach = [n for n in nested if n not in ALLOWED_NESTED]
        if reach:
            problems.append(f"its script tool reaches tools a run does not allow: {', '.join(reach)}")
        if not any(text.strip() == system.strip() for _, text in texts(requests[0])):
            problems.append("it did not send the run's system prompt as it is")
        for note in notes:
            if not (note["role"] == "developer" and note["text"].startswith(ALLOWED_NOTES)):
                problems.append(f"it adds a text of its own ({note['role']}): {note['text'][:160]!r}")
        if problems:
            raise BackendError(f"the Codex CLI ({self.version}) with the model {self.cfg['agent_model']} is not closed off as a run "
                               "needs: " + "; ".join(problems) + f". What it sent is in {Path(where) / 'codex_check.json'}; "
                               "see docs/OPENAI.md")
        return seen

    # a round ---------------------------------------------------------------------------------
    def available(self, index: int) -> bool:
        return True

    def _wait(self, attempt: int) -> float:
        wait = min(self.retries["max_wait_s"], self.retries["first_wait_s"] * 2**attempt)
        return wait * (1 + 0.1 * random.random())

    def _classify(self, text: str) -> tuple[bool, str]:
        """(retry?, description) for an error the CLI reported."""
        low = text.lower()
        if any(w in low for w in MODEL_WORDS):
            return False, f"the model {self.cfg['agent_model']} cannot be used under this login: {text[:300]}"
        if any(w in low for w in LIMIT_WORDS):
            return True, f"usage or rate limit: {text[:300]}"
        if any(w in low for w in LOGIN_WORDS):
            return False, f"Codex login problem: {text[:300]}"
        return True, f"Codex error: {text[:300]}"

    def respond(self, prompt_text: str, index: int, where, session) -> Response:
        where = Path(where)
        where.mkdir(parents=True, exist_ok=True)
        system = SYSTEM_PROMPT if session.TOOLS else SUMMARY_SYSTEM_PROMPT
        instructions = where / "codex_instructions.md"
        atomic_write(instructions, system)
        checked = None
        if self.check_isolation:
            key = (self.command, self.version, self.cfg["agent_model"], str(self.cfg.get("effort")), tuple(session.TOOLS), str(self.home))
            if key not in _CHECKED:
                _CHECKED[key] = self.isolation(session, where, instructions, system)
                self.log(f"  Codex {self.version}: closed off as a run needs (besides the session's tools it offers "
                         f"{', '.join(n for n in _CHECKED[key]['tools'] if not n.startswith(NAMESPACE)) or 'nothing'})")
            checked = _CHECKED[key]
        cmd = self.command_line(session, instructions)
        t0 = time.monotonic()
        tries, timeout = int(self.retries["tries"]), float(self.cfg["session"]["seconds"])
        max_calls = int(self.cfg["session"]["max_turns"])
        failures, restarts, request_prompts = [], [], []
        got, text, stop_reason, stop_details, usage = parse_events(""), "", "error", None, {}
        for attempt in range(tries):
            session.reload()  # the tools run in the tool server's process; an earlier try may have used some
            note = prompt.restart_note(session)
            if note:
                restarts.append(dict(session.state["used"]))
            n = next_stream(where)
            request_text = prompt_text + note
            request_file = f"request_{n:02d}.md"
            atomic_write(where / request_file, request_text)
            request_prompts.append(request_file)
            with tempfile.TemporaryDirectory(prefix="lv1-codex-") as cwd:
                try:
                    session.begin_attempt(timeout)
                    r = self.run(cmd, request_text, self.env, cwd, timeout, max_calls)
                except OSError as exc:
                    raise StopRun(f"could not run the Codex CLI: {exc}") from exc
            # Everything the CLI emitted on this try, as it came (a try stopped at a limit: what it emitted until then).
            atomic_write(where / f"stream_{n:02d}.jsonl", r.stdout or "")
            got = parse_events(r.stdout or "")
            usage = got["usage"] or usage
            text = "\n".join(m.rstrip("\n") for m in got["messages"])
            if got["foreign"]:
                raise StopRun(f"the Codex session used what a run does not allow ({', '.join(sorted(set(got['foreign'])))}); the run "
                              f"stops, and this round's record is in {where}")
            if getattr(r, "stopped", None) == "timeout":
                failures.append(f"the session did not finish within {timeout:g} s")
                self.log(f"  the session reached its time limit ({timeout:g} s); what it submitted stands")
                text = ("(the session ran out of time; partial response follows)\n\n" + text) if text else "(the session ran out of time)"
                stop_reason, stop_details = "timeout", {"timeout_s": timeout}
                break
            if getattr(r, "stopped", None) == "turn_limit":
                text = text or "(the session reached its turn limit)"
                stop_reason, stop_details = "turn_limit", {"max_turns": max_calls}
                break
            if r.returncode == 0 and got["completed"] and not got["failed"]:
                stop_reason, stop_details = "end_turn", None
                break
            said = got["failed"] or "; ".join(e for e in got["errors"] if e) or (r.stderr or "")[-STDERR_TAIL:] or f"exit code {r.returncode}"
            retry, what = self._classify(said)
            failures.append(what)
            if not retry:
                raise StopRun(f"Codex stopped the run: {what}")
            if attempt + 1 >= tries:
                raise StopRun(f"Codex kept failing after {tries} tries; last: {what}. Resume later with `python -m level1.loop resume`.")
            wait = self._wait(attempt)
            self.log(f"  Codex: {what}; trying again in {wait:.0f} s")
            self.sleep(wait)

        atomic_write(where / "response_raw.json", json.dumps({"assistant_messages": got["messages"], "usage": usage, "items": got["items"]}))
        session.reload()  # the tools ran in the tool server's process; the counters are in the saved session
        total, cached = int(usage.get("input_tokens") or 0), int(usage.get("cached_input_tokens") or 0)
        written = int(usage.get("cache_write_input_tokens") or 0)
        meta = {
            "backend": self.name,
            "source": got["thread"],
            "model": self.cfg["agent_model"],
            "stop_reason": stop_reason,
            "stop_details": stop_details,
            "timed_out": stop_reason == "timeout",
            "models_used": {self.cfg["agent_model"]: int(usage.get("output_tokens") or 0)},  # the CLI's events name no model
            "model_events": [],
            "input_tokens": max(0, total - cached - written),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "reasoning_tokens": int(usage.get("reasoning_output_tokens") or 0),
            "cache_creation_input_tokens": written,
            "cache_read_input_tokens": cached,
            "auth": self.auth,
            "billed_to": "none",  # a ChatGPT plan is not billed per call
            "cost_usd": 0.0,
            "api_equivalent_usd": None,
            "codex_version": self.version,
            "codex_check": checked,
            "num_turns": got["tool_calls"],
            "assistant_messages": len(got["messages"]),
            "tools_used": dict(session.state["used"]),
            "submissions": len(session.receipts()),
            "request": {"command": cmd[:13], "prompt_files": request_prompts},
            "failed_tries": failures,
            "restarted_with_used": restarts,
            "seconds": round(time.monotonic() - t0, 1),
        }
        return Response(text=str(text or ""), meta=meta)
