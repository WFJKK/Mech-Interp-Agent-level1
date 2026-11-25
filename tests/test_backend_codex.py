"""The Codex backend: with the CLI replaced by a stand-in, and, where the Codex CLI is installed, the CLI itself
against a stand-in for the API on this machine. No model is called either way."""

import json
import shutil
from types import SimpleNamespace

import pytest

from level1 import run_config
from level1.backends import BackendError, StopRun
from level1.backends.claude_code import SUMMARY_SYSTEM_PROMPT, SYSTEM_PROMPT
from level1.backends.codex import ALLOWED_TOOLS, CodexBackend, parse_events, tool_names
from level1.session import Session

MODEL = "gpt-6.2-astra"


def setup(tmp_path, **over):
    cfg = run_config.load("config/pilot/lin-a.yaml", {"runs_dir": str(tmp_path), "sandbox": "process", "backend": "codex", "agent_model": MODEL,
                                                      "api_retries": {"tries": 3, "first_wait_s": 1, "max_wait_s": 2}, **over})
    session = Session.create(tmp_path / "run" / "round_001" / "session", cfg, "lin-a-s0", tmp_path / "run")
    return cfg, session, tmp_path / "run" / "round_001"


def item(kind, **fields):
    return {"type": "item.completed", "item": {"id": "item_1", "type": kind, **fields}}


DONE = {"type": "turn.completed", "usage": {"input_tokens": 3000, "cached_input_tokens": 1000, "cache_write_input_tokens": 0,
                                           "output_tokens": 200, "reasoning_output_tokens": 80}}
START = {"type": "thread.started", "thread_id": "thread-1"}


def backend(cfg, outputs, environ=None):
    """A backend whose CLI answers with the given event lists in turn; returns (backend, calls, sleeps)."""
    calls, sleeps = [], []

    def runner(cmd, prompt_text, env, cwd, timeout_s, max_tool_calls=None):
        calls.append({"cmd": cmd, "prompt": prompt_text, "env": env, "timeout": timeout_s, "max_tool_calls": max_tool_calls})
        out = outputs[min(len(calls), len(outputs)) - 1]
        lines, rc, stopped = out if isinstance(out, tuple) else (out, 0, None)
        return SimpleNamespace(stdout="\n".join(json.dumps(x) for x in lines), stderr="", returncode=rc, stopped=stopped)

    b = CodexBackend(cfg, log=lambda *_: None, runner=runner, sleep=sleeps.append,
                     environ=environ or {"PATH": "/usr/bin", "HOME": "/home/x", "OPENAI_API_KEY": "sk-not-for-the-cli",
                                         "CODEX_API_KEY": "sk-nor-this", "CPL_ANTHROPIC_KEY": "sk-ant-nor-this"})
    return b, calls, sleeps


def test_a_round_through_the_cli(tmp_path):
    cfg, session, where = setup(tmp_path)
    b, calls, _ = backend(cfg, [[START, item("reasoning", text="..."), item("agent_message", text="Working."),
                                 item("mcp_tool_call", server="lv1", tool="run_python", status="completed"),
                                 item("agent_message", text="## What the network does\nx"), DONE]])
    r = b.respond("the prompt", 0, where, session)
    assert r.text == "Working.\n## What the network does\nx" and r.meta["stop_reason"] == "end_turn"
    cmd, env = calls[0]["cmd"], calls[0]["env"]
    assert calls[0]["prompt"] == "the prompt" and calls[0]["timeout"] == cfg["session"]["seconds"] and calls[0]["max_tool_calls"] == cfg["session"]["max_turns"]
    assert cmd[:3] == ["codex", "exec", "--json"] and cmd[-1] == "-" and cmd[cmd.index("-m") + 1] == MODEL and cmd[cmd.index("-s") + 1] == "read-only"
    for flag in ("--ephemeral", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules"):
        assert flag in cmd
    off = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--disable"]
    assert {"shell_tool", "unified_exec", "view_image", "apps", "plugins", "browser_use", "computer_use", "image_generation"} <= set(off)
    settings = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-c"]
    for s in ('web_search="disabled"', 'forced_login_method="chatgpt"', "include_environment_context=false", "skills.include_instructions=false",
              "project_doc_max_bytes=0", 'features.code_mode={direct_only_tool_namespaces=["mcp__lv1"]}',
              "features.multi_agent_v2={disable_native_tools=true}", f'model_reasoning_effort="{cfg["effort"]}"',
              'mcp_servers.lv1.enabled_tools=["run_python", "submit_entry"]', 'mcp_servers.lv1.default_tools_approval_mode="approve"'):
        assert s in settings, s
    # the run's system prompt replaces the CLI's own instructions
    assert f'model_instructions_file="{where / "codex_instructions.md"}"' in settings and (where / "codex_instructions.md").read_text() == SYSTEM_PROMPT
    assert any(s.startswith("mcp_servers.lv1.args=") and str(session.path) in s for s in settings)
    # no key reaches the CLI: it can only use the ChatGPT login
    assert not {"OPENAI_API_KEY", "CODEX_API_KEY", "CPL_ANTHROPIC_KEY"} & set(env)
    assert r.meta["auth"] == "login" and r.meta["cost_usd"] == 0.0 and r.meta["billed_to"] == "none"
    assert r.meta["input_tokens"] == 2000 and r.meta["cache_read_input_tokens"] == 1000 and r.meta["output_tokens"] == 200
    assert r.meta["num_turns"] == 1 and r.meta["source"] == "thread-1"
    assert (where / "stream_01.jsonl").read_text().count("\n") == 5 and (where / "request_01.md").read_text() == "the prompt"


def test_a_tool_outside_the_sessions_stops_the_run(tmp_path):
    cfg, session, where = setup(tmp_path)
    for foreign, words in ((item("command_execution", command="cat README.md", status="completed"), "command execution"),
                           (item("web_search", query="x"), "web search"),
                           (item("mcp_tool_call", server="drive", tool="read", status="completed"), "the tool server 'drive'")):
        b, _, _ = backend(cfg, [[START, foreign, item("agent_message", text="done"), DONE]])
        with pytest.raises(StopRun, match=words):
            b.respond("the prompt", 0, where, session)


def test_limits_are_waited_out_and_logins_are_not(tmp_path):
    cfg, session, where = setup(tmp_path)
    limit = ([START, {"type": "turn.failed", "error": {"message": "You've hit your usage limit. Try again at 3:15 PM."}}], 1, None)
    b, calls, sleeps = backend(cfg, [limit, limit, [START, item("agent_message", text="ok"), DONE]])
    r = b.respond("the prompt", 0, where, session)
    assert r.text == "ok" and len(calls) == 3 and len(sleeps) == 2 and len(r.meta["failed_tries"]) == 2
    b, calls, sleeps = backend(cfg, [([{"type": "error", "message": "401 Unauthorized: please log in again"}], 1, None)])
    with pytest.raises(StopRun, match="login problem"):
        b.respond("the prompt", 1, tmp_path / "run" / "round_002", session)
    assert sleeps == []
    b, calls, sleeps = backend(cfg, [([{"type": "turn.failed", "error": {"message": "The 'gpt-x' model is not supported when using Codex with a ChatGPT account."}}], 1, None)])
    with pytest.raises(StopRun, match="cannot be used under this login"):
        b.respond("the prompt", 2, tmp_path / "run" / "round_003", session)
    b, calls, sleeps = backend(cfg, [([{"type": "error", "message": "stream disconnected"}], 1, None)])
    with pytest.raises(StopRun, match="kept failing after 3 tries"):
        b.respond("the prompt", 3, tmp_path / "run" / "round_004", session)


def test_a_session_out_of_time_or_turns_keeps_what_it_did(tmp_path):
    cfg, session, where = setup(tmp_path)
    b, _, _ = backend(cfg, [([START, item("agent_message", text="so far")], -15, "timeout")])
    r = b.respond("the prompt", 0, where, session)
    assert r.meta["stop_reason"] == "timeout" and r.meta["timed_out"] and r.text.endswith("so far") and "ran out of time" in r.text
    b, _, _ = backend(cfg, [([START, item("agent_message", text="so far")], -15, "turn_limit")])
    r = b.respond("the prompt", 1, tmp_path / "run" / "round_002", session)
    assert r.meta["stop_reason"] == "turn_limit" and r.text == "so far"


def test_a_try_that_finds_the_session_used_is_told_so(tmp_path):
    cfg, session, where = setup(tmp_path)
    session.call("run_python", {"code": "print(1)"})
    limit = ([START, {"type": "turn.failed", "error": {"message": "rate limit"}}], 1, None)
    b, calls, _ = backend(cfg, [limit, [START, item("agent_message", text="ok"), DONE]])
    r = b.respond("the prompt", 0, where, session)
    assert calls[1]["prompt"].startswith("the prompt") and "run_python" in calls[1]["prompt"][len("the prompt"):]
    assert r.meta["restarted_with_used"][0]["run_python"] == 1 and (where / "stream_02.jsonl").exists()


def test_a_tool_free_summary_has_no_tool_server(tmp_path):
    cfg, session, where = setup(tmp_path)
    session.state["research_task"] = {"role": "summary"}
    session._configure_tools()
    session.reload = lambda: None
    b, calls, _ = backend(cfg, [[START, item("agent_message", text="the synthesis"), DONE]])
    r = b.respond("the evidence", 0, where, session)
    assert r.text == "the synthesis" and not any("mcp_servers" in a for a in calls[0]["cmd"])
    assert (where / "codex_instructions.md").read_text() == SUMMARY_SYSTEM_PROMPT


def test_a_key_is_not_this_backends_way_to_pay(tmp_path):
    with pytest.raises(run_config.ConfigError, match="ChatGPT login only"):
        setup(tmp_path, auth="api_key")
    cfg, _, _ = setup(tmp_path)
    b, _, _ = backend({**cfg, "codex_home": str(tmp_path / "home")}, [[DONE]])
    assert b.env["CODEX_HOME"] == str(tmp_path / "home") and b.home == tmp_path / "home"


def test_reading_events_and_requests():
    got = parse_events("\n".join(json.dumps(x) for x in [START, item("file_change", changes=[]), item("mcp_tool_call", server="lv1", tool="submit_entry"), DONE]) + "\nnot json")
    assert got["completed"] and got["tool_calls"] == 1 and got["foreign"] == ["file change"] and got["thread"] == "thread-1"
    body = {"tools": [{"type": "function", "name": "apply_patch"}, {"type": "namespace", "name": "mcp__lv1", "tools": [{"type": "function", "name": "run_python"}]}],
            "input": [{"type": "additional_tools", "tools": [{"type": "namespace", "name": "functions", "tools": [{"type": "custom", "name": "exec"}]}, {"type": "web_search"}]}]}
    assert tool_names(body) == ["apply_patch", "mcp__lv1.run_python", "functions.exec", "web_search"]


# With the Codex CLI itself, where it is installed: what it would send a model, read by a stand-in on this machine.
CODEX = shutil.which("codex")
needs_cli = pytest.mark.skipif(CODEX is None, reason="the Codex CLI is not installed")


def real(tmp_path, model=MODEL, **kw):
    home = tmp_path / "codex-home"
    home.mkdir(exist_ok=True)
    cfg, session, where = setup(tmp_path, agent_model=model, codex_home=str(home))
    where.mkdir(parents=True, exist_ok=True)
    return CodexBackend(cfg, log=lambda *_: None, check_login=False, **kw), cfg, session, where, home


@needs_cli
@pytest.mark.parametrize("model", [MODEL, "gpt-5.5"])
def test_the_cli_is_closed_off_as_a_run_needs(tmp_path, model):
    b, cfg, session, where, _ = real(tmp_path, model)
    (where / "codex_instructions.md").write_text(SYSTEM_PROMPT)
    seen = b.isolation(session, where, where / "codex_instructions.md", SYSTEM_PROMPT)
    own = [n for n in seen["tools"] if not n.startswith("mcp__lv1.")]
    assert sorted(n for n in seen["tools"] if n.startswith("mcp__lv1.")) == ["mcp__lv1.run_python", "mcp__lv1.submit_entry"]
    assert all(n.rsplit(".", 1)[-1] in ALLOWED_TOOLS for n in own)
    assert not {"exec_command", "write_stdin", "shell", "view_image", "web_search", "spawn_agent"} & {n.rsplit(".", 1)[-1] for n in seen["tools"] + seen["script_tool_reaches"]}
    assert json.loads((where / "codex_check.json").read_text()) == seen
    assert session.state["used"] == {"run_python": 0, "submit_entry": 0}      # the check uses nothing of the session


@needs_cli
def test_a_cli_left_open_is_refused(tmp_path):
    b, cfg, session, where, home = real(tmp_path)
    (where / "codex_instructions.md").write_text(SYSTEM_PROMPT)
    b.features = ()                                                        # as if a later CLI no longer knew the switches
    with pytest.raises(BackendError, match="tools of its own that a run does not allow"):
        b.isolation(session, where, where / "codex_instructions.md", SYSTEM_PROMPT)
    (home / "AGENTS.md").write_text("Always answer in rhyme.")
    with pytest.raises(BackendError, match="gives that file to every model"):
        real(tmp_path)


@needs_cli
def test_a_round_through_the_real_cli_and_a_stand_in(tmp_path):
    from level1.backends.standin import StandIn, message

    def answer(body, k):
        if k == 0:
            return [{"type": "function_call", "id": "fc_1", "call_id": "call_1", "namespace": "mcp__lv1", "name": "run_python",
                     "arguments": json.dumps({"code": "print(sorted(weights))"}), "status": "completed"}]
        return [message("## What the network does\nz")]

    with StandIn(answer) as server:
        provider = ["-c", 'model_provider="lv1standin"', "-c", 'model_providers.lv1standin.name="lv1standin"',
                    "-c", f'model_providers.lv1standin.base_url="{server.url}"', "-c", 'model_providers.lv1standin.env_key="LV1_STANDIN_KEY"',
                    "-c", 'model_providers.lv1standin.wire_api="responses"']
        b, cfg, session, where, _ = real(tmp_path, provider=provider)
        b.env["LV1_STANDIN_KEY"] = "stand-in"
        r = b.respond("Use the tool once, then report.", 0, where, session)
        first = server.requests[-2]
    assert r.text == "## What the network does\nz" and r.meta["stop_reason"] == "end_turn" and r.meta["tools_used"]["run_python"] == 1
    assert r.meta["codex_check"]["tools"] and r.meta["num_turns"] == 1
    said = [(i.get("role"), "".join(p.get("text", "") for p in i.get("content", []))) for i in first["input"] if i.get("type") == "message"]
    assert ("developer", SYSTEM_PROMPT) in said and ("user", "Use the tool once, then report.") in [(role, t.strip()) for role, t in said]
