"""The Claude Code backend with the CLI replaced by a stand-in, and the tool server's protocol."""

import io
import json
import subprocess
from types import SimpleNamespace

import pytest

from level1 import mcp_server, prompt, run_config
from level1.backends import StopRun
from level1.backends.claude_code import ClaudeCodeBackend, join, parse_output
from level1.session import Session

MODEL = "claude-fable-5-1"


@pytest.fixture()
def setup(tmp_path):
    cfg = run_config.load("config/pilot/lin-a.yaml", {"runs_dir": str(tmp_path), "sandbox": "process",
                                                      "api_retries": {"tries": 3, "first_wait_s": 1, "max_wait_s": 2}})
    session = Session.create(tmp_path / "run" / "round_001" / "session", cfg, "lin-a-s0", tmp_path / "run")
    return cfg, session, tmp_path / "run" / "round_001"


def assistant(text, stop="end_turn"):
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}], "stop_reason": stop}}


def result(**over):
    base = {"type": "result", "subtype": "success", "is_error": False, "result": "last piece", "session_id": "abc", "num_turns": 7,
            "total_cost_usd": 1.25, "usage": {"input_tokens": 100, "output_tokens": 50},
            "modelUsage": {MODEL: {"outputTokens": 50}}}
    return {**base, **over}


def backend(cfg, outputs, environ=None):
    """A backend whose CLI answers with the given outputs in turn; returns (backend, calls, sleeps)."""
    calls, sleeps = [], []

    def runner(cmd, prompt_text, env, cwd, timeout_s):
        calls.append({"cmd": cmd, "prompt": prompt_text, "env": env, "timeout": timeout_s})
        out = outputs[min(len(calls), len(outputs)) - 1]
        if isinstance(out, Exception):
            raise out
        lines, rc = out if isinstance(out, tuple) else (out, 0)
        return SimpleNamespace(stdout="\n".join(json.dumps(x) for x in lines), stderr="", returncode=rc)

    b = ClaudeCodeBackend(cfg, log=lambda *_: None, runner=runner, sleep=sleeps.append,
                          environ=environ or {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk-not-for-the-cli", "HOME": "/home/x"})
    return b, calls, sleeps


def test_a_round_through_the_cli(setup):
    cfg, session, where = setup
    b, calls, _ = backend(cfg, [[assistant("Working.", None), assistant(" More.\n"), assistant("## What the network does\nx"), result()]])
    r = b.respond("the prompt", 0, where, session)
    assert r.text == "Working. More.\n## What the network does\nx"
    cmd, env = calls[0]["cmd"], calls[0]["env"]
    assert calls[0]["prompt"] == "the prompt" and calls[0]["timeout"] == cfg["session"]["seconds"]
    assert cmd[:4] == ["claude", "-p", "--tools", ""] and "--strict-mcp-config" in cmd and cmd[cmd.index("--model") + 1] == MODEL
    assert cmd[cmd.index("--allowedTools") + 1] == "mcp__lv1__run_python,mcp__lv1__submit_entry"
    assert cmd[cmd.index("--max-turns") + 1] == "150" and cmd[cmd.index("--effort") + 1] == "high"
    assert "ANTHROPIC_API_KEY" not in env and env["CLAUDE_CODE_NO_MODEL_FALLBACK"] == "1"
    mcp = json.loads((where / "mcp.json").read_text())["mcpServers"]["lv1"]
    assert str(session.path) in mcp["args"][1] and "level1.mcp_server" in mcp["args"][1]
    assert r.meta["stop_reason"] == "end_turn" and r.meta["output_tokens"] == 50 and r.meta["cost_usd"] == 0.0
    assert r.meta["api_equivalent_usd"] == 1.25 and r.meta["models_used"] == {MODEL: 50} and r.meta["submissions"] == 0
    assert json.loads((where / "response_raw.json").read_text())["session_id"] == "abc"


def test_another_model_answering_stops_the_run(setup):
    cfg, session, where = setup
    b, _, _ = backend(cfg, [[assistant("x"), result(modelUsage={MODEL: {"outputTokens": 1}, "claude-opus-5-5": {"outputTokens": 9}})]])
    with pytest.raises(StopRun, match="claude-opus-5-5"):
        b.respond("p", 0, where, session)
    b, _, _ = backend(cfg, [[{"type": "system", "subtype": "model_refusal_fallback", "original_model": MODEL, "fallback_model": "other"},
                             assistant("x"), result()]])
    with pytest.raises(StopRun, match="instead of"):
        b.respond("p", 0, where, session)


def test_a_refusal_is_recorded_as_one(setup):
    cfg, session, where = setup
    b, calls, _ = backend(cfg, [[{"type": "system", "subtype": "model_refusal_no_fallback", "api_refusal_category": "x"}, result(is_error=True)]])
    r = b.respond("p", 0, where, session)
    assert r.meta["stop_reason"] == "refusal" and r.meta["stop_details"] == {"category": "x"} and len(calls) == 1


def test_limits_are_waited_out_and_logins_are_not(setup):
    cfg, session, where = setup
    limited = [result(is_error=True, result="You have hit your usage limit")]
    b, calls, sleeps = backend(cfg, [limited, limited, [assistant("done"), result()]])
    r = b.respond("p", 0, where, session)
    assert r.text == "done" and len(calls) == 3 and len(sleeps) == 2 and len(r.meta["failed_tries"]) == 2
    b, calls, sleeps = backend(cfg, [limited])
    with pytest.raises(StopRun, match="kept failing after 3 tries"):
        b.respond("p", 0, where, session)
    b, calls, sleeps = backend(cfg, [[result(is_error=True, result="Not logged in")]])
    with pytest.raises(StopRun, match="login problem"):
        b.respond("p", 0, where, session)
    assert len(calls) == 1 and not sleeps


def test_a_try_that_finds_the_session_used_is_told_so(setup):
    """A dropped connection or a resumed run: the next try gets the prompt with the budgets as they stand,
    the saved entries and the submissions already made."""
    cfg, session, where = setup
    assert prompt.restart_note(session) == ""
    calls = []

    def runner(cmd, prompt_text, env, cwd, timeout_s):
        calls.append(prompt_text)
        if len(calls) == 1:  # the first try works a little, then the connection drops
            s = Session(session.path)
            s.run_python("g = graph()\nsave_entry('first', g.entry(predict=g.token(0)))")
            s.submit_entry("first", 0.5, 0.5)
            return SimpleNamespace(stdout=json.dumps(result(is_error=True, result="overloaded")), stderr="", returncode=1)
        return SimpleNamespace(stdout="\n".join(json.dumps(x) for x in [assistant("done"), result()]), stderr="", returncode=0)

    b = ClaudeCodeBackend(cfg, log=lambda *_: None, runner=runner, sleep=lambda s: None, environ={"PATH": "/usr/bin"})
    r = b.respond("the prompt", 0, where, session)
    assert calls[0] == "the prompt" and calls[1].startswith("the prompt\n\n## This session was started before")
    assert f"{cfg['session']['python_calls'] - 1} of {cfg['session']['python_calls']} `run_python` calls" in calls[1]
    assert f"{cfg['session']['submissions'] - 1} of {cfg['session']['submissions']} submissions" in calls[1]
    assert "- `first`: reward -3.9243, coverage 1.0000, fidelity 0.50757" in calls[1]  # token 0 as the answer, everywhere
    assert "still in `entries`: `first`." in calls[1]
    assert r.meta["restarted_with_used"] == [{"run_python": 1, "submit_entry": 1}] and r.meta["submissions"] == 1
    assert "{" not in prompt.restart_note(session).replace("`", "")


def test_a_session_out_of_time_or_turns_keeps_what_it_did(setup):
    cfg, session, where = setup
    b, calls, _ = backend(cfg, [subprocess.TimeoutExpired("claude", 1)])
    r = b.respond("p", 0, where, session)
    assert r.text == "(the session ran out of time)" and len(calls) == 1
    b, _, _ = backend(cfg, [([assistant("so far"), result(is_error=True, subtype="error_max_turns", result=None)], 1)])
    assert b.respond("p", 0, where, session).text == "so far"


def test_parse_and_join():
    out = "\n".join(["not json", json.dumps(assistant("a", "max_tokens")), json.dumps(assistant("b")), json.dumps(assistant("c")), "[1]",
                     json.dumps(result())])
    data, messages = parse_output(out)
    assert data["type"] == "result" and [m["text"] for m in messages] == ["a", "b", "c"]
    assert join(messages) == "ab\nc"  # a cut-off piece continues exactly; a finished one ends its line


def test_the_tool_server_protocol(setup):
    cfg, session, where = setup
    ask = lambda method, **params: mcp_server.handle(session, {"jsonrpc": "2.0", "id": 1, "method": method, "params": params})  # noqa: E731
    assert ask("initialize", protocolVersion="2025-03-26")["result"]["protocolVersion"] == "2025-03-26"
    assert [t["name"] for t in ask("tools/list")["result"]["tools"]] == ["run_python", "submit_entry"]
    ok = ask("tools/call", name="run_python", arguments={"code": "print(info['n_total'])"})["result"]
    assert ok["isError"] is False and json.loads(ok["content"][0]["text"])["output"] == "4096\n"
    no = ask("tools/call", name="submit_entry", arguments={"name": "missing", "claimed_fidelity": 1})["result"]
    assert no["isError"] is True and no["content"][0]["text"].startswith("refused: no saved entry")
    assert ask("tools/call", name="read_file", arguments={"path": "x"})["result"]["isError"] is True
    assert ask("nothing")["error"]["code"] == -32601
    assert mcp_server.handle(session, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    inp = io.StringIO('garbage\n{"jsonrpc": "2.0", "id": 5, "method": "ping"}\n\n')
    out = io.StringIO()
    mcp_server.serve(session, inp, out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert replies[0]["error"]["code"] == -32700 and replies[1] == {"jsonrpc": "2.0", "id": 5, "result": {}}
