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
    # everything the CLI printed is kept as it came: the model's text between its tool calls, in order
    stream = [json.loads(line) for line in (where / "stream_01.jsonl").read_text().splitlines() if line.strip()]
    assert [e["type"] for e in stream] == ["assistant", "assistant", "assistant", "result"]


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


# billed to an API key --------------------------------------------------------------------------------------
KEY = "sk-ant-test-0000-not-a-real-key-0000"


def init(source):
    return {"type": "system", "subtype": "init", "apiKeySource": source, "claude_code_version": "2.1.322",
            "tools": ["mcp__lv1__run_python", "mcp__lv1__submit_entry"]}


def billed(tmp_path, **over):
    base = {"runs_dir": str(tmp_path), "sandbox": "process", "auth": "api_key", "budget": {"run_usd": 5, "round_usd": 3},
            "api_retries": {"tries": 3, "first_wait_s": 1, "max_wait_s": 2}}
    cfg = run_config.load("config/pilot/lin-a.yaml", run_config.merge(base, over))
    return cfg, tmp_path / "run"


def round_of(cfg, run, k):
    where = run / f"round_{k:03d}"
    return Session.create(where / "session", cfg, "lin-a-s0", run), where


ENV = {"PATH": "/usr/bin", "HOME": "/home/x", "ANTHROPIC_API_KEY": "sk-not-for-the-cli", "ANTHROPIC_AUTH_TOKEN": "t",
       "CPL_ANTHROPIC_KEY": KEY}


def test_api_key_mode_bills_the_key_and_reads_no_login(tmp_path):
    cfg, run = billed(tmp_path)
    session, where = round_of(cfg, run, 1)
    b, calls, _ = backend(cfg, [[init("apiKeyHelper"), assistant("## What the network does\nx"),
                                 result(total_cost_usd=1.25)]], environ=dict(ENV))
    r = b.respond("the prompt", 0, where, session)
    cmd, env = calls[0]["cmd"], calls[0]["env"]
    assert cmd[:3] == ["claude", "-p", "--bare"]  # --bare reads no login: the key is the only way to pay
    assert json.loads(cmd[cmd.index("--settings") + 1]) == {"apiKeyHelper": "printenv CPL_ANTHROPIC_KEY"}
    assert cmd[cmd.index("--max-budget-usd") + 1] == "3.00" and cmd[cmd.index("--tools") + 1] == ""
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env and env["CPL_ANTHROPIC_KEY"] == KEY
    assert KEY not in " ".join(cmd)
    code = json.loads((where / "mcp.json").read_text())["mcpServers"]["lv1"]["args"][1]
    assert "os.environ.pop" in code and "'CPL_ANTHROPIC_KEY'" in code and "'ANTHROPIC_API_KEY'" in code  # the tool server drops them
    m = r.meta
    assert m["auth"] == "api_key" and m["billed_to"] == "apiKeyHelper" and m["cost_usd"] == 1.25 and m["claude_code_version"] == "2.1.322"
    assert m["budget"]["caps_usd"] == [3.0] and m["budget"]["run_counted_usd"] == 1.25 and m["budget"]["unreported_tries"] == 0
    spend = [json.loads(line) for line in (run / "spend.jsonl").read_text().splitlines()]
    assert [(e["event"], e["round"], e["try"], e.get("cap_usd"), e.get("cost_usd")) for e in spend] == \
        [("start", 1, 1, 3.0, None), ("end", 1, 1, None, 1.25)]
    # the key's value is in no file the run wrote
    assert not [p for p in run.rglob("*") if p.is_file() and KEY in p.read_text(errors="replace")]


def test_a_login_run_keeps_the_key_out_and_bills_nothing(setup):
    cfg, session, where = setup
    b, calls, _ = backend(cfg, [[init("none"), assistant("done"), result()]], environ=dict(ENV))
    r = b.respond("p", 0, where, session)
    cmd, env = calls[0]["cmd"], calls[0]["env"]
    assert "--bare" not in cmd and "--max-budget-usd" not in cmd and "--settings" not in cmd
    assert not {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CPL_ANTHROPIC_KEY"} & set(env)
    assert r.meta["auth"] == "login" and r.meta["billed_to"] == "none" and r.meta["cost_usd"] == 0.0 and "budget" not in r.meta
    assert not (where.parent / "spend.jsonl").exists()


def test_a_try_billed_another_way_stops_the_run(setup, tmp_path):
    cfg, session, where = setup
    b, _, _ = backend(cfg, [[init("ANTHROPIC_API_KEY"), assistant("x"), result()]])
    with pytest.raises(StopRun, match="billed 'ANTHROPIC_API_KEY'"):
        b.respond("p", 0, where, session)
    b, _, _ = backend(cfg, [[init("/login managed key"), assistant("x"), result(total_cost_usd=2.5)]])
    assert b.respond("p", 0, where, session).meta["cost_usd"] == 2.5  # a Console login is billed per token: say so
    cfg2, run = billed(tmp_path / "b")
    session2, where2 = round_of(cfg2, run, 1)
    b, _, _ = backend(cfg2, [[init("none"), assistant("x"), result()]], environ=dict(ENV))
    with pytest.raises(StopRun, match="billed 'none'"):
        b.respond("p", 0, where2, session2)


def test_api_key_mode_needs_its_variable(tmp_path):
    from level1.backends import BackendError

    cfg, _ = billed(tmp_path)
    with pytest.raises(BackendError, match="CPL_ANTHROPIC_KEY is not set"):
        backend(cfg, [[]], environ={"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk-not-for-the-cli"})
    with pytest.raises(BackendError, match="other than ANTHROPIC_API_KEY"):
        ClaudeCodeBackend({**cfg, "api_key_env": "ANTHROPIC_API_KEY"}, runner=lambda *a: None, environ=dict(ENV))
    with pytest.raises(BackendError, match="workspace ID"):
        backend(cfg, [[]], environ={**ENV, "CPL_ANTHROPIC_WORKSPACE": "x\nInjected: 1"})
    b, _, _ = backend(cfg, [[]], environ={**ENV, "CPL_ANTHROPIC_WORKSPACE": "wrkspc_01AbC"})
    assert b.env["ANTHROPIC_CUSTOM_HEADERS"] == "anthropic-workspace-id: wrkspc_01AbC"
    with pytest.raises(BackendError, match="line breaks") as caught:
        backend(cfg, [[]], environ={**ENV, "CPL_ANTHROPIC_KEY": KEY + "\r"})  # a key file with Windows line endings
    assert KEY not in str(caught.value)
    with pytest.raises(run_config.ConfigError, match="api_key_env must name"):
        billed(tmp_path, api_key_env="ANTHROPIC_API_KEY")
    with pytest.raises(run_config.ConfigError, match="budget.run_usd must be a positive"):
        billed(tmp_path, budget={"run_usd": 0})
    with pytest.raises(run_config.ConfigError, match="auth must be"):
        billed(tmp_path, auth="subscription")


def test_the_key_is_checked_before_the_run(tmp_path):
    from level1.backends import BackendError

    cfg, _ = billed(tmp_path)
    asked = []

    def http(status, raise_=None):
        def get(url, headers):
            asked.append((url, headers))
            if raise_:
                raise raise_
            return status, '{"type": "error", "error": {"message": "fake"}}'
        return get

    ok = ClaudeCodeBackend(cfg, runner=lambda *a: None, environ={**ENV, "CPL_ANTHROPIC_WORKSPACE": "wrkspc_1"}, http=http(200))
    url, headers = asked[-1]
    assert url == "https://api.anthropic.com/v1/models/claude-fable-5-1" and headers["x-api-key"] == KEY
    assert headers["anthropic-version"] == "2023-06-01" and headers["anthropic-workspace-id"] == "wrkspc_1" and ok.auth == "api_key"
    with pytest.raises(BackendError, match="refused the key in CPL_ANTHROPIC_KEY"):
        ClaudeCodeBackend(cfg, runner=lambda *a: None, environ=dict(ENV), http=http(401))
    with pytest.raises(BackendError, match="cannot use the model claude-fable-5-1"):
        ClaudeCodeBackend(cfg, runner=lambda *a: None, environ=dict(ENV), http=http(404))
    said = []
    ClaudeCodeBackend(cfg, log=said.append, runner=lambda *a: None, environ=dict(ENV), http=http(0, OSError("no network")))
    assert "could not check the API key" in said[0] and KEY not in said[0]
    ClaudeCodeBackend(cfg, runner=lambda *a: None, environ={**ENV, "ANTHROPIC_BASE_URL": "http://127.0.0.1:9/"}, http=http(200))
    assert asked[-1][0] == "http://127.0.0.1:9/v1/models/claude-fable-5-1"
    retired = lambda url, headers: (200, json.dumps({"type": "model", "id": "claude-fable-5-1", "lifecycle": "retired"}))  # noqa: E731
    with pytest.raises(BackendError, match="is retired"):
        ClaudeCodeBackend(cfg, runner=lambda *a: None, environ=dict(ENV), http=retired)


def test_the_budget_holds_across_tries_rounds_and_resumes(tmp_path):
    """A failed try's cost counts; a try that reported no cost counts at its cap; a spent budget ends the round
    with what it submitted, and the loop's question says so; a new budget lets the run go on."""
    from level1 import budget

    cfg, run = billed(tmp_path)  # $5 for the run, $3 a round
    session, where = round_of(cfg, run, 1)
    b, calls, sleeps = backend(cfg, [([init("apiKeyHelper"), result(is_error=True, result="Overloaded", total_cost_usd=0.5)], 1),
                                     [init("apiKeyHelper"), assistant("## What the network does\nx"), result(total_cost_usd=2.0)]],
                               environ=dict(ENV))
    r = b.respond("p", 0, where, session)
    assert [c["cmd"][c["cmd"].index("--max-budget-usd") + 1] for c in calls] == ["3.00", "2.50"]  # the round's $3 less the failed try
    assert r.meta["cost_usd"] == 2.5 and r.meta["budget"]["round_counted_usd"] == 2.5 and len(sleeps) == 1
    assert b.exhausted(run) is None

    # round 2: the session runs out of time; the CLI reported no cost, so the try counts at its cap
    session, where = round_of(cfg, run, 2)
    partial = (json.dumps(init("apiKeyHelper")) + "\n" + json.dumps(assistant("half", None)) + "\n").encode()
    b, calls, _ = backend(cfg, [subprocess.TimeoutExpired("claude", 1, output=partial)], environ=dict(ENV))
    r = b.respond("p", 1, where, session)
    assert calls[0]["cmd"][calls[0]["cmd"].index("--max-budget-usd") + 1] == "2.50"  # what is left of the run's $5
    assert r.text == "(the session ran out of time)" and r.meta["budget"]["unreported_tries"] == 1
    assert (where / "stream_01.jsonl").read_text().startswith('{"type": "system"')  # what it printed until then is kept
    assert b.exhausted(run) == "the run's API budget is spent: $5.00 counted of $5.00"

    # round 3: nothing is left, so the round ends before the CLI is started
    session, where = round_of(cfg, run, 3)
    b, calls, _ = backend(cfg, [[result()]], environ=dict(ENV))
    r = b.respond("p", 2, where, session)
    assert not calls and r.meta["stop_reason"] == "budget" and r.text.startswith("(the run's API budget is spent")

    budget.set_run_budget(run, 9)
    assert b.exhausted(run) is None and budget.cap(budget.state(run, cfg["budget"]), 3) == 3.0
    assert budget.cap(budget.state(run, cfg["budget"]), 1) == 0.5  # round 1 has $0.50 of its $3 left


def test_a_session_at_its_spending_cap_keeps_what_it_did(tmp_path):
    cfg, run = billed(tmp_path)
    session, where = round_of(cfg, run, 1)
    capped = [init("apiKeyHelper"), assistant("so far", None),
              result(is_error=True, subtype="error_max_budget_usd", result=None, total_cost_usd=3.4)]
    b, calls, _ = backend(cfg, [(capped, 1)], environ=dict(ENV))
    r = b.respond("p", 0, where, session)
    assert len(calls) == 1 and r.text == "so far" and r.meta["stop_reason"] == "budget" and r.meta["cost_usd"] == 3.4
    assert r.meta["stop_details"] == {"budget": "the session reached its spending cap ($3.00)"}


def test_the_key_never_reaches_a_record(tmp_path):
    """Should the CLI ever print the key, it is taken out of the stream and the messages before they are written."""
    cfg, run = billed(tmp_path)
    session, where = round_of(cfg, run, 1)
    b, _, _ = backend(cfg, [([init("apiKeyHelper"), assistant(f"echo {KEY}"), result(is_error=True, result=f"Invalid API key {KEY}")], 1)],
                      environ=dict(ENV))
    with pytest.raises(StopRun) as caught:
        b.respond("p", 0, where, session)
    assert KEY not in str(caught.value) and "<the API key>" in str(caught.value)
    assert not [p for p in run.rglob("*") if p.is_file() and KEY in p.read_text(errors="replace")]


def test_a_subscription_limit_is_still_waited_out(setup):
    """The API's words for an account that cannot pay must not catch a subscription's usage limit."""
    cfg, session, where = setup
    limited = [init("none"), result(is_error=True, result="Claude usage limit reached. You'll regain access at 5pm; see billing.")]
    b, calls, sleeps = backend(cfg, [(limited, 1), [init("none"), assistant("done"), result()]])
    assert b.respond("p", 0, where, session).text == "done" and len(calls) == 2 and len(sleeps) == 1


def test_a_model_that_cannot_be_used_stops_the_run(setup):
    cfg, session, where = setup
    said = ("There's an issue with the selected model (claude-fable-5-1). It may not exist or you may not have access to it. "
            "Run --model to pick a different model.")  # what Claude Code 2.1.322 says when the API answers 404
    b, calls, sleeps = backend(cfg, [([init("none"), result(is_error=True, result=said, total_cost_usd=0)], 1)])
    with pytest.raises(StopRun, match="cannot be used here"):
        b.respond("p", 0, where, session)
    assert len(calls) == 1 and not sleeps


def test_an_account_that_cannot_pay_stops_the_run(tmp_path):
    cfg, run = billed(tmp_path)
    session, where = round_of(cfg, run, 1)
    for said in ("Credit balance is too low",
                 "API Error: 400 You have reached your specified API usage limits. You will regain access on 2026-12-01 at 00:00 UTC."):
        b, calls, sleeps = backend(cfg, [([init("apiKeyHelper"), result(is_error=True, result=said, total_cost_usd=0)], 1)],
                                   environ=dict(ENV))
        with pytest.raises(StopRun, match="cannot pay for more"):
            b.respond("p", 0, where, session)
        assert len(calls) == 1 and not sleeps


def test_streams_are_numbered_on_and_never_written_over(setup):
    cfg, session, where = setup
    where.mkdir(parents=True, exist_ok=True)
    (where / "stream_01.jsonl").write_text("an earlier try\n")
    (where / "stream_02.jsonl").write_text("another\n")
    b, _, _ = backend(cfg, [[assistant("done"), result()]])
    b.respond("p", 0, where, session)
    assert (where / "stream_01.jsonl").read_text() == "an earlier try\n" and (where / "stream_03.jsonl").exists()
