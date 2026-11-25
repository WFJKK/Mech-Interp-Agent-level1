"""The OpenAI API backend with the API replaced by a stand-in: no request leaves the machine."""

import json

import pytest

from level1 import budget, run_config
from level1.backends import BackendError, StopRun, make
from level1.backends.claude_code import SUMMARY_SYSTEM_PROMPT, SYSTEM_PROMPT
from level1.backends.openai_api import OpenAIBackend, price_of
from level1.session import Session
from level1.util import read_jsonl

MODEL = "gpt-6.2-sol"
KEY = "sk-stand-in-0123456789"
OVER = {"backend": "openai_api", "auth": "api_key", "api_key_env": "CPL_OPENAI_KEY", "agent_model": MODEL,
        "api_retries": {"tries": 3, "first_wait_s": 1, "max_wait_s": 2}}


def setup(tmp_path, **over):
    cfg = run_config.load("config/pilot/lin-a.yaml", {"runs_dir": str(tmp_path), "sandbox": "process", **OVER, **over})
    session = Session.create(tmp_path / "run" / "round_001" / "session", cfg, "lin-a-s0", tmp_path / "run")
    return cfg, session, tmp_path / "run" / "round_001"


def usage(inp=1000, cached=0, out=100, written=0):
    return {"input_tokens": inp, "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written},
            "output_tokens": out, "output_tokens_details": {"reasoning_tokens": 40}, "total_tokens": inp + out}


def answer(*items, model=MODEL + "-2026-10-01", status="completed", **over):
    resp = {"id": "resp_1", "object": "response", "status": status, "model": model, "output": list(items), "usage": usage(), **over}
    return 200, [{"type": "response.created", "response": {"id": "resp_1"}},
                 {"type": "response.completed" if status == "completed" else f"response.{status}", "response": resp}], "", {}


def call(name, args, call_id="call_1"):
    return {"type": "function_call", "id": "fc_" + call_id, "call_id": call_id, "name": name, "arguments": json.dumps(args), "status": "completed"}


def said(text):
    return {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": text, "annotations": []}]}


REASONING = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAAA-encrypted"}


class StandIn:
    """Answers each post with the next scripted answer (the last one again when they run out)."""
    check_key = True

    def __init__(self, answers, get=(200, "{}")):
        self.answers, self.posts, self.gets, self.get_answer = list(answers), [], [], get

    def get(self, url, headers, timeout_s=30):
        self.gets.append({"url": url, "headers": headers})
        return self.get_answer

    def post(self, url, headers, body, idle_s, deadline):
        self.posts.append({"url": url, "headers": headers, "body": json.loads(json.dumps(body))})
        out = self.answers[min(len(self.posts), len(self.answers)) - 1]
        if isinstance(out, Exception):
            raise out
        return out


def backend(cfg, answers, **kw):
    t = StandIn(answers, **kw)
    sleeps = []
    return OpenAIBackend(cfg, log=lambda *_: None, transport=t, sleep=sleeps.append, environ={"CPL_OPENAI_KEY": KEY, "OPENAI_API_KEY": "sk-not-this-one"}), t, sleeps


def test_a_round_through_the_api(tmp_path):
    cfg, session, where = setup(tmp_path)
    b, t, _ = backend(cfg, [answer(REASONING, call("run_python", {"code": "print(sorted(weights))"})),
                            answer(said("## What the network does\nx"))])
    r = b.respond("the prompt", 0, where, session)
    assert r.text == "## What the network does\nx" and r.meta["stop_reason"] == "end_turn" and r.meta["num_turns"] == 2
    # the key is checked once, free of charge, and goes in the Authorization header only
    assert t.gets[0]["url"] == "https://api.openai.com/v1/models/gpt-6.2-sol"
    first, second = t.posts[0], t.posts[1]
    assert first["url"] == "https://api.openai.com/v1/responses" and first["headers"] == {"authorization": f"Bearer {KEY}"}
    body = first["body"]
    # the session's two tools and nothing else; nothing stored at OpenAI; the reasoning comes back encrypted
    assert [t_["name"] for t_ in body["tools"]] == ["run_python", "submit_entry"] and all(t_["type"] == "function" for t_ in body["tools"])
    assert body["store"] is False and body["include"] == ["reasoning.encrypted_content"] and body["stream"] is True
    assert body["model"] == MODEL and body["reasoning"] == {"effort": cfg["effort"]}
    assert body["input"][0] == {"role": "developer", "content": SYSTEM_PROMPT}
    assert body["input"][1]["role"] == "user" and body["input"][1]["content"].startswith("the prompt")
    assert "This attempt's API allowance" in body["input"][1]["content"]
    # the second turn carries the first answer's items as they came, and the tool's result after them
    items = second["body"]["input"]
    assert items[2] == REASONING and items[3]["type"] == "function_call"
    assert items[4]["type"] == "function_call_output" and items[4]["call_id"] == "call_1"
    assert json.loads(items[4]["output"])["status"] == "ok" and "embed" in json.loads(items[4]["output"])["output"]
    assert session.state["used"]["run_python"] == 1 and r.meta["tools_used"]["run_python"] == 1
    # what it cost, by the reported tokens at the config's prices: two answers of 1000 in and 100 out
    assert r.meta["cost_usd"] == pytest.approx(2 * (1000 * 2.00 + 100 * 10.00) / 1e6)
    assert r.meta["auth"] == "api_key" and r.meta["billed_to"] == "the key in CPL_OPENAI_KEY" and r.meta["models_used"] == {MODEL + "-2026-10-01": 200}
    spend = read_jsonl(tmp_path / "run" / "spend.jsonl")
    assert [e["event"] for e in spend] == ["start", "end"] and spend[1]["cost_usd"] == pytest.approx(r.meta["cost_usd"]) and spend[1]["ended"] == "done"
    record = read_jsonl(where / "stream_01.jsonl")
    assert [e["event"] for e in record] == ["request", "response", "response"] and (where / "request_01.md").read_text().startswith("the prompt")


def test_the_key_never_reaches_a_record(tmp_path):
    cfg, session, where = setup(tmp_path)
    b, t, _ = backend(cfg, [(401, [], json.dumps({"error": {"code": "invalid_api_key", "message": f"Incorrect API key provided: {KEY}"}}), {})])
    with pytest.raises(StopRun) as exc:
        b.respond("the prompt", 0, where, session)
    assert KEY not in str(exc.value) and "CPL_OPENAI_KEY was refused" in str(exc.value)
    for path in (tmp_path / "run").rglob("*"):
        if path.is_file():
            assert KEY not in path.read_text(errors="replace"), path
    # what the try cost so far (nothing) is counted, so the budget's log has no open try
    assert read_jsonl(tmp_path / "run" / "spend.jsonl")[-1] == {**read_jsonl(tmp_path / "run" / "spend.jsonl")[-1], "event": "end", "cost_usd": 0.0}


def test_prices(tmp_path):
    p = {"input": 2.0, "cached_input": 0.1, "output": 10.0, "long_context": {"above_input_tokens": 272000, "input": 4.0, "cached_input": 0.2, "output": 15.0}}
    assert price_of(usage(1000, 400, 100), p) == pytest.approx((600 * 2.0 + 400 * 0.1 + 100 * 10.0) / 1e6)
    # tokens written to the cache are billed as input where the model has no price for them, at theirs where it has
    assert price_of(usage(1000, 0, 0, written=500), p) == pytest.approx(1000 * 2.0 / 1e6)
    assert price_of(usage(1000, 0, 0, written=500), {**p, "cache_write": 2.5}) == pytest.approx((500 * 2.0 + 500 * 2.5) / 1e6)
    # a request over the long-context line is billed at those prices throughout
    assert price_of(usage(300000, 100000, 1000), p) == pytest.approx((200000 * 4.0 + 100000 * 0.2 + 1000 * 15.0) / 1e6)
    assert price_of({}, p) == 0.0


def test_a_session_at_its_spending_cap_keeps_what_it_did(tmp_path):
    cfg, session, where = setup(tmp_path, budget={"run_usd": 0.02, "round_usd": 0.01})
    costly = answer(call("run_python", {"code": "print(1)"}), usage=usage(4000, 0, 400))   # $0.012 at 2 and 10 a million
    b, t, _ = backend(cfg, [costly, answer(said("never reached"))])
    r = b.respond("the prompt", 0, where, session)
    # the cap is checked after the answer, and the tools that answer called still ran
    assert len(t.posts) == 1 and r.meta["stop_reason"] == "budget" and session.state["used"]["run_python"] == 1
    assert "spending cap ($0.01)" in r.text and r.meta["budget"]["caps_usd"] == [0.01]
    assert budget.state(tmp_path / "run", cfg["budget"])["spent"] == pytest.approx(0.012)
    # less than a cent is left of the run's two: no round can start, and one that is asked makes no request
    assert "the run's API budget is spent" in b.exhausted(tmp_path / "run")
    session2 = Session.create(tmp_path / "run" / "round_002" / "session", cfg, "lin-a-s0", tmp_path / "run")
    r2 = b.respond("the prompt", 1, tmp_path / "run" / "round_002", session2)
    assert len(t.posts) == 1 and r2.meta["stop_reason"] == "budget" and r2.meta["cost_usd"] == 0.0


def test_limits_are_waited_out_in_place_and_the_rest_stops_the_run(tmp_path):
    cfg, session, where = setup(tmp_path)
    limit = (429, [], json.dumps({"error": {"code": "rate_limit_exceeded", "message": "Rate limit reached"}}), {"retry-after": "1.5"})
    b, t, sleeps = backend(cfg, [answer(call("run_python", {"code": "print(1)"})), limit, ConnectionResetError("dropped"), answer(said("done"))])
    r = b.respond("the prompt", 0, where, session)
    # the same turn is asked again with the conversation as it stood: nothing of the session is lost or repeated
    assert r.text == "done" and len(t.posts) == 4 and t.posts[1]["body"] == t.posts[2]["body"] == t.posts[3]["body"]
    assert session.state["used"]["run_python"] == 1 and len(sleeps) == 2 and 1.5 <= sleeps[0] <= 1.66 and len(r.meta["failed_tries"]) == 2
    for body, words in ((json.dumps({"error": {"code": "insufficient_quota", "message": "You exceeded your current quota"}}), "cannot pay"),
                        (json.dumps({"error": {"code": "model_not_found", "message": "The model does not exist or you do not have access to it"}}), "cannot be used"),
                        (json.dumps({"error": {"code": "invalid_request_error", "message": "Unsupported value: reasoning.effort"}}), "did not take the request")):
        b, t, sleeps = backend(cfg, [(429 if "quota" in body else 400, [], body, {})])
        with pytest.raises(StopRun, match=words):
            b.respond("the prompt", 1, tmp_path / "run" / f"round_{len(body)}", session)
        assert sleeps == []
    b, t, sleeps = backend(cfg, [(503, [], "upstream overloaded", {})])
    with pytest.raises(StopRun, match="kept failing after 3 tries"):
        b.respond("the prompt", 2, tmp_path / "run" / "round_9", session)
    assert len(t.posts) == 3


def test_a_refusal_another_model_and_a_full_context(tmp_path):
    cfg, session, where = setup(tmp_path)
    refusal = {"type": "message", "id": "m", "role": "assistant", "status": "completed", "content": [{"type": "refusal", "refusal": "I can't help with that."}]}
    b, t, _ = backend(cfg, [answer(refusal)])
    r = b.respond("the prompt", 0, where, session)
    assert r.meta["stop_reason"] == "refusal" and r.meta["stop_details"] == {"category": "I can't help with that."}
    b, t, _ = backend(cfg, [answer(said("x"), model="gpt-6-luna-2026-05-01")])
    with pytest.raises(StopRun, match="answered with gpt-6-luna-2026-05-01 instead of gpt-6.2-sol"):
        b.respond("the prompt", 1, tmp_path / "run" / "round_002", session)
    full = (400, [], json.dumps({"error": {"code": "context_length_exceeded", "message": "Your input exceeds the context window of this model."}}), {})
    b, t, _ = backend(cfg, [answer(call("run_python", {"code": "print(1)"})), full])
    r = b.respond("the prompt", 2, tmp_path / "run" / "round_003", session)
    assert r.meta["stop_reason"] == "context_limit" and session.state["used"]["run_python"] == 1


def test_turns_time_and_a_used_session(tmp_path):
    cfg, session, where = setup(tmp_path, session={"max_turns": 2})
    b, t, _ = backend(cfg, [answer(said("thinking"), call("run_python", {"code": "print(1)"}))])
    r = b.respond("the prompt", 0, where, session)
    assert r.meta["stop_reason"] == "turn_limit" and len(t.posts) == 2 and r.text == "thinking\nthinking"
    # a try that finds the session used (a resumed run) is told so, and its record is numbered on
    b, t, _ = backend(cfg, [TimeoutError("out of time")])
    r = b.respond("the prompt", 0, where, session)
    assert r.meta["stop_reason"] == "timeout" and r.meta["timed_out"] and r.meta["restarted_with_used"] == [{"run_python": 2, "submit_entry": 0}]
    assert "run_python" in t.posts[0]["body"]["input"][1]["content"].split("the prompt", 1)[1]
    assert (where / "stream_02.jsonl").exists() and (where / "request_02.md").exists()


def test_a_bad_tool_call_is_refused_not_fatal(tmp_path):
    cfg, session, where = setup(tmp_path)
    bad = {"type": "function_call", "id": "f", "call_id": "c9", "name": "read_file", "arguments": "{not json"}
    b, t, _ = backend(cfg, [answer(bad, call("submit_entry", {"name": "nope", "claimed_fidelity": 1.0}, "c10")), answer(said("ok"))])
    r = b.respond("the prompt", 0, where, session)
    outs = [i for i in t.posts[1]["body"]["input"] if i.get("type") == "function_call_output"]
    assert [o["call_id"] for o in outs] == ["c9", "c10"] and all(o["output"].startswith("refused: ") for o in outs) and r.text == "ok"


def test_a_tool_free_summary(tmp_path):
    cfg, session, where = setup(tmp_path)
    session.state["research_task"] = {"role": "summary"}
    session._configure_tools()
    session.reload = lambda: None
    b, t, _ = backend(cfg, [answer(said("the synthesis"))])
    r = b.respond("the evidence", 0, where, session)
    body = t.posts[0]["body"]
    assert "tools" not in body and body["input"][0] == {"role": "developer", "content": SUMMARY_SYSTEM_PROMPT} and r.text == "the synthesis"


def test_the_config_says_who_pays_and_at_what_price(tmp_path):
    load = lambda **over: run_config.load("config/pilot/lin-a.yaml", {"runs_dir": str(tmp_path), **OVER, **over})
    with pytest.raises(run_config.ConfigError, match="bills an API key"):
        load(auth="login")
    with pytest.raises(run_config.ConfigError, match="other than"):
        load(api_key_env="OPENAI_API_KEY")
    with pytest.raises(run_config.ConfigError, match="no entry for the model"):
        load(agent_model="gpt-of-next-year")
    with pytest.raises(run_config.ConfigError, match="cannot set"):
        load(openai={"request": {"store": True}})
    with pytest.raises(run_config.ConfigError, match="price"):
        load(openai={"prices": {MODEL: {"input": 0, "output": 10}}})
    cfg = load(openai={"request": {"service_tier": "priority"}})
    with pytest.raises(BackendError, match="CPL_OPENAI_KEY is not set"):
        OpenAIBackend(cfg, environ={"OPENAI_API_KEY": "sk-not-this-one"}, transport=StandIn([]))
    with pytest.raises(BackendError, match="reads like an Anthropic key"):
        OpenAIBackend({**cfg, "api_key_env": "CPL_ANTHROPIC_KEY"}, environ={"CPL_ANTHROPIC_KEY": "sk-ant"}, transport=StandIn([]))
    with pytest.raises(BackendError, match="refused the key"):
        OpenAIBackend(cfg, environ={"CPL_OPENAI_KEY": KEY}, transport=StandIn([], get=(401, "no")))
    with pytest.raises(BackendError, match="cannot use the model"):
        OpenAIBackend(cfg, environ={"CPL_OPENAI_KEY": KEY}, transport=StandIn([], get=(404, "no such model")))
    b = OpenAIBackend(cfg, environ={"CPL_OPENAI_KEY": KEY}, transport=StandIn([]))
    assert b.request_body([], [])["service_tier"] == "priority"
    assert isinstance(make.__module__, str)


def test_the_real_http_path_against_a_stand_in_on_this_machine(tmp_path):
    """Transport itself (urllib, the event stream), the whole way: a server on 127.0.0.1 stands in for the API."""
    from level1.backends.standin import StandIn as Server, message

    def answers(body, k):
        return [REASONING, call("run_python", {"code": "print(len(weights))"})] if k == 0 else [message("## What the network does\ny")]

    with Server(answers, usage=usage(2000, 500, 300), model=MODEL + "-2026-10-01") as server:
        cfg, session, where = setup(tmp_path, openai={"base_url": server.url})
        b = OpenAIBackend(cfg, log=lambda *_: None, environ={"CPL_OPENAI_KEY": KEY})
        r = b.respond("the prompt", 0, where, session)
        assert r.text == "## What the network does\ny" and session.state["used"]["run_python"] == 1 and len(server.requests) == 2
        assert server.headers[0]["authorization"] == f"Bearer {KEY}" and server.requests[1]["input"][4]["type"] == "function_call_output"
        assert r.meta["cost_usd"] == pytest.approx(2 * (1500 * 2.00 + 500 * 0.10 + 300 * 10.00) / 1e6)
    for path in (tmp_path / "run").rglob("*"):
        if path.is_file():
            assert KEY not in path.read_text(errors="replace"), path
