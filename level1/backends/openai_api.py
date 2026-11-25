"""The OpenAI API backend: a round is one conversation with OpenAI's Responses API, held by this process.

No CLI stands between the loop and the model here. Each turn posts

    POST <openai.base_url>/responses
    {"model": <agent_model>, "input": [...], "tools": [the session's tools as function tools],
     "reasoning": {"effort": <effort>}, "store": false, "include": ["reasoning.encrypted_content"],
     "stream": true, "prompt_cache_key": ...}

with the run's system prompt as a developer message at the head of the input (where it is cached) and the
round's prompt as the user message after it. The model's answer is a list of items: its reasoning (encrypted,
since nothing is stored at OpenAI), calls of the session's tools, text. Every item goes back into the input as
it came, each call's result after it, and the next turn is posted, until an answer calls no tool. The tools
run in this process, against the same session, sandbox and verifier as under any other backend; the model has
no other tool, no file and no web.

Who pays: an API key only (`auth: api_key`), billed per token. The key stays in the variable `api_key_env`
names (CPL_OPENAI_KEY in the configs here), never OPENAI_API_KEY; its value goes into the request's
Authorization header and nowhere else, and is taken out of anything written. Before the first round the key
is checked against the API's model list (free).

The budget is the run's as under Claude Code (budget.py: spend.jsonl, a cap per try from what is left of the
round's and of the run's), with one difference: there is no CLI to count the dollars, so this backend counts
them, from the tokens every answer reports at the prices in the config (openai.prices.<agent_model>). After
every answer the count is compared with the try's cap; a try at its cap ends there, after the tools its last
answer called, and what it submitted stands. So a try can go over its cap by one answer, as with the CLI.
The count is this pipeline's, not OpenAI's invoice: with a wrong price in the config it is wrong.

A rate limit, an overload or a dropped connection is waited out and the same turn asked again; the
conversation is in this process, so nothing is lost. A refused key, an account that cannot pay or a model the
key cannot use stops the run (`resume` continues it later, with a new try that is told what the session
already used). An answer whose model is not the run's stops the run. A refusal is recorded as one. The
session's wall clock is session.seconds and its turns session.max_turns: one answer is one turn.

Each try's record is stream_NN.jsonl in the round's folder: the request's fixed part, then for every turn the
items added to the input and the whole answer, as they came.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .. import budget, prompt, run_config
from ..session import ToolError
from ..util import append_jsonl, atomic_write
from . import BackendError, Response, StopRun
from .claude_code import SYSTEM_PROMPT, next_stream

RESERVED = run_config.KEY_VARIABLES + run_config.OPENAI_KEY_VARIABLES
FINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")
# What the API says when waiting does not help, by its error codes and words.
BILLING_WORDS = ("insufficient_quota", "billing_hard_limit", "exceeded your current quota", "billing")
MODEL_WORDS = ("model_not_found", "does not exist or you do not have access", "does not have access to model")
CONTEXT_WORDS = ("context_length_exceeded", "context window", "maximum context length")
MILLION = 1_000_000.0


class Transport:
    """The two HTTP calls, kept apart so that tests replace them: nothing here is reached by a test."""

    def get(self, url: str, headers: dict, timeout_s: float = 30) -> tuple[int, str]:
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                return r.status, r.read(1 << 16).decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(1 << 16).decode("utf-8", errors="replace")

    def post(self, url: str, headers: dict, body: dict, idle_s: float, deadline: float) -> tuple[int, list, str, dict]:
        """(status, the stream's events, the body of an error, the response's headers). The answer is read as a
        server-sent event stream, so a long answer keeps the connection alive; idle_s is how long it may send
        nothing, and deadline (time.monotonic) when the session's clock runs out (TimeoutError)."""
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={**headers, "content-type": "application/json", "accept": "text/event-stream"})
        try:
            with urllib.request.urlopen(req, timeout=idle_s) as r:
                events = []
                for raw in r:
                    if time.monotonic() > deadline:
                        raise TimeoutError("the session's time ran out while the model was answering")
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        ev = json.loads(payload)
                    except ValueError:
                        continue
                    if isinstance(ev, dict):
                        events.append(ev)
                return r.status, events, "", dict(r.headers)
        except urllib.error.HTTPError as exc:
            return exc.code, [], exc.read(1 << 16).decode("utf-8", errors="replace"), dict(exc.headers or {})


def price_of(usage: dict, prices: dict) -> float:
    """Dollars for one answer's usage. input_tokens counts every input token: those read from the cache, those
    written to it and the rest; output_tokens includes the reasoning."""
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
    total = max(0, int(usage.get("input_tokens") or 0))
    cached = min(total, max(0, int(details.get("cached_tokens") or 0)))
    written = min(total - cached, max(0, int(details.get("cache_write_tokens") or 0)))
    out = max(0, int(usage.get("output_tokens") or 0))
    long = prices.get("long_context")
    p = long if isinstance(long, dict) and total > int(long["above_input_tokens"]) else prices
    p_in = float(p["input"])
    return ((total - cached - written) * p_in + cached * float(p.get("cached_input", p_in))
            + written * float(p.get("cache_write", p_in)) + out * float(p["output"])) / MILLION


def tool_specs(tools: dict) -> list[dict]:
    return [{"type": "function", "name": name, "description": spec["description"], "parameters": spec["schema"], "strict": False}
            for name, spec in tools.items()]


def error_of(status: int, body: str) -> tuple[str, str]:
    """(code, message) of an error answer, from its JSON when it has one."""
    try:
        err = json.loads(body).get("error") or {}
        return str(err.get("code") or err.get("type") or ""), str(err.get("message") or body)[:400]
    except (ValueError, AttributeError):
        return "", " ".join(str(body).split())[:400] or f"HTTP {status}"


class OpenAIBackend:
    name = "openai_api"

    def __init__(self, cfg: dict, log=print, transport=None, sleep=time.sleep, environ=None, check_key: bool = True):
        self.cfg = cfg
        self.log = log
        self.sleep = sleep
        self.retries = cfg["api_retries"]
        self.auth = cfg.get("auth") or "login"
        if self.auth != "api_key":
            raise BackendError("the openai_api backend bills an API key: set auth: api_key and api_key_env")
        self.key_var = str(cfg.get("api_key_env") or "")
        if not self.key_var.isidentifier() or self.key_var in RESERVED:
            raise BackendError(f"api_key_env must name an environment variable other than {' and '.join(RESERVED)}, not {self.key_var!r}")
        if "ANTHROPIC" in self.key_var.upper():
            raise BackendError(f"api_key_env is {self.key_var}, which reads like an Anthropic key's variable: an openai_api run "
                               "sends its key to OpenAI, so name the variable that holds the OpenAI key (CPL_OPENAI_KEY)")
        env = os.environ if environ is None else environ
        self.key = env.get(self.key_var) or ""
        if not self.key:
            raise BackendError(f"auth is api_key, but {self.key_var} is not set: put your OpenAI API key in it first")
        if not self.key.isprintable() or any(c.isspace() for c in self.key):
            raise BackendError(f"the value of {self.key_var} has spaces or line breaks, which an API key does not have "
                               "(a line ending from the file it came from?)")
        errors = budget.check(cfg.get("budget"))
        if errors:
            raise BackendError("; ".join(errors))
        try:
            run_config.check_openai(cfg)
            self.prices = run_config.openai_prices(cfg)
        except run_config.ConfigError as exc:
            raise BackendError(str(exc)) from None
        self.base = cfg["openai"]["base_url"].rstrip("/")
        self.idle = float(cfg["openai"]["idle_seconds"])
        self.extra = dict(cfg["openai"].get("request") or {})
        self.transport = transport or Transport()
        if check_key and (transport is None or getattr(transport, "check_key", False)):
            self._check_key()

    # the key ---------------------------------------------------------------------------------
    def headers(self) -> dict:
        return {"authorization": f"Bearer {self.key}"}

    def scrub(self, text) -> str:
        """The text with the key's value taken out, should anything ever echo it."""
        return str(text).replace(self.key, "<the API key>") if self.key else str(text)

    def _check_key(self) -> None:
        """Ask the API whether it takes the key and lets it use the run's model (GET /models/<model>, free)."""
        model = self.cfg["agent_model"]
        try:
            status, body = self.transport.get(f"{self.base}/models/{urllib.parse.quote(model, safe='')}", self.headers())
        except (OSError, ValueError) as exc:  # no network, no certificates: the first round will tell
            self.log(f"WARNING: could not check the API key before the run ({type(exc).__name__}: {self.scrub(exc)[:200]}); going on")
            return
        said = self.scrub(" ".join(str(body).split())[:300])
        if status in (401, 403):
            raise BackendError(f"OpenAI's API refused the key in {self.key_var} (HTTP {status}): {said}")
        if status == 404:
            raise BackendError(f"the key in {self.key_var} cannot use the model {model} (HTTP 404): {said}")
        if status != 200:
            self.log(f"WARNING: checking the API key before the run gave HTTP {status}: {said}; going on")

    # the loop's questions ----------------------------------------------------------------------
    def available(self, index: int) -> bool:
        return True

    def run_path(self, session) -> Path:
        return Path(session.state["run_path"])

    def exhausted(self, run_path: Path) -> str | None:
        """Why no round can start, when the run's budget is spent; None otherwise."""
        st = budget.state(run_path, self.cfg["budget"])
        return budget.spent_why(st) if budget.left(st) < budget.MIN_CAP_USD else None

    def _wait(self, attempt: int, headers: dict | None = None) -> float:
        """How long to wait before asking again: what the API asks for when it says, else a growing pause."""
        asked = None
        for name in ("retry-after", "Retry-After"):
            try:
                asked = float((headers or {}).get(name))
            except (TypeError, ValueError):
                continue
        wait = min(self.retries["max_wait_s"], self.retries["first_wait_s"] * 2**attempt)
        if asked is not None and asked >= 0:
            wait = min(self.retries["max_wait_s"], max(asked, 1.0))
        return wait * (1 + 0.1 * random.random())

    def _classify(self, status: int, code: str, message: str) -> tuple[str, str]:
        """(what to do, description) for an error: retry, stop the run, or end the session at the context's limit."""
        low = f"{code} {message}".lower()
        said = f"{message[:300]}" + (f" ({code})" if code else "")
        if any(w in low for w in BILLING_WORDS):
            return "stop", f"the API account cannot pay for more: {said}"
        if status in (401, 403) or "invalid_api_key" in low or "incorrect api key" in low:
            return "stop", f"the API key in {self.key_var} was refused: {said}"
        if status == 404 or any(w in low for w in MODEL_WORDS):
            return "stop", f"the model {self.cfg['agent_model']} cannot be used here: {said}"
        if any(w in low for w in CONTEXT_WORDS):
            return "context", f"the conversation no longer fits the model's context: {said}"
        if status in (408, 409, 429) or status >= 500 or status == 0:
            return "retry", f"rate limit or overload: {said}"
        if 400 <= status < 500:
            return "stop", f"the API did not take the request (HTTP {status}): {said}"
        return "retry", f"API error: {said}"

    def request_body(self, items: list, tools: list) -> dict:
        body = {"model": self.cfg["agent_model"], "input": items, "store": False, "stream": True,
                "include": ["reasoning.encrypted_content"]}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        if self.cfg.get("effort") is not None:
            body["reasoning"] = {"effort": self.cfg["effort"]}
        return {**body, **self.extra}

    def _call_tool(self, session, name, arguments) -> str:
        """The tool's result as the text the model reads; a refused call says why, as the tool server does."""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except ValueError:
            return "refused: the arguments are not valid JSON"
        try:
            return json.dumps(session.call(name, args), indent=1)
        except ToolError as exc:
            return f"refused: {exc}"
        except Exception as exc:  # a tool that failed for the pipeline's reasons: say so, keep going
            self.log(f"  tool {name} failed: {exc!r}")
            return f"the tool failed: {type(exc).__name__}"

    # one try ---------------------------------------------------------------------------------
    @staticmethod
    def _blank() -> dict:
        return {"kind": "interrupted", "messages": [], "cost": 0.0, "turns": 0, "failures": [], "declined": None, "why": None,
                "usage": {"input": 0, "cached": 0, "written": 0, "output": 0, "reasoning": 0}, "source": None, "models": {}}

    def _converse(self, out: dict, request_text: str, session, cap: float, stream: Path, timeout: float, cache_key: str) -> dict:
        """One conversation, until the model stops calling tools or a limit ends it. What happened is written into
        `out` as it happens (so a try that is interrupted still knows what it cost); the record is appended to
        `stream` turn by turn."""
        model = self.cfg["agent_model"]
        tools = tool_specs(session.TOOLS)
        items = [{"role": "developer", "content": SYSTEM_PROMPT}, {"role": "user", "content": request_text}]
        fixed = self.request_body([], tools)
        fixed.pop("input")
        fixed["prompt_cache_key"] = cache_key
        append_jsonl(stream, {"event": "request", "url": f"{self.base}/responses", "fixed": fixed, "input": items})
        max_turns = int(self.cfg["session"]["max_turns"])
        tries = int(self.retries["tries"])
        deadline = time.monotonic() + timeout
        failed = 0
        while True:
            if out["turns"] >= max_turns:
                out["kind"] = "turn_limit"
                break
            if time.monotonic() >= deadline:
                out["kind"] = "timeout"
                break
            body = {**fixed, "input": items}
            try:
                status, events, text, headers = self.transport.post(f"{self.base}/responses", self.headers(), body, self.idle, deadline)
            except TimeoutError:
                out["kind"] = "timeout"
                break
            except (OSError, ValueError) as exc:  # the connection dropped or never opened
                status, events, text, headers = 0, [], f"{type(exc).__name__}: {exc}", {}
            final = next((e for e in reversed(events) if e.get("type") in FINAL_EVENTS and isinstance(e.get("response"), dict)), None)
            resp = final["response"] if final else None
            problem = None
            if status != 200:
                problem = self._classify(status, *error_of(status, self.scrub(text)))
            elif resp is None:
                err = next((e for e in events if e.get("type") == "error"), None)
                problem = (self._classify(0, str((err or {}).get("code") or ""), str((err or {}).get("message") or "the answer's stream ended early"))
                           if err else ("retry", "the answer's stream ended before it was complete"))
            elif resp.get("status") == "failed":
                err = resp.get("error") if isinstance(resp.get("error"), dict) else {}
                problem = self._classify(0, str(err.get("code") or ""), str(err.get("message") or "the response failed"))
            if resp is not None:  # an answer that came is paid for, whatever it says
                usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
                details = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
                out["cost"] += price_of(usage, self.prices)
                out["usage"]["input"] += int(usage.get("input_tokens") or 0)
                out["usage"]["cached"] += int(details.get("cached_tokens") or 0)
                out["usage"]["written"] += int(details.get("cache_write_tokens") or 0)
                out["usage"]["output"] += int(usage.get("output_tokens") or 0)
                out["usage"]["reasoning"] += int((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
            if problem is not None:
                what, said = problem
                append_jsonl(stream, {"event": "error", "turn": out["turns"] + 1, "status": status, "what": said,
                                      "response": resp})
                out["failures"].append(said)
                if what == "context":  # what the session submitted stands
                    out.update(kind="context", why=said)
                    break
                failed += 1
                if what == "stop":
                    out.update(kind="stop", why=f"OpenAI's API stopped the run: {said}")
                    break
                if failed >= tries:
                    out.update(kind="stop", why=f"OpenAI's API kept failing after {tries} tries; last: {said}. "
                                                "Resume later with `python -m level1.loop resume`.")
                    break
                wait = self._wait(failed - 1, headers)
                self.log(f"  OpenAI API: {said}; asking again in {wait:.0f} s")
                self.sleep(wait)
                deadline += wait  # the wait is not the session's time
                continue
            failed = 0
            out["turns"] += 1
            out["source"] = resp.get("id") or out["source"]
            answered = str(resp.get("model") or "")
            out["models"][answered or model] = out["models"].get(answered or model, 0) + int((resp.get("usage") or {}).get("output_tokens") or 0)
            if answered and not (answered.startswith(model) or model.startswith(answered)):
                append_jsonl(stream, {"event": "response", "turn": out["turns"], "response": resp})
                out.update(kind="stop", why=f"OpenAI's API answered with {answered} instead of {model}; the run stops rather than "
                                            f"record another model's work as {model}'s")
                break
            output = [it for it in (resp.get("output") or []) if isinstance(it, dict)]
            calls, texts, refusal = [], [], None
            for it in output:
                if it.get("type") == "function_call":
                    calls.append(it)
                elif it.get("type") == "message":
                    for part in it.get("content") or []:
                        if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                            texts.append(part["text"])
                        elif isinstance(part, dict) and part.get("type") == "refusal":
                            refusal = str(part.get("refusal") or "")
            incomplete = (resp.get("incomplete_details") or {}).get("reason") if resp.get("status") == "incomplete" else None
            if incomplete == "content_filter" and refusal is None:
                refusal = "content_filter"
            if texts:
                out["messages"].append({"text": "".join(texts), "stop_reason": incomplete or ("tool_use" if calls else "end_turn")})
            items = items + output
            added = []
            for call in calls:
                added.append({"type": "function_call_output", "call_id": call.get("call_id"),
                              "output": self._call_tool(session, call.get("name"), call.get("arguments"))})
            items = items + added
            append_jsonl(stream, {"event": "response", "turn": out["turns"], "response": resp, "tool_outputs": added,
                                  "cost_usd": round(out["cost"], 6)})
            if refusal is not None:  # what the session submitted before it stands; the loop counts refusals
                out.update(kind="refusal", declined=refusal)
                break
            if out["cost"] >= cap:
                out["kind"] = "budget"
                break
            if not calls:
                if incomplete:
                    out["why"] = f"the last answer was cut off ({incomplete})"
                out["kind"] = "done"
                break
        return out

    def respond(self, prompt_text: str, index: int, where, session) -> Response:
        where = Path(where)
        where.mkdir(parents=True, exist_ok=True)
        k, run_path = index + 1, self.run_path(session)
        timeout = float(self.cfg["session"]["seconds"])
        model = self.cfg["agent_model"]
        t0 = time.monotonic()
        session.reload()  # an earlier try (before a stop and a resume) may have used some of the session
        note = prompt.restart_note(session)
        restarts = [dict(session.state["used"])] if note else []
        n = next_stream(where)
        cap = budget.cap(budget.state(run_path, self.cfg["budget"]), k)
        ended, caps, request_prompts = None, [], []
        if cap <= 0:  # what the session submitted stands; the loop ends the rounds if the run's is spent
            ended = budget.spent_why(budget.state(run_path, self.cfg["budget"]), k)
            self.log(f"  {ended}; the round ends with what it submitted")
            got = {**self._blank(), "kind": "budget"}
        else:
            budget.start(run_path, k, n, cap)
            caps.append(cap)
            request_text = prompt_text + note
            request_file = f"request_{n:02d}.md"
            atomic_write(where / request_file, self.scrub(request_text))
            request_prompts.append(request_file)
            stream = where / f"stream_{n:02d}.jsonl"
            atomic_write(stream, "")
            cache_key = hashlib.sha256(f"{session.path}/{n}".encode()).hexdigest()[:32]
            got = self._blank()
            try:
                self._converse(got, request_text, session, cap, stream, timeout, cache_key)
            finally:  # whatever ended the try, even a stop from the keyboard, what it cost so far is counted
                budget.end(run_path, k, n, got["cost"], got["kind"])
            if got["kind"] == "stop":
                raise StopRun(self.scrub(got["why"]))
            if got["kind"] == "budget":
                ended = f"the session reached its spending cap (${cap:.2f})"
                self.log(f"  {ended}; the round ends with what it submitted")

        messages = got["messages"]
        text = "\n".join(m["text"].rstrip("\n") for m in messages)
        if got["kind"] == "timeout":
            self.log(f"  the session reached its time limit ({timeout:g} s); what it submitted stands")
            text = ("(the session ran out of time; partial response follows)\n\n" + text) if text else "(the session ran out of time)"
            stop_reason, stop_details = "timeout", {"timeout_s": timeout}
        elif got["kind"] == "refusal":
            self.log(f"  OpenAI API: the model declined to go on ({got['declined'] or 'no reason given'})")
            stop_reason, stop_details = "refusal", {"category": got["declined"]}
        elif ended:
            text = text or f"({ended})"
            stop_reason, stop_details = "budget", {"budget": ended}
        elif got["kind"] == "turn_limit":
            text = text or "(the session reached its turn limit)"
            stop_reason, stop_details = "turn_limit", {"max_turns": int(self.cfg["session"]["max_turns"])}
        elif got["kind"] == "context":
            self.log(f"  OpenAI API: {got['why']}; the round ends with what it submitted")
            text = text or "(the conversation outgrew the model's context)"
            stop_reason, stop_details = "context_limit", {"why": got["why"]}
        else:
            stop_reason, stop_details = "end_turn", ({"note": got["why"]} if got.get("why") else None)
        u = got["usage"]
        atomic_write(where / "response_raw.json", json.dumps({"assistant_messages": messages, "usage": u, "turns": got["turns"],
                                                               "cost_usd": round(got["cost"], 6), "source": got["source"]}))
        meta = {
            "backend": self.name,
            "source": got["source"],
            "model": model,
            "stop_reason": stop_reason,
            "stop_details": stop_details,
            "timed_out": got["kind"] == "timeout",
            "models_used": got["models"],
            "model_events": [],
            "input_tokens": u["input"] - u["cached"] - u["written"],
            "output_tokens": u["output"],
            "reasoning_tokens": u["reasoning"],
            "cache_creation_input_tokens": u["written"],
            "cache_read_input_tokens": u["cached"],
            "auth": self.auth,
            "billed_to": f"the key in {self.key_var}",
            "cost_usd": round(got["cost"], 6),  # this pipeline's count: the reported tokens at the config's prices
            "api_equivalent_usd": round(got["cost"], 6),
            "prices_usd_per_million": self.prices,
            "num_turns": got["turns"],
            "assistant_messages": len(messages),
            "tools_used": dict(session.state["used"]),
            "submissions": len(session.receipts()),
            "request": {"url": f"{self.base}/responses", "prompt_files": request_prompts},
            "failed_tries": got["failures"],
            "restarted_with_used": restarts,
            "seconds": round(time.monotonic() - t0, 1),
        }
        st = budget.state(run_path, self.cfg["budget"])
        meta["budget"] = {"caps_usd": caps, "round_counted_usd": round(st["rounds"].get(k, 0.0), 6),
                          "run_counted_usd": round(st["spent"], 6), "round_usd": st["round_usd"], "run_usd": st["run_usd"],
                          "unreported_tries": st["unreported"], "stopped_tries": st["stopped"]}
        self.log(f"  API: round {k} ${st['rounds'].get(k, 0.0):.2f}; the run ${st['spent']:.2f} of ${st['run_usd']:.2f}"
                 + (f" ({st['unreported']} tries counted at their caps, {st['stopped']} of them stopped with the loop)" if st["unreported"] else ""))
        return Response(text=self.scrub(text), meta=meta)
