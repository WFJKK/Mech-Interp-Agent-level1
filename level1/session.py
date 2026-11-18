"""One agent session: its two tools, their budgets and their log.

A session lives in a round's folder:

    round_k/session/
        session.json          what the tools need (model, network, limits, reward) and what was used so far
        tools.jsonl           every tool call, appended as it happens: tool, arguments, outcome, seconds
        python/<n>.py, .txt   every program run_python was given, in full, and what went back to the agent
        python/<n>.audit.json what the guard inside Python refused that program, when it refused anything or
                              its record is incomplete
        workspace/entries/    the entries saved from run_python (and the ones carried from earlier rounds)
        submissions/<n>/      one folder per submit_entry call: entry.json, receipt.json

The tools:

- run_python(code): run code in the sandbox with the network's weights, the entry language and the model's
  definition in scope; returns what it printed. save_entry(name, entry) in that code keeps an entry in the
  session's workspace. Counts against the python budget.
- submit_entry(name, claimed_fidelity, expected_fidelity_other): hand a saved entry to the verifier, which
  measures it on its own inputs and returns the receipt. Counts against the submission budget.

The verifier is the judge; this file is its client. Budgets persist in session.json, so a session that is
started again (a resumed run) continues them, and its earlier submissions stay.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import replace
from pathlib import Path

from . import lang
from .networks import load_model
from .util import append_jsonl, atomic_write, now, read_json, write_json

MAX_TEXT = 20000  # characters of program output returned to the agent
MAX_SOURCE = 200_000  # characters of one program
ENTRY_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class ToolError(ValueError):
    """A tool refused the call: the message goes back to the agent."""


WRAPPER = '''"""run_python: the agent's code, with the network and the entry language in scope."""
import contextlib, io, json, traceback, warnings
import numpy as np

# On some machines numpy's matrix product raises floating-point flags when nothing went wrong, and prints
# "divide by zero / overflow / invalid value encountered in matmul" though the result is exact. Elsewhere
# the same code prints nothing. Those three messages are not shown, so the output is the same on every machine;
# a real overflow still shows in the values (inf, nan).
warnings.filterwarnings("ignore", message=r"(divide by zero|overflow|invalid value) encountered in matmul",
                        category=RuntimeWarning)

import lv1
import model
import rewards
import regions

CODE = @@CODE@@


def make_result(weights, info, extra):
    n_ctx = info["sizes"]["n_ctx"]
    entries = dict(extra["entries"])
    saved = {}
    cache = {}

    def graph(example=None):
        return lv1.Graph(weights, n_ctx, example, real=lv1.real_inputs(info["data_set"]))

    def reference_entry():
        g = graph()
        return g.entry(predict=model.reference(g, lv1))

    def evaluation_domain(scope):
        cfg = dict(extra["verifier"])
        if scope is not None:
            if not extra.get("allow_claim_scopes"):
                raise lv1.EntryError("narrower claim domains are available only in research sessions")
            cfg["scope"] = regions.intersect_scopes(cfg.get("scope"), scope)
        return cfg, json.dumps(cfg.get("scope"), sort_keys=True)

    def network_outputs(scope=None):
        """(inputs, the network's outputs on them, operations of one forward pass). The inputs are every
        input of the data set when info["check"] is "all"; otherwise a sample of info["sample_size"] inputs
        drawn from it. An optional scope is intersected with the assigned domain in research sessions.
        Local calls repeat the sample; submission sampling follows the configured policy."""
        cfg, key = evaluation_domain(scope)
        if key not in cache:
            sample_seed = cfg.get("sample_seed")
            x, how, scope_meta = regions.prepare_inputs(info, weights, cfg,
                                                       seed=0 if sample_seed is None else sample_seed,
                                                       limits=extra["limits"])
            answers, ops = lv1.run_reference(reference_entry(), weights, n_ctx, x)
            cache[key] = {"out": (x, answers, ops), "how": how, "scope": scope_meta}
        if scope is None:
            info["check"] = cache[key]["how"]["mode"]
            if cache[key]["scope"] is not None:
                info.setdefault("n_total_original", info["n_total"])
                info["n_total"] = cache[key]["scope"]["cardinality"]
                info["scope"] = cache[key]["scope"]
        return cache[key]["out"]

    def measure(entry, scope=None):
        """What the verifier measures of an entry, with the reward: the same code on the same kind of inputs."""
        lv1.check_structure(entry, extra["limits"]["max_nodes"], extra["limits"]["max_const_scalars"])
        cfg, key = evaluation_domain(scope)
        x, answers, ops = network_outputs(scope)
        parts = regions.measure(entry, weights, n_ctx, x, answers, ops, how=cache[key]["how"],
                                scope=cache[key]["scope"], verifier_cfg=cfg, limits=extra["limits"])
        parts["reward"] = rewards.compute(parts, extra["reward"])["reward"]
        if cache[key]["scope"] is not None:
            parts["scope"] = cache[key]["scope"]
        return parts

    def save_entry(name, entry):
        """Keep an entry in the session's workspace under a name, to submit it or to build on it later."""
        size = lv1.check_structure(entry, extra["limits"]["max_nodes"], extra["limits"]["max_const_scalars"])
        saved[str(name)] = entry
        entries[str(name)] = entry
        return size

    scope = {"weights": weights, "info": info, "np": np, "lv1": lv1, "model": model, "graph": graph,
             "entries": entries, "save_entry": save_entry, "measure": measure, "network_outputs": network_outputs,
             "reference_entry": reference_entry, "__name__": "__main__"}
    buf = io.StringIO()
    error = None
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(compile(CODE, "<run_python>", "exec"), scope)
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            buf.write(traceback.format_exc())
    return {"stdout": buf.getvalue()[-20000:], "entries": saved, "error": error}
'''

TOOLS = {
    "run_python": {
        "description": (
            "Run Python code in a sandbox and get back what it printed. In scope: `weights` (a dict of numpy arrays), "
            "`info` (sizes, input_space, the data set and how it is checked), `np`, `lv1` (the entry language), "
            "`model` (the network's definition, model.reference(g, lv1)), `graph()` (a new lv1.Graph on these weights), "
            "`network_outputs()` (inputs of the data set, every one of them or a fixed sample as `info['check']` says, "
            "the network's outputs on them, and the operations of a forward pass), "
            "`measure(entry)` (what the verifier measures: coverage, fidelity, cost and the reward), `entries` (the "
            "saved entries by name) and `save_entry(name, entry)`. The code may read its own folder and Python's "
            "libraries and write only to tmp/ in its folder; other files, the network and other programs are closed "
            "to it (PermissionError). Nothing persists between calls except saved entries. Each call counts against "
            "the python budget. Research sessions also accept `measure(entry, scope=...)` and "
            "`network_outputs(scope=...)` to inspect a narrower domain intersected with the assignment. "
            "Local measurements are not formal submissions. Replies include remaining tool quotas and, "
            "when an attempt deadline exists, wall_seconds_remaining."),
        "schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    },
    "submit_entry": {
        "description": (
            "Hand a saved entry to the verifier. It measures the entry on its own inputs and returns the receipt: "
            "status scored or refused (with the reason), coverage, fidelity, cost (with its three parts) and the "
            "reward. claimed_fidelity is the share of the entry's region on which you say its interpretation gives "
            "the network's output (0 to 1); expected_fidelity_other is the same for the second network, which you "
            "do not see. Each call counts against the submission budget."),
        "schema": {"type": "object", "properties": {
            "name": {"type": "string", "description": "the name the entry was saved under"},
            "claimed_fidelity": {"type": "number"},
            "expected_fidelity_other": {"type": "number"}},
            "required": ["name", "claimed_fidelity"]},
    },
}


RESEARCH_SUBMIT_TOOL = {
    "description": TOOLS["submit_entry"]["description"] + (
        " In a research session, optional scope declares a separately scored domain inside the assigned "
        "domain. The evaluator intersects them, so the claim cannot broaden the assignment. Use kind all, "
        "lengths, predicate or intersection. For a predicate, scope_entry_name can name a saved entry whose "
        "region test defines it. This consumes one ordinary submission; it does not change reward, prices "
        "or workload policy. Without scope, evaluation uses the assigned domain."),
    "schema": {**TOOLS["submit_entry"]["schema"], "properties": {
        **TOOLS["submit_entry"]["schema"]["properties"],
        "scope": {"type": "object", "description": "Optional narrower claim domain, intersected with the assignment."},
        "scope_entry_name": {"type": "string", "description": "Saved entry defining a predicate claim domain."},
    }},
}


RESEARCH_TOOLS = {
    "assign_region": {
        "description": (
            "Assign a bounded investigation to a regional researcher. The scope is the fixed input domain of "
            "the task, and the researcher optimizes the same reward conditioned on it. Use kind all, lengths "
            "with a list of permitted lengths, or predicate with a saved entry whose region test defines the "
            "scope. For predicate, pass that saved entry's name as entry_name. The task describes a question, "
            "evidence to examine and expected output. This schedules work for the next research phase; it "
            "does not run another researcher immediately. Assignments cannot alter reward parameters, prices "
            "or workload policy. Their scopes and results are retained separately from the global task."),
        "schema": {"type": "object", "properties": {
            "scope": {"type": "object", "description": "An input domain: kind all, lengths or predicate."},
            "task": {"type": "string", "description": "A focused research question and the evidence to return."},
            "entry_name": {"type": "string", "description": "For a predicate scope, the saved entry defining its region."}},
            "required": ["scope", "task"]},
    },
}


def agent_info(model, verifier_cfg: dict) -> dict:
    """What the agent's code is handed about the network, its data set and how the verifier checks it. No
    name in it says more than the model folder's own agent-facing files."""
    from .verifier import DEFAULTS, resolve_check

    cfg = {**DEFAULTS, **(verifier_cfg or {})}
    check = resolve_check(model, cfg) if cfg.get("scope") is None else cfg["check"]
    out = {**model.info(), "n_total": model.n_total, "check": check, "sample_size": int(cfg["sample_size"])}
    if cfg.get("scope") is not None:
        out["scope"] = cfg["scope"]
        out["workload_size"] = cfg.get("workload_size")
    return out


def box_files(model) -> dict:
    """The modules the agent's code may import, as they are placed in the sandbox folder."""
    here = Path(__file__).parent
    return {"lv1.py": (here / "lang.py").read_text(), "rewards.py": (here / "rewards.py").read_text(),
            "regions.py": (here / "regions.py").read_text(), "model.py": model.source()}


def surface(model, verifier_cfg: dict) -> dict:
    """Every text of a session the agent can reach, for the hygiene check: the tools' descriptions, the files
    of the sandbox folder, the wrapper and what its code is handed. (The prompt is checked with them.)"""
    out = {f"tool {name}": json.dumps(spec) for name, spec in TOOLS.items()}
    out.update({f"sandbox file {name}": text for name, text in box_files(model).items()})
    out["sandbox file harness.py"] = (Path(__file__).parent / "harness.py").read_text()
    out["sandbox wrapper"] = WRAPPER
    out["sandbox info"] = json.dumps(agent_info(model, verifier_cfg))
    out["weights' names"] = " ".join(model.config["weights"])
    return out


class Session:
    TOOLS = TOOLS

    def __init__(self, path: Path):
        self.path = Path(path)
        self.state_path = self.path / "session.json"
        self.state = read_json(self.state_path)
        self._configure_tools()
        self.entries_dir = self.path / "workspace" / "entries"
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self._verifier = None
        self._scoped_verifiers = {}
        self._model = None

    # creation --------------------------------------------------------------------------------
    @classmethod
    def create(cls, path: Path, cfg: dict, network_id: str, run_path: Path, carried: dict | None = None) -> "Session":
        """Make (or reopen) a round's session folder. Budgets carry over if it exists. `carried` are entries
        of earlier rounds, {name: entry}, placed in the workspace."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state_path = path / "session.json"
        research = dict(cfg.get("research_task") or {})
        summary = research.get("role") == "summary"
        if summary:
            research["max_assignments"] = 0
        if not state_path.exists():
            s = cfg["session"]
            state = {
                "model": str(cfg["model"]), "network": network_id, "run_path": str(Path(run_path).resolve()),
                "budgets": {"run_python": 0 if summary else int(s["python_calls"]),
                            "submit_entry": 0 if summary else int(s["submissions"])},
                "used": {"run_python": 0, "submit_entry": 0},
                "python": dict(cfg["python"]), "verifier": dict(cfg["verifier"]), "entry": dict(cfg["entry"]),
                "reward": dict(cfg["reward"]), "sandbox": cfg["sandbox"], "sandbox_image": cfg.get("sandbox_image"),
                "sandbox_dir": cfg.get("sandbox_dir"), "runs_dir": cfg.get("runs_dir"), "created": now(),
            }
            if research:
                state["research_task"] = research
                state["assignments"] = []
            write_json(state_path, state)
            session = cls(path)
            for name, entry in (carried or {}).items():
                write_json(session.entries_dir / f"{name}.json", entry)
            return session
        session = cls(path)
        if summary != ((session.state.get("research_task") or {}).get("role") == "summary"):
            raise ToolError("cannot change a persisted session's summary role")
        return session

    def reload(self) -> None:
        """Re-read the saved state: the tools run in the tool server's process, which saves its counters here."""
        self.state = read_json(self.state_path)
        self._configure_tools()

    # shared ----------------------------------------------------------------------------------
    def _configure_tools(self) -> None:
        research = self.state.get("research_task") or {}
        self.TOOLS = {} if research.get("role") == "summary" else dict(TOOLS)
        if research.get("role") in ("lead", "regional"):
            self.TOOLS["submit_entry"] = RESEARCH_SUBMIT_TOOL
        if research.get("role") == "lead" and int(research.get("max_assignments", 0)) > 0:
            self.TOOLS.update(RESEARCH_TOOLS)

    def _require_research(self) -> None:
        if (self.state.get("research_task") or {}).get("role") == "summary":
            raise ToolError("the final summary has no research tools")

    def _log(self, tool: str, args: dict, outcome: dict, seconds: float) -> None:
        short = {k: (v if not isinstance(v, str) or len(v) < 4000 else v[:4000] + f"... ({len(v)} chars)") for k, v in args.items()}
        append_jsonl(self.path / "tools.jsonl", {"at": now(), "tool": tool, "args": short, "outcome": outcome, "seconds": round(seconds, 2)})

    def _spend(self, tool: str) -> None:
        self._require_attempt_open()
        used, budget = self.state["used"][tool], self.state["budgets"][tool]
        if used >= budget:
            raise ToolError(f"the {tool} budget of this session ({budget} calls) is used up")
        if tool == "run_python":
            checkpoint = self._checkpoint()
            if (checkpoint is not None and checkpoint["status"] != "scored"
                    and checkpoint["bootstrap_repair_calls"]
                    and used >= checkpoint["initial_python_calls"] + checkpoint["repair_calls"]):
                self.state["checkpoint_bootstrap_used"] = True
        self.state["used"][tool] = used + 1
        write_json(self.state_path, self.state)

    def _require_attempt_open(self) -> None:
        if self.state.get("attempt_deadline") is not None and time.time() >= self.state["attempt_deadline"]:
            raise ToolError("the current attempt has reached its wall-clock deadline")

    def _attempt_deadline(self) -> float | None:
        """Convert the cross-process wall deadline to this process's monotonic clock."""
        deadline = self.state.get("attempt_deadline")
        return None if deadline is None else time.monotonic() + max(0.0, deadline - time.time())

    def begin_attempt(self, timeout_s: float) -> None:
        """Publish the current CLI attempt's deadline without refreshing any tool quota."""
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("an attempt needs a positive finite timeout")
        self.reload()
        started = time.time()
        self.state.update(attempt_started_at=started, attempt_deadline=started + float(timeout_s))
        write_json(self.state_path, self.state)

    def remaining(self) -> dict:
        out = {t: self.state["budgets"][t] - self.state["used"][t] for t in self.state["budgets"]}
        if self.state.get("attempt_deadline") is not None:
            out["wall_seconds_remaining"] = max(0.0, self.state["attempt_deadline"] - time.time())
        checkpoint = self._checkpoint()
        if checkpoint is not None:
            out["checkpoint"] = checkpoint
        return out

    def _checkpoint(self) -> dict | None:
        """A formal receipt, rather than a mutable draft or local score, satisfies the session's gate."""
        from .regions import scope_id

        research = self.state.get("research_task") or {}
        allowance = research.get("checkpoint_python_calls")
        baseline = research.get("role") == "baseline"
        integration = research.get("role") == "lead" and research.get("cycle", 0) > 0
        if not (baseline or integration) or allowance is None:
            return None  # persisted sessions without this policy retain their original behavior
        assigned = self.state["verifier"].get("scope")
        sid = scope_id(assigned)
        initial = min(int(allowance), self.state["budgets"]["run_python"])
        receipts = [r for r in self.receipts() if scope_id(r.get("claim_scope", assigned)) == sid]

        def qualifies(receipt):
            if receipt.get("status") != "scored" or receipt.get("reward") is None:
                return False
            if not baseline:
                return True
            parts = receipt.get("parts") or {}
            count = parts.get("n_inputs", 0)
            return (count > 0 and parts.get("coverage") == 1.0
                    and parts.get("n_in_region") == count and parts.get("n_agree") == count)

        scored = next((r for r in receipts if qualifies(r)), None)
        repaired = sum(r.get("status") == "refused" or
                       (baseline and r.get("status") == "scored" and not qualifies(r)) for r in receipts)
        bootstrap = int(bool(self.state.get("checkpoint_bootstrap_used"))
                        or (self.state["used"]["run_python"] >= initial and not self.entries()))
        remaining_submissions = self.state["budgets"]["submit_entry"] - self.state["used"]["submit_entry"]
        left = max(0, min(initial + repaired + bootstrap - self.state["used"]["run_python"],
                          self.state["budgets"]["run_python"] - self.state["used"]["run_python"]))
        if remaining_submissions <= 0:
            left = 0
        out = {"status": "scored" if scored is not None else "pending" if left else "blocked",
               "scope_id": sid, "initial_python_calls": initial, "repair_calls": repaired,
               "bootstrap_repair_calls": bootstrap,
               "python_calls_until_checkpoint": left}
        if scored is not None:
            out["submission"] = scored["submission"]
        return out

    def _require_checkpoint(self) -> None:
        checkpoint = self._checkpoint()
        if checkpoint is not None and checkpoint["status"] == "blocked":
            raise ToolError("submit an evaluated checkpoint on the assigned domain before further run_python calls; "
                            "saved drafts, local measurements and narrower claims do not satisfy this checkpoint")

    def model(self):
        if self._model is None:
            self._model = load_model(self.state["model"])
        return self._model

    def verifier(self, scope: dict | None = None):
        if scope is not None and scope != self.state["verifier"].get("scope"):
            from .verifier import Verifier

            key = json.dumps(scope, sort_keys=True)
            if key not in self._scoped_verifiers:
                cfg = {"verifier": {**self.state["verifier"], "scope": scope},
                       "entry": self.state["entry"], "reward": self.state["reward"]}
                self._scoped_verifiers[key] = Verifier(self.model(), self.state["network"], cfg, deadline=self._attempt_deadline())
            return self._scoped_verifiers[key]
        if self._verifier is None:
            from .verifier import Verifier

            cfg = {"verifier": self.state["verifier"], "entry": self.state["entry"], "reward": self.state["reward"]}
            self._verifier = Verifier(self.model(), self.state["network"], cfg, deadline=self._attempt_deadline())
        return self._verifier

    def entries(self) -> dict:
        out = {}
        for p in sorted(self.entries_dir.glob("*.json")):
            if ENTRY_NAME.match(p.stem) and p.is_file() and not p.is_symlink():
                try:
                    out[p.stem] = read_json(p)
                except (OSError, ValueError):
                    pass
        return out

    def info(self) -> dict:
        return agent_info(self.model(), self.state["verifier"])

    def box_files(self) -> dict:
        return box_files(self.model())

    # the tools -------------------------------------------------------------------------------
    def run_python(self, code: str) -> dict:
        t0 = time.monotonic()
        args = {"code": code}
        try:
            self._require_research()
            if not isinstance(code, str) or len(code) > MAX_SOURCE:
                raise ToolError("code must be a string of at most 200,000 characters")
            self._require_checkpoint()
            self._spend("run_python")
            n = self.state["used"]["run_python"]
            atomic_write(self.path / "python" / f"{n:03d}.py", code)  # in full: the log line below is cut short
            from .runner import Limits, run_recipe
            from .sandbox import choose, protected_paths

            m = self.model()
            py = self.state["python"]
            limits = Limits.from_config(py)
            if self.state.get("attempt_deadline") is not None:
                limits = replace(limits, time_s=min(limits.time_s, max(0.0, self.state["attempt_deadline"] - time.time())))
            cfg = {"sandbox": self.state["sandbox"], "sandbox_image": self.state["sandbox_image"], "runs_dir": self.state["runs_dir"]}
            sb = choose(cfg, protected_paths(cfg, m, Path(self.state["run_path"])))
            extra = {"entries": self.entries(), "reward": self.state["reward"], "limits": self.state["entry"],
                     "verifier": self.state["verifier"],
                     "allow_claim_scopes": (self.state.get("research_task") or {}).get("role") in ("lead", "regional")}
            ex = run_recipe(WRAPPER.replace("@@CODE@@", repr(code)), self.box_files(), m.load_weights(self.state["network"]),
                            self.info(), extra, limits, self.state["sandbox_dir"], sandbox=sb)
            saved, problems, error = [], [], None
            if ex.status == "ok" and isinstance(ex.result, dict):
                output = str(ex.result.get("stdout", ""))
                error = ex.result.get("error")
                got = ex.result.get("entries")
                for name, entry in (got.items() if isinstance(got, dict) else []):
                    try:
                        if not ENTRY_NAME.match(str(name)):
                            raise lang.EntryError("an entry's name is 1 to 40 letters, digits, - or _")
                        lang.check_structure(entry, self.state["entry"]["max_nodes"], self.state["entry"]["max_const_scalars"])
                        write_json(self.entries_dir / f"{name}.json", entry)
                        saved.append(str(name))
                    except lang.EntryError as exc:
                        problems.append(f"{name}: not saved: {exc}")
            else:
                output = f"({ex.status}) {ex.error or ''}"
            status = "error" if error or problems else ex.status
            out = {"status": status, "output": output[-MAX_TEXT:], "saved_entries": sorted(saved), "runtime_s": ex.runtime_s,
                   "sandbox": ex.mode, "remaining": self.remaining()}
            if error:
                out["error"] = error
            if problems:
                out["problems"] = problems
            atomic_write(self.path / "python" / f"{n:03d}.txt", out["output"])
            audit = ex.audit or {"present": False, "complete": False, "n_blocked": 0, "blocked": []}
            record = "complete" if audit.get("complete") else ("incomplete" if audit.get("present") else "missing")
            if audit.get("n_blocked") or record != "complete":
                write_json(self.path / "python" / f"{n:03d}.audit.json", {**audit, "record": record, "status": status})
            self._log("run_python", args, {"status": status, "output": output[:500], "saved": sorted(saved), "sandbox": ex.mode,
                                           "blocked": int(audit.get("n_blocked") or 0), "guard_record": record}, time.monotonic() - t0)
            return out
        except ToolError as exc:
            self._log("run_python", args, {"error": str(exc)}, time.monotonic() - t0)
            raise

    def _claim_scope(self, scope: dict | None, scope_entry_name: str | None) -> dict | None:
        """Resolve the claim using saved data; its effective domain cannot exceed the assignment."""
        from .regions import intersect_scopes, normalize_scope

        research = (self.state.get("research_task") or {}).get("role") in ("lead", "regional")
        if scope is None:
            if scope_entry_name is not None:
                raise ToolError("scope_entry_name needs a predicate scope")
            return normalize_scope(self.state["verifier"].get("scope")) if research else None
        if not research:
            raise ToolError("narrower claim domains are available only in research sessions")
        if not isinstance(scope, dict):
            raise ToolError("scope must be an object")
        spec = dict(scope)
        if scope_entry_name is not None:
            if spec.get("kind") != "predicate" or "entry" in spec:
                raise ToolError("scope_entry_name is only for a predicate scope without an inline entry")
            if not isinstance(scope_entry_name, str) or not ENTRY_NAME.match(scope_entry_name):
                raise ToolError("scope_entry_name must name a saved entry")
            entries = self.entries()
            if scope_entry_name not in entries:
                raise ToolError(f"no saved entry named {scope_entry_name!r}")
            spec["entry"] = entries[scope_entry_name]
        try:
            return intersect_scopes(self.state["verifier"].get("scope"), spec)
        except (ValueError, RuntimeError) as exc:
            raise ToolError(str(exc)) from None

    def submit_entry(self, name: str, claimed_fidelity: float, expected_fidelity_other: float | None = None,
                     scope: dict | None = None, scope_entry_name: str | None = None) -> dict:
        t0 = time.monotonic()
        args = {"name": name, "claimed_fidelity": claimed_fidelity, "expected_fidelity_other": expected_fidelity_other}
        if scope is not None or scope_entry_name is not None:
            args.update(scope=scope, scope_entry_name=scope_entry_name)
        try:
            self._require_research()
            if not isinstance(name, str) or not ENTRY_NAME.match(name):
                raise ToolError("name must be the name an entry was saved under")
            for label, v in (("claimed_fidelity", claimed_fidelity), ("expected_fidelity_other", expected_fidelity_other)):
                if v is None and label == "expected_fidelity_other":
                    continue
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
                    raise ToolError(f"{label} must be a number from 0 to 1")
            path = self.entries_dir / f"{name}.json"
            if not path.is_file() or path.is_symlink():
                raise ToolError(f"no saved entry named {name!r}; the saved entries are {sorted(self.entries())}")
            entry = read_json(path)
            self._require_attempt_open()
            claim_scope = self._claim_scope(scope, scope_entry_name)
            checkpoint = self._checkpoint()
            if checkpoint is not None and checkpoint["status"] != "scored":
                from .regions import scope_id

                if (scope_id(claim_scope) != checkpoint["scope_id"]
                        and self.state["budgets"]["submit_entry"] - self.state["used"]["submit_entry"] <= 1):
                    raise ToolError("the last submission is reserved for an evaluated checkpoint on the assigned domain")
            try:
                verifier = (self.verifier(claim_scope) if claim_scope is not None and claim_scope != self.state["verifier"].get("scope")
                            else self.verifier())
                if (claim_scope is not None and verifier.scope.cardinality is None
                        and self.state["verifier"].get("workload_size") is None):
                    raise ToolError("this scope has no known finite size; the run needs an explicit verifier.workload_size")
            except (ValueError, RuntimeError) as exc:
                raise ToolError(str(exc)) from None
            self._spend("submit_entry")
            n = self.state["used"]["submit_entry"]
            folder = self.path / "submissions" / f"{n:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / "entry.json", entry)
            claim = {}
            if claim_scope is not None:
                from .regions import scope_id

                claim = {"claim_scope": claim_scope, "assigned_scope_id": scope_id(self.state["verifier"].get("scope"))}
                write_json(folder / "claim.json", claim)
            deadline = self._attempt_deadline()
            receipt = (verifier.score(entry, claimed=float(claimed_fidelity), deadline=deadline) if deadline is not None
                       else verifier.score(entry, claimed=float(claimed_fidelity)))
            receipt.update(name=name, submission=n, at=now(),
                           expected_fidelity_other=None if expected_fidelity_other is None else float(expected_fidelity_other), **claim)
            write_json(folder / "receipt.json", receipt)
            self._log("submit_entry", args, {"status": receipt["status"], "reward": receipt["reward"], "reason": receipt["reason"]}, time.monotonic() - t0)
            out = {"status": receipt["status"], "reason": receipt["reason"], "reward": receipt["reward"], "parts": receipt["parts"],
                   "size": receipt["size"], "inputs": receipt["inputs"], "overclaim": receipt.get("overclaim"),
                   "remaining": self.remaining()}
            if "scope" in receipt:
                out["scope"] = receipt["scope"]
            return out
        except ToolError as exc:
            self._log("submit_entry", args, {"error": str(exc)}, time.monotonic() - t0)
            raise

    def receipts(self) -> list:
        """The receipts of this session's submissions, in order."""
        out = []
        sub = self.path / "submissions"
        if sub.is_dir():
            for d in sorted(p for p in sub.iterdir() if p.is_dir() and p.name.isdigit()):
                try:
                    out.append(read_json(d / "receipt.json"))
                except (OSError, ValueError):
                    pass  # a submission cut off before its receipt: it spent its budget and scored nothing
        return out

    def assign_region(self, scope: dict, task: str, entry_name: str | None = None) -> dict:
        """Validate and record an investigation. Its scope is frozen before any worker submissions."""
        from .regions import Scope, normalize_scope, scope_id

        t0 = time.monotonic()
        args = {"scope": scope, "task": task, "entry_name": entry_name}
        try:
            self._require_research()
            self._require_attempt_open()
            research = self.state.get("research_task") or {}
            limit = int(research.get("max_assignments", 0))
            if research.get("role") != "lead" or limit <= 0:
                raise ToolError("this session cannot assign regional investigations")
            assignments = self.state.get("assignments", [])
            if len(assignments) >= limit:
                raise ToolError(f"this session has already assigned its {limit} investigations")
            if not isinstance(task, str) or not task.strip() or len(task) > MAX_TEXT:
                raise ToolError("task must be nonempty text within the tool's output limit")
            if not isinstance(scope, dict):
                raise ToolError("scope must be an object")
            spec = dict(scope)
            if entry_name is not None:
                if spec.get("kind") != "predicate" or "entry" in spec:
                    raise ToolError("entry_name is only for a predicate scope without an inline entry")
                if not isinstance(entry_name, str) or not ENTRY_NAME.match(entry_name):
                    raise ToolError("entry_name must name a saved entry")
                entries = self.entries()
                if entry_name not in entries:
                    raise ToolError(f"no saved entry named {entry_name!r}")
                spec["entry"] = entries[entry_name]
            try:
                spec = normalize_scope(spec)
                model = self.model()
                deadline = t0 + float(self.state["verifier"]["time_limit_s"])
                if self.state.get("attempt_deadline") is not None:
                    deadline = min(deadline, self._attempt_deadline())
                resolved = Scope(model.input_space, model.info()["data_set"], model.load_weights(self.state["network"]),
                                 model.n_ctx, spec, limits=self.state["entry"],
                                 enumerate_up_to=int(self.state["verifier"]["enumerate_up_to"]),
                                 deadline=deadline)
                if resolved.cardinality is None and self.state["verifier"].get("workload_size") is None:
                    raise ToolError("this scope has no known finite size; the run needs an explicit verifier.workload_size")
            except (ValueError, RuntimeError) as exc:
                raise ToolError(str(exc)) from None
            record = {"assignment": len(assignments) + 1, "scope": spec, "scope_id": scope_id(spec),
                      "task": task.strip(), "at": now()}
            self.state["assignments"] = [*assignments, record]
            write_json(self.state_path, self.state)
            out = {"status": "assigned", "assignment": record["assignment"], "scope_id": record["scope_id"],
                   "scope": resolved.metadata(), "remaining_assignments": limit - len(assignments) - 1}
            self._log("assign_region", args, out, time.monotonic() - t0)
            return out
        except ToolError as exc:
            self._log("assign_region", args, {"error": str(exc)}, time.monotonic() - t0)
            raise

    def call(self, tool: str, arguments: dict) -> dict:
        if tool not in self.TOOLS:
            raise ToolError(f"no tool {tool!r}; the tools are {sorted(self.TOOLS)}")
        if not isinstance(arguments, dict):
            raise ToolError("arguments must be an object")
        schema = self.TOOLS[tool]["schema"]
        missing = [k for k in schema["required"] if k not in arguments]
        if missing:
            raise ToolError(f"{tool} needs {missing}")
        extra = sorted(set(arguments) - set(schema["properties"]))
        if extra:
            raise ToolError(f"{tool} does not take {extra}")
        return getattr(self, tool)(**arguments)
