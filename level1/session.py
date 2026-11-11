"""One agent session: its two tools, their budgets and their log.

A session lives in a round's folder:

    round_k/session/
        session.json          what the tools need (model, network, limits, reward) and what was used so far
        tools.jsonl           every tool call, appended as it happens: tool, arguments, outcome, seconds
        python/<n>.py, .txt   every program run_python was given, in full, and what went back to the agent
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
import re
import time
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
import contextlib, io, traceback, warnings
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

CODE = @@CODE@@


def make_result(weights, info, extra):
    n_ctx = info["sizes"]["n_ctx"]
    entries = dict(extra["entries"])
    saved = {}
    cache = {}

    def graph(example=None):
        return lv1.Graph(weights, n_ctx, example)

    def reference_entry():
        g = graph()
        return g.entry(predict=model.reference(g, lv1))

    def network_outputs():
        """(inputs, the network's outputs on them, operations of one forward pass). The inputs are every
        input of the data set when info["check"] is "all"; otherwise a sample of info["sample_size"] inputs
        drawn from it (the same sample at every call: the verifier draws a fresh one for each submission)."""
        if "out" not in cache:
            if info["check"] == "all":
                x = lv1.all_inputs(info["input_space"])
            else:
                x = lv1.draw_inputs(info["data_set"], info["input_space"], info["sample_size"], np.random.default_rng(0))
            answers, ops = lv1.run_reference(reference_entry(), weights, n_ctx, x)
            cache["out"] = (x, answers, ops)
        return cache["out"]

    def measure(entry):
        """What the verifier measures of an entry, with the reward: the same code on the same kind of inputs."""
        lv1.check_structure(entry, extra["limits"]["max_nodes"], extra["limits"]["max_const_scalars"])
        x, answers, ops = network_outputs()
        parts = lv1.measure(entry, weights, n_ctx, x, answers, ops)
        sampled = info["check"] != "all" and parts["n_in_region"]
        parts["fidelity_low"] = lv1.wilson_lower(parts["n_agree"], parts["n_in_region"]) if sampled else parts["fidelity"]
        parts["reward"] = rewards.compute(parts, extra["reward"])["reward"]
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
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(compile(CODE, "<run_python>", "exec"), scope)
        except BaseException:
            buf.write(traceback.format_exc())
    return {"stdout": buf.getvalue()[-20000:], "entries": saved}
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
            "saved entries by name) and `save_entry(name, entry)`. No network access; nothing persists between "
            "calls except saved entries. Each call counts against the python budget."),
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


def agent_info(model, verifier_cfg: dict) -> dict:
    """What the agent's code is handed about the network, its data set and how the verifier checks it. No
    name in it says more than the model folder's own agent-facing files."""
    from .verifier import DEFAULTS, resolve_check

    cfg = {**DEFAULTS, **(verifier_cfg or {})}
    return {**model.info(), "n_total": model.n_total, "check": resolve_check(model, cfg), "sample_size": int(cfg["sample_size"])}


def box_files(model) -> dict:
    """The modules the agent's code may import, as they are placed in the sandbox folder."""
    here = Path(__file__).parent
    return {"lv1.py": (here / "lang.py").read_text(), "rewards.py": (here / "rewards.py").read_text(),
            "model.py": model.source()}


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
        self.entries_dir = self.path / "workspace" / "entries"
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self._verifier = None
        self._model = None

    # creation --------------------------------------------------------------------------------
    @classmethod
    def create(cls, path: Path, cfg: dict, network_id: str, run_path: Path, carried: dict | None = None) -> "Session":
        """Make (or reopen) a round's session folder. Budgets carry over if it exists. `carried` are entries
        of earlier rounds, {name: entry}, placed in the workspace."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state_path = path / "session.json"
        if not state_path.exists():
            s = cfg["session"]
            state = {
                "model": str(cfg["model"]), "network": network_id, "run_path": str(Path(run_path).resolve()),
                "budgets": {"run_python": int(s["python_calls"]), "submit_entry": int(s["submissions"])},
                "used": {"run_python": 0, "submit_entry": 0},
                "python": dict(cfg["python"]), "verifier": dict(cfg["verifier"]), "entry": dict(cfg["entry"]),
                "reward": dict(cfg["reward"]), "sandbox": cfg["sandbox"], "sandbox_image": cfg.get("sandbox_image"),
                "sandbox_dir": cfg.get("sandbox_dir"), "runs_dir": cfg.get("runs_dir"), "created": now(),
            }
            write_json(state_path, state)
            session = cls(path)
            for name, entry in (carried or {}).items():
                write_json(session.entries_dir / f"{name}.json", entry)
            return session
        return cls(path)

    def reload(self) -> None:
        """Re-read the saved state: the tools run in the tool server's process, which saves its counters here."""
        self.state = read_json(self.state_path)

    # shared ----------------------------------------------------------------------------------
    def _log(self, tool: str, args: dict, outcome: dict, seconds: float) -> None:
        short = {k: (v if not isinstance(v, str) or len(v) < 4000 else v[:4000] + f"... ({len(v)} chars)") for k, v in args.items()}
        append_jsonl(self.path / "tools.jsonl", {"at": now(), "tool": tool, "args": short, "outcome": outcome, "seconds": round(seconds, 2)})

    def _spend(self, tool: str) -> None:
        used, budget = self.state["used"][tool], self.state["budgets"][tool]
        if used >= budget:
            raise ToolError(f"the {tool} budget of this session ({budget} calls) is used up")
        self.state["used"][tool] = used + 1
        write_json(self.state_path, self.state)

    def remaining(self) -> dict:
        return {t: self.state["budgets"][t] - self.state["used"][t] for t in self.state["budgets"]}

    def model(self):
        if self._model is None:
            self._model = load_model(self.state["model"])
        return self._model

    def verifier(self):
        if self._verifier is None:
            from .verifier import Verifier

            cfg = {"verifier": self.state["verifier"], "entry": self.state["entry"], "reward": self.state["reward"]}
            self._verifier = Verifier(self.model(), self.state["network"], cfg)
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
            if not isinstance(code, str) or len(code) > MAX_SOURCE:
                raise ToolError("code must be a string of at most 200,000 characters")
            self._spend("run_python")
            n = self.state["used"]["run_python"]
            atomic_write(self.path / "python" / f"{n:03d}.py", code)  # in full: the log line below is cut short
            from .runner import Limits, run_recipe
            from .sandbox import choose, protected_paths

            m = self.model()
            py = self.state["python"]
            limits = Limits.from_config(py)
            cfg = {"sandbox": self.state["sandbox"], "sandbox_image": self.state["sandbox_image"], "runs_dir": self.state["runs_dir"]}
            sb = choose(cfg, protected_paths(cfg, m, Path(self.state["run_path"])))
            extra = {"entries": self.entries(), "reward": self.state["reward"], "limits": self.state["entry"]}
            ex = run_recipe(WRAPPER.replace("@@CODE@@", repr(code)), self.box_files(), m.load_weights(self.state["network"]),
                            self.info(), extra, limits, self.state["sandbox_dir"], sandbox=sb)
            saved, problems = [], []
            if ex.status == "ok" and isinstance(ex.result, dict):
                output = str(ex.result.get("stdout", ""))
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
            out = {"status": ex.status, "output": output[-MAX_TEXT:], "saved_entries": sorted(saved), "runtime_s": ex.runtime_s,
                   "sandbox": ex.mode, "remaining": self.remaining()}
            if problems:
                out["problems"] = problems
            atomic_write(self.path / "python" / f"{n:03d}.txt", out["output"])
            self._log("run_python", args, {"status": ex.status, "output": output[:500], "saved": sorted(saved), "sandbox": ex.mode}, time.monotonic() - t0)
            return out
        except ToolError as exc:
            self._log("run_python", args, {"error": str(exc)}, time.monotonic() - t0)
            raise

    def submit_entry(self, name: str, claimed_fidelity: float, expected_fidelity_other: float | None = None) -> dict:
        t0 = time.monotonic()
        args = {"name": name, "claimed_fidelity": claimed_fidelity, "expected_fidelity_other": expected_fidelity_other}
        try:
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
            self._spend("submit_entry")
            n = self.state["used"]["submit_entry"]
            folder = self.path / "submissions" / f"{n:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / "entry.json", entry)
            receipt = self.verifier().score(entry, claimed=float(claimed_fidelity))
            receipt.update(name=name, submission=n, at=now(),
                           expected_fidelity_other=None if expected_fidelity_other is None else float(expected_fidelity_other))
            write_json(folder / "receipt.json", receipt)
            self._log("submit_entry", args, {"status": receipt["status"], "reward": receipt["reward"], "reason": receipt["reason"]}, time.monotonic() - t0)
            return {"status": receipt["status"], "reason": receipt["reason"], "reward": receipt["reward"], "parts": receipt["parts"],
                    "size": receipt["size"], "inputs": receipt["inputs"], "overclaim": receipt.get("overclaim"),
                    "remaining": self.remaining()}
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
