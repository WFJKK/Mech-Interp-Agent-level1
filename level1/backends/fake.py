"""The fake backend: replays scripted tool calls from JSON files, in order, against the real session code,
so the whole loop (tools, sandbox, verifier, receipts, records, resume) runs without a model. A script is

    {"calls": [{"tool": "run_python", "args": {"code": {"$file": "explore.py"}}},
               {"tool": "submit_entry", "args": {"name": "first", "claimed_fidelity": 1.0}}],
     "response": "the closing notes"}

and a string argument may be {"$file": "relative path"} to read it from a file next to the script. For tests
of the loop, "stop_run": "why" makes the backend stop the run after the calls (as a usage limit would), and
"crash": "why" makes it fail there (as a killed process would).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..session import ToolError
from . import BackendError, Response, StopRun


class FakeBackend:
    name = "fake"

    def __init__(self, scripts):
        self.items: list[Path] = []
        for src in ([scripts] if isinstance(scripts, (str, Path)) else scripts):
            p = Path(src)
            if p.is_dir():
                self.items += sorted(p.glob("*.json"))
            elif p.is_file():
                self.items.append(p)
            else:
                raise BackendError(f"no scripts at {p}")

    def available(self, index: int) -> bool:
        return 0 <= index < len(self.items)

    @staticmethod
    def _resolve(value, base: Path):
        if isinstance(value, dict) and set(value) == {"$file"}:
            return (base / value["$file"]).read_text()
        if isinstance(value, dict):
            return {k: FakeBackend._resolve(v, base) for k, v in value.items()}
        if isinstance(value, list):
            return [FakeBackend._resolve(v, base) for v in value]
        return value

    def respond(self, prompt_text: str, index: int, where, session) -> Response:
        if not self.available(index):
            raise BackendError(f"the fake backend has {len(self.items)} scripts; asked for number {index + 1}")
        path = self.items[index]
        script = json.loads(path.read_text())
        t0 = time.monotonic()
        outcomes = []
        for call in script.get("calls", []):
            tool, args = call["tool"], self._resolve(call.get("args", {}), path.parent)
            try:
                result = session.call(tool, args)
                outcomes.append({"tool": tool, "ok": True, "status": result.get("status"), "reward": result.get("reward")})
            except ToolError as exc:
                outcomes.append({"tool": tool, "ok": False, "error": str(exc)})
        if script.get("stop_run"):
            raise StopRun(str(script["stop_run"]))
        if script.get("crash"):
            raise RuntimeError(str(script["crash"]))
        meta = {"backend": self.name, "source": path.name, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                "stop_reason": script.get("stop_reason", "end_turn"), "tool_calls": outcomes, "tools_used": dict(session.state["used"]),
                "submissions": len(session.receipts()), "seconds": round(time.monotonic() - t0, 1)}
        return Response(text=script.get("response", ""), meta=meta)
