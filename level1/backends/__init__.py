"""Agent backends: who answers a round.

Every backend has `available(index)` and `respond(prompt_text, index, where, session)`: index counts the
rounds this backend has answered in the run, so a resumed run asks for the same response again; where is
the round's folder; session is the round's Session (its tools, budgets and log). The answer is the agent's
closing notes; what it did, it did through the session's tools.

    claude_code   the Claude Code CLI, headless, under the user's own login, with the two tools attached
    fake          scripted tool calls replayed from JSON files: tests, dry runs
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Response:
    text: str
    meta: dict = field(default_factory=dict)


class BackendError(RuntimeError):
    pass


class StopRun(RuntimeError):
    """The run cannot go on now (a login problem, a usage limit that did not lift, another model answering).
    The loop stops cleanly; `resume` continues it later."""


def make(cfg: dict, log=print, fake_scripts=None):
    if cfg["backend"] == "fake":
        from .fake import FakeBackend

        return FakeBackend(fake_scripts or cfg.get("fake_scripts") or [])
    if cfg["backend"] == "claude_code":
        from .claude_code import ClaudeCodeBackend

        return ClaudeCodeBackend(cfg, log=log)
    raise BackendError(f"no backend named {cfg['backend']!r}; there are claude_code and fake")
