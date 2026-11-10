"""The agent is not told what the model does: a check on everything the agent can reach.

The forbidden strings are the run config's `hygiene` list and the model's own (withheld/<model>/task.yaml,
which the agent's tools never reach). Before a run starts, and before every prompt is sent, each text the
agent can see is searched for them, whatever the letter case: the prompt, the tools' descriptions, the files
of the sandbox folder, the network's id and the information handed to the agent's code. One hit stops the
run before a model call is made.

This is a check on the surface the pipeline gives the agent. It cannot see what a weak sandbox lets the
agent's code read for itself (sandbox.py: process mode).
"""

from __future__ import annotations


class HygieneError(RuntimeError):
    pass


def hits(texts: dict, forbidden: list) -> list:
    """[(where, forbidden string, the line it is on)] for every occurrence, letter case ignored."""
    out = []
    for where, text in texts.items():
        low_lines = str(text).lower().splitlines()
        lines = str(text).splitlines()
        for s in forbidden:
            needle = str(s).lower()
            if not needle:
                continue
            for i, line in enumerate(low_lines):
                if needle in line:
                    out.append((where, str(s), lines[i].strip()[:160]))
    return out


def check(texts: dict, forbidden: list) -> None:
    found = hits(texts, forbidden)
    if found:
        shown = "\n".join(f"  {where}: {s!r} in: {line}" for where, s, line in found[:20])
        raise HygieneError(f"the agent would be told what the model does ({len(found)} places):\n{shown}")
