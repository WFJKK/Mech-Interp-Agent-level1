"""The session's tools as an MCP server over stdio, for the Claude Code CLI.

    python -m level1.mcp_server --session <round>/session

Speaks JSON-RPC 2.0, one message per line (the Model Context Protocol's stdio transport): initialize,
notifications/initialized, ping, tools/list and tools/call. Each tool call runs the session's tool
(session.py) and returns its result as one text block of JSON; a refused call is a text block with
isError set, so the agent reads the reason and goes on. Nothing is written to stdout but protocol
messages; diagnostics go to stderr.

Taken from the proof pipeline (core/open_lane/mcp_server.py of Compact-Proof-Agent).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .session import Session, ToolError

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_NAME = "lv1"
SERVER_INFO = {"name": "level1-tools", "version": "1"}


def tool_list(session: Session) -> list[dict]:
    return [{"name": name, "description": spec["description"], "inputSchema": spec["schema"]}
            for name, spec in session.TOOLS.items()]


def handle(session: Session, msg: dict) -> dict | None:
    """The response to one request, or None for a notification."""
    method, params, mid = msg.get("method"), msg.get("params") or {}, msg.get("id")
    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return {"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": version, "capabilities": {"tools": {}},
                                                       "serverInfo": SERVER_INFO}}
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None
    if mid is None:
        return None  # an unknown notification
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": tool_list(session)}}
    if method == "tools/call":
        name, arguments = params.get("name"), params.get("arguments") or {}
        try:
            result = session.call(name, arguments)
            return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": json.dumps(result, indent=1)}],
                                                           "isError": False}}
        except ToolError as exc:
            return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": f"refused: {exc}"}], "isError": True}}
        except Exception as exc:  # a tool that failed for the pipeline's reasons: say so, keep serving
            print(f"tool {name} failed: {exc!r}", file=sys.stderr, flush=True)
            return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": f"the tool failed: {type(exc).__name__}"}], "isError": True}}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}


def serve(session: Session, inp=None, out=None) -> None:
    inp = inp or sys.stdin
    out = out or sys.stdout
    for line in inp:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            out.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}) + "\n")
            out.flush()
            continue
        if not isinstance(msg, dict):
            continue
        reply = handle(session, msg)
        if reply is not None:
            out.write(json.dumps(reply) + "\n")
            out.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="the session folder (with session.json)")
    args = ap.parse_args(argv)
    serve(Session(Path(args.session)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
