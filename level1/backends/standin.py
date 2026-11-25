"""A stand-in for OpenAI's Responses API on this machine: no model, no network beyond 127.0.0.1.

It answers every POST .../responses with scripted output items, as the API's event stream, and keeps what it was
sent. Two uses: the codex backend asks the Codex CLI what it would send a model before a run trusts it with one
(codex.py, the isolation check), and the tests drive the real HTTP paths against it.

    with StandIn(lambda body, k: [message("ok")]) as s:
        ... point a client at s.url ...
        s.requests        every request's JSON body, in order
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def message(text: str) -> dict:
    return {"type": "message", "id": "msg_standin", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def script_call(code: str, call_id: str = "call_standin") -> dict:
    """A call of the Codex CLI's script tool (its code mode), which runs JavaScript over the nested tools."""
    return {"type": "custom_tool_call", "id": "ctc_standin", "call_id": call_id, "namespace": "functions", "name": "exec",
            "input": code, "status": "completed"}


def decode(raw: bytes, encoding: str | None) -> bytes:
    if encoding == "zstd":  # the CLI compresses requests to OpenAI's own servers; to another address it does not
        import zstandard

        return zstandard.ZstdDecompressor().decompressobj().decompress(raw)
    if encoding == "gzip":
        import gzip

        return gzip.decompress(raw)
    return raw


class StandIn:
    def __init__(self, answer, usage: dict | None = None, model: str | None = None):
        """answer(body, k) gives the output items of the k-th request (from 0), or an (http status, text) pair."""
        self.answer, self.requests, self.headers = answer, [], []
        self.usage = usage or {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 10,
                               "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 110}
        self.model = model
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, status: int, obj) -> None:
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._json(200, {"object": "list", "data": []})

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("content-length") or 0))
                try:
                    body = json.loads(decode(raw, self.headers.get("content-encoding")))
                except Exception:
                    body = {}
                if not self.path.rstrip("/").endswith("/responses"):
                    return self._json(404, {"error": {"code": "not_found", "message": "the stand-in answers /responses only"}})
                with outer.lock:
                    k = len(outer.requests)
                    outer.requests.append(body)
                    outer.headers.append({name.lower(): value for name, value in self.headers.items()})
                out = outer.answer(body, k)
                if isinstance(out, tuple):
                    return self._json(out[0], {"error": {"code": "standin", "message": out[1]}})
                rid = f"resp_standin_{k}"
                done = {"id": rid, "object": "response", "status": "completed", "model": outer.model or body.get("model"),
                        "output": out, "usage": outer.usage}
                events = [{"type": "response.created", "response": {"id": rid, "object": "response", "status": "in_progress", "output": []}}]
                for i, item in enumerate(out):
                    events.append({"type": "response.output_item.added", "output_index": i, "item": item})
                    events.append({"type": "response.output_item.done", "output_index": i, "item": item})
                events.append({"type": "response.completed", "response": done})
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.end_headers()
                for ev in events:
                    self.wfile.write(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode())
                    self.wfile.flush()

        self.lock = threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
