"""proxy.py - drop-in proxy in front of breeze-server: same API, German-normalized text.

breeze-server runs internally (by default HTTP on 127.0.0.1:8147 and its
WebSocket on :8148). This proxy takes the ports clients already use (by default
8137/8138), normalizes text - numbers, units, abbreviations, markdown, emojis,
see german_normalizer.py - and passes everything else through unchanged. For
clients nothing changes, except that "3.000 m" is now spoken as "dreitausend
Meter" instead of with English digits.

Only the fields "text" and "ref_text" are rewritten:
  HTTP       form fields (multipart or urlencoded) and query parameters,
             e.g. POST /v1/audio/speech, /v1/voices, /v1/audio/convert
  WebSocket  "text" messages sentence by sentence through SentenceBuffer
             (token streams from a chat model), "end"/"flush" with the rest,
             "ref_text" in "start"
/health reports the proxy's own WebSocket port, so that the Breeze web UI
connects here and not past the proxy.

Configuration via environment variables (defaults in brackets):
  PROXY_HOST            interface to bind (0.0.0.0)
  PROXY_HTTP_PORT       HTTP port (8137)
  PROXY_WS_PORT         WebSocket port (8138)
  BREEZE_UPSTREAM_HTTP  breeze-server HTTP base URL (http://127.0.0.1:8147)
  BREEZE_UPSTREAM_WS    breeze-server WebSocket URL (ws://127.0.0.1:8148)

There is no authentication, just like breeze-server itself.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

from german_normalizer import SentenceBuffer, normalize

UPSTREAM_HTTP = os.environ.get("BREEZE_UPSTREAM_HTTP", "http://127.0.0.1:8147")
UPSTREAM_WS = os.environ.get("BREEZE_UPSTREAM_WS", "ws://127.0.0.1:8148")
HOST = os.environ.get("PROXY_HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("PROXY_HTTP_PORT", "8137"))
WS_PORT = int(os.environ.get("PROXY_WS_PORT", "8138"))
TEXT_FIELDS = {"text", "ref_text"}
# not forwarded as-is: length and encoding no longer match once the text is rewritten
HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "content-encoding"}

app = FastAPI(title="Breeze German text proxy", docs_url=None, redoc_url=None, openapi_url=None)
client = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10))


def _normalize_field(key: str, value: str) -> str:
    return normalize(value).strip() if key in TEXT_FIELDS and value else value


def _log_rewrite(where: str, before: str, after: str) -> None:
    if before != after:
        print(f"[text] {where}: {before[:80]!r} -> {after[:80]!r}", flush=True)


@app.get("/health")
async def health() -> Response:
    r = await client.get(UPSTREAM_HTTP + "/health")
    try:
        d = r.json()
        d["ws_port"] = WS_PORT
        return Response(json.dumps(d), status_code=r.status_code, media_type="application/json")
    except ValueError:
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def forward(path: str, req: Request) -> Response:
    url = f"{UPSTREAM_HTTP}/{path}"
    params = [(k, _normalize_field(k, v)) for k, v in req.query_params.multi_items()]
    headers = {k: v for k, v in req.headers.items() if k.lower() not in HOP_HEADERS}
    content_type = req.headers.get("content-type", "")
    is_form = "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type
    if req.method == "POST" and is_form:
        form = await req.form()
        fields = []
        for k, v in form.multi_items():
            if hasattr(v, "read"):          # uploaded file (ref_audio, source)
                fields.append((k, (v.filename or k, await v.read(), v.content_type or "application/octet-stream")))
            else:
                new = _normalize_field(k, v)
                _log_rewrite(f"{path} {k}", v, new)
                fields.append((k, (None, new)))  # (None, value) = plain form field
        headers.pop("content-type", None)    # httpx sets the new multipart boundary itself
        if "multipart/form-data" in content_type:
            upstream_req = client.build_request(req.method, url, params=params, files=fields, headers=headers)
        else:
            upstream_req = client.build_request(req.method, url, params=params, headers=headers,
                                                data={k: v[1] for k, v in fields})
    else:
        upstream_req = client.build_request(req.method, url, params=params, headers=headers,
                                            content=await req.body())
    try:
        r = await client.send(upstream_req, stream=True)
    except httpx.HTTPError as exc:
        return Response(json.dumps({"error": f"breeze-server unreachable: {type(exc).__name__}"}),
                        status_code=502, media_type="application/json")
    # stream it through: /v1/audio/speech delivers PCM while it is being generated
    return StreamingResponse(r.aiter_raw(), status_code=r.status_code,
                             headers={k: v for k, v in r.headers.items() if k.lower() not in HOP_HEADERS},
                             background=BackgroundTask(r.aclose))


async def ws_relay(conn) -> None:
    """Relay one WebSocket connection to breeze-server, normalizing text on the way."""
    path = conn.request.path or "/"
    try:
        upstream = await websockets.connect(UPSTREAM_WS + path, max_size=None)
    except Exception as exc:                          # noqa: BLE001
        await conn.send(json.dumps({"type": "error", "message": f"breeze-server unreachable: {exc}"}))
        return
    buffer = [SentenceBuffer()]

    async def client_to_upstream() -> None:
        async for raw in conn:
            if isinstance(raw, bytes):
                await upstream.send(raw)
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await upstream.send(raw)
                continue
            kind = msg.get("type")
            if kind == "text":
                ready = buffer[0].push(msg.get("text", ""))
                if ready:
                    await upstream.send(json.dumps({"type": "text", "text": ready}, ensure_ascii=False))
            elif kind in ("end", "flush"):
                msg["text"] = buffer[0].flush(msg.get("text", ""))
                await upstream.send(json.dumps(msg, ensure_ascii=False))
            elif kind == "cancel":
                buffer[0] = SentenceBuffer()          # anything held back belongs to the cancelled utterance
                await upstream.send(raw)
            elif kind == "start" and msg.get("ref_text"):
                msg["ref_text"] = _normalize_field("ref_text", msg["ref_text"])
                await upstream.send(json.dumps(msg, ensure_ascii=False))
            else:
                await upstream.send(raw)

    async def upstream_to_client() -> None:
        async for raw in upstream:
            await conn.send(raw)

    a, b = asyncio.create_task(client_to_upstream()), asyncio.create_task(upstream_to_client())
    try:
        # Ends as soon as EITHER side closes. breeze-server never closes on its
        # own (not even after "done") - the client decides when it is over.
        _, pending = await asyncio.wait({a, b}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
    finally:
        await upstream.close()


async def main() -> None:
    # table (re)loads of the normalizer are worth seeing; library chatter (httpx per request) is not
    logging.basicConfig(format="[%(name)s] %(message)s")
    logging.getLogger("german_normalizer").setLevel(logging.INFO)
    async with websockets.serve(ws_relay, HOST, WS_PORT, max_size=None):
        print(f"[proxy] HTTP {HOST}:{HTTP_PORT} and WebSocket {HOST}:{WS_PORT} -> "
              f"{UPSTREAM_HTTP} / {UPSTREAM_WS}", flush=True)
        await uvicorn.Server(uvicorn.Config(app, host=HOST, port=HTTP_PORT, log_level="warning")).serve()


if __name__ == "__main__":
    asyncio.run(main())
