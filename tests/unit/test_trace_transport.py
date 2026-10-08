from __future__ import annotations

import gzip
import http.server
import threading
import urllib.error

import pytest


@pytest.mark.parametrize("compression,expected_encoding", [("gzip", "gzip"), ("none", None)])
def test_transport_preserves_custom_export_path_and_compression(
    hook_module, monkeypatch, compression, expected_encoding,
):
    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            encoding = self.headers.get("Content-Encoding")
            received.append((self.path, encoding, gzip.decompress(raw) if encoding else raw))
            self.send_response(200)
            self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = hook_module.LangfuseConfig("public", "secret", f"http://127.0.0.1:{server.server_port}", None)
    monkeypatch.setenv("LANGFUSE_OTEL_TRACES_EXPORT_PATH", "custom/traces")
    monkeypatch.setenv("LANGFUSE_OTEL_COMPRESSION", compression)
    try:
        hook_module.send_trace_batch(config, b"exact-otel-payload", 2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert received == [("/custom/traces", expected_encoding, b"exact-otel-payload")]


def test_redirect_cannot_acknowledge_a_post_as_a_get(hook_module):
    methods = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            methods.append("POST")
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(302)
            self.send_header("Location", "/redirected")
            self.end_headers()

        def do_GET(self):
            methods.append("GET")
            self.send_response(200)
            self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = hook_module.LangfuseConfig("public", "secret", f"http://127.0.0.1:{server.server_port}", None)
    try:
        with pytest.raises(urllib.error.HTTPError):
            hook_module.send_trace_batch(config, b"otel-payload", 2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert methods == ["POST"]


def test_transport_preserves_otel_headers_and_sdk_auth_precedence(hook_module, monkeypatch):
    captured = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            captured.append(dict(self.headers))
            self.send_response(200)
            self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = hook_module.LangfuseConfig("public", "secret", f"http://127.0.0.1:{server.server_port}", None)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-tenant=generic")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "x-tenant=trace%20tenant,authorization=override")
    try:
        hook_module.send_trace_batch(config, b"otel-payload", 2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    headers = {k.lower():v for k,v in captured[0].items()}
    assert headers["x-tenant"] == "trace tenant"
    assert headers["authorization"] == "Basic cHVibGljOnNlY3JldA=="
    assert headers["x-langfuse-sdk-name"] == "python"
    assert headers["x-langfuse-public-key"] == "public"
