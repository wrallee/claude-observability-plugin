from __future__ import annotations

import base64
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest


@pytest.fixture
def sdk_delivery(tmp_path):
    requests = []
    responses = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            reply = responses.pop(0) if responses else (200, b"")
            code, body = reply[:2]
            content_type = reply[2] if len(reply) == 3 else "application/x-protobuf"
            requests.append((code, payload, body))
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    transcript = tmp_path / "sdk-session.jsonl"
    rows = []
    for index in range(3):
        rows.extend([
            {"type": "user", "uuid": f"u{index}", "sessionId": "sdk-session",
             "timestamp": f"2026-01-01T00:00:0{index*2}Z",
             "message": {"role": "user", "content": "question " + "x" * 19000}},
            {"type": "assistant", "uuid": f"a{index}",
             "timestamp": f"2026-01-01T00:00:0{index*2+1}Z",
             "message": {"role": "assistant", "id": f"msg-{index}", "model": "claude-test",
                         "content": [{"type": "text", "text": "answer " + "y" * 19000}]}},
        ])
    transcript.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_file = state_dir / "langfuse_state.json"
    state_file.write_text(json.dumps({"unrelated": {"offset": 7}}))
    env = {k:v for k,v in os.environ.items()
           if not k.startswith(("LANGFUSE_", "CC_LANGFUSE_", "CLAUDE_PLUGIN_OPTION_", "OTEL_"))}
    env.update(LANGFUSE_BASE_URL=f"http://127.0.0.1:{server.server_port}",
               LANGFUSE_PUBLIC_KEY="pk-lf-sdk-test", LANGFUSE_SECRET_KEY="sk-lf-sdk-test",
               LANGFUSE_CAPTURE_IMAGES="false", CC_LANGFUSE_STATE_DIR=str(state_dir),
               LANGFUSE_OTEL_MAX_BATCH_SIZE_BYTES="131072")
    hook = Path(__file__).resolve().parents[2] / "hooks/langfuse_hook.py"

    def fire(slow_export=False, source=None):
        if source is not None:
            copied_rows = [dict(json.loads(line), sessionId="sdk-session")
                           for line in source.read_text().splitlines()]
            transcript.write_text("\n".join(json.dumps(row) for row in copied_rows) + "\n")
            import shutil
            shutil.copytree(source.with_suffix("") / "subagents",
                            transcript.with_suffix("") / "subagents")
        command = [sys.executable, str(hook)]
        if slow_export:
            code = """
import importlib.util,sys,time
spec=importlib.util.spec_from_file_location('sdk_hook',sys.argv[1])
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
stage=m.BufferedSpanExporter.stage
def slow_stage(self,spans):
    time.sleep(0.05)
    return stage(self,spans)
m.BufferedSpanExporter.stage=slow_stage
sys.exit(m.main())
"""
            command = [sys.executable, "-c", code, str(hook)]
        result = subprocess.run(
            command, text=True,
            input=json.dumps({"session_id":"sdk-session", "transcript_path":str(transcript),
                              "hook_event_name":"SessionEnd"}),
            env=env, cwd=tmp_path, capture_output=True, timeout=20,
        )
        assert result.returncode == 0, result.stderr

    try:
        yield fire, requests, responses, state_file, state_dir, env
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def decode_core(payloads):
    # Tests stub OTel in the parent interpreter. Decode in a clean process
    # so both emission and decoding use the installed SDK's real protobufs.
    code = """
import base64,json,sys
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
cores=[]
for payload in json.load(sys.stdin):
    request=ExportTraceServiceRequest.FromString(base64.b64decode(payload))
    for resource in request.resource_spans:
        for scope in resource.scope_spans:
            for span in scope.spans:
                attrs={a.key:a.value for a in span.attributes}
                cores.append({'id':span.span_id.hex(),'trace':span.trace_id.hex(),
                    'parent':span.parent_span_id.hex(),'name':span.name,
                    'root':attrs['langfuse.internal.is_app_root'].bool_value})
print(json.dumps(cores))
"""
    result = subprocess.run([sys.executable, "-c", code],
                            input=json.dumps([base64.b64encode(p).decode() for p in payloads]),
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def test_real_sdk_retries_partial_delivery_without_duplicate_spans(sdk_delivery):
    fire, requests, responses, state_file, state_dir, _ = sdk_delivery
    before = state_file.read_bytes()
    responses.extend([(200, b""), (503, b"")])
    fire()
    assert state_file.read_bytes() == before
    assert len(list((state_dir / "langfuse_delivery").glob("*/manifest.json"))) == 1
    failed = requests[-1][1]
    fire()
    assert requests[2][1] == failed
    assert all(len(payload) <= 131072 for _,payload,_ in requests)
    accepted = [payload for code,payload,body in requests if code == 200 and not body]
    cores = decode_core(accepted)
    assert len(cores) == len({s["id"] for s in cores}) == 6
    roots = {s["id"]:s for s in cores if s["root"]}
    assert len(roots) == 3
    assert all(s["parent"] in roots for s in cores if not s["root"])
    assert len({s["trace"] for s in cores}) == 3
    state = json.loads(state_file.read_text())
    assert state.pop("unrelated") == {"offset": 7}
    assert next(iter(state.values()))["turn_count"] == 3
    assert not list((state_dir / "langfuse_delivery").glob("*/manifest.json"))
    count = len(requests)
    fire()
    assert len(requests) == count


def test_real_sdk_serialization_failure_does_not_consume_the_turn(sdk_delivery):
    fire, requests, _, state_file, _, env = sdk_delivery
    before = state_file.read_bytes()
    env["LANGFUSE_OTEL_MAX_BATCH_SIZE_BYTES"] = "10"
    fire()
    assert requests == []
    assert state_file.read_bytes() == before
    env["LANGFUSE_OTEL_MAX_BATCH_SIZE_BYTES"] = "131072"
    fire()
    assert requests
    assert state_file.read_bytes() != before


@pytest.mark.parametrize("body", [
    b"{}",
    (Path(__file__).parents[1] / "fixtures/otel_responses/langfuse_ingestion_job.json").read_bytes(),
    b'{"partialSuccess":{"rejectedSpans":"0","errorMessage":"warning"}}',
    b'{"partialSuccess":null,"futureField":null}',
    b'{"partialSuccess":{"rejectedSpans":0,"futureWarning":"NaN"},"futureField":true}',
], ids=["empty", "langfuse-job", "warning", "null", "future-fields"])
def test_json_acknowledgement_commits_progress_without_resending(sdk_delivery, body):
    fire, requests, responses, state_file, state_dir, _ = sdk_delivery
    responses.append((200, body, "application/json; charset=utf-8"))
    fire()
    state = json.loads(state_file.read_text())
    assert state.pop("unrelated") == {"offset": 7}
    assert len(state) == 1
    assert next(iter(state.values()))["turn_count"] == 3
    assert not list((state_dir / "langfuse_delivery").glob("*/manifest.json"))
    cores = decode_core([payload for code, payload, _ in requests if code == 200])
    assert len(cores) == len({s["id"] for s in cores}) == 6
    count = len(requests)
    fire()
    assert len(requests) == count


@pytest.mark.parametrize("body", [
    b'{"partialSuccess":{"rejectedSpans":"1","errorMessage":"rejected"}}',
    b'{"partialSuccess":{"rejectedSpans":1},"name":"otel-ingestion-job","id":"job-1"}',
    b'{"partialSuccess":{"rejectedSpans":"invalid"}}',
    b'{"partialSuccess":[]}',
    b'{"partialSuccess":{"rejectedSpans":"1","rejected_spans":"0"}}',
    b'{"partialSuccess":{"rejectedSpans":"1"},"partial_success":{"rejected_spans":"0"}}',
    b'{"partialSuccess":{"rejectedSpans":"1"},"partialSuccess":{"rejectedSpans":"0"}}',
    b'{"partialSuccess":{"rejectedSpans":"1","rejectedSpans":"0"}}',
    b'{"error":"ingestion failed"}',
    b'{"name":"otel-ingestion-job","id":""}',
    b'{"name":"otel-ingestion-job","id":" "}',
    b'{"name":"otel-ingestion-job","id":123}',
    b'{"name":"unrelated-job","id":"job-1"}',
    b"{bad-json",
    b"[]",
    b'{"partialSuccess":{},"newField":NaN}',
    b'{"name":"otel-ingestion-job","id":"job-1","data":Infinity}',
    b'{"partialSuccess":{},"newField":-Infinity}',
])
def test_bad_json_acknowledgement_keeps_the_batch_and_session_state(sdk_delivery, body):
    fire, requests, responses, state_file, state_dir, _ = sdk_delivery
    before = state_file.read_bytes()
    responses.append((200, body, "application/json"))
    fire()
    assert state_file.read_bytes() == before
    assert len(list((state_dir / "langfuse_delivery").glob("*/manifest.json"))) == 1
    failed = requests[0][1]
    fire()
    assert requests[1][1] == failed
    cores = decode_core([payload for code, payload, response in requests if code == 200 and not response])
    assert len(cores) == len({s["id"] for s in cores}) == 6


def test_real_sdk_rejected_spans_keep_the_original_batch_for_retry(sdk_delivery):
    fire, requests, responses, state_file, _, _ = sdk_delivery
    before = state_file.read_bytes()
    # OTLP partial_success.rejected_spans = 1.
    responses.append((200, b"\x0a\x02\x08\x01"))
    fire()
    assert state_file.read_bytes() == before
    rejected = requests[0][1]
    fire()
    assert requests[1][1] == rejected
    cores = decode_core([payload for code,payload,body in requests if code == 200 and not body])
    assert len(cores) == len({s["id"] for s in cores}) == 6


def test_real_sdk_cannot_drop_spans_when_its_queue_fills(sdk_delivery):
    fire, requests, _, _, _, env = sdk_delivery
    env.update(OTEL_BSP_MAX_QUEUE_SIZE="2", LANGFUSE_FLUSH_AT="1")
    fire(slow_export=True)
    cores = decode_core([payload for code,payload,body in requests if code == 200 and not body])
    assert len(cores) == len({s["id"] for s in cores}) == 6


def test_real_sdk_keeps_nested_parents_when_they_end_consecutively(
    sdk_delivery, fixture_transcript_path,
):
    fire, requests, _, _, _, env = sdk_delivery
    env.update(OTEL_BSP_MAX_QUEUE_SIZE="2", LANGFUSE_FLUSH_AT="1")
    fire(slow_export=True, source=fixture_transcript_path("nested_subagents"))
    cores = decode_core([payload for code,payload,body in requests if code == 200 and not body])
    by_name = {s["name"]:s for s in cores}
    assert "Subagent: Inner agent" in by_name
    assert "Subagent: Outer agent" in by_name
    assert by_name["Subagent: Inner agent"]["parent"] == by_name["Subagent: Outer agent"]["id"]
    assert len(cores) == len({s["id"] for s in cores}) == 10
