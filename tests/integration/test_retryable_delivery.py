from __future__ import annotations

import json

import pytest


@pytest.mark.parametrize("event", ["SessionEnd", "Stop", "StopFailure"])
def test_emission_error_keeps_original_turn_and_progress_for_retry(
    hook_module, fake_langfuse, fixture_transcript_path, isolated_hook_state, monkeypatch,
    event,
):
    transcript = fixture_transcript_path("simple_turn")
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", None)
    key = hook_module.get_session_state_key("session-test", str(transcript))
    original = hook_module.SessionState(turn_progress={"user-1": {"emitted_keys": ["gen:old"]}})
    hook_module.save_session_state({}, key, original)
    before = hook_module.STATE_FILE.read_bytes()
    emit = hook_module.emit_turn

    def fail(*args, **kwargs):
        raise RuntimeError("Emission failed")

    monkeypatch.setattr(hook_module, "emit_turn", fail)
    with pytest.raises(RuntimeError, match="Emission failed"):
        hook_module.emit_new_turns_from_transcript(
            fake_langfuse, config, "session-test", transcript,
            flush_deferred_agent_turns=event == "SessionEnd",
            stop_failure={"error":"rate_limit"} if event == "StopFailure" else None,
        )
    assert hook_module.STATE_FILE.read_bytes() == before
    monkeypatch.setattr(hook_module, "emit_turn", emit)
    assert hook_module.emit_new_turns_from_transcript(
        fake_langfuse, config, "session-test", transcript, flush_deferred_agent_turns=True,
    ) == 1


@pytest.fixture
def json_batches(hook_module, monkeypatch):
    def encode(spans):
        return json.dumps(spans, ensure_ascii=False).encode("utf-8")

    monkeypatch.setattr(hook_module, "serialize_span_batch", encode, raising=False)
    return encode


def test_batches_are_split_by_encoded_bytes_without_changing_spans(
    hook_module, json_batches,
):
    spans = [{"id": str(i), "input": "한글" * 8} for i in range(8)]
    exporter = hook_module.BufferedSpanExporter(max_batch_bytes=160)
    exporter.stage(spans)
    packets = sorted(exporter.directory.glob("*.bin"))
    assert len(packets) > 1
    assert all(len(path.read_bytes()) <= 160 for path in packets)
    assert [span for path in packets for span in json.loads(path.read_bytes())] == spans


def test_an_oversized_observation_is_not_silently_dropped(hook_module, json_batches):
    exporter = hook_module.BufferedSpanExporter(max_batch_bytes=64)
    with pytest.raises(ValueError, match="single observation"):
        exporter.stage([{"input": "한글" * 100}])


def test_partial_delivery_retries_original_ids_and_commits_only_after_success(
    hook_module, json_batches, isolated_hook_state, monkeypatch,
):
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", None)
    key = "delivery-session-key"
    hook_module.save_session_state({}, key, hook_module.SessionState(offset=7))
    before = hook_module.STATE_FILE.read_bytes()
    exporter = hook_module.BufferedSpanExporter(max_batch_bytes=90)
    exporter.stage([{"id": "root-id", "input": "x" * 30},
                    {"id": "child-id", "parent": "root-id", "input": "y" * 20}])
    hook_module.prepare_pending_delivery(key, config, hook_module.SessionState(offset=42), exporter)
    attempted = []

    def failing_send(config, payload, timeout):
        attempted.append(payload)
        if len(attempted) == 2:
            raise OSError("Server unavailable")

    monkeypatch.setattr(hook_module, "send_trace_batch", failing_send)
    with pytest.raises(OSError, match="Server unavailable"):
        hook_module.finish_pending_delivery(key, config)
    assert hook_module.STATE_FILE.read_bytes() == before
    first, failed = attempted
    monkeypatch.setattr(hook_module, "send_trace_batch",
                        lambda config, payload, timeout: attempted.append(payload))
    hook_module.finish_pending_delivery(key, config)
    assert attempted == [first, failed, failed]
    assert hook_module.load_hook_state()[key]["offset"] == 42
    assert not hook_module.get_pending_delivery_path(key).exists()


def test_lost_acknowledgement_checkpoint_replays_identical_payload(
    hook_module, json_batches, isolated_hook_state, monkeypatch,
):
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", None)
    hook_module.save_session_state({}, "key", hook_module.SessionState(offset=7))
    exporter = hook_module.BufferedSpanExporter(max_batch_bytes=256)
    exporter.stage([{"id": "root-id", "child": "child-id"}])
    hook_module.prepare_pending_delivery("key", config, hook_module.SessionState(offset=42), exporter)
    sent = []
    monkeypatch.setattr(hook_module, "send_trace_batch", lambda c, p, t: sent.append(p))
    write_manifest = hook_module.write_delivery_manifest

    def fail_checkpoint(path, manifest):
        raise OSError("Acknowledgement checkpoint failed")

    monkeypatch.setattr(hook_module, "write_delivery_manifest", fail_checkpoint)
    with pytest.raises(OSError, match="checkpoint"):
        hook_module.finish_pending_delivery("key", config)
    assert hook_module.load_hook_state()["key"]["offset"] == 7
    monkeypatch.setattr(hook_module, "write_delivery_manifest", write_manifest)
    hook_module.finish_pending_delivery("key", config)
    assert len(sent) == 2
    assert sent[0] == sent[1]
    assert hook_module.load_hook_state()["key"]["offset"] == 42


def test_pending_delivery_is_not_sent_to_another_project(
    hook_module, json_batches, isolated_hook_state, monkeypatch,
):
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", None)
    exporter = hook_module.BufferedSpanExporter(max_batch_bytes=256)
    exporter.stage([{"id": "root-id"}])
    hook_module.prepare_pending_delivery("key", config, hook_module.SessionState(offset=42), exporter)
    other = hook_module.LangfuseConfig("other-public", "secret", config.host, None)
    sent = []
    monkeypatch.setattr(hook_module, "send_trace_batch",
                        lambda *args: sent.append(args))
    with pytest.raises(ValueError, match="another Langfuse project"):
        hook_module.finish_pending_delivery("key", other)
    assert sent == []
    assert hook_module.get_pending_delivery_path("key").exists()


def test_shutdown_timeout_does_not_advance_session_state(
    hook_module, fake_langfuse, fixture_transcript_path, isolated_hook_state,
    monkeypatch, json_batches,
):
    transcript = fixture_transcript_path("simple_turn")
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", None)
    key = hook_module.get_session_state_key("session-test", str(transcript))
    hook_module.save_session_state({}, key, hook_module.SessionState())
    before = hook_module.STATE_FILE.read_bytes()
    fake_langfuse._cc_delivery_exporter = hook_module.BufferedSpanExporter()
    monkeypatch.setattr(hook_module, "flush_and_shutdown_langfuse_client", lambda client: False)
    with pytest.raises(TimeoutError, match="shutdown"):
        hook_module.emit_new_turns_from_transcript(
            fake_langfuse, config, "session-test", transcript, flush_deferred_agent_turns=True,
        )
    assert hook_module.STATE_FILE.read_bytes() == before


def test_state_write_failure_retains_delivery_checkpoint_without_resending(
    hook_module, json_batches, isolated_hook_state, monkeypatch,
):
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", None)
    exporter = hook_module.BufferedSpanExporter(max_batch_bytes=256)
    exporter.stage([{"id": "root-id"}])
    hook_module.prepare_pending_delivery("key", config, hook_module.SessionState(offset=42), exporter)
    sent = []
    monkeypatch.setattr(hook_module, "send_trace_batch", lambda c, p, t: sent.append(p))
    replace = hook_module.os.replace

    def fail_state_write(source, destination):
        if destination == hook_module.STATE_FILE:
            raise OSError("Disk full")
        return replace(source, destination)

    monkeypatch.setattr(hook_module.os, "replace", fail_state_write)
    with pytest.raises(OSError, match="state"):
        hook_module.finish_pending_delivery("key", config)
    assert hook_module.get_pending_delivery_path("key").exists()
    monkeypatch.setattr(hook_module.os, "replace", replace)
    hook_module.finish_pending_delivery("key", config)
    assert len(sent) == 1
    assert hook_module.load_hook_state()["key"]["offset"] == 42


def test_completed_delivery_cleanup_failure_does_not_block_next_hook(
    hook_module, json_batches, isolated_hook_state, monkeypatch,
):
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", None)
    exporter = hook_module.BufferedSpanExporter(max_batch_bytes=256)
    exporter.stage([{"id": "root-id"}])
    hook_module.prepare_pending_delivery("key", config, hook_module.SessionState(offset=42), exporter)
    sent = []
    monkeypatch.setattr(hook_module, "send_trace_batch", lambda c, p, t: sent.append(p))

    def interrupt_cleanup(path):
        (path / "manifest.json").unlink()
        raise OSError("Directory removal failed")

    monkeypatch.setattr(hook_module.shutil, "rmtree", interrupt_cleanup)
    hook_module.finish_pending_delivery("key", config)
    assert not hook_module.get_pending_delivery_path("key").exists()
    hook_module.finish_pending_delivery("key", config)
    assert len(sent) == 1
    assert hook_module.load_hook_state()["key"]["offset"] == 42


def test_completed_delivery_rename_failure_keeps_readable_checkpoint(
    hook_module, json_batches, isolated_hook_state, monkeypatch,
):
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", None)
    exporter = hook_module.BufferedSpanExporter(max_batch_bytes=256)
    exporter.stage([{"id": "root-id"}])
    hook_module.prepare_pending_delivery("key", config, hook_module.SessionState(offset=42), exporter)
    pending = hook_module.get_pending_delivery_path("key")
    sent = []
    monkeypatch.setattr(hook_module, "send_trace_batch", lambda c, p, t: sent.append(p))
    rename = hook_module.Path.rename

    def fail_rename(path, target):
        if path == pending:
            raise OSError("Rename failed")
        return rename(path, target)

    monkeypatch.setattr(hook_module.Path, "rename", fail_rename)
    with pytest.raises(OSError, match="Rename failed"):
        hook_module.finish_pending_delivery("key", config)
    assert json.loads((pending / "manifest.json").read_text())["pending_batches"] == []
    monkeypatch.setattr(hook_module.Path, "rename", rename)
    hook_module.finish_pending_delivery("key", config)
    assert len(sent) == 1
    assert not pending.exists()
    assert hook_module.load_hook_state()["key"]["offset"] == 42


def test_queue_drain_flushes_spans_without_joining_the_media_queue(
    hook_module, monkeypatch, fake_langfuse,
):
    class Provider:
        def force_flush(self, timeout_millis):
            return True

    def media_wait():
        raise AssertionError("The media queue must not block span staging")

    monkeypatch.setenv("OTEL_BSP_MAX_QUEUE_SIZE", "2")
    monkeypatch.setattr(hook_module.otel_trace_api, "get_tracer_provider", lambda: Provider(), raising=False)
    monkeypatch.setattr(fake_langfuse, "flush", media_wait, raising=False)
    hook_module.BufferedSpanExporter().before_observation_end(fake_langfuse)


def test_queue_drain_timeout_is_bounded(hook_module, monkeypatch, fake_langfuse):
    import threading
    release = threading.Event()

    class Provider:
        def force_flush(self, timeout_millis):
            release.wait()
            return True

    monkeypatch.setenv("OTEL_BSP_MAX_QUEUE_SIZE", "2")
    monkeypatch.setattr(hook_module, "get_langfuse_shutdown_timeout_seconds", lambda: 0.01)
    monkeypatch.setattr(hook_module.otel_trace_api, "get_tracer_provider", lambda: Provider(), raising=False)
    monkeypatch.setattr(fake_langfuse, "flush", lambda: None, raising=False)
    try:
        with pytest.raises(TimeoutError, match="flush"):
            hook_module.BufferedSpanExporter().before_observation_end(fake_langfuse)
    finally:
        release.set()
