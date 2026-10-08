from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import pytest


ERROR_TEXT = "You've hit your org's monthly spend limit"


@pytest.fixture
def failed_session(hook_module, fixture_transcript_path, fake_langfuse, isolated_hook_state,
                   tmp_path, monkeypatch):
    transcript = tmp_path / "session-agent-deferred.jsonl"
    transcript.write_text(
        fixture_transcript_path("async_agent_deferred").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    shutil.copytree(
        fixture_transcript_path("async_agent_deferred").with_suffix("") / "subagents",
        transcript.with_suffix("") / "subagents",
    )
    error_row = {
        "type": "assistant", "uuid": "api-error-1",
        "timestamp": "2026-01-01T00:03:04.000Z",
        "isApiErrorMessage": True, "error": "rate_limit",
        "message": {"id": "error-message-1", "role": "assistant",
                    "content": [{"type": "text", "text": ERROR_TEXT}]},
    }
    with transcript.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(error_row) + "\n")
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", "user-1")
    monkeypatch.setattr(hook_module, "get_langfuse_config", lambda: config)
    monkeypatch.setattr(hook_module, "create_langfuse_client", lambda _config: fake_langfuse)
    monkeypatch.setattr(hook_module, "flush_and_shutdown_langfuse_client", lambda _client: None)

    def fire(event, *, camel_case=False, error="rate_limit", message=ERROR_TEXT, details=None,
             prompt_id=None):
        payload = {"session_id": "session-agent-deferred", "transcript_path": str(transcript),
                   "hookEventName" if camel_case else "hook_event_name": event,
                   "error": error}
        if message is not None:
            payload["last_assistant_message"] = message
        if details is not None:
            payload["error_details"] = details
        if prompt_id is not None:
            payload["prompt_id"] = prompt_id
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
        assert hook_module.main() == 0

    return transcript, fire


@pytest.mark.parametrize("camel_case", [False, True])
def test_stop_failure_exports_the_error_while_an_agent_is_running(
    failed_session, fake_langfuse, camel_case,
):
    _, fire = failed_session
    fire("StopFailure", camel_case=camel_case)

    roots = [o for o in fake_langfuse.observations if o.name == "Conversational Turn"]
    assert len(roots) == 1
    assert roots[0].kwargs["level"] == "ERROR"
    assert roots[0].output == {"role": "assistant", "content": ERROR_TEXT}
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    assert len(errors) == 1
    assert errors[0].kwargs["status_message"] == ERROR_TEXT


@pytest.mark.parametrize("separate_batches", [False, True])
@pytest.mark.parametrize("delayed_assistant", [False, True])
def test_delayed_transcript_errors_reconcile_multiple_failure_payloads(
    failed_session, fake_langfuse, separate_batches, delayed_assistant,
):
    transcript, fire = failed_session
    rows = [json.loads(line) for line in transcript.read_text().splitlines()]
    error_row = rows.pop()
    transcript.write_text("".join(json.dumps(row) + "\n" for row in rows))
    fire("StopFailure")
    fire("StopFailure", error="authentication_failed", message="API Error: Invalid key")
    with transcript.open("a", encoding="utf-8") as stream:
        if delayed_assistant:
            stream.write(json.dumps({
                "type": "assistant", "uuid": "preceding-error",
                "message": {"id": "preceding-error-message", "role": "assistant",
                            "content": "Preparing the next step."},
            }) + "\n")
        stream.write(json.dumps(error_row) + "\n")
    if separate_batches:
        fire("StopFailure", error="authentication_failed", message="API Error: Invalid key")
        fire("StopFailure", error="authentication_failed", message="API Error: Invalid key")
    with transcript.open("a", encoding="utf-8") as stream:
        second_error = dict(error_row, uuid="api-error-2", error="authentication_failed",
                            message={"id": "error-message-2", "role": "assistant",
                                     "content": "API Error: Invalid key"})
        stream.write(json.dumps(second_error) + "\n")
    fire("StopFailure", error="authentication_failed", message="API Error: Invalid key")
    fire("SessionEnd")
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    assert [o.kwargs["status_message"] for o in errors] == [ERROR_TEXT, "API Error: Invalid key"]


def test_stop_keeps_waiting_for_the_agent_after_an_api_error(failed_session, fake_langfuse):
    _, fire = failed_session
    fire("Stop")
    assert fake_langfuse.observations == []


def test_failure_details_without_assistant_message_preserve_new_failures(
    failed_session, fake_langfuse,
):
    transcript, fire = failed_session
    rows = [json.loads(line) for line in transcript.read_text().splitlines()][:-1]
    transcript.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for details in ["Model A limit", "Model B limit", "Model A limit"]:
        fire("StopFailure", message=None, details=details)
        fire("StopFailure", message=None, details=details)
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    assert [o.kwargs["status_message"] for o in errors] == [
        "Model A limit", "Model B limit", "Model A limit",
    ]


@pytest.mark.parametrize("repeat_original_payload", [False, True])
@pytest.mark.parametrize("has_error_code", [False, True])
def test_delayed_error_with_different_rendered_text_reuses_the_failure(
    failed_session, fake_langfuse, isolated_hook_state, repeat_original_payload, has_error_code,
):
    transcript, fire = failed_session
    rows = [json.loads(line) for line in transcript.read_text().splitlines()]
    actual_error = rows.pop()
    if not has_error_code:
        actual_error.pop("error")
    transcript.write_text("".join(json.dumps(row) + "\n" for row in rows))
    fire("StopFailure", message=None, details="429 model A limit")
    with transcript.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(actual_error) + "\n")
    for _ in range(2):
        if repeat_original_payload:
            fire("StopFailure", message=None, details="429 model A limit")
        else:
            fire("StopFailure", message=ERROR_TEXT)
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    assert len(errors) == 1
    state = json.loads((isolated_hook_state / "langfuse_state.json").read_text())
    held = next(iter(state.values()))["open_turn"]["rows"]
    assert held[-1]["message"]["content"] == actual_error["message"]["content"]
    assert not any(row.get("stop_failure_fallback") for row in held)


def test_missing_transcript_error_code_does_not_hide_a_new_hook_error(
    failed_session, fake_langfuse,
):
    transcript, fire = failed_session
    rows = [json.loads(line) for line in transcript.read_text().splitlines()]
    rows[-1].pop("error")
    transcript.write_text("".join(json.dumps(row) + "\n" for row in rows))
    fire("StopFailure", message=None)
    fire("StopFailure", error="authentication_failed", message=None)
    fire("StopFailure", error="authentication_failed", message=None)
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    assert len(errors) == 2


@pytest.mark.parametrize("has_error_code", [False, True])
def test_same_code_failures_reconcile_once_across_separate_batches(
    failed_session, fake_langfuse, has_error_code,
):
    transcript, fire = failed_session
    rows = [json.loads(line) for line in transcript.read_text().splitlines()]
    first_error = rows.pop()
    if not has_error_code:
        first_error.pop("error")
    transcript.write_text("".join(json.dumps(row) + "\n" for row in rows))
    fire("StopFailure", message=None, details="Model A limit", prompt_id="prompt-1")
    fire("StopFailure", message=None, details="Model B limit", prompt_id="prompt-2")
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    initial_ids = [o.id for o in errors]
    assert [o.kwargs["status_message"] for o in errors] == ["Model A limit", "Model B limit"]
    first_error["message"]["content"] = "Model A limit"
    with transcript.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(first_error) + "\n")
    for _ in range(2):
        fire("StopFailure", message=None, details="Model B limit", prompt_id="prompt-2")
    second_error = dict(first_error, uuid="api-error-2",
                        message=dict(first_error["message"], id="error-message-2",
                                     content="Model B limit"))
    with transcript.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(second_error) + "\n")
    fire("StopFailure", message=None, details="Model B limit", prompt_id="prompt-2")
    fire("SessionEnd")
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    assert [o.id for o in errors] == initial_ids


def test_returning_to_an_earlier_failure_is_a_new_failure(failed_session, fake_langfuse):
    transcript, fire = failed_session
    rows = [json.loads(line) for line in transcript.read_text().splitlines()][:-1]
    transcript.write_text("".join(json.dumps(row) + "\n" for row in rows))
    fire("StopFailure")
    fire("StopFailure", error="authentication_failed", message="API Error: Invalid key")
    fire("StopFailure")
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    assert [o.kwargs["status_message"] for o in errors] == [
        ERROR_TEXT, "API Error: Invalid key", ERROR_TEXT,
    ]


def test_a_new_transcript_failure_follows_an_already_recorded_failure(
    failed_session, fake_langfuse,
):
    transcript, fire = failed_session
    fire("StopFailure")
    previous = json.loads(transcript.read_text().splitlines()[-1])
    new_error = dict(previous, uuid="api-error-2", error="authentication_failed",
                     message={"id": "error-message-2", "role": "assistant",
                              "content": "API Error: Invalid key"})
    with transcript.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(new_error) + "\n")
    fire("StopFailure", error="authentication_failed", message="API Error: Invalid key")
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    assert [o.kwargs["status_message"] for o in errors] == [ERROR_TEXT, "API Error: Invalid key"]


def test_a_transcript_error_without_an_error_code_is_not_duplicated(
    failed_session, fake_langfuse,
):
    transcript, fire = failed_session
    rows = [json.loads(line) for line in transcript.read_text().splitlines()]
    del rows[-1]["error"]
    transcript.write_text("".join(json.dumps(row) + "\n" for row in rows))
    fire("StopFailure")
    fire("SessionEnd")
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    assert len(errors) == 1


def test_a_distinct_failure_after_retry_is_not_hidden_by_the_previous_error(
    failed_session, fake_langfuse,
):
    _, fire = failed_session
    fire("StopFailure")
    fire("StopFailure", error="authentication_failed", message="API Error: Invalid key")
    fire("StopFailure", error="authentication_failed", message="API Error: Invalid key")
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    assert [o.kwargs["status_message"] for o in errors] == [ERROR_TEXT, "API Error: Invalid key"]
    assert len([o for o in fake_langfuse.observations if o.name == "Conversational Turn"]) == 1


def test_repeated_failure_payload_with_whitespace_does_not_duplicate_the_error(
    failed_session, fake_langfuse,
):
    _, fire = failed_session
    for _ in range(3):
        fire("StopFailure", message=f"  {ERROR_TEXT}\n")
    fire("SessionEnd")
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    assert len(errors) == 1


def test_delayed_assistant_rows_stay_before_the_actual_failure(
    failed_session, fake_langfuse, isolated_hook_state,
):
    transcript, fire = failed_session
    rows = [json.loads(line) for line in transcript.read_text().splitlines()]
    error_row = rows.pop()
    transcript.write_text("".join(json.dumps(row) + "\n" for row in rows))
    fire("StopFailure")
    preceding = {
        "type": "assistant", "uuid": "preceding-error", "timestamp": "2026-01-01T00:03:03.900Z",
        "message": {"id": "preceding-error-message", "role": "assistant",
                    "content": "Preparing the next step."},
    }
    with transcript.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(preceding) + "\n")
        stream.write(json.dumps(error_row) + "\n")
    fire("StopFailure")
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    assert len(errors) == 1
    state = json.loads((isolated_hook_state / "langfuse_state.json").read_text())
    held = next(iter(state.values()))["open_turn"]["rows"]
    assert held[-2]["uuid"] == "preceding-error"
    assert held[-1]["isApiErrorMessage"] is True


@pytest.mark.parametrize("prompt_id", [None, "prompt-1"])
def test_delayed_prefix_in_a_separate_batch_does_not_repeat_the_failure(
    failed_session, fake_langfuse, prompt_id,
):
    transcript, fire = failed_session
    rows = [json.loads(line) for line in transcript.read_text().splitlines()]
    actual_error = rows.pop()
    transcript.write_text("".join(json.dumps(row) + "\n" for row in rows))
    fire("StopFailure", prompt_id=prompt_id)
    with transcript.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "type": "assistant", "uuid": "delayed-prefix",
            "timestamp": "2026-01-01T00:03:03.900Z",
            "message": {"id": "delayed-prefix-message", "role": "assistant",
                        "content": "Preparing the next step."},
        }) + "\n")
    for _ in range(2):
        fire("StopFailure", prompt_id=prompt_id)
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    assert len(errors) == 1
    with transcript.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(actual_error) + "\n")
    fire("StopFailure", prompt_id=prompt_id)
    assert len([o for o in fake_langfuse.observations
                if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]) == 1


@pytest.mark.parametrize("delayed_transcript", [False, True])
def test_identical_failure_from_a_new_prompt_is_exported_immediately(
    failed_session, fake_langfuse, delayed_transcript,
):
    transcript, fire = failed_session
    if delayed_transcript:
        rows = [json.loads(line) for line in transcript.read_text().splitlines()][:-1]
        transcript.write_text("".join(json.dumps(row) + "\n" for row in rows))
    fire("StopFailure", prompt_id="prompt-1")
    fire("StopFailure", prompt_id="prompt-2")
    fire("StopFailure", prompt_id="prompt-2")
    assert len([o for o in fake_langfuse.observations
                if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]) == 2


def test_retry_after_a_recorded_error_exports_the_same_failure_again(
    failed_session, fake_langfuse,
):
    transcript, fire = failed_session
    fire("StopFailure")
    with transcript.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "type": "assistant", "uuid": "retry-after-error",
            "message": {"id": "retry-after-error-message", "role": "assistant",
                        "content": "Retrying."},
        }) + "\n")
    fire("StopFailure")
    fire("StopFailure")
    assert len([o for o in fake_langfuse.observations
                if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]) == 2


def test_repeated_identical_failures_match_their_delayed_rows_in_order(
    failed_session, fake_langfuse, isolated_hook_state,
):
    transcript, fire = failed_session
    rows = [json.loads(line) for line in transcript.read_text().splitlines()]
    first_error = rows.pop()
    transcript.write_text("".join(json.dumps(row) + "\n" for row in rows))
    fire("StopFailure")
    retry = {
        "type": "assistant", "uuid": "retry", "timestamp": "2026-01-01T00:03:05.000Z",
        "message": {"id": "retry-message", "role": "assistant", "content": "Retrying."},
    }
    with transcript.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(retry) + "\n")
    fire("StopFailure")
    second_error = dict(first_error, uuid="api-error-2", timestamp="2026-01-01T00:03:06.000Z",
                        message=dict(first_error["message"], id="error-message-2"))
    with transcript.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(first_error) + "\n")
        stream.write(json.dumps(second_error) + "\n")
    fire("StopFailure")
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    assert len(errors) == 2
    state = json.loads((isolated_hook_state / "langfuse_state.json").read_text())
    held = next(iter(state.values()))["open_turn"]["rows"]
    assert [row["timestamp"] for row in held if row.get("isApiErrorMessage")] == [
        "2026-01-01T00:03:04.000Z", "2026-01-01T00:03:06.000Z",
    ]


@pytest.mark.parametrize("has_assistant", [False, True])
def test_failure_payload_covers_a_transcript_that_has_not_written_the_error(
    failed_session, fake_langfuse, has_assistant,
):
    transcript, fire = failed_session
    rows = [json.loads(line) for line in transcript.read_text().splitlines()]
    error_row = rows.pop()
    if not has_assistant:
        rows = [row for row in rows if row.get("type") != "assistant"
                and row.get("uuid") != "tool-result-agent-deferred"]
    transcript.write_text("".join(json.dumps(row) + "\n" for row in rows))

    fire("StopFailure")
    root = next(o for o in fake_langfuse.observations if o.name == "Conversational Turn")
    assert root.kwargs["level"] == "ERROR"
    assert root.output == {"role": "assistant", "content": ERROR_TEXT}
    count = len(fake_langfuse.observations)
    fire("StopFailure")
    assert len(fake_langfuse.observations) == count

    with transcript.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "type": "user", "uuid": "injected-context", "isMeta": True,
            "message": {"role": "user", "content": "Additional context"},
        }) + "\n")
        stream.write(json.dumps(error_row) + "\n")
    fire("StopFailure")
    fire("SessionEnd")
    assert len([o for o in fake_langfuse.observations if o.name == "Conversational Turn"]) == 1
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    assert len(errors) == 1


@pytest.mark.parametrize("nested", [False, True])
def test_failure_progress_survives_until_late_agent_completion(
    failed_session, fake_langfuse, isolated_hook_state, fixture_transcript_path, nested,
):
    transcript, fire = failed_session
    fire("StopFailure")
    assert any(o.kwargs.get("level") == "ERROR" for o in fake_langfuse.observations)
    count_after_failure = len(fake_langfuse.observations)
    assert not any(o.name.startswith("Subagent") for o in fake_langfuse.observations)
    fire("StopFailure")
    assert len(fake_langfuse.observations) == count_after_failure

    state = json.loads((isolated_hook_state / "langfuse_state.json").read_text())
    entry = next(iter(state.values()))
    assert entry["open_turn"]["rows"]
    assert entry["turn_progress"]
    progress = next(iter(entry["turn_progress"].values()))
    root = next(o for o in fake_langfuse.observations if o.name == "Conversational Turn")
    assert progress["root_span_id"] == root.id
    assert progress["trace_id"] == root.trace_id

    notification = (
        "<task-notification><task-id>agent-deferred</task-id>"
        "<tool-use-id>toolu_agent_deferred</tool-use-id>"
        "<status>completed</status><result>Research finished.</result></task-notification>"
    )
    with transcript.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "type": "user", "uuid": "notification-1",
            "timestamp": "2026-01-01T00:04:00.000Z",
            "origin": {"kind": "task-notification"},
            "message": {"role": "user", "content": notification},
        }) + "\n")
    agent_transcript = transcript.with_suffix("") / "subagents" / "agent-agent-deferred.jsonl"
    if nested:
        nested_agents = fixture_transcript_path("nested_subagents").with_suffix("") / "subagents"
        for filename in ("agent-inner-agent.meta.json", "agent-inner-agent.jsonl"):
            shutil.copyfile(nested_agents / filename, agent_transcript.parent / filename)
        with agent_transcript.open("a", encoding="utf-8") as stream:
            stream.write((nested_agents / "agent-outer-agent.jsonl").read_text())
    with agent_transcript.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "type": "assistant", "uuid": "subagent-final",
            "timestamp": "2026-01-01T00:06:00.000Z",
            "message": {"id": "subagent-final-message", "role": "assistant",
                        "model": "claude-test", "content": "Research finished."},
        }) + "\n")
    fire("Stop")

    roots = [o for o in fake_langfuse.observations if o.name == "Conversational Turn"]
    assert len(roots) == 1
    errors = [o for o in fake_langfuse.observations
              if o.as_type == "generation" and o.kwargs.get("level") == "ERROR"]
    assert len(errors) == 1
    agent = next(o for o in fake_langfuse.observations if o.name == "Subagent: Long research")
    assert agent.output == {"role": "assistant", "content": "Research finished."}
    parent_context = agent._otel_span.parent.get_span_context()
    assert parent_context.trace_id == int(root.trace_id, 16)
    assert parent_context.span_id == int(root.id, 16)
    assert agent._otel_span.attributes["langfuse.internal.is_app_root"] is False
    assert not agent._otel_span.attributes.get("langfuse.internal.as_root", False)
    agent_generations = [o for o in fake_langfuse.observations if o.name == "Subagent LLM Call"]
    assert agent_generations
    if nested:
        inner = next(o for o in fake_langfuse.observations if o.name == "Subagent: Inner agent")
        assert inner._otel_span.parent is agent._otel_span
        assert inner._otel_span.attributes["langfuse.internal.is_app_root"] is False
        assert not inner._otel_span.attributes.get("langfuse.internal.as_root", False)
        assert inner.output == {"role": "assistant", "content": "Inner result."}
        assert any(o._otel_span.parent is inner._otel_span for o in agent_generations)
    else:
        assert all(o._otel_span.parent is agent._otel_span for o in agent_generations)
    assert all(o._otel_span.attributes["langfuse.internal.is_app_root"] is False
               for o in agent_generations)
    observation_count = len(fake_langfuse.observations)
    fire("SessionEnd")
    assert len(fake_langfuse.observations) == observation_count
    state = json.loads((isolated_hook_state / "langfuse_state.json").read_text())
    entry = next(iter(state.values()))
    assert entry["turn_count"] == 1
    assert entry["open_turn"] == {}
    assert entry["turn_progress"] == {}


def test_plugin_registers_failure_export():
    repo = Path(__file__).resolve().parents[2]
    hooks = json.loads((repo / "hooks" / "hooks.json").read_text())["hooks"]
    assert "StopFailure" in hooks
    assert hooks["StopFailure"] == hooks["Stop"]
