from __future__ import annotations

import json
import shutil

import pytest


@pytest.fixture(autouse=True)
def bound_recursive_regressions(fake_langfuse, monkeypatch):
    create = fake_langfuse._create_observation_from_otel_span

    def bounded_create(**kwargs):
        if len(fake_langfuse.observations) >= 50:
            raise AssertionError("Agent observations repeat instead of completing")
        return create(**kwargs)

    monkeypatch.setattr(fake_langfuse, "_create_observation_from_otel_span", bounded_create)


def test_forked_agent_emits_its_work_once_beneath_its_parent(
    hook_module, fake_langfuse, fixture_transcript_path, tmp_path,
):
    source = fixture_transcript_path("nested_subagents")
    transcript = tmp_path / "transcript.jsonl"
    shutil.copyfile(source, transcript)
    agents = transcript.with_suffix("") / "subagents"
    shutil.copytree(source.with_suffix("") / "subagents", agents)
    parent_launch = source.read_text().splitlines()[:6]
    outer = agents / "agent-outer-agent.jsonl"
    outer_own = outer.read_text()
    inner = agents / "agent-inner-agent.jsonl"
    inner_own = inner.read_text()
    outer.write_text("\n".join(parent_launch) + "\n" + outer_own)
    inner.write_text("\n".join(parent_launch) + "\n"
                     + "\n".join(outer_own.splitlines()[:2]) + "\n" + inner_own)
    mapping = hook_module.get_subagent_transcripts_by_tool_use_id(transcript)
    rows = [json.loads(line) for line in transcript.read_text().splitlines()]
    turn = hook_module.build_turns(rows)[0]

    hook_module.emit_turn(fake_langfuse, "session-nested", 1, turn, transcript,
                          subagent_transcripts_by_tool_use_id=mapping)

    observations = fake_langfuse.observations
    roots = [o for o in observations if o.name == "Conversational Turn"]
    outer_spans = [o for o in observations if o.name == "Subagent: Outer agent"]
    inner_spans = [o for o in observations if o.name == "Subagent: Inner agent"]
    assert len(roots) == len(outer_spans) == len(inner_spans) == 1
    assert outer_spans[0]._otel_span.parent is roots[0]._otel_span
    assert inner_spans[0]._otel_span.parent is outer_spans[0]._otel_span
    assert len([o for o in observations if o.as_type == "generation"]) == 5
    assert all(o.end_time is not None for o in observations)
    inner_generation = next(o for o in observations
                            if o._otel_span.parent is inner_spans[0]._otel_span)
    assert inner_generation.kwargs["input"][-1] == {
        "role": "user", "content": "Return inner result",
    }
    assert any(m.get("tool_calls") for m in inner_generation.kwargs["input"])


@pytest.mark.parametrize("alias", [False, True])
def test_cyclic_agent_metadata_cannot_reenter_an_ancestor(
    hook_module, fake_langfuse, tmp_path, alias,
):
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    for index, path in enumerate(paths):
        rows = [
            {"type": "user", "uuid": f"u{index}", "timestamp": "2026-01-01T00:00:00Z",
             "message": {"role": "user", "content": "Work"}},
            {"type": "assistant", "uuid": f"a{index}", "timestamp": "2026-01-01T00:00:01Z",
             "message": {"role": "assistant", "id": f"m{index}", "content": [
                 {"type": "tool_use", "id": f"launch-{1-index}", "name": "Agent",
                  "input": {"description": "Next"}},
             ]}},
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    target = paths[0]
    if alias:
        target = tmp_path / "alias.jsonl"
        target.symlink_to(paths[0])
    mapping = {"launch-0": {"path": target, "description": "A"},
               "launch-1": {"path": paths[1], "description": "B"}}

    hook_module.emit_subagent_observations(
        fake_langfuse, None, {"path": paths[0], "description": "A"}, None,
        subagent_transcripts_by_tool_use_id=mapping,
    )

    assert [o.name for o in fake_langfuse.observations if o.name.startswith("Subagent:")] == [
        "Subagent: A", "Subagent: B",
    ]
    assert all(o.end_time is not None for o in fake_langfuse.observations)


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("startup_result_only", [False, True])
@pytest.mark.parametrize("top_level_content", [False, True])
def test_fork_startup_tool_result_does_not_hide_its_work(
    hook_module, fake_langfuse, fixture_transcript_path, tmp_path, failure, startup_result_only,
    top_level_content,
):
    source = fixture_transcript_path("nested_subagents")
    transcript = tmp_path / "transcript.jsonl"
    shutil.copyfile(source, transcript)
    agents = transcript.with_suffix("") / "subagents"
    shutil.copytree(source.with_suffix("") / "subagents", agents)
    outer = agents / "agent-outer-agent.jsonl"
    outer_rows = [json.loads(s) for s in outer.read_text().split("\n") if s]
    # Actual Claude forks start with the inherited launch's tool_result and
    # their directive in one user message, rather than a text-only prompt.
    inner = agents / "agent-inner-agent.jsonl"
    inner_rows = [json.loads(s) for s in inner.read_text().split("\n") if s]
    startup = inner_rows[0]
    startup["message"]["content"] = [
        {"type": "tool_result", "tool_use_id": "toolu_inner_agent",
         "content": [{"type": "text", "text": "Fork started — processing in background"}]},
    ]
    if not startup_result_only:
        startup["message"]["content"].append({"type": "text", "text": "Return inner result"})
    if top_level_content:
        startup["content"] = startup.pop("message")["content"]
    inherited_rows = [json.loads(s) for s in transcript.read_text().split("\n") if s][:3] + outer_rows[:2]
    inner.write_text("".join(json.dumps(r) + "\n" for r in inherited_rows + inner_rows))
    if failure:
        with transcript.open("a") as stream:
            stream.write(json.dumps({
                "type": "assistant", "uuid": "limit-error", "error": "rate_limit",
                "isApiErrorMessage": True, "timestamp": "2026-01-01T00:05:08.000Z",
                "message": {"id": "limit-error", "role": "assistant", "model": "<synthetic>",
                            "content": "You've hit your org's monthly spend limit"},
            }) + "\n")
    mapping = hook_module.get_subagent_transcripts_by_tool_use_id(transcript)
    turn = hook_module.build_turns(hook_module.read_subagent_jsonl(transcript))[0]

    progress = hook_module.emit_turn(
        fake_langfuse, "session-nested", 1, turn, transcript,
        subagent_transcripts_by_tool_use_id=mapping, close=not failure,
    )

    observations = fake_langfuse.observations
    root = next(o for o in observations if o.name == "Conversational Turn")
    outer_span = next(o for o in observations if o.name == "Subagent: Outer agent")
    inner_spans = [o for o in observations if o.name == "Subagent: Inner agent"]
    assert len(inner_spans) == 1
    inner_span = inner_spans[0]
    assert outer_span._otel_span.parent is root._otel_span
    assert inner_span._otel_span.parent is outer_span._otel_span
    assert inner_span._otel_span.attributes["langfuse.internal.is_app_root"] is False
    assert inner_span.output == {"role": "assistant", "content": "Inner result."}
    assert len([o for o in observations if o.as_type == "generation"]) == 5 + failure
    generation = next(o for o in observations
                      if o.as_type == "generation" and o._otel_span.parent is inner_span._otel_span)
    assert generation.kwargs["input"][-1] == {"role": "user", "content": "Return inner result"}
    # The launch result remains in the inherited context, once.
    launch_results = [r for m in generation.kwargs["input"] for r in m.get("tool_results", [])
                      if r.get("tool_use_id") == "toolu_inner_agent"]
    assert len(launch_results) == 1
    count = len(observations)
    hook_module.emit_turn(fake_langfuse, "session-nested", 1, turn, transcript,
                          subagent_transcripts_by_tool_use_id=mapping, progress=progress)
    assert len(observations) == count
