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
