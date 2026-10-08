import json

import pytest


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_subagent_reader_preserves_unicode_separators(hook_module, tmp_path, separator, newline):
    rows = [
        {"type": "user", "toolUseResult": {"stdout": f"compressed{separator}data"}},
        {"type": "assistant", "message": {"content": "next record"}},
    ]
    path = tmp_path / "agent.jsonl"
    path.write_bytes((newline.join(json.dumps(row, ensure_ascii=False) for row in rows)
                      + newline).encode("utf-8"))

    assert hook_module.read_subagent_jsonl(path) == rows


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_workflow_results_preserve_unicode_separators(hook_module, tmp_path, separator):
    result = {"output": f"compressed{separator}data"}
    rows = [
        {"type": "result", "agentId": "first", "result": result},
        {"type": "result", "agentId": "second", "result": "done"},
    ]
    (tmp_path / "journal.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    assert hook_module.get_workflow_journal_results(tmp_path) == {"first": result, "second": "done"}
