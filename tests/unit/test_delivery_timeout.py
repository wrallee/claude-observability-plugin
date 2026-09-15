from __future__ import annotations

from typing import Any


def test_client_uses_30_second_request_timeout_by_default(hook_module: Any, monkeypatch: Any) -> None:
    monkeypatch.delenv("LANGFUSE_TIMEOUT", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_LANGFUSE_TIMEOUT", raising=False)

    class RecordingLangfuse:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(hook_module, "Langfuse", RecordingLangfuse)
    config = hook_module.LangfuseConfig(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example.com",
        user_id=None,
    )

    client = hook_module.create_langfuse_client(config)

    assert client is not None
    assert client.kwargs["timeout"] == 30


def test_shutdown_uses_request_timeout_plus_grace_and_does_not_double_flush(
    hook_module: Any, monkeypatch: Any
) -> None:
    monkeypatch.delenv("LANGFUSE_TIMEOUT", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_LANGFUSE_TIMEOUT", raising=False)
    monkeypatch.delenv("CC_LANGFUSE_FLUSH_TIMEOUT", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_CC_LANGFUSE_FLUSH_TIMEOUT", raising=False)
    joined_with: list[float] = []

    class ImmediateThread:
        def __init__(self, *, target: Any, daemon: bool) -> None:
            assert daemon is True
            self.target = target

        def start(self) -> None:
            self.target()

        def join(self, timeout: float) -> None:
            joined_with.append(timeout)

        def is_alive(self) -> bool:
            return False

    class RecordingClient:
        def __init__(self) -> None:
            self.flush_calls = 0
            self.shutdown_calls = 0

        def flush(self) -> None:
            self.flush_calls += 1

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    monkeypatch.setattr(hook_module.threading, "Thread", ImmediateThread)
    client = RecordingClient()

    hook_module.flush_and_shutdown_langfuse_client(client)

    assert client.shutdown_calls == 1
    assert client.flush_calls == 0
    assert joined_with == [35.0]


def test_timeout_overrides_are_applied(hook_module: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("LANGFUSE_TIMEOUT", "45")
    monkeypatch.setenv("CC_LANGFUSE_FLUSH_TIMEOUT", "60")
    joined_with: list[float] = []

    class RecordingLangfuse:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class ImmediateThread:
        def __init__(self, *, target: Any, daemon: bool) -> None:
            self.target = target

        def start(self) -> None:
            self.target()

        def join(self, timeout: float) -> None:
            joined_with.append(timeout)

        def is_alive(self) -> bool:
            return False

    class Client:
        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(hook_module, "Langfuse", RecordingLangfuse)
    monkeypatch.setattr(hook_module.threading, "Thread", ImmediateThread)
    config = hook_module.LangfuseConfig(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example.com",
        user_id=None,
    )

    client = hook_module.create_langfuse_client(config)
    hook_module.flush_and_shutdown_langfuse_client(Client())

    assert client is not None
    assert client.kwargs["timeout"] == 45
    assert joined_with == [60.0]
