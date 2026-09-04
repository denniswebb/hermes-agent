"""MCP lifecycle isolation for duplicate logical names under multiplexing."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import tools.mcp_tool as mcp


class _AsyncLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _Server:
    def __init__(self, name, response):
        self.name = name
        self._tools = [
            SimpleNamespace(
                name="query",
                description="Query metrics",
                inputSchema={"type": "object", "properties": {}},
            )
        ]
        self.tool_timeout = 5
        self.initialize_result = None
        self._rpc_lock = _AsyncLock()
        self._inflight_tasks = set()
        self._pending_call_context = None
        self._registered_tool_names = []
        self.shutdown_calls = 0

        async def call_tool(_tool_name, *, arguments):
            return SimpleNamespace(
                isError=False,
                content=[SimpleNamespace(text=f"{response}:{arguments['query']}")],
                structuredContent=None,
            )

        self.session = SimpleNamespace(call_tool=call_tool)

    def _is_recycled_stdio(self):
        return False

    async def shutdown(self):
        self.shutdown_calls += 1


@contextmanager
def _profile_scope(home):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


@pytest.fixture
def multiplex_mcp_state(monkeypatch):
    """Give this test private MCP state without disturbing singleton tests."""
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from tools.registry import registry

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    for name, value in {
        "_servers": {},
        "_server_scope_keys": {},
        "_server_connecting": set(),
        "_server_connect_errors": {},
        "_lazy_server_configs": {},
        "_lazy_server_fingerprints": {},
        "_lazy_server_tool_names": {},
        "_server_connect_retry_after": {},
        "_server_connect_failures": {},
        "_server_error_counts": {},
        "_server_breaker_opened_at": {},
        "_server_trust_levels": {},
        "_tool_read_only_hints": {},
        "_parallel_safe_servers": set(),
        "_mcp_tool_server_names": {},
    }.items():
        monkeypatch.setattr(mcp, name, value)

    scoped_tools = {scope: dict(entries) for scope, entries in registry._scoped_tools.items()}
    monkeypatch.setattr(registry, "_scoped_tools", scoped_tools)
    try:
        yield
    finally:
        set_multiplex_active(previous_multiplex)
        mcp._stop_mcp_loop()


def test_same_logical_server_isolated_by_profile_scope(tmp_path, monkeypatch, multiplex_mcp_state):
    """Two profiles may independently own a same-named MCP connection/tool."""
    from hermes_constants import hermes_home_key
    from tools.registry import registry

    home_a = tmp_path / "platform"
    home_b = tmp_path / "sre"
    home_a.mkdir()
    home_b.mkdir()
    scope_a = hermes_home_key(home_a)
    scope_b = hermes_home_key(home_b)
    discoveries = []

    async def discover(name, config):
        response = config["headers"]["X-Profile"]
        server = _Server(name, response)
        discoveries.append((name, response))
        with mcp._lock:
            mcp._servers[name] = server
            mcp._server_scope_keys[name] = mcp._server_registry_scope(name)
        names = mcp._register_server_tools(name, server, config)
        server._registered_tool_names = names
        return names

    monkeypatch.setattr(mcp, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp, "_discover_and_register_server", discover)

    config_a = {
        "victoriametrics": {
            "url": "https://metrics.platform.example/mcp",
            "headers": {"X-Profile": "platform"},
            "supports_parallel_tool_calls": True,
        }
    }
    config_b = {
        "victoriametrics": {
            "url": "https://metrics.sre.example/mcp",
            "headers": {"X-Profile": "sre"},
            "supports_parallel_tool_calls": True,
        }
    }

    with _profile_scope(home_a):
        mcp.register_mcp_servers(config_a)
        key_a = mcp._server_identity("victoriametrics")
        # A failed profile A connection must not place profile B into cooldown.
        mcp._record_connect_failure(key_a)

    with _profile_scope(home_b):
        mcp.register_mcp_servers(config_b)
        key_b = mcp._server_identity("victoriametrics")

        assert key_a != key_b
        assert len(mcp._servers) == 2
        assert discoveries == [(key_a, "platform"), (key_b, "sre")]
        assert mcp._connect_cooldown_active(key_a) is True
        assert mcp._connect_cooldown_active(key_b) is False
        assert mcp.is_mcp_tool_parallel_safe("mcp__victoriametrics__query") is True
        # Capability-derived prompt logic must see only this profile's tools,
        # not a process-wide union from its sibling profile.
        mcp._track_mcp_tool_server(
            "mcp__sre_helper__query", mcp._server_identity("sre-helper")
        )
        assert mcp.get_registered_mcp_server_names() == {
            "victoriametrics", "sre-helper"
        }

        entry_b = registry.snapshot_registration(
            "mcp__victoriametrics__query", scope=scope_b
        )
        assert entry_b is not None
        assert "sre:up" in entry_b.handler({"query": "up"})

    with _profile_scope(home_a):
        assert mcp.is_mcp_tool_parallel_safe("mcp__victoriametrics__query") is True
        assert mcp.get_registered_mcp_server_names() == {"victoriametrics"}
        entry_a = registry.snapshot_registration(
            "mcp__victoriametrics__query", scope=scope_a
        )
        assert entry_a is not None
        assert "platform:up" in entry_a.handler({"query": "up"})

    mcp.shutdown_mcp_servers(scope=scope_a)
    assert key_a not in mcp._servers
    assert key_b in mcp._servers
    assert mcp._servers[key_b].shutdown_calls == 0
    assert key_a not in mcp._parallel_safe_servers
    assert key_b in mcp._parallel_safe_servers

    registry.deregister("mcp__victoriametrics__query", scope=scope_a)
    registry.deregister("mcp__victoriametrics__query", scope=scope_b)
