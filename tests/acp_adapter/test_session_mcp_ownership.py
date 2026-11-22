"""The gateway owns each session's editor MCP servers as session state (#134889 ACP-8/ACP-10).

A changed editor list re-freezes through one owner path (equal schemas keep the manifest), and a
viewed session keeps its connections between executions instead of respawning them per turn.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway.session_policy import policy_scope
from persistence.core.errors import AuthorityRefusal
from tests.acp_adapter.test_session_mcp_policy import owner, policy
from tests.tools.test_mcp_multiplex_connection_keys import _server as shared_server, two_profiles


def _editor_servers():
    from tools.mcp_tool import _servers
    return {key: server for key, server in _servers.items() if key[-1].startswith('editor_')}


def test_changed_credentials_with_equal_schemas_keep_the_manifest(server_spec, tmp_path):
    from gateway.session_local_mcp import attach_editor_mcp, rebind_editor_mcp
    from tools.registry import registry
    authority = owner()
    before = policy(authority, 'rebind', server_spec, tmp_path)
    changed = json.loads(json.dumps(server_spec))
    changed['env'][0]['value'] = 'PRIVATE_B'
    after, tools_changed = rebind_editor_mcp(authority, 'rebind', before, [changed])
    old, new = json.loads(before.editor_mcp_json), json.loads(after.editor_mcp_json)
    # The prompt-cache prefix (tool manifest + toolsets) is byte-stable; only the fingerprint moved.
    assert not tools_changed
    assert new['schemas'] == old['schemas'] and new['names'] == old['names'] and new['scope'] == old['scope']
    assert after.toolsets == before.toolsets
    assert new['fingerprint'] != old['fingerprint']
    assert 'PRIVATE_B' not in after.editor_mcp_json
    attach_editor_mcp(authority, after, [changed])
    with policy_scope(after, authority=authority):
        names = [n for n in registry.get_all_tool_names() if n.startswith('mcp__')]
        assert 'PRIVATE_B' in registry.dispatch(names[0], {})


def test_changed_schema_is_reported_as_a_manifest_change(server_spec, tmp_path):
    from gateway.session_local_mcp import rebind_editor_mcp
    authority = owner()
    before = policy(authority, 'schema', server_spec, tmp_path)
    script = tmp_path / 'owned_mcp.py'
    script.write_text(script.read_text().replace('owned echo', 'renamed echo'))
    after, tools_changed = rebind_editor_mcp(authority, 'schema', before, [server_spec])
    assert tools_changed
    schemas = json.loads(after.editor_mcp_json)['schemas']
    assert 'renamed echo' in json.dumps(schemas)
    assert after.toolsets == before.toolsets  # same server, same ledger name


def test_viewed_session_keeps_editor_connections_between_executions(server_spec, tmp_path):
    from gateway.session_local_mcp import release_editor_mcp
    authority = owner()
    live = SimpleNamespace(subscribers={'editor': object()})
    authority.sessions = {'viewed': live}
    p = policy(authority, 'viewed', server_spec, tmp_path)
    with policy_scope(p, authority=authority):
        first = _editor_servers()
    assert first, 'a viewed session keeps its editor servers connected after the execution'
    with policy_scope(p, authority=authority):
        assert _editor_servers() == first  # the same connection objects: no respawn per turn
    # Another session with the same editor config never adopts these connections.
    authority.sessions['other'] = SimpleNamespace(subscribers={'editor': object()})
    other = policy(authority, 'other', server_spec, tmp_path)
    with policy_scope(other, authority=authority):
        assert set(_editor_servers()) - set(first)
    live.subscribers.clear()
    authority.sessions['other'].subscribers.clear()
    release_editor_mcp(authority, ['viewed', 'other'])
    assert not _editor_servers()


def test_unviewed_session_releases_editor_connections_after_execution(server_spec, tmp_path):
    authority = owner()
    authority.sessions = {'quiet': SimpleNamespace(subscribers={})}
    p = policy(authority, 'quiet', server_spec, tmp_path)
    with policy_scope(p, authority=authority):
        assert _editor_servers()
    assert not _editor_servers()


def test_profile_mcp_servers_remain_callable_with_editor_servers(server_spec, tmp_path, monkeypatch):
    """An editor server must not disconnect the profile's already-live MCP tools."""
    from tools.mcp_tool_discovery import register_mcp_servers
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.registry import registry
    import tools.mcp_tool_config as config_module

    authority = owner()
    profile_spec = dict(server_spec, name='configured-global')
    profile_config = {
        'command': profile_spec['command'],
        'args': profile_spec['args'],
        'env': {'BORROWED_CREDENTIAL': 'PRIVATE_GLOBAL'},
        'trust': 'full',
    }
    monkeypatch.setattr(config_module, '_load_mcp_config', lambda: {'configured-global': profile_config})
    # The ACP overlay has no discovery record on the unfixed path. Its trust gate must therefore
    # resolve this profile's real config and reach connection resolution, not stop at fail-closed
    # "server not in config" before exercising the regression.
    from hermes_constants import get_hermes_home
    (get_hermes_home() / 'config.yaml').write_text(json.dumps({'mcp_servers': {
        'configured-global': profile_config,
    }}))
    try:
        register_mcp_servers({'configured-global': profile_config})
        assert 'PRIVATE_GLOBAL' in registry.dispatch('mcp__configured_global__echo', {})
        p = policy(authority, 'mixed', server_spec, tmp_path)
        with policy_scope(p, authority=authority):
            assert 'PRIVATE_GLOBAL' in registry.dispatch('mcp__configured_global__echo', {})
    finally:
        shutdown_mcp_servers()


def test_acp_shared_connection_uses_consuming_profile_trust_and_cost(
        server_spec, tmp_path, monkeypatch, two_profiles):
    """An ACP overlay on B inherits B's policy, never owner A's elevated trust or cost table."""
    from tools import approval_prompt, mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_registration as reg
    from tools.mcp_tool_costs import config_cost_policy
    from tools.mcp_tool_scope import _server_key
    from tools.registry import registry
    from gateway.session_local_mcp import session_scope

    route = {'url': 'https://mcp.example/shared', 'headers': {'Authorization': 'Bearer shared'}}
    config_a = dict(route, trust='full', tool_costs={'t': 11}, cost_unit='owner-credits')
    config_b = dict(route, trust='untrusted', tool_costs={'t': 3}, spend_limit=5,
                    cost_unit='borrower-credits')
    asked = []
    monkeypatch.setattr(approval_prompt, 'request_elicitation_consent',
                        lambda *args, **kwargs: asked.append(args) or 'deny')
    monkeypatch.setattr(core, '_server_cost_policies', {})

    scope_a = two_profiles('a')
    server_a = shared_server('shared', config_a)
    disc._adopt_server('shared', server_a)
    server_a._registered_tool_names = reg._register_server_tools('shared', server_a, config_a)

    scope_b = two_profiles('b')
    assert reg.register_connected_into_current_scope({'shared': config_b}) == 1
    authority = owner()
    with disc._owner_secret_scope():
        p = policy(authority, 'shared-policy', server_spec, tmp_path)
        with policy_scope(p, authority=authority, home=scope_b):
            scope = session_scope(authority, 'shared-policy')
            policy_key = _server_key('shared', scope, current=False)
            assert core._server_trust_levels[policy_key] == core._TRUST_UNTRUSTED
            assert core._server_cost_policies[policy_key] == config_cost_policy('shared', config_b)
            result = json.loads(registry.dispatch('mcp__shared__t', {}))
            assert "untrusted server 'shared'" in result['error']
            assert 'NOT run' in result['error']
            assert len(asked) == 1

    # Releasing the ACP overlay leaves the owner connection and both profiles' policies intact.
    assert core._servers[_server_key('shared', scope_a, current=False)] is server_a
    assert core._server_trust_levels[_server_key('shared', scope_a, current=False)] == core._TRUST_FULL
    assert core._server_cost_policies[_server_key('shared', scope_a, current=False)] == \
        config_cost_policy('shared', config_a)
    assert core._server_cost_policies[_server_key('shared', scope_b, current=False)] == \
        config_cost_policy('shared', config_b)


def test_editor_scopes_are_per_session_and_per_profile():
    from gateway.session_local_mcp import session_scope
    one, other = owner(), SimpleNamespace(profile_id='other-profile')
    assert session_scope(one, 'parent') != session_scope(one, 'child')
    assert session_scope(one, 'parent') != session_scope(other, 'parent')


@pytest.mark.parametrize(('actor_profile', 'capabilities', 'reason'), [
    ('owned-profile', {'session:read', 'session:submit'}, 'permission_denied'),
    ('foreign-profile', {'session:read', 'session:submit', 'session:control'}, 'profile_mismatch'),
])
def test_editor_mcp_change_requires_control_in_the_owning_profile(server_spec, actor_profile, capabilities, reason):
    from gateway.session_contract import SessionActor, SessionRef
    from gateway.session_mutations import _mutate_session
    authority = owner()
    actor = SessionActor('viewer', actor_profile, frozenset(capabilities), 'viewer-transport')
    params = {'session_id': 'held', 'request_id': 'r', 'expected_revision': 0, 'expected_generation': 0,
              'operation': 'editor_mcp', 'payload': {'mcp_servers': [server_spec]}}
    with pytest.raises(AuthorityRefusal, match=reason):
        asyncio.run(_mutate_session(authority, actor, SessionRef('owned-profile', 'held'), params))
    assert not _editor_servers()  # refused before any discovery


def test_store_payload_carries_only_the_fingerprint(server_spec):
    from gateway.session_local_mcp import fingerprint, private_mutation_payload
    from persistence.authority.runtime import parse_mutation
    payload, servers = private_mutation_payload('editor_mcp', {'mcp_servers': [server_spec]})
    assert servers == [server_spec]
    assert payload == {'mcp_servers': {'fingerprint': fingerprint([server_spec])}}
    assert 'PRIVATE_A' not in json.dumps(payload)
    parse_mutation('editor_mcp', payload)
    with pytest.raises(AuthorityRefusal, match='invalid_params'):
        parse_mutation('editor_mcp', {'mcp_servers': [server_spec]})
    branch, _ = private_mutation_payload('branch', {'mcp_servers': [server_spec]})
    parse_mutation('branch', branch)
