"""Refresh diagnostics disclose fixed outcomes without changing recovery decisions."""
import logging
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from agent.client_lifecycle import ClientLifecycleMixin
from hermes_cli.auth_constants import AuthError


@pytest.mark.parametrize("provider", ["openai-codex", "xai-oauth"])
@pytest.mark.parametrize("case", [
    "read_failure", "read_auth_failure", "identity_guard", "refresh_failure",
    "known_auth_failure", "unknown_auth_failure", "invalid_credentials",
    "unchanged_credentials", "rebuild_refused", "rebuild_exception", "refreshed", "not_forced",
])
def test_diagnostics_preserve_resolver_and_adoption_decisions(monkeypatch, caplog, provider, case):
    from hermes_cli import auth

    marker = "synthetic-private-error-detail"
    old = {"api_key": "fixture-old", "base_url": "https://example.invalid"}
    fresh = {"api_key": "fixture-new", "base_url": old["base_url"]}
    scenarios = {
        "read_failure": {"read": RuntimeError(marker), "outcome": "credential_read_failed"},
        "read_auth_failure": {"read": AuthError(marker, code="codex_auth_missing", relogin_required=True),
                              "outcome": "credential_read_failed", "code": "codex_auth_missing", "relogin": True},
        "identity_guard": {"read": {"api_key": "fixture-other"}, "outcome": "identity_guard"},
        "refresh_failure": {"refresh": RuntimeError(marker), "outcome": "refresh_failed"},
        "known_auth_failure": {"refresh": AuthError(marker, code="invalid_grant", relogin_required=True),
                               "outcome": "refresh_failed", "code": "invalid_grant", "relogin": True},
        "unknown_auth_failure": {"refresh": AuthError(marker, code=marker), "outcome": "refresh_failed",
                                 "code": "unknown_auth_error", "relogin": False},
        "invalid_credentials": {"refresh": {"api_key": None, "base_url": None}, "outcome": "invalid_credentials"},
        "unchanged_credentials": {"refresh": old, "outcome": "unchanged_credentials"},
        "rebuild_refused": {"outcome": "client_rebuild_failed"},
        "rebuild_exception": {"rebuild": RuntimeError(marker), "outcome": "client_rebuild_failed"},
        "refreshed": {}, "not_forced": {},
    }
    scenario = scenarios[case]
    outcome = scenario.get("outcome", "refreshed")
    rebuild_fault = scenario.get("rebuild")
    resolve = Mock(side_effect=[scenario.get("read", old), scenario.get("refresh", fresh)])
    unused = Mock(side_effect=AssertionError("wrong provider resolver"))
    monkeypatch.setattr(auth, "resolve_codex_runtime_credentials", resolve if provider == "openai-codex" else unused)
    monkeypatch.setattr(auth, "resolve_xai_oauth_runtime_credentials", resolve if provider == "xai-oauth" else unused)
    adopt = Mock(side_effect=rebuild_fault, return_value=case != "rebuild_refused")
    agent = SimpleNamespace(api_mode="codex_responses", provider=provider, api_key=old["api_key"],
                            _adopt_openai_credentials=adopt)
    caplog.set_level(logging.DEBUG, logger="run_agent")
    forced = case != "not_forced"
    if rebuild_fault:
        with pytest.raises(RuntimeError) as raised:
            ClientLifecycleMixin._try_refresh_codex_client_credentials(agent, force=forced)
        assert raised.value is rebuild_fault
    else:
        assert ClientLifecycleMixin._try_refresh_codex_client_credentials(agent, force=forced) is (outcome == "refreshed")

    invoked = case not in {"read_failure", "read_auth_failure", "identity_guard"}
    assert resolve.call_args_list == [call(refresh_if_expiring=False)] + ([call(force_refresh=forced)] if invoked else [])
    unused.assert_not_called()
    if outcome in {"refreshed", "client_rebuild_failed"}:
        adopt.assert_called_once_with("fixture-new", old["base_url"], reason=f"{provider}_credential_refresh")
    else:
        adopt.assert_not_called()
    expected = {"outcome": outcome, "resolver_invoked": invoked, "force_requested": invoked and forced,
                "code": scenario.get("code"), "relogin_required": scenario.get("relogin")}
    assert agent._codex_refresh_outcome == expected
    records = [r for r in caplog.records if r.name == "run_agent"]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert str(expected) in records[0].getMessage()
    assert not any(s in caplog.text for s in (marker, "fixture-old", "fixture-new", "fixture-other", old["base_url"]))
    assert not any(r.exc_info for r in records)


@pytest.mark.parametrize("provider,mode", [
    ("openrouter", "chat_completions"), ("openai-codex", "chat_completions"), ("xai-oauth", "chat_completions"),
])
def test_ineligible_route_does_not_resolve_or_adopt(monkeypatch, provider, mode):
    from hermes_cli import auth

    resolve = Mock(side_effect=AssertionError("ineligible resolver"))
    monkeypatch.setattr(auth, "resolve_codex_runtime_credentials", resolve)
    monkeypatch.setattr(auth, "resolve_xai_oauth_runtime_credentials", resolve)
    agent = SimpleNamespace(provider=provider, api_mode=mode, _adopt_openai_credentials=Mock())
    assert ClientLifecycleMixin._try_refresh_codex_client_credentials(agent) is False
    resolve.assert_not_called()
    agent._adopt_openai_credentials.assert_not_called()
    assert agent._codex_refresh_outcome == {
        "outcome": "route_ineligible", "resolver_invoked": False, "force_requested": False,
        "code": None, "relogin_required": None,
    }
