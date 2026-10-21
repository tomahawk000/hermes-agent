"""Codex recovery preserves identity, bounds retries and exports only safe evidence."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest

from agent.client_lifecycle import ClientLifecycleMixin
from agent.codex_auth_recovery import codex_auth_failure, codex_auth_notice, refresh_outcome
from agent.error_surface import build_error_surface_from_result
from agent.turn_recovery import _refresh_credentials_after_401, nonretryable_client_error_result
from agent.turn_retry_state import TurnRetryState
from agent.error_classifier import FailoverReason
from gateway.config import Platform
from gateway.run import _prepare_gateway_status_message, _sanitize_gateway_final_response
from hermes_cli.auth_constants import AuthError


@pytest.mark.parametrize("case,reason,attempted,code,relogin", [
    ("success", "refreshed", True, None, None),
    ("identity", "identity_guard", False, None, None),
    ("missing", "credential_read_failed", False, "codex_auth_missing_refresh_token", True),
    ("revoked", "refresh_failed", True, "invalid_grant", True),
    ("rotated", "refresh_failed", True, "refresh_token_reused", True),
    ("transient", "refresh_failed", True, "codex_refresh_failed", False),
    ("unknown", "refresh_failed", True, None, None),
    ("unknown_code", "refresh_failed", True, "unknown_auth_error", False),
    ("unchanged", "unchanged_credentials", True, None, None),
    ("invalid", "invalid_credentials", True, None, None),
    ("rebuild", "client_rebuild_failed", True, None, None),
])
def test_refresh_outcome_is_bounded_and_secret_free(monkeypatch, caplog, case, reason, attempted, code, relogin):
    from hermes_cli import auth
    agent = SimpleNamespace(api_mode="codex_responses", provider="openai-codex", api_key="synthetic-old",
                            _buffer_vprint=Mock(), session_id="offline-session", _current_api_request_id="offline-turn:api:0")
    calls = []
    def resolve(**kwargs):
        calls.append(kwargs)
        if not kwargs.get("force_refresh"):
            if case == "missing":
                raise AuthError("PRIVATE-DIAGNOSTIC", code="codex_auth_missing_refresh_token", relogin_required=True)
            return {"api_key": "synthetic-other" if case == "identity" else "synthetic-old"}
        errors = {"revoked": ("invalid_grant", True), "rotated": ("refresh_token_reused", True),
                  "transient": ("codex_refresh_failed", False), "unknown_code": ("PRIVATE-DIAGNOSTIC", False)}
        if case in errors:
            raise AuthError("PRIVATE-DIAGNOSTIC", code=errors[case][0], relogin_required=errors[case][1])
        if case == "unknown":
            raise RuntimeError("PRIVATE-DIAGNOSTIC")
        return {"api_key": None if case == "invalid" else "synthetic-old" if case == "unchanged" else "synthetic-fresh",
                "base_url": "https://chatgpt.com/backend-api/codex"}
    monkeypatch.setattr(auth, "resolve_codex_runtime_credentials", resolve)
    agent._adopt_openai_credentials = Mock(return_value=case != "rebuild")
    agent._try_refresh_codex_client_credentials = lambda **kw: ClientLifecycleMixin._try_refresh_codex_client_credentials(agent, **kw)
    state = TurnRetryState()
    error = RuntimeError("Provided authentication token is expired")
    with caplog.at_level("INFO"):
        assert _refresh_credentials_after_401(agent, error, state, 401) is (case == "success")
        assert not _refresh_credentials_after_401(agent, error, state, 401)
    assert state.codex_refresh_outcome == {"resolver_invoked": attempted, "force_requested": attempted, "outcome": reason, "code": code, "relogin_required": relogin}
    assert sum(bool(c.get("force_refresh")) for c in calls) == int(attempted)
    assert agent.api_key == "synthetic-old"
    assert agent._adopt_openai_credentials.call_count == int(case in {"success", "rebuild"})
    assert "PRIVATE-DIAGNOSTIC" not in caplog.text and "synthetic-" not in caplog.text


@pytest.mark.parametrize("kind,body,text,status,refresh", [
    ("access_token_expired", {}, "Provided authentication token is expired PRIVATE-DIAGNOSTIC", 401, None),
    ("credential_rejected", {}, "401 PRIVATE-DIAGNOSTIC", 401, refresh_outcome("refresh_failed", resolver_invoked=True, force_requested=True)),
    ("account_rejected", {"error": {"code": "account_suspended"}}, "PRIVATE-DIAGNOSTIC", 403, None),
    ("access_token_expired", {"error": {"code": "access_token_expired"}}, "PRIVATE-DIAGNOSTIC", 401,
     refresh_outcome("refresh_failed", resolver_invoked=True, force_requested=True, exc=AuthError("PRIVATE-DIAGNOSTIC", code="invalid_grant", relogin_required=True))),
])
def test_codex_terminal_status_and_final_have_one_chat_owner(kind, body, text, status, refresh):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid"))
    exc = openai.AuthenticationError(text, response=response, body=body)
    agent = SimpleNamespace(_fallback_chain=[], log_prefix="", session_id="offline-session", _current_api_request_id="offline-turn",
        _flush_status_buffer=Mock(), _persist_session=Mock(), verbose_logging=False,
        _emit_status=Mock(), _emit_status_kind=Mock(), _summarize_api_error=Mock(side_effect=AssertionError("raw summary not allowed")))
    classified = SimpleNamespace(is_auth=True, reason=FailoverReason.auth, retryable=False)
    result = nonretryable_client_error_result(agent, exc, classified, status_code=status, api_kwargs=None,
        api_messages=[], messages=[{"role": "user", "content": "offline request"}], conversation_history=[],
        api_call_count=1, approx_tokens=1, provider="openai-codex", base_url="https://example.invalid", model="gpt-6-astra",
        retry_state=TurnRetryState(codex_refresh_outcome=refresh))
    assert result["auth_failure"]["kind"] == kind
    assert not result["auth_failure"]["fallback_configured"]
    assert result["failed"] and not result["completed"]
    surface = build_error_surface_from_result(result, provider="openai-codex", model="gpt-6-astra")
    assert surface["code"] == "codex_" + kind and surface["retryable"] is False
    assert surface["recovery"] == result["auth_failure"]
    agent._emit_status.assert_not_called()
    event, status_copy = agent._emit_status_kind.call_args.args
    final = _sanitize_gateway_final_response(Platform.TELEGRAM, result["final_response"])
    deliveries = [m for m in (_prepare_gateway_status_message(Platform.TELEGRAM, event, status_copy), final) if m]
    assert deliveries == [result["final_response"]]
    assert _prepare_gateway_status_message(Platform.LOCAL, event, status_copy) == status_copy
    assert _prepare_gateway_status_message(Platform.TELEGRAM, "lifecycle", "Working on your request") == "Working on your request"
    assert "PRIVATE-DIAGNOSTIC" not in json.dumps(result)
    assert "Nous" not in final and "portal" not in final and "/login" not in final
    assert "send it again" not in final and "/retry" not in final
    assert ("new Codex sign-in" in final) is bool(refresh and refresh["relogin_required"])
    assert codex_auth_notice(codex_auth_failure(exc))[0:5] == "Codex"
