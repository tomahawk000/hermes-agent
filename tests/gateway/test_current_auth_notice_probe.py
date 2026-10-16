"""Offline counterexamples for contracts outside the diagnostics-only slice."""
import pytest

from gateway.run import GatewayRunner, _prepare_gateway_status_message
from tests.agent.test_failed_turn_chat_copy import _Agent, _nonretryable


@pytest.mark.parametrize("contract", ["single_terminal_producer", "no_generic_resend"])
def test_current_codex_terminal_notice_contract(contract):
    statuses = []

    class Recorder(_Agent):
        def _emit_diagnostic_status(self, message):
            statuses.append(("lifecycle", message))

        def _emit_status_kind(self, kind, message, *, origin):
            statuses.append((kind, message))

    result = _nonretryable(401, "HTTP 401: Provided authentication token is expired",
                          provider="openai-codex", model="fixture-model", agent=Recorder())
    runner = object.__new__(GatewayRunner)
    notice = runner._hmwa_failed_turn_notice(result)
    final = runner._hmwa_add_failed_turn_notice(result["final_response"], notice)
    forwarded = [_prepare_gateway_status_message("telegram", kind, s) for kind, s in statuses]
    print({"contract": contract, "terminal_status_producers": sum(bool(s) for s in forwarded),
           "final_producer": bool(final), "generic_resend_advice": "Send it again" in final})
    if contract == "single_terminal_producer":
        assert not any(forwarded), "Terminal auth status and final remain separate producers"
    else:
        assert "Send it again" not in final, "Codex failure still receives generic resend advice"
