"""Fixed-vocabulary diagnostics for the shared Codex/xAI refresh seam.

Resolver invocation and force intent do not establish that a wire refresh occurred.
Provider bodies, credential values, arbitrary exception codes and exception text
must not enter this record.
"""
from typing import Any

from hermes_cli.auth_constants import AuthError

_SAFE_CODES = frozenset({
    "codex_auth_missing", "codex_auth_invalid_shape", "codex_auth_missing_access_token",
    "codex_auth_missing_refresh_token", "codex_refresh_failed", "codex_refresh_invalid_json",
    "codex_refresh_invalid_response", "codex_refresh_missing_access_token",
    "invalid_grant", "invalid_token", "invalid_request", "refresh_token_reused",
    "codex_rate_limited", "auth_store_lock_timeout",
})


def refresh_outcome(reason: str, *, resolver_invoked: bool = False,
                    force_requested: bool = False, exc: Exception | None = None) -> dict:
    outcome = {"outcome": reason, "resolver_invoked": resolver_invoked,
               "force_requested": force_requested, "code": None, "relogin_required": None}
    if isinstance(exc, AuthError):
        outcome["code"] = exc.code if isinstance(exc.code, str) and exc.code in _SAFE_CODES else "unknown_auth_error"
        outcome["relogin_required"] = exc.relogin_required is True
    return outcome


def codex_auth_failure(exc: Any, refresh: dict | None = None, *, fallback_configured: bool = False,
                       fallback_attempted: bool = False) -> dict:
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = error.get("code") if isinstance(error, dict) else None
    # An expired access token says nothing about the subscription/account or refresh grant.
    expired = isinstance(code, str) and code in {"token_expired", "access_token_expired"} or (
        "provided authentication token is expired" in str(exc).lower()
    )
    account = isinstance(code, str) and code in {"account_deactivated", "account_suspended", "account_rejected"}
    return {
        "provider": "openai-codex",
        "kind": "account_rejected" if account else "access_token_expired" if expired else "credential_rejected",
        "refresh": refresh or refresh_outcome("not_attempted"),
        "fallback_configured": bool(fallback_configured),
        "fallback_attempted": bool(fallback_attempted),
        "fallback_exhausted": bool(fallback_configured and fallback_attempted),
    }


def codex_auth_notice(failure: dict) -> str:
    leads = {
        "access_token_expired": "Codex's access token expired, so this turn stopped.",
        "account_rejected": "Codex rejected access to this account, so this turn stopped.",
        "credential_rejected": "Codex rejected the active credential, so this turn stopped.",
    }
    from agent.turn_failure_copy import oauth_relogin_command

    refresh = failure.get("refresh") or {}
    command = oauth_relogin_command("openai-codex")
    if refresh.get("relogin_required") is True:
        guidance = (
            "Credential recovery requires a new Codex sign-in. Ask the operator to use the secure "
            "Codex OAuth flow for this Hermes profile. Do not share sign-in codes or tokens in chat."
        )
    else:
        guidance = "Credential recovery did not restore access. Ask the operator to check Codex authentication for this Hermes profile."
    return (f"{leads.get(failure.get('kind'), leads['credential_rejected'])} {guidance} "
            f"If a new sign-in is needed, the operator can use the protected profile-scoped flow: `{command}`. "
            "Nothing will be replayed automatically.")
