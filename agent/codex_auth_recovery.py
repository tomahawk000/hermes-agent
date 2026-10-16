"""Fixed-vocabulary diagnostics for the shared Codex/xAI refresh seam.

Resolver invocation and force intent do not establish that a wire refresh occurred.
Provider bodies, credential values, arbitrary exception codes and exception text
must not enter this record.
"""
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
