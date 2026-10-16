"""Durable replay and turn admission checks on a single SQLite snapshot."""
import json

from agent.context_compressor import user_originated_turn_view


class CodexAuthResumeBlocked(ValueError):
    """A restart continuation cannot revive a terminal or uncertain Codex input."""


def resolve_turn_admission_target(conn, session_id):
    """Strict continuation resolution for execution, including inherited fork markers.

    UI resume resolution is best-effort; execution cannot fall back to a closed
    parent after a read failure. All route reads use the caller's transaction.
    """
    from hermes_state_common import _legacy_reset_child_sql, _RESET_END_REASONS_SQL, _sql_session_last_active

    current, best, seen = session_id, None, set()
    for _ in range(100):
        if current in seen:
            raise ValueError("turn admission continuation is cyclic")
        seen.add(current)
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (current,)).fetchone()
        if row is None:
            raise ValueError("turn admission continuation is unavailable")
        if conn.execute("SELECT 1 FROM messages WHERE session_id=? LIMIT 1", (current,)).fetchone():
            best = current
        compressed = row["end_reason"] == "compression"
        order = ("CASE WHEN child.end_reason='compression' THEN 0 WHEN child.ended_at IS NULL THEN 1 ELSE 2 END, "
                 f"{_sql_session_last_active('child')} DESC, ") if compressed else ""
        children = conn.execute(
            "SELECT child.* FROM sessions child WHERE child.parent_session_id=? "
            "AND COALESCE(child.source,'')!='tool' "
            f"AND NOT {_legacy_reset_child_sql('child', _RESET_END_REASONS_SQL)} "
            f"ORDER BY {order}child.started_at DESC, child.id DESC", (current,)).fetchall()
        target = None
        for child in children:
            try:
                cfg = json.loads(child["model_config"]) if child["model_config"] else {}
            except (ValueError, TypeError) as exc:
                raise ValueError("turn admission routing metadata is unreadable") from exc
            if not isinstance(cfg, dict):
                raise ValueError("turn admission routing metadata is invalid")
            # Markers bind to their original parent. Compression copies them;
            # inherited markers must not hide the continuation of a fork.
            if current not in (cfg.get("_branched_from"), cfg.get("_delegate_from"), cfg.get("_reset_from")):
                target = child["id"]
                break
        if target is None:
            if compressed:
                raise ValueError("turn admission compression continuation is unavailable")
            return best or session_id
        current = target
    raise ValueError("turn admission continuation exceeds the depth limit")


def clear_owned_codex_pending(db, conn, session_id, user, *, failed=False):
    """Resolve all active compaction copies of this exact owned input in the caller's transaction."""
    owner = (db._decode_display_metadata(user["display_metadata"]) or {}).get("gateway_input_owner")
    rows = conn.execute("SELECT id,display_metadata FROM messages WHERE session_id=? AND role='user' AND active=1",
                        (session_id,)).fetchall()
    for row in rows:
        meta = db._decode_display_metadata(row["display_metadata"]) or {}
        same_input = meta.get("gateway_input_owner") == owner if owner else row["id"] == user["id"]
        if same_input and (meta.get("codex_auth_pending") is True or failed):
            if meta.get("codex_auth_pending") is True:
                meta["codex_auth_pending"] = False
            if failed:
                meta["codex_auth_failure"] = True
            conn.execute("UPDATE messages SET display_metadata=? WHERE id=?",
                         (db._encode_display_metadata(meta), row["id"]))


def validate_history_auth_state(rows, *, reject_pending=False, reject_failure=False):
    """Check the fetched snapshot before projection/alternation can hide a pending input."""
    failed_turn = False
    for row in rows:
        if row["role"] == "user":
            failed_turn = False
        if row["display_metadata"]:
            try:
                meta = json.loads(row["display_metadata"])
            except (ValueError, TypeError) as exc:
                raise ValueError("retry safety metadata is unreadable") from exc
            if not isinstance(meta, dict) or any(
                key in meta and not isinstance(meta[key], bool)
                for key in ("codex_auth_failure", "codex_auth_pending")
            ):
                raise ValueError("retry safety metadata is invalid")
            if reject_pending and row["role"] == "user" and meta.get("codex_auth_pending") is True:
                error = CodexAuthResumeBlocked if reject_failure else ValueError
                raise error("Codex input has no durable completion or failure boundary")
            if meta.get("codex_auth_failure") is True:
                failed_turn = True
    if reject_failure and failed_turn:
        raise CodexAuthResumeBlocked("Automatic continuation refused after a recorded Codex auth failure")


def reject_archived_codex_auth_resume(conn, target, lease_key):
    """Inspect request metadata across this lease's compression lineage, never prompt content.

    Summary-only rows cannot erase an archived failed request. A newer real user
    request supersedes it. Rewind-discarded rows are deliberately excluded.
    """
    lineage, current = [], target
    for _ in range(100):
        if current in lineage:
            raise CodexAuthResumeBlocked("Automatic continuation lineage is cyclic")
        lineage.append(current)
        if current == lease_key:
            break
        row = conn.execute("SELECT parent_session_id FROM sessions WHERE id=?", (current,)).fetchone()
        if row is None or not row["parent_session_id"]:
            raise CodexAuthResumeBlocked("Automatic continuation lineage is unavailable")
        current = row["parent_session_id"]
    else:
        raise CodexAuthResumeBlocked("Automatic continuation lineage exceeds the depth limit")
    placeholders = ",".join("?" for _ in lineage)
    rows = conn.execute(
        "SELECT display_metadata,display_kind,_compressed_summary FROM messages WHERE "
        f"session_id IN ({placeholders}) AND role='user' AND (active=1 OR compacted=1) ORDER BY id DESC",
        lineage)
    for row in rows:
        try:
            meta = json.loads(row["display_metadata"]) if row["display_metadata"] else {}
        except (ValueError, TypeError) as exc:
            raise CodexAuthResumeBlocked("Automatic continuation metadata is unreadable") from exc
        if not isinstance(meta, dict) or any(
            key in meta and not isinstance(meta[key], bool)
            for key in ("codex_auth_pending", "codex_auth_failure")
        ):
            raise CodexAuthResumeBlocked("Automatic continuation metadata is invalid")
        if meta.get("codex_auth_pending") is True or meta.get("codex_auth_failure") is True:
            raise CodexAuthResumeBlocked("Automatic continuation refused after an uncertain or failed Codex input")
        owner = meta.get("gateway_input_owner")
        if (isinstance(owner, str) and owner) or (not row["_compressed_summary"] and not row["display_kind"]):
            return


def validate_retry_history(db, conn, session_id, expected_history):
    rows = conn.execute("SELECT * FROM messages WHERE session_id=? AND active=1 ORDER BY id",
                        (session_id,)).fetchall()
    validate_history_auth_state(rows)
    raw = db._rows_to_conversation(rows, session_id=session_id, include_ancestors=False,
                                   repair_alternation=False)
    target = next((i for i in range(len(raw) - 1, -1, -1)
                   if user_originated_turn_view(raw[i]) is not None), None)
    if target is None:
        raise ValueError("retry target is no longer available")
    turn = raw[target:]
    if (turn[0].get("display_metadata") or {}).get("codex_auth_pending") is True or any(
        (m.get("display_metadata") or {}).get("codex_auth_failure") is True for m in turn
    ):
        raise ValueError("Cannot retry this Codex authentication failure safely")
    # A missing closing row is uncertain execution, not proof that tools did not run.
    tail = next((m for m in reversed(turn) if m["role"] not in {"system", "session_meta"}), None)
    if tail is None or tail["role"] != "assistant" or tail.get("tool_calls"):
        raise ValueError("retry cannot establish a completed turn boundary")
    current = db._rows_to_conversation(rows, session_id=session_id, include_ancestors=False,
                                       repair_alternation=True, include_summary_markers=True)
    def comparison(history):
        return [{k: v for k, v in m.items() if k != "_row_id"} for m in history]
    if comparison(current) != comparison(expected_history):
        raise ValueError("session history changed before retry could be persisted")
