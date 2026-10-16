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


# Terminal durability and permission to replay are independent facts.
INPUT_OUTCOMES = frozenset({'completed', 'retryable_failure', 'review_required', 'auth_failure'})
REPLAY_BLOCKED_OUTCOMES = frozenset({'review_required', 'auth_failure'})


def input_metadata(raw):
    """Reject unreadable safety metadata instead of treating it as absent."""
    meta = json.loads(raw) if raw else {}
    if not isinstance(meta, dict) or any(
        k in meta and not isinstance(meta[k], bool) for k in ('codex_auth_pending', 'codex_auth_failure')
    ) or ('gateway_input_outcome' in meta and (not isinstance(meta['gateway_input_outcome'], str)
        or meta['gateway_input_outcome'] not in INPUT_OUTCOMES)):
        raise ValueError('owned input safety metadata is invalid')
    return meta


def owned_input_span(rows, *, input_owner=None, user_row_id=None, platform_message_id=''):
    """Select the root and its same-invocation copies/steers, refusing a newer independent input.

    A legacy typed steer without identity is accepted only between the pinned root
    and its closing boundary. Plain unowned user rows are never assumed to be steers.
    """
    users = [(i, r, input_metadata(r['display_metadata'])) for i, r in enumerate(rows) if r['role'] == 'user']
    match = next(((i, r, m) for i, r, m in users if user_row_id is not None and r['id'] == user_row_id), None)
    if match is None and user_row_id is None:
        match = next(((i, r, m) for i, r, m in users if
                      (input_owner and m.get('gateway_input_owner') == input_owner) or
                      (platform_message_id and r['platform_message_id'] == platform_message_id)), None)
    if match is None:
        return None
    start, root, meta = match
    owner = meta.get('gateway_input_owner')
    if owner:
        # Compaction may retain several carriers for this same accepted input.
        for i, prior, prior_meta in reversed(users):
            if prior['id'] >= root['id']:
                continue
            if prior_meta.get('gateway_input_owner') != owner:
                break
            start = i
    if (input_owner and owner is not None and owner != input_owner) or (
        platform_message_id and root['platform_message_id'] and root['platform_message_id'] != platform_message_id
    ):
        return None
    # A platform id or pinned row may locate a legacy input whose owner is absent.
    owner = owner or input_owner
    for _, row, row_meta in users:
        if row['id'] <= root['id']:
            continue
        row_owner = row_meta.get('gateway_input_owner')
        if owner and row_owner == owner:
            continue
        if row['display_kind'] == 'steer' and row_owner is None and not row['platform_message_id']:
            continue
        return None
    return start, root, meta, owner


def settle_owned_input(db, conn, session_id, *, input_owner=None, user_row_id=None,
                       platform_message_id='', completed=False, auth_failure=False,
                       interrupted=False, failed=False, notice='', turn_lease_holder=None):
    """Settle one bounded durable invocation under its holder-qualified writer snapshot."""
    import time
    from collections import Counter
    from hermes_state_messages import _TURN_LEASE_ROW_SQL
    from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND

    unknown = {'saved': False, 'current': False, 'settled': False, 'tools_recorded': None}
    key = db._session_turn_lease_key_on_conn(conn, session_id)
    if turn_lease_holder:
        lease = conn.execute(_TURN_LEASE_ROW_SQL, (key,)).fetchone()
        if lease is None or lease['holder'] != turn_lease_holder or float(lease['expires_at']) <= time.time():
            raise ValueError('Owned input settlement no longer owns the admitted lease')
    target = resolve_turn_admission_target(conn, session_id)
    if db._session_turn_lease_key_on_conn(conn, target) != key:
        raise ValueError('Owned input settlement changed conversation lineage')
    db._check_transcript_write_guards(conn, target, None, turn_lease_holder=turn_lease_holder,
                                      reject_active_turn_lease=not turn_lease_holder)
    # No provider payload or prompt content enters the receipt. Tool IDs suffice to
    # reject a supposedly completed turn with an unfinished call.
    rows = conn.execute("""SELECT id,role,display_kind,display_metadata,platform_message_id,tool_call_id,
        CASE WHEN tool_calls IS NULL OR tool_calls='[]' THEN 0 ELSE 1 END AS has_calls,
        CASE WHEN json_valid(tool_calls) AND json_type(tool_calls)='array'
             THEN (SELECT json_group_array(json_extract(value,'$.id')) FROM json_each(tool_calls))
             ELSE NULL END AS call_ids
        FROM messages WHERE session_id=? AND active=1 AND compacted=0 ORDER BY id""", (target,)).fetchall()
    span = owned_input_span(rows, input_owner=input_owner, user_row_id=user_row_id,
                            platform_message_id=platform_message_id)
    if span is None:
        return unknown
    start, user, meta, owner = span
    receipt = {'saved': True, 'current': True, 'settled': False, 'row_id': user['id'],
               'input_owner': owner, 'session_id': target, 'tools_recorded': False}
    turn = rows[start:]
    receipt['tools_recorded'] = any(r['role'] in {'tool','function'} or r['has_calls'] for r in turn)
    if not (meta.get('codex_auth_pending') is True or meta.get('gateway_input_outcome') or auth_failure):
        return unknown
    tail = next((r for r in reversed(turn) if r['role'] in {'user','assistant','tool','function'}), None)
    previous = meta.get('gateway_input_outcome')
    if previous:
        if any(input_metadata(r['display_metadata']).get('codex_auth_pending') is True for r in turn if r['role'] == 'user'):
            return receipt  # contradictory/new pending evidence is not an idempotent close
        same = (auth_failure and previous == 'auth_failure') or (completed and previous == 'completed') or (
            failed and not auth_failure and previous in {'retryable_failure','review_required'})
        if not same:
            return receipt
        receipt.update(settled=True, outcome=previous, retry_blocked=previous in REPLAY_BLOCKED_OUTCOMES,
            _row_updates=[{'id':r['id'],'display_metadata':input_metadata(r['display_metadata'])}
                          for r in turn if r['role']=='user'])
        return receipt
    if completed:
        calls, results = Counter(), Counter()
        invalid = False
        for row in turn:
            if row['has_calls']:
                ids = json.loads(row['call_ids']) if row['call_ids'] else []
                if not ids or not all(isinstance(i, str) and i for i in ids):
                    invalid = True
                else:
                    calls.update(ids)
            if row['role'] == 'tool':
                results.update([row['tool_call_id']])
        if (interrupted or failed or tail is None or tail['role'] != 'assistant' or tail['has_calls']
                or invalid or calls - results or meta.get('codex_auth_failure') is True
                or any(input_metadata(r['display_metadata']).get('codex_auth_failure') is True for r in turn)):
            return receipt
        outcome = 'completed'
    elif auth_failure:
        outcome = 'auth_failure'
    elif failed and not interrupted and tail and tail['role'] == 'assistant' and not tail['has_calls']:
        outcome = 'review_required' if receipt['tools_recorded'] else 'retryable_failure'
    else:
        return receipt  # interrupted/unclosed/unknown execution retains the pending fence
    updates = []
    for row in turn:
        if row['role'] != 'user':
            continue
        row_meta = input_metadata(row['display_metadata'])
        row_meta.update(codex_auth_pending=False, gateway_input_outcome=outcome)
        if owner:
            row_meta['gateway_input_owner'] = owner
        if auth_failure:
            row_meta['codex_auth_failure'] = True
        conn.execute('UPDATE messages SET display_metadata=? WHERE id=?',
                     (db._encode_display_metadata(row_meta), row['id']))
        updates.append({'id':row['id'], 'display_metadata':row_meta})
    if auth_failure:
        if tail and tail['role'] == 'assistant':
            tail_meta = input_metadata(tail['display_metadata'])
            tail_meta['codex_auth_failure'] = True
            conn.execute('UPDATE messages SET display_metadata=? WHERE id=?',
                         (db._encode_display_metadata(tail_meta),tail['id']))
            update = {'id':tail['id'],'display_metadata':tail_meta}
            if notice and tail['display_kind'] == FAILED_TURN_DISPLAY_KIND:
                conn.execute('UPDATE messages SET content=? WHERE id=?', (db._encode_content(notice), tail['id']))
                from agent.transcript_repair import transcript_row_snapshot
                update['content'] = notice
                update['_db_row_snapshot'] = transcript_row_snapshot(
                    conn.execute('SELECT * FROM messages WHERE id=?', (tail['id'],)).fetchone())
            updates.append(update)
        else:
            msg = {'content':notice or 'Codex authentication failed; review this request before another attempt.',
                   'display_kind':FAILED_TURN_DISPLAY_KIND, 'display_metadata':{'codex_auth_failure':True}}
            msg['role'] = 'assistant'
            inserted, calls = db._insert_message_rows(conn, target, [msg], prune_checkpoints=False)
            db._bump_session_counters(conn, target, inserted, calls, unit=True)
            updates.append({'id':msg['_row_id'], '_new_message':msg,
                            'display_metadata':msg['display_metadata']})
    receipt.update(settled=True, outcome=outcome, retry_blocked=outcome in REPLAY_BLOCKED_OUTCOMES,
                   boundary_saved=True, _row_updates=updates)
    return receipt


def validate_history_auth_state(rows, *, reject_pending=False, reject_failure=False):
    """Check the fetched snapshot before projection/alternation can hide a pending input."""
    failed_turn = False
    for row in rows:
        if row["role"] == "user":
            failed_turn = False
        if row["display_metadata"]:
            try:
                meta = input_metadata(row["display_metadata"])
            except (ValueError, TypeError) as exc:
                raise ValueError("retry safety metadata is invalid") from exc
            if reject_pending and row["role"] == "user" and meta.get("codex_auth_pending") is True:
                error = CodexAuthResumeBlocked if reject_failure else ValueError
                raise error("Codex input has no durable completion or failure boundary")
            if meta.get("codex_auth_failure") is True or meta.get("gateway_input_outcome") in INPUT_OUTCOMES - {"completed"}:
                failed_turn = True
    if reject_failure and failed_turn:
        raise CodexAuthResumeBlocked("Automatic continuation refused after a recorded Codex auth failure")


def reject_archived_codex_auth_resume(conn, target, lease_key, *, automatic=True):
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
            meta = input_metadata(row["display_metadata"])
        except (ValueError, TypeError) as exc:
            raise CodexAuthResumeBlocked("Replay safety metadata is invalid") from exc
        if (meta.get("codex_auth_pending") is True or meta.get("codex_auth_failure") is True
                or meta.get("gateway_input_outcome") in (INPUT_OUTCOMES - {"completed"} if automatic else REPLAY_BLOCKED_OUTCOMES)):
            raise CodexAuthResumeBlocked("Automatic continuation refused after an uncertain or failed Codex input")
        owner = meta.get("gateway_input_owner")
        if (isinstance(owner, str) and owner) or (not row["_compressed_summary"] and not row["display_kind"]):
            return


def validate_retry_history(db, conn, session_id, expected_history):
    rows = conn.execute("SELECT * FROM messages WHERE session_id=? AND active=1 ORDER BY id",
                        (session_id,)).fetchall()
    validate_history_auth_state(rows, reject_pending=True)
    reject_archived_codex_auth_resume(conn, session_id, db._session_turn_lease_key_on_conn(conn, session_id), automatic=False)
    raw = db._rows_to_conversation(rows, session_id=session_id, include_ancestors=False,
                                   repair_alternation=False)
    target = next((i for i in range(len(raw) - 1, -1, -1)
                   if user_originated_turn_view(raw[i]) is not None), None)
    if target is None:
        raise ValueError("retry target is no longer available")
    turn = raw[target:]
    if (turn[0].get("display_metadata") or {}).get("codex_auth_pending") is True or any(
        (m.get("display_metadata") or {}).get("codex_auth_failure") is True or
        (m.get("display_metadata") or {}).get("gateway_input_outcome") in REPLAY_BLOCKED_OUTCOMES for m in turn
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
