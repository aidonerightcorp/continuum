"""Wiring between the right-sizing loop and Hermes's public plugin surfaces.

    pre_llm_call           one fingerprint and one choice per turn; with the grant, may add the record of a
                           near-identical good answer from the same platform and sender
    llm_request middleware applies the choice to the provider request: model, settings, tool list
    post_api_request       records each model call
    api_request_error      records each failed model call; a failure that Hermes retries goes back to the
                           user's own model, a failure that Hermes does not retry blocks the cheaper model
    post_tool_call         records each tool call
    post_llm_call          stores the result of the turn; on a sample of the turns that a cheaper model answered,
                           asks the user's own model one PASS or FAIL question (ctx.side_question)
    on_session_end         audits the policy (this hook fires at the end of every turn)
    on_session_finalize    closes open outcomes and tunes the similarity weights

Nothing in a request is changed without the llm.model_override grant. With the setting `mode: off` every
callback returns at once and no new record is made.

Every callback is wrapped: a fault in Continuum is logged and the turn goes on unchanged.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .engine import RightSizer

logger = logging.getLogger("continuum.rightsize")

HOOKS = ("pre_llm_call", "post_llm_call", "post_api_request", "api_request_error", "post_tool_call",
         "on_session_end", "on_session_finalize")
MIDDLEWARE = ("llm_request",)
CAPABILITY = "llm.model_override"  # Hermes's consent id for a plugin that chooses the model of a call
CHECK_CAPABILITY = "llm.side_question"  # consent id for one short question to the user's own model after a turn
GRANT_NOTE = ("the model switch is not granted (to grant it, run `hermes plugins enable continuum` in a terminal "
              "and accept llm.model_override; spend follows the chosen model)")
RECALL_TOOL = "continuum_recall"
RECALL_SCHEMA = {
    "name": RECALL_TOOL,
    "description": ("Find how similar Tasks from the same user were answered before. Returns the closest recorded "
                    "Tasks with their answers, their recorded outcome and the tools they used. Local records only."),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The Task to look up, in plain words."},
            "limit": {"type": "integer", "description": "How many records to return (1 to 10).", "default": 3},
        },
        "required": ["query"],
    },
}

HELP = """Continuum commands
  /continuum            status of the right-sizing loop
  /continuum why        how the last turn was fingerprinted, chosen and recorded
  /continuum good|bad   record the outcome of the last turn (strongest signal)
  /continuum pin|unpin  keep this session on your own model, or release it
  /continuum unblock    try a cheaper model again after the provider refused it
  /continuum audit      check the policy on held-out records now
  /continuum tune       tune the similarity weights on the records now
  /continuum export     write fingerprints and outcomes (no text) to the plugin data folder
  /continuum import F   read an export file F from the plugin data folder's inbox/
  /continuum forget-text  remove stored Task and answer text; fingerprints and outcomes stay
  /continuum compact    rewrite the record file after text was removed, so no old copy stays in it
  /continuum lab        status of the experimental engines (with enable_lab_tools)
On a chat platform, the commands that act on the whole record file (unblock, audit, tune, export, import,
forget-text, compact) run only from a terminal on the machine that runs Hermes: hermes continuum <command>"""


def _safe(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception:  # never break a turn
            logger.warning("continuum: %s failed; the turn goes on unchanged", fn.__name__, exc_info=True)
            return None
    wrapper.__name__ = fn.__name__
    return wrapper


def _is_check(turn_id: Any = "", api_request_id: Any = "") -> bool:
    """True for the PASS or FAIL question that Hermes sends for Continuum. That request is no part of a user
    turn: it is never changed and never recorded as a model attempt of a turn."""
    return ":side_question" in str(turn_id or "") or ":side_question" in str(api_request_id or "")


def install(ctx: Any, sizer: RightSizer, data_dir: Path) -> dict[str, Callable[..., Any]]:
    """Register the hooks, the middleware and the recall tool. Returns the callbacks for tests."""

    def switch_allowed() -> bool:
        probe = getattr(ctx, "has_capability", None)
        return bool(probe(CAPABILITY)) if callable(probe) else False  # no consent surface: fail closed

    sizer.switch_allowed = switch_allowed
    sizer.switch_denied_note = GRANT_NOTE

    @_safe
    def pre_llm_call(session_id: str = "", user_message: Any = "", model: str = "", platform: str = "",
                     sender_id: Any = "", **_: Any):
        text = user_message if isinstance(user_message, str) else json.dumps(user_message, default=str)
        dec = sizer.start_turn(session_id or "default", text, model=model or "", platform=str(platform or ""),
                               sender_id=str(sender_id or ""))
        if dec.mode == "off":
            return None
        context = sizer.recall_context(session_id or "default")
        return {"context": context} if context else None

    @_safe
    def llm_request(request: Any = None, session_id: str = "", model: str = "", api_call_count: int = 0,
                    provider: Any = "", base_url: Any = "", turn_id: Any = "", api_request_id: Any = "", **_: Any):
        if _is_check(turn_id, api_request_id):
            return None
        shaped = sizer.shape_request(request or {}, session_id or "default", model or "", int(api_call_count or 0),
                                     provider=str(provider or ""), base_url=str(base_url or ""))
        if not shaped:
            return None
        return {"request": shaped[0], "source": "continuum", "reason": shaped[1][:300]}

    @_safe
    def post_api_request(session_id: str = "", model: str = "", response_model: Any = None, usage: Any = None,
                         api_duration: float = 0.0, finish_reason: str = "", assistant_tool_call_count: int = 0,
                         api_request_id: str = "", turn_id: Any = "", **_: Any):
        if _is_check(turn_id, api_request_id):
            return
        sizer.record_attempt(session_id or "default", ok=True, model=str(response_model or model or ""), usage=usage,
                             duration=api_duration or 0.0, finish=finish_reason or "",
                             tool_calls=assistant_tool_call_count or 0, api_request_id=api_request_id or "")

    @_safe
    def api_request_error(session_id: str = "", model: str = "", error: Any = None, api_duration: float = 0.0,
                          api_request_id: str = "", request: Any = None, retryable: Any = None,
                          status_code: Any = None, reason: Any = "", turn_id: Any = "", **_: Any):
        if _is_check(turn_id, api_request_id):
            return
        # Only the kind of the error goes to the record, never the provider's message text.
        kind = (str(error.get("type") or "") if isinstance(error, dict)
                else type(error).__name__ if isinstance(error, BaseException) else "")
        sent = request.get("model") if isinstance(request, dict) else None
        try:
            status = int(status_code or 0)
        except (TypeError, ValueError):
            status = 0
        sizer.record_attempt(session_id or "default", ok=False, model=str(sent or model or ""), error=kind,
                             duration=api_duration or 0.0, api_request_id=api_request_id or "",
                             retryable=retryable if isinstance(retryable, bool) else None, status=status,
                             reason=str(getattr(reason, "value", reason) or ""))

    @_safe
    def post_tool_call(tool_name: str = "", result: Any = None, session_id: str = "", status: Any = None, **_: Any):
        failed = str(status or "").lower() in ("error", "failed", "blocked")
        sizer.record_tool(session_id or "default", tool_name or "", '{"error": "tool failed"}' if failed else result)

    @_safe
    def reference_check(token: str, turn_key: str, prompt: str, timeout: float) -> str:
        t0 = time.monotonic()
        answer = ctx.side_question(token, prompt, timeout=timeout)
        return sizer.check_result(turn_key, answer, duration=time.monotonic() - t0)

    def check_allowed() -> bool:
        probe = getattr(ctx, "has_capability", None)
        return bool(probe(CHECK_CAPABILITY)) if callable(probe) else False

    @_safe
    def post_llm_call(session_id: str = "", assistant_response: Any = "", side_question_token: Any = None, **_: Any):
        text = assistant_response if isinstance(assistant_response, str) else json.dumps(assistant_response, default=str)
        ended = sizer.end_turn(session_id or "default", text)
        # Hermes gives a token only when the user granted llm.side_question. The hook must not wait for a model.
        if ended and ended["check"] and isinstance(side_question_token, str) and side_question_token \
                and callable(getattr(ctx, "side_question", None)) and check_allowed():
            worker = threading.Thread(target=reference_check, name="continuum-check", daemon=True,
                                      args=(side_question_token, ended["turn_key"], ended["prompt"], ended["timeout_s"]))
            worker.start()
        return None  # the hook returns nothing to Hermes

    @_safe
    def on_session_end(session_id: str = "", interrupted: bool = False, **_: Any):
        if sizer.mode == "off":
            return
        if interrupted:
            sizer.drop_open(session_id or "default", "interrupted")
        sizer.audit_if_new()  # this hook fires after every turn: the audit runs only when an outcome is new

    @_safe
    def on_session_finalize(session_id: Any = None, **_: Any):
        sizer.end_session(str(session_id or "default"))

    callbacks = {"pre_llm_call": pre_llm_call, "post_llm_call": post_llm_call, "post_api_request": post_api_request,
                 "api_request_error": api_request_error, "post_tool_call": post_tool_call,
                 "on_session_end": on_session_end, "on_session_finalize": on_session_finalize}
    for name in HOOKS:
        ctx.register_hook(name, callbacks[name])
    ctx.register_middleware("llm_request", llm_request)
    callbacks["llm_request"] = llm_request

    def recall_handler(args: dict[str, Any], session_id: Any = "", **_: Any) -> str:
        # Records are returned only to the platform and sender of the turn that is running in this session.
        try:
            if sizer.mode == "off":
                return json.dumps({"records": [], "note": "Continuum is off."})
            if not sizer.may_change():  # observe mode, or no consent: Continuum adds nothing to a turn
                return json.dumps({"records": [], "note": "Continuum only observes here (setting `mode`, or the "
                                                          "llm.model_override consent is not given), so no "
                                                          "records are shown."})
            scope = sizer.scope_of(str(session_id or "default"))
            if scope is None:
                return json.dumps({"records": [], "note": "No running turn was found for this session, so no "
                                                          "records are shown."})
            return json.dumps({"records": sizer.recall(str(args.get("query") or ""), int(args.get("limit") or 3),
                                                       scope=scope)})
        except Exception as exc:
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    ctx.register_tool(name=RECALL_TOOL, toolset="continuum", schema=RECALL_SCHEMA, handler=recall_handler,
                      description=RECALL_SCHEMA["description"])
    callbacks[RECALL_TOOL] = recall_handler
    return callbacks


OFF_NOTE = "Continuum is off (setting `mode`). Nothing was recorded and nothing was changed."
NO_SESSION_NOTE = ("Hermes gave no session id for this chat, so Continuum cannot tell which turn is yours. "
                   "Nothing was shown or changed.")
# These commands act on the whole record file: every user's text, the shared policy, an export of everything.
WHOLE_FILE = ("forget-text", "forget_text", "compact", "unblock", "export", "import", "tune", "audit")
NOT_REWRITTEN = ("The record file could not be rewritten now (another Hermes process holds it, or the disk has no "
                 "room): old copies of the removed text can still be in the file or its side file. Close the other "
                 "Hermes process and run: hermes continuum compact. A start of Hermes also does the rewrite, for a "
                 "record file up to the size limit in config/rightsize.json (32 MiB).")
OWNER_NOTE = ("`/continuum {arg}` acts on the whole record file, so it runs only for the local user. On the machine "
              "that runs Hermes, type: hermes continuum {arg}")


def command(sizer: RightSizer, data_dir: Path, raw_args: str, session_id: str = "", platform: str = "") -> str | None:
    """The /continuum subcommands of the right-sizing loop. Returns None for a subcommand it does not own."""
    parts = (raw_args or "").strip().split(None, 1)
    arg = parts[0].lower() if parts else "status"
    rest = parts[1].strip() if len(parts) > 1 else ""
    sid = session_id or None  # a named session never falls back to another session's turn
    # Without a session id the newest turn is taken. That is right for one local user and wrong on a chat
    # platform, where the newest turn can be another person's.
    local = str(platform or "") in sizer.cfg["recall"].get("single_user_platforms", [])
    if arg in ("status", ""):
        # On a chat platform the counts are shown; the provider's name and the reason of a block are for
        # the local user only.
        return sizer.status_text(detail=local)
    if arg == "help":
        return HELP
    if sizer.mode == "off" and arg in ("good", "bad", "pin", "unpin", "unblock", "audit", "tune", "import"):
        return OFF_NOTE
    if not local and arg in WHOLE_FILE:
        return OWNER_NOTE.format(arg=arg.replace("_", "-"))
    if not sid and not local and arg in ("why", "good", "bad", "pin", "unpin"):
        return NO_SESSION_NOTE
    if arg == "why":
        return sizer.why_text(sid)
    if arg in ("good", "bad"):
        row = sizer.feedback(arg == "good", sid)
        if not row:
            return "Continuum has no recorded turn to mark" + (" in this session yet." if sid else " yet.")
        if row.get("source") in ("host_error", "rung_blocked"):
            return f"Turn {row['turn_key']} had no answer from {row['model'] or row['rung']} ({row['source']}), so it is not scored."
        if arg == "good" and int(row.get("escalated") or 0):
            return (f"Turn {row['turn_key']} fell back to your own model after {row['model'] or row['rung']} failed, "
                    f"so {row['model'] or row['rung']} keeps its failure record.")
        return f"Recorded: turn {row['turn_key']} marked {arg} ({row['model'] or row['rung']})."
    if arg in ("pin", "unpin"):
        sizer.pin(sid or "default", arg == "pin")
        return ("This session now stays on your own model." if arg == "pin"
                else "This session is right-sized again.")
    if arg == "unblock":
        had_state = bool(sizer.store.get_policy("rung_home", {}) or sizer.store.get_policy("rung_strikes", {})
                         or getattr(sizer, "_held", None))
        names = sizer.unblock()
        if names:
            return "Unblocked: " + ", ".join(names) + ". Continuum may try it again."
        if had_state:
            return "No cheaper model was blocked; cleared the remembered provider and refusal counts."
        return "No cheaper model is blocked."
    if arg in ("export", "import") and sizer.store.in_memory:
        return "Continuum has no record file here, so there is nothing to export and no place to import to."
    if arg in ("forget-text", "forget_text"):
        n = sizer.forget_text()
        text = (f"Removed the stored Task and answer text of {n} turns. Fingerprints and outcomes stay. "
                "Set store_text: false to store no text from now on.")
        try:
            others = sorted(p.name for p in Path(data_dir).glob("*") if p.is_file() and (
                ".unreadable-" in p.name or p.name.startswith("continuum.db")))
        except OSError:
            others = []  # the folder cannot be listed: the text is removed all the same
        if others:
            text += ("\nNot touched by this command, in " + str(data_dir) + ": " + ", ".join(others[:8])
                     + (" and more" if len(others) > 8 else "")
                     + ". These are files Continuum set aside because it could not read them, or the file of "
                     "the experimental engines. Delete them yourself if you do not want them.")
        if sizer.store.compact_pending():
            text += "\n" + NOT_REWRITTEN
        return text
    if arg == "compact":
        if sizer.store.in_memory:
            note = str(getattr(sizer, "store_note", "") or "")
            return ("Continuum is not using a record file in this run, so there is nothing to rewrite."
                    + (" " + note if note else ""))
        if not sizer.store.compact_pending():
            return "The record file needs no rewrite: no text was removed since the last one."
        return "The record file was rewritten. No old copy of removed text stays in it." if sizer.store.compact() \
            else NOT_REWRITTEN
    if arg == "audit":
        rep = sizer.audit_now()
        lines = [f"Audit: {rep['action']} · floor {rep['state']['floor']:.2f} · evidence needed {rep['state']['min_evidence']:.1f}"]
        if rep.get("advised"):
            lines.append(f"  the records advise '{rep['advised']}'; not applied: {rep.get('note', '')}")
        for view, v in rep.get("views", {}).items():
            lines.append(f"  {view}: cheaper rungs passed {v['cheap_rate']:.0%} on {v['cheap_evidence']:.1f} units "
                         f"(interval {v['lower']:.2f} to {v['upper']:.2f}); reference {v['reference_rate']:.0%} on "
                         f"{v['reference_evidence']:.1f} units; target {v['target']:.2f}")
        return "\n".join(lines)
    if arg == "tune":
        rep = sizer.tune_now()
        if rep.get("skipped"):
            return f"Tune: {rep['skipped']}. Similarity preset stays {rep['chosen']}."
        errs = ", ".join(f"{k} {v:.4f}" for k, v in sorted(rep["fit_error"].items(), key=lambda kv: kv[1]))
        held = (", ".join(f"{k} {v:.4f}" for k, v in rep["held_out"].items())
                or ("no held-out records yet (preset kept)" if rep.get("holdout_records", 0) == 0
                    else "not needed (the current preset is best on the fit records)"))
        return f"Tune: preset {rep['chosen']} (was {rep['current']}).\n  error on fit records: {errs}\n  held-out check: {held}"
    if arg == "export":
        path = sizer.export(data_dir / "exports")
        return (f"Wrote {path} (fingerprints and outcomes only; no Task text, no answers). A fingerprint is not "
                "text, but a person who guesses a Task can test the guess against it: share the file with care.")
    if arg == "import":
        name = Path(rest).name
        path = data_dir / "inbox" / name
        if not name or not path.is_file():
            return f"Put the export file in {data_dir / 'inbox'} and run /continuum import <file name>."
        return (f"Imported {sizer.import_file(path)} records from {name}. They are labelled, carry reduced weight "
                "and never decide a choice without your own records.")
    return None
