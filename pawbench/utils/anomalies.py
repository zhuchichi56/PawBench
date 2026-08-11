# -*- coding: utf-8 -*-
"""Rule-based anomaly detection for pawbench task runs.

Each anomaly represents an infrastructure or execution problem that makes a
run's score unreliable (e.g. OOM kill, empty transcript, grading script error).
Anomaly data is attached to TaskResult and written into checkpoint JSON so
downstream analysis can filter or flag affected runs.

Ported from examples/QwenClawBench/scripts/lib_anomalies.py.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


# ── Anomaly severity ──────────────────────────────────────────────────────────

ERROR = "error"       # score-impacting: run result is unreliable / incomplete
WARNING = "warning"   # transient: anomaly detected but score may still be valid


# ── Transcript text extraction ────────────────────────────────────────────────

def _get_transcript_text(transcript: List[Dict[str, Any]]) -> str:
    """Flatten a transcript list to a single string for keyword searches.

    Only looks at text-like fields to avoid false positives from binary
    content or argument objects.
    """
    parts: List[str] = []
    for entry in transcript:
        entry_type = entry.get("type", "")
        if entry_type == "message":
            msg = entry.get("message", {})
            for item in msg.get("content", []):
                if isinstance(item, dict):
                    text = item.get("text", "") or item.get("content", "")
                    if text:
                        parts.append(str(text))
        elif entry_type in ("toolResult", "toolCall"):
            content = entry.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text", "") or item.get("content", "")
                        if text:
                            parts.append(str(text))
            elif isinstance(content, str):
                parts.append(content)
    return "\n".join(parts)


def _scan_text_for_keywords(text: str, keywords: List[str]) -> bool:
    """Return True if any keyword appears in text (case-insensitive)."""
    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


def _extract_llm_api_errors(transcript: List[Dict[str, Any]]) -> str:
    """Extract only LLM API error messages from transcript.

    Collects errorMessage from assistant messages where stopReason='error'.
    Targets actual provider-level failures (rate limits, 5xx, dropped
    connections) and excludes task content that happens to mention error terms.
    """
    parts: List[str] = []
    for entry in transcript:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        if msg.get("role") == "assistant" and msg.get("stopReason") == "error":
            err = msg.get("errorMessage", "")
            if err:
                parts.append(err)
    return "\n".join(parts)


# ── Individual rule checks ────────────────────────────────────────────────────
# Each function receives (execution_result, grade_notes, transcript_text, stderr)
# and returns either None (no anomaly) or a description string.

def _check_execution_exception(exec_r, notes, transcript_text, stderr):
    if exec_r.get("exit_code") == -1:
        cause = ""
        if stderr:
            lines = [l.strip() for l in stderr.splitlines() if l.strip()]
            if lines:
                cause = f": {lines[-1][:200]}"
        return f"Docker/subprocess failed before agent ran (exit_code=-1){cause}"
    return None


def _check_oom_kill(exec_r, notes, transcript_text, stderr):
    if exec_r.get("exit_code") == 137:
        return "Process killed by SIGKILL (exit_code=137), likely OOM or manual kill"
    return None


def _check_exit_code_nonzero(exec_r, notes, transcript_text, stderr):
    code = exec_r.get("exit_code")
    timed_out = exec_r.get("timed_out", False)
    if code not in (0, -1, 137, None) and not timed_out:
        return f"Non-zero exit code (exit_code={code}) without timeout"
    return None


def _check_task_timed_out(exec_r, notes, transcript_text, stderr):
    if exec_r.get("timed_out"):
        return "Task execution timed out — agent may have been mid-work when killed"
    return None


def _check_empty_transcript(exec_r, notes, transcript_text, stderr):
    tlen = exec_r.get("transcript_length", 0)
    status = exec_r.get("status", "")
    if tlen == 0 and status != "timeout":
        return "Empty transcript — agent never ran or transcript collection failed"
    return None


def _check_short_transcript(exec_r, notes, transcript_text, stderr):
    tlen = exec_r.get("transcript_length", 0)
    if not 0 < tlen < 5:
        return None

    # Brevity alone is not an infrastructure failure. A complete short
    # exchange is normal for one-action PawBench tasks (for example:
    # user -> assistant tool call -> tool result -> final assistant). Only
    # flag a short transcript when its conversational envelope is incomplete.
    transcript = exec_r.get("transcript") or []
    roles = [
        event.get("message", {}).get("role")
        for event in transcript
        if isinstance(event, dict) and isinstance(event.get("message"), dict)
    ]
    if (
        exec_r.get("status") == "success"
        and roles
        and roles[0] == "user"
        and roles[-1] == "assistant"
        and transcript_text.strip()
    ):
        return None
    return f"Very short incomplete transcript ({tlen} entries) — agent may have crashed immediately after starting"


def _check_quick_exit(exec_r, notes, transcript_text, stderr):
    elapsed = exec_r.get("execution_time", float("inf"))
    status = exec_r.get("status", "")
    if elapsed < 10.0 and status == "error":
        return (
            f"Container exited very quickly ({elapsed:.1f}s) with error — "
            "possible missing assets or Docker startup failure"
        )
    return None


def _check_grading_script_error(exec_r, notes, transcript_text, stderr):
    if notes and "Grading failed:" in notes:
        short = notes[:150].replace("\n", " ")
        return f"Grading script raised an exception: {short}"
    return None


def _check_grading_missing_function(exec_r, notes, transcript_text, stderr):
    if notes and (
        "grading function missing" in notes.lower()
        or "No automated grading code" in notes
    ):
        return "Grading function not found in task definition — score is forced to 0.0"
    return None


def _check_api_rate_limit(exec_r, notes, transcript_text, stderr):
    # Primary: inspect LLM API error messages (assistant stopReason='error'
    # errorMessage) and stderr — avoids false positives from task content that
    # mentions rate-limit terminology.
    transcript = exec_r.get("transcript", [])
    api_errors = _extract_llm_api_errors(transcript)
    combined = (api_errors + "\n" + (stderr or "")).lower()
    keywords = [
        "429", "rate limit", "rate_limit", "too many requests", "ratelimit",
        "resource_exhausted", "resource has been exhausted",
        "you exceeded your current quota",
        "request rate increased too quickly",
        "api rate limit reached",
    ]
    if _scan_text_for_keywords(combined, keywords):
        return "API rate limit detected in LLM API error response"

    # Fallback: some agents (e.g. qwenpaw) retry 429s internally and the
    # transcript never gets a stopReason=error entry. In those cases the
    # runner passes log_text (from agent log files) via execution_result.
    #
    # Use strong, context-aware patterns only to avoid false positives from:
    #   - UUIDs containing "429" (e.g. "c818074c-9028-4296-993d")
    #   - floating-point numbers (e.g. "158.42997...")
    #   - file names (e.g. "1cb429f7...")
    #   - HTTP port numbers (e.g. ":42910")
    #   - initialization log lines ("rate limiter initialized")
    log_text = (exec_r.get("log_text") or "")
    if log_text:
        # Each pattern must unambiguously point to a real API rate-limit error.
        # \b429\b guards against substring matches within longer digit sequences.
        _LOG_STRONG_RE = re.compile(
            r"rawError=429"
            r"|Error code:\s*429"
            r"|reason=rate_limit"
            r"|API rate limit reached"
            r"|You exceeded your current quota"
            r"|Request rate increased too quickly"
            r"|\binsufficient_quota\b"
            r"|\bresource_exhausted\b"
            r"|LLM judge API returned 429"
            r"|status[_\s]?code[=:\s]+429\b"
            r"|HTTP/\S+\s+429\b",
            re.IGNORECASE,
        )
        _LOG_BENIGN_RE = re.compile(
            r"rate limiter initialized"
            r"|LLM rate limiter initialized"
            r"|max_qpm=",
            re.IGNORECASE,
        )
        for line in log_text.splitlines():
            if _LOG_BENIGN_RE.search(line):
                continue
            if _LOG_STRONG_RE.search(line):
                return "API rate limit detected in agent log (internal retry masked the 429 in transcript)"

    return None


def _check_api_server_error(exec_r, notes, transcript_text, stderr):
    transcript = exec_r.get("transcript", [])
    combined = _extract_llm_api_errors(transcript) + "\n" + (stderr or "")
    if _scan_text_for_keywords(combined, [
        "502", "503", "500",
        "service unavailable", "bad gateway", "internal server error",
        "internal error has occurred",
        "backend buffer overflow",
        "requesttimeout",
        "unavailable",
    ]):
        return "API server error (5xx or server-side failure) detected in LLM API error response"
    return None


def _check_zero_token_response(exec_r, notes, transcript_text, stderr):
    """Detect cases where the transcript exists but the model generated 0 tokens."""
    tlen = exec_r.get("transcript_length", 0)
    if tlen == 0:
        return None  # covered by EMPTY_TRANSCRIPT

    transcript = exec_r.get("transcript", [])
    assistant_msgs = [
        e.get("message", {})
        for e in transcript
        if e.get("type") == "message" and e.get("message", {}).get("role") == "assistant"
    ]

    if any(msg.get("content") for msg in assistant_msgs):
        return None

    usage = exec_r.get("usage", {})
    total_tokens = usage.get("total_tokens", -1)
    if total_tokens == 0:
        return (
            "Transcript exists but all assistant messages are empty with 0 tokens — "
            "model API returned empty responses (silent failure, possibly "
            "mis-configured model endpoint or auth error)"
        )

    if assistant_msgs and all(
        msg.get("usage", {}).get("totalTokens", -1) == 0
        for msg in assistant_msgs
    ):
        return (
            "All assistant messages have empty content and 0 tokens — "
            "model API returned empty responses (silent failure)"
        )

    return None


def _check_terminal_api_failure(exec_r, notes, transcript_text, stderr):
    """Detect runs that ended because the final API call returned an error.

    Catches the 'partial run killed mid-task' case: the transcript has content
    from early turns (so EMPTY_TRANSCRIPT and ZERO_TOKEN_RESPONSE don't fire),
    but the run's last assistant message has stopReason=error with empty content
    — the model was cut off and the task never completed.

    Typical cause: API quota exhausted or rate limit hit repeatedly until the
    agent runner gave up, even though some initial turns succeeded.
    """
    transcript = exec_r.get("transcript", [])
    if not transcript:
        return None  # covered by EMPTY_TRANSCRIPT

    assistant_msgs = [
        e.get("message", {})
        for e in transcript
        if e.get("type") == "message" and e.get("message", {}).get("role") == "assistant"
    ]
    if not assistant_msgs:
        return None

    last_msg = assistant_msgs[-1]
    if last_msg.get("stopReason") == "error" and not last_msg.get("content"):
        error_msg = str(last_msg.get("errorMessage", "unknown API error"))
        return (
            "Run ended with API error on final turn (content=[], stopReason=error) — "
            f"task never completed: {error_msg[:150]}"
        )

    if (
        not last_msg.get("content")
        and last_msg.get("stopReason") == "stop"
        and last_msg.get("usage", {}).get("totalTokens", -1) == 0
    ):
        return (
            "Run ended silently on final turn (content=[], stopReason=stop, tokens=0) — "
            "model produced no output after last tool results (silent task abandonment)"
        )

    return None


# ── Rule registry ─────────────────────────────────────────────────────────────
# Each entry: (rule_id, severity, check_function)

_RULES: List[tuple] = [
    ("EXECUTION_EXCEPTION",      ERROR,   _check_execution_exception),
    ("EXIT_CODE_OOM",            ERROR,   _check_oom_kill),
    ("EXIT_CODE_NONZERO",        ERROR,   _check_exit_code_nonzero),
    ("TASK_TIMED_OUT",           ERROR,   _check_task_timed_out),
    ("EMPTY_TRANSCRIPT",         ERROR,   _check_empty_transcript),
    ("SHORT_TRANSCRIPT",         ERROR,   _check_short_transcript),
    ("QUICK_EXIT_SUSPICIOUS",    ERROR,   _check_quick_exit),
    ("GRADING_SCRIPT_ERROR",     ERROR,   _check_grading_script_error),
    ("GRADING_MISSING_FUNCTION", ERROR,   _check_grading_missing_function),
    ("API_RATE_LIMIT",           WARNING, _check_api_rate_limit),
    ("API_SERVER_ERROR",         WARNING, _check_api_server_error),
    ("ZERO_TOKEN_RESPONSE",      ERROR,   _check_zero_token_response),
    ("TERMINAL_API_FAILURE",     ERROR,   _check_terminal_api_failure),
]

# Rule IDs that indicate infrastructure failure (not model capability)
API_FAILURE_IDS = frozenset({
    "API_RATE_LIMIT",
    "API_SERVER_ERROR",
    "TERMINAL_API_FAILURE",
    "EMPTY_TRANSCRIPT",
    "ZERO_TOKEN_RESPONSE",
})


# ── Public API ────────────────────────────────────────────────────────────────

def detect_anomalies(
    execution_result: Dict[str, Any],
    grade_notes: str,
) -> Dict[str, Any]:
    """Detect anomalies in a single task run.

    Parameters
    ----------
    execution_result:
        The dict from ``_run_agent_async`` / ``execute_task_in_docker``.
        Expected keys: ``exit_code``, ``timed_out``, ``status``,
        ``execution_time``, ``transcript_length``, ``transcript`` (list),
        ``stderr``.
    grade_notes:
        The ``notes`` string from the GradeResult.

    Returns
    -------
    dict with keys:
        ``is_anomalous``  — True if any rule triggered.
        ``has_error``     — True if any error-severity rule triggered.
        ``has_api_error`` — True if an API-related anomaly triggered.
        ``items``         — List of {id, severity, description} dicts.
    """
    transcript = execution_result.get("transcript", [])
    transcript_text = _get_transcript_text(transcript)
    stderr = execution_result.get("stderr", "") or ""

    items: List[Dict[str, str]] = []
    for rule_id, severity, check_fn in _RULES:
        description = check_fn(execution_result, grade_notes, transcript_text, stderr)
        if description is not None:
            items.append({"id": rule_id, "severity": severity, "description": description})

    # Post-process: if API_RATE_LIMIT / API_SERVER_ERROR co-occurs with a fatal
    # error that produced zero useful output, upgrade those warnings to ERROR.
    _fatal_error_ids = {
        "EMPTY_TRANSCRIPT",
        "ZERO_TOKEN_RESPONSE",
        "TERMINAL_API_FAILURE",
        "EXECUTION_EXCEPTION",
        "EXIT_CODE_OOM",
    }
    _api_warning_ids = {"API_RATE_LIMIT", "API_SERVER_ERROR"}
    triggered_ids = {item["id"] for item in items}
    has_fatal_error = bool(triggered_ids & _fatal_error_ids)
    if has_fatal_error:
        for item in items:
            if item["id"] in _api_warning_ids:
                item["severity"] = ERROR
                item["description"] += " [upgraded to error: co-occurs with fatal execution failure]"

    has_error = any(item["severity"] == ERROR for item in items)
    has_api_error = bool(triggered_ids & API_FAILURE_IDS)

    return {
        "is_anomalous": len(items) > 0,
        "has_error": has_error,
        "has_api_error": has_api_error,
        "items": items,
    }
