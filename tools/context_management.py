import json
from typing import List


AUTO_ARCHIVE_MARKER = "[Auto-archived browser result]"
BROWSER_TOOL_NAMES = ("browser.search", "browser.open", "browser.find")


def _is_archived(content: str) -> bool:
    return AUTO_ARCHIVE_MARKER in content or content.startswith("[Auto-archived |")


def _is_browser_error_result(content: str) -> bool:
    text = (content or "").lstrip()
    if not text:
        return False
    if text.startswith("Error executing browser."):
        return True
    if text.startswith("Error rendering browser."):
        return True
    if text.startswith("Error executing ") and any(
        f"browser.{name}" in text[:80] for name in ("search", "open", "find")
    ):
        return True
    if text.startswith("Error rendering ") and any(
        f"browser.{name}" in text[:80] for name in ("search", "open", "find")
    ):
        return True
    if text.startswith("Error: Invalid arguments for function"):
        return True
    if text.startswith("An unexpected error occurred while executing function"):
        return True
    if text.startswith('{"error"'):
        try:
            parsed = json.loads(text)
        except Exception:
            return False
        return isinstance(parsed, dict) and "error" in parsed
    return False


def _args_desc(raw_args) -> str:
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except Exception:
            raw_args = {"raw": raw_args}
    if not isinstance(raw_args, dict) or not raw_args:
        return "no args"
    return ", ".join(f"{k}={repr(v)[:80]}" for k, v in raw_args.items())


def _call_info_by_id(messages: List[dict]) -> dict:
    call_id_to_info = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            tid = tc.get("id", "")
            fn = tc.get("function", {})
            name = fn.get("name", "")
            if tid:
                call_id_to_info[tid] = (name, fn.get("arguments", {}))
    return call_id_to_info


def _auto_archive_placeholder(tool_name: str, args_desc: str) -> str:
    return f"{AUTO_ARCHIVE_MARKER} tool={tool_name} | args: {args_desc}"


def _auto_archive_old_results(
    messages: List[dict],
    round_num: int,
    force_archive_after_turns: int,
    error_tool_call_ids: set[str] = None,
) -> int:
    """Replace old successful browser tool results in the working context."""
    if force_archive_after_turns <= 0:
        return 0

    error_tool_call_ids = set(error_tool_call_ids or ())

    def _has_non_error_browser_call(msg: dict) -> bool:
        for tc in msg.get("tool_calls") or []:
            tid = tc.get("id", "")
            name = tc.get("function", {}).get("name", "")
            if name in BROWSER_TOOL_NAMES and tid not in error_tool_call_ids:
                return True
        return False

    effective_asst_indices = [
        i for i, msg in enumerate(messages)
        if msg.get("role") == "assistant" and _has_non_error_browser_call(msg)
    ]
    if len(effective_asst_indices) < force_archive_after_turns:
        return 0

    cutoff_idx = effective_asst_indices[-force_archive_after_turns]
    call_id_to_info = _call_info_by_id(messages)
    archived = 0

    for i in range(cutoff_idx):
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str) or _is_archived(content):
            continue

        tool_id = msg.get("tool_call_id", "")
        tool_name, raw_args = call_id_to_info.get(tool_id, ("", {}))
        if tool_name not in BROWSER_TOOL_NAMES:
            continue
        if tool_id in error_tool_call_ids or _is_browser_error_result(content):
            continue

        messages[i] = {
            **msg,
            "content": _auto_archive_placeholder(tool_name, _args_desc(raw_args)),
        }
        archived += 1

    if archived:
        print(
            f"[AUTO_ARCHIVE] Archived {archived} old browser result(s) "
            f"(older than {force_archive_after_turns} assistant turns, round={round_num})"
        )
    return archived


def _gptoss_content_text(content) -> str:
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    return str(content or "")


def _gptoss_parse_args(content) -> dict:
    raw = _gptoss_content_text(content).strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except Exception:
        return {"raw": raw}


def _gptoss_tool_call_info_by_result_idx(msg_dicts: List[dict]) -> dict:
    pending_calls: list[tuple[str, dict]] = []
    result: dict[int, tuple[str, dict]] = {}

    for idx, msg in enumerate(msg_dicts):
        role = msg.get("role")
        if role == "assistant":
            recipient = str(msg.get("recipient", "") or "")
            if recipient in BROWSER_TOOL_NAMES:
                pending_calls.append((recipient, _gptoss_parse_args(msg.get("content", ""))))
            continue

        if role != "tool":
            continue
        tool_name = str(msg.get("name", "") or "")
        if tool_name not in BROWSER_TOOL_NAMES:
            continue

        matched_pos = None
        for pos in range(len(pending_calls) - 1, -1, -1):
            if pending_calls[pos][0] == tool_name:
                matched_pos = pos
                break
        if matched_pos is None:
            continue

        result[idx] = pending_calls.pop(matched_pos)

    return result


def _auto_archive_old_results_gptoss(
    msg_dicts: List[dict],
    round_num: int,
    force_archive_after_turns: int,
) -> List[dict]:
    """Replace old successful browser tool results for GPT-OSS dict messages."""
    if force_archive_after_turns <= 0:
        return msg_dicts

    asst_has_success: set[int] = set()
    for idx, msg in enumerate(msg_dicts):
        if msg.get("role") != "tool":
            continue
        tool_name = str(msg.get("name", "") or "")
        if tool_name not in BROWSER_TOOL_NAMES:
            continue
        content_text = _gptoss_content_text(msg.get("content", ""))
        if _is_browser_error_result(content_text):
            continue
        for j in range(idx - 1, -1, -1):
            if msg_dicts[j].get("role") == "assistant":
                asst_has_success.add(j)
                break

    effective_asst_indices = [
        i for i, msg in enumerate(msg_dicts)
        if msg.get("role") == "assistant" and i in asst_has_success
    ]
    if len(effective_asst_indices) < force_archive_after_turns:
        return msg_dicts

    cutoff_idx = effective_asst_indices[-force_archive_after_turns]
    result = list(msg_dicts)
    call_info_by_idx = _gptoss_tool_call_info_by_result_idx(result)
    archived = 0

    for i in range(cutoff_idx):
        msg = result[i]
        if msg.get("role") != "tool":
            continue

        tool_name = str(msg.get("name", "") or "")
        if tool_name not in BROWSER_TOOL_NAMES:
            continue

        content_text = _gptoss_content_text(msg.get("content", ""))
        if _is_archived(content_text) or _is_browser_error_result(content_text):
            continue

        _, call_args = call_info_by_idx.get(i, (tool_name, {}))
        result[i] = {
            **msg,
            "content": _auto_archive_placeholder(tool_name, _args_desc(call_args)),
        }
        archived += 1

    if archived:
        print(
            f"[AUTO_ARCHIVE] Archived {archived} old browser result(s) "
            f"(gptoss, round={round_num})"
        )
    return result
