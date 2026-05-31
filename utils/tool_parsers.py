import json5
import re
from typing import Any


def coerce_tool_arguments(raw_args: Any) -> Any:
    """Parse tool arguments if the model emitted them as a JSON string."""
    if isinstance(raw_args, str):
        try:
            return json5.loads(raw_args)
        except Exception:
            return raw_args
    return raw_args


def normalise_tool_call_obj(obj: Any) -> list:
    """Convert JSON tool-call variants into [{'name': ..., 'arguments': ...}]."""
    if isinstance(obj, list):
        calls = []
        for item in obj:
            calls.extend(normalise_tool_call_obj(item))
        return calls

    if not isinstance(obj, dict):
        return []

    if isinstance(obj.get("tool_calls"), list):
        calls = []
        for item in obj["tool_calls"]:
            calls.extend(normalise_tool_call_obj(item))
        return calls

    if isinstance(obj.get("function"), dict):
        fn = obj["function"]
        name = fn.get("name", "")
        args = coerce_tool_arguments(fn.get("arguments", {}))
    else:
        name = obj.get("name", "")
        args = coerce_tool_arguments(obj.get("arguments", {}))

    return [{"name": name, "arguments": args}] if name else []


def parse_xml_tool_calls(tool_call_text: str) -> list:
    """Parse Qwen-style XML tool calls; supports one or more <function=...> blocks."""
    matches = list(re.finditer(
        r'<function=([\w.]+)>(.*?)(?:</function>|$)',
        tool_call_text,
        re.DOTALL,
    ))
    if not matches:
        return []

    calls = []
    for func_match in matches:
        tool_name = func_match.group(1)
        body = func_match.group(2)
        tool_args = {}
        for p in re.finditer(
            r'<parameter=([\w]+)>\s*(.*?)\s*</parameter>',
            body,
            re.DOTALL,
        ):
            pv = p.group(2).strip()
            if pv.startswith('"') and pv.endswith('"'):
                pv = pv[1:-1]
            try:
                pv = json5.loads(pv)
            except Exception:
                if isinstance(pv, str) and pv.isdigit():
                    pv = int(pv)
            tool_args[p.group(1)] = pv
        calls.append({"name": tool_name, "arguments": tool_args})
    return calls


def parse_tool_call_block(tool_call_text: str) -> list:
    """Parse one <tool_call> body into normalized tool calls."""
    try:
        parsed = json5.loads(tool_call_text)
        calls = normalise_tool_call_obj(parsed)
        if calls:
            print(f"[NATIVE_TOOLS] Parsed tool call block (JSON): {parsed}")
            return calls
    except Exception as e:
        print(f"[NATIVE_TOOLS] JSON parsing failed, trying XML format: {e}")

    calls = parse_xml_tool_calls(tool_call_text)
    if calls:
        print(f"[NATIVE_TOOLS] Parsed tool call block (XML): {calls}")
        return calls

    print(f"[NATIVE_TOOLS] Failed to parse tool call: {tool_call_text}")
    return []


def build_parsed_tool_calls(raw_calls: list, round_num: int) -> list:
    """Attach stable ids and OpenAI-compatible shape to normalized tool calls."""
    parsed = []
    total = len(raw_calls)
    for idx, call in enumerate(raw_calls, start=1):
        call_id = f"{round_num}" if total == 1 else f"{round_num}_{idx}"
        parsed.append({
            "id": call_id,
            "type": "function",
            "function": {
                "name": call.get("name", ""),
                "arguments": call.get("arguments", {}),
            },
        })
    return parsed


def is_parallel_tool_call_batch(tool_calls: list) -> bool:
    """True for multiple tool calls or a browser.search with a list query."""
    if not tool_calls:
        return False
    if len(tool_calls) > 1:
        return True
    fn = tool_calls[0].get("function", {})
    if fn.get("name") != "browser.search":
        return False
    args = coerce_tool_arguments(fn.get("arguments", {}))
    return isinstance(args, dict) and isinstance(args.get("query"), list) and len(args["query"]) > 1


QWEN_STYLE_PARALLEL_TOOL_INSTRUCTION = (
    "When several function calls are independent, you may issue parallel "
    "tool calls by outputting multiple consecutive `<tool_call>...</tool_call>` "
    "blocks in the same assistant turn, one function call per block."
)


# ── DeepSeek tool-name normalisation ─────────────────────────────────────────
# DeepSeek API rejects names that don't match ^[a-zA-Z0-9_-]+$ (no dots).
# We map dotted names → underscore names before sending and reverse on receipt.

_TOOL_NAME_TO_API: dict[str, str] = {
    "browser.search": "browser_search",
    "browser.open":   "browser_open",
    "browser.find":   "browser_find",
    "browser.scroll": "browser_scroll",
}
_TOOL_NAME_FROM_API: dict[str, str] = {v: k for k, v in _TOOL_NAME_TO_API.items()}


def normalize_tool_name(name: str) -> str:
    """browser.search → browser_search (for APIs that forbid dots in names)."""
    return _TOOL_NAME_TO_API.get(name, name.replace(".", "_"))


def denormalize_tool_name(name: str) -> str:
    """browser_search → browser.search (convert back after API call)."""
    return _TOOL_NAME_FROM_API.get(name, name)


def normalize_tools_for_api(tools: list) -> list:
    """Return a copy of the tools list with all function names normalised."""
    out = []
    for tool in tools:
        t = dict(tool)
        if "function" in t:
            f = dict(t["function"])
            f["name"] = normalize_tool_name(f["name"])
            t["function"] = f
        out.append(t)
    return out


# DeepSeek stop-token string (signals end of tool-call block)
DEEPSEEK_TOOL_CALL_STOP = "<｜tool▁calls▁end｜>"


def parse_deepseek_tool_calls(text: str) -> list:
    """Parse DeepSeek's <｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜> format.

    Each call block:
        <｜tool▁call▁begin｜>function<｜tool▁sep｜>tool_name
        ```json
        {"arg": "val"}
        ```<｜tool▁call▁end｜>
    """
    calls = []
    # Split on tool_call_begin delimiter
    blocks = re.split(r'<｜tool▁call▁begin｜>', text)
    for block in blocks[1:]:
        sep_idx = block.find('<｜tool▁sep｜>')
        if sep_idx == -1:
            continue
        # Everything between tool▁sep and tool▁call▁end is "type\nname\n```json\n{...}\n```"
        after_sep = block[sep_idx + len('<｜tool▁sep｜>'):]
        # Name is up to first newline
        nl = after_sep.find('\n')
        tool_name = after_sep[:nl].strip() if nl != -1 else after_sep.strip()
        tool_name = denormalize_tool_name(tool_name)
        remainder = after_sep[nl + 1:] if nl != -1 else ""
        # Extract JSON from ```json ... ``` block
        json_match = re.search(r'```json\s*(.*?)\s*```', remainder, re.DOTALL)
        if json_match:
            raw_args = json_match.group(1).strip()
        else:
            raw_args = remainder.split('<｜tool▁call▁end｜>')[0].strip()
        try:
            args = json5.loads(raw_args)
        except Exception:
            args = {"query": raw_args} if tool_name in ("browser.search", "browser_search") else {}
        calls.append({"name": tool_name, "arguments": args})
    return calls


DSML_TOKEN = "｜DSML｜"


def _decode_dsml_parameter(value: str, is_string: str) -> Any:
    if is_string == "true":
        return value
    try:
        return json5.loads(value)
    except Exception:
        return value


def parse_deepseek_v4_dsml_tool_calls(text: str) -> list:
    """Parse DeepSeek-V4 DSML tool calls.

    Expected form:
        <｜DSML｜tool_calls>
        <｜DSML｜invoke name="browser_search">
        <｜DSML｜parameter name="query" string="true">...</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜tool_calls>
    """
    calls = []
    block_re = re.compile(
        rf'<{re.escape(DSML_TOKEN)}tool_calls>\s*(.*?)\s*</{re.escape(DSML_TOKEN)}tool_calls>',
        re.DOTALL,
    )
    invoke_re = re.compile(
        rf'<{re.escape(DSML_TOKEN)}invoke\s+name="([^"]+)">\s*(.*?)\s*</{re.escape(DSML_TOKEN)}invoke>',
        re.DOTALL,
    )
    param_re = re.compile(
        rf'<{re.escape(DSML_TOKEN)}parameter\s+name="([^"]+)"\s+string="(true|false)">(.*?)</{re.escape(DSML_TOKEN)}parameter>',
        re.DOTALL,
    )

    for block in block_re.findall(text):
        for tool_name, body in invoke_re.findall(block):
            args = {}
            for key, is_string, value in param_re.findall(body):
                args[key] = _decode_dsml_parameter(value, is_string)
            calls.append({"name": denormalize_tool_name(tool_name), "arguments": args})
    return calls


def strip_deepseek_v4_dsml_tool_calls(text: str) -> str:
    """Remove DeepSeek-V4 DSML tool-call blocks from assistant content."""
    return re.sub(
        rf'\s*<{re.escape(DSML_TOKEN)}tool_calls>.*?</{re.escape(DSML_TOKEN)}tool_calls>\s*',
        '',
        text,
        flags=re.DOTALL,
    ).strip()


def parallel_tool_instruction_for_model(tokenizer: Any, model_id: str) -> str:
    """Return a template-specific parallel tool-call hint, or empty if unsupported."""
    model_id = (model_id or "").lower()

    # DeepSeek: uses /chat/completions with native tool_calls — model already does parallel
    if "deepseek" in model_id:
        return (
            "When several function calls are independent, issue them as MULTIPLE parallel "
            "tool calls in the same response."
        )

    template = str(getattr(tokenizer, "chat_template", "") or "")
    if "<tool_call>" in template and "<function=" in template:
        return QWEN_STYLE_PARALLEL_TOOL_INSTRUCTION
    # JSON-in-<tool_call> templates, e.g. Nemotron/OpenResearcher and MiroThinker.
    if "<tool_call>" in template:
        return (
            "When several function calls are independent, issue them in parallel "
            "by outputting multiple consecutive <tool_call>...</tool_call> blocks "
            "in the same response, one function call per block."
        )
    return ""
