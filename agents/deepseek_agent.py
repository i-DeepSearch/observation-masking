import datetime
import json
import re
import time
from typing import Any, List

from tools.context_management import _auto_archive_old_results
from utils.prompts import DEVELOPER_CONTENT, TOOL_CONTENT
from utils.tool_parsers import (
    build_parsed_tool_calls,
    denormalize_tool_name,
    is_parallel_tool_call_batch,
    normalize_tools_for_api,
    parallel_tool_instruction_for_model,
    parse_deepseek_v4_dsml_tool_calls,
    strip_deepseek_v4_dsml_tool_calls,
)


async def _generate_deepseek_round(
    generator: Any,
    messages: List[dict],
    api_tools: list,
    round_num: int,
) -> tuple:
    """Generate one round for DeepSeek using /chat/completions native tool calling."""
    api_msgs = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        entry: dict = {"role": role, "content": content}
        if m.get("reasoning_content"):
            entry["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            tc_list = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                tc_list.append({
                    "id": tc.get("id", f"call_{round_num}"),
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args},
                })
            entry["tool_calls"] = tc_list
        if m.get("tool_call_id"):
            entry["tool_call_id"] = m["tool_call_id"]
        api_msgs.append(entry)

    response = await generator.chat_completion(
        messages=api_msgs,
        tools=api_tools,
        tool_choice="auto",
        temperature=1.0,
        max_tokens=8192,
        thinking_mode="thinking",
        reasoning_effort="max",
    )

    choice = response["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning_content = msg.get("reasoning_content") or None
    api_tool_calls = msg.get("tool_calls") or []
    usage = response.get("usage", {})
    n_output = usage.get("completion_tokens", 0)

    parsed_tool_calls = None
    parallel_tool_calls = False
    if api_tool_calls:
        raw = []
        for tc in api_tool_calls:
            fn = tc.get("function", {})
            name = denormalize_tool_name(fn.get("name", ""))
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except Exception:
                args = {}
            raw.append({"name": name, "arguments": args})
        parsed_tool_calls = build_parsed_tool_calls(raw, round_num) or None
        parallel_tool_calls = is_parallel_tool_call_batch(parsed_tool_calls or [])

    return content, reasoning_content, parsed_tool_calls, parallel_tool_calls, n_output


def _deepseek_text_tool_instruction(api_tools: list) -> str:
    """Official DeepSeek-V4 DSML tool instruction for non-native tool calling."""
    tool_schemas = "\n".join(
        json.dumps(tool["function"], ensure_ascii=False)
        for tool in api_tools
        if "function" in tool
    )
    return (
        "## Tools\n\n"
        "You have access to a set of tools to help answer the user's question. "
        "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n"
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
        "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        "<｜DSML｜invoke name=\"$TOOL_NAME2\">\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>\n"
        "String parameters should be specified as is and set `string=\"true\"`. "
        "For all other types (numbers, booleans, arrays, objects), pass the value "
        "in JSON format and set `string=\"false\"`.\n"
        "If thinking_mode is enabled (triggered by <think>), you MUST output your "
        "complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n"
        "Otherwise, output directly after </think> with tool calls or final response.\n"
        "### Available Tool Schemas\n"
        f"{tool_schemas}\n"
        "You MUST strictly follow the above defined tool name and parameter schemas "
        "to invoke tool calls."
    )


async def _generate_deepseek_text_round(
    generator: Any,
    messages: List[dict],
    round_num: int,
) -> tuple:
    """DeepSeek generation without API-native tools; parse text tool-call blocks locally."""
    response = await generator.chat_completion(
        messages=messages,
        tools=None,
        tool_choice="none",
        temperature=1.0,
        max_tokens=8192,
        thinking_mode="thinking",
        reasoning_effort="max",
    )

    choice = response["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning_content = msg.get("reasoning_content") or msg.get("reasoning") or None
    if reasoning_content is None:
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if think_match:
            reasoning_content = think_match.group(1).strip()
            content = content.replace(think_match.group(0), "", 1).strip()
    usage = response.get("usage", {})
    n_output = usage.get("completion_tokens", 0)

    raw_tool_calls = parse_deepseek_v4_dsml_tool_calls(content)
    parsed_tool_calls = build_parsed_tool_calls(raw_tool_calls, round_num) or None
    parallel_tool_calls = is_parallel_tool_call_batch(parsed_tool_calls or [])

    if parsed_tool_calls:
        content = strip_deepseek_v4_dsml_tool_calls(content)

    return content, reasoning_content, parsed_tool_calls, parallel_tool_calls, n_output


async def run_one_deepseek(
    question: str,
    qid: Any,
    generator: Any,
    browser_pool,
    force_archive_after_turns: int,
    max_rounds: int = 200,
    enable_parallel_tool_calls: bool = True,
) -> tuple:
    """Agent loop for DeepSeek-V4 using DSML text tool calls."""
    tool_config = browser_pool.init_session(qid)

    if hasattr(generator, '_init_tokenizer'):
        await generator._init_tokenizer()

    model_id = (getattr(generator, 'model_name', '') or '').lower()
    parallel_tool_instruction = (
        parallel_tool_instruction_for_model(generator.tokenizer, model_id)
        if enable_parallel_tool_calls
        else ""
    )

    system_prompt = (
        DEVELOPER_CONTENT
        + f"\n\nToday's date: {datetime.datetime.now().strftime('%Y-%m-%d')}"
        + (f"\n\n{parallel_tool_instruction}" if parallel_tool_instruction else "")
    )
    init_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    messages = list(init_msgs)
    full_messages = list(init_msgs)
    tool_error_call_ids: set[str] = set()

    tools = json.loads(TOOL_CONTENT)
    api_tools = normalize_tools_for_api(tools)
    messages[0]["content"] += "\n\n" + _deepseek_text_tool_instruction(api_tools)
    full_messages[0]["content"] = messages[0]["content"]

    round_num = 0
    turn_stats: list = []
    last_input_tokens = 0
    prev_input_tokens = 0

    try:
        while round_num < max_rounds:
            round_num += 1

            print(f"\n{'='*60}")
            print(f"Round {round_num}")
            print(f"{'='*60}")

            _auto_archive_old_results(
                messages,
                round_num,
                force_archive_after_turns,
                tool_error_call_ids,
            )

            text_for_count = " ".join(str(m.get("content") or "") for m in messages)
            tokens = generator.tokenizer.encode(text_for_count, add_special_tokens=False)
            current_token_count = len(tokens)
            print(f"[CONTEXT] ~{current_token_count} tokens, {len(messages)} messages")

            last_input_tokens = current_token_count
            t_round_start = time.time()

            (content, reasoning_content, parsed_tool_calls,
             parallel_tool_calls, n_output_tokens) = await _generate_deepseek_text_round(
                generator, messages, round_num
            )
            cached_est = 0
            turn_stats.append({
                "round": round_num,
                "input_tokens": current_token_count,
                "output_tokens": n_output_tokens,
                "latency_s": round(time.time() - t_round_start, 3),
                "cached_input_tokens_est": cached_est,
            })
            prev_input_tokens = current_token_count

            print(f"[DEEPSEEK] tool_calls={len(parsed_tool_calls or [])}  parallel={parallel_tool_calls}")
            assistant_msg = {
                "role": "assistant",
                "content": content,
                "reasoning_content": reasoning_content,
                "tool_calls": parsed_tool_calls,
                "parallel_tool_calls": parallel_tool_calls,
            }
            messages.append(assistant_msg)
            full_messages.append({**assistant_msg})

            if parsed_tool_calls:
                print(f"[DEEPSEEK] Tool calls: {len(parsed_tool_calls)}")
                if parallel_tool_calls:
                    print("[DEEPSEEK] Detected parallel tool-call batch")

                def _append_tool_result(tool_call: dict, function_name: str, result: str, is_error: bool = False) -> None:
                    tool_id = tool_call["id"]
                    if is_error:
                        tool_error_call_ids.add(tool_id)
                        tool_err_msg = {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result,
                        }
                        messages.append(tool_err_msg)
                        full_messages.append(tool_err_msg)
                        return
                    tool_error_call_ids.discard(tool_id)

                    max_tool_result_chars = 12000
                    result_for_ctx = result if len(result) <= max_tool_result_chars \
                        else result[:max_tool_result_chars] + "\n...[truncated]"

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result_for_ctx,
                    }
                    messages.append(tool_msg)
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result,
                    })

                def _parse_tool_args(tool_call: dict) -> dict:
                    function_args_raw = tool_call["function"]["arguments"]
                    if isinstance(function_args_raw, dict):
                        return function_args_raw
                    return json.loads(function_args_raw)

                def _append_tool_error(tool_call: dict, function_name: str, error: Exception | str) -> None:
                    error_msg = f"Error executing {function_name}: {str(error)}"
                    print(f"[DEEPSEEK] Error: {error_msg}")
                    tool_error_call_ids.add(tool_call["id"])
                    tool_err_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": error_msg,
                    }
                    messages.append(tool_err_msg)
                    full_messages.append(tool_err_msg)

                async def _execute_browser_batch(browser_batch: list[tuple[dict, str, dict]]) -> None:
                    if not browser_batch:
                        return

                    print(f"[DEEPSEEK] Executing {len(browser_batch)} browser tool calls concurrently")
                    specs = []
                    for tool_call, function_name, function_args in browser_batch:
                        actual_function_name = function_name.split(".", 1)[1].lower()
                        specs.append({
                            "tool_name": actual_function_name,
                            "tool_args": function_args,
                        })
                        print(f"\n[DEEPSEEK] === Parallel Browser Tool Call ===")
                        print(f"[DEEPSEEK] Tool ID: {tool_call['id']}")
                        print(f"[DEEPSEEK] Function: {function_name}")
                        print(f"[DEEPSEEK] Arguments (full):\n{json.dumps(function_args, indent=2, ensure_ascii=False)}")

                    search_calls = [
                        item for item in browser_batch
                        if item[1].split(".", 1)[1].lower() == "search"
                    ]
                    search_reasoning = reasoning_content
                    if search_calls:
                        print(
                            "[DEEPSEEK] Search reasoning context: "
                            f"{'enabled' if search_reasoning else 'disabled'} "
                            f"(round={round_num}, browser_batch_size={len(browser_batch)})"
                        )

                    results = await browser_pool.call_browser_tools_concurrently(
                        qid,
                        specs,
                        reasoning=search_reasoning,
                    )
                    for (tool_call, function_name, _), result in zip(browser_batch, results):
                        _append_tool_result(
                            tool_call,
                            function_name,
                            result if result else f"{function_name} completed",
                            is_error=(result or "").startswith(("Error executing ", "Error rendering ")),
                        )
                        print(f"[DEEPSEEK] Tool Result (full):\n{result}")
                    print("[DEEPSEEK] === End Parallel Browser Tool Batch ===\n")

                browser_batch: list[tuple[dict, str, dict]] = []
                for tool_call in parsed_tool_calls:
                    function_name = tool_call["function"]["name"]
                    tool_name_map = {
                        "search": "browser.search",
                        "visit": "browser.open",
                        "web_search": "browser.search",
                        "web_browse": "browser.open",
                    }
                    if function_name in tool_name_map:
                        function_name = tool_name_map[function_name]
                        tool_call["function"]["name"] = function_name

                    try:
                        function_args = _parse_tool_args(tool_call)
                        print(f"\n[DEEPSEEK] === Tool Call ===")
                        print(f"[DEEPSEEK] Tool ID: {tool_call['id']}")
                        print(f"[DEEPSEEK] Function: {function_name}")
                        print(f"[DEEPSEEK] Arguments (full):\n{json.dumps(function_args, indent=2, ensure_ascii=False)}")

                        if function_name.startswith("browser."):
                            actual_function_name = function_name.split(".", 1)[1]
                            if actual_function_name.lower() in ['search', 'find', 'open']:
                                query_val = function_args.get("query") if function_name == "browser.search" else None
                                if isinstance(query_val, list) and len(query_val) > 1:
                                    for sub_idx, single_q in enumerate(query_val):
                                        sub_args = {**function_args, "query": single_q}
                                        sub_tc = {
                                            **tool_call,
                                            "id": f"{tool_call['id']}_{sub_idx}",
                                            "function": {
                                                **tool_call["function"],
                                                "arguments": json.dumps(sub_args, ensure_ascii=False),
                                            },
                                        }
                                        browser_batch.append((sub_tc, function_name, sub_args))
                                    if parsed_tool_calls:
                                        expanded = []
                                        for tc in parsed_tool_calls:
                                            if tc["id"] == tool_call["id"] and tc["function"]["name"] == "browser.search":
                                                for sub_idx, single_q in enumerate(query_val):
                                                    sub_args = {**function_args, "query": single_q}
                                                    expanded.append({
                                                        **tc,
                                                        "id": f"{tc['id']}_{sub_idx}",
                                                        "function": {
                                                            **tc["function"],
                                                            "arguments": sub_args,
                                                        },
                                                    })
                                            else:
                                                expanded.append(tc)
                                        for m in reversed(messages):
                                            if m.get("role") == "assistant" and m.get("tool_calls"):
                                                m["tool_calls"] = expanded
                                                break
                                        for m in reversed(full_messages):
                                            if m.get("role") == "assistant" and m.get("tool_calls"):
                                                m["tool_calls"] = expanded
                                                break
                                else:
                                    browser_batch.append((tool_call, function_name, function_args))
                                continue

                            await _execute_browser_batch(browser_batch)
                            browser_batch = []
                            _append_tool_result(tool_call, function_name, f"Tool {function_name} not available")
                            continue

                        await _execute_browser_batch(browser_batch)
                        browser_batch = []
                        _append_tool_result(tool_call, function_name, f"Tool {function_name} not available")

                    except Exception as e:
                        await _execute_browser_batch(browser_batch)
                        browser_batch = []
                        _append_tool_error(tool_call, function_name, e)

                await _execute_browser_batch(browser_batch)
                continue

            content_lower = content.lower()
            if '<answer>' in content_lower and '</answer>' in content_lower:
                print(f"\n✅ Found <answer> tag - conversation completed")
                break

            if "exact answer:" in content_lower and "confidence:" in content_lower:
                print(f"\n✅ Found 'Exact Answer:' and 'Confidence:' - conversation completed")
                break

            if "final answer:" in content_lower or "answer:" in content_lower:
                print(f"\n✅ Found 'Final Answer:' or 'Answer:' - conversation completed")
                break

        total_input = sum(s["input_tokens"] for s in turn_stats)
        total_output = sum(s["output_tokens"] for s in turn_stats)
        total_cached_est = sum(s.get("cached_input_tokens_est", 0) for s in turn_stats)
        token_summary = {
            "final_messages_tokens": last_input_tokens,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cached_input_tokens_est": total_cached_est,
        }
        visited_urls = browser_pool.get_visited_urls(qid)
        return full_messages, list(messages), turn_stats, token_summary, visited_urls

    finally:
        browser_pool.cleanup(qid)
