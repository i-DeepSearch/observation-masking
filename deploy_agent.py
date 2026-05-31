# -*- coding: utf-8 -*-
import argparse
import os, json, time, traceback, glob
from typing import List, Any
import multiprocessing as mp
import asyncio
import datetime
import tqdm
from tools.browser import BrowserPool
from utils.data_utils import load_dataset, list_available_datasets
from utils.prompts import DEVELOPER_CONTENT, TOOL_CONTENT
from tools.context_management import _auto_archive_old_results
from agents.deepseek_agent import run_one_deepseek
from agents.gptoss_agent import run_one_gptoss
from utils.tool_parsers import (
    build_parsed_tool_calls,
    is_parallel_tool_call_batch,
    parallel_tool_instruction_for_model,
    parse_tool_call_block,
)
import dotenv
import re

dotenv.load_dotenv()

os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BROWSECOMP_PLUS_QID_ORDER_FILE = os.path.join(
    PROJECT_ROOT,
    "results/browsecomp-plus/qid_preemption_aware_order.csv",
)

# Pre-import transformers in main process to avoid multiprocessing issues
try:
    import transformers
    print(f"Pre-loaded transformers version: {transformers.__version__}")
except ImportError:
    print("Warning: transformers not available")

def _load_qid_order(path: str) -> List[Any]:
    """Load qid order from a JSON list or a CSV with a qid column."""
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            order = json.load(f)
        if not isinstance(order, list):
            raise ValueError(f"qid order JSON must contain a list: {path}")
        return order

    if path.endswith(".csv"):
        import csv
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "qid" not in (reader.fieldnames or []):
                raise ValueError(f"qid order CSV must contain a 'qid' column: {path}")
            return [row["qid"] for row in reader]

    raise ValueError(f"Unsupported qid order file format: {path}. Use .json or .csv")


def _apply_qid_order(data: list, qid_order_file: str) -> list:
    order = _load_qid_order(qid_order_file)
    rank = {str(qid): i for i, qid in enumerate(order)}
    original_pos = {str(item["qid"]): i for i, item in enumerate(data)}
    ordered = sorted(
        data,
        key=lambda item: (
            rank.get(str(item["qid"]), len(rank) + original_pos[str(item["qid"])]),
            original_pos[str(item["qid"])],
        ),
    )
    matched = sum(1 for item in data if str(item["qid"]) in rank)
    print(
        f"[QID_ORDER] Applied {qid_order_file}: matched {matched}/{len(data)} qids; "
        f"unmatched items kept after ordered items in original order."
    )
    return ordered


def _default_qid_order_file_for_dataset(dataset_name: str) -> str | None:
    dataset_key = (dataset_name or "").replace("_", "-")
    if dataset_key != "browsecomp-plus":
        return None
    if os.path.exists(DEFAULT_BROWSECOMP_PLUS_QID_ORDER_FILE):
        return DEFAULT_BROWSECOMP_PLUS_QID_ORDER_FILE
    return None


def _build_prompt(generator: Any, messages: List[dict], tools: list) -> tuple:
    """
    Apply chat template and return (prompt_text, token_list).
    Detects once whether the tokenizer supports save_history_reasoning_content
    (LongCat feature) and caches the result on the generator object to avoid
    a double apply_chat_template call every round.
    """
    template_kwargs = dict(tools=tools, tokenize=False, add_generation_prompt=True)

    if not hasattr(generator, '_supports_save_history_rc'):
        try:
            generator.tokenizer.apply_chat_template(
                messages[:2],
                save_history_reasoning_content=False,
                **template_kwargs
            )
            generator._supports_save_history_rc = True
        except TypeError:
            generator._supports_save_history_rc = False

    if generator._supports_save_history_rc:
        prompt = generator.tokenizer.apply_chat_template(
            messages,
            save_history_reasoning_content=True,
            **template_kwargs
        )
    else:
        prompt = generator.tokenizer.apply_chat_template(messages, **template_kwargs)

    tokens = generator.tokenizer.encode(prompt, add_special_tokens=False)
    return prompt, tokens


async def _generate_with_retry(
    generator: Any,
    tokens: List[int],
    stop_strings: List[str],
    max_retries: int = 20
) -> str:
    """
    HuggingFace-based generation (interface matches _generate_with_retry)
    Args:
        generator: Generator (vLLMAsyncGenerator or OpenAIAsyncGenerator) with tokenizer
        tokens: Pre-tokenized input
        stop_strings: Stop strings (e.g., ["\n<tool_response>", "<tool_response>"])
        max_retries: Max retry attempts
    Returns:
        Generated text string
    """
    assert max_retries > 0
    last_exception = None

    # Retry only the generation part
    for attempt in range(1, max_retries + 1):
        stream = generator.generate(tokens, stop_strings=stop_strings)
        try:
            # Generate and collect tokens with client-side stop checking
            generated_tokens = []
            accumulated_text = ""

            async for token_id in stream:
                generated_tokens.append(token_id)

                # Periodically check for stop strings (every 10 tokens)
                if len(generated_tokens) % 10 == 0:
                    accumulated_text = generator.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    # Check if we hit any stop string
                    for stop_str in stop_strings:
                        if stop_str in accumulated_text:
                            print(f"[DEBUG] Client-side stop detected: found '{stop_str}' in generated text")
                            break
                    else:
                        continue
                    break

            # Final decode
            generated_text = generator.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Remove any stop strings from the end
            for stop_str in stop_strings:
                if stop_str in generated_text:
                    pos = generated_text.find(stop_str)
                    generated_text = generated_text[:pos]

            print(f"[DEBUG] Generated {len(generated_tokens)} tokens, text length: {len(generated_text)}")
            return generated_text, len(generated_tokens)

        except Exception as e:
            last_exception = e
            print(f"\n--- Generation failed on attempt {attempt}/{max_retries} ---")
            import traceback as _tb
            print(_tb.format_exc())

        finally:
            try:
                await stream.aclose()
            except Exception:
                pass

    if last_exception:
        raise last_exception
    raise RuntimeError("Generation failed after retries without a captured exception.")


async def run_one(
    question: str,
    qid: Any,
    generator: Any,
    browser_pool: BrowserPool,
    force_archive_after_turns: int,
    max_rounds: int = 200,
    enable_parallel_tool_calls: bool = True,
) -> List[dict]:
    """
    Helper function for native tool calling using tokenizer's chat template
    Uses tokenizer.apply_chat_template with tools parameter instead of OpenAI API
    """
    # Initialize browser session
    tool_config = browser_pool.init_session(qid)

    # Initialize tokenizer
    if hasattr(generator, '_init_tokenizer'):
        await generator._init_tokenizer()

    _model_id = (getattr(generator, 'model_name', '') or '').lower()
    parallel_tool_instruction = (
        parallel_tool_instruction_for_model(generator.tokenizer, _model_id)
        if enable_parallel_tool_calls
        else ""
    )

    # Initialize messages (Standard approach)
    system_prompt = (
        DEVELOPER_CONTENT
        + f"\n\nToday's date: {datetime.datetime.now().strftime('%Y-%m-%d')}"
        + (f"\n\n{parallel_tool_instruction}" if parallel_tool_instruction else "")
    )
    init_msgs = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question,
        }
    ]
    # messages: working copy fed to the model, with old browser results auto-archived
    # full_messages: complete, unmodified record saved to JSONL
    messages = list(init_msgs)
    full_messages = list(init_msgs)
    tool_error_call_ids: set[str] = set()

    tools = json.loads(TOOL_CONTENT)
    stop_strings = ["\n<tool_response>", "<tool_response>"]

    round_num = 0
    turn_stats: list = []   # per-round timing + token stats
    last_input_tokens = 0   # tracks final-round input token count
    prev_input_tokens = 0   # previous round's input — used as cache hit estimate

    try:
        while round_num < max_rounds:
            round_num += 1

            print(f"\n{'='*60}")
            print(f"Round {round_num}")
            print(f"{'='*60}")

            # Compact old browser results in the working context. The unmodified
            # originals remain in full_messages for the saved JSONL record.
            _auto_archive_old_results(
                messages,
                round_num,
                force_archive_after_turns,
                tool_error_call_ids,
            )

            # Count tokens for telemetry only.
            _, tokens = _build_prompt(generator, messages, tools)
            current_token_count = len(tokens)
            print(f"[CONTEXT] ~{current_token_count} tokens, {len(messages)} messages")

            last_input_tokens = current_token_count

            t_round_start = time.time()

            content, n_output_tokens = await _generate_with_retry(generator, tokens, stop_strings)
            cached_est = min(current_token_count, prev_input_tokens)
            turn_stats.append({
                "round": round_num,
                "input_tokens": current_token_count,
                "output_tokens": n_output_tokens,
                "latency_s": round(time.time() - t_round_start, 3),
                "cached_input_tokens_est": cached_est,
            })
            prev_input_tokens = current_token_count
            non_thinking_content = content

            print(f'[NATIVE_TOOLS] Round {round_num}: {content[:500] if len(content) > 500 else content}')

            reasoning_content = None
            parsed_tool_calls = None
            tool_call_text = None
            parallel_tool_calls = False

            # ── Qwen3 / default parsing ───────────────────────────────────────
            # Remove tool_response marker if present
            if '<tool_response>' in content:
                content = content[:content.find('<tool_response>')]

            # Step 1: Extract and remove <think> tags
            if '<think>' in content and '</think>' in content:
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                if think_match:
                    reasoning_content = think_match.group(1).strip()
                    content = content.replace(think_match.group(0), "").strip()
            elif '</think>' in content:
                think_match = re.search(r'^(.*?)</think>', content, re.DOTALL)
                if think_match:
                    reasoning_content = think_match.group(1).strip()
                    content = content.replace(think_match.group(0), "").strip()

            # Step 2: Extract and remove one or more <tool_call> tags.
            tool_call_blocks = []
            tool_call_matches = list(re.finditer(
                r'<tool_call>(.*?)</tool_call>',
                content,
                re.DOTALL,
            ))
            if tool_call_matches:
                for m in tool_call_matches:
                    tool_call_blocks.append(m.group(1).strip())
                content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
            elif '</tool_call>' in content:
                tool_call_match = re.search(r'^(.*?)</tool_call>', content, re.DOTALL)
                if tool_call_match:
                    tool_call_blocks.append(tool_call_match.group(1).strip())
                    content = content.replace(tool_call_match.group(0), "").strip()

            if tool_call_blocks:
                tool_call_text = "\n".join(tool_call_blocks)
                raw_tool_calls = []
                for block in tool_call_blocks:
                    raw_tool_calls.extend(parse_tool_call_block(block))
                parsed_tool_calls = build_parsed_tool_calls(raw_tool_calls, round_num) or None
                parallel_tool_calls = (
                    enable_parallel_tool_calls
                    and is_parallel_tool_call_batch(parsed_tool_calls or [])
                )
                if parsed_tool_calls:
                    print(f"[NATIVE_TOOLS] Parallel tool calls: {parallel_tool_calls}")

            print(f"[NATIVE_TOOLS] Assistant response (cleaned):\n{content}")
            if reasoning_content:
                print(f"[NATIVE_TOOLS] Reasoning content:\n{reasoning_content}")

            if tool_call_text is None:
                non_thinking_content = (
                    non_thinking_content.split('</think>', 1)[1].strip()
                    if '</think>' in non_thinking_content
                    else non_thinking_content.strip()
                )
            assistant_msg = {
                "role": "assistant",
                "content": non_thinking_content if tool_call_text is None else "",
                "reasoning_content": reasoning_content,
                "tool_calls": parsed_tool_calls,
                "parallel_tool_calls": parallel_tool_calls,
            }
            messages.append(assistant_msg)
            full_messages.append({**assistant_msg})

            # Check if there are tool calls
            if parsed_tool_calls:
                print(f"[NATIVE_TOOLS] Tool calls: {len(parsed_tool_calls)}")
                if parallel_tool_calls:
                    print("[NATIVE_TOOLS] Detected parallel tool-call batch")

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

                    # Truncate large tool results for working context. Full
                    # results are preserved in full_messages for JSONL logs.
                    MAX_TOOL_RESULT_CHARS = 12000
                    result_for_ctx = result if len(result) <= MAX_TOOL_RESULT_CHARS \
                        else result[:MAX_TOOL_RESULT_CHARS] + "\n...[truncated]"

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result_for_ctx,
                    }
                    messages.append(tool_msg)
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result,  # full, untruncated for JSONL log
                    })
                def _parse_tool_args(tool_call: dict) -> dict:
                    function_args_raw = tool_call["function"]["arguments"]
                    if isinstance(function_args_raw, dict):
                        return function_args_raw
                    return json.loads(function_args_raw)

                def _append_tool_error(tool_call: dict, function_name: str, error: Exception | str) -> None:
                    error_msg = f"Error executing {function_name}: {str(error)}"
                    print(f"[NATIVE_TOOLS] Error: {error_msg}")
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

                    print(f"[NATIVE_TOOLS] Executing {len(browser_batch)} browser tool calls concurrently")
                    specs = []
                    for tool_call, function_name, function_args in browser_batch:
                        actual_function_name = function_name.split(".", 1)[1].lower()
                        specs.append({
                            "tool_name": actual_function_name,
                            "tool_args": function_args,
                        })
                        print(f"\n[NATIVE_TOOLS] === Parallel Browser Tool Call ===")
                        print(f"[NATIVE_TOOLS] Tool ID: {tool_call['id']}")
                        print(f"[NATIVE_TOOLS] Function: {function_name}")
                        print(f"[NATIVE_TOOLS] Arguments (full):\n{json.dumps(function_args, indent=2, ensure_ascii=False)}")

                    search_calls = [
                        item for item in browser_batch
                        if item[1].split(".", 1)[1].lower() == "search"
                    ]
                    search_reasoning = reasoning_content
                    if search_calls:
                        print(
                            "[NATIVE_TOOLS] Search reasoning context: "
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
                        print(f"[NATIVE_TOOLS] Tool Result (full):\n{result}")
                    print("[NATIVE_TOOLS] === End Parallel Browser Tool Batch ===\n")

                browser_batch: list[tuple[dict, str, dict]] = []
                for tool_call in parsed_tool_calls:
                    function_name = tool_call["function"]["name"]  # e.g., "browser.search"

                    # Map DeepResearch-style tool names → browser.* equivalents
                    _TOOL_NAME_MAP = {
                        "search": "browser.search",
                        "visit":  "browser.open",
                        "web_search": "browser.search",
                        "web_browse": "browser.open",
                    }
                    if function_name in _TOOL_NAME_MAP:
                        function_name = _TOOL_NAME_MAP[function_name]
                        tool_call["function"]["name"] = function_name

                    try:
                        function_args = _parse_tool_args(tool_call)
                        print(f"\n[NATIVE_TOOLS] === Tool Call ===")
                        print(f"[NATIVE_TOOLS] Tool ID: {tool_call['id']}")
                        print(f"[NATIVE_TOOLS] Function: {function_name}")
                        print(f"[NATIVE_TOOLS] Arguments (full):\n{json.dumps(function_args, indent=2, ensure_ascii=False)}")

                        if function_name.startswith("browser."):
                            actual_function_name = function_name.split(".", 1)[1]
                            if actual_function_name.lower() in ['search', 'find', 'open']:
                                # Expand list-query browser.search into concurrent single-query calls.
                                # DeepResearch (and some other models) pass query as a list; we split
                                # each element into its own call so the model sees N distinct results.
                                query_val = function_args.get("query") if function_name == "browser.search" else None
                                if isinstance(query_val, list) and len(query_val) > 1:
                                    for sub_idx, single_q in enumerate(query_val):
                                        sub_args = {**function_args, "query": single_q}
                                        sub_tc = {
                                            **tool_call,
                                            "id": f"{tool_call['id']}_{sub_idx}",
                                            "function": {**tool_call["function"],
                                                         "arguments": json.dumps(sub_args, ensure_ascii=False)},
                                        }
                                        browser_batch.append((sub_tc, function_name, sub_args))
                                    # Patch the assistant message's tool_calls to match the expansion
                                    if parsed_tool_calls:
                                        expanded = []
                                        for tc in parsed_tool_calls:
                                            if tc["id"] == tool_call["id"] and tc["function"]["name"] == "browser.search":
                                                for sub_idx, single_q in enumerate(query_val):
                                                    sub_args = {**function_args, "query": single_q}
                                                    expanded.append({
                                                        **tc,
                                                        "id": f"{tc['id']}_{sub_idx}",
                                                        "function": {**tc["function"],
                                                                     "arguments": sub_args},
                                                    })
                                            else:
                                                expanded.append(tc)
                                        # Update the last assistant message's tool_calls in place
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

                # Continue to next round
                continue

            # Check for answer termination
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

        # Return complete history, final working context, per-round stats.
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


def worker_entry(
    worker_idx,
    num_workers,
    args,
    gpu_ids,
):
    # Set visible GPUs for this worker (empty list for API mode)
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gid) for gid in gpu_ids)
    else:
        # API mode - no GPUs needed
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    node_rank = int(os.getenv("RANK", 0))
    node_size = int(os.getenv("WORLD_SIZE", 1))

    async def _run():
        try:
            # Initialize generator based on mode
            if args.vllm_server_url:
                # Get the server URL for this worker
                if hasattr(args, 'vllm_server_urls') and len(args.vllm_server_urls) > 1:
                    server_url = args.vllm_server_urls[worker_idx % len(args.vllm_server_urls)]
                else:
                    server_url = args.vllm_server_url

                # Use OpenAI API with optional native tools support
                from utils.openai_generator import OpenAIAsyncGenerator
                import os as _os
                model_key = (args.model_name_or_path or "").lower()
                is_gptoss = "gpt-oss" in model_key
                is_deepseek = "deepseek" in model_key
                _api_key = (
                    getattr(args, "api_key", None)
                    or _os.getenv("OPENAI_API_KEY")
                    or _os.getenv("DEEPSEEK_API_KEY")
                    or "EMPTY"
                )
                generator = OpenAIAsyncGenerator(
                    base_url=server_url,
                    model_name=args.model_name_or_path,
                    api_key=_api_key,
                    use_native_tools=not is_gptoss,
                    served_model_name=getattr(args, "served_model_name", None),
                )

                if is_gptoss:
                    from openai_harmony import load_harmony_encoding, HarmonyEncodingName
                    gptoss_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
                    print(f"[Worker {worker_idx}] GPT-OSS mode: harmony encoding loaded")
                else:
                    gptoss_encoding = None

                mode = "gptoss harmony" if is_gptoss else ("deepseek dsml" if is_deepseek else "native function calling")
                print(f"[Worker {worker_idx}] Using OpenAI API ({mode}) at {server_url}")
            else:
                # Use local vLLM engine (slow startup)
                from utils.vllm_generator import vLLMAsyncGenerator
                generator = vLLMAsyncGenerator(
                    args.model_name_or_path,
                    tensor_parallel_size=args.tensor_parallel_size
                )
                is_gptoss = False
                is_deepseek = "deepseek" in (args.model_name_or_path or "").lower()
                gptoss_encoding = None
                print(f"[Worker {worker_idx}] Using local vLLM engine")


            browser_pool = BrowserPool(args.search_url, browser_backend=args.browser_backend)
            sem = asyncio.Semaphore(args.max_concurrency_per_worker)

            shard_path = os.path.join(args.output_dir, f"node_{node_rank}_shard_{worker_idx}.jsonl")
            os.makedirs(args.output_dir, exist_ok=True)

            # Load completed tasks from ALL shard files (not just completed_qids.txt)
            # This ensures we don't reprocess tasks even if they were completed by different workers
            processed_qids = set()
            print(f"[Worker {worker_idx}] Scanning all shard files for completed tasks...")

            for shard_file in glob.glob(os.path.join(args.output_dir, "node_*_shard_*.jsonl")):
                try:
                    with open(shard_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                record = json.loads(line)
                                processed_qids.add(record['qid'])
                            except Exception:
                                continue
                except Exception as e:
                    print(f"[Worker {worker_idx}] Warning: Could not read {shard_file}: {e}")

            print(f"[Worker {worker_idx}] Found {len(processed_qids)} completed tasks across all shards.")

            # Load dataset using unified loader
            # If data_path provided, pass it (for backward compatibility with browsecomp-plus)
            if args.data_path:
                # Legacy mode: explicit data_path for browsecomp-plus
                data = load_dataset(args.dataset_name, data_path=args.data_path)
            else:
                # New unified mode: load from HuggingFace
                data = load_dataset(args.dataset_name)

            qid_order_file = _default_qid_order_file_for_dataset(args.dataset_name)
            if qid_order_file:
                data = _apply_qid_order(data, qid_order_file)

            total_workers = node_size * num_workers
            global_worker_idx = num_workers * node_rank + worker_idx

            # Dynamic load balancing: redistribute unprocessed tasks among all workers
            # This ensures all workers stay busy even if previous runs were interrupted
            all_unprocessed_tasks = [x for x in data if x['qid'] not in processed_qids]
            tasks_to_process = all_unprocessed_tasks[global_worker_idx::total_workers]

            print(f"[Worker {worker_idx}] Total tasks: {len(data)}, "
                  f"Unprocessed: {len(all_unprocessed_tasks)}, "
                  f"Assigned to this worker: {len(tasks_to_process)}")
            
            if not tasks_to_process:
                print(f"[Worker {worker_idx}] Nothing to do.")
                return

            async def process_item(item_data: dict) -> dict:
                async with sem:
                    qid = item_data['qid']
                    question = item_data['question']
                    MAX_RETRY = 5
                    attempt = 0
                    error_msg = None
                    t0 = time.time()
                    full_msgs: list = []
                    final_msgs: list = []
                    t_stats: list = []
                    tok_summary: dict = {"final_messages_tokens": 0, "total_input_tokens": 0, "total_output_tokens": 0}
                    while attempt < MAX_RETRY:
                        attempt += 1
                        try:
                            if is_gptoss:
                                full_msgs, final_msgs, t_stats, tok_summary, visited_urls = await run_one_gptoss(
                                    question=question,
                                    qid=qid,
                                    generator=generator,
                                    browser_pool=browser_pool,
                                    encoding=gptoss_encoding,
                                    force_archive_after_turns=args.force_archive_after_turns,
                                    max_rounds=500,
                                    enable_parallel_tool_calls=not args.disable_parallel_tool_calls,
                                )
                            elif is_deepseek:
                                full_msgs, final_msgs, t_stats, tok_summary, visited_urls = await run_one_deepseek(
                                    question=question,
                                    qid=qid,
                                    generator=generator,
                                    browser_pool=browser_pool,
                                    force_archive_after_turns=args.force_archive_after_turns,
                                    max_rounds=500,
                                    enable_parallel_tool_calls=not args.disable_parallel_tool_calls,
                                )
                            else:
                                full_msgs, final_msgs, t_stats, tok_summary, visited_urls = await run_one(
                                    question=question,
                                    qid=qid,
                                    generator=generator,
                                    browser_pool=browser_pool,
                                    force_archive_after_turns=args.force_archive_after_turns,
                                    max_rounds=500,
                                    enable_parallel_tool_calls=not args.disable_parallel_tool_calls,
                                )
                            dt = time.time() - t0
                            rec = item_data.copy()
                            rec.update({
                                "full_messages": full_msgs,
                                "final_messages": final_msgs,
                                "turn_stats": t_stats,
                                **tok_summary,
                                "latency_s": dt,
                                "error": None,
                                "attempts": attempt,
                                "status": "success",
                                "retrieved_urls": visited_urls,
                            })
                            return rec
                        except Exception as e:
                            error_msg = traceback.format_exc()
                            print(f"[Worker {worker_idx}] qid {qid} attempt {attempt}/{MAX_RETRY} failed: {e}")
                    rec = item_data.copy()
                    rec.update({
                        "full_messages": full_msgs,
                        "final_messages": final_msgs,
                        "turn_stats": t_stats,
                        **tok_summary,
                        "latency_s": 0.0,
                        "error": error_msg,
                        "attempts": attempt,
                        "status": "fail",
                        "retrieved_urls": browser_pool.get_visited_urls(qid),
                    })
                    return rec

            tasks = [asyncio.create_task(process_item(task)) for task in tasks_to_process]

            with open(shard_path, "a", encoding="utf-8") as writer:
                for fut in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Worker {worker_idx}"):
                    rec = await fut
                    writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    writer.flush()
        finally:
            print(f"[Worker {worker_idx}] Done.")

    asyncio.run(_run())
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--search_url", required=True)
    parser.add_argument("--dataset_name", type=str, default="browsecomp-plus",
                        help=f"Dataset name (default: browsecomp-plus). Available: {', '.join(list_available_datasets())}")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to local data files (only required for browsecomp-plus dataset)")
    parser.add_argument("--browser_backend", type=str, default="local", choices=["local", "serper"],
                        help="Browser backend: 'local' (default) or 'serper'")
    parser.add_argument("--max_concurrency_per_worker", type=int, default=8)
    parser.add_argument("--reasoning_effort", default='high')
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for local vLLM (default: 1, ignored if using --vllm_server_url)")
    parser.add_argument("--vllm_server_url", type=str, default=None,
                        help="URL(s) of vLLM/OpenAI-compatible server. "
                             "Single: http://localhost:8001/v1  "
                             "Multiple (comma-sep): http://localhost:8001/v1,http://localhost:8002/v1")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for the inference server. Falls back to OPENAI_API_KEY, "
                             "then DEEPSEEK_API_KEY. Use 'EMPTY' for local vLLM.")
    parser.add_argument("--served_model_name", type=str, default=None,
                        help="Model ID as exposed by the API server. "
                             "Overrides auto-detection from /models. The --model_name_or_path is still used "
                             "for tokenizer loading.")
    parser.add_argument("--disable_parallel_tool_calls", action="store_true", default=False,
                        help="Disable model-specific parallel tool-call prompting and concurrent browser.search execution.")
    parser.add_argument("--force_archive_after_turns", type=int, default=4,
                        help="Browser results older than this many assistant turns are auto-archived. "
                             "Default: 4.")

    args = parser.parse_args()
    print(f"[CONFIG] force_archive_after_turns = {args.force_archive_after_turns}")
    print(f"[CONFIG] PARALLEL_TOOL_CALLS = {not args.disable_parallel_tool_calls}")
    print(args)

    # Auto-detect number of available CUDA devices
    import torch

    if args.vllm_server_url:
        # Parse server URLs (support comma-separated list)
        server_urls = [url.strip() for url in args.vllm_server_url.split(',')]
        args.vllm_server_urls = server_urls  # Store as list

        # Using external vLLM server - create one worker per server URL
        NUM_WORKERS = len(server_urls)
        available_gpu_ids = []

        print(f"Using {NUM_WORKERS} external vLLM server(s):")
        for i, url in enumerate(server_urls):
            print(f"  - Server {i+1}: {url}")
        print(f"Launching {NUM_WORKERS} worker(s) (CPU-based, no local model loading)")
    else:
        # Using local vLLM engine - need GPU allocation
        # Get the list of available GPU IDs from CUDA_VISIBLE_DEVICES
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices is not None:
            # User specified GPU IDs
            available_gpu_ids = [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()]
            print(f"Using user-specified GPUs: {available_gpu_ids}")
        else:
            # Auto-detect all available GPUs
            num_gpus = torch.cuda.device_count()
            available_gpu_ids = list(range(num_gpus))
            print(f"Auto-detected {num_gpus} CUDA device(s)")

        if len(available_gpu_ids) == 0:
            raise RuntimeError("No CUDA devices found. Cannot proceed without GPUs.")

        # Calculate number of workers based on tensor_parallel_size
        tp_size = args.tensor_parallel_size
        if len(available_gpu_ids) % tp_size != 0:
            raise ValueError(
                f"Number of GPUs ({len(available_gpu_ids)}) must be divisible by "
                f"tensor_parallel_size ({tp_size})"
            )

        NUM_WORKERS = len(available_gpu_ids) // tp_size
        print(f"Launching {NUM_WORKERS} worker(s) with tensor_parallel_size={tp_size}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Scan all existing shard files and collect completed qids
    print("Scanning for completed tasks across all shards...")
    completed_qids = set()
    node_rank = int(os.getenv("RANK", 0))

    # Check all possible shard files from all nodes
    for shard_file in glob.glob(os.path.join(args.output_dir, "node_*_shard_*.jsonl")):
        try:
            with open(shard_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        completed_qids.add(record['qid'])
                    except Exception:
                        continue
        except Exception as e:
            print(f"Warning: Could not read {shard_file}: {e}")

    if completed_qids:
        print(f"Found {len(completed_qids)} completed tasks from existing shards.")
        # Write to global completed file
        global_completed_path = os.path.join(args.output_dir, "completed_qids.txt")
        with open(global_completed_path, "w", encoding="utf-8") as f:
            for qid in sorted(completed_qids):
                f.write(f"{qid}\n")
        print(f"Wrote completed qids to {global_completed_path}")
    else:
        print("No completed tasks found. Starting fresh.")

    procs: List[mp.Process] = []
    for i in range(NUM_WORKERS):
        if args.vllm_server_url:
            # No GPU assignment needed for API mode
            worker_gpu_ids = []
            if hasattr(args, 'vllm_server_urls') and len(args.vllm_server_urls) > 1:
                server_url = args.vllm_server_urls[i % len(args.vllm_server_urls)]
                print(f"Worker {i} → Server: {server_url}")
            else:
                print(f"Worker {i} → Server: {args.vllm_server_url}")
        else:
            # Assign GPU IDs for this worker based on tensor parallelism
            tp_size = args.tensor_parallel_size
            worker_gpu_ids = available_gpu_ids[i * tp_size:(i + 1) * tp_size]
            print(f"Worker {i} assigned GPUs: {worker_gpu_ids}")

        p = mp.Process(
            target=worker_entry,
            args=(i, NUM_WORKERS, args, worker_gpu_ids)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=None)
        if p.exitcode != 0:
            print(f"Worker process {p.pid} exited with code {p.exitcode}")

    print("All workers finished. Script done.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
