import json
import time
from typing import Any, List

from tools.context_management import _auto_archive_old_results_gptoss
from utils.prompts import build_gptoss_messages


async def _generate_gptoss_round(
    generator: Any,
    tokens: List[int],
    stop_token_ids: List[int],
    encoding,
    max_retries: int = 20,
) -> tuple:
    """One generation round for GPT-OSS using harmony encoding."""
    import httpx
    from openai_harmony import Role

    if not getattr(generator, '_api_model_name', None):
        try:
            r = await generator.client.get(f"{generator.base_url}/models")
            data = r.json()
            if data.get("data"):
                generator._api_model_name = data["data"][0]["id"]
                print(f"[GPTOSS] Auto-detected model: {generator._api_model_name}")
        except Exception as e:
            print(f"[GPTOSS] Warning: could not fetch model name: {e}")
            generator._api_model_name = generator.model_name

    prompt_text = encoding.decode(tokens)

    async def _call_completions() -> dict:
        request_data = {
            "model": generator._api_model_name,
            "prompt": prompt_text,
            "max_tokens": 8192,
            "temperature": 1.0,
            "stop_token_ids": stop_token_ids,
            "skip_special_tokens": False,
        }
        response = await generator.client.post(
            f"{generator.base_url}/completions",
            json=request_data,
            headers={"Authorization": f"Bearer {generator.api_key}"},
        )
        response.raise_for_status()
        return response.json()

    def _parse_response(data: dict) -> tuple:
        choice = data["choices"][0]
        text = choice["text"]
        stop_reason = choice.get("stop_reason")
        n_output = data.get("usage", {}).get("completion_tokens", 0)

        gen_tokens = encoding.encode(text, allowed_special="all")
        if isinstance(stop_reason, int) and stop_reason in stop_token_ids:
            gen_tokens.append(stop_reason)

        new_messages = encoding.parse_messages_from_completion_tokens(
            gen_tokens, Role.ASSISTANT, strict=False
        )
        return new_messages, n_output

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            data = await _call_completions()
        except httpx.HTTPStatusError as e:
            last_exc = e
            if 400 <= e.response.status_code < 500:
                print(f"[GPTOSS] Client error {e.response.status_code} on attempt {attempt}: not retrying")
                break
            print(f"[GPTOSS] Server error {e.response.status_code} on attempt {attempt}/{max_retries}")
            continue
        except Exception as e:
            last_exc = e
            print(f"[GPTOSS] Network/transport error on attempt {attempt}/{max_retries}: {e}")
            import traceback as _tb
            print(_tb.format_exc())
            continue

        try:
            new_messages, n_output = _parse_response(data)
            print(f"[GPTOSS] Generated {n_output} tokens, stop_reason={data['choices'][0].get('stop_reason')}, "
                  f"{len(new_messages)} new message(s)")
            return new_messages, n_output
        except Exception as e:
            last_exc = e
            print(f"[GPTOSS] Response parsing failed on attempt {attempt}/{max_retries}: {e}")
            import traceback as _tb
            print(_tb.format_exc())
            continue

    raise last_exc or RuntimeError("GPTOSS generation failed after retries.")


def _extract_gptoss_reasoning(messages) -> str:
    """Return text of the most recent analysis message without a recipient."""
    from openai_harmony import Role

    for m in reversed(messages):
        if (m.author.role == Role.ASSISTANT
                and getattr(m, "channel", None) == "analysis"
                and not getattr(m, "recipient", None)):
            for c in m.content:
                if hasattr(c, "text"):
                    return c.text
    return ""


async def run_one_gptoss(
    question: str,
    qid: Any,
    generator: Any,
    browser_pool,
    encoding,
    force_archive_after_turns: int,
    max_rounds: int = 200,
    enable_parallel_tool_calls: bool = True,
) -> tuple:
    """Agent loop for GPT-OSS using openai_harmony message objects."""
    from openai_harmony import Author, Conversation, Message, Role, TextContent

    tool_config = browser_pool.init_session(qid)
    messages = build_gptoss_messages(question, tool_config)
    full_messages = list(messages)

    stop_token_ids = encoding.stop_tokens_for_assistant_actions()
    turn_stats: list = []
    last_input_tokens = 0
    prev_input_tokens = 0
    round_num = 0

    try:
        while round_num < max_rounds:
            last_message = messages[-1]
            recipient = getattr(last_message, "recipient", None)

            if recipient and str(recipient).startswith("browser."):
                tool_name = str(recipient).split(".", 1)[1]
                raw_args = next(
                    (c.text for c in last_message.content if hasattr(c, "text")), "{}"
                )
                try:
                    tool_args = json.loads(raw_args)
                except Exception:
                    tool_args = {}
                reasoning = _extract_gptoss_reasoning(messages)
                result_text = await browser_pool.call_tool(
                    qid, tool_name, tool_args, reasoning=reasoning
                )
                tool_msg = Message(
                    author=Author(role=Role.TOOL, name=str(recipient)),
                    content=[TextContent(text=result_text)],
                )
                messages.append(tool_msg)
                full_messages.append(tool_msg)
                continue

            if (last_message.author.role == Role.ASSISTANT
                    and getattr(last_message, "channel", None) == "final"):
                break

            round_num += 1
            print(f"\n{'='*60}\nRound {round_num}\n{'='*60}")

            msg_dicts = [m.to_dict() for m in messages]
            msg_dicts = _auto_archive_old_results_gptoss(
                msg_dicts,
                round_num,
                force_archive_after_turns,
            )
            messages = [Message.from_dict(d) for d in msg_dicts]

            conversation = Conversation.from_messages(messages)
            tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
            current_token_count = len(tokens)
            print(f"[GPTOSS] Input tokens: {current_token_count}")
            last_input_tokens = current_token_count

            t_start = time.time()
            new_messages, n_output = await _generate_gptoss_round(
                generator, tokens, stop_token_ids, encoding
            )
            cached_est = min(current_token_count, prev_input_tokens)
            turn_stats.append({
                "round": round_num,
                "input_tokens": current_token_count,
                "output_tokens": n_output,
                "latency_s": round(time.time() - t_start, 3),
                "cached_input_tokens_est": cached_est,
            })
            prev_input_tokens = current_token_count

            if not new_messages:
                print("[GPTOSS] Warning: empty generation - stopping.")
                break

            messages += new_messages
            full_messages += new_messages

            for m in new_messages:
                ch = getattr(m, "channel", None)
                rc = getattr(m, "recipient", None)
                preview = next(
                    (c.text[:300] for c in m.content if hasattr(c, "text")), ""
                )
                print(f"[GPTOSS] msg: role={m.author.role} channel={ch} "
                      f"recipient={rc} | {preview}")

        total_input = sum(s["input_tokens"] for s in turn_stats)
        total_output = sum(s["output_tokens"] for s in turn_stats)
        total_cached_est = sum(s.get("cached_input_tokens_est", 0) for s in turn_stats)
        token_summary = {
            "final_messages_tokens": last_input_tokens,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cached_input_tokens_est": total_cached_est,
        }

        full_msgs_dicts = [m.to_dict() for m in full_messages]
        final_msgs_dicts = [m.to_dict() for m in messages]
        visited_urls = browser_pool.get_visited_urls(qid)
        return full_msgs_dicts, final_msgs_dicts, turn_stats, token_summary, visited_urls

    finally:
        browser_pool.cleanup(qid)
