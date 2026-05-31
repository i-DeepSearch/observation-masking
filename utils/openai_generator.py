"""
OpenAI API-compatible generator for agent inference
Works with vLLM OpenAI-compatible server or any OpenAI-compatible API
"""
from typing import List, Optional, AsyncIterator
import httpx
import json

# Pre-import transformers to avoid issues in multiprocessing
try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print(f"Warning: transformers not available: {e}")
    _TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None


class OpenAIAsyncGenerator:
    """
    Async generator using OpenAI-compatible API
    Compatible with vLLM's OpenAI-compatible server
    """

    def __init__(
        self,
        base_url: str,
        model_name: str = None,
        api_key: str = "EMPTY",
        timeout: float = 300.0,
        use_native_tools: bool = False,
        served_model_name: str = None,
    ):
        """
        Args:
            base_url: Base URL of the OpenAI-compatible API (e.g., "http://localhost:8001/v1")
            model_name: HuggingFace model path used for tokenizer loading.
            api_key: API key ("EMPTY" for local vLLM, real key for cloud APIs).
            timeout: Request timeout in seconds.
            use_native_tools: If True, use chat/completions API with native tools support.
            served_model_name: Override the model ID sent to the API server. When None the
                server-reported model ID is used.
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name   # used for tokenizer loading (HF path)
        self._api_model_name = served_model_name  # pre-set if provided, else auto-detected
        self.api_key = api_key
        self.timeout = timeout
        self.use_native_tools = use_native_tools
        self.client = httpx.AsyncClient(timeout=timeout)
        self._closed = False

        # Fetch tokenizer info from server
        self.tokenizer = None

    async def _init_tokenizer(self):
        """Initialize tokenizer and resolve the server-side API model name."""
        if self.tokenizer is not None:
            return

        # If served_model_name was pre-set in __init__, skip auto-detection.
        if not self._api_model_name:
            try:
                response = await self.client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                models_data = response.json()
                if models_data.get("data"):
                    self._api_model_name = models_data["data"][0]["id"]
                    print(f"[API] Server model ID: {self._api_model_name}")
            except Exception as e:
                print(f"Warning: Could not fetch model list from server: {e}")

        # Fall back to model_name if still unresolved
        if not self._api_model_name:
            self._api_model_name = self.model_name

        # If no model_name was provided, use the server's ID for tokenizer too
        if not self.model_name:
            self.model_name = self._api_model_name

        if not self.model_name:
            raise ValueError("No model name provided and could not fetch from server")

        # Use pre-imported AutoTokenizer if available
        if not _TRANSFORMERS_AVAILABLE or AutoTokenizer is None:
            raise ImportError("transformers library not available")

        print(f"Loading tokenizer for: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        print(f"Tokenizer loaded successfully")

    @staticmethod
    def _parse_chat_messages(prompt_text: str) -> list:
        """Parse a Qwen chat-template string back into [{role, content}] messages.

        Handles <|im_start|>role\\ncontent<|im_end|> blocks produced by
        AutoTokenizer.apply_chat_template.  The trailing incomplete assistant
        block (if any) is dropped because /chat/completions will generate it.
        """
        messages = []
        parts = prompt_text.split("<|im_start|>")
        for part in parts[1:]:
            if not part.strip():
                continue
            nl = part.find("\n")
            if nl == -1:
                continue
            role = part[:nl].strip()
            content = part[nl + 1:]
            content = content.split("<|im_end|>")[0].strip()
            # skip incomplete trailing assistant block (no content yet)
            if role == "assistant" and not content:
                continue
            messages.append({"role": role, "content": content})
        return messages

    async def _generate_via_chat(
        self,
        prompt_text: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[int]:
        """Fallback: use /chat/completions when /completions is not supported.

        Handles both regular text responses AND tool-call responses (delta.tool_calls).
        For tool calls, the reconstructed JSON is emitted as tokens so the caller's
        tool-call parser can handle them transparently.
        """
        messages = self._parse_chat_messages(prompt_text)
        if not messages:
            messages = [{"role": "user", "content": prompt_text}]

        request_data: dict = {
            "model": self._api_model_name,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens or 8192,
        }
        print(f"[OpenAI API / chat] model={self._api_model_name}, "
              f"messages={len(messages)}, max_tokens={request_data['max_tokens']}")

        # Accumulate streaming tool_call fragments keyed by index
        tool_call_accum: dict[int, dict] = {}

        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=request_data,
            headers={"Authorization": f"Bearer {self.api_key}"},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choice = (data.get("choices") or [{}])[0]
                        delta = choice.get("delta", {})
                        text = delta.get("content") or ""
                        if text:
                            for tok in self.tokenizer.encode(text, add_special_tokens=False):
                                yield tok

                        # Accumulate streaming tool_call fragments
                        for tc_delta in delta.get("tool_calls") or []:
                            idx = tc_delta.get("index", 0)
                            if idx not in tool_call_accum:
                                tool_call_accum[idx] = {
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                }
                            fn = tc_delta.get("function") or {}
                            tool_call_accum[idx]["function"]["name"] += fn.get("name") or ""
                            tool_call_accum[idx]["function"]["arguments"] += fn.get("arguments") or ""

                        finish = choice.get("finish_reason")
                        if finish:
                            # Emit accumulated tool calls as DeepSeek-format text tokens
                            if tool_call_accum and finish == "tool_calls":
                                tc_text = "<｜tool▁calls▁begin｜>"
                                for tc in tool_call_accum.values():
                                    name = tc["function"]["name"]
                                    args = tc["function"]["arguments"]
                                    tc_text += (
                                        f"<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n"
                                        f"```json\n{args}\n```<｜tool▁call▁end｜>"
                                    )
                                tc_text += "<｜tool▁calls▁end｜>"
                                for tok in self.tokenizer.encode(tc_text, add_special_tokens=False):
                                    yield tok
                            break
                    except json.JSONDecodeError:
                        continue

    async def generate(
        self,
        prompt_tokens: List[int],
        stop_tokens: Optional[List[int]] = None,
        stop_strings: Optional[List[str]] = None,
        temperature: float = 1.0,
        max_tokens: int = 0,
        return_logprobs: bool = False
    ) -> AsyncIterator[int]:
        """
        Generate tokens using OpenAI API streaming.
        Tries /completions first; falls back to /chat/completions automatically
        for servers that do not expose /completions.
        """
        await self._init_tokenizer()

        prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        max_tok = max_tokens if max_tokens and max_tokens > 0 else 8192

        # If we already know /completions is unsupported, go straight to chat path
        if getattr(self, "_force_chat_api", False):
            async for tok in self._generate_via_chat(prompt_text, max_tok, temperature):
                yield tok
            return

        request_data = {
            "model": self._api_model_name,
            "prompt": prompt_text,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tok,
        }
        if return_logprobs:
            request_data["logprobs"] = 1
        print(f"[OpenAI API] Request: model={self._api_model_name}, max_tokens={max_tok}")

        # Make streaming request; auto-retry via chat API if /completions unsupported
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/completions",
                json=request_data,
                headers={"Authorization": f"Bearer {self.api_key}"}
            ) as response:
                # 404 or 400 "model not supported" → switch to chat API permanently
                if response.status_code in (400, 404):
                    body = await response.aread()
                    try:
                        err = json.loads(body).get("error", {})
                    except Exception:
                        err = {}
                    err_msg = err.get("message", "").lower()
                    fallback = (
                        response.status_code == 404
                        or "not_supported" in str(err.get("code", ""))
                        or "unsupported model" in err_msg
                        or "not supported" in err_msg
                        or "completions api is only available" in err_msg  # DeepSeek
                        or "only available when using beta" in err_msg     # DeepSeek
                    )
                    if fallback:
                        print(f"[OpenAI API] /completions not available (HTTP {response.status_code}), "
                              f"switching to /chat/completions permanently")
                        self._force_chat_api = True
                        async for tok in self._generate_via_chat(prompt_text, max_tok, temperature):
                            yield tok
                        return
                    response.raise_for_status()
                else:
                    response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])

                            if choices:
                                choice = choices[0]
                                text = choice.get("text", "")
                                finish_reason = choice.get("finish_reason")

                                if text:
                                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                                    for token in tokens:
                                        yield token

                                if finish_reason is not None and finish_reason != "":
                                    print(f"[OpenAI API] Stream finished with reason: {finish_reason}")
                                    break

                        except json.JSONDecodeError:
                            continue

        except Exception as exc:
            # Surface unexpected errors (not the "unsupported model" path above)
            raise

    async def chat_completion(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        tool_choice: str = "auto",
        temperature: float = 1.0,
        max_tokens: int = 4096,
        use_reasoning_content: bool = True,
        thinking_mode: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        **extra_params,
    ) -> dict:
        """
        Create a chat completion with optional tool calling using OpenAI Chat API

        Args:
            messages: List of message dicts with 'role' and 'content'/'reasoning_content'
            tools: List of tool definitions in OpenAI format
            tool_choice: "auto", "none", or specific tool
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_reasoning_content: If True, use 'reasoning_content' field for assistant messages

        Returns:
            Response dict from API
        """
        await self._init_tokenizer()

        # Convert messages to API format
        # Only keep valid fields and ensure content is never None
        api_messages = []
        is_deepseek = "deepseek" in (self.model_name or "").lower()
        for msg in messages:
            api_msg = {
                "role": msg.get("role", "user"),
                "content": msg.get("content") or "",
            }
            if msg.get("reasoning_content"):
                api_msg["reasoning_content"] = msg["reasoning_content"]
            if msg.get("tool_calls"):
                api_tool_calls = []
                for tc in msg["tool_calls"]:
                    tc_copy = dict(tc)
                    fn = dict(tc_copy.get("function", {}))
                    args = fn.get("arguments", {})
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    fn["arguments"] = args
                    if is_deepseek and fn.get("name"):
                        fn["name"] = fn["name"].replace(".", "_")
                    tc_copy["function"] = fn
                    api_tool_calls.append(tc_copy)
                api_msg["tool_calls"] = api_tool_calls
            if msg.get("tool_call_id"):
                api_msg["tool_call_id"] = msg["tool_call_id"]
            api_messages.append(api_msg)

        request_data = {
            "model": self._api_model_name,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add tools if provided
        if tools:
            request_data["tools"] = tools
            request_data["tool_choice"] = tool_choice

        # DeepSeek-V4: top-level thinking_mode and reasoning_effort params
        if thinking_mode is not None:
            request_data["thinking_mode"] = thinking_mode
        if reasoning_effort is not None:
            request_data["reasoning_effort"] = reasoning_effort

        # Any additional caller-supplied params
        request_data.update(extra_params)

        print(f"[OpenAI Chat API] Request: model={self._api_model_name}, "
              f"messages={len(api_messages)}, tools={len(tools) if tools else 0}"
              f"{f', thinking_mode={thinking_mode}' if thinking_mode else ''}")

        # Make request
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=request_data,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            # Print detailed error for 400 Bad Request
            print(f"[OpenAI Chat API] Error: {e.response.status_code} {e.response.reason_phrase}")
            print(f"[OpenAI Chat API] Response body: {e.response.text}")
            raise

    def shutdown(self) -> None:
        """Close the HTTP client"""
        if self._closed:
            return
        try:
            import asyncio
            asyncio.create_task(self.client.aclose())
        except Exception:
            pass
        finally:
            self._closed = True

    def __del__(self):
        try:
            if not getattr(self, "_closed", True):
                self.shutdown()
        except Exception:
            pass
