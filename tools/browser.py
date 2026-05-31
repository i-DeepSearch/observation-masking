import logging
from aiohttp import ClientSession
from typing import AsyncIterator, Any, Dict, List, Union
import time,json,html
import asyncio
import requests
import contextvars
import inspect
import re
from gpt_oss.tools.simple_browser.page_contents import (
    process_html,
)
from gpt_oss.tools.simple_browser.backend import (
    VIEW_SOURCE_PREFIX,
    BackendError,
    maybe_truncate,
)

from openai_harmony import (
    Author,
    Message,
    Role,
    TextContent,
)

from gpt_oss.tools.simple_browser.simple_browser_tool import SimpleBrowserTool,maybe_get_function_args,function_the_model_can_call,handle_errors,_live_function_name
from gpt_oss.tools.simple_browser.simple_browser_tool import get_end_loc, join_lines, wrap_lines
from gpt_oss.tools.simple_browser.simple_browser_tool import ToolUsageError, run_find_in_page
from gpt_oss.tools.simple_browser.backend import BackendError, maybe_truncate

import os
import textwrap


logger = logging.getLogger(__name__)
_ACTIVE_REASONING = contextvars.ContextVar("browser_active_reasoning", default=None)
SEARCH_RESULT_VIEW_LINES = 80
URL_REF_RE = re.compile(r"【\d+†(?:https?://|web-search://|web://)[^】]+】")


def _wrap_lines_preserving_url_refs(text: str, width: int = 80) -> list[str]:
    wrapped: list[str] = []
    for line in text.split("\n"):
        if not line:
            wrapped.append("")
            continue
        url_ref = URL_REF_RE.search(line)
        if url_ref:
            wrapped.append(line[:url_ref.end()])
            rest = line[url_ref.end():]
            if rest:
                wrapped.extend(textwrap.wrap(
                    rest,
                    width=width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                ) or [""])
        else:
            wrapped.extend(wrap_lines(line, width=width))
    return wrapped


def _render_search_result_items(title_url_summary_all: List[tuple[str, str, str]]) -> str:
    return "".join(
        f"<li><a href='{url}'>{url}</a> Title: {title}. {summary}</li>"
        for title, url, summary in title_url_summary_all
    )


def _normalize_search_result_link_markers(page):
    """Render search result refs as 【id†real_url】 instead of adding title/domain."""
    text = re.sub(
        r"【(\d+)†((?:https?://|web-search://|web://)[^†】]+)†[^】]+】",
        r"【\1†\2】",
        page.text,
    )
    if text == page.text:
        return page
    if hasattr(page, "model_copy"):
        return page.model_copy(update={"text": text})
    page.text = text
    return page


def sanitize_dict_keys(d):
    """Remove None keys from dictionary."""
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if k is not None}


def _coerce_topn(tool_args: Dict[str, Any], default: int = 15) -> int:
    raw_topn = tool_args.get("topn", tool_args.get("top_n", default))
    try:
        topn = int(raw_topn)
    except Exception:
        topn = default
    return max(1, topn)


class BrowserTool(SimpleBrowserTool):
    def _latest_search_result_cursor(self) -> int:
        """Scan the page stack backward and return the cursor of the most recent
        search-result page (identified by url starting with 'web-search://').
        Returns -1 (current top) if no search-result page is found."""
        stack = self.tool_state.page_stack
        for i in range(len(stack) - 1, -1, -1):
            if stack[i].startswith('web-search://'):
                return i
        return -1

    async def show_page(self, loc: int = 0, num_lines: int = -1) -> Message:
        page = self.tool_state.get_page()
        cursor = self.tool_state.current_cursor
        lines = _wrap_lines_preserving_url_refs(text=page.text)
        total_lines = len(lines)

        if loc >= total_lines:
            raise ToolUsageError(
                f"Invalid location parameter: `{loc}`. "
                f"Cannot exceed page maximum of {total_lines - 1}."
            )

        end_loc = get_end_loc(
            loc, num_lines, total_lines, lines, self.view_tokens, self.encoding_name
        )
        body = join_lines(lines[loc:end_loc], add_line_numbers=True, offset=loc)
        scrollbar = f"viewing lines [{loc} - {end_loc - 1}] of {total_lines - 1}"
        return self._make_response(page, cursor, body, scrollbar)

    @property
    def tool_config(self):
        config = super().tool_config
        for tool in config.tools:
            if tool.name != "search":
                continue
            properties = tool.parameters.get("properties", {})
            if "topn" in properties:
                properties["topn"]["default"] = 15
            if "top_n" in properties:
                properties["top_n"]["default"] = 15
        return config

    @function_the_model_can_call
    @handle_errors
    async def search(
        self,
        query: Union[str, List[str]],
        topn: int = 15,
        top_n: int = 15,
        source: str | None = None,
    ) -> AsyncIterator[Message]:
        """Search while honoring the model-provided topn/top_n value."""
        del source
        if topn == 15 and top_n != 15:
            topn = top_n
        try:
            topn = int(topn)
        except Exception:
            topn = 15
        topn = max(1, topn)

        try:
            async with ClientSession() as session:
                search_params = inspect.signature(self.backend.search).parameters
                kwargs = {
                    "query": query,
                    "topn": topn,
                    "session": session,
                }
                reasoning = _ACTIVE_REASONING.get() or getattr(self.backend, '_active_reasoning', None)
                if "reasoning" in search_params and reasoning:
                    kwargs["reasoning"] = reasoning
                search_page = await self.backend.search(**kwargs)
        except Exception as e:
            msg = maybe_truncate(str(e))
            raise BackendError(f"Error during search for `{query}`: {msg}") from e

        self.tool_state.add_page(search_page)
        yield await self.show_page_safely(loc=0, num_lines=SEARCH_RESULT_VIEW_LINES)

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        def make_error_message(error: str) -> Message:
            return self.make_response(
                content=TextContent(text=json.dumps({"error": error})),
                author=Author(role=Role.TOOL, name=message.recipient),
            )

        function_args = maybe_get_function_args(message, tool_name=self.name)
        if function_args is None:
            yield make_error_message("Invalid function arguments")
            return

        _, function_name = message.recipient.split(".")
        if function_name not in ["search", "open", "find"]:
            yield make_error_message(f"Unknown function: {function_name}")
            return
        try:
            if function_name == "search":
                function_args.pop("include_reasoning", None)
                async for msg in self.search(**function_args):
                    yield msg
            elif function_name == "open":
                id_val = function_args.get("id", -1)
                cursor_val = function_args.get("cursor", -1)
                # When opening by link index without an explicit cursor, resolve
                # against the most recent search-result page rather than the
                # current top-of-stack (which may be an opened URL page with no
                # link-id index).
                if not isinstance(id_val, str) and id_val >= 0 and cursor_val == -1:
                    effective = self._latest_search_result_cursor()
                    if effective != -1:
                        function_args = dict(function_args, cursor=effective)
                async for msg in self.open(**function_args):
                    yield msg
            elif function_name == "find":
                async for msg in self.find(**function_args):
                    yield msg
        except TypeError as e:
            error_text = f"Error: Invalid arguments for function '{function_name}'. Please check the function signature. Details: {e}"
            error_content = TextContent(text=error_text)
            yield self.make_response(
                content=error_content,
                author=Author(role=Role.TOOL, name=message.recipient)
            )
        except Exception as e:
            error_text = f"An unexpected error occurred while executing function '{function_name}': {e}"
            error_content = TextContent(text=error_text)
            yield self.make_response(
                content=error_content,
                author=Author(role=Role.TOOL, name=message.recipient)
            )
    
    async def _open_url(self, url: str, direct_url_open: bool):
        """Use the cache, if available."""
        backend = self.backend
        # direct_url_open should be regarded as a refresh
        if not direct_url_open and (page := self.tool_state.get_page_by_url(url)):
            assert page.url == url
            return page

        try:
            async with ClientSession() as session:
                page = await backend.fetch(url, session=session)
            return page
        except Exception as e:
            msg = maybe_truncate(str(e))
            raise BackendError(
                f"Error fetching URL `{maybe_truncate(url)}`: {msg}"
            ) from e

class LocalServiceBrowserBackend:
    source = "web"
    
    def __init__(self,base_url):
        self.base_url = base_url
        
    async def _post(self, session: ClientSession, endpoint: str, payload: dict) -> dict:
        t0 = time.time()
        async with session.post(f"{self.base_url}{endpoint}", json=payload) as resp:
            if resp.status != 200:
                raise BackendError(
                    f"Search error {resp.status}: {await resp.text()}"
                )
            return await resp.json()
                
    async def _search_single(
        self,
        query: str,
        topn: int,
        session: ClientSession,
        reasoning: str = None,
    ) -> tuple[str, list]:
        """Execute a single search query and return query with results."""
        effective_reasoning = reasoning or _ACTIVE_REASONING.get() or getattr(self, '_active_reasoning', None)
        payload = {"query": query, "topn": topn}
        if effective_reasoning:
            payload["reasoning"] = effective_reasoning
        data = await self._post(session, "/search", payload)
        results = data.get("results", [])
        if not results:
            logger.warning(f"No results returned for query: '{query}'")
            return query, []

        title_url_summary = []
        for result in results:
            title_url_summary.append((
                html.escape(result['title'], quote=True),
                html.escape(result['url'], quote=True),
                html.escape(result['summary'], quote=True)
            ))
        return query, title_url_summary

    async def search(
        self,
        query: Union[str, List[str]],
        topn: int = 15,
        session = None,
        reasoning: str = None,
    ):
        """Search for one or more queries. If query is a list, searches are executed in parallel."""
        # Handle single query
        if isinstance(query, str):
            query_list = [query]
            title_str = query
        else:
            query_list = query
            # Create a title with all query names
            title_str = " | ".join(query_list)

        # Execute searches in parallel
        tasks = [self._search_single(q, topn, session, reasoning) for q in query_list]
        all_results = await asyncio.gather(*tasks)

        # Merge all results
        title_url_summary_all = []
        for query_str, title_url_summary in all_results:
            # Add results from each query
            if title_url_summary:
                title_url_summary_all.extend(title_url_summary)
        title_url_summary_all = title_url_summary_all[:topn]

        # If no results from any query, raise error
        if not title_url_summary_all:
            raise BackendError(f"No results returned for any query: {query_list}")

        html_page = f"""
<html><body>
<h1>Search Results</h1>
<ul>
{_render_search_result_items(title_url_summary_all)}
</ul>
</body></html>
"""

        pseudo_url = f"web-search://ts={time.time_ns()}"
        return _normalize_search_result_link_markers(process_html(
            html=html_page,
            url=pseudo_url,
            title=title_str,
            display_urls=True,
            session=session,
        ))
        
    async def fetch(self, url:str, session=None):
        is_view_source = url.startswith(VIEW_SOURCE_PREFIX)
        if is_view_source:
            url = url[len(VIEW_SOURCE_PREFIX) :]
        
        data = await self._post(
            session,
            "/get_content",
            {"url": url},
        )
        
        if not data or not data.get("content"):
            raise BackendError(f"No content returned for {url}")
        return process_html(
            html=data.get("content", ""),
            url=url,
            title=data.get("title", ""),
            display_urls=True,
            session=session,
        )

class SerperServiceBrowserBackend:
    """Browser backend using Serper API for search and scraping."""
    source = "web"

    def __init__(self):
        self.api_key = os.getenv("SERPER_API_KEY")
        self.search_url = "https://google.serper.dev/search"
        self.scrape_url = "https://scrape.serper.dev/"

    # Maps Unicode block ranges → (gl country code, hl language code)
    # Order matters: more specific scripts first (Kana before CJK, Hangul before CJK)
    _SCRIPT_LOCALE: list = [
        (0x3040, 0x30FF, "jp", "ja"),        # Hiragana / Katakana → Japanese (before CJK)
        (0xAC00, 0xD7A3, "kr", "ko"),        # Hangul → Korean (before CJK)
        (0x4E00, 0x9FFF, "cn", "zh-cn"),    # CJK Unified → Chinese
        (0x0600, 0x06FF, "sa", "ar"),        # Arabic
        (0x0400, 0x04FF, "ru", "ru"),        # Cyrillic → Russian
        (0x0900, 0x097F, "in", "hi"),        # Devanagari → Hindi
        (0x0E00, 0x0E7F, "th", "th"),        # Thai
    ]

    @classmethod
    def _detect_locale(cls, query: str):
        """Return (gl, hl) based on the dominant script in query, or (None, None).

        Special rule: any Hiragana/Katakana char → Japanese, because Japanese text
        mixes Kana with CJK ideographs and CJK count alone would misclassify it.
        """
        # Fast-path: Hiragana (3040-309F) or Katakana (30A0-30FF) → Japanese
        for ch in query:
            if 0x3040 <= ord(ch) <= 0x30FF:
                return "jp", "ja"

        counts: dict[tuple, int] = {}
        for ch in query:
            cp = ord(ch)
            for lo, hi, gl, hl in cls._SCRIPT_LOCALE:
                if lo <= cp <= hi:
                    key = (gl, hl)
                    counts[key] = counts.get(key, 0) + 1
                    break
        if not counts:
            return None, None
        (gl, hl), _ = max(counts.items(), key=lambda x: x[1])
        return gl, hl

    async def _search_single(
        self,
        query: str,
        topn: int,
        session: ClientSession,
    ) -> tuple[str, list]:
        """Execute a single search query and return query with results."""
        payload = {
            "q": query,
            "num": topn
        }
        # Auto-detect region/language from query script
        gl, hl = self._detect_locale(query)
        if gl:
            payload["gl"] = gl
        if hl:
            payload["hl"] = hl
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

        async with session.post(self.search_url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise BackendError(
                    f"Search error {resp.status}: {await resp.text()}"
                )
            data = await resp.json()

        results = data.get("organic", [])
        if not results:
            logger.warning(f"No results returned for query: '{query}'")
            return query, []

        title_url_summary = []
        for result in results:
            title_url_summary.append((
                html.escape(result.get('title', ''), quote=True),
                html.escape(result.get('link', ''), quote=True),
                html.escape(result.get('snippet', ''), quote=True)
            ))
        return query, title_url_summary

    async def search(
        self,
        query: Union[str, List[str]],
        topn: int = 15,
        session = None,
    ):
        """Search using Serper API. Supports single query or list of queries (parallel)."""
        # Handle single query
        if isinstance(query, str):
            query_list = [query]
            title_str = query
        else:
            query_list = query
            # Create a title with all query names
            title_str = " | ".join(query_list)

        # Execute searches in parallel
        tasks = [self._search_single(q, topn, session) for q in query_list]
        all_results = await asyncio.gather(*tasks)

        # Merge all results
        title_url_summary_all = []
        for query_str, title_url_summary in all_results:
            # Add results from each query
            if title_url_summary:
                title_url_summary_all.extend(title_url_summary)
        title_url_summary_all = title_url_summary_all[:topn]

        # If no results from any query, raise error
        if not title_url_summary_all:
            raise BackendError(f"No results returned for any query: {query_list}")

        html_page = f"""
<html><body>
<h1>Search Results</h1>
<ul>
{_render_search_result_items(title_url_summary_all)}
</ul>
</body></html>
"""

        pseudo_url = f"web-search://ts={time.time_ns()}"
        return _normalize_search_result_link_markers(process_html(
            html=html_page,
            url=pseudo_url,
            title=title_str,
            display_urls=True,
            session=session,
        ))

    async def fetch(self, url: str, session=None):
        """Fetch and scrape a URL using Serper API."""
        is_view_source = url.startswith(VIEW_SOURCE_PREFIX)
        if is_view_source:
            url = url[len(VIEW_SOURCE_PREFIX):]

        payload = {
            "url": url
        }
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

        async with session.post(self.scrape_url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise BackendError(
                    f"Fetch error {resp.status}: {await resp.text()}"
                )
            data = await resp.json()

        # Sanitize the response data to remove any None keys
        data = sanitize_dict_keys(data)

        if not data:
            raise BackendError(f"No content returned for {url}")

        # Serper scrape API returns 'text' (extracted content) and 'metadata' (with title)
        text_content = data.get("text", "")
        metadata = data.get("metadata", {})
        title = metadata.get("title", "") if isinstance(metadata, dict) else ""

        if not text_content:
            raise BackendError(f"No content returned for {url}")

        return process_html(
            html=text_content,
            url=url,
            title=title,
            display_urls=True,
            session=session,
        )




def _extract_text_from_harmony(messages: List) -> str:
    text_parts = []
    for msg in messages:
        if hasattr(msg, 'content') and isinstance(msg.content, list):
            for item in msg.content:
                if hasattr(item, 'text'):
                    text_parts.append(item.text)
                elif isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
    return '\n'.join(text_parts) if text_parts else ""


class BrowserPool:
    def __init__(self, search_url: str, browser_backend: str = 'local'):
        self.search_url = search_url
        self.browser_backend = browser_backend
        self.sessions: Dict[Any, BrowserTool] = {}

    def init_session(self, qid: Any) -> dict:
        if self.browser_backend == 'serper':
            backend = SerperServiceBrowserBackend()
        else:
            backend = LocalServiceBrowserBackend(base_url=self.search_url)
        tool = BrowserTool(backend=backend)
        self.sessions[qid] = tool
        return tool.tool_config

    async def call_tool(self, qid: Any, tool_name: str, tool_args: Dict[str, Any],
                        reasoning: str = None) -> str:
        tool = self.sessions[qid]
        tool_args = dict(tool_args)
        tool_args.pop("include_reasoning", None)
        reasoning_token = _ACTIVE_REASONING.set(
            reasoning if tool_name.lower() == 'search' else None
        )

        recipient_map = {
            'search': 'browser.search',
            'find': 'browser.find',
            'open': 'browser.open',
        }
        try:
            recipient = recipient_map.get(tool_name.lower())
            if not recipient:
                return f"Unknown browser tool: {tool_name}"

            args_json = json.dumps(tool_args, ensure_ascii=False)
            tool_msg = Message.from_role_and_content(Role.ASSISTANT, TextContent(text=args_json))
            tool_msg.recipient = recipient

            results = []
            async for msg in tool.process(tool_msg):
                results.append(msg)

            return _extract_text_from_harmony(results)
        finally:
            _ACTIVE_REASONING.reset(reasoning_token)

    async def call_search_tools_concurrently(
        self,
        qid: Any,
        tool_args_list: List[Dict[str, Any]],
        reasoning: str = None,
    ) -> List[str]:
        """Run multiple browser.search backend requests concurrently.

        The network-bound backend.search calls run in parallel, while page-state
        mutation and cursor rendering happen afterward in the original call
        order. This avoids races in SimpleBrowserTool.show_page(), which reads
        the shared current page from tool_state.
        """
        import inspect

        tool = self.sessions[qid]
        search_params = inspect.signature(tool.backend.search).parameters
        supports_reasoning = "reasoning" in search_params

        async def _search_page(tool_args: Dict[str, Any]):
            query = tool_args.get("query", "")
            topn = _coerce_topn(tool_args)
            async with ClientSession() as session:
                kwargs = {
                    "query": query,
                    "topn": topn,
                    "session": session,
                }
                if supports_reasoning and reasoning:
                    kwargs["reasoning"] = reasoning
                return await tool.backend.search(**kwargs)

        pages = await asyncio.gather(
            *(_search_page(args) for args in tool_args_list),
            return_exceptions=True,
        )

        results: List[str] = []
        for page in pages:
            if isinstance(page, Exception):
                results.append(f"Error executing browser.search: {page}")
                continue
            try:
                tool.tool_state.add_page(page)
                function_token = _live_function_name.set("search")
                try:
                    result_msg = await tool.show_page_safely(
                        loc=0,
                        num_lines=SEARCH_RESULT_VIEW_LINES,
                    )
                    results.append(_extract_text_from_harmony([result_msg]))
                finally:
                    _live_function_name.reset(function_token)
            except Exception as e:
                results.append(f"Error rendering browser.search result: {e}")
        return results

    async def call_browser_tools_concurrently(
        self,
        qid: Any,
        tool_specs: List[Dict[str, Any]],
        reasoning: str = None,
    ) -> List[str]:
        """Run a batch of browser.search/open/find calls concurrently.

        Browser state is intentionally snapshotted before the batch. Any
        browser.open/browser.find cursor or link id is resolved against that
        pre-batch state, so an open/find in the same assistant turn cannot depend
        on search results produced earlier in that same turn. Network-bound
        fetch/search work runs concurrently; page-stack mutation and rendering
        happen afterward in the original call order.
        """
        import inspect

        tool = self.sessions[qid]
        snapshot_stack = list(tool.tool_state.page_stack)
        snapshot_pages = dict(tool.tool_state.pages)
        snapshot_current_cursor = len(snapshot_stack) - 1

        def _snapshot_page(cursor: int = -1):
            if snapshot_current_cursor < 0:
                raise ToolUsageError("No pages to access!")
            if cursor == -1 or cursor == snapshot_current_cursor:
                return snapshot_pages[snapshot_stack[-1]]
            if not isinstance(cursor, int):
                raise ToolUsageError(
                    f"`cursor` should be an integer, not `{type(cursor).__name__}`"
                )
            try:
                return snapshot_pages[snapshot_stack[cursor]]
            except IndexError as e:
                raise ToolUsageError(
                    f"Cursor `{cursor}` is out of range. "
                    f"Available cursor indices: [0 - {snapshot_current_cursor}]."
                ) from e

        def _latest_search_cursor_in_snapshot() -> int:
            """Return the cursor index of the most recent search-result page in
            the pre-batch snapshot (url starts with 'web-search://'), or -1 if none."""
            for i in range(len(snapshot_stack) - 1, -1, -1):
                if snapshot_stack[i].startswith('web-search://'):
                    return i
            return -1

        async def _prepare_search(tool_args: Dict[str, Any]):
            search_params = inspect.signature(tool.backend.search).parameters
            supports_reasoning = "reasoning" in search_params
            topn = _coerce_topn(tool_args)
            async with ClientSession() as session:
                kwargs = {
                    "query": tool_args.get("query", ""),
                    "topn": topn,
                    "session": session,
                }
                if supports_reasoning and reasoning:
                    kwargs["reasoning"] = reasoning
                page = await tool.backend.search(**kwargs)
            return {
                "tool_name": "search",
                "page": page,
                "loc": 0,
                "num_lines": SEARCH_RESULT_VIEW_LINES,
            }

        async def _prepare_open(tool_args: Dict[str, Any]):
            link_id = tool_args.get("id", -1)
            cursor = tool_args.get("cursor", -1)
            loc = tool_args.get("loc", -1)
            num_lines = tool_args.get("num_lines", -1)
            view_source = tool_args.get("view_source", False)

            curr_page = None
            snippet = None
            direct_url_open = False
            stay_on_current_page = False

            if isinstance(link_id, str):
                url = link_id
                direct_url_open = True
            else:
                # When opening by link index without an explicit cursor, resolve
                # against the most recent search-result page in the snapshot.
                if link_id >= 0 and cursor == -1:
                    effective_cursor = _latest_search_cursor_in_snapshot()
                    curr_page = _snapshot_page(effective_cursor)
                else:
                    curr_page = _snapshot_page(cursor)
                if link_id >= 0:
                    try:
                        url = curr_page.urls[str(link_id)]
                    except KeyError as e:
                        used_cursor = effective_cursor if (link_id >= 0 and cursor == -1) else cursor
                        max_id = max((int(k) for k in curr_page.urls if k.isdigit()), default=-1)
                        if max_id < 0:
                            raise ToolUsageError(
                                f"Invalid link id `{link_id}`: cursor [{used_cursor}] "
                                f"({curr_page.url}) has no link ids."
                            ) from e
                        raise ToolUsageError(
                            f"Invalid link id `{link_id}`: cursor [{used_cursor}] "
                            f"({curr_page.url}) only has link ids [0 - {max_id}]."
                        ) from e
                    snippet = (curr_page.snippets or {}).get(str(link_id))
                else:
                    if not view_source:
                        stay_on_current_page = True
                    url = curr_page.url

            if view_source:
                url = f"{VIEW_SOURCE_PREFIX}{url}"
                snippet = None

            if stay_on_current_page:
                page = curr_page
            else:
                page = await tool._open_url(url, direct_url_open)

            if loc < 0:
                if snippet is not None and snippet.line_idx is not None:
                    loc = max(snippet.line_idx - 4, 0)
                else:
                    loc = 0

            return {
                "tool_name": "open",
                "page": page,
                "loc": loc,
                "num_lines": num_lines,
            }

        async def _prepare_find(tool_args: Dict[str, Any]):
            pattern = str(tool_args.get("pattern", "")).lower()
            cursor = tool_args.get("cursor", -1)
            page = _snapshot_page(cursor)
            if page.snippets is not None:
                raise ToolUsageError(
                    "Cannot run `find` on search results page or find results page"
                )
            result_page = await run_find_in_page(pattern=pattern, page=page)
            return {
                "tool_name": "find",
                "page": result_page,
                "loc": 0,
                "num_lines": -1,
            }

        async def _prepare(spec: Dict[str, Any]):
            tool_name = spec["tool_name"].lower()
            tool_args = spec.get("tool_args") or {}
            if tool_name == "search":
                return await _prepare_search(tool_args)
            if tool_name == "open":
                return await _prepare_open(tool_args)
            if tool_name == "find":
                return await _prepare_find(tool_args)
            raise ToolUsageError(f"Unknown browser tool: {tool_name}")

        prepared = await asyncio.gather(
            *(_prepare(spec) for spec in tool_specs),
            return_exceptions=True,
        )

        results: List[str] = []
        for spec, prepared_item in zip(tool_specs, prepared):
            tool_name = spec["tool_name"].lower()
            if isinstance(prepared_item, Exception):
                results.append(f"Error executing browser.{tool_name}: {prepared_item}")
                continue

            try:
                tool.tool_state.add_page(prepared_item["page"])
                function_token = _live_function_name.set(prepared_item["tool_name"])
                try:
                    result_msg = await tool.show_page_safely(
                        loc=prepared_item["loc"],
                        num_lines=prepared_item["num_lines"],
                    )
                    results.append(_extract_text_from_harmony([result_msg]))
                finally:
                    _live_function_name.reset(function_token)
            except Exception as e:
                results.append(f"Error rendering browser.{tool_name} result: {e}")
        return results

    async def call(self, qid: Any, tool_call_msg_dict: dict, reasoning: str = None) -> list:
        """Execute a harmony-format tool call (gptoss path).

        Takes the tool-call message as a dict (from Message.to_dict()), dispatches
        it through BrowserTool.process(), and returns results as a list of dicts
        (each produced by Message.to_dict()).
        """
        from openai_harmony import Message as _Msg
        tool = self.sessions[qid]
        msg = _Msg.from_dict(tool_call_msg_dict)

        recipient = str(getattr(msg, "recipient", "") or "")
        reasoning_token = _ACTIVE_REASONING.set(reasoning if reasoning and "search" in recipient else None)

        try:
            results = []
            async for result_msg in tool.process(msg):
                results.append(result_msg.to_dict())

            return results
        finally:
            _ACTIVE_REASONING.reset(reasoning_token)

    def get_visited_urls(self, qid: Any) -> list:
        """Return deduplicated list of URLs seen during this session.

        Includes:
        - URLs listed in browser.search result pages (appeared in snippet list)
        - URLs fetched via browser.open

        Excludes internal pseudo-URLs (web-search://) and find-result pages.
        """
        tool = self.sessions.get(qid)
        if tool is None:
            return []
        seen: set = set()
        urls: list = []

        def _add(url: str) -> None:
            if url and url not in seen:
                seen.add(url)
                urls.append(url)

        for page_url in tool.tool_state.page_stack:
            if page_url.startswith("web-search://"):
                # search result page — collect every linked URL from the snippet list
                page = tool.tool_state.pages.get(page_url)
                if page and page.urls:
                    for linked_url in page.urls.values():
                        if isinstance(linked_url, str) and linked_url.startswith(("http://", "https://")):
                            _add(linked_url)
            elif page_url.startswith(("http://", "https://")) and "/find?pattern=" not in page_url:
                _add(page_url)

        return urls

    def cleanup(self, qid: Any):
        if qid in self.sessions:
            del self.sessions[qid]
