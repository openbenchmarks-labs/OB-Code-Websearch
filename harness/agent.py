"""Coding-agent loop with web search and fetch tools. Gold tokens in queries are flagged, not blocked."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .logger import request_scope, write_hop_pair, write_json
from .search import (
    DEFAULT_MAX_RESULTS,
    QUERY_PARAMETER,
    UNSCORED_FAILURES,
    SearchBackend,
    classify_failure,
    fit_search_budget,
    get_backend,
    hit_text_stats,
    page_text_stats,
)

DEFAULT_MODEL = "gpt-6-sol"
REASONING_EFFORT = "medium"

WEB_SEARCH_TOOL = {
    "type": "function",
    "name": "web_search",
    "description": (
        "Search the web. Required before you return an answer or file. "
        "Call it to find current docs. You may call it more than once "
        "with a new query. If you are unsure, search again rather than "
        "returning a wrong file."
    ),
    "parameters": {
        "type": "object",
        "properties": {"query": QUERY_PARAMETER},
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}

WEB_FETCH_TOOL = {
    "type": "function",
    "name": "web_fetch",
    "description": (
        "Fetch a specific URL through this vendor's native extract/scrape API. "
        "Use after search when snippets are not enough."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "https URL to fetch."},
            "objective": {
                "type": "string",
                "description": "What you need from the page. Empty string for the full page.",
            },
        },
        "required": ["url", "objective"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass
class SearchCall:
    query: str
    backend: str
    hits: list[dict[str, str]]
    error: str | None = None
    latency_ms: int = 0
    n_hits: int = 0
    chars: int = 0
    tokens_est: int = 0
    turn: int = 0
    seq: int = 0
    endpoint: str = ""
    http_status: int | None = None
    http_ms: int | None = None
    vendor_request_id: str = ""
    hit_urls: list[str] = field(default_factory=list)
    empty: bool = False
    gold_token: str | None = None
    budget_cut_chars: int = 0
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    # Set only under the "counts only when it returned a result" budget rule: whether this call used one of the searches.
    counted: bool | None = None
    # Queries the request carried: each one is one of the max_searches when the request returns a result.
    n_queries: int = 1
    failure_class: str | None = None
    request_attempted: bool = False
    notices: list[dict[str, Any]] = field(default_factory=list)

    def hop_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "backend": self.backend,
            "n_hits": self.n_hits,
            "chars": self.chars,
            "tokens_est": self.tokens_est,
            "latency_ms": self.latency_ms,
            "http_ms": self.http_ms,
            "http_status": self.http_status,
            "endpoint": self.endpoint,
            "vendor_request_id": self.vendor_request_id,
            "hit_urls": self.hit_urls,
            "empty": self.empty,
            "error": self.error,
            "failure_class": self.failure_class,
            "request_attempted": self.request_attempted,
            "notices": self.notices,
            "gold_token": self.gold_token,
            "budget_cut_chars": self.budget_cut_chars,
            "n_queries": self.n_queries,
            "turn": self.turn,
            "seq": self.seq,
            "hits": [
                {
                    "url": str(hit.get("url") or ""),
                    "title": str(hit.get("title") or ""),
                    "snippet": str(hit.get("snippet") or hit.get("text") or hit.get("content") or ""),
                }
                for hit in (self.hits or [])
                if isinstance(hit, dict)
            ],
            **({"counted": self.counted} if self.counted is not None else {}),
        }


@dataclass
class FetchCall:
    url: str
    backend: str
    title: str = ""
    content: str = ""
    objective: str = ""
    error: str | None = None
    latency_ms: int = 0
    chars: int = 0
    tokens_est: int = 0
    turn: int = 0
    seq: int = 0
    endpoint: str = ""
    http_status: int | None = None
    http_ms: int | None = None
    vendor_request_id: str = ""
    preview: str = ""
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    failure_class: str | None = None
    request_attempted: bool = False

    def hop_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "backend": self.backend,
            "title": self.title,
            "objective": self.objective,
            "chars": self.chars,
            "tokens_est": self.tokens_est,
            "latency_ms": self.latency_ms,
            "http_ms": self.http_ms,
            "http_status": self.http_status,
            "endpoint": self.endpoint,
            "vendor_request_id": self.vendor_request_id,
            "error": self.error,
            "failure_class": self.failure_class,
            "request_attempted": self.request_attempted,
            "turn": self.turn,
            "seq": self.seq,
            "preview": (self.preview or self.content or "")[:800],
            "content": self.content,
        }


@dataclass
class LlmTurn:
    turn: int
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    n_tool_calls: int
    tool_names: list[str] = field(default_factory=list)
    ended: bool = False
    error: str | None = None
    usage_available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentRun:
    question: str
    answer: str
    model: str
    backend: str
    used_search: bool
    used_fetch: bool = False
    run_id: str = ""
    searches: list[SearchCall] = field(default_factory=list)
    fetches: list[FetchCall] = field(default_factory=list)
    llm_turns: list[LlmTurn] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    wall_ms: int = 0
    llm_ms: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    hop_seq: int = 0
    error: str | None = None
    failure_origin: str | None = None
    evidence_dir: Path | None = None

    def metrics(self) -> dict[str, Any]:
        return {
            "wall_ms": self.wall_ms,
            "llm_ms": self.llm_ms,
            "search_count": sum(call.request_attempted for call in self.searches),
            # The cost column's inputs: each successful search is billed, an empty one only where the vendor
            # bills empty results (`bills_empty` in tools/*.yaml); failed calls cost 0 for every vendor.
            "successful_search_count": sum(1 for call in self.searches if call.request_attempted and not call.error),
            "empty_search_count": sum(
                1 for call in self.searches if call.request_attempted and not call.error and call.n_hits == 0),
            "fetch_count": sum(call.request_attempted for call in self.fetches),
            "search_tool_call_count": len(self.searches),
            "fetch_tool_call_count": len(self.fetches),
            "counted_searches": sum(call.n_queries for call in self.searches if call.counted is not False),
            "model_call_count": len(self.llm_turns),
            "model_usage_complete": all(turn.usage_available for turn in self.llm_turns),
            "search_ms": sum(call.latency_ms for call in self.searches),
            "fetch_ms": sum(call.latency_ms for call in self.fetches),
            "llm_prompt_tokens": self.llm_prompt_tokens,
            "llm_completion_tokens": self.llm_completion_tokens,
            "gold_token_queries": sum(1 for call in self.searches if call.gold_token),
        }


def run_agent(
    question: str,
    *,
    client: Any,
    backend: SearchBackend | None = None,
    backend_name: str = "parallel_fast_tool",
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = REASONING_EFFORT,
    max_turns: int = 32,
    max_searches: int = 5,
    max_fetches: int = 0,
    max_results: int = DEFAULT_MAX_RESULTS,
    search_fn: Callable[[str], list[dict[str, str]]] | None = None,
    fetch_fn: Callable[[str, str], dict[str, str]] | None = None,
    system: str,
    allow_search: bool = True,
    allow_fetch: bool = True,
    leak_needles: list[str] | None = None,
    text_format: dict[str, Any] | None = None,
    evidence_root: Path | None = None,
) -> AgentRun:
    backend = backend or (None if search_fn else get_backend(backend_name))
    provider = backend.name if backend is not None else backend_name
    if not allow_search:
        provider = "no_search"
    run = AgentRun(
        question=question,
        answer="",
        model=model,
        backend=provider,
        used_search=False,
        run_id=uuid.uuid4().hex,
    )
    if evidence_root is not None:
        run.evidence_dir = evidence_root / run.run_id
        run.evidence_dir.mkdir(parents=True, exist_ok=False)
    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    tools = []
    if allow_search:
        tools.append(_search_tool(backend))
    if allow_fetch:
        tools.append(WEB_FETCH_TOOL)
    # Every lane: each search call takes a turn, including separate calls in one model response, so empty or failed
    # searches, which do not use one of the max_searches, are still capped by max_turns.
    extra_search_turns = 0
    started = time.perf_counter()
    for model_turn in range(max_turns):
        turn = model_turn + extra_search_turns
        if turn >= max_turns:
            break
        llm_started = time.perf_counter()
        request = {
            "model": model, "reasoning": {"effort": reasoning_effort},
            "tools": tools, "input": conversation, "max_output_tokens": 4000, "store": False,
            **({"text": {"format": text_format}} if text_format else {}),
        }
        if run.evidence_dir is not None:
            write_json(run.evidence_dir / "model" / f"{model_turn + 1:02d}.request.json", request)
        try:
            with request_scope(run.evidence_dir):
                response = client.responses.create(**request)
            if run.evidence_dir is not None:
                payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else vars(response)
                write_json(run.evidence_dir / "model" / f"{model_turn + 1:02d}.response.json", payload)
        except Exception as exc:  # Preserve completed evidence when the model service fails.
            llm_ms = int((time.perf_counter() - llm_started) * 1000)
            run.llm_ms += llm_ms
            run.error = f"{type(exc).__name__}: {exc}"
            run.failure_origin = _model_failure_origin(exc)
            if run.evidence_dir is not None:
                write_json(run.evidence_dir / "model" / f"{model_turn + 1:02d}.error.json",
                           {"error": run.error, "failure_origin": run.failure_origin})
            run.llm_turns.append(LlmTurn(turn=turn, latency_ms=llm_ms, prompt_tokens=0,
                                        completion_tokens=0, n_tool_calls=0, ended=True,
                                        error=run.error, usage_available=False))
            break
        llm_ms = int((time.perf_counter() - llm_started) * 1000)
        prompt_tokens, completion_tokens = _usage(response)
        run.llm_ms += llm_ms
        run.llm_prompt_tokens += prompt_tokens
        run.llm_completion_tokens += completion_tokens
        calls = _function_calls(response)
        run.llm_turns.append(
            LlmTurn(
                turn=turn,
                latency_ms=llm_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                n_tool_calls=len(calls),
                tool_names=[_call_name(call) for call in calls],
                ended=not calls,
                usage_available=getattr(response, "usage", None) is not None,
            )
        )
        if not calls:
            run.answer = (getattr(response, "output_text", None) or "").strip()
            run.wall_ms = int((time.perf_counter() - started) * 1000)
            return run
        _append_model_output(conversation, response)
        search_calls_this_response = 0
        for call in calls:
            name = _call_name(call)
            if name == "web_search":
                if not allow_search:
                    conversation.append(_tool_output(call, {"error": "web_search is not available"}))
                    continue
                if search_calls_this_response:
                    if turn + 1 >= max_turns:
                        conversation.append(_tool_output(call, {"error": "turn budget exhausted; answer now"}))
                        run.notes.append("turn budget exhausted")
                        continue
                    turn += 1
                    extra_search_turns += 1
                search_calls_this_response += 1
                _run_search(
                    call,
                    run=run,
                    conversation=conversation,
                    backend=backend,
                    provider=provider,
                    max_searches=max_searches,
                    max_results=max_results,
                    search_fn=search_fn,
                    turn=turn,
                    leak_needles=leak_needles,
                    client_model=model,
                )
            elif name == "web_fetch":
                if not allow_fetch:
                    conversation.append(_tool_output(call, {"error": "web_fetch is not available; search again"}))
                    continue
                _run_fetch(
                    call,
                    run=run,
                    conversation=conversation,
                    backend=backend,
                    provider=provider,
                    max_fetches=max_fetches,
                    fetch_fn=fetch_fn,
                    turn=turn,
                )
            else:
                conversation.append(_tool_output(call, {"error": f"unknown tool {name}"}))
            if run.error:
                break
        if run.error:
            break
    run.notes.append("stopped after local/model failure" if run.error else "hit max turns without a final answer")
    run.wall_ms = int((time.perf_counter() - started) * 1000)
    return run


def _next_seq(run: AgentRun) -> int:
    run.hop_seq += 1
    return run.hop_seq


def _hidden_token_in_query(query: str, needles: list[str] | None) -> str | None:
    if not needles:
        return None
    hay = query.casefold()
    for needle in needles:
        token = (needle or "").strip()
        if len(token) < 4:
            continue
        if token.casefold() in hay:
            return token
    return None


def _vendor_meta(backend: SearchBackend | None) -> dict[str, Any]:
    return dict(getattr(backend, "last_meta", None) or {})


def _failed_call_meta(backend: SearchBackend | None, before: Any) -> dict[str, Any]:
    """Meta of the vendor HTTP call made by the tool call that just failed.

    vendor_call replaces backend.last_meta on every HTTP call, so an unchanged
    object means the failure came before any request: record nothing rather than
    an earlier call's endpoint and status.
    """
    after = getattr(backend, "last_meta", None)
    return dict(after) if after is not None and after is not before else {}


MULTI_QUERY_NOTE = "Each query in the list counts as one of your searches."


def _search_tool(backend: SearchBackend | None) -> dict[str, Any]:
    """web_search as every lane sees it; a lane with its vendor's documented parameters (`tool_parameters`) gets those
    parameters under the same name and description, and a lane whose call carries several queries (`queries_field`) is
    told that each query uses one search."""
    parameters = getattr(backend, "tool_parameters", None)
    if not parameters:
        return WEB_SEARCH_TOOL
    note = f" {MULTI_QUERY_NOTE}" if getattr(backend, "queries_field", None) else ""
    return {**WEB_SEARCH_TOOL, "description": WEB_SEARCH_TOOL["description"] + note, "parameters": parameters}


def _run_search(
    call: Any,
    *,
    run: AgentRun,
    conversation: list[dict[str, Any]],
    backend: SearchBackend | None,
    provider: str,
    max_searches: int,
    max_results: int,
    search_fn: Callable[[str], list[dict[str, str]]] | None,
    turn: int,
    leak_needles: list[str] | None,
    client_model: str = "",
) -> None:
    # One rule for every lane: each query is one search, so a request carrying N queries uses N of the max_searches
    # (the lane's tool text says so); a search with no results, or one that failed, does not use any; the turn it took
    # still counts. Billing is separate: one billed API request is one price, whatever it carries.
    args = _call_args(call)
    # A lane with its own tool parameters sends the model's fields; `query` records them as JSON.
    tool_lane = bool(getattr(backend, "tool_parameters", None)) and not search_fn
    request = backend.tool_request(args) if tool_lane else None
    if tool_lane:
        query = json.dumps(request, ensure_ascii=False) if request else ""
    else:
        query = str(args.get("query") or "").strip()
    if not query:
        fields = backend.tool_parameters["required"] if tool_lane else ["query"]
        conversation.append(_tool_output(call, {"error": f"{' and '.join(fields)} required"}))
        return
    queries_field = getattr(backend, "queries_field", None) if tool_lane else None
    left = max_searches - sum(done.n_queries for done in run.searches if done.n_hits > 0)
    if left <= 0:
        conversation.append(_tool_output(call, {"error": "search budget exhausted; answer now"}))
        run.notes.append("search budget exhausted")
        return
    budget_note = ""
    if queries_field and len(request[queries_field]) > left:
        request[queries_field] = request[queries_field][:left]
        query = json.dumps(request, ensure_ascii=False)
        budget_note = f"only {left} of your searches were left, so only the first {left} queries were run"
    n_queries = len(request[queries_field]) if queries_field else 1
    search_started = time.perf_counter()
    meta: dict[str, Any] = {}
    gold_token = _hidden_token_in_query(query, leak_needles)
    if gold_token:
        run.notes.append(f"query contained gold token:{gold_token}")
    before = getattr(backend, "last_meta", None)
    try:
        with request_scope(run.evidence_dir):
            if search_fn:
                hits = search_fn(query)
            elif tool_lane:
                hits = backend.search_tool(request, max_results=max_results, client_model=client_model)
            else:
                hits = backend.search(query, max_results=max_results)
        if not search_fn:
            meta = _vendor_meta(backend)
        error = None
    except Exception as exc:  # noqa: BLE001
        hits = []
        error = f"{type(exc).__name__}: {exc}"
        meta = _failed_call_meta(backend, before)
    # Vendor latency is the HTTP call alone (`http_ms`), not our rate-pacing wait before it.
    latency_ms = int(meta["http_ms"] if meta.get("http_ms") is not None else (time.perf_counter() - search_started) * 1000)
    hits, budget_cut_chars = fit_search_budget(hits)
    stats = hit_text_stats(hits)
    hit_urls = [str(hit.get("url") or "") for hit in hits if hit.get("url")]
    hop = SearchCall(
        query=query,
        backend=provider,
        hits=hits,
        error=error,
        latency_ms=latency_ms,
        n_hits=stats["n_hits"],
        chars=stats["chars"],
        tokens_est=stats["tokens_est"],
        turn=turn,
        seq=_next_seq(run),
        endpoint=str(meta.get("endpoint") or ""),
        http_status=meta.get("http_status"),
        http_ms=meta.get("http_ms"),
        vendor_request_id=str(meta.get("vendor_request_id") or ""),
        hit_urls=hit_urls,
        empty=stats["n_hits"] == 0,
        gold_token=gold_token,
        budget_cut_chars=budget_cut_chars,
        request=meta.get("request") if isinstance(meta.get("request"), dict) else None,
        response=meta.get("response") if isinstance(meta.get("response"), dict) else None,
        counted=stats["n_hits"] > 0,
        n_queries=n_queries,
        failure_class=classify_failure(meta, error),
        request_attempted=bool(meta.get("request")),
        notices=meta.get("notices", []),
    )
    run.used_search = True
    run.searches.append(hop)
    if run.evidence_dir is not None:
        stem = f"search-{len(run.searches):02d}"
        write_hop_pair(run.evidence_dir / "hops", stem, hop.request, hop.response)
        write_json(run.evidence_dir / "hops" / f"{stem}.delivered.json", hop.hop_dict())
    # Exa's own tool output leads with the notices, then the results: exa_py/tools.py:32-37.
    tool_result: dict[str, Any] = {**(request if tool_lane else {"query": query}),
                                   **({"notices": hop.notices} if hop.notices else {}),
                                   **({"budget": _budget_text(n_queries, hop.n_hits > 0, left, budget_note)}
                                      if queries_field else {}),
                                   "hits": hits}
    if error:
        # The vendor's own error text, the same for every vendor, so the model can act on it.
        vendor_text = str(meta.get("error_body") or "")[:500]
        tool_result["error"] = f"search failed ({hop.failure_class}, HTTP {hop.http_status})" + (
            f": {vendor_text}" if vendor_text else "")
        if hop.failure_class in UNSCORED_FAILURES:
            run.error, run.failure_origin = error, hop.failure_class
        if hop.failure_class == "account_billing":
            # Out of credit fails every later search of this vendor: stop the round, so no attempt is scored on it.
            raise BudgetStop(f"{provider} account out of credit (HTTP {hop.http_status})")
    conversation.append(_tool_output(call, tool_result))


def _budget_text(n_queries: int, counted: bool, left: int, truncated: str) -> str:
    """What a multi-query call used of the search budget, in the tool result."""
    used = n_queries if counted else 0
    text = f"{n_queries} queries: used {used} of your searches, {left - used} left" if n_queries > 1 else (
        f"used {used} of your searches, {left - used} left")
    return f"{truncated}; {text}" if truncated else text


def _run_fetch(
    call: Any,
    *,
    run: AgentRun,
    conversation: list[dict[str, Any]],
    backend: SearchBackend | None,
    provider: str,
    max_fetches: int,
    fetch_fn: Callable[[str, str], dict[str, str]] | None,
    turn: int,
) -> None:
    if len(run.fetches) >= max_fetches:
        conversation.append(_tool_output(call, {"error": "fetch budget exhausted; answer now"}))
        run.notes.append("fetch budget exhausted")
        return
    args = _call_args(call)
    url = str(args.get("url") or "").strip()
    objective = str(args.get("objective") or "").strip()
    if not url.startswith(("http://", "https://")):
        conversation.append(_tool_output(call, {"error": "url must be http(s)"}))
        return
    fetch_started = time.perf_counter()
    meta: dict[str, Any] = {}
    before = getattr(backend, "last_meta", None)
    try:
        with request_scope(run.evidence_dir):
            page = fetch_fn(url, objective) if fetch_fn else backend.fetch(url, objective=objective)
        meta = dict(page.pop("_meta", None) or {})
        if not meta and not fetch_fn:
            meta = _vendor_meta(backend)
        error = None
    except Exception as exc:  # noqa: BLE001
        page = {"url": url, "title": "", "content": ""}
        error = f"{type(exc).__name__}: {exc}"
        meta = _failed_call_meta(backend, before)
    latency_ms = int((time.perf_counter() - fetch_started) * 1000)
    stats = page_text_stats(page)
    content = str(page.get("content") or "")
    hop = FetchCall(
        url=str(page.get("url") or url),
        backend=provider,
        title=str(page.get("title") or ""),
        content=content,
        objective=objective,
        error=error,
        latency_ms=latency_ms,
        chars=stats["chars"],
        tokens_est=stats["tokens_est"],
        turn=turn,
        seq=_next_seq(run),
        endpoint=str(meta.get("endpoint") or ""),
        http_status=meta.get("http_status"),
        http_ms=meta.get("http_ms"),
        vendor_request_id=str(meta.get("vendor_request_id") or ""),
        preview=content[:800],
        request=meta.get("request") if isinstance(meta.get("request"), dict) else None,
        response=meta.get("response") if isinstance(meta.get("response"), dict) else None,
        failure_class=classify_failure(meta, error),
        request_attempted=bool(meta.get("request")),
    )
    run.used_fetch = True
    run.fetches.append(hop)
    if run.evidence_dir is not None:
        stem = f"fetch-{len(run.fetches):02d}"
        write_hop_pair(run.evidence_dir / "hops", stem, hop.request, hop.response)
        write_json(run.evidence_dir / "hops" / f"{stem}.delivered.json", hop.hop_dict())
    fetch_result: dict[str, Any] = {
        "url": page.get("url") or url,
        "title": page.get("title") or "",
        "content": page.get("content") or "",
    }
    if error:
        fetch_result["error"] = f"fetch failed ({hop.failure_class}, HTTP {hop.http_status})"
        if hop.failure_class in UNSCORED_FAILURES:
            run.error, run.failure_origin = error, hop.failure_class
    conversation.append(_tool_output(call, fetch_result))


def _function_calls(response: Any) -> list[Any]:
    return [item for item in (getattr(response, "output", None) or []) if _item_type(item) == "function_call"]


def _item_type(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("type") or "")
    return str(getattr(item, "type", "") or "")


def _call_name(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("name") or "")
    return str(getattr(call, "name", "") or "")


def _call_id(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("call_id") or call.get("id") or "")
    return str(getattr(call, "call_id", None) or getattr(call, "id", "") or "")


def _call_args(call: Any) -> dict[str, Any]:
    raw = call.get("arguments") if isinstance(call, dict) else getattr(call, "arguments", "")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _append_model_output(conversation: list[dict[str, Any]], response: Any) -> None:
    for item in getattr(response, "output", None) or []:
        if _item_type(item) in {"reasoning", "message"}:
            continue
        conversation.append(_item_as_input(item))


def _item_as_input(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    payload = {
        "type": getattr(item, "type", None),
        "id": getattr(item, "id", None),
        "call_id": getattr(item, "call_id", None),
        "name": getattr(item, "name", None),
        "arguments": getattr(item, "arguments", None),
        "status": getattr(item, "status", None),
    }
    return {key: value for key, value in payload.items() if value is not None}


class BudgetStop(RuntimeError):
    """The vendor account is out of credit; the round stops here."""


# Azure's content filter refuses a prompt with HTTP 400 and one of these error codes; the same request has passed
# on a later attempt, so the refusal is transient.
CONTENT_FILTER_CODES = frozenset({"content_filter", "invalid_prompt"})


def _model_failure_origin(exc: BaseException) -> str:
    """A budget stop is ours and final. Transient, so rerun on resume: the model client's connection or timeout
    failure (the SDK raises APIConnectionError, or its subclass APITimeoutError, only for a transport error); an HTTP
    400 whose error code names the content filter; an HTTP 429 (our key's rate limit or our concurrency); an HTTP 5xx or
    408 (the model service's own failure to answer). Any other model error (another refusal, a bad request, any other
    4xx) is the model's and final."""
    if isinstance(exc, BudgetStop):
        return "local"
    from openai import (
        APIConnectionError,
        APIStatusError,
        BadRequestError,
        InternalServerError,
        RateLimitError,
    )

    if isinstance(exc, APIConnectionError):
        return "model_transport"
    if isinstance(exc, BadRequestError) and exc.code in CONTENT_FILTER_CODES:
        return "model_content_filter"
    if isinstance(exc, RateLimitError):
        return "model_rate_limit"
    if isinstance(exc, InternalServerError) or (isinstance(exc, APIStatusError) and exc.status_code == 408):
        return "model_server"
    return "model"


def _tool_output(call: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": _call_id(call), "output": json.dumps(payload, ensure_ascii=False)}


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    return int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)
