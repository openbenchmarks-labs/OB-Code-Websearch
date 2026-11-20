"""Pluggable web-search backends.

HTTP call + hit shape for Parallel, Firecrawl, Exa, Linkup, Tavily, Brave
(LLM Context), SERP RapidAPI, You.com, TinyFish, Tako, Perplexity, Nimble and
Keenable. Vendors are locked to search-only vs search-fetch boards in
SEARCH_ONLY_BACKENDS / SEARCH_FETCH_BACKENDS. Nimble, Tavily Basic, Firecrawl,
You and TinyFish sit on both. Keenable, Tako, Perplexity and SERP are search-only.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections.abc import Callable
from typing import Any, Protocol

import requests

# --- Nimble REST helper ---
"""Shared Nimble request shapes and source-only response parsing."""


import math

_NIMBLE_SEARCH_URL = "https://sdk.nimbleway.com/v2/search"
_NIMBLE_EXTRACT_URL = "https://sdk.nimbleway.com/v2/extract"
SEARCH_PRICES = {"lite": 0.0011, "standard": 0.005}


def search_body(query: str, depth: str, max_results: int) -> dict[str, Any]:
    if depth not in SEARCH_PRICES:
        raise ValueError(f"Unsupported Nimble search depth: {depth}")
    return {"query": query, "search_depth": depth, "full_content": False,
            "focus": "general", "max_results": max_results}


def _nimble_search_hits(payload: Any, max_results: int) -> list[dict[str, str]]:
    rows = payload.get("results") if isinstance(payload, dict) else None
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("url"), str):
            continue
        url = row["url"].strip()
        if not url or url in seen:
            continue
        seen.add(url)
        hits.append({"url": url, "title": str(row.get("title") or ""),
                     "snippet": str(row.get("description") or "")})
        if len(hits) >= max_results:
            break
    return hits


def _nimble_extract_page(payload: Any, url: str, max_chars: int) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("status") not in (None, "success"):
        raise RuntimeError("Nimble Extract returned an unsuccessful response")
    status = payload.get("status_code")
    if status is not None and not 200 <= int(status) < 300:
        raise RuntimeError(f"Nimble Extract upstream HTTP {status}")
    data = payload.get("data") or {}
    text = data.get("markdown") if isinstance(data, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Nimble Extract returned no markdown content")
    return {"requested_url": url, "final_url": payload.get("url") or url,
            "title": "", "text": text[:max_chars], "truncated": len(text) > max_chars,
            "fetch_provider": "nimble_extract"}


def extract_price() -> float | None:
    raw = os.getenv("NIMBLE_EXTRACT_USD_PER_REQUEST", "").strip()
    if not raw:
        return None
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError("NIMBLE_EXTRACT_USD_PER_REQUEST must be finite and nonnegative")
    return value
# --- End Nimble REST helper ---

from .logger import begin_request, end_request

SEARCH_PREVIEW_BUDGET_CHARS = 32_000

PARALLEL_SEARCH_URL = "https://api.parallel.ai/v1/search"
PARALLEL_EXTRACT_URL = "https://api.parallel.ai/v1/extract"
FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v2/search"
FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"
EXA_SEARCH_URL = "https://api.exa.ai/search"
EXA_CONTENTS_URL = "https://api.exa.ai/contents"
LINKUP_SEARCH_URL = "https://api.linkup.so/v1/search"
LINKUP_FETCH_URL = "https://api.linkup.so/v1/fetch"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
BRAVE_LLM_CONTEXT_URL = "https://api.search.brave.com/res/v1/llm/context"
SERP_SEARCH_URL = "https://google-search74.p.rapidapi.com/"
SERP_HOST = "google-search74.p.rapidapi.com"
YOU_SEARCH_URL = "https://ydc-index.io/v1/search"
YOU_CONTENTS_URL = "https://ydc-index.io/v1/contents"
TINYFISH_SEARCH_URL = "https://api.search.tinyfish.ai"
TINYFISH_FETCH_URL = "https://api.fetch.tinyfish.ai"
TINYFISH_FETCH_TIMEOUT_S = 150
TAKO_SEARCH_URL = "https://tako.com/api/v3/search"
PERPLEXITY_SEARCH_URL = "https://api.perplexity.ai/search"
STRING_SEARCH_URL = "https://request.usestring.ai/v1/search"
STRING_FETCH_URL = "https://request.usestring.ai/v1/fetch"
KEENABLE_SEARCH_URL = "https://api.keenable.ai/v1/search"
HEADER_REDACT_TOKENS = (
    "authorization",
    "api-key",
    "api_key",
    "apikey",
    "x-api-key",
    "x-rapidapi-key",
    "rapidapi",
    "token",
    "secret",
    "password",
    "cookie",
)
# Failures on our side: the attempt stops and is not scored against the search tool. A transport failure (TLS,
# connection reset, no HTTP status) is ours; a 429 is our
# key's rate limit or our concurrency; the model client's connection or timeout failure (openai APIConnectionError) is
# ours and transient too. `served_mode_mismatch`: the vendor served another mode than the lane sent, so the row would
# measure a different product.
UNSCORED_FAILURES = frozenset({"local_error", "auth_configuration", "redirect_not_followed", "account_billing",
                               "transport_error", "rate_limit", "vendor_server", "model_transport",
                               "served_mode_mismatch"})
# Transient: `--resume` runs such an attempt again, so every lane ends with the same question count. The first three
# are ours; `model_content_filter` is Azure's content-filter refusal (HTTP 400, error code `content_filter` or
# `invalid_prompt`), which is not the search tool's doing and has passed on an identical retry; `model_rate_limit` is
# the model service's HTTP 429, as a search vendor's is `rate_limit`, and `model_server` its HTTP 5xx or 408, as a
# search vendor's is `vendor_server`: one rule for both sides, under the same try cap. `recording_timeout` is the
# stand-in recording of a starter or reference fix that did not finish in time: ours, and a question asked without its
# "The program sends this request today" line would measure nothing.
RERUN_FAILURES = frozenset({"transport_error", "rate_limit", "vendor_server", "model_transport",
                            "model_content_filter", "model_rate_limit", "model_server", "recording_timeout"})
DEFAULT_MAX_RESULTS = 8
DEFAULT_MAX_FETCH_CHARS = 16000
FETCH_TIMEOUT_S = 90
# One search timeout for every lane: the largest any release lane used (Firecrawl, Nimble, Parallel basic).
SEARCH_TIMEOUT_S = 60
QUERY_PARAMETER = {"type": "string", "description": "Focused search query for this hop."}


class SearchHit(dict):
    """url / title / snippet."""


class SearchBackend(Protocol):
    name: str

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        ...

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        ...


# One rule for a model-written `site:`, the same for every vendor, set per lane by `site_lift` from its vendor's docs
# (cited at each class):
#   True   the lane sends the vendor's documented domain filter: every `site:` host moves into it and the path is
#          dropped; path segments never become query words.
#   False  the lane has no domain filter in its request and its vendor applies `site:` in the query: the agent's
#          `site:host/path` stays as written.
#   None   the query goes out exactly as the agent wrote it (comparison arms that measure the vendor without the rule).
# A query left with no terms besides operators (`site:host/path` alone) gets the product name of each host as its
# terms, since vendors reject or misread a query that is only an operator; under `None` it goes out as written.
_SITE_INCLUDE_RE = re.compile(r"(?<![\w-])site:([^\s]+)", re.I)
_OPERATOR_RE = re.compile(r"(?<![\w])-?site:\S+|(?<!\S)(?:OR|AND)(?!\S)", re.I)
_CONJUNCTIONS = frozenset({"or", "and"})
_HOST_RE = re.compile(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+")
_SECOND_LEVEL_LABELS = frozenset({"co", "com", "org", "net", "gov", "edu", "ac"})
SiteLift = bool | None


def _site_host(value: str) -> str:
    """The host of a `site:` value, with scheme, `*.`, `www.`, port, path, query string and fragment removed."""
    raw = re.sub(r"^[a-z][a-z0-9+.-]*://", "", value.strip("\"'"), flags=re.I)
    host = re.split(r"[/?#]", raw, maxsplit=1)[0]
    return host.lower().removeprefix("*.").removeprefix("www.").split(":", 1)[0].strip(".")


def _product_name(host: str) -> str:
    """The host's product name: its label before the public suffix (`help.sap.com` is `sap`)."""
    labels = host.split(".")
    return labels[-3] if len(labels) >= 3 and labels[-2] in _SECOND_LEVEL_LABELS and len(labels[-1]) == 2 else labels[-2]


def rewrite_site(query: str, *, lift: bool) -> tuple[str, list[str]]:
    """(query, hosts moved into a domain filter). Under `lift` each valid `site:` value is removed from the query and
    its host goes to the filter, the path dropped; otherwise the query keeps its `site:` values as written and no
    filter is sent. A query with no terms left besides operators gets the product name of each host as its terms.
    `-site:` and words ending in `site:` (website:) stay as written; a `site:` value with no valid host stays as
    written."""
    hosts: list[str] = []

    def _take(match: re.Match[str]) -> str:
        host = _site_host(match.group(1))
        if not _HOST_RE.fullmatch(host):
            return match.group(0)
        if host not in hosts:
            hosts.append(host)
        return "" if lift else match.group(0)

    rewritten = _SITE_INCLUDE_RE.sub(_take, query)
    if not hosts:
        return query, []
    words = rewritten.split()
    words = [word for index, word in enumerate(words) if not (
        word.lower() in _CONJUNCTIONS and (index == 0 or index == len(words) - 1 or words[index - 1].lower() in _CONJUNCTIONS))]
    cleaned = " ".join(words)
    if not _OPERATOR_RE.sub("", cleaned).strip():
        cleaned = " ".join([cleaned, *dict.fromkeys(_product_name(host) for host in hosts)]).strip()
    return cleaned, hosts if lift else []


def site_request(backend: Any, query: str) -> tuple[str, list[str]]:
    """The query text and domain filter a backend sends, under its `site_lift`."""
    lift = getattr(backend, "site_lift", False)
    return (query, []) if lift is None else rewrite_site(query, lift=lift)


def site_request_all(backend: Any, queries: list[str]) -> tuple[list[str], list[str]]:
    """The queries and the one request-level domain filter for a request carrying several queries. The filter scopes
    every query in the request, so hosts are lifted only when every query names the same hosts; otherwise no filter is
    sent and each query keeps its `site:` as written."""
    lift = getattr(backend, "site_lift", False)
    if lift is None:
        return list(queries), []
    lifted = [rewrite_site(query, lift=True) for query in queries] if lift else []
    if lifted and lifted[0][1] and all(set(hosts) == set(lifted[0][1]) for _, hosts in lifted):
        return [sent for sent, _ in lifted], lifted[0][1]
    return [rewrite_site(query, lift=False)[0] for query in queries], []


class IntentTool:
    """The one tool-wording rule, the same for every vendor in tools/*.yaml: a lane uses the vendor's official
    MCP server or SDK tool text (the description and wording of each field the model writes, under the vendor's field
    names) where the vendor publishes one; otherwise QUERY_PARAMETER. Each class cites its source. All 8 vendors
    publish one: Exa (exa-py), Linkup (`linkup_search`), TinyFish (`SEARCH_TOOL`), Firecrawl (`firecrawl_search`), Brave
    (`brave_llm_context`), Tavily (`tavily_search`), You (`you-search`) and Parallel (`search_web`, ParallelFastTool and
    its mode subclasses; Parallel's non-tool arms are
    comparison arms that measure it without its tool). The tool name and
    its description stay the shared `web_search` ones (agent.WEB_SEARCH_TOOL), so every lane sees one tool. Clauses about
    tools or request options this board does not have are left out, and each class names them. Fields the board fixes
    (mode, depth, result count, output type, content mode, domain filter) stay fixed and hidden; only the vendor's intent
    field and the optional fields listed per class are model-written.
    """

    query_description: str = ""
    intent_field: str | None = None
    intent_description: str = ""
    # Optional model-written array fields with an enum; the strict tool lists them as required and nullable.
    optional_fields: dict[str, dict[str, Any]] = {}

    @property
    def tool_parameters(self) -> dict[str, Any]:
        properties: dict[str, Any] = {"query": {"type": "string", "description": self.query_description}}
        if self.intent_field:
            properties[self.intent_field] = {"type": "string", "description": self.intent_description}
        properties.update(self.optional_fields)
        return {"type": "object", "properties": properties, "required": list(properties),
                "additionalProperties": False}

    def tool_request(self, args: dict[str, Any]) -> dict[str, Any] | None:
        """The model's fields as sent; None when `query` or the intent field is missing or empty, or an optional field
        is outside its schema. A null or empty optional field is left out of the request."""
        request: dict[str, Any] = {}
        for name in ["query", *([self.intent_field] if self.intent_field else [])]:
            value = args.get(name)
            if not isinstance(value, str) or not value.strip():
                return None
            request[name] = value.strip()
        for name, schema in self.optional_fields.items():
            value = args.get(name)
            if value is None or value == []:
                continue
            if (not isinstance(value, list) or len(value) > schema["maxItems"]
                    or any(item not in schema["items"]["enum"] for item in value)):
                return None
            request[name] = list(value)
        return request

    def search_tool(self, request: dict[str, Any], *, max_results: int, client_model: str) -> list[dict[str, str]]:
        del client_model
        fields = {name: value for name, value in request.items() if name != "query"}
        return self.search(request["query"], max_results=max_results, **fields)


def _record_site_lift(backend: Any, sent: str, include_domains: list[str]) -> None:
    if include_domains:
        backend.last_meta["include_domains"] = include_domains
        backend.last_meta["query_stripped"] = sent


class ParallelBasic:
    name = "parallel_basic"
    mode = "basic"
    # Parallel's guide says not to put `site:` operators in queries (parallel-best-practices.md:118); its
    # source_policy.include_domains takes the host (parallel-api-search.md:540).
    site_lift: SiteLift = True
    last_meta: dict[str, Any] | None = None

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        key = os.environ.get("PARALLEL_API_KEY")
        if not key:
            raise RuntimeError(f"PARALLEL_API_KEY is required for {self.name}")
        sent, include_domains = site_request(self, query)
        advanced: dict[str, Any] = {"max_results": max_results}
        if include_domains:
            advanced["source_policy"] = {"include_domains": include_domains}
        payload, meta = vendor_call(self,
            "POST",
            PARALLEL_SEARCH_URL,
            headers={"x-api-key": key, "Content-Type": "application/json"},
            # `objective` is Parallel's intent field; only the tool lanes, where the model writes it, send one.
            json_body={
                "search_queries": [sent],
                "mode": self.mode,
                "advanced_settings": advanced,
            },
            timeout=SEARCH_TIMEOUT_S,
        )
        hits = parse_parallel_hits(payload, max_results=max_results)
        self.last_meta = _search_meta(meta, hits)
        _record_site_lift(self, sent, include_domains)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        key = os.environ.get("PARALLEL_API_KEY")
        if not key:
            raise RuntimeError(f"PARALLEL_API_KEY is required for {self.name}")
        body: dict[str, Any] = {
            "urls": [url],
            "max_chars_total": DEFAULT_MAX_FETCH_CHARS,
            "advanced_settings": {"full_content": True},
        }
        if objective.strip():
            body["objective"] = objective.strip()
        payload, meta = vendor_call(self,
            "POST",
            PARALLEL_EXTRACT_URL,
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json_body=body,
            timeout=FETCH_TIMEOUT_S,
        )
        page = parse_parallel_extract(payload, url=url)
        flags = _pop_flags(page)
        meta = {**meta, **flags, "extract_id": flags.get("extract_id") or meta.get("extract_id")}
        self.last_meta = meta
        page["_meta"] = meta
        return page


class ParallelFast(ParallelBasic):
    name = "parallel_fast"
    mode = "fast"


class ParallelFastRaw(ParallelFast):
    """Parallel fast with the model's query sent unchanged as the one `search_queries` entry, `site:` included, and no
    `objective` (settings comparison arm)."""

    name = "parallel_fast_raw"
    billed_sku = "parallel_fast"
    site_lift: SiteLift = None


class ParallelFastTool(ParallelFast):
    """Parallel fast with Parallel's documented search tool, the headline lane.

    The model sees `web_search` with Parallel's `search_web` parameters (`objective`, `search_queries`); each field
    description follows Parallel's own tool text (docs.parallel.ai/search/best-practices.md "Search Tool Definition", read
    9 Nov 2026). The model writes `objective` and 1-3 `search_queries` on every call (2-3 asked, 1-3 accepted). The harness adds `client_model` (the run's model) and one `session_id` per
    question (one backend object per attempt). `source_policy.include_domains` is request-level
    (`advanced_settings.source_policy`, parallel-api-search.md:540) and Parallel says to use it "only when absolutely
    necessary" (docs.parallel.ai/resources/source-policy.md), so `site:` hosts are lifted only when every query in the
    call names the same hosts (`site_request_all`); otherwise each query keeps its `site:` as written. Each query in the call is
    one of the searches in the budget (`queries_field`); the request is billed once, as its mode.
    """

    name = "parallel_fast_tool"
    billed_sku = "parallel_fast"
    queries_field = "search_queries"
    tool_parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "objective": {
                "type": "string",
                "description": ("Describe what you are trying to accomplish in one concise, standalone sentence: the "
                                "product or library, its version, and the error or goal."),
            },
            "search_queries": {
                "type": "array",
                "description": (
                    "Provide 2-3 short keyword variants of 3-6 words each. Include the key entity or topic in every "
                    "query. Vary names, synonyms, or angles. Do not use sentences, instructions, or site: operators."
                ),
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 3,
            },
        },
        "required": ["objective", "search_queries"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.session_id = f"session_{uuid.uuid4().hex}"

    def tool_request(self, args: dict[str, Any]) -> dict[str, Any] | None:
        """Validate the declared tool shape before spending a search request."""
        queries, objective = args.get("search_queries"), args.get("objective")
        if (not isinstance(objective, str) or not objective.strip()
                or not isinstance(queries, list) or not 1 <= len(queries) <= 3
                or any(not isinstance(query, str) or not query.strip() for query in queries)):
            return None
        return {"objective": objective.strip(), "search_queries": [query.strip() for query in queries]}

    def search_tool(self, request: dict[str, Any], *, max_results: int, client_model: str) -> list[dict[str, str]]:
        key = os.environ.get("PARALLEL_API_KEY")
        if not key:
            raise RuntimeError(f"PARALLEL_API_KEY is required for {self.name}")
        sent, include_domains = site_request_all(self, request["search_queries"])
        advanced: dict[str, Any] = {"max_results": max_results}
        if include_domains:
            advanced["source_policy"] = {"include_domains": include_domains}
        payload, meta = vendor_call(self,
            "POST",
            PARALLEL_SEARCH_URL,
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json_body={
                "objective": request["objective"],
                "search_queries": sent,
                "mode": self.mode,
                "client_model": client_model,
                "session_id": self.session_id,
                "advanced_settings": advanced,
            },
            timeout=SEARCH_TIMEOUT_S,
        )
        hits = parse_parallel_hits(payload, max_results=max_results)
        self.last_meta = _search_meta(meta, hits)
        _record_site_lift(self, " | ".join(sent), include_domains)
        return hits


class ParallelAdvancedTool(ParallelFastTool):
    """Settings option: Parallel's higher-accuracy search mode, search only. Parallel's API defaults to `advanced`
    when `mode` is omitted (parallel-api-search.md:262-265). One 60 s search timeout, as every lane."""

    name = "parallel_advanced_tool"
    billed_sku = "parallel_advanced"
    mode = "advanced"


class NoSearch:
    """No web_search / web_fetch. Memory baseline for the locked tickets."""

    name = "no_search"
    last_meta: dict[str, Any] | None = None

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        del query, max_results
        raise RuntimeError("no_search has no web_search")

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del url, objective
        raise RuntimeError("no_search has no web_fetch")


class FirecrawlSearch(IntentTool):
    """Firecrawl search + scrape: plain `/v2/search`, `{query, limit}`, Firecrawl's general search endpoint. The model
    sees Firecrawl's own search tool's `query` (firecrawl-mcp-server src/tool-helpers.ts and server.ts, read 10 Nov
    2026) and nothing else. Left out: `categories` (documented, docs.firecrawl.dev/
    api-reference/endpoint/search.md:311-320, but Firecrawl's tool pushes `developer` for every programming question,
    which this board does not), `scrapeOptions` (fetches pages), `sources` and `domainTools` (Alexandria tools),
    `limit` (8 for every vendor), `includeDomains`/`excludeDomains` (the shared `site:` rule owns the filter)."""

    name = "firecrawl"
    # Firecrawl documents `site:` as a query operator ("Only returns results from a specified website",
    # docs.firecrawl.dev/api-reference/endpoint/search.md:19): `site:` stays in the query as written.
    site_lift: SiteLift = False
    # tool-helpers.ts:59 "Query for web and semantic tool discovery. Operators include ...; the set is non-exhaustive.
    # Catalogue browsing is available through firecrawl_find_tools.", with tool discovery and the catalogue tool, which
    # this board does not have, left out.
    query_description = ("Query for web search. Operators include quoted phrases, `-term`, `site:host`, `inurl:term`, "
                         "`intitle:term`, and `related:host`; the set is non-exhaustive.")
    last_meta: dict[str, Any] | None = None

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        key = os.environ.get("FIRECRAWL_API_KEY")
        if not key:
            raise RuntimeError("FIRECRAWL_API_KEY is required for firecrawl")
        sent, include_domains = site_request(self, query)
        body: dict[str, Any] = {"query": sent, "limit": max_results}
        if include_domains:
            body["includeDomains"] = include_domains
        payload, meta = vendor_call(self,
            "POST",
            FIRECRAWL_SEARCH_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json_body=body,
            timeout=SEARCH_TIMEOUT_S,
        )
        hits = parse_firecrawl_hits(payload, max_results=max_results)
        self.last_meta = _search_meta(meta, hits)
        _record_site_lift(self, sent, include_domains)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del objective
        key = os.environ.get("FIRECRAWL_API_KEY")
        if not key:
            raise RuntimeError("FIRECRAWL_API_KEY is required for firecrawl")
        payload, meta = vendor_call(self,
            "POST",
            FIRECRAWL_SCRAPE_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json_body={"url": url, "formats": ["markdown"]},
            timeout=FETCH_TIMEOUT_S,
        )
        page = parse_firecrawl_scrape(payload, url=url)
        flags = _pop_flags(page)
        self.last_meta = {**meta, **flags}
        page["_meta"] = self.last_meta
        return page


class ExaSearch(IntentTool):
    """Exa search. Fetch is POST /contents with text=true (get_contents)."""

    name = "exa_auto"
    search_type = "auto"
    # Exa honours `site:` in the query and relaxes it when nothing matches (RESTRICTIONS_RELAXED,
    # exa-best-practices.md:324); includeDomains never relaxes. So `site:` stays in the query as written. exa-spec.yaml:3237
    # prefers includeDomains; that filter is the settings-round option `exa_auto_domains`, not the headline.
    site_lift: SiteLift = False
    # Exa's own web-search tool, exa-py 2.28.0 exa_py/tools.py:64-68 `_WebSearchInput.query` (best practices
    # "Describe the pages you want, not a bag of keywords", exa-best-practices.md:76-83).
    query_description = ("Natural language search query. Should be a semantically rich description of the ideal page, "
                         "not just keywords.")
    # exa-spec.yaml:3419-3424 `objective`, with the description Exa gives for exposing it as an agent tool parameter
    # (exa_py/tools.py:24-27, required in the advertised schema: tools.py:59-61).
    intent_field = "objective"
    intent_description = ("Goal for this search turn; say which documents should rank first, which should be "
                          "excluded, and what specific facts or figures to pull from them.")
    last_meta: dict[str, Any] | None = None

    def _key(self) -> str:
        key = os.environ.get("EXA_API_KEY")
        if not key:
            raise RuntimeError(f"EXA_API_KEY is required for {self.name}")
        return key

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS, objective: str | None = None
               ) -> list[dict[str, str]]:
        sent, include_domains = site_request(self, query)
        payload, meta = vendor_call(self,
            "POST",
            EXA_SEARCH_URL,
            headers={"x-api-key": self._key(), "Content-Type": "application/json"},
            json_body={
                "query": sent,
                **({"objective": objective} if objective else {}),
                "type": self.search_type,
                "numResults": max_results,
                "contents": {"highlights": True},
                **({"includeDomains": include_domains} if include_domains else {}),
            },
            timeout=120 if self.search_type == "deep" else SEARCH_TIMEOUT_S,
        )
        hits = parse_exa_hits(payload, max_results=max_results)
        self.last_meta = _search_meta(meta, hits)
        _record_site_lift(self, sent, include_domains)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del objective
        payload, meta = vendor_call(self,
            "POST",
            EXA_CONTENTS_URL,
            headers={"x-api-key": self._key(), "Content-Type": "application/json"},
            json_body={"urls": [url], "text": True},
            timeout=FETCH_TIMEOUT_S,
        )
        page = parse_exa_contents(payload, url=url)
        flags = _pop_flags(page)
        self.last_meta = {**meta, **flags}
        page["_meta"] = self.last_meta
        return page


class ExaAutoDomains(ExaSearch):
    """Settings arm: Exa `includeDomains`, "List of domains or domain paths to include in the search ... Use this
    parameter for domain or path filtering instead of adding a `site:` operator to the query" (exa-spec.yaml:3231-3237).
    Every `site:` host moves into it, and `includeDomains` never relaxes, as `site:` in the query does."""

    name = "exa_auto_domains"
    billed_sku = "exa_auto"
    site_lift: SiteLift = True


class ExaFast(ExaSearch):
    """Legacy Exa mode: "`fast`: use `lite`" (exa-changelog.md:27-30; exa-spec.yaml:3347 "Legacy. Use `lite`
    instead"). A comparison arm, never Exa's headline."""

    name = "exa_fast"
    search_type = "fast"


class ExaInstant(ExaSearch):
    """Legacy Exa mode: "`instant`: use `lite`" (exa-changelog.md:27-29; exa-spec.yaml:3346). A comparison arm,
    never Exa's headline."""

    name = "exa_instant"
    search_type = "instant"


class ExaLite(ExaSearch):
    """Exa `type: "lite"`, Exa's recommended replacement for the legacy `instant` (changelog 19 Oct 2026); body and
    timeout as `exa_instant`."""

    name = "exa_lite"
    search_type = "lite"


class LinkupSearch(IntentTool):
    """Linkup search + fetch.

    Search: depth=fast or standard (subclass), outputType=searchResults ($0.005 each). The model sees `query` from
    Linkup's published tool definition (linkup-for-agents.md:79-95, `linkup_search`). Depth, outputType and the
    result count are fixed by the board; includeDomains comes from the shared `site:` rule.

    Fetch: mode=standard, renderJs=true. JS-rendered markdown extract.
    """

    name = "linkup_fast"
    depth = "fast"
    output_type = "searchResults"
    fetch_mode = "standard"
    render_js = True
    # Linkup's docs never mention `site:`, and includeDomains takes domains: "Restrict the search to up to 100
    # domains" (linkup-for-agents.md:114; OpenAPI "The domains you want to search on").
    site_lift: SiteLift = True
    # linkup-for-agents.md:95, without its last sentence, which is about search-then-scrape chains (deep depth and
    # page scraping, outside this search-only board).
    query_description = ("Natural-language question or instruction. Be specific. Include the key entity, date range, "
                         "or domain when known.")
    last_meta: dict[str, Any] | None = None

    def _key(self) -> str:
        key = os.environ.get("LINKUP_API_KEY")
        if not key:
            raise RuntimeError(f"LINKUP_API_KEY is required for {self.name}")
        return key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"}

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        q, include_domains = site_request(self, query)
        body: dict[str, Any] = {
            "q": q,
            "depth": self.depth,
            "outputType": self.output_type,
            "maxResults": max_results,
        }
        if include_domains:
            body["includeDomains"] = include_domains
        try:
            payload, meta = vendor_call(self,
                "POST",
                LINKUP_SEARCH_URL,
                headers=self._headers(),
                json_body=body,
                timeout=SEARCH_TIMEOUT_S,
            )
        except requests.HTTPError as exc:
            # Linkup answers "Search query yields no result" with HTTP 400 SEARCH_QUERY_NO_RESULT
            # (docs.linkup.so/pages/documentation/platform/errors): an empty result, not a rejected request.
            meta = dict(getattr(exc, "vendor_meta", None) or {})
            if meta.get("http_status") != 400 or "SEARCH_QUERY_NO_RESULT" not in str(meta.get("error_body") or ""):
                raise
            payload = {"results": []}
        hits = parse_linkup_hits(payload, max_results=max_results)
        self.last_meta = _search_meta(meta, hits)
        _record_site_lift(self, q, include_domains)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del objective
        payload, meta = vendor_call(self,
            "POST",
            LINKUP_FETCH_URL,
            headers=self._headers(),
            json_body={
                "url": url,
                "mode": self.fetch_mode,
                "renderJs": self.render_js,
            },
            timeout=FETCH_TIMEOUT_S,
        )
        page = parse_linkup_fetch(payload, url=url)
        flags = _pop_flags(page)
        self.last_meta = {**meta, **flags}
        page["_meta"] = self.last_meta
        return page


class LinkupStandard(LinkupSearch):
    name = "linkup_standard"
    depth = "standard"


class TavilySearch(IntentTool):
    """Tavily search + extract.

    `tavily_fast` (search_depth=fast) is not on either board: the boards run Tavily basic and advanced only. The
    `tavily` alias is the headline,
    `tavily_advanced`. include_answer and include_raw_content stay off: the agent writes the answer; fetch owns page
    bodies. Fetch: extract_depth=basic, format=markdown.

    Every call of one attempt carries one `X-Session-Id` (one backend object per attempt): "opaque identifier for a
    session of related requests ... all requests from one user task" (docs.tavily.com/documentation/api-headers.md,
    "Session Tracking"; used for attribution and analytics).
    """

    name = "tavily_fast"
    search_depth = "fast"
    extract_depth = "basic"
    chunks_per_source = 3
    # Tavily's include_domains takes hosts only (docs.tavily.com/changelog, Aug 2026, include_domains_mode).
    site_lift: SiteLift = True
    # Tavily's official MCP server, `tavily_search` `query` (github.com/tavily-ai/tavily-mcp src/index.ts:173-176 at
    # ff166aa8, 5 Nov 2026). Its other fields are request options the board fixes.
    query_description = ("Search query. Keep it short and specific, in the words the page itself would use; one idea "
                         "per query.")
    last_meta: dict[str, Any] | None = None

    def __init__(self) -> None:
        self.session_id = str(uuid.uuid4())

    def _key(self) -> str:
        key = os.environ.get("TAVILY_API_KEY")
        if not key:
            raise RuntimeError(f"TAVILY_API_KEY is required for {self.name}")
        return key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json",
                "X-Session-Id": self.session_id}

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        sent, include_domains = site_request(self, query)
        body: dict[str, Any] = {
            "query": sent,
            "search_depth": self.search_depth,
            "max_results": max_results,
            "chunks_per_source": self.chunks_per_source,
            "include_answer": False,
            "include_raw_content": False,
            "topic": "general",
        }
        if include_domains:
            body["include_domains"] = include_domains
        payload, meta = vendor_call(self,
            "POST",
            TAVILY_SEARCH_URL,
            headers=self._headers(),
            json_body=body,
            timeout=SEARCH_TIMEOUT_S,
        )
        hits = parse_tavily_hits(payload, max_results=max_results)
        self.last_meta = _search_meta(meta, hits)
        _record_site_lift(self, sent, include_domains)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        body: dict[str, Any] = {
            "urls": [url],
            "extract_depth": self.extract_depth,
            "format": "markdown",
            "timeout": 60,
        }
        if objective.strip():
            body["query"] = objective.strip()
        payload, meta = vendor_call(self,
            "POST",
            TAVILY_EXTRACT_URL,
            headers=self._headers(),
            json_body=body,
            timeout=FETCH_TIMEOUT_S,
        )
        page = parse_tavily_extract(payload, url=url)
        flags = _pop_flags(page)
        self.last_meta = {**meta, **flags}
        page["_meta"] = self.last_meta
        return page


class TavilyBasic(TavilySearch):
    name = "tavily_basic"
    search_depth = "basic"
    extract_depth = "basic"


class TavilyAdvanced(TavilySearch):
    name = "tavily_advanced"
    search_depth = "advanced"
    extract_depth = "advanced"


class NimbleLite(IntentTool):
    """Nimble Search + standalone Extract.

    The model writes `query` with the description in Nimble's documented function-calling schema `nimble_search`
    (nimble-openai-tools.md:199); the Nimble MCP schema is not public without an authenticated call
    (nimble-mcp-server.md:118-124). The body is the one Nimble's own OpenCode coding-agent plugin sends, `query`,
    `search_depth` and `max_results` (nimble-opencode-plugin-readme.md:71-75): `full_content` and `focus` stay at their
    API defaults, false and general (nimble-api-search.md:61, 110), and are not sent. Fetch: /v2/extract, markdown.
    """

    name = "nimble_lite"
    depth = "lite"
    focus: str | None = None
    # include_domains: "List of domains to include in search results. Maximum 50 domains."
    # (nimble-api-search.md:115-123); no saved Nimble page documents `site:` in the query.
    site_lift: SiteLift = True
    query_description = "The search query to execute"
    last_meta: dict[str, Any] | None = None

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        key = os.environ.get("NIMBLE_API_KEY")
        if not key:
            raise RuntimeError(f"NIMBLE_API_KEY is required for {self.name}")
        sent, include_domains = site_request(self, query)
        body: dict[str, Any] = {"query": sent, "search_depth": self.depth, "max_results": max_results}
        if self.focus:
            body["focus"] = self.focus
        if include_domains:
            body["include_domains"] = include_domains
        payload, meta = vendor_call(self, "POST", _NIMBLE_SEARCH_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json_body=body, timeout=SEARCH_TIMEOUT_S)
        hits = _nimble_search_hits(payload, max_results)
        self.last_meta = _search_meta(meta, hits)
        _record_site_lift(self, sent, include_domains)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        key = os.environ.get("NIMBLE_API_KEY")
        if not key:
            raise RuntimeError(f"NIMBLE_API_KEY is required for {self.name}")
        payload, meta = vendor_call(self, "POST", _NIMBLE_EXTRACT_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json_body={"url": url, "formats": ["markdown"]}, timeout=FETCH_TIMEOUT_S)
        extracted = _nimble_extract_page(payload, url, DEFAULT_MAX_FETCH_CHARS)
        self.last_meta = {**meta, "truncated": extracted["truncated"]}
        return {"url": extracted["final_url"], "title": extracted["title"],
                "content": extracted["text"], "_meta": self.last_meta}


class NimbleStandard(NimbleLite):
    """Nimble's headline: standard is "the default depth when a request omits search_depth"
    (nimble-search-best-practices.md:32), "Rich content (~2K chars) optimized for AI agents"
    (nimble-api-search.md:205)."""

    name = "nimble_standard"
    depth = "standard"


class NimbleLiteCoding(NimbleLite):
    """Settings arm: lite with `focus: "coding"` ("Code examples, debugging, API documentation", nimble-search.md:499;
    "Focus modes only work with search_depth: lite", :508). Billed as lite (nimble-pricing.md:47)."""

    name = "nimble_lite_coding"
    billed_sku = "nimble_lite"
    focus = "coding"


class BraveSearch(IntentTool):
    """Brave LLM Context search-only.

    POST /res/v1/llm/context — the agent/RAG endpoint, not Web Search
    (/res/v1/web/search). Returns pre-extracted snippets per URL so search-only
    runs have grounding text without a fetch hop. fetch() is not implemented.
    """

    name = "brave"
    maximum_number_of_tokens: int | None = None
    # Brave applies `site:` itself (`operators` default true, brave-llm-context.txt:28; `site:` "Returns web pages from
    # only a specific website or domain", api-dashboard.search.brave.com/documentation/resources/search-operators):
    # `site:` stays in the query as written.
    site_lift: SiteLift = False
    # Brave's official MCP server, `brave_llm_context` (github.com/brave/brave-search-mcp-server
    # src/tools/llm_context/index.ts:13 and schemas/input.ts:119-127 at e7da0f19, 9 Nov 2026), whose `query` is sent
    # as `q`. Its limits are stricter than the API's 600 characters and 75 words (brave-llm-context.txt:28).
    query_description = ("The user's search query term. Query can not be empty. Maximum of 400 characters and 50 words "
                         "in the query.")
    last_meta: dict[str, Any] | None = None

    def _key(self) -> str:
        key = os.environ.get("BRAVE_SEARCH_API_KEY")
        if not key:
            raise RuntimeError("BRAVE_SEARCH_API_KEY is required for brave")
        return key

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        n = max(1, min(int(max_results), 50))
        sent, _ = site_request(self, query)
        payload, meta = vendor_call(
            self,
            "POST",
            BRAVE_LLM_CONTEXT_URL,
            headers={
                "X-Subscription-Token": self._key(),
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            # `count` omitted: "leave count ... unset" (brave-llm-context.txt:28). `enable_local`
            # omitted: "For most use cases, you can omit enable_local entirely" (brave-llm-context.txt:28, Local Recall).
            json_body={
                "q": sent,
                "maximum_number_of_urls": n,
                "maximum_number_of_tokens": self.maximum_number_of_tokens or min(8192, max(2048, n * 1024)),
            },
            timeout=SEARCH_TIMEOUT_S,
        )
        hits = parse_brave_hits(payload, max_results=n)
        self.last_meta = _search_meta(meta, hits)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del url, objective
        raise RuntimeError("brave is LLM Context search-only; web_fetch is not wired")


class BraveTokens(BraveSearch):
    """Settings arm: Brave `maximum_number_of_tokens` (1024-32768), the one size control Brave documents: "To fit your
    LLM's context window, set only `maximum_number_of_tokens`" (brave-llm-context.txt:28) and "Use the smallest
    `maximum_number_of_tokens` that meets your needs" (:97). 6,100 is the largest multiple of 100 under which no saved
    Brave search would have passed the 32,000-character preview; at the default 8,192 some did."""

    name = "brave_tokens_6100"
    billed_sku = "brave"
    maximum_number_of_tokens = 6100


def _rapidapi_upstream_refusal(meta: dict[str, Any]) -> int | None:
    """401 or 403 when the RapidAPI gateway's `x-service-info` header lists one: google-search74 can reply HTTP 200
    with `{"c":[401]}` and no results when the key is refused."""
    headers = (meta.get("response") or {}).get("headers") or {}
    raw = next((value for key, value in headers.items() if key.lower() == "x-service-info"), "")
    try:
        info = json.loads(raw) if raw else {}
    except ValueError:
        return None
    codes = info.get("c") if isinstance(info, dict) else None
    return next((code for code in codes if code in (401, 403)), None) if isinstance(codes, list) else None


class SerpSearch:
    """RapidAPI Google Search (google-search74) search-only.

    GET / with `query` (Google operators, `site:` included, serp-rapidapi-google-search74.md §4) and `limit`.
    `related_keywords` stays off ("Enable related_keywords only when you need it", §10). No vendor agent tool is
    published, so the model sees QUERY_PARAMETER. An HTTP 200 whose `x-service-info` lists an upstream 401 or 403 is
    the key refused, not an empty search: `upstream_status` makes it `auth_configuration`. fetch() is not implemented.
    """

    name = "serp"
    # Google operators, `site:` included (serp-rapidapi-google-search74.md §4); no domain filter: `site:` stays as written.
    site_lift: SiteLift = False
    locale: dict[str, str] = {}
    last_meta: dict[str, Any] | None = None

    def _key(self) -> str:
        key = os.environ.get("RAPIDAPI_KEY")
        if not key:
            raise RuntimeError("RAPIDAPI_KEY is required for serp")
        return key

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        n = max(1, min(int(max_results), 10))
        sent, _ = site_request(self, query)
        payload, meta = vendor_call(
            self,
            "GET",
            SERP_SEARCH_URL,
            headers={
                "x-rapidapi-key": self._key(),
                "x-rapidapi-host": SERP_HOST,
                "Accept": "application/json",
            },
            params={"query": sent, "limit": n, **self.locale},
            timeout=SEARCH_TIMEOUT_S,
        )
        refused = _rapidapi_upstream_refusal(meta)
        if refused:
            self.last_meta = {**meta, "upstream_status": refused}
            raise RuntimeError(f"RapidAPI x-service-info reports upstream HTTP {refused} behind HTTP 200")
        hits = parse_serp_hits(payload, max_results=n)
        self.last_meta = _search_meta(meta, hits)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del url, objective
        raise RuntimeError("serp is RapidAPI search-only; web_fetch is not wired")


class SerpEnUs(SerpSearch):
    """Settings arm: `hl=en`, `gl=us`, listed in the playground Params tab ("Language code", "Country code";
    serp-rapidapi-google-search74.md:27); its locale "appears to be decided server-side and is not stable. Test
    explicitly" (§11)."""

    name = "serp_en_us"
    billed_sku = "serp"
    locale = {"hl": "en", "gl": "us"}


def _first_env(*names: str) -> str:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


class YouSearch(IntentTool):
    """You.com Web Search + Contents.

    Search: POST /v1/search with snippets only (no extraction). Fetch: Contents
    API for known URLs — not extraction-on-search, so the two boards differ.
    Dual-split, same pattern as Firecrawl.
    """

    name = "you"
    # You honours `site:` in the query ("Searches for webpages from a particular domain",
    # you.com/docs/guides/search/search-operators.md:21): `site:` stays in the query as written.
    site_lift: SiteLift = False
    # You's official MCP server, `you-search` `query` (you.com/docs/agents/mcp-server.md:329, 462-469; schema text from
    # the hosted server's tools/list as listed at glama.ai/mcp/servers/youdotcom-oss/mcp/tools/you-search, v4.0.6, read
    # 11 Nov 2026), with `lang:` and `loc:` left out: they map to You's language and country parameters, which this
    # board does not send, and You's API operator list has `site:` and `filetype:` (search-operators.md:21-22).
    query_description = ("Short natural-language phrase describing one thing to find, e.g. \"tariff exemptions for "
                         "semiconductor imports 2026\". Inline `site:` filters are mapped automatically. Max 50 words "
                         "/ 400 characters after operator mapping.")
    last_meta: dict[str, Any] | None = None

    def _key(self) -> str:
        key = _first_env("YDC_API_KEY", "YOU_API_KEY", "YOU_KEY")
        if not key:
            raise RuntimeError("YDC_API_KEY is required for you")
        return key

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._key(), "Content-Type": "application/json"}

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        n = max(1, min(int(max_results), 100))
        sent, _ = site_request(self, query)
        payload, meta = vendor_call(
            self,
            "POST",
            YOU_SEARCH_URL,
            headers=self._headers(),
            json_body={"query": sent, "count": n},
            timeout=SEARCH_TIMEOUT_S,
        )
        hits = parse_you_hits(payload, max_results=n)
        self.last_meta = _search_meta(meta, hits)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del objective
        payload, meta = vendor_call(
            self,
            "POST",
            YOU_CONTENTS_URL,
            headers=self._headers(),
            json_body={"urls": [url], "formats": ["markdown"], "crawl_timeout": 20},
            timeout=FETCH_TIMEOUT_S,
        )
        page = parse_you_contents(payload, url=url)
        flags = _pop_flags(page)
        self.last_meta = {**meta, **flags}
        page["_meta"] = self.last_meta
        return page


class YouHighlights(YouSearch):
    """You.com Highlights with separate Contents API fetch.

    POST /v1/search with extraction.extraction_mode=highlights. Query-aware
    passages land in contents.highlights; default snippets are omitted. Fetch
    uses the separate Contents API on search + fetch boards.
    """

    name = "you_highlights"
    knowledge: str | None = None
    # Approximate highlight tokens for the whole response (You docs: 1024-8192, default 8192 when omitted).
    total_max_tokens: int | None = None

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        n = max(1, min(int(max_results), 100))
        sent, include_domains = site_request(self, query)
        highlights = {"total_max_tokens": self.total_max_tokens} if self.total_max_tokens else {}
        payload, meta = vendor_call(
            self,
            "POST",
            YOU_SEARCH_URL,
            headers=self._headers(),
            json_body={
                "query": sent,
                "count": n,
                "extraction": {"extraction_mode": "highlights", **({"highlights": highlights} if highlights else {})},
                **({"knowledge": self.knowledge} if self.knowledge else {}),
                **({"include_domains": include_domains} if include_domains else {}),
            },
            timeout=SEARCH_TIMEOUT_S,
        )
        hits = parse_you_hits(payload, max_results=n)
        self.last_meta = _search_meta(meta, hits)
        _record_site_lift(self, sent, include_domains)
        return hits

class YouHighlightsCore(YouHighlights):
    name = "you_highlights_core"
    knowledge = "core"


class YouHighlightsCoreDocumented(YouHighlightsCore):
    """Settings arm, not a headline lane. You's documented `include_domains`, a strict
    domain allowlist (you.com/docs/api-reference/search/v1-search.md:42), and `extraction.highlights.total_max_tokens` (line 136). 7,000 tokens is about 28,000
    characters, so You's own budget, not the shared 32,000-character one, decides what text arrives."""

    name = "you_highlights_core_documented"
    billed_sku = "you_highlights_core"
    site_lift: SiteLift = True
    total_max_tokens = 7000


class FirecrawlDomains(FirecrawlSearch):
    """Settings arm, not a headline lane. Firecrawl's documented `includeDomains`
    ("hostnames only, without protocol or path", docs.firecrawl.dev/api-reference/endpoint/search.md:73) on its
    default web search."""

    name = "firecrawl_domains"
    billed_sku = "firecrawl"
    site_lift: SiteLift = True


class TinyfishSearch(IntentTool):
    """TinyFish Search + Fetch.

    Search: GET api.search.tinyfish.ai (no count param; slice to max_results). The model sees TinyFish's own search
    tool fields (docs.tinyfish.ai/integrations/fireworks.md, `SEARCH_TOOL`, read 11 Nov 2026): `query` "Search query"
    and `purpose`, required because the tool says "Always provide purpose". `domain_type` stays at its "web" default.
    Fetch: POST api.fetch.tinyfish.ai format=markdown. Dual-split.
    """

    name = "tinyfish"
    # TinyFish calls `site:` in the query "Deprecated for domain filtering" and says to prefer include_domains, a list
    # of domains (tinyfish-search-reference.md:29-33, 51-53).
    site_lift: SiteLift = True
    query_description = "Search query"
    # fireworks.md `purpose` parameter text; the OpenAPI `purpose` description is the same.
    intent_field = "purpose"
    intent_description = ("Why this search is being run — the underlying goal or task the results will be used for. "
                          "Used to better rank results against your intent.")
    last_meta: dict[str, Any] | None = None

    def _key(self) -> str:
        key = _first_env("TINYFISH_API_KEY", "TINYFISH_KEY")
        if not key:
            raise RuntimeError("TINYFISH_API_KEY is required for tinyfish")
        return key

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._key(), "Accept": "application/json"}

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS, purpose: str | None = None
               ) -> list[dict[str, str]]:
        sent, include_domains = site_request(self, query)
        params: dict[str, Any] = {"query": sent, **({"purpose": purpose[:2000]} if purpose else {})}
        if include_domains:
            params["include_domains"] = ",".join(include_domains)
        payload, meta = vendor_call(
            self,
            "GET",
            TINYFISH_SEARCH_URL,
            headers=self._headers(),
            params=params,
            timeout=SEARCH_TIMEOUT_S,
        )
        hits = parse_tinyfish_hits(payload, max_results=max_results)
        self.last_meta = _search_meta(meta, hits)
        _record_site_lift(self, sent, include_domains)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        body: dict[str, Any] = {"urls": [url], "format": "markdown"}
        if objective.strip():
            body["purpose"] = objective.strip()[:2000]
        payload, meta = vendor_call(
            self,
            "POST",
            TINYFISH_FETCH_URL,
            headers={**self._headers(), "Content-Type": "application/json"},
            json_body=body,
            timeout=TINYFISH_FETCH_TIMEOUT_S,
        )
        page = parse_tinyfish_fetch(payload, url=url)
        flags = _pop_flags(page)
        self.last_meta = {**meta, **flags}
        page["_meta"] = self.last_meta
        return page


class TinyfishNoPurpose(TinyfishSearch):
    """Settings arm: TinyFish's documented API default, `purpose` omitted: "An optional short statement of why you are
    searching ... When omitted, search behaves exactly as before" (tinyfish-search-reference.md:35-40). The headline
    sends the agent-written `purpose`, as TinyFish's own tool tells the agent to ("Always provide purpose")."""

    name = "tinyfish_no_purpose"
    billed_sku = "tinyfish"
    intent_field = None
    intent_description = ""


class TakoSearch(IntentTool):
    """Tako Search, web source only, search-only.

    The body is the one Tako's own `tako_search` tool builds (tako-mcp-tako-search-ts.md:154-194): `sources.web` with
    `highlights` true, and `effort` omitted, which is fast, the documented default (tako-search-for-coding-agent.md:76).
    Left out under content_mode: the data source (Tako datasets and variable-cost cards,
    tako-search-v3-reference.md:913-916), `include_contents` (full page text, :975-977) and the Answer endpoint.
    No fetch.
    """

    name = "tako"
    highlights = True
    snippet_max_chars: int | None = None
    # include_domains: "Send a bare host" (tako-search-v3-reference.md:993-996); Tako's docs never mention `site:`.
    site_lift: SiteLift = True
    # tako-mcp-tako-search-ts.md:92-93 QUERY_DESCRIBE; TRAFFIC_QUERY_SENTENCE (:94-95) is about traffic data cards this
    # lane does not request.
    query_description = ("Natural-language search query (e.g. \"US GDP growth\", \"Intel vs Nvidia revenue\"). "
                         "Double-quote a multi-word name to keep it one entity (\"tesla motors\" club revenue); an "
                         "unpaired quote disables quoting.")
    last_meta: dict[str, Any] | None = None

    def _key(self) -> str:
        key = _first_env("TAKO_API_KEY")
        if not key:
            raise RuntimeError(f"TAKO_API_KEY is required for {self.name}")
        return key

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        n = max(1, min(int(max_results), 20))
        sent, include_domains = site_request(self, query)
        web: dict[str, Any] = {"count": n}
        if self.highlights:
            web["highlights"] = True
        if self.snippet_max_chars:
            web["snippet_max_chars"] = self.snippet_max_chars
        if include_domains:
            web["include_domains"] = include_domains
        payload, meta = vendor_call(
            self,
            "POST",
            TAKO_SEARCH_URL,
            headers={"X-API-Key": self._key(), "Content-Type": "application/json"},
            json_body={"query": sent, "sources": {"web": web}},
            timeout=SEARCH_TIMEOUT_S,
        )
        hits = parse_tako_hits(payload, max_results=n)
        self.last_meta = _search_meta(meta, hits)
        _record_site_lift(self, sent, include_domains)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del url, objective
        raise RuntimeError(f"{self.name} is Tako search-only; web_fetch is not wired")


class TakoPlain(TakoSearch):
    """Settings arm: the API default snippet, the page's opening text (`highlights` default false,
    tako-search-v3-reference.md:1020-1030)."""

    name = "tako_plain"
    billed_sku = "tako"
    highlights = False


class TakoSnippet3400(TakoSearch):
    """Settings arm: `snippet_max_chars` 3400, which Tako "uses exactly" (tako-search-v3-reference.md:1014-1017): 8 hits
    fit the 32,000-character preview after the largest v1 url + title total."""

    name = "tako_snippet_3400"
    billed_sku = "tako"
    snippet_max_chars = 3400


class PerplexitySearch:
    """Perplexity Search API `POST /search`, search-only: Perplexity has no page-open API.

    No `search_context_size`: it is not a Search API field (perplexity-api-search-post.md:107-169). `search_type`
    omitted is web, the default (:124-135). One request is billed once whatever the number of queries in it
    (perplexity-search-quickstart.md:281) and is one search in the budget.
    """

    billed_sku = "perplexity_web"
    search_type: str | None = None
    budget: dict[str, int] = {}
    # search_domain_filter takes root domains, max 20 (perplexity-domain-filter.md:13, 80-110); the docs document no
    # `site:` operator.
    site_lift: SiteLift = True
    last_meta: dict[str, Any] | None = None

    def _post(self, query: str | list[str], include_domains: list[str], max_results: int) -> list[dict[str, str]]:
        key = _first_env("PERPLEXITY_API_KEY", "PERPLEXITY_API")
        if not key:
            raise RuntimeError(f"PERPLEXITY_API_KEY is required for {self.name}")
        body: dict[str, Any] = {"query": query, "max_results": max_results, **self.budget}
        if self.search_type:
            body["search_type"] = self.search_type
        if include_domains:
            body["search_domain_filter"] = include_domains
        payload, meta = vendor_call(
            self,
            "POST",
            PERPLEXITY_SEARCH_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json_body=body,
            timeout=SEARCH_TIMEOUT_S,
        )
        hits = parse_perplexity_hits(payload, max_results=max_results)
        self.last_meta = _search_meta(meta, hits)
        _record_site_lift(self, query if isinstance(query, str) else " | ".join(query), include_domains)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del url, objective
        raise RuntimeError(f"{self.name} is Perplexity search-only; web_fetch is not wired")


class PerplexityWebTool(PerplexitySearch):
    """Perplexity's headline: the Search API as a function tool for an agent on another provider
    (perplexity-search-overview.md:224-231). The model writes `queries`, an array of strings, under Perplexity's own
    parameter wording (perplexity-agent-sdk-openai.md:58-62); entries after the third are dropped, as Perplexity's
    handler does (`queries[:3]`, :146-147), and the array is sent as `query`. `search_domain_filter` is request-wide, so
    `site:` hosts are lifted only when every query names the same hosts (`site_request_all`). Each query in the array is
    one of the searches in the budget (`queries_field`); the request is billed once."""

    name = "perplexity_web_tool"
    queries_field = "queries"
    tool_parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "description": ("An array of keyword-based search queries. Each query should be short, as longer "
                                "queries may reduce performance. Do not provide more than three queries to maintain "
                                "efficiency."),
                "items": {"type": "string"},
            },
        },
        "required": ["queries"],
        "additionalProperties": False,
    }

    def tool_request(self, args: dict[str, Any]) -> dict[str, Any] | None:
        queries = args.get("queries")
        if (not isinstance(queries, list) or not queries
                or any(not isinstance(query, str) or not query.strip() for query in queries)):
            return None
        return {"queries": [query.strip() for query in queries[:3]]}

    def search_tool(self, request: dict[str, Any], *, max_results: int, client_model: str) -> list[dict[str, str]]:
        del client_model
        sent, include_domains = site_request_all(self, request["queries"])
        return self._post(sent, include_domains, max_results)

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        return self.search_tool({"queries": [query]}, max_results=max_results, client_model="")


class PerplexityFastTool(PerplexityWebTool):
    """Settings arm: `search_type: "fast"`, for "Day-to-day agentic work: tool calls inside agent loops"
    (perplexity-fast-search.md:21-24)."""

    name = "perplexity_fast_tool"
    billed_sku = "perplexity_fast"
    search_type = "fast"


class PerplexityWebToolBudget(PerplexityWebTool):
    """Settings arm: "Cap the text you hand the model with max_tokens_per_page ... and max_tokens"
    (perplexity-search-overview.md:124); 1024 per page is the MCP server's default
    (perplexity-mcp-search-tool.md:31)."""

    name = "perplexity_web_tool_budget"
    budget = {"max_tokens": 8000, "max_tokens_per_page": 1024}


class PerplexityWebToolSiteq(PerplexityWebTool):
    """Comparison arm: every query goes out with its `site:` unchanged and no domain filter."""

    name = "perplexity_web_tool_siteq"
    site_lift: SiteLift = None


class PerplexityWebMcp(IntentTool, PerplexitySearch):
    """Settings arm: Perplexity's MCP `perplexity_search` tool, one `query` "Search query string"
    (perplexity-mcp-search-tool.md:9) and the handler's `max_tokens_per_page` default 1024 (:31); its other fields are
    filters and options the board fixes."""

    name = "perplexity_web_mcp"
    budget = {"max_tokens_per_page": 1024}
    query_description = "Search query string"

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        sent, include_domains = site_request(self, query)
        return self._post(sent, include_domains, max_results)


class KeenableSearch(IntentTool):
    """Keenable Search, keyed `POST /v1/search`, search-only.

    The model writes `query` in the wording of Keenable's SDK SEARCH_TOOL (keenable 0.1.3, keenable-sdk-tools.md:44);
    the tool's optional `site` and `published_after` (:215-219) are filters the board fixes, so they are left out.
    `mode` is sent, since an organization can have another default (keenable-credits.md:41), and the reply's `mode`,
    "The mode that served the request" (keenable-api-reference-search.md:221-223), must equal it; otherwise the call is
    `served_mode_mismatch`. Cards, results Keenable writes itself, are off (`disable_cards`, :205-207). Every call of
    one attempt carries one `X-Keenable-Session-Id` (:209-211; one backend object per attempt).
    """

    name = "keenable_pro"
    mode = "pro"
    snippet_max_length: int | None = None
    # `site:` in the query keeps pages on that site (keenable-api-reference-search.md:514-522); Keenable has no domain
    # filter, so `site:` stays in the query as written.
    site_lift: SiteLift = False
    query_description = "What to look for, described in natural language rather than keywords."
    last_meta: dict[str, Any] | None = None

    def __init__(self) -> None:
        self.session_id = str(uuid.uuid4())

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        key = os.environ.get("KEENABLE_API_KEY")
        if not key:
            raise RuntimeError(f"KEENABLE_API_KEY is required for {self.name}")
        sent, _ = site_request(self, query)
        body: dict[str, Any] = {"query": sent, "mode": self.mode, "max_results": max_results, "disable_cards": True}
        if self.snippet_max_length:
            body["snippet_max_length"] = self.snippet_max_length
        payload, meta = vendor_call(
            self,
            "POST",
            KEENABLE_SEARCH_URL,
            headers={"X-API-Key": key, "Content-Type": "application/json", "X-Keenable-Title": "OpenBenchmarks",
                     "X-Keenable-Session-Id": self.session_id},
            json_body=body,
            timeout=SEARCH_TIMEOUT_S,
        )
        data = payload if isinstance(payload, dict) else {}
        meta = {**meta, "served_mode": data.get("mode"), **{k: data[k] for k in ("usage", "warning") if k in data}}
        if data.get("mode") != self.mode:
            self.last_meta = {**meta, "served_mode_mismatch": True}
            raise RuntimeError(f"{self.name} sent mode {self.mode!r}; Keenable served {data.get('mode')!r}")
        hits = parse_keenable_hits(payload, max_results=max_results)
        self.last_meta = _search_meta(meta, hits)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del url, objective
        raise RuntimeError(f"{self.name} is Keenable search-only; web_fetch is not wired")


class KeenableRealtime(KeenableSearch):
    """Settings arm: the other documented mode, "The fastest response" (keenable-best-practices.md:164), billed as
    `search.realtime` (keenable-api-reference-search.md:157)."""

    name = "keenable_realtime"
    mode = "realtime"


class KeenableProSnippet4000(KeenableSearch):
    """Settings arm: `snippet_max_length` 4000 (range 180-10000, keenable-api-reference-search.md:201-203): 8 x 4,000
    is the 32,000-character preview."""

    name = "keenable_pro_snippet4000"
    billed_sku = "keenable_pro"
    snippet_max_length = 4000


def parse_parallel_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in (payload or {}).get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        url = str(item["url"])
        if url in seen:
            continue
        seen.add(url)
        excerpts = item.get("excerpts") or []
        snippet = "\n".join(part for part in excerpts if isinstance(part, str))
        hits.append({
            "url": url,
            "title": str(item.get("title") or ""),
            "snippet": snippet,
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_firecrawl_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        rows = data.get("web") or data.get("results") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("link")
        if not url:
            continue
        url = str(url)
        if url in seen:
            continue
        seen.add(url)
        snippet = item.get("description") or item.get("snippet") or item.get("markdown") or ""
        hits.append({
            "url": url,
            "title": str(item.get("title") or ""),
            "snippet": str(snippet),
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_parallel_extract(
    payload: Any,
    *,
    url: str,
    max_chars: int = DEFAULT_MAX_FETCH_CHARS,
) -> dict[str, str]:
    errors = (payload or {}).get("errors") or []
    results = (payload or {}).get("results") or []
    if errors and not results:
        first = errors[0] if isinstance(errors[0], dict) else {}
        detail = first.get("content") or first.get("error_type") or "extract failed"
        raise RuntimeError(f"{url}: {detail}")
    item = results[0] if results and isinstance(results[0], dict) else {}
    excerpts = item.get("excerpts") or []
    excerpt_text = "\n\n".join(part for part in excerpts if isinstance(part, str))
    raw = str(item.get("full_content") or excerpt_text or "")
    return {
        "url": str(item.get("url") or url),
        "title": str(item.get("title") or ""),
        "content": raw[:max_chars],
        "_truncated": len(raw) > max_chars,
        "_extract_id": str((payload or {}).get("extract_id") or ""),
        "_session_id": str((payload or {}).get("session_id") or ""),
        "_had_full_content": bool(item.get("full_content")),
        "_n_excerpts": len(excerpts),
    }


def parse_firecrawl_scrape(
    payload: Any,
    *,
    url: str,
    max_chars: int = DEFAULT_MAX_FETCH_CHARS,
) -> dict[str, str]:
    if isinstance(payload, dict) and payload.get("success") is False:
        raise RuntimeError(str(payload.get("error") or "scrape failed"))
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, dict):
        data = payload if isinstance(payload, dict) else {}
    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    content = str(data.get("markdown") or data.get("content") or "")
    if not content:
        raise RuntimeError(f"{url}: scrape returned no markdown")
    return {
        "url": str(meta.get("sourceURL") or meta.get("url") or data.get("url") or url),
        "title": str(meta.get("title") or data.get("title") or ""),
        "content": content[:max_chars],
        "_truncated": len(content) > max_chars,
        "_status_code": meta.get("statusCode") or meta.get("status_code"),
    }


def join_preview(parts: list[str]) -> str:
    """Every text field a result returned, in API order; empty and repeated text dropped."""
    kept: list[str] = []
    for part in parts:
        text = part.strip()
        if text and not any(text in prev for prev in kept):
            kept.append(text)
    return "\n".join(kept)


def hit_text_stats(hits: list[dict[str, str]]) -> dict[str, int]:
    """Chars and ~tokens of title+url+snippet actually fed back to the model."""
    chars = 0
    for hit in hits:
        chars += len(str(hit.get("url") or ""))
        chars += len(str(hit.get("title") or ""))
        chars += len(str(hit.get("snippet") or ""))
    return {
        "n_hits": len(hits),
        "chars": chars,
        "tokens_est": chars // 4,
    }


def fit_search_budget(hits: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Hits as the agent sees them under SEARCH_PREVIEW_BUDGET_CHARS, the same for every vendor.

    Every hit's url + title is reserved first, so no returned result disappears; the rest of the budget fills
    previews in API order, and the hit that overflows keeps its preview cut there. Hits whose url + title alone
    exceed what is left are dropped from the end. Returns the shown hits and the number of characters cut.
    """
    kept_hits: list[dict[str, str]] = []
    left = SEARCH_PREVIEW_BUDGET_CHARS
    for hit in hits:
        size = len(str(hit.get("url") or "")) + len(str(hit.get("title") or ""))
        if size > left:
            break
        left -= size
        kept_hits.append(hit)
    shown: list[dict[str, str]] = []
    for hit in kept_hits:
        snippet = str(hit.get("snippet") or "")[:left]
        left -= len(snippet)
        shown.append({**hit, "snippet": snippet})
    return shown, hit_text_stats(hits)["chars"] - hit_text_stats(shown)["chars"]


def page_text_stats(page: dict[str, str]) -> dict[str, int]:
    return hit_text_stats([{
        "url": str(page.get("url") or ""),
        "title": str(page.get("title") or ""),
        "snippet": str(page.get("content") or ""),
    }])


def parse_exa_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in (payload or {}).get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        url = str(item["url"])
        if url in seen:
            continue
        seen.add(url)
        highlights = item.get("highlights") or []
        if isinstance(highlights, list) and highlights:
            snippet = "\n".join(part for part in highlights if isinstance(part, str))
        else:
            snippet = str(item.get("summary") or item.get("text") or "")
        hits.append({
            "url": url,
            "title": str(item.get("title") or ""),
            "snippet": snippet,
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_exa_contents(
    payload: Any,
    *,
    url: str,
    max_chars: int = DEFAULT_MAX_FETCH_CHARS,
) -> dict[str, str]:
    results = (payload or {}).get("results") or []
    item = results[0] if results and isinstance(results[0], dict) else {}
    raw = item.get("text") or item.get("excerpt") or ""
    if isinstance(raw, dict):
        raw = raw.get("text") or ""
    content = str(raw)
    if not content:
        raise RuntimeError(f"{url}: contents returned no text")
    return {
        "url": str(item.get("url") or url),
        "title": str(item.get("title") or ""),
        "content": content[:max_chars],
        "_truncated": len(content) > max_chars,
    }


def parse_linkup_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in (payload or {}).get("results") or []:
        if not isinstance(item, dict) or item.get("type") == "image" or not item.get("url"):
            continue
        url = str(item["url"])
        if url in seen:
            continue
        seen.add(url)
        hits.append({
            "url": url,
            "title": str(item.get("name") or item.get("title") or ""),
            "snippet": str(item.get("content") or item.get("snippet") or ""),
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_linkup_fetch(
    payload: Any,
    *,
    url: str,
    max_chars: int = DEFAULT_MAX_FETCH_CHARS,
) -> dict[str, str]:
    data = payload if isinstance(payload, dict) else {}
    content = str(data.get("markdown") or data.get("content") or "")
    if not content:
        raise RuntimeError(f"{url}: fetch returned no markdown")
    return {
        "url": url,
        "title": str(data.get("title") or ""),
        "content": content[:max_chars],
        "_truncated": len(content) > max_chars,
    }


def parse_tavily_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in (payload or {}).get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        url = str(item["url"])
        if url in seen:
            continue
        seen.add(url)
        hits.append({
            "url": url,
            "title": str(item.get("title") or ""),
            "snippet": str(item.get("content") or ""),
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_brave_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    """Map LLM Context grounding.generic[] snippets plus sources[url].snippet to url/title/snippet."""
    grounding = payload.get("grounding") if isinstance(payload, dict) else None
    rows = grounding.get("generic") if isinstance(grounding, dict) else None
    if not isinstance(rows, list):
        rows = []
    sources = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(sources, dict):
        sources = {}
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        url = str(item["url"])
        if url in seen:
            continue
        seen.add(url)
        chunks = item.get("snippets") or []
        parts = [str(part) for part in chunks if part] if isinstance(chunks, list) else [str(chunks)]
        source = sources.get(url)
        if isinstance(source, dict):
            parts.append(str(source.get("snippet") or ""))
        hits.append({
            "url": url,
            "title": str(item.get("title") or ""),
            "snippet": join_preview(parts),
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_serp_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    """Map google-search74 organic results to url/title/snippet."""
    rows = (payload or {}).get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        rows = []
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("link")
        if not url:
            continue
        url = str(url)
        if url in seen:
            continue
        seen.add(url)
        snippet = item.get("description") or item.get("snippet") or item.get("content") or ""
        hits.append({
            "url": url,
            "title": str(item.get("title") or ""),
            "snippet": str(snippet),
        })
        if len(hits) >= max_results:
            break
    return hits


def _you_hit_text(item: dict[str, Any]) -> str:
    contents = item.get("contents") if isinstance(item.get("contents"), dict) else {}
    highlights = (contents or {}).get("highlights")
    if not isinstance(highlights, list):
        highlights = item.get("highlights")
    parts: list[str] = []
    if isinstance(highlights, list):
        for part in highlights:
            if isinstance(part, str) and part.strip():
                parts.append(part.strip())
            elif isinstance(part, dict):
                text = part.get("text") or part.get("snippet") or part.get("content") or ""
                if str(text).strip():
                    parts.append(str(text).strip())
    chunks = item.get("snippets") or []
    if isinstance(chunks, list):
        parts.extend(str(part) for part in chunks if part)
    parts.append(str(item.get("description") or item.get("snippet") or ""))
    return join_preview(parts)


def parse_you_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    data = payload.get("results") if isinstance(payload, dict) else payload
    rows: list[Any] = []
    if isinstance(data, dict):
        rows.extend(data.get("web") or [])
        rows.extend(data.get("news") or [])
    elif isinstance(data, list):
        rows = data
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not url:
            continue
        url = str(url)
        if url in seen:
            continue
        seen.add(url)
        snippet = _you_hit_text(item)
        hits.append({
            "url": url,
            "title": str(item.get("title") or ""),
            "snippet": snippet,
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_you_contents(
    payload: Any,
    *,
    url: str,
    max_chars: int = DEFAULT_MAX_FETCH_CHARS,
) -> dict[str, str]:
    rows = payload if isinstance(payload, list) else None
    if rows is None and isinstance(payload, dict):
        rows = payload.get("results") or payload.get("data") or []
        if isinstance(rows, dict):
            rows = [rows]
    if not isinstance(rows, list):
        rows = []
    item = {}
    for row in rows:
        if isinstance(row, dict):
            item = row
            if str(row.get("url") or "") == url:
                break
    content = str(item.get("markdown") or item.get("html") or item.get("content") or "")
    if not content:
        raise RuntimeError(f"{url}: contents returned no markdown")
    return {
        "url": str(item.get("url") or url),
        "title": str(item.get("title") or ""),
        "content": content[:max_chars],
        "_truncated": len(content) > max_chars,
    }


def parse_tinyfish_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    rows = (payload or {}).get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        rows = []
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        url = str(item["url"])
        if url in seen:
            continue
        seen.add(url)
        hits.append({
            "url": url,
            "title": str(item.get("title") or ""),
            "snippet": str(item.get("snippet") or item.get("description") or ""),
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_tinyfish_fetch(
    payload: Any,
    *,
    url: str,
    max_chars: int = DEFAULT_MAX_FETCH_CHARS,
) -> dict[str, str]:
    data = payload if isinstance(payload, dict) else {}
    errors = data.get("errors") or []
    results = data.get("results") or []
    if errors and not results:
        first = errors[0] if isinstance(errors[0], dict) else {}
        detail = first.get("error") or first.get("message") or "fetch failed"
        raise RuntimeError(f"{url}: {detail}")
    item = results[0] if results and isinstance(results[0], dict) else {}
    text = item.get("text")
    if isinstance(text, dict):
        content = json.dumps(text)
    else:
        content = str(text or item.get("content") or item.get("markdown") or "")
    if not content:
        raise RuntimeError(f"{url}: fetch returned no text")
    return {
        "url": str(item.get("final_url") or item.get("url") or url),
        "title": str(item.get("title") or ""),
        "content": content[:max_chars],
        "_truncated": len(content) > max_chars,
    }


def parse_tako_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    data = payload if isinstance(payload, dict) else {}
    rows = data.get("web_results") or data.get("results") or []
    if not isinstance(rows, list):
        rows = []
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("webpage_url")
        if not url:
            continue
        url = str(url)
        if url in seen:
            continue
        seen.add(url)
        hits.append({
            "url": url,
            "title": str(item.get("title") or ""),
            "snippet": str(item.get("snippet") or item.get("summary") or ""),
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_perplexity_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    """The hits shown from one Perplexity /search reply, one or several queries.

    Merge rule, not yet checked against a live call: the docs say neither whether `max_results` counts per query or
    per request nor how a multi-query reply orders its pages (the schema lists `results` as pages,
    perplexity-api-search-post.md; Perplexity's own handler reads them as one list, perplexity-agent-sdk-openai.md).
    A flat `results` is kept in API order. A `results` of per-query lists is interleaved by rank: every query's first
    page, then every query's second, and so on. Either way repeated URLs are dropped and the first `max_results` shown.
    """
    rows = (payload or {}).get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        rows = []
    groups = rows if rows and all(isinstance(row, list) for row in rows) else [rows]
    ordered = [group[rank] for rank in range(max(len(group) for group in groups)) for group in groups
               if rank < len(group)]
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in ordered:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        url = str(item["url"])
        if url in seen:
            continue
        seen.add(url)
        hits.append({
            "url": url,
            "title": str(item.get("title") or ""),
            "snippet": str(item.get("snippet") or item.get("content") or ""),
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_keenable_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    """results[].url / title / snippet; `description` is "Always an empty string"
    (keenable-api-reference-search.md:251-253)."""
    rows = (payload or {}).get("results") if isinstance(payload, dict) else None
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in rows if isinstance(rows, list) else []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        url = str(item["url"])
        if url in seen:
            continue
        seen.add(url)
        hits.append({"url": url, "title": str(item.get("title") or ""), "snippet": str(item.get("snippet") or "")})
        if len(hits) >= max_results:
            break
    return hits


def parse_tavily_extract(
    payload: Any,
    *,
    url: str,
    max_chars: int = DEFAULT_MAX_FETCH_CHARS,
) -> dict[str, str]:
    failed = (payload or {}).get("failed_results") or []
    results = (payload or {}).get("results") or []
    if failed and not results:
        first = failed[0] if isinstance(failed[0], dict) else {}
        detail = first.get("error") or "extract failed"
        raise RuntimeError(f"{url}: {detail}")
    item = results[0] if results and isinstance(results[0], dict) else {}
    content = str(item.get("raw_content") or item.get("content") or "")
    if not content:
        raise RuntimeError(f"{url}: extract returned no content")
    usage = (payload or {}).get("usage") if isinstance((payload or {}).get("usage"), dict) else {}
    return {
        "url": str(item.get("url") or url),
        "title": str(item.get("title") or ""),
        "content": content[:max_chars],
        "_truncated": len(content) > max_chars,
        "_credits": usage.get("credits"),
        "_request_id": str((payload or {}).get("request_id") or ""),
    }


def vendor_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    text_body: bool = False,
) -> tuple[Any, dict[str, Any]]:
    request_audit = {
        "method": method,
        "url": url,
        "headers": redact_headers(headers),
        "params": redact_payload(params),
        "json": redact_payload(json_body),
    }
    record = begin_request("vendor", request_audit)
    started = time.perf_counter()
    try:
        response = requests.request(
            method,
            url,
            headers=headers,
            json=json_body,
            params=params,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        end_request(record, {"state": "unknown", "reason": "response unavailable"})
        meta = {
            "endpoint": url,
            "method": method,
            "http_status": None,
            "http_ms": int((time.perf_counter() - started) * 1000),
            "response_bytes": 0,
            "vendor_request_id": "",
            "error": f"{type(exc).__name__}: {exc}",
            "request": request_audit,
            "response": None,
        }
        error = RuntimeError(f"{type(exc).__name__}: {exc}")
        setattr(error, "vendor_meta", meta)
        raise error from exc
    decode_error = False
    if text_body:
        payload: Any = response.text or ""
    else:
        try:
            payload = response.json() if response.content else {}
        except ValueError:
            decode_error = True
            payload = {"raw": response.text or ""}
    meta: dict[str, Any] = {
        "endpoint": url,
        "method": method,
        "http_status": response.status_code,
        "http_ms": int((time.perf_counter() - started) * 1000),
        "response_bytes": len(response.content or b""),
        "vendor_request_id": _header_id(response),
        "request": request_audit,
        "response": {
            "http_status": response.status_code,
            "headers": redact_headers(dict(response.headers)),
            "body": payload,
        },
    }
    end_request(record, {"state": "response", **meta["response"]})
    if decode_error and response.ok:
        error = ValueError("vendor returned a non-JSON response")
        setattr(error, "vendor_meta", meta)
        raise error
    if isinstance(payload, dict):
        for key in ("extract_id", "search_id", "id", "session_id", "request_id", "requestId"):
            if payload.get(key):
                meta[key] = str(payload[key])
        if payload.get("requestId"):
            meta["vendor_request_id"] = str(payload["requestId"])
        if url == EXA_SEARCH_URL and isinstance(payload.get("notices"), list):
            meta["notices"] = payload["notices"]
    if not response.ok:
        meta["error_body"] = str(payload)[:800]
        error = requests.HTTPError(f"{response.status_code} for {url}", response=response)
        setattr(error, "vendor_meta", meta)
        raise error
    return payload, meta


def redact_headers(headers: dict[str, str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in (headers or {}).items():
        lowered = str(key).lower()
        if any(token in lowered for token in HEADER_REDACT_TOKENS):
            out[str(key)] = "***REDACTED***"
        else:
            out[str(key)] = str(value)
    return out


def redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key).lower().replace("-", "_")
            credential = name in {"authorization", "api_key", "apikey", "token", "secret", "password", "cookie"}
            if credential or name.endswith(("_api_key", "_token", "_secret", "_password")):
                out[str(key)] = "***REDACTED***"
            else:
                out[str(key)] = redact_payload(item)
        return out
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    return value


def classify_failure(meta: dict[str, Any], error: str | None) -> str | None:
    """Classify the observed boundary, without guessing who caused a timeout."""
    if not error:
        return None
    if not meta.get("request"):
        return "local_error"
    status = meta.get("http_status")
    if status is None:
        return "transport_error"
    if meta.get("served_mode_mismatch"):
        return "served_mode_mismatch"
    if meta.get("upstream_status") in (401, 403):
        return "auth_configuration"
    if 300 <= status < 400:
        return "redirect_not_followed"
    # 402 Payment Required: our account ran out (TinyFish: wallet at zero after the free daily allowance).
    if status == 402:
        return "account_billing"
    if status in (432, 433) and meta.get("endpoint") == TAVILY_SEARCH_URL:
        return "account_billing"
    # TinyFish documents 403 as an upstream Search denial, not a bad API key.
    if status == 403 and meta.get("endpoint", "").rstrip("/") == TINYFISH_SEARCH_URL:
        return "provider_http"
    # Keenable: "If `query` was a URL, the target site may have blocked the fetch"
    # (keenable-api-reference-search.md:296), code target_blocked, a target_error (keenable-errors.md, 403 section);
    # other 403s are the key.
    if (status == 403 and meta.get("endpoint") == KEENABLE_SEARCH_URL
            and "target_blocked" in str(meta.get("error_body") or "")):
        return "provider_http"
    if status in (401, 403):
        return "auth_configuration"
    if status == 429:
        # Linkup documents 429 for "Rate limit exceeded or insufficient credits": out of credit is our account.
        if meta.get("endpoint") == LINKUP_SEARCH_URL and "credit" in str(meta.get("error_body") or "").lower():
            return "account_billing"
        # Tako: "With error_type BUDGET_EXCEEDED, the API key or its owner has used a spend budget: do not retry
        # automatically" (tako-search-v3-reference.md:93-99).
        if meta.get("endpoint") == TAKO_SEARCH_URL and "BUDGET_EXCEEDED" in str(meta.get("error_body") or ""):
            return "account_billing"
        return "rate_limit"
    # A vendor's own failure to answer, the same rule as the model service's `model_server`: HTTP 5xx, or 408 (Tako:
    # "The request timed out. Retry", tako-search-for-coding-agent.md:248; TinyFish 503: "retry with backoff",
    # tinyfish-search-reference.md Error Codes).
    if status >= 500 or status == 408:
        return "vendor_server"
    if status >= 400:
        return "request_rejected"
    return "response_processing"


def vendor_call(backend: Any, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """vendor_request that leaves this call's meta on backend.last_meta, also when it fails."""
    try:
        payload, meta = vendor_request(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        backend.last_meta = dict(getattr(exc, "vendor_meta", None) or {})
        raise
    backend.last_meta = meta
    return payload, meta


def _header_id(response: Any) -> str:
    headers = getattr(response, "headers", None) or {}
    for key in ("x-request-id", "x-parallel-request-id", "cf-ray", "fly-request-id"):
        value = headers.get(key)
        if value:
            return str(value)
    return ""


def _search_meta(meta: dict[str, Any], hits: list[dict[str, str]]) -> dict[str, Any]:
    urls = [str(hit.get("url") or "") for hit in hits if hit.get("url")]
    return {
        **meta,
        "n_hits": len(hits),
        "hit_urls": urls,
        "empty": len(hits) == 0,
    }


def _pop_flags(page: dict[str, Any]) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    for key in list(page):
        if key.startswith("_") and key != "_meta":
            flags[key[1:]] = page.pop(key)
    return flags


class StringSearch:
    """String Web Access search + readable markdown fetch; dual-split."""

    name = "string"
    engine = "google"
    last_meta: dict[str, Any] | None = None

    def _headers(self) -> dict[str, str]:
        key = os.environ.get("STRING_API_KEY")
        if not key:
            raise RuntimeError("STRING_API_KEY is required for string")
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def search(self, query: str, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
        payload, meta = vendor_call(
            self,
            "POST",
            STRING_SEARCH_URL,
            headers=self._headers(),
            json_body={"query": query, "engine": self.engine},
            timeout=SEARCH_TIMEOUT_S,
        )
        hits = parse_string_hits(payload, max_results=max_results)
        self.last_meta = _search_meta(meta, hits)
        return hits

    def fetch(self, url: str, *, objective: str = "") -> dict[str, str]:
        del objective
        payload, meta = vendor_call(
            self,
            "POST",
            STRING_FETCH_URL,
            headers=self._headers(),
            json_body={"url": url, "format": "markdown", "markdownMode": "readable"},
            timeout=FETCH_TIMEOUT_S,
            text_body=True,
        )
        headers = (meta.get("response") or {}).get("headers") or {}
        status = next((value for key, value in headers.items() if key.lower() == "x-status-code"), None)
        page = parse_string_fetch(payload, url=url, status_code=int(status) if status else None)
        self.last_meta = {**meta, **_pop_flags(page)}
        page["_meta"] = self.last_meta
        return page


def parse_string_hits(payload: Any, *, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]]:
    rows = (payload or {}).get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        rows = []
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        url = str(item["url"])
        if url in seen:
            continue
        seen.add(url)
        hits.append({
            "url": url,
            "title": str(item.get("title") or ""),
            "snippet": str(item.get("snippet") or "")[:1200],
        })
        if len(hits) >= max_results:
            break
    return hits


def parse_string_fetch(
    payload: str, *, url: str, status_code: int | None, max_chars: int = DEFAULT_MAX_FETCH_CHARS,
) -> dict[str, Any]:
    """String's `format: markdown` fetch answers with a text/markdown body; the page's own status is `x-status-code`."""
    if not payload:
        raise RuntimeError(f"{url}: fetch returned no markdown")
    return {
        "url": url,
        "title": "",
        "content": payload[:max_chars],
        "_truncated": len(payload) > max_chars,
        "_status_code": status_code,
    }


# Canonical ids. Aliases below keep old CLI names working.
SEARCH_ONLY_BACKENDS: tuple[str, ...] = (
    "parallel_fast_tool",
    "parallel_advanced_tool",
    "exa_auto",
    "exa_auto_domains",
    "tavily_advanced",
    "tavily_basic",
    "linkup_standard",
    "linkup_fast",
    "tinyfish",
    "tinyfish_no_purpose",
    "brave",
    "brave_tokens_6100",
    "you_highlights_core",
    "you_highlights",
    "firecrawl",
    "firecrawl_domains",
    "keenable_pro",
    "keenable_realtime",
    "nimble_standard",
    "nimble_lite_coding",
    "perplexity_web_tool",
    "perplexity_fast_tool",
    "tako",
    "tako_plain",
    "serp",
    "serp_en_us",
    "string",
)
SEARCH_FETCH_BACKENDS: tuple[str, ...] = (
    "nimble_standard",
    "exa_auto",
    "tavily_basic",
    "tavily_advanced",
    "linkup_standard",
    "firecrawl",
    "you_highlights",
    "you_highlights_core",
    "tinyfish",
    "string",
)
BACKEND_ALIASES = {
    "exa": "exa_auto",
    "tavily": "tavily_advanced",
    "linkup": "linkup_standard",
}
BACKEND_ALIAS_SPLIT = {alias: "search-only" for alias in BACKEND_ALIASES}
# Search-only exclusive rows cannot turn fetch on. Search+fetch exclusive rows
# cannot turn fetch off. Dual-split vendors sit on both boards.
DUAL_SPLIT_BACKENDS = frozenset({"nimble_standard", "tavily_basic", "tavily_advanced", "exa_auto", "firecrawl", "you_highlights", "you_highlights_core", "tinyfish", "linkup_standard", "string"})
FETCH_FORBIDDEN = frozenset(SEARCH_ONLY_BACKENDS) - DUAL_SPLIT_BACKENDS
FETCH_REQUIRED = frozenset(SEARCH_FETCH_BACKENDS) - DUAL_SPLIT_BACKENDS

BACKENDS: dict[str, Callable[[], SearchBackend]] = {
    "nimble_lite": NimbleLite,
    "nimble_standard": NimbleStandard,
    "nimble_lite_coding": NimbleLiteCoding,
    "no_search": NoSearch,
    "parallel_fast": ParallelFast,
    "parallel_fast_raw": ParallelFastRaw,
    "parallel_fast_tool": ParallelFastTool,
    "parallel_advanced_tool": ParallelAdvancedTool,
    "firecrawl": FirecrawlSearch,
    "firecrawl_domains": FirecrawlDomains,
    "exa_auto": ExaSearch,
    "exa_auto_domains": ExaAutoDomains,
    "exa_fast": ExaFast,
    "exa_instant": ExaInstant,
    "exa_lite": ExaLite,
    "linkup_fast": LinkupSearch,
    "linkup_standard": LinkupStandard,
    "tavily_basic": TavilyBasic,
    "tavily_advanced": TavilyAdvanced,
    "brave": BraveSearch,
    "brave_tokens_6100": BraveTokens,
    "serp": SerpSearch,
    "serp_en_us": SerpEnUs,
    "string": StringSearch,
    "you_highlights": YouHighlights,
    "you_highlights_core": YouHighlightsCore,
    "you_highlights_core_documented": YouHighlightsCoreDocumented,
    "tinyfish": TinyfishSearch,
    "tinyfish_no_purpose": TinyfishNoPurpose,
    "tako": TakoSearch,
    "tako_plain": TakoPlain,
    "tako_snippet_3400": TakoSnippet3400,
    "perplexity_web_tool": PerplexityWebTool,
    "perplexity_fast_tool": PerplexityFastTool,
    "perplexity_web_tool_budget": PerplexityWebToolBudget,
    "perplexity_web_mcp": PerplexityWebMcp,
    "perplexity_web_tool_siteq": PerplexityWebToolSiteq,
    "keenable_pro": KeenableSearch,
    "keenable_realtime": KeenableRealtime,
    "keenable_pro_snippet4000": KeenableProSnippet4000,
}


def resolve_backend(name: str) -> str:
    return BACKEND_ALIASES.get(name, name)


def get_backend(name: str) -> SearchBackend:
    canonical = resolve_backend(name)
    factory = BACKENDS.get(canonical)
    if factory is None:
        known = ", ".join(sorted(set(resolve_backend(item) for item in BACKENDS)))
        raise KeyError(f"unknown search backend {name!r}; wired now: {known}")
    return factory()
