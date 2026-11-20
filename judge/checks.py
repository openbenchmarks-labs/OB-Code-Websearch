"""Public-set deterministic checks: scoring helpers, execution check, operation check, facts check, and answer grading."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.run import CodingTask

# === Scoring Helpers ===


import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlsplit

FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", re.S)
COMMENT_URL = re.compile(r"https?://[^\s#]+", re.I)
JUDGE_VERSION = "needles-v5"


@dataclass
class JudgeScore:
    passed: bool
    compiled: bool
    output: str
    notes: list[str] = field(default_factory=list)
    version: str = JUDGE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "compiled": self.compiled,
            "output": self.output,
            "notes": list(self.notes),
            "judge_version": self.version,
        }


def extract_code(answer: str, outfile: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""
    fences = [block.strip() for block in FENCE.findall(text) if block.strip()]
    if fences:
        return fences[-1]
    if text.startswith("```"):
        return ""
    if outfile.endswith(".py") and ("import " in text or "def " in text):
        return text
    return text


def comment_source_urls(source: str) -> list[str]:
    found: list[str] = []
    for line in (source or "").splitlines():
        if not line.lstrip().startswith("#"):
            continue
        for match in COMMENT_URL.findall(line):
            url = match.rstrip(").,;\"'")
            if url not in found:
                found.append(url)
    return found


def needle_source_urls(source: str, needles: list[str]) -> dict[str, str | None]:
    cited: dict[str, str | None] = {needle: None for needle in needles}
    last_url: str | None = None
    for line in (source or "").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            urls = comment_source_urls(line)
            if urls:
                last_url = urls[-1]
            continue
        for needle in needles:
            if needle_in_source(needle, line) and last_url:
                cited[needle] = last_url
    return cited


def normalize_cite_url(url: str) -> tuple[str, str, tuple[tuple[str, str], ...]] | None:
    """Host, path, and query identity. Fragments and trailing slashes are not distinct documents.

    http and https name the same page. Path case and query values stay intact, so a parent path,
    a longer sibling, or a different query is a different document.
    """
    raw = (url or "").strip().rstrip(").,;\"'")
    if not raw:
        return None
    parts = urlsplit(raw if "://" in raw else f"https://{raw}")
    host = (parts.hostname or "").lower().removeprefix("www.").rstrip(".")
    if not host:
        return None
    path = parts.path or "/"
    if len(path) > 1:
        path = path.rstrip("/")
    query = tuple(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return host, path, query


def url_was_observed(cited: str, observed: list[str]) -> bool:
    key = normalize_cite_url(cited)
    if key is None:
        return False
    return any(normalize_cite_url(item) == key for item in observed)


def hop_urls(run: Any) -> list[str]:
    found: list[str] = []
    for search in _searches(run):
        for url in list(getattr(search, "hit_urls", None) or (search.get("hit_urls") if isinstance(search, dict) else None) or []):
            if url and url not in found:
                found.append(url)
        for hit in _hits(search):
            url = str((hit or {}).get("url") or "")
            if url and url not in found:
                found.append(url)
    for fetch in _fetches(run):
        if not _fetch_returned_content(fetch):
            continue
        url = str(_fetch_url(fetch) or "")
        if url and url not in found:
            found.append(url)
    return found


def hop_retrieved_text(run: Any) -> str:
    """URL + title + preview of each hit and title + body of each fetch: what the model was shown. Queries excluded."""
    parts: list[str] = []
    for search in _searches(run):
        for hit in _hits(search):
            if not isinstance(hit, dict):
                continue
            parts.append(str(hit.get("url") or ""))
            parts.append(str(hit.get("title") or ""))
            parts.append(str(hit.get("snippet") or hit.get("text") or hit.get("content") or ""))
    for fetch in _fetches(run):
        parts.append(_fetch_title(fetch))
        parts.append(_fetch_body(fetch))
    return "\n".join(part for part in parts if part)


# A version segment ("/v1/", "/v2.1/", "/{api_version}/"). ServiceNow documents many endpoints both with
# and without one (/api/sn_devops/devops/... and /api/sn_devops/v1/devops/...), so a needle written without a
# version also matches a path that has one. A needle that names its version still needs that version.
VERSION_SEGMENT = re.compile(r"/(?:v\d+(?:\.\d+)*|\{[a-z_]*version\})(?=/)", re.I)


# A fixture value the ticket writer put where the docs have a placeholder ("interaction_hard" for {interaction},
# "object_hard" for {object}). Any single path segment stands in for it, in the file and in the docs.
FIXTURE_VALUE = re.compile(r":?[A-Za-z0-9]+_hard\b")


def _pattern(needle: str) -> re.Pattern[str] | None:
    """A regex for a needle holding fixture values; None when it holds none (plain substring match)."""
    if not FIXTURE_VALUE.search(needle):
        return None
    parts = FIXTURE_VALUE.split(needle)
    return re.compile(r"[^/\s\"'?&]+".join(re.escape(p) for p in parts), re.I)


def _contains(text: str, needle: str) -> bool:
    pattern = _pattern(needle)
    return needle in text if pattern is None else bool(pattern.search(text))


def unversioned(text: str) -> str:
    return VERSION_SEGMENT.sub("", text)


def has_version(needle: str) -> bool:
    return bool(VERSION_SEGMENT.search("/" + needle.strip("/") + "/"))


def needle_in_source(needle: str, source: str) -> bool:
    return _contains(source, needle) or (not has_version(needle) and _contains(unversioned(source), needle))


def _searches(run: Any) -> list[Any]:
    if run is None:
        return []
    if isinstance(run, dict):
        return list(run.get("searches") or run.get("search_hops") or [])
    return list(getattr(run, "searches", None) or [])


def _fetches(run: Any) -> list[Any]:
    if run is None:
        return []
    if isinstance(run, dict):
        return list(run.get("fetches") or run.get("fetch_hops") or [])
    return list(getattr(run, "fetches", None) or [])


def _hits(search: Any) -> list[Any]:
    if isinstance(search, dict):
        return list(search.get("hits") or [])
    return list(getattr(search, "hits", None) or [])


def _fetch_url(fetch: Any) -> str:
    if isinstance(fetch, dict):
        return str(fetch.get("url") or "")
    return str(getattr(fetch, "url", "") or "")


def _fetch_title(fetch: Any) -> str:
    if isinstance(fetch, dict):
        return str(fetch.get("title") or "")
    return str(getattr(fetch, "title", "") or "")


def _fetch_body(fetch: Any) -> str:
    if isinstance(fetch, dict):
        return str(fetch.get("content") or fetch.get("preview") or "")
    return str(getattr(fetch, "content", "") or getattr(fetch, "preview", "") or "")


def _fetch_returned_content(fetch: Any) -> bool:
    """A fetch counts as returned only if it has no error and a non-empty body.

    JS shells such as "Loading application..." still count: the vendor returned
    the page, it just failed to render it.
    """
    error = fetch.get("error") if isinstance(fetch, dict) else getattr(fetch, "error", None)
    return not error and bool(_fetch_body(fetch).strip())

# === Execution Check ===


import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextvars import ContextVar
from pathlib import Path

_LISTENER: ContextVar[socket.socket | None] = ContextVar("execcheck_listener", default=None)

DRIVER = r'''
import importlib.util, inspect, json, os, re, socket, sys, threading, typing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

SEEN = []

class Stand(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _any(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n).decode("utf-8", "replace") if n else ""
        u = urlsplit(self.path)
        body_keys, soap = [], None
        if raw.lstrip().startswith("<"):
            m = re.search(r"<(?:[\w-]+:)?Body\b[^>]*>\s*<(?:[\w-]+:)?([A-Za-z]\w*)", raw)
            soap = m.group(1) if m else None
        elif raw:
            try:
                data = json.loads(raw)
                body_keys = sorted(data) if isinstance(data, dict) else (sorted(data[0]) if isinstance(data, list) and data and isinstance(data[0], dict) else [])
            except ValueError:
                body_keys = sorted(parse_qs(raw))
        SEEN.append({"method": self.command.lower(), "path": u.path, "query": {k: v[0] for k, v in parse_qs(u.query).items()},
                     "body_keys": body_keys, "soap": soap})
        xml = raw.lstrip().startswith("<")
        out = (b"<soapenv:Envelope xmlns:soapenv='http://schemas.xmlsoap.org/soap/envelope/'><soapenv:Body/></soapenv:Envelope>" if xml else b"{}")
        self.send_response(200)
        self.send_header("Content-Type", "text/xml" if xml else "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)
    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _any

if len(sys.argv) > 2:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stand, bind_and_activate=False)
    srv.socket.close()
    srv.socket = socket.socket(fileno=int(sys.argv[2]))
    srv.server_address = srv.socket.getsockname()
    srv.server_name, srv.server_port = srv.server_address[0], srv.server_address[1]
else:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stand)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}"
src = open(sys.argv[1]).read()
# Point base URLs at the stand-in; leave every other setting at the fix's own default. A setting the fix reads with
# no default gets a test value (name_hard), as the test harness would have to supply one.
for name, default in re.findall(r"""(?:environ\.get|getenv)\(\s*["']([A-Za-z0-9_]+)["']\s*(?:,\s*(["'][^"']*["']|[^)]+))?\)""", src) + \
        [(n, "") for n in re.findall(r"""environ\[\s*["']([A-Za-z0-9_]+)["']\s*\]""", src)]:
    default = default.strip().strip("'\"")
    if default.startswith(("http://", "https://")) or (not default and re.search(r"BASE|URL|HOST|ENDPOINT", name, re.I)):
        os.environ[name] = base
    elif not default and name not in os.environ:
        os.environ[name] = "sk-judge" if re.search(r"TOKEN|KEY|SECRET|PASSWORD", name, re.I) else f"{name.lower()}_hard"
# Every request, to any host and over http or https, goes to the stand-in: httpx's transports are wrapped to rewrite
# the address to the stand-in's, keeping path and query, so a hard-coded vendor URL is recorded too.
import httpx
for p in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
    os.environ.pop(p, None)
_host, _port = srv.server_address

def _to_stand(request):
    request.url = request.url.copy_with(scheme="http", host=_host, port=_port)
    request.headers["host"] = f"{_host}:{_port}"
    return request

_sync, _async = httpx.HTTPTransport.handle_request, httpx.AsyncHTTPTransport.handle_async_request
httpx.HTTPTransport.handle_request = lambda self, request: _sync(self, _to_stand(request))
async def _ah(self, request):
    return await _async(self, _to_stand(request))
httpx.AsyncHTTPTransport.handle_async_request = _ah

def value(name, ann):
    ann = ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))
    low = (name + " " + ann).lower()
    if "client" in low or "session" in low:
        import httpx
        return httpx.Client()
    if "int" in ann.lower() or name.lower() in ("limit", "page_size", "pagesize", "size", "top", "count", "page"):
        return 25
    if "bool" in ann.lower():
        return True
    if "dict" in ann.lower() or "mapping" in ann.lower():
        return {}
    if "list" in ann.lower() or "sequence" in ann.lower():
        return []
    return f"{name}_hard"  # reads as a test value, so the grader treats it as filling an ID slot

error = None
try:
    spec = importlib.util.spec_from_file_location("fix", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "run")
    kwargs = {p.name: value(p.name, p.annotation) for p in inspect.signature(fn).parameters.values()
              if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    fn(**kwargs)
except BaseException as exc:
    error = f"{type(exc).__name__}: {str(exc)[:200]}"
print("@@RESULT@@" + json.dumps({"requests": SEEN, "error": error}))
sys.stdout.flush()
os._exit(0)
'''


@contextlib.contextmanager
def recorder_listener(fd: int) -> Iterator[None]:
    """Within this scope, `record` runs its stand-in on the caller's listening socket `fd`.

    `fd` must be an AF_INET, SOCK_STREAM socket bound to 127.0.0.1 and already listening; otherwise
    OSError or ValueError is raised before the scope is entered. The caller owns binding, listening, the port and
    closing `fd`; this scope only duplicates it, lends the duplicate to each driver process, and closes the duplicate
    on exit, so `fd` stays open and usable after success, exception or timeout. The listener is shared, so `record`
    calls in the scope must be sequential. The scope is held in a ContextVar: it covers the calling thread/context,
    nests (the inner listener wins until it exits) and is restored on exit.
    """
    owned = os.dup(fd)
    try:
        sock = socket.socket(fileno=owned)
    except BaseException:
        os.close(owned)
        raise
    try:
        if sock.family != socket.AF_INET or sock.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM:
            raise ValueError("recorder listener must be an AF_INET SOCK_STREAM socket")
        if sock.getsockname()[0] != "127.0.0.1":
            raise ValueError("recorder listener must be bound to 127.0.0.1")
        if sys.platform == "darwin":
            # tcp.h exposes tcpi_state as byte 0; tcp_fsm.h defines TCPS_LISTEN as 1.
            listening = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_CONNECTION_INFO, 1)[0] == 1
        else:
            listening = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
        if not listening:
            raise ValueError("recorder listener must be listening")
        token = _LISTENER.set(sock)
        try:
            yield
        finally:
            _LISTENER.reset(token)
    finally:
        sock.close()


def record(source: str, timeout: float = 15.0) -> dict:
    """Run `source` once against the stand-in; return the requests it sent and the error it raised, if any."""
    with tempfile.TemporaryDirectory() as tmp:
        fix, driver = Path(tmp) / "fix.py", Path(tmp) / "driver.py"
        fix.write_text(source or "")
        driver.write_text(DRIVER)
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "VIRTUAL_ENV", "PYTHONPATH", "TMPDIR")}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        listener = _LISTENER.get()
        args = [str(driver), str(fix)] + ([] if listener is None else [str(listener.fileno())])
        argv = [sys.executable, "-B", *args]
        pass_fds = () if listener is None else (listener.fileno(),)
        try:
            out = subprocess.run(argv, cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout, pass_fds=pass_fds)
        except subprocess.TimeoutExpired as exc:
            text = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            return {"requests": [], "error": "Timeout"} if "@@RESULT@@" not in text else json.loads(text.split("@@RESULT@@")[-1])
        line = next((l for l in out.stdout.splitlines() if l.startswith("@@RESULT@@")), None)
        if line is None:
            return {"requests": [], "error": (out.stderr.strip().splitlines() or ["no result"])[-1][:200]}
        return json.loads(line.removeprefix("@@RESULT@@"))

# === Operation Check ===


import ast
import re
from dataclasses import dataclass

JUDGE_VERSION = "ops-v2"
METHODS = ("get", "post", "put", "patch", "delete")
SLOT = "{}"
_HOLE = "\x00"  # stands for a value the source computes at run time
# Test values the ticket writer put where the docs have a placeholder: object_record_hard, object_hard, abc1a2b3c4d5e6.
FIXTURE = re.compile(r":?[A-Za-z0-9_.-]+_hard\b|[a-z]{3}1a2b3c4d5e6")
VERSION = re.compile(r"^v\d+(?:\.\d+)*$")
VERSION_LIKE = re.compile(r"^(?:v?\d+(?:\.\d+)*|latest)$")  # a version written as 11.13.18.05, or latest
# Words that read as data, not as a route: Veeva object names (person__sys), codes with digits or dashes (en-us).
PAGING = {"$format", "$top", "$skip", "$count", "$inlinecount", "$skiptoken", "format", "limit", "offset", "cursor", "page", "pagesize", "page_size"}
VALUE_LIKE = re.compile(r"__|\d|-")
SOAP_ELEMENT = re.compile(r"<(?:[\w-]+:)?Body\b[^>]*>\s*<(?:[\w-]+:)?([A-Za-z]\w*)")  # the first element in the SOAP body


# A placeholder in any style: {id}, {{id}}, <id>, URL-encoded %7Bid%7D.
PLACEHOLDER = re.compile(r"\{\{[^{}/\s]*\}\}|\{[^{}/\s]*\}|%7B[^/\s]*?%7D|<[\w.-]+>", re.I)


@dataclass(frozen=True)
class Op:
    method: str
    path: tuple[str, ...]
    soap: str | None = None


def _segment(raw: str) -> str:
    seg = FIXTURE.sub(_HOLE, PLACEHOLDER.sub(_HOLE, raw)).lower()
    if not seg.replace(_HOLE, "").strip(":"):
        return SLOT
    # An OData key reads the same entity set: BenefitInsurancePlan('abc') and BenefitInsurancePlan are one resource.
    seg = re.sub(r"\(.*\)$", "", seg)
    return seg.replace(_HOLE, SLOT)


def path_of(url: str) -> tuple[str, ...]:
    url = re.sub(r"^[a-z][a-z0-9+.-]*://[^/]*", "", url.strip(), flags=re.I)
    url = re.split(r"[?#]", url, maxsplit=1)[0]
    return tuple(_segment(s) for s in url.split("/") if s)


class _Reader:
    """Resolves the URL argument of each call in one source file to text, with holes for run-time values."""

    def __init__(self, tree: ast.AST) -> None:
        self.names: dict[str, ast.AST] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                self.names[node.targets[0].id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                self.names[node.target.id] = node.value

    def text(self, node: ast.AST | None, depth: int = 0) -> str:
        if node is None or depth > 6:
            return _HOLE
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(self.text(v.value if isinstance(v, ast.FormattedValue) else v, depth + 1) for v in node.values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self.text(node.left, depth + 1) + self.text(node.right, depth + 1)
        if isinstance(node, ast.Name) and node.id in self.names:
            return self.text(self.names[node.id], depth + 1)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("get", "getenv") and len(node.args) >= 2:  # os.environ.get("BASE", "http://host")
                return self.text(node.args[1], depth + 1)
            if node.func.attr == "format":
                return self.text(node.func.value, depth + 1)
        return _HOLE

    def ops(self, tree: ast.AST, soap: str | None) -> list[Op]:
        found: list[Op] = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            attr, args = node.func.attr.lower(), list(node.args)
            url_node = next((k.value for k in node.keywords if k.arg == "url"), None)
            if attr in METHODS:
                method, url_node = attr, url_node or (args[0] if args else None)
            elif attr == "request" and args:
                method = self.text(args[0]).lower()
                url_node = url_node or (args[1] if len(args) > 1 else None)
            else:
                continue
            url = self.text(url_node)
            if method not in METHODS or "/" not in url:  # dict.get("key") and friends
                continue
            if url.startswith(_HOLE):  # f"{BASE}/..." with BASE unknown: the host part
                url = url.lstrip(_HOLE)
            found.append(Op(method, path_of(url), soap if method == "post" else None))
        return found


def operations(source: str) -> list[Op]:
    """Every HTTP call in a Python source file, as operations. Empty when the file does not parse."""
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return []
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    soap = next((m.group(1).lower() for s in strings for m in [SOAP_ELEMENT.search(s)] if m), None)
    return _Reader(tree).ops(tree, soap)


def _core(path: tuple[str, ...], drop_versions: bool) -> tuple[str, ...]:
    p = path[1:] if path[:1] == ("api",) else path
    return tuple(s for s in p if not (drop_versions and VERSION.match(s)))


def _segments_match(agent: tuple[str, ...], ref: tuple[str, ...]) -> bool:
    # A variable where the reference names its version (f".../{api_version}/...") is the same address.
    return len(agent) == len(ref) and all(a == r or (a == SLOT and VERSION.match(r)) for a, r in zip(agent, ref))


def answer_tail(ref: Op, needles: list[str]) -> tuple[str, ...]:
    """The part of the reference address the ticket tests: from the first segment its answer strings name to the end.

    The vendor's base path before it (/odata/v2, /api/sn_devops/v1/...) is context the agent may write in
    another documented form; everything from the tested part to the end must match, so extra segments fail.
    """
    core = _core(ref.path, True)
    starts = []
    for needle in needles:
        segs = _core(path_of("/" + needle.strip().strip("/")), True)
        for i in range(len(core) - len(segs) + 1):
            if segs and _segments_match(core[i:i + len(segs)], segs):
                starts.append(i)
                break
    return core[min(starts):] if starts else core[-1:]


def _in_address(needle: str, ref: Op) -> bool:
    """Whether an answer string names part of the reference address (or its SOAP element) rather than a parameter."""
    if ref.soap and needle.strip().lower() == ref.soap:
        return True
    segs, core = _core(path_of("/" + needle.strip().strip("/")), True), _core(ref.path, True)
    return bool(segs) and any(_segments_match(core[i:i + len(segs)], segs) for i in range(len(core) - len(segs) + 1))


_ROUTES: dict[str, dict[str, set[tuple[str, str]]]] = {}


def documented_pairs(task: CodingTask) -> set[tuple[str, str]] | None:
    """Adjacent (parent, child) address segments of every route the vendor's manual documents, read from the file
    CODING_SEARCH_DOCUMENTED_ROUTES names. None when it is unset or the
    vendor has no list."""
    site = task.id.split("_")[1] if task.id.count("_") >= 2 else ""
    path = os.environ.get("CODING_SEARCH_DOCUMENTED_ROUTES")
    if not path:
        return None
    key = str(Path(path).resolve())
    if key not in _ROUTES:
        _ROUTES[key] = {}
        for name, routes in json.loads(Path(key).read_text()).items():
            pairs = set()
            for r in routes:
                segs = _core(path_of(r.split(" ", 1)[1]), True)
                pairs.update(zip(segs, segs[1:]))
            _ROUTES[key][name] = pairs
    return _ROUTES[key].get(site) or None


def same_operation(agent: Op, ref: Op, tail: tuple[str, ...], retrieved_text: str = "",
                   documented: set[tuple[str, str]] | None = None) -> bool:
    if agent.method != ref.method:
        return False
    mine = _core(agent.path, True)
    if ref.soap:  # the service is in the address; the operation is the request element
        return agent.soap == ref.soap and all(s in mine for s in tail if s != SLOT and not VERSION.match(s))
    if len(mine) < len(tail):
        return False
    text = (retrieved_text or "").lower().replace("\\_", "_")
    # The base path before the tested part must be the reference's (version aliases aside), or another form the search
    # tool showed: another base before /devops/orchestration/stepMapping passes only if a result showed that path.
    prefix = tuple(s for s in mine[:len(mine) - len(tail)] if not VERSION_LIKE.match(s) and s != SLOT)
    rc = _core(ref.path, True)
    ref_prefix = tuple(s for s in rc[:len(rc) - len(tail)] if not VERSION_LIKE.match(s) and s != SLOT)
    if prefix != ref_prefix and not (prefix and "/".join(prefix) in text):
        return False
    end = mine[len(mine) - len(tail):]
    for i, (a, r) in enumerate(zip(end, tail)):
        if a == r or (a == SLOT and VERSION.match(r)):
            continue
        # A word where the reference has a slot is a value (en-us, person__sys) unless it is its own route under the
        # same parent: /vobjects/{object}/actions is another operation, not a record id. The vendor's manual decides
        # when we have its route list; otherwise the search results and the word's shape do.
        if r == SLOT and a != SLOT and i > 0 and end[i - 1] != SLOT:
            route = (end[i - 1], a) in documented if documented is not None else (
                not VALUE_LIKE.search(a) and bool(re.search(rf"/{re.escape(end[i - 1])}/{re.escape(a)}(?![\w-])", text)))
            if not route:
                continue
        elif r == SLOT and a != SLOT:
            continue
        return False
    return True


def _shown_pattern(ref: Op, tail: tuple[str, ...]) -> re.Pattern[str]:
    if ref.soap:
        name = re.escape(ref.soap.removesuffix("_request"))
        return re.compile(rf"(?<![\w]){name}(?:_request)?(?![\w])", re.I)
    slot = r"(?:\{\{[^{}/\s]*\}\}|\{[^{}/\s]*\}|%7B[^/\s]*?%7D|<[\w.-]+>|:[\w.-]+|[^/\s\"'<>()?#{}]+)"
    parts = [slot if s == SLOT else re.escape(s).replace(re.escape(SLOT), slot) + r"(?:\([^)/\s]*\))?" for s in tail]
    version = r"(?:/(?:v\d+(?:\.\d+)*|\{[^}/\s]*version[^}/\s]*\}))?"
    return re.compile(r"(?<![A-Za-z0-9_])" + (version + "/").join(parts) + r"(?!/?[\w{:(])", re.I)


def shown(ref: Op, tail: tuple[str, ...], retrieved_text: str) -> bool:
    text = (retrieved_text or "").replace("\\_", "_")
    return bool(_shown_pattern(ref, tail).search(text))


def reference_ops(task: CodingTask) -> list[Op]:
    try:
        return operations(task.gold_source())
    except OSError:
        return []


_RECORDED: dict[tuple[str, str], dict] = {}


def _recorded(task: CodingTask, which: str) -> dict:
    """The requests the reference fix (or the starter) sends to the stand-in, recorded once per ticket."""
    key = (task.id, which)
    if key not in _RECORDED:
        src = task.gold_source() if which == "gold" else task.starter_source()
        _RECORDED[key] = record(src) if src else {"requests": [], "error": "no file"}
    return _RECORDED[key]


def _as_op(req: dict) -> Op:
    return Op(req["method"], path_of(req["path"]), (req.get("soap") or "").lower() or None)


def _keys(req: dict) -> dict[str, str]:
    """Query parameters and top-level body keys a request sent, by their original names."""
    return {**dict(req.get("query", {})), **{k: "" for k in req.get("body_keys", [])}}


def score_ops(source: str, task: CodingTask, *, observed_urls: list[str] | None = None,
              retrieved_text: str | None = None, require_cite: bool = False, execute: bool = True) -> JudgeScore:
    """Grade one fix. With execute (the default), the fix and the reference fix are run against the stand-in and their
    actual requests are compared: same request (method, address, SOAP element), every parameter the reference adds over
    the starter, and its value when the ticket names that value. Without it, requests are read from the code."""
    text = source or ""
    if not text.strip():
        return JudgeScore(False, False, "", ["empty submission"], JUDGE_VERSION)
    try:
        compile(text, task.outfile or "submission.py", "exec")
    except SyntaxError as exc:
        return JudgeScore(False, False, "OPS_FAIL", [f"{exc.__class__.__name__}: {exc}"], JUDGE_VERSION)
    needles = [n for n in list(task.required) + list(task.query_forbidden) if len((n or "").strip()) >= 4]
    refs, mine, notes = reference_ops(task), operations(text), []
    gold = _recorded(task, "gold") if execute else {"requests": []}
    run_reqs: list[dict] = []
    if gold["requests"]:
        refs = [_as_op(gold["requests"][0])]
        sent = record(text)
        run_reqs = sent["requests"]
        if run_reqs:
            mine = [_as_op(r) for r in run_reqs]
        # A fix that stops before sending anything here (it wants an input file, a date in a setting, a real login host)
        # may fail only because the stand-in's inputs aren't real ones; it is then read from its code, as without run.
    if not refs:
        notes.append("no_reference: the reference fix makes no readable call")
    for ref in refs:
        tail = answer_tail(ref, needles)
        label = f"{ref.method.upper()} …/{'/'.join(tail)}" + (f" {ref.soap}" if ref.soap else "")
        documented = documented_pairs(task)
        match = next((i for i, op in enumerate(mine) if same_operation(op, ref, tail, retrieved_text or "", documented)), None)
        if match is None:
            if mine:
                near = [op for op in mine if op.path and _core(op.path, True)[-1:] == _core(ref.path, True)[-1:]] or mine
                notes.append(f"wrong_call:{label} (sent {near[0].method.upper()} /{'/'.join(near[0].path)})")
            else:
                notes.append(f"no_request:{label} (the fix makes no request)")
            continue
        if run_reqs:
            # Parameters the ticket asks for: what the reference sends that the starter does not.
            starter = _recorded(task, "starter")["requests"]
            had = {k.lower() for k in (_keys(starter[0]) if starter else {})}
            added = {k: v for k, v in _keys(gold["requests"][0]).items() if k.lower() not in had}
            got = {k.lower(): v for k, v in _keys(run_reqs[match]).items()}
            prompt = task.prompt.lower()
            needle_text = " ".join(needles).lower()
            for k, v in added.items():
                # Required only when the ticket asks for it: the parameter's words are in the question or the answer key.
                # Paging and response-format switches ($top, $format) are the fix's own choice, not the ticket's.
                words = [w for w in re.split(r"[_\W]+|(?<=[a-z])(?=[A-Z])", k) if w]
                if k.lower() not in needle_text and (k.lower() in PAGING or len(k.strip("$")) < 4 or not all(w.lower() in prompt for w in words)
                                                    or any(f"optional {w.lower()}" in prompt for w in words)):  # "an optional description"
                    continue
                if k.lower() not in got:
                    notes.append(f"missing_param:{k}")
                elif v and not FIXTURE.search(v) and len(v) >= 3 and v.lower() in prompt and got[k.lower()].lower() != v.lower():
                    notes.append(f"wrong_value:{k}={got[k.lower()]} (ticket asks for {v})")
        if retrieved_text is not None and not shown(ref, tail, retrieved_text):
            notes.append(f"not_shown:{label}")
        if require_cite and task.cite_urls:
            cited = next(iter(needle_source_urls(text, [ref.path[-1]]).values()), None)
            if not cited or not url_was_observed(cited, observed_urls or []):
                notes.append(f"uncited:{label}")
    # Answer strings that are not part of the address (a query parameter, a body field) must be in the fix's code.
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    for needle in needles:
        if not any(_in_address(needle, r) for r in refs) and needle.strip() not in code:
            notes.append(f"missing:{needle.strip()}")
    passed = bool(refs) and not notes
    return JudgeScore(passed, True, "OPS_OK" if passed else "OPS_FAIL " + "; ".join(notes), notes, JUDGE_VERSION)

# === Facts Check ===


import re

JUDGE_VERSION = "facts-v2"

VENDORS = {"gen": "Genesys Cloud", "sf": "SAP SuccessFactors", "sap": "SAP S/4HANA Cloud", "sn": "ServiceNow",
           "ukg": "UKG Pro Workforce Management", "wd": "Workday", "vv": "Veeva Vault"}

ANSWER_SCHEMA = {
    "type": "json_schema",
    "name": "request",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "method": {"type": "string", "description": "HTTP method: GET, POST, PUT, PATCH or DELETE."},
            "path": {"type": "string", "description": "Address path after the host, as the docs write it; a value the caller supplies as {name}."},
            "operation": {"type": "string", "description": "For SOAP, the request element (for example Get_Workers_Request); empty otherwise."},
            "fields": {"type": "array", "items": {"type": "string"},
                       "description": "Query parameters or body fields the ticket needs sent, as name, or name=value when the docs fix the value (for example $expand=someNavigation); empty if none."},
        },
        "required": ["method", "path", "operation", "fields"],
        "additionalProperties": False,
    },
}

_RULES = """Answer with the request only: method, path, SOAP operation if the API is SOAP, and the query parameters or body fields the ticket needs. Do not write code. Write any value the caller supplies as {name}. When the docs fix a value the ticket needs (a navigation to expand, an action name), give it as name=value."""
SYSTEM = f"""You find the exact API request a ticket needs in the vendor's current public documentation.
You have web_search and web_fetch. You must call web_search before you answer. Do not rely on memory for paths, operation names or field names: use a name only if you saw it in a result. Fetch a specific URL when snippets are not enough. If you are unsure, search again; using more of the search budget is better than a wrong answer.
{_RULES}"""
SYSTEM_SEARCH_ONLY = f"""You find the exact API request a ticket needs in the vendor's current public documentation.
You have web_search. You do not have web_fetch. You must call web_search before you answer. Do not rely on memory for paths, operation names or field names: use a name only if you saw it in a result. If snippets are not enough, search again with a more specific query. If you are unsure, search again; using more of the search budget is better than a wrong answer.
{_RULES}"""
SYSTEM_NO_SEARCH = f"""You state the exact API request a ticket needs, from your own knowledge of the vendor's current public documentation. You cannot look anything up.
{_RULES}"""


def ticket_text(task: CodingTask) -> str:
    """The ticket sentence(s) from the locked prompt, without the code-editing instructions."""
    text = task.prompt.split("Desired state (product ticket):", 1)[-1].strip().split("\n\n")[0]
    return re.split(r"\s*Cite the https URL", text)[0].strip()


def question(task: CodingTask) -> str:
    starter = _recorded(task, "starter")["requests"]
    now = ""
    if starter:
        r = starter[0]
        now = f"{r['method'].upper()} {r['path']}" + (f" (SOAP operation {r['soap']})" if r.get("soap") else "")
    vendor = VENDORS.get(task.id.split("_")[1], "") if task.id.count("_") >= 2 else ""
    return (f"Vendor: {vendor}\n" if vendor else "") + (f"The program sends this request today: {now}\n" if now else "") + \
        f"Ticket: {ticket_text(task)}\n\nWhich request does the ticket need?"


def parse_answer(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and data.get("path") else None


def _words(name: str) -> list[str]:
    return [w.lower() for w in re.split(r"[_\W]+|(?<=[a-z])(?=[A-Z])", name) if len(w) >= 3]


def asked_fields(task: CodingTask) -> list[str]:
    """Parameters and body fields the reference adds over the starter that the ticket asks for."""
    gold, starter = _recorded(task, "gold")["requests"], _recorded(task, "starter")["requests"]
    if not gold:
        return []
    keys = lambda r: list(r.get("query", {})) + list(r.get("body_keys", []))
    had = {k.lower() for k in (keys(starter[0]) if starter else [])}
    ticket = ticket_text(task).lower()
    needles = " ".join(task.required + task.query_forbidden).lower()
    stem = lambda w: w[:-1] if w.endswith("s") else w
    ticket_words = {stem(w) for w in re.findall(r"[a-z]+", ticket)}
    out = []
    for k in keys(gold[0]):
        if k.lower() in had or k.lower() in PAGING:
            continue
        words = _words(k)
        asked = k.lower() in needles or (words and all(stem(w) in ticket_words for w in words))
        if asked and not any(f"optional {w}" in ticket for w in words):
            out.append(k)
    return out

def score_facts(answer: str, task: CodingTask, *, retrieved_text: str | None = None) -> JudgeScore:
    """A search run passes when the search results showed the right address; the model's stated answer is recorded
    (`answer:` notes) but does not decide it. Without search (the memory baseline) the answer itself is graded."""
    gold = _recorded(task, "gold")["requests"]
    # A reference fix that stops before sending is read from its code, as the operation judge does.
    refs = [Op(gold[0]["method"], path_of(gold[0]["path"]), (gold[0].get("soap") or "").lower() or None)] if gold else reference_ops(task)
    if not refs:
        return JudgeScore(False, True, "FACTS_FAIL no_reference", ["no_reference"], JUDGE_VERSION)
    ref = refs[0]
    needles = [n for n in task.required + task.query_forbidden if len((n or "").strip()) >= 4]
    tail = answer_tail(ref, needles)
    label = f"{ref.method.upper()} …/{'/'.join(tail)}" + (f" {ref.soap}" if ref.soap else "")
    data = parse_answer(answer)
    said = _answer_notes(data, task, ref, tail, needles, label, retrieved_text or "") if data else ["no_answer"]
    if retrieved_text is None:
        passed, notes = not said, said
    else:
        passed = shown(ref, tail, retrieved_text)
        notes = ([] if passed else [f"not_shown:{label}"]) + [f"answer:{n}" for n in said]
    return JudgeScore(passed, data is not None, ("FACTS_OK" if passed else "FACTS_FAIL") + ("" if not notes else " " + "; ".join(notes)),
                      notes, JUDGE_VERSION)


def _answer_notes(data: dict, task: CodingTask, ref: Op, tail: tuple[str, ...], needles: list[str], label: str, text: str) -> list[str]:
    """What is wrong with the stated request: another request, a field the ticket asks for left out, or an answer
    string outside the address not named."""
    raw = str(data.get("path") or "")
    url = urlsplit(raw if re.match(r"^[a-z]+://", raw, re.I) else "http://x" + ("" if raw.startswith("/") else "/") + raw)
    mine = Op(str(data.get("method") or "").lower(), path_of(url.path),
              (str(data.get("operation") or "").strip().split(":")[-1].lower()) or None)
    if ref.soap and mine.soap:
        mine = Op(mine.method, mine.path, mine.soap if mine.soap.endswith("_request") else mine.soap + "_request")
    notes = []
    if not same_operation(mine, ref, tail, text, documented_pairs(task)):
        notes.append(f"wrong_request:{label} (answered {mine.method.upper()} {raw}{' ' + mine.soap if mine.soap else ''})")
    named = {str(f).split("=", 1)[0].strip().lower() for f in data.get("fields") or []} | {k.lower() for k, _ in parse_qsl(url.query)}
    notes += [f"missing_field:{k}" for k in asked_fields(task) if k.lower() not in named]
    # Answer strings that are not part of the address (an expand target, a field) must appear in the answer.
    said = json.dumps(data).lower()
    whole = lambda n: re.search(rf"(?<![a-z0-9_]){re.escape(n.strip().lower())}(?![a-z0-9_])", said)
    notes += [f"missing:{n}" for n in dict.fromkeys(n.strip() for n in needles) if not _in_address(n, ref) and not whole(n)]
    return notes

# === Answer Grading ===

import re
from dataclasses import dataclass

# SAP prints some form-section keys both as sectIndex and sectionIndex.
LABEL_ALIAS = {'sectindex': 'sectionindex'}

UNREADABLE, NO_CALL = 'unreadable answer', 'no call'
WRONG_METHOD, WRONG_SOAP, WRONG_PATH, WRONG_KEYS, NO_EXPAND = (
    'wrong method', 'wrong SOAP service/operation', 'wrong path', 'wrong key labels', 'missing expand')
# How far a miss got through the checks, least to most: the answer is named by the right answer it came nearest to.
NEARNESS = (WRONG_METHOD, WRONG_SOAP, WRONG_PATH, WRONG_KEYS, NO_EXPAND)


@dataclass(frozen=True)
class Grade:
    passed: bool
    reason: str
    kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {'passed': self.passed, 'reason': self.reason, 'kind': self.kind}


def _answer_keys(inside: str) -> str:
    labels = (x.split('=')[0].strip().lower() for x in inside.split(',') if '=' in x)
    return '(' + ','.join(sorted(LABEL_ALIAS.get(l, l) for l in labels)) + ')'


def _segments(path: str) -> list[str]:
    path = re.sub(r'^https?://[^/]+', '', (path or '').split('?')[0]).rstrip('/')
    out = []
    for s in path.split('/'):
        s = re.sub(r'\(([^)]*)\)', lambda m: _answer_keys(m.group(1)), s)
        free = re.fullmatch(r'\{[^}]*\}|v\d+(\.\d+)?', s)
        if s:
            out.append('{}' if free else s.lower())
    return out[1:] if out[:1] == ['api'] else out


def _service(address: str) -> str:
    m = re.search(r'/ccx/service/[^/]+/([^/?]+)', address)
    assert m, f'a SOAP right answer without a Workday service: {address}'
    return m.group(1).lower()


def _expands(address: str) -> list[str]:
    """The top-level names a right answer expands ($expand=a,b/c,d(...)): the bare call returns only links to them."""
    m = re.search(r'[?&]\$expand=([^&]*)', address)
    if not m:
        return []
    names, depth, cur = [], 0, ''
    for ch in m.group(1) + ',':
        depth += (ch == '(') - (ch == ')')
        if ch == ',' and depth == 0:
            names.append(re.split(r'[(/]', cur)[0].strip())
            cur = ''
        else:
            cur += ch
    return [n for n in names if n]


def _op(s: str | None) -> str:
    return re.sub(r'_Request$', '', (s or '').strip())


def _same(got: list[str], want: list[str]) -> bool:
    # a placeholder in the right answer matches anything; the agent's own placeholder matches only a placeholder
    return len(got) == len(want) and all(a == b or b == '{}' for a, b in zip(got, want))


def _rest_miss(call: dict, right: dict) -> str | None:
    """None when the call is this right answer, else the first check it fails."""
    if (call.get('method') or '').upper() != right['method'].upper():
        return WRONG_METHOD
    got, want = _segments(call.get('path') or ''), _segments(right['address'])
    if not _same(got, want):
        bare = lambda segments: [re.sub(r'\(.*\)$', '', s) for s in segments]
        return WRONG_KEYS if _same(bare(got), bare(want)) else WRONG_PATH
    query = str(call.get('path')) + '&' + '&'.join(map(str, call.get('fields') or []))
    need = ([right['require_query']] if right.get('require_query') else []) + _expands(right['address'])
    return None if all(re.search(r'(?:^|[?&\s])\$?expand=[^&\s]*\b' + re.escape(n) + r'\b', query, re.I) for n in need) else NO_EXPAND


def grade(accepted: list[dict], answer: str | dict | None) -> Grade:
    try:
        call = json.loads(answer) if isinstance(answer, str) else (answer or {})
    except ValueError:
        return Grade(False, 'no readable answer', UNREADABLE)
    if not isinstance(call, dict) or not (call.get('path') or call.get('operation')):
        return Grade(False, 'no call in the answer', NO_CALL)
    misses = []
    for right in accepted:
        if right.get('soap'):
            # Workday owns each operation in one service; the call must be sent to that service
            sent_to = [x.lower() for x in re.split(r'[/?]', call.get('path') or '')]
            if call.get('operation') and _op(call['operation']) == _op(right['soap']) and _service(right['address']) in sent_to:
                return Grade(True, f"SOAP {right['soap']}")
            misses.append(WRONG_SOAP)
        else:
            miss = _rest_miss(call, right)
            if miss is None:
                return Grade(True, f"{right['method']} {right['address']}")
            misses.append(miss)
    return Grade(False, 'not one of the right answers', max(misses, key=NEARNESS.index))
