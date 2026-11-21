#!/usr/bin/env python3
"""CLI runner for OB-Code-Websearch (selftest and benchmark execution)."""

from __future__ import annotations

import argparse
import json

# --- Environment & LLM client ---
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from judge import checks as facts
from judge.grader import grade_call

from .agent import DEFAULT_MODEL, run_agent

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

AZURE_KEY_ENV = "AZURE_OPENAI_NEXTGEN_DEPLOYMENT_KEY"
AZURE_URL_ENV = "AZURE_OPENAI_NEXTGEN_DEPLOYMENT_URL"


def load_environment() -> None:
    path = ROOT / ".env"
    if path.is_file():
        from dotenv import load_dotenv

        load_dotenv(path)


def llm_transport() -> str:
    if os.environ.get(AZURE_URL_ENV) and os.environ.get(AZURE_KEY_ENV):
        return "azure"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "missing"


def make_llm_client(*, timeout: float = 180, max_retries: int = 1) -> Any:
    """Azure Foundry OpenAI-v1 deployment when NEXTGEN env is set, else native OpenAI."""
    from openai import OpenAI

    azure_url = (os.environ.get(AZURE_URL_ENV) or "").rstrip("/")
    azure_key = os.environ.get(AZURE_KEY_ENV) or ""
    if azure_url and azure_key:
        return OpenAI(
            base_url=azure_url,
            api_key=azure_key,
            timeout=timeout,
            max_retries=max_retries,
        )
    key = os.environ.get("OPENAI_API_KEY") or ""
    if not key:
        raise SystemExit(
            f"set {AZURE_URL_ENV} + {AZURE_KEY_ENV} (preferred) or OPENAI_API_KEY"
        )
    return OpenAI(api_key=key, timeout=timeout, max_retries=max_retries)
# --- End Environment ---
# --- Task loader ---
from dataclasses import asdict, dataclass, field
from pathlib import Path

from judge.checks import extract_code, hop_retrieved_text, hop_urls, score_ops
from judge.retrieval import POLICY_VERSIONS, score_retrieval

from .logger import new_run_stamp, parse_run_stamp, split_label, write_arm_summary
from .search import (
    BACKEND_ALIAS_SPLIT,
    BACKENDS,
    FETCH_FORBIDDEN,
    FETCH_REQUIRED,
    SEARCH_FETCH_BACKENDS,
    SEARCH_ONLY_BACKENDS,
    get_backend,
    resolve_backend,
)


@dataclass
class CodingTask:
    id: str
    prompt: str
    starter_code: str
    gold_code: str
    outfile: str
    retrieval_key: dict[str, Any]
    required: list[str] = field(default_factory=list)
    query_forbidden: list[str] = field(default_factory=list)
    cite_urls: bool = True
    known_issue: str | None = None

    def leak_needles(self) -> list[str]:
        """Gold path tokens. Logged when a search query contains them; not withheld."""
        return list(self.query_forbidden)

    def gold_source(self) -> str:
        return self.gold_code

    def starter_source(self) -> str:
        return self.starter_code


def load_tasks(path: Path, *, ids: list[str] | None = None) -> list[CodingTask]:
    wanted = set(ids or [])
    out: list[CodingTask] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if wanted and row["task_id"] not in wanted:
            continue
        out.append(
            CodingTask(
                id=row["task_id"],
                prompt=row["prompt"],
                starter_code=row["starter_code"],
                gold_code=row["gold_code"],
                outfile=row["outfile"],
                retrieval_key=row["retrieval_key"],
                required=list(row["required"]),
                query_forbidden=list(row["query_forbidden"]),
                known_issue=row.get("known_issue"),
            )
        )
    if wanted and len(out) != len(wanted):
        missing = wanted - {task.id for task in out}
        raise KeyError(f"unknown task ids: {sorted(missing)}")
    return out
# --- End Task loader ---

SPLITS = ("search-fetch", "search-only")

PATCH_SYSTEM = """You edit an existing production script. Make the smallest change that satisfies the ticket.
Do not rewrite the file from scratch. Use the latest public APIs. Do not invent symbols that are not in the current docs.
You have web_search and web_fetch. You must call web_search before you return the file. Do not rely on memory for params, paths, or versions. Search for the current docs, then fetch a specific URL when snippets are not enough.
For every API version or field you add, put a `# source: https://...` comment on the line above with the search or fetch URL where you saw it. Do not use a name you cannot cite from a result you received.
Do not return until you are sure the path, version, and field names match current docs. If you are unsure, search or fetch again. Using more of the search budget is better than returning a wrong file. Only stop early when you are sure. Return only the full updated file."""
PATCH_SYSTEM_SEARCH_ONLY = """You edit an existing production script. Make the smallest change that satisfies the ticket.
Do not rewrite the file from scratch. Use the latest public APIs. Do not invent symbols that are not in the current docs.
You have web_search. You do not have web_fetch. You must call web_search before you return the file. Do not rely on memory for params, paths, or versions. Search for the current docs. If snippets are not enough, search again with a more specific query.
For every API version or field you add, put a `# source: https://...` comment on the line above with the search-result URL where you saw it. Do not use a name you cannot cite from a result you received.
Do not return until you are sure the path, version, and field names match current docs. If you are unsure, search again. Using more of the search budget is better than returning a wrong file. Only stop early when you are sure. Return only the full updated file."""


def parse_split(raw: str) -> str:
    name = (raw or "").strip().lower().replace("_", "-")
    aliases = {"fetch": "search-fetch", "search": "search-only"}
    name = aliases.get(name, name)
    if name not in SPLITS:
        raise SystemExit(f"unknown split {raw!r}; expected {', '.join(SPLITS)}")
    return name


def vendor_slug(backend: str) -> str:
    return backend.replace("_", "-")


def parse_arms(raw: str, *, split: str = "") -> tuple[str, ...]:
    text = (raw or "all").strip()
    if text == "all":
        if split == "search-only":
            return SEARCH_ONLY_BACKENDS
        if split == "search-fetch":
            return SEARCH_FETCH_BACKENDS
        raise SystemExit("--backend all requires --split search-only or search-fetch")
    arms = tuple(resolve_backend(item.strip()) for item in text.split(",") if item.strip())
    unknown = [item for item in arms if item not in BACKENDS]
    if unknown:
        known = ", ".join(sorted(set(resolve_backend(name) for name in BACKENDS)))
        raise SystemExit(f"unknown backend {unknown}; expected {known}")
    if split == "search-only":
        allowed = set(SEARCH_ONLY_BACKENDS)
        label = "search-only"
    elif split == "search-fetch":
        allowed = set(SEARCH_FETCH_BACKENDS)
        label = "search-fetch"
    else:
        allowed = None
        label = ""
    if allowed is not None:
        wrong = [item for item in arms if item not in allowed]
        if wrong:
            raise SystemExit(
                f"{wrong} is not on the {label} matrix; "
                f"use {', '.join(sorted(allowed))}"
            )
    return arms


def parse_fetch_modes(raw: str) -> tuple[bool, ...]:
    if raw == "on":
        return (True,)
    if raw == "off":
        return (False,)
    if raw == "both":
        return (True, False)
    raise SystemExit(f"unknown fetch-mode {raw!r}")


def arm_label(backend: str, allow_fetch: bool) -> str:
    return f"{backend}+fetch" if allow_fetch else f"{backend}+search"


def run_arm(
    task,
    *,
    client: Any,
    backend: str,
    allow_fetch: bool,
    model: str,
    max_turns: int,
    max_searches: int,
    max_fetches: int,
    raw_dir: Path | None = None,
    split: str = "",
    run_stamp: str = "",
    answer: str,
    retrieval_policy: str,
) -> dict[str, Any]:
    label = arm_label(backend, allow_fetch)
    evidence_root = (
        raw_dir / split_label(allow_fetch, split) / vendor_slug(backend) / run_stamp / task.id
        if raw_dir is not None
        else None
    )
    is_facts = answer == "facts"
    if is_facts:
        system = facts.SYSTEM if allow_fetch else facts.SYSTEM_SEARCH_ONLY
    else:
        system = PATCH_SYSTEM if allow_fetch else PATCH_SYSTEM_SEARCH_ONLY
    run = run_agent(
        facts.question(task) if is_facts else task.prompt,
        client=client,
        backend_name=backend,
        model=model,
        max_turns=max_turns,
        max_searches=max_searches,
        max_fetches=0 if not allow_fetch else max_fetches,
        system=system,
        allow_search=True,
        allow_fetch=allow_fetch,
        leak_needles=task.leak_needles(),
        text_format=facts.ANSWER_SCHEMA if is_facts else None,
        evidence_root=evidence_root,
    )
    source = run.answer if is_facts else extract_code(run.answer, task.outfile)
    if is_facts:
        verdict = grade_call(task.retrieval_key["accepted"], run.answer)
        retrieval = score_retrieval(task, search_hops=run.searches, fetch_hops=run.fetches, policy=retrieval_policy)
        passed = verdict.passed
        retrieval_passed = retrieval["passed"]
        details = {"answer": asdict(verdict), "retrieval": retrieval}
    else:
        score = score_ops(source, task, observed_urls=hop_urls(run), retrieved_text=hop_retrieved_text(run))
        passed = score.passed
        retrieval_passed = None
        details = score.to_dict()
    result = {
        "id": task.id,
        "backend": label,
        "search_backend": backend,
        "allow_fetch": allow_fetch,
        "used_search": run.used_search,
        "used_fetch": run.used_fetch,
        "search_count": len(run.searches),
        "fetch_count": len(run.fetches),
        "queries": [call.query for call in run.searches],
        "fetch_urls": [call.url for call in run.fetches],
        "search_hops": [call.hop_dict() for call in run.searches],
        "fetch_hops": [call.hop_dict() for call in run.fetches],
        "passed": passed,
        "retrieval_passed": retrieval_passed,
        "answer_mode": answer,
        "score": details,
        "metrics": run.metrics(),
        "source": source,
        "notes": run.notes,
        "query_contained_gold_token": any(bool(call.gold_token) for call in run.searches),
        "error": None,
    }
    if run.evidence_dir is not None:
        write_arm_summary(run=run, row=result)
        result["raw_dir"] = str(run.evidence_dir)
    return result


def failed_arm(task, *, backend: str, allow_fetch: bool, exc: BaseException) -> dict[str, Any]:
    label = arm_label(backend, allow_fetch)
    return {
        "id": task.id,
        "backend": label,
        "search_backend": backend,
        "allow_fetch": allow_fetch,
        "used_search": False,
        "used_fetch": False,
        "search_count": 0,
        "fetch_count": 0,
        "queries": [],
        "fetch_urls": [],
        "search_hops": [],
        "fetch_hops": [],
        "passed": False,
        "retrieval_passed": None,
        "score": {"passed": False, "compiled": False, "output": "RUN_ERROR", "notes": [str(exc)]},
        "metrics": {},
        "source": "",
        "notes": [f"{type(exc).__name__}: {exc}"],
        "query_contained_gold_token": False,
        "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
    }


def selftest(path: Path) -> int:
    tasks = load_tasks(path)
    if not tasks:
        print("FAIL no tasks loaded")
        return 1
    failed = 0
    starter_passes: list[str] = []
    for task in tasks:
        gold = score_ops(task.gold_source(), task)
        starter = score_ops(task.starter_source(), task)
        bad = []
        if not gold.passed:
            bad.append(f"gold {gold.output}")
        if starter.passed:
            starter_passes.append(task.id)
        if starter.passed != (task.known_issue is not None):
            bad.append("starter passed" if starter.passed else "starter failed")
        for reason in bad:
            print(f"FAIL {task.id} {reason}")
        failed += bool(bad)
    known = sorted(task.id for task in tasks if task.known_issue is not None)
    print(f"starters passing ({len(starter_passes)}) match known_issue rows ({len(known)}): {sorted(starter_passes) == known}")
    print(f"judge gold/starter ok={len(tasks) - failed} failed={failed}")
    names = tuple(dict.fromkeys((*SEARCH_ONLY_BACKENDS, *SEARCH_FETCH_BACKENDS)))
    for name in names:
        backend = get_backend(name)
        print(f"runner {name} constructed name={backend.name}")
    print(f"tasks={len(tasks)} llm={llm_transport()}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("selftest", help="Judge gold and starter code offline and construct the runners.")
    st.add_argument("--tasks", type=Path, required=True, help="JSONL file of tasks, one per line.")
    run = sub.add_parser("run", help="Score one or more tickets.")
    run.add_argument("--ids", default="")
    run.add_argument("--tasks", type=Path, required=True, help="JSONL file of tasks, one per line.")
    run.add_argument(
        "--backend",
        default="all",
        help=(
            "Vendor id, comma list, or all (default). all requires --split. "
            f"search-only: {', '.join(SEARCH_ONLY_BACKENDS)}. "
            f"search-fetch: {', '.join(SEARCH_FETCH_BACKENDS)}. "
            "Aliases: exa=exa_auto, tavily=tavily_advanced, linkup=linkup_standard. "
            "A bare alias runs its headline setup, search-only."
        ),
    )
    run.add_argument("--fetch-mode", default=None, choices=("on", "off", "both"))
    run.add_argument(
        "--answer",
        default="facts",
        choices=("code", "facts"),
        help="facts: grade the final API call against the key (default). code: run the submitted file.",
    )
    run.add_argument("--retrieval-policy", default="L1", choices=tuple(POLICY_VERSIONS), help="Retrieval policy for the retrieval verdict.")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--max-turns", type=int, default=32)
    run.add_argument("--max-searches", type=int, default=5)
    run.add_argument("--max-fetches", type=int, default=None, help="Fetch budget per task. Default: 0 for search-only, else 5.")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument(
        "--split",
        default="",
        help="search-only (no fetch) or search-fetch. Locks fetch on/off.",
    )
    run.add_argument("--out", type=Path, default=Path("/tmp/coding-search-eval.json"))
    run.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("runs/web-coding"),
        help="Write redacted tool hops under {split}/{tool}/{run_stamp}/{task_id}/. Use none to skip.",
    )
    run.add_argument(
        "--run-stamp",
        default="",
        help="UTC folder name for this dump (default: now, e.g. 2026-08-24T16-29-16Z). Distinguishes reruns.",
    )
    args = parser.parse_args(argv)
    load_environment()
    if args.cmd == "selftest":
        return selftest(args.tasks)
    ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    tasks = load_tasks(args.tasks, ids=ids or None)
    if not args.split:
        args.split = BACKEND_ALIAS_SPLIT.get(args.backend.strip(), "")
    split = parse_split(args.split) if args.split else ""
    if args.max_fetches is None:
        args.max_fetches = 0 if split == "search-only" else 5
    backends = parse_arms(args.backend, split=split)
    if args.fetch_mode is None:
        if split == "search-only":
            args.fetch_mode = "off"
        elif split == "search-fetch":
            args.fetch_mode = "on"
        elif backends and all(name in FETCH_FORBIDDEN for name in backends):
            args.fetch_mode = "off"
        elif backends and all(name in FETCH_REQUIRED for name in backends):
            args.fetch_mode = "on"
        else:
            args.fetch_mode = "both"
    fetch_modes = parse_fetch_modes(args.fetch_mode)
    if split == "search-only" and True in fetch_modes:
        raise SystemExit("search-only has no fetch; omit --fetch-mode or use --fetch-mode off")
    if split == "search-fetch" and False in fetch_modes:
        raise SystemExit("search-fetch requires fetch; omit --fetch-mode or use --fetch-mode on")
    fetch_on = [name for name in backends if name in FETCH_FORBIDDEN]
    if fetch_on and True in fetch_modes:
        raise SystemExit(
            f"{', '.join(fetch_on)} is search-only; use --split search-only (no fetch)"
        )
    fetch_off = [name for name in backends if name in FETCH_REQUIRED]
    if fetch_off and False in fetch_modes:
        raise SystemExit(
            f"{', '.join(fetch_off)} is search+fetch; use --split search-fetch"
        )
    run_stamp = parse_run_stamp(args.run_stamp) or new_run_stamp()
    client = make_llm_client(timeout=180, max_retries=1)
    jobs = [
        (task, backend, allow_fetch)
        for task in tasks
        for backend in backends
        for allow_fetch in fetch_modes
    ]
    workers = max(1, args.concurrency)
    print(
        f"split={split or '-'} "
        f"llm={llm_transport()} model={args.model} n={len(jobs)} concurrency={workers}",
        flush=True,
    )

    raw_dir = None if str(args.raw_dir) in {"", "-", "none"} else args.raw_dir
    if raw_dir is not None:
        print(f"raw_dir={raw_dir.resolve()} run_stamp={run_stamp}", flush=True)

    def _one(job: tuple) -> dict[str, Any]:
        task, backend, allow_fetch = job
        try:
            return run_arm(
                task,
                client=client,
                backend=backend,
                allow_fetch=allow_fetch,
                model=args.model,
                max_turns=args.max_turns,
                max_searches=args.max_searches,
                max_fetches=args.max_fetches,
                raw_dir=raw_dir,
                split=split,
                run_stamp=run_stamp,
                answer=args.answer,
                retrieval_policy=args.retrieval_policy,
            )
        except Exception as exc:  # noqa: BLE001
            return failed_arm(task, backend=backend, allow_fetch=allow_fetch, exc=exc)

    results: list[dict[str, Any]] = []
    if workers == 1:
        for job in jobs:
            row = _one(job)
            results.append(row)
            print(row["id"], row["backend"], "pass" if row["passed"] else "fail", row.get("error") or "", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_one, job): job for job in jobs}
            for fut in as_completed(futs):
                row = fut.result()
                results.append(row)
                print(row["id"], row["backend"], "pass" if row["passed"] else "fail", row.get("error") or "", flush=True)
    results.sort(key=lambda row: (row["id"], row["backend"]))
    passed = sum(1 for row in results if row["passed"])
    summary = {
        "n": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "backend": list(backends),
        "fetch_mode": args.fetch_mode,
        "model": args.model,
        "concurrency": workers,
        "raw_dir": str(raw_dir) if raw_dir else None,
        "run_stamp": run_stamp if raw_dir else None,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"passed={passed}/{len(results)} wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
