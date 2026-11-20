# OB-Code-Websearch

Open, independent benchmark of web search APIs for developers and coding agents: which search API helps a coding agent write working API calls for enterprise products (Workday, SAP, UKG, Veeva, Genesys and ServiceNow). The model, tickets, search budget and run day stay fixed; only the search tool changes. Search APIs measured: Brave, Exa, Firecrawl, Keenable, Linkup, Nimble, Parallel, Perplexity, SERP (RapidAPI), Tako, Tavily, TinyFish and You.com.

**Live leaderboard:** [openbenchmarks.com/web-search-for-coding-agents](https://openbenchmarks.com/web-search-for-coding-agents) · **Public dataset:** [`openbenchmarks/OB-Code-Websearch`](https://huggingface.co/datasets/openbenchmarks/OB-Code-Websearch)

Published and maintained by **[OpenBenchmarks Labs](https://openbenchmarks.com)**.

## Methodology

Each task gives a coding agent a realistic integration ticket against an enterprise product API (such as Workday, SAP SuccessFactors, UKG, Veeva, Genesys Cloud, or ServiceNow) along with starter code. The agent searches the web using the configured search API and produces a final API call.

- **Two boards:** **Search only** and **Search & fetch** are ranked separately.
- **Fixed protocol:** Every tool runs with the same coding model, turn limit, search budget, and day (`harness/config.yaml`).
- **Scoring:** `judge/` grades each final answer against ground truth, checking the constructed request, code extraction, citations, and retrieval (`judge/config.yaml`). Primary metrics are grounded task completion accuracy, latency, and token efficiency.

## Tools evaluated

14 search tools configured in [`tools/`](tools/README.md):

`brave`, `exa`, `firecrawl`, `keenable`, `linkup`, `nimble`, `parallel`, `perplexity`, `serp`, `string`, `tako`, `tavily`, `tinyfish`, `you`

## Dataset

- **Public split:** 30 enterprise API tickets are committed in [`data/questions.jsonl`](data/questions.jsonl) and mirrored on Hugging Face at [`openbenchmarks/OB-Code-Websearch`](https://huggingface.co/datasets/openbenchmarks/OB-Code-Websearch).
- **Held-out split:** The private evaluation set used for the published leaderboard is kept out of this repository to prevent contamination and training on test items.

See [`data/README.md`](data/README.md) for the full file format and field definitions.

## Repository structure

- [`data/`](data/README.md): 30 public benchmark questions and schema documentation.
- [`harness/`](harness/README.md): Coding agent loop, search API adapters, and CLI runner (`harness/run.py`).
- [`tools/`](tools/README.md): One `<tool>.yaml` config per search API (14 tools).
- [`judge/`](judge/README.md): Grading rules, code/citation/retrieval checks, and board aggregation.
- [`benchmark.yaml`](benchmark.yaml): Benchmark contract, scope, and refresh cadence.
- [`AGENTS.md`](AGENTS.md): Entry guide for coding agents.

## Quick start

Run the offline verification check (no API keys required):

```sh
uv venv && uv pip install -r requirements.txt
.venv/bin/python -B -m harness.run selftest --tasks data/questions.jsonl
```

To run against live APIs, copy `.env.example` to `.env`, add the required API keys, and inspect runner options:

```sh
cp .env.example .env
.venv/bin/python -m harness.run --help
```

## Adding or updating a tool

1. Add or edit `tools/<tool>.yaml` (and update `harness/search.py` and `.env.example` if adding a new search API).
2. Run the offline `selftest` command above.
3. Open a pull request.

## Links

- [Live leaderboard](https://openbenchmarks.com/web-search-for-coding-agents)
- [Most accurate](https://openbenchmarks.com/web-search-for-coding-agents/most-accurate-search-api), [fastest](https://openbenchmarks.com/web-search-for-coding-agents/fastest-search-api) and [most token-efficient](https://openbenchmarks.com/web-search-for-coding-agents/most-token-efficient-search-api) search APIs for coding agents
- [Hugging Face dataset](https://huggingface.co/datasets/openbenchmarks/OB-Code-Websearch)
- [`llms.txt`](https://openbenchmarks.com/llms.txt)

## Related benchmarks

- [OB-Company-News](https://github.com/openbenchmarks-labs/OB-Company-News): Company news search and news-index APIs ([leaderboard](https://openbenchmarks.com/company-news))
- [OB-Company-Search-Multi-Turn](https://github.com/openbenchmarks-labs/OB-Company-Search-Multi-Turn): Multi-hop company research agents ([leaderboard](https://openbenchmarks.com/multi-turn-company-search))
- [OB-FinSearch](https://github.com/openbenchmarks-labs/OB-FinSearch): Financial research search APIs
- [All OpenBenchmarks leaderboards](https://openbenchmarks.com)

## License

MIT. See [LICENSE](LICENSE).
