# harness

Runs the benchmark and calls the search APIs. No grading code lives here.

- `config.yaml` — Model, reasoning effort, turn, search and fetch limits, the default spend cap and the model's token prices.
- `run.py` — Command-line runner (`selftest` and live benchmark runs).
- `agent.py` — Runs the coding agent loop and collects its final API call.
- `spend.py` — The run's hard spend cap: refuses the next request that would pass it.
- `search.py` — API adapters for every search tool.
- `logger.py` — Saves run logs under `runs/` with API keys stripped out.
