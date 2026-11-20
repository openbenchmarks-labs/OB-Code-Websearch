# AGENTS.md — OB-Code-Websearch

Quick navigation guide for coding agents working in `OB-Code-Websearch`.

## Structure (Zero Overlap)

| Folder | Owns | Files |
|---|---|---|
| **`data/`** | Public benchmark questions (30 questions) | `data/README.md`, `data/questions.jsonl` |
| **`harness/`** | Execution only: agent limits, CLI runner, agent loop, search adapters, logger | `harness/config.yaml`, `harness/run.py`, `harness/agent.py`, `harness/search.py`, `harness/logger.py` |
| **`tools/`** | Tool setups & pricing (one file per tool for clean PRs) | `tools/<tool>.yaml` (14 tools) |
| **`judge/`** | Grading only: rules & board config, v2 judge, checks, retrieval, verifier prompt | `judge/config.yaml`, `judge/grader.py`, `judge/checks.py`, `judge/retrieval.py`, `judge/verifier_prompt.txt` |

Benchmark order & win condition live at the root in `benchmark.yaml`.

## Commands

```sh
# Offline selftest (no API keys)
uv venv && uv pip install -r requirements.txt
.venv/bin/python -B -m harness.run selftest --tasks data/questions.jsonl
```

## Guidelines

- **Keep folders strictly separated:** Questions in `data/`; execution in `harness/`; per-tool configs in `tools/<tool>.yaml`; grading in `judge/`.
- **Adding or tuning a tool:** Edit `tools/<tool>.yaml` (and `harness/search.py` for new adapters), then verify with `selftest`.
- **No secrets or raw run logs in git:** Keep API keys in `.env` and run outputs in `runs/` (both gitignored).
- **Consistent terminology:** Use `tool`, `setup`, `question`, `ground truth`, `v2 deterministic judge`, `v2 LLM verifier`, `retrieval check`, `ranked questions`, `question group`, `board`, `main score`.
