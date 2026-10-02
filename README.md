---
title: Arxiv Agent Lab
emoji: 🔥
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 6.27.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Agentic AI Lab
---

# Autonomous arXiv Research Agent

A multi-agent research assistant built with [smolagents](https://github.com/huggingface/smolagents) and Gradio. It searches arXiv, reads full papers, writes structured summaries, and produces accurate BibTeX.

## How it works

```
user ─► input guardrail ─► research_manager (CodeAgent)
                              ├─ search_arxiv / get_paper_details / format_bibtex
                              ├─ paper_summarizer (sub-agent) ─► read_paper_pdf
                              └─ final_answer ─► output guardrails ─(fail)─► agent retries
```

| File | Purpose |
|---|---|
| `app.py` | Gradio chat UI (entry point) |
| `research_agents.py` | Builds the manager agent and the `paper_summarizer` sub-agent |
| `tools.py` | arXiv tools the agents can call |
| `guardrails.py` | Input checks and final-answer checks |
| `smoke_test.py` | End-to-end test script that sends example prompts to the running app |

## Features

### Paper retrieval
- **`search_arxiv`**: search with `sort_by` (`relevance`, `newest`, `updated`) and an optional `category` filter such as `cs.AI`. Every word in the query must appear in the paper, so `agentic AI` doesn't return papers that only mention "AI". Advanced arXiv syntax like `ti:"large language models" AND abs:agents` also works. The category filter also matches papers cross-listed in that category.
- **`get_paper_details`**: look up one paper by ID or any arXiv URL (`1706.03762`, `arXiv:1706.03762v7`, `https://arxiv.org/abs/1706.03762`).
- **`format_bibtex`**: builds BibTeX from real arXiv metadata, so author lists and years are never made up.

> *"Find 3 recent papers on RAG in cs.CL and give me BibTeX for each."*

### PDF processing
- **`read_paper_pdf`**: downloads a paper's PDF and extracts the text with pypdf, returning it in chunks. To keep reading, call it again with the `next_start_char` value from the previous chunk. Each PDF is downloaded once and cached.
- PDFs come straight from `arxiv.org/pdf/<id>`, so reading works even when arXiv's search API is rate-limiting.
- Safety limits: the URL is always built from a validated arXiv ID (never an arbitrary link), files up to 30 MB, and the first 40 pages.

### Summarization
- **`paper_summarizer`**: a sub-agent that reads the full paper and returns a summary with these sections: **Problem, Method, Key results, Limitations, Takeaway**. The manager calls it whenever you ask for more than the abstract. The long PDF text stays in the sub-agent's memory, and the manager only gets the finished summary.
- smolagents' default sub-agent report template is replaced with a plain hand-off, so the manager passes the summary through unchanged instead of trying to parse it.

> *"Summarize 1706.03762, focusing on how attention works."*

### Guardrails
- **Input** (checked before the agent runs): blocks empty or over-long messages (limit 2,000 characters), common prompt-injection phrases, and requests for secrets or shell commands. These are keyword checks, so they stop simple attempts rather than every possible one.
- **Output** (`final_answer_checks`): rejects empty answers and any cited arXiv ID that doesn't exist. When a check fails, the agent is told why and gets another step to fix its answer.
  - IDs the user typed are allowed, so *"No paper found for 2601.99999"* isn't rejected.
  - Papers the tools already returned are checked from the cache, so the check rarely calls arXiv. If arXiv is down, the check is skipped instead of trapping the agent in retries.

### Chat UI
- Each response has two parts: a collapsible **🧠 Agent steps** box (each step's code, tools used, timing, and shortened output), then the clean final answer.
- BibTeX in answers is put in code blocks automatically so its line breaks are kept.
- Each browser session gets its own agent memory, so follow-up questions such as *"now compare the first two"* keep their context.

### Reliability
arXiv's API often rate-limits (HTTP 429), so the tools are built to fail fast instead of hanging:
- Each arXiv request times out after 20 s and retries once.
- Identical queries are cached for 15 minutes.
- After a failure, arXiv calls fail immediately for 60 s, and the agent is told to stop and tell the user to try again shortly.
- One step's code may run up to 90 s (smolagents' default is 30 s), leaving room for PDF downloads. LLM calls time out after 60 s.

### Logging
Logs print in the terminal and are saved to `logs/agent.log`. They include:
- each message, and whether the input guardrail blocked it
- every tool call with its arguments and duration
- every arXiv request, cache hit, and HTTP error
- one line per agent step: which agent, duration, tokens, tools called, and any error
- output-guardrail rejections and the total time per request

Set `LOG_LEVEL=DEBUG` for more detail.

## Running it

**Locally:**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt spaces
hf auth login                   # paste a token with "Make calls to Inference Providers" permission
export MODEL_ID=...             # optional, defaults to Qwen/Qwen2.5-Coder-32B-Instruct
python app.py                   # then open http://127.0.0.1:7860
```
Instead of `hf auth login`, you can `export HF_TOKEN=hf_...`.

**On Hugging Face Spaces:** add `HF_TOKEN` as a Space secret. You can also set `MODEL_ID` as a variable.

## Testing

With the app running, in a second terminal:
```bash
python smoke_test.py                  # all tests
python smoke_test.py BibTeX Memory    # only the named tests
```
It sends one prompt per feature (input guardrail, retrieval, BibTeX, PDF summary, follow-up memory, fake arXiv ID) and prints each answer, the tools called, and how long it took. See `logs/agent.log` for step-by-step details. If many tests report "arXiv is busy", arXiv is rate-limiting you; wait a few minutes and try again.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
