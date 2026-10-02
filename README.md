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

## Features

### Paper retrieval
- **`search_arxiv`**: search with `sort_by` (`relevance`, `newest`, `updated`) and an optional `category` filter such as `cs.AI`. The category filter also matches papers cross-listed in that category.
- **`get_paper_details`**: look up one paper by ID or any arXiv URL (`1706.03762`, `arXiv:1706.03762v7`, `https://arxiv.org/abs/1706.03762`).
- **`format_bibtex`**: builds BibTeX from real arXiv metadata, so author lists and years are never made up.

> *"Find 3 recent papers on RAG in cs.CL and give me BibTeX for each."*

### PDF processing
- **`read_paper_pdf`**: downloads a paper's PDF and extracts the text with pypdf, returning it in chunks. To keep reading, call it again with the `next_start_char` value from the previous chunk. Each PDF is downloaded once and cached.
- Safety limits: downloads only from arxiv.org, files up to 30 MB, and the first 40 pages.

### Summarization
- **`paper_summarizer`**: a sub-agent that reads the full paper and returns a summary with these sections: **Problem, Method, Key results, Limitations, Takeaway**. The manager calls it whenever you ask for more than the abstract. The long PDF text stays in the sub-agent's memory, and the manager only gets the finished summary.

> *"Summarize 1706.03762, focusing on how attention works."*

### Guardrails
- **Input** (checked before the agent runs): blocks empty or over-long messages (limit 2,000 characters), common prompt-injection phrases, and requests for secrets or shell commands. These are keyword checks, so they stop simple attempts rather than every possible one.
- **Output** (`final_answer_checks`): rejects empty answers and any cited arXiv ID that doesn't exist. When a check fails, the agent is told why and gets another step to fix its answer.

### Chat UI
- A multi-turn chat that shows each agent step as a collapsible message.
- Each browser session gets its own agent memory, so follow-up questions such as *"now compare the first two"* keep their context.

## Running it

**Locally:**
```bash
pip install -r requirements.txt spaces
export HF_TOKEN=hf_...          # Hugging Face token with Inference Providers access
export MODEL_ID=...             # optional, defaults to Qwen/Qwen2.5-Coder-32B-Instruct
python app.py
```

**On Hugging Face Spaces:** add `HF_TOKEN` as a Space secret. You can also set `MODEL_ID` as a variable.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
