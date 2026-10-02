"""
Builds the multi-agent system:

    research_manager (CodeAgent)            <- talks to the user, plans, searches, formats citations
      ├── tools: search_arxiv, get_paper_details, format_bibtex
      └── managed agent: paper_summarizer (CodeAgent)
            └── tools: read_paper_pdf, get_paper_details

The manager calls the summarizer like a tool: `paper_summarizer(task="Summarize 1706.03762 ...")`. Long PDF text stays
inside the summarizer's own memory, so the manager's context only receives the finished summary.
"""
import os

from smolagents import CodeAgent, InferenceClientModel

from guardrails import FINAL_ANSWER_CHECKS
from tools import format_bibtex, get_paper_details, read_paper_pdf, search_arxiv

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-Coder-32B-Instruct")

SUMMARIZER_INSTRUCTIONS = """You are a careful research paper summarizer.
Use read_paper_pdf to read the paper; print each chunk so you can see it. Read the beginning (abstract, introduction)
and keep reading with next_start_char until you have seen the method, results, and conclusion (you do not need the
references or appendix). If the PDF cannot be read, fall back to get_paper_details and say the summary is abstract-only.

Return your summary in exactly this Markdown format:
### <Title> (arXiv:<id>)
- **Problem:** what question the paper addresses and why it matters
- **Method:** the core idea or approach
- **Key results:** main findings, with concrete numbers when the paper reports them
- **Limitations:** weaknesses stated by the authors or evident from the paper
- **Takeaway:** one sentence a busy reader should remember

Only state things that appear in the paper text. Never invent numbers."""

MANAGER_INSTRUCTIONS = """You are an arXiv research assistant. You only help with finding, understanding, comparing,
and citing academic papers; politely decline anything else.
- Use search_arxiv to find papers (use sort_by='newest' when the user asks for recent work).
- When the user wants a summary or details beyond the abstract, delegate to the paper_summarizer team member,
  giving it the arXiv ID and what the user cares about. Summarize one paper per call.
- Always use format_bibtex for BibTeX; never write BibTeX yourself.
- Only cite papers that your tools actually returned, and include each paper's arXiv ID and URL.
- Format the final answer as clean Markdown."""


def build_agent() -> CodeAgent:
    model = InferenceClientModel(model_id=MODEL_ID, token=os.environ.get("HF_TOKEN"))

    summarizer = CodeAgent(
        tools=[read_paper_pdf, get_paper_details],
        model=model,
        name="paper_summarizer",
        description=(
            "Reads the full PDF of one arXiv paper and returns a structured summary (problem, method, key results, "
            "limitations, takeaway). Pass a task containing the arXiv ID and any specific focus, e.g. "
            "'Summarize 1706.03762, focusing on the attention mechanism'."
        ),
        instructions=SUMMARIZER_INSTRUCTIONS,
        max_steps=6,
    )

    return CodeAgent(
        tools=[search_arxiv, get_paper_details, format_bibtex],
        model=model,
        managed_agents=[summarizer],
        instructions=MANAGER_INSTRUCTIONS,
        final_answer_checks=FINAL_ANSWER_CHECKS,
        add_base_tools=False,
        max_steps=10,
    )
