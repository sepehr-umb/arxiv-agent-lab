"""
Builds the multi-agent system:

    research_manager (CodeAgent)            <- talks to the user, plans, searches, formats citations
      ├── tools: search_arxiv, get_paper_details, format_bibtex
      └── managed agent: paper_summarizer (CodeAgent)
            └── tools: read_paper_pdf, get_paper_details

The manager calls the summarizer like a tool: `paper_summarizer(task="Summarize 1706.03762 ...")`. Long PDF text stays
inside the summarizer's own memory, so the manager's context only receives the finished summary.
"""
import logging
import os

from smolagents import ActionStep, CodeAgent, InferenceClientModel

from guardrails import FINAL_ANSWER_CHECKS
from tools import format_bibtex, get_paper_details, read_paper_pdf, search_arxiv

logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-Coder-32B-Instruct")
LLM_TIMEOUT_SECONDS = 60   # Fail an LLM call after 60 s instead of the 120 s default
CODE_TIMEOUT_SECONDS = 90  # Max time for one step's code (smolagents defaults to 30 s, too short for PDF downloads)

SUMMARIZER_INSTRUCTIONS = """You are a careful research paper summarizer. Work in exactly two steps:
1. Read the paper: `chunk = read_paper_pdf(arxiv_id="<id>", max_chars=15000)` then `print(chunk["text"])`.
   If the user asked about a specific topic that is not covered yet, read one more chunk from chunk["next_start_char"].
2. After reading the printed text, write the summary YOURSELF as one Markdown string literal and pass it to
   final_answer. Do not build the summary with code from variables or dict fields, and do not use regex.
If read_paper_pdf fails, use get_paper_details instead and say the summary is based on the abstract only.
You can only call read_paper_pdf and get_paper_details; there is no web search.

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
- Do only what the user asked. For a search request, list each paper's title, authors, date, arXiv ID, URL, and a
  one-sentence description; do not summarize papers or produce BibTeX unless the user asks for it.
- Use search_arxiv to find papers (use sort_by='newest' when the user asks for recent work).
- Print tool results before formatting them, and only use the dict keys listed in each tool's description.
- When the user wants a summary or details beyond the abstract, delegate to the paper_summarizer team member,
  giving it the arXiv ID and what the user cares about. Summarize one paper per call.
- paper_summarizer returns a finished Markdown summary. Pass it to final_answer as-is (you may add text around it);
  never parse or split it with code. For follow-up questions about it, answer from what you already know.
- Always use format_bibtex for BibTeX; never write BibTeX yourself.
- Only cite papers that your tools actually returned, and include each paper's arXiv ID and URL.
- Trust search_arxiv's ranking; do not filter results by exact keyword matches and re-search.
- You can only call the tools listed above; there is no web search.
- If a tool says arXiv is unavailable, stop immediately and call final_answer explaining that to the user.
  This only applies to the current request: arXiv may be back, so always try your tools again on a new user message.
- Format the final answer as clean Markdown."""


def log_step(step, agent=None) -> None:
    """Step callback: logs one line per agent step so slow or failing steps are easy to spot."""
    if not isinstance(step, ActionStep):
        return
    name = agent.name if agent and agent.name else "research_manager"
    tokens = f"{step.token_usage.input_tokens} in / {step.token_usage.output_tokens} out" if step.token_usage else "n/a"
    code = " ".join((step.code_action or "").split())
    callable_names = list(agent.tools) + list(agent.managed_agents) if agent else []
    tools = [n for n in callable_names if f"{n}(" in code]
    code = code[:200]
    logger.info("[%s] step %d took %.1fs | tokens %s | tools %s | code: %s",
                name, step.step_number, step.timing.duration or 0, tokens, tools, code)
    if step.error:
        logger.warning("[%s] step %d error: %s", name, step.step_number, step.error)
    if step.is_final_answer:
        logger.info("[%s] final answer: %.300r", name, step.action_output)


def build_agent() -> CodeAgent:
    model = InferenceClientModel(model_id=MODEL_ID, token=os.environ.get("HF_TOKEN"), timeout=LLM_TIMEOUT_SECONDS)
    common = dict(
        model=model,
        step_callbacks=[log_step],
        executor_kwargs={"timeout_seconds": CODE_TIMEOUT_SECONDS},
    )

    summarizer = CodeAgent(
        tools=[read_paper_pdf, get_paper_details],
        name="paper_summarizer",
        description=(
            "Reads the full PDF of one arXiv paper and returns a structured summary (problem, method, key results, "
            "limitations, takeaway). Pass a task containing the arXiv ID and any specific focus, e.g. "
            "'Summarize 1706.03762, focusing on the attention mechanism'."
        ),
        instructions=SUMMARIZER_INSTRUCTIONS,
        max_steps=6,
        **common,
    )
    # smolagents' default sub-agent prompt demands a 3-part "Task outcome" report, which clashes with the summary
    # format above and confuses the manager. Replace it with a plain hand-off.
    summarizer.prompt_templates["managed_agent"]["task"] = (
        "You're '{{name}}'. Your manager gave you this task:\n---\n{{task}}\n---\n"
        "Your final_answer must be only the finished summary in the required Markdown format."
    )
    summarizer.prompt_templates["managed_agent"]["report"] = "{{final_answer}}"

    return CodeAgent(
        tools=[search_arxiv, get_paper_details, format_bibtex],
        managed_agents=[summarizer],
        instructions=MANAGER_INSTRUCTIONS,
        final_answer_checks=FINAL_ANSWER_CHECKS,
        add_base_tools=False,
        max_steps=10,
        **common,
    )
