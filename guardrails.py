"""
Guardrails for the arXiv agent.

- Input guardrail: runs on the user's message *before* the agent sees it.
- Output guardrails: passed to the agent as `final_answer_checks`. smolagents calls each one with the proposed final
  answer; if a check raises, the error is fed back to the LLM and it gets another step to fix its answer.
"""
import logging
import re

import arxiv

from tools import ARXIV_ID_PATTERN, known_arxiv_ids, query_arxiv

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 2000

# Common prompt-injection phrasings. A keyword list is not bulletproof, but it blocks the low-effort attempts.
INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (your|the) (instructions|system prompt|rules)",
    r"(reveal|print|show) (your|the) (system prompt|instructions)",
    r"you are no longer",
    r"\bjailbreak\b",
]

# The agent can execute Python, so refuse requests that try to steer it toward the host machine.
DANGEROUS_PATTERNS = [
    r"\b(os\.system|subprocess|shutil\.rmtree|rm -rf)\b",
    r"\b(api[_ ]?key|hf_token|password|environment variables?)\b",
]


def check_user_input(message: str) -> str | None:
    """Returns a refusal message if the input should be blocked, otherwise None."""
    text = message.strip()
    refusal = _input_refusal(text)
    if refusal:
        logger.warning("input blocked (%s): %.100r", refusal, text)
    return refusal


def _input_refusal(text: str) -> str | None:
    if not text:
        return "Please enter a question about research papers."
    if len(text) > MAX_INPUT_CHARS:
        return f"Your message is {len(text)} characters long; please keep it under {MAX_INPUT_CHARS}."
    lowered = text.lower()
    if any(re.search(p, lowered) for p in INJECTION_PATTERNS):
        return "I can't follow requests to override my instructions. Ask me about arXiv papers instead."
    if any(re.search(p, lowered) for p in DANGEROUS_PATTERNS):
        return "I can only help with finding, reading, and summarizing arXiv papers."
    return None


def answer_is_not_empty(final_answer, memory, **kwargs) -> bool:
    """Rejects empty or trivially short answers."""
    if final_answer is None or len(str(final_answer).strip()) < 20:
        logger.warning("final answer rejected: too short (%r)", final_answer)
        raise ValueError("The final answer is empty or too short. Give the user a complete answer.")
    return True


def cited_arxiv_ids_exist(final_answer, memory, agent=None, **kwargs) -> bool:
    """Catches hallucinated citations: every arXiv ID in the final answer must be a real paper."""
    cited_ids = set(ARXIV_ID_PATTERN.findall(str(final_answer)))
    # IDs the user typed themselves may appear in the answer, e.g. "No paper found for ID 2601.99999".
    user_ids = set(ARXIV_ID_PATTERN.findall(getattr(agent, "task", "") or ""))
    cited_ids -= user_ids
    # Papers the tools already returned are real; only look up the rest, so this check rarely hits the arXiv API.
    cited_ids -= known_arxiv_ids()
    if not cited_ids:
        return True

    search = arxiv.Search(id_list=sorted(cited_ids), max_results=len(cited_ids))
    try:
        papers = query_arxiv(search, cache_key=("ids", tuple(sorted(cited_ids))))
    except RuntimeError as e:
        # Don't trap the agent in a retry loop when arXiv itself is unavailable.
        logger.warning("citation check skipped, arXiv unavailable: %s", e)
        return True

    found = set()
    for paper in papers:
        short_id = paper.get_short_id()
        found.add(short_id)
        found.add(re.sub(r"v\d+$", "", short_id))  # a citation may omit the version suffix

    missing = cited_ids - found
    if missing:
        logger.warning("final answer rejected: nonexistent arXiv IDs %s", sorted(missing))
        raise ValueError(
            f"These arXiv IDs in your answer do not exist: {sorted(missing)}. "
            "Only cite papers returned by your tools, then call final_answer again."
        )
    logger.info("citation check passed for %s", sorted(cited_ids))
    return True


FINAL_ANSWER_CHECKS = [answer_is_not_empty, cited_arxiv_ids_exist]
