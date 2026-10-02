import io
import logging
import re
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

import arxiv
import requests
from pypdf import PdfReader
from smolagents import tool

logger = logging.getLogger(__name__)

# Matches new-style (2301.01234, 2301.01234v2) and old-style (cs/0112017) arXiv IDs.
ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)")

SORT_OPTIONS = {
    "relevance": arxiv.SortCriterion.Relevance,
    "newest": arxiv.SortCriterion.SubmittedDate,
    "updated": arxiv.SortCriterion.LastUpdatedDate,
}

MAX_PDF_BYTES = 30 * 1024 * 1024  # Refuse to download PDFs larger than 30 MB
MAX_PDF_PAGES = 40                # Only extract the first 40 pages
_pdf_text_cache: dict[str, str] = {}  # arxiv_id -> extracted text, so paging doesn't re-download

ARXIV_TIMEOUT_SECONDS = 20       # Give up on a single arXiv request after 20 s instead of hanging
CACHE_TTL_SECONDS = 15 * 60      # Reuse identical arXiv queries for 15 minutes to avoid rate limits
ARXIV_COOLDOWN_SECONDS = 60      # After arXiv fails, skip further requests for 60 s instead of waiting on each
_query_cache: dict[tuple, tuple[float, list]] = {}  # cache_key -> (time fetched, results)
_arxiv_down_until = 0.0  # time.time() before which arXiv is treated as unavailable


class _TimeoutSession(requests.Session):
    """arxiv.Client has no timeout option, so give its HTTP session a default one."""

    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", ARXIV_TIMEOUT_SECONDS)
        return super().request(*args, **kwargs)


_client = arxiv.Client(page_size=10, delay_seconds=3, num_retries=1)
_client._session = _TimeoutSession()


@contextmanager
def _log_tool(name: str, **args):
    """Logs each tool call with its arguments, duration, and any error."""
    start = time.perf_counter()
    logger.info("tool %s called with %s", name, args)
    try:
        yield
    except Exception as e:
        logger.warning("tool %s failed after %.1fs: %s", name, time.perf_counter() - start, e)
        raise
    logger.info("tool %s finished in %.1fs", name, time.perf_counter() - start)


def known_arxiv_ids() -> set[str]:
    """IDs of every paper the tools have fetched recently (with and without version suffix), from the cache."""
    ids = set()
    for _, results in _query_cache.values():
        for paper in results:
            short_id = paper.get_short_id()
            ids.update({short_id, re.sub(r"v\d+$", "", short_id)})
    return ids


def query_arxiv(search: arxiv.Search, cache_key: tuple) -> list[arxiv.Result]:
    """Runs an arXiv API query with caching, logging, a failure cooldown, and errors the LLM can act on."""
    global _arxiv_down_until
    cached = _query_cache.get(cache_key)
    if cached and time.time() - cached[0] < CACHE_TTL_SECONDS:
        logger.info("arXiv cache hit for %s", cache_key)
        return cached[1]

    unavailable = ("arXiv is unavailable right now ({reason}). Stop calling tools and call final_answer telling the "
                   "user: arXiv is busy right now, please try again in a minute.")
    if time.time() < _arxiv_down_until:
        logger.warning("arXiv cooldown active, skipping %s", cache_key)
        raise RuntimeError(unavailable.format(reason="it failed moments ago"))

    start = time.perf_counter()
    try:
        results = list(_client.results(search))
    except arxiv.HTTPError as e:
        logger.error("arXiv HTTP %s for %s after %.1fs", e.status, cache_key, time.perf_counter() - start)
        if e.status in (429, 503):
            _arxiv_down_until = time.time() + ARXIV_COOLDOWN_SECONDS
            raise RuntimeError(unavailable.format(reason=f"rate limited, HTTP {e.status}")) from e
        raise RuntimeError(f"arXiv API error (HTTP {e.status}).") from e
    except requests.RequestException as e:
        logger.error("arXiv request failed for %s after %.1fs: %r", cache_key, time.perf_counter() - start, e)
        _arxiv_down_until = time.time() + ARXIV_COOLDOWN_SECONDS
        raise RuntimeError(unavailable.format(reason=type(e).__name__)) from e

    logger.info("arXiv query %s returned %d results in %.1fs", cache_key, len(results), time.perf_counter() - start)
    _query_cache[cache_key] = (time.time(), results)
    return results


def normalize_arxiv_id(raw: str) -> str:
    """Turns 'arXiv:2301.01234v2', 'https://arxiv.org/abs/2301.01234' or a pdf link into '2301.01234v2'."""
    match = ARXIV_ID_PATTERN.search(raw.strip())
    if not match:
        raise ValueError(f"'{raw}' does not look like an arXiv ID or arXiv URL.")
    return match.group(1)


def build_search_query(query: str, category: str = "") -> str:
    """
    Turns plain words into an arXiv query that requires every word, e.g. 'agentic AI' -> 'all:agentic AND all:AI'.
    arXiv otherwise matches papers containing *any* word, which gives off-topic results when sorting by date.
    Queries that already use arXiv syntax (ti:, abs:, au:, quotes, AND/OR) are passed through unchanged.
    """
    query = query.strip()
    if not re.search(r'\b(ti|abs|au|all|cat):|"|\b(AND|OR|ANDNOT)\b', query):
        terms = re.findall(r"[\w\-.]+", query)
        query = " AND ".join(f"all:{term}" for term in terms)
    return f"cat:{category} AND ({query})" if category else query


def _paper_to_dict(paper: arxiv.Result) -> dict:
    return {
        "arxiv_id": paper.get_short_id(),
        "title": paper.title,
        "authors": [author.name for author in paper.authors],
        "year": paper.published.year,
        "published": paper.published.strftime("%Y-%m-%d"),
        "primary_category": paper.primary_category,
        "url": paper.entry_id,
        "pdf_url": paper.pdf_url,
        "abstract": paper.summary.replace("\n", " "),
    }


def _fetch_paper(arxiv_id: str) -> arxiv.Result:
    arxiv_id = normalize_arxiv_id(arxiv_id)
    results = query_arxiv(arxiv.Search(id_list=[arxiv_id]), cache_key=("id", arxiv_id))
    if not results:
        raise ValueError(f"No arXiv paper found with ID '{arxiv_id}'.")
    return results[0]


@tool
def search_arxiv(query: str, max_results: int = 3, sort_by: str = "relevance", category: str = "") -> list:
    """
    Searches arXiv for academic papers based on a query.

    Args:
        query: The search terms (e.g., 'Neural SDEs time series'). Every word must appear in the paper, so keep it
            to 2-4 key words. Advanced: arXiv syntax like 'ti:"large language models" AND abs:agents' is also accepted.
        max_results: The maximum number of papers to return, between 1 and 10. Defaults to 3.
        sort_by: How to order results: 'relevance' (default), 'newest' (most recently submitted), or 'updated'.
        category: Optional arXiv category to restrict the search to, e.g. 'cs.AI', 'cs.LG', 'stat.ML'. Empty for all.

    Returns:
        A list of dictionaries. Each dict contains 'arxiv_id', 'title', 'authors' (a list of names), 'year', 'published',
        'primary_category', 'url', 'pdf_url', and 'abstract'.
    """
    with _log_tool("search_arxiv", query=query, max_results=max_results, sort_by=sort_by, category=category):
        if sort_by not in SORT_OPTIONS:
            raise ValueError(f"sort_by must be one of {list(SORT_OPTIONS)}, got '{sort_by}'.")
        max_results = max(1, min(int(max_results), 10))

        full_query = build_search_query(query, category)
        search = arxiv.Search(query=full_query, max_results=max_results, sort_by=SORT_OPTIONS[sort_by])
        results = query_arxiv(search, cache_key=("search", full_query, max_results, sort_by))
        return [_paper_to_dict(paper) for paper in results]


@tool
def get_paper_details(arxiv_id: str) -> dict:
    """
    Looks up a single arXiv paper by its ID or URL.

    Args:
        arxiv_id: An arXiv ID such as '1706.03762' or a URL like 'https://arxiv.org/abs/1706.03762'.

    Returns:
        A dict with 'arxiv_id', 'title', 'authors' (a list of names), 'year', 'published', 'primary_category', 'url',
        'pdf_url', and 'abstract'.
    """
    with _log_tool("get_paper_details", arxiv_id=arxiv_id):
        return _paper_to_dict(_fetch_paper(arxiv_id))


def _download_pdf_text(arxiv_id: str) -> str:
    # PDFs live at a fixed arxiv.org address, so this skips the (often rate-limited) export.arxiv.org search API.
    # Guardrail: the URL is built from a validated arXiv ID, so the agent can never make us fetch an arbitrary URL.
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    start = time.perf_counter()
    request = urllib.request.Request(pdf_url, headers={"User-Agent": "arxiv-agent-lab/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(MAX_PDF_BYTES + 1)
    except urllib.error.HTTPError as e:
        logger.error("PDF download %s failed: HTTP %s", pdf_url, e.code)
        if e.code == 404:
            raise ValueError(f"No arXiv paper found with ID '{arxiv_id}'.") from e
        raise RuntimeError(f"Could not download the PDF (HTTP {e.code}). Tell the user to try again later.") from e
    if len(data) > MAX_PDF_BYTES:
        raise ValueError("PDF is too large to process (over 30 MB).")
    logger.info("downloaded %s (%d KB) in %.1fs", pdf_url, len(data) // 1024, time.perf_counter() - start)

    start = time.perf_counter()
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages[:MAX_PDF_PAGES]]
    logger.info("extracted text from %d of %d pages in %.1fs", len(pages), len(reader.pages), time.perf_counter() - start)
    text = "\n".join(pages)
    text = re.sub(r"[ \t]+", " ", text)      # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse blank lines
    return text.strip()


@tool
def read_paper_pdf(arxiv_id: str, start_char: int = 0, max_chars: int = 8000) -> dict:
    """
    Downloads an arXiv paper's PDF and returns a chunk of its full text. Papers are long, so the text is returned
    in chunks: to keep reading, call again with start_char set to the previous 'next_start_char'.

    Args:
        arxiv_id: An arXiv ID such as '1706.03762' or an arXiv URL.
        start_char: Character offset to start reading from. Defaults to 0 (the beginning of the paper).
        max_chars: How many characters to return, between 1000 and 15000. Defaults to 8000.

    Returns:
        A dict with 'arxiv_id', 'total_chars', 'start_char', 'next_start_char' (None when the end is reached),
        and 'text' (the extracted text chunk; the paper's title is at the start of the first chunk).
    """
    with _log_tool("read_paper_pdf", arxiv_id=arxiv_id, start_char=start_char, max_chars=max_chars):
        short_id = normalize_arxiv_id(arxiv_id)
        if short_id not in _pdf_text_cache:
            _pdf_text_cache[short_id] = _download_pdf_text(short_id)
        full_text = _pdf_text_cache[short_id]

        max_chars = max(1000, min(int(max_chars), 15000))
        start_char = max(0, int(start_char))
        end_char = min(start_char + max_chars, len(full_text))
        return {
            "arxiv_id": short_id,
            "total_chars": len(full_text),
            "start_char": start_char,
            "next_start_char": end_char if end_char < len(full_text) else None,
            "text": full_text[start_char:end_char],
        }


@tool
def format_bibtex(arxiv_id: str) -> str:
    """
    Produces an accurate BibTeX entry for an arXiv paper from its real metadata. Always use this instead of
    writing BibTeX by hand.

    Args:
        arxiv_id: An arXiv ID such as '1706.03762' or an arXiv URL.

    Returns:
        A BibTeX @misc entry as a string.
    """
    with _log_tool("format_bibtex", arxiv_id=arxiv_id):
        paper = _fetch_paper(arxiv_id)
        short_id = paper.get_short_id()
        base_id = re.sub(r"v\d+$", "", short_id)  # drop the version suffix, e.g. 1706.03762v7 -> 1706.03762
        first_author_last = re.sub(r"[^A-Za-z]", "", paper.authors[0].name.split()[-1]) or "anon"
        first_title_word = next((w for w in re.findall(r"[A-Za-z]+", paper.title) if len(w) > 3), "paper")
        key = f"{first_author_last.lower()}{paper.published.year}{first_title_word.lower()}"

        return (
            f"@misc{{{key},\n"
            f"  title         = {{{paper.title}}},\n"
            f"  author        = {{{' and '.join(a.name for a in paper.authors)}}},\n"
            f"  year          = {{{paper.published.year}}},\n"
            f"  eprint        = {{{base_id}}},\n"
            f"  archivePrefix = {{arXiv}},\n"
            f"  primaryClass  = {{{paper.primary_category}}},\n"
            f"  url           = {{https://arxiv.org/abs/{base_id}}}\n"
            f"}}"
        )
