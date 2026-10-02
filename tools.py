import io
import re
import urllib.request

import arxiv
from pypdf import PdfReader
from smolagents import tool

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

_client = arxiv.Client(page_size=10, delay_seconds=3, num_retries=3)


def normalize_arxiv_id(raw: str) -> str:
    """Turns 'arXiv:2301.01234v2', 'https://arxiv.org/abs/2301.01234' or a pdf link into '2301.01234v2'."""
    match = ARXIV_ID_PATTERN.search(raw.strip())
    if not match:
        raise ValueError(f"'{raw}' does not look like an arXiv ID or arXiv URL.")
    return match.group(1)


def _paper_to_dict(paper: arxiv.Result) -> dict:
    return {
        "arxiv_id": paper.get_short_id(),
        "title": paper.title,
        "authors": ", ".join(author.name for author in paper.authors),
        "year": paper.published.year,
        "published": paper.published.strftime("%Y-%m-%d"),
        "primary_category": paper.primary_category,
        "url": paper.entry_id,
        "pdf_url": paper.pdf_url,
        "abstract": paper.summary.replace("\n", " "),
    }


def _fetch_paper(arxiv_id: str) -> arxiv.Result:
    arxiv_id = normalize_arxiv_id(arxiv_id)
    results = list(_client.results(arxiv.Search(id_list=[arxiv_id])))
    if not results:
        raise ValueError(f"No arXiv paper found with ID '{arxiv_id}'.")
    return results[0]


@tool
def search_arxiv(query: str, max_results: int = 3, sort_by: str = "relevance", category: str = "") -> list:
    """
    Searches arXiv for academic papers based on a query.

    Args:
        query: The search terms (e.g., 'Neural SDEs time series').
        max_results: The maximum number of papers to return, between 1 and 10. Defaults to 3.
        sort_by: How to order results: 'relevance' (default), 'newest' (most recently submitted), or 'updated'.
        category: Optional arXiv category to restrict the search to, e.g. 'cs.AI', 'cs.LG', 'stat.ML'. Empty for all.

    Returns:
        A list of dictionaries. Each dict contains 'arxiv_id', 'title', 'authors', 'year', 'published',
        'primary_category', 'url', 'pdf_url', and 'abstract'.
    """
    if sort_by not in SORT_OPTIONS:
        raise ValueError(f"sort_by must be one of {list(SORT_OPTIONS)}, got '{sort_by}'.")
    max_results = max(1, min(int(max_results), 10))

    full_query = f"cat:{category} AND ({query})" if category else query
    search = arxiv.Search(query=full_query, max_results=max_results, sort_by=SORT_OPTIONS[sort_by])
    return [_paper_to_dict(paper) for paper in _client.results(search)]


@tool
def get_paper_details(arxiv_id: str) -> dict:
    """
    Looks up a single arXiv paper by its ID or URL.

    Args:
        arxiv_id: An arXiv ID such as '1706.03762' or a URL like 'https://arxiv.org/abs/1706.03762'.

    Returns:
        A dict with 'arxiv_id', 'title', 'authors', 'year', 'published', 'primary_category', 'url', 'pdf_url', and 'abstract'.
    """
    return _paper_to_dict(_fetch_paper(arxiv_id))


def _download_pdf_text(paper: arxiv.Result) -> str:
    # Guardrail: only ever fetch from arxiv.org, never an arbitrary URL.
    if not paper.pdf_url.startswith(("https://arxiv.org/", "http://arxiv.org/")):
        raise ValueError(f"Refusing to download PDF from non-arXiv URL: {paper.pdf_url}")

    request = urllib.request.Request(paper.pdf_url, headers={"User-Agent": "arxiv-agent-lab/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise ValueError("PDF is too large to process (over 30 MB).")

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages[:MAX_PDF_PAGES]]
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
        A dict with 'arxiv_id', 'title', 'total_chars', 'start_char', 'next_start_char' (None when the end is reached),
        and 'text' (the extracted text chunk).
    """
    paper = _fetch_paper(arxiv_id)
    short_id = paper.get_short_id()
    if short_id not in _pdf_text_cache:
        _pdf_text_cache[short_id] = _download_pdf_text(paper)
    full_text = _pdf_text_cache[short_id]

    max_chars = max(1000, min(int(max_chars), 15000))
    start_char = max(0, int(start_char))
    end_char = min(start_char + max_chars, len(full_text))
    return {
        "arxiv_id": short_id,
        "title": paper.title,
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
