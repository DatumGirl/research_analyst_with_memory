"""Search and fetch tools backed by Tavily.

Each tool returns two things: text for the model to read, and a list of Source
objects as the artifact, which the graph records as citable evidence. Results
are cached, so a repeated query or URL costs no API call.
"""

from langchain_core.tools import BaseTool, tool
from tavily import TavilyClient

from research_analyst.cache import ToolCache
from research_analyst.schemas import Source

SEARCH_TOOL_NAME = "search"
FETCH_TOOL_NAME = "fetch"
TRUNCATION_NOTICE = "\n\n[Page cut off here: it exceeded the {limit}-character fetch limit.]"


def format_search_results(sources: list[Source]) -> str:
    """Render search hits as text the model can scan and choose from."""
    if not sources:
        return "No results found. Try a different query."
    return "\n\n".join(
        f"Title: {source.title}\nURL: {source.url}\nSnippet: {source.content}" for source in sources
    )


def truncate_page(text: str, max_chars: int) -> str:
    """Cut page text to max_chars, saying so explicitly when anything is dropped."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + TRUNCATION_NOTICE.format(limit=max_chars)


def build_search_tool(client: TavilyClient, max_results: int, cache: ToolCache) -> BaseTool:
    """Create the web search tool.

    Args:
        client: Tavily client used to run the search.
        max_results: Number of results returned per query.
        cache: Where results are stored and looked up.
    """

    @tool(SEARCH_TOOL_NAME, response_format="content_and_artifact")
    def search(query: str) -> tuple[str, list[Source]]:
        """Search the web. Returns titles, URLs, and short snippets.

        Snippets are only a preview. Call fetch on a URL to read the full page
        before relying on it for a claim.

        Args:
            query: A focused search query, as you would type into a search engine.
        """
        arguments = {"query": query, "max_results": max_results}
        hits = cache.get(SEARCH_TOOL_NAME, arguments)
        if hits is None:
            hits = client.search(query, max_results=max_results)["results"]
            cache.put(SEARCH_TOOL_NAME, arguments, hits)
        sources = [
            Source(url=hit["url"], title=hit["title"], content=hit["content"]) for hit in hits
        ]
        return format_search_results(sources), sources

    return search


def build_fetch_tool(client: TavilyClient, max_chars: int, cache: ToolCache) -> BaseTool:
    """Create the page fetch tool.

    Args:
        client: Tavily client used to extract page text.
        max_chars: Page text kept before the rest is cut off.
        cache: Where fetched pages are stored and looked up.
    """

    @tool(FETCH_TOOL_NAME, response_format="content_and_artifact")
    def fetch(url: str) -> tuple[str, list[Source]]:
        """Fetch the full text of a web page.

        Args:
            url: The exact URL to read, usually one returned by search.
        """
        arguments = {"url": url}
        page = cache.get(FETCH_TOOL_NAME, arguments)
        if page is None:
            pages = client.extract(urls=[url])["results"]
            if not pages:
                return f"Could not fetch {url}. Try a different source.", []
            page = pages[0]
            cache.put(FETCH_TOOL_NAME, arguments, page)
        text = truncate_page(page["raw_content"], max_chars)
        return text, [Source(url=url, title=page.get("title") or url, content=text)]

    return fetch
