"""
Multi-source research agent.
Uses OpenRouter (DeepSeek default) as orchestrator with multiple research sources:
- Perplexity Sonar (via OpenRouter) for AI-powered web research
- SearXNG for image, video, news, and document search
- Jina Reader for page scraping
- youtube-transcript-api for YouTube transcripts
Yields structured events for the UI via websocket.
"""

import json
import os
import re
import sys
from typing import Generator

import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = os.environ.get("MODEL", "deepseek/deepseek-chat")
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng.example.com:8888")
JINA_PREFIX = "https://r.jina.ai/"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_research",
            "description": (
                "Research a topic using AI-powered web search. Returns a detailed answer "
                "with citations and source URLs. This is your PRIMARY research tool. "
                "Use it for any factual question, background research, or investigation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The research question or topic to investigate"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_images",
            "description": (
                "Search for images related to a query. Returns URLs, titles, and thumbnails. "
                "Use for finding photos, satellite imagery, visual evidence, or illustrations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Image search query"},
                    "limit": {"type": "integer", "description": "Max results (default 5)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_videos",
            "description": (
                "Search for videos related to a query. Returns URLs, titles, durations, and thumbnails. "
                "Use for finding video evidence, testimonies, documentaries, or news footage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Video search query"},
                    "limit": {"type": "integer", "description": "Max results (default 5)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "Search for recent news articles. Returns URLs, titles, dates, and sources. "
                "Use for finding current coverage, breaking news, or recent developments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "News search query"},
                    "limit": {"type": "integer", "description": "Max results (default 5)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search for PDF documents and reports. Returns URLs, titles, and sources. "
                "Use for finding official reports, legal documents, academic papers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Document search query"},
                    "limit": {"type": "integer", "description": "Max results (default 5)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": (
                "Read the full content of a web page as clean markdown. "
                "Use to get detailed content from a specific URL found via research or search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to read"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_video_transcript",
            "description": (
                "Get the transcript of a YouTube video. "
                "Use to extract spoken content from YouTube videos found via search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "youtube_url": {"type": "string", "description": "YouTube video URL or video ID"},
                },
                "required": ["youtube_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_social",
            "description": (
                "Search social media platforms. Use when the topic involves public discourse, "
                "eyewitness accounts, activist posts, or community discussions. "
                "Supported platforms: twitter (uses AI-powered X search), facebook, instagram, "
                "reddit, telegram. Use this when social media perspectives would add value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "platform": {
                        "type": "string",
                        "description": "Platform to search: twitter, facebook, instagram, reddit, telegram",
                        "enum": ["twitter", "facebook", "instagram", "reddit", "telegram"],
                    },
                },
                "required": ["query", "platform"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are a deep research agent. Your job is to find, extract, and synthesize \
information from the web on any topic the user asks about.

You have access to these tools:

- **web_research**: Your PRIMARY tool. It searches the web and returns AI-synthesized \
answers with source URLs. Use it for any factual question or investigation topic.
- **search_images**: Find relevant photos, satellite imagery, or visual evidence.
- **search_videos**: Find video evidence, testimonies, documentaries.
- **search_news**: Find recent news coverage and developments.
- **search_documents**: Find reports, PDFs, legal documents.
- **read_page**: Get full content of a specific URL as markdown.
- **get_video_transcript**: Get transcripts of YouTube videos found via search.
- **search_social**: Search social media (twitter, facebook, instagram, reddit, telegram). \
Use when eyewitness accounts, activist posts, or public discourse would add value.

Research methodology (YOU MUST FOLLOW ALL STEPS):
1. Start with web_research to get an overview and key sources
2. ALWAYS call search_images to find relevant photos and visual evidence
3. ALWAYS call search_videos to find video evidence and testimonies
4. ALWAYS call search_news to find recent news coverage
5. ALWAYS call search_documents to find PDF reports and official documents
6. Use read_page to dive deeper into specific sources when needed
7. Use get_video_transcript for important YouTube videos

IMPORTANT: You MUST use at least web_research, search_images, search_videos, \
search_news, and search_documents before writing your report. Never skip media searches.

When you have enough information, write a structured research report with:
- **Summary**: Key findings in 2-3 sentences
- **Sources**: List of URLs with what each contributed
- **Key Findings**: Detailed findings organized by theme
- **Gaps**: What information is still missing or unverified

Do NOT include images, videos, or documents sections in your report. \
Those will be appended automatically from your search results."""


http = httpx.Client(timeout=60.0)


def _sonar_research(query: str) -> str:
    """Use Perplexity Sonar via OpenRouter for AI-powered web research."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    response = client.chat.completions.create(
        model="perplexity/sonar-pro",
        messages=[
            {"role": "system", "content": "You are a research assistant. Provide detailed, factual answers with source URLs."},
            {"role": "user", "content": query},
        ],
        max_tokens=4096,
    )
    content = response.choices[0].message.content or ""
    return json.dumps({"answer": content}, indent=2)


def _searxng_search(query: str, category: str, limit: int = 5) -> str:
    """Query SearXNG for a specific category."""
    params = {"q": query, "categories": category, "format": "json"}
    r = http.get(f"{SEARXNG_URL}/search", params=params)
    r.raise_for_status()
    data = r.json().get("results", [])[:limit]

    if category == "images":
        results = [
            {
                "url": item.get("img_src") or item.get("url", ""),
                "title": item.get("title", ""),
                "source": item.get("source", "") or item.get("engine", ""),
                "thumbnail": item.get("thumbnail_src") or item.get("img_src") or item.get("url", ""),
            }
            for item in data
        ]
    elif category == "videos":
        results = [
            {
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "duration": item.get("length") or item.get("duration", ""),
                "thumbnail": item.get("thumbnail", "") or item.get("img_src", ""),
            }
            for item in data
        ]
    elif category == "news":
        results = [
            {
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "date": item.get("publishedDate", "") or item.get("pubdate", ""),
                "source": item.get("engine", "") or item.get("source", ""),
            }
            for item in data
        ]
    else:  # files
        results = [
            {
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "source": item.get("engine", "") or item.get("source", ""),
            }
            for item in data
        ]

    return json.dumps(results, indent=2)


def _jina_read(url: str) -> str:
    """Read a page via Jina Reader."""
    r = http.get(f"{JINA_PREFIX}{url}", headers={"Accept": "text/markdown"})
    r.raise_for_status()
    content = r.text[:12000]
    return json.dumps({"url": url, "content": content}, indent=2)


def _extract_video_id(url_or_id: str) -> str:
    """Extract YouTube video ID from URL or return as-is if already an ID."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return url_or_id


def _youtube_transcript(youtube_url: str) -> str:
    """Get transcript from a YouTube video."""
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = _extract_video_id(youtube_url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    lines = [entry["text"] for entry in transcript]
    full_text = " ".join(lines)
    # Truncate if very long
    if len(full_text) > 10000:
        full_text = full_text[:10000] + "... [truncated]"
    return json.dumps({"video_id": video_id, "transcript": full_text}, indent=2)


SITE_FILTERS = {
    "facebook": "site:facebook.com",
    "instagram": "site:instagram.com",
    "reddit": "site:reddit.com",
    "telegram": "site:t.me",
}


def _search_social(query: str, platform: str) -> str:
    """Search social media. Twitter uses Grok, others use SearXNG site: filter."""
    if platform == "twitter":
        return _grok_x_search(query)

    site_filter = SITE_FILTERS.get(platform, "")
    search_query = f"{query} {site_filter}".strip()
    params = {"q": search_query, "format": "json"}
    r = http.get(f"{SEARXNG_URL}/search", params=params)
    r.raise_for_status()
    data = r.json().get("results", [])[:8]
    results = [
        {
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "snippet": (item.get("content", "") or "")[:300],
            "platform": platform,
        }
        for item in data
    ]
    return json.dumps(results, indent=2)


def _grok_x_search(query: str) -> str:
    """Search X/Twitter using Grok via OpenRouter."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    response = client.chat.completions.create(
        model="x-ai/grok-3-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Search X/Twitter for relevant posts, threads, and discussions. "
                    "Return the most relevant tweets with usernames, dates, and content. "
                    "Include URLs to the original tweets when possible."
                ),
            },
            {"role": "user", "content": f"Search X/Twitter for: {query}"},
        ],
        max_tokens=4096,
    )
    content = response.choices[0].message.content or ""
    return json.dumps({"platform": "twitter", "results": content}, indent=2)


def execute_tool(name: str, args: dict) -> str:
    try:
        if name == "web_research":
            return _sonar_research(args["query"])
        elif name == "search_images":
            return _searxng_search(args["query"], "images", args.get("limit", 5))
        elif name == "search_videos":
            return _searxng_search(args["query"], "videos", args.get("limit", 5))
        elif name == "search_news":
            return _searxng_search(args["query"], "news", args.get("limit", 5))
        elif name == "search_documents":
            return _searxng_search(args.get("query", "") + " filetype:pdf", "files", args.get("limit", 5))
        elif name == "read_page":
            return _jina_read(args["url"])
        elif name == "get_video_transcript":
            return _youtube_transcript(args["youtube_url"])
        elif name == "search_social":
            return _search_social(args["query"], args["platform"])
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _build_media_appendix(images, videos, news, docs, social=None) -> str:
    """Build a markdown appendix with all collected media, injected server-side."""
    sections = []

    if images:
        lines = ["---", "## Images"]
        for img in images[:8]:
            url = img.get("url", "")
            title = img.get("title", "").replace("[", "").replace("]", "") or "Image"
            source = img.get("source", "")
            if url:
                lines.append(f"![{title}]({url})")
                if source:
                    lines.append(f"*Source: {source}*")
                lines.append("")
        sections.append("\n".join(lines))

    if videos:
        lines = ["---", "## Videos"]
        for vid in videos[:6]:
            url = vid.get("url", "")
            title = vid.get("title", "") or "Video"
            duration = vid.get("duration", "")
            thumb = vid.get("thumbnail", "")
            if url:
                dur_str = f" ({duration})" if duration else ""
                if thumb:
                    lines.append(f"[![{title}]({thumb})]({url})")
                lines.append(f"[{title}{dur_str}]({url})")
                lines.append("")
        sections.append("\n".join(lines))

    if news:
        lines = ["---", "## Recent News"]
        for item in news[:6]:
            url = item.get("url", "")
            title = item.get("title", "") or "Article"
            date = item.get("date", "")
            source = item.get("source", "")
            if url:
                meta = " | ".join(filter(None, [source, date[:10] if date else ""]))
                lines.append(f"- [{title}]({url})" + (f" *({meta})*" if meta else ""))
        sections.append("\n".join(lines))

    if docs:
        lines = ["---", "## Documents & Reports"]
        for doc in docs[:6]:
            url = doc.get("url", "")
            title = doc.get("title", "") or "Document"
            if url:
                lines.append(f"- [{title}]({url})")
        sections.append("\n".join(lines))

    if social:
        lines = ["---", "## Social Media"]
        for item in social:
            if isinstance(item, dict):
                if item.get("platform") == "twitter" and item.get("results"):
                    lines.append(f"### X/Twitter\n{item['results']}")
                elif item.get("url"):
                    platform = item.get("platform", "").title()
                    title = item.get("title", "") or "Post"
                    snippet = item.get("snippet", "")
                    lines.append(f"- [{title}]({item['url']})" + (f" - *{snippet[:100]}*" if snippet else ""))
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def event(kind: str, **data) -> dict:
    return {"type": kind, **data}


TOOL_SOURCE_MAP = {
    "images": "search_images",
    "videos": "search_videos",
    "news": "search_news",
    "docs": "search_documents",
    "social": "search_social",
}


def run(query: str, config: dict = None) -> Generator[dict, None, None]:
    """Run the agent loop, yielding events for each step."""
    config = config or {}
    sources = config.get("sources", {})
    model = config.get("model") or MODEL

    # Filter tools based on UI toggles
    disabled_tools = set()
    for source_key, tool_name in TOOL_SOURCE_MAP.items():
        if sources.get(source_key) is False:
            disabled_tools.add(tool_name)

    active_tools = [t for t in TOOLS if t["function"]["name"] not in disabled_tools]

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    yield event("start", query=query, model=model)

    # Collect media from tool results to append to report
    collected_images = []
    collected_videos = []
    collected_news = []
    collected_docs = []
    collected_social = []

    iteration = 0
    max_iterations = 15

    while iteration < max_iterations:
        iteration += 1
        yield event("thinking", iteration=iteration)

        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            tools=active_tools,
            messages=messages,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                label = args.get("query") or args.get("url") or args.get("youtube_url", "")
                yield event("tool_call", tool=tc.function.name, args=args, label=label[:120])

                result = execute_tool(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

                # Collect media results for the appendix
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, list):
                        yield event("tool_result", tool=tc.function.name, count=len(parsed))
                        if tc.function.name == "search_images":
                            collected_images.extend(parsed)
                        elif tc.function.name == "search_videos":
                            collected_videos.extend(parsed)
                        elif tc.function.name == "search_news":
                            collected_news.extend(parsed)
                        elif tc.function.name == "search_documents":
                            collected_docs.extend(parsed)
                        elif tc.function.name == "search_social":
                            collected_social.extend(parsed)
                    elif "error" in parsed:
                        if tc.function.name == "search_social":
                            collected_social.append(parsed)
                        yield event("tool_error", tool=tc.function.name, error=parsed["error"])
                    else:
                        title = parsed.get("title", "") or parsed.get("video_id", "")
                        yield event("tool_result", tool=tc.function.name, title=title)
                except Exception:
                    yield event("tool_result", tool=tc.function.name)
            continue

        content = msg.content or ""

        # DeepSeek outputs planning text between tool rounds. Detect and redirect.
        is_planning = (
            len(content) < 1000
            and not any(h in content for h in ["## ", "### ", "**Summary**", "**Sources**", "**Key"])
        )

        if is_planning and iteration < max_iterations - 1:
            messages.append(msg)
            messages.append({
                "role": "user",
                "content": "Don't narrate. Use your tools now, then write the final report when done.",
            })
            continue

        # Append collected media to report (don't rely on the model to format these)
        appendix = _build_media_appendix(collected_images, collected_videos, collected_news, collected_docs, collected_social)
        if appendix:
            content = content.rstrip() + "\n\n" + appendix

        yield event("report", content=content)
        yield event("done")
        return

    yield event("error", message="Max iterations reached")
    yield event("done")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run agent.py 'your research query'")
        print(f"Default model: {MODEL}")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    for ev in run(query):
        if ev["type"] == "tool_call":
            print(f"  -> {ev['tool']}({ev['label']})")
        elif ev["type"] == "report":
            print(f"\n{'='*60}")
            print(ev["content"])
        elif ev["type"] == "start":
            print(f"\nResearch: {ev['query']}")
            print(f"Model: {ev['model']}\n")
