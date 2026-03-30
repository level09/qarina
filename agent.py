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
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

import httpx
from openai import OpenAI
from dotenv import load_dotenv

import history
import knowledge

load_dotenv()

MODEL = os.environ.get("MODEL", "google/gemini-2.5-flash")
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


def _generate_plan(client, model: str, query: str) -> str:
    """Generate a research plan before executing."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research planner. Given a query, output a brief research plan "
                    "as a numbered list (3-5 steps). Each step should be one sentence. "
                    "Focus on WHAT you'll search for, not HOW. No preamble, just the list."
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content or ""


def _generate_followups(client, model: str, query: str) -> list[str]:
    """Generate clarifying questions before research."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[
            {
                "role": "system",
                "content": (
                    "You help refine research queries. Given a query, generate exactly 3 short "
                    "follow-up questions that would help narrow the research. Format: one question "
                    "per line, no numbering, no bullets. Keep each under 60 characters."
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    text = response.choices[0].message.content or ""
    return [q.strip() for q in text.strip().split("\n") if q.strip()][:3]


def _run_agent_loop(client, model, messages, active_tools, collected, events):
    """Run the core tool-calling loop. Appends events to the list, returns report content."""
    iteration = 0
    max_iterations = 12

    while iteration < max_iterations:
        iteration += 1
        events.append(event("thinking", iteration=iteration))

        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            tools=active_tools,
            messages=messages,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            key_map = {
                "search_images": "images",
                "search_videos": "videos",
                "search_news": "news",
                "search_documents": "docs",
                "search_social": "social",
            }

            # Log all tool calls first
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                label = args.get("query") or args.get("url") or args.get("youtube_url", "")
                events.append(event("tool_call", tool=tc.function.name, args=args, label=label[:120]))

            # Execute tools in parallel
            def _exec(tc):
                args = json.loads(tc.function.arguments)
                return tc, execute_tool(tc.function.name, args)

            with ThreadPoolExecutor(max_workers=6) as pool:
                results = list(pool.map(_exec, msg.tool_calls))

            for tc, result in results:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, list):
                        events.append(event("tool_result", tool=tc.function.name, count=len(parsed)))
                        if tc.function.name in key_map:
                            collected[key_map[tc.function.name]].extend(parsed)
                    elif "error" in parsed:
                        if tc.function.name == "search_social":
                            collected["social"].append(parsed)
                        events.append(event("tool_error", tool=tc.function.name, error=parsed["error"]))
                    else:
                        title = parsed.get("title", "") or parsed.get("video_id", "")
                        events.append(event("tool_result", tool=tc.function.name, title=title))
                except Exception:
                    events.append(event("tool_result", tool=tc.function.name))
            continue

        content = msg.content or ""

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

        return content

    return ""


def run(query: str, config: dict = None) -> Generator[dict, None, None]:
    """Run the full research pipeline with plan, questions, research, and gap analysis."""
    config = config or {}
    sources = config.get("sources", {})
    model = config.get("model") or MODEL

    disabled_tools = set()
    for source_key, tool_name in TOOL_SOURCE_MAP.items():
        if sources.get(source_key) is False:
            disabled_tools.add(tool_name)

    active_tools = [t for t in TOOLS if t["function"]["name"] not in disabled_tools]

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    yield event("start", query=query, model=model)

    # Phase 1: Run followups, knowledge check, and plan in parallel
    yield event("phase", name="Preparing research...")

    followups = []
    prior_knowledge = None
    plan = ""

    def _get_followups():
        nonlocal followups
        try:
            followups = _generate_followups(client, model, query)
        except Exception:
            pass

    def _get_knowledge():
        nonlocal prior_knowledge
        try:
            prior_knowledge = knowledge.get_prior_knowledge(query)
        except Exception:
            pass

    def _get_plan():
        nonlocal plan
        try:
            plan = _generate_plan(client, model, query)
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=3) as pool:
        pool.submit(_get_followups)
        pool.submit(_get_knowledge)
        f_plan = pool.submit(_get_plan)
        f_plan.result()  # wait for all to finish

    if followups:
        yield event("followups", questions=followups)
    if prior_knowledge:
        yield event("prior_knowledge", found=True, summary=prior_knowledge[:500])
    else:
        yield event("prior_knowledge", found=False)
    if plan:
        yield event("plan", content=plan)

    # Phase 3: Execute research
    yield event("phase", name="Researching...")

    # Build dynamic system prompt based on active tools
    active_tool_names = {t["function"]["name"] for t in active_tools}
    tool_lines = []
    for t in active_tools:
        name = t["function"]["name"]
        desc = t["function"]["description"].split(".")[0]
        tool_lines.append(f"- **{name}**: {desc}.")

    must_use = []
    if "search_images" in active_tool_names:
        must_use.append("search_images")
    if "search_videos" in active_tool_names:
        must_use.append("search_videos")
    if "search_news" in active_tool_names:
        must_use.append("search_news")
    if "search_documents" in active_tool_names:
        must_use.append("search_documents")

    must_use_str = ", ".join(must_use)
    must_use_instruction = f"\nYou MUST use at least web_research and {must_use_str} before writing your report." if must_use else ""

    dynamic_prompt = f"""\
You are a deep research agent. Your job is to find, extract, and synthesize \
information from the web on any topic the user asks about.

Your available tools:
{chr(10).join(tool_lines)}

Use web_research as your PRIMARY tool for factual investigation.{must_use_instruction}

When done, write a structured report with:
- **Summary**: Key findings in 2-3 sentences
- **Sources**: URLs with what each contributed
- **Key Findings**: Organized by theme
- **Gaps**: What's still missing or unverified

Do NOT include images, videos, or documents sections. Those are appended automatically."""

    if prior_knowledge:
        dynamic_prompt += (
            "\n\n## Prior Research Context\n"
            "You have relevant knowledge from previous research sessions:\n"
            f"{prior_knowledge}\n\n"
            "Use this as context but verify key claims with fresh sources."
        )

    messages = [
        {"role": "system", "content": dynamic_prompt},
        {"role": "user", "content": query},
    ]

    collected = {"images": [], "videos": [], "news": [], "docs": [], "social": []}

    report_content = _run_agent_loop(client, model, messages, active_tools, collected, event_sink := [])
    for ev in event_sink:
        yield ev

    # Phase 4: Gap analysis and second pass
    if report_content and "Gaps" in report_content:
        yield event("phase", name="Analyzing gaps, doing follow-up research...")
        messages.append({"role": "assistant", "content": report_content})
        messages.append({
            "role": "user",
            "content": (
                "Look at the gaps you identified. Do ONE more round of targeted research "
                "to fill the most important gaps. Use your tools, then rewrite the full report "
                "incorporating the new findings. Keep the same structure."
            ),
        })
        report_content = _run_agent_loop(client, model, messages, active_tools, collected, event_sink2 := [])
        for ev in event_sink2:
            yield ev

    # Backfill: force-call any enabled media tools that returned nothing
    backfill = {
        "search_images": "images",
        "search_videos": "videos",
        "search_news": "news",
        "search_documents": "docs",
    }
    for tool_name, key in backfill.items():
        if not collected[key] and tool_name not in disabled_tools:
            yield event("tool_call", tool=tool_name, args={"query": query}, label=f"(backfill) {query[:80]}")
            try:
                result = execute_tool(tool_name, {"query": query, "limit": 5})
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    collected[key].extend(parsed)
                    yield event("tool_result", tool=tool_name, count=len(parsed))
            except Exception:
                pass

    # Index report into knowledge graph (background, don't block response)
    if report_content:
        threading.Thread(
            target=lambda: knowledge.index_report(query, report_content),
            daemon=True,
        ).start()

    # Append media
    appendix = _build_media_appendix(
        collected["images"], collected["videos"], collected["news"],
        collected["docs"], collected["social"]
    )
    if appendix:
        report_content = (report_content or "").rstrip() + "\n\n" + appendix

    yield event("report", content=report_content or "No results found.")

    # Save to history
    try:
        session_id = history.save(query, report_content or "", model, sources)
        yield event("saved", session_id=session_id)
    except Exception:
        pass

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
