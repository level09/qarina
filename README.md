# Research Agent

Multi-source AI research agent. Type a question, get a report with images, videos, news, documents, and social media posts. Single HTML file, no framework, streams results over WebSocket.

## Quick Start

```bash
cp .env.example .env
# Edit .env with your OpenRouter API key
uv sync
uv run python main.py
# Open http://localhost:8000
```

## What It Does

Type a research query. The agent will:

1. Research the topic using Perplexity Sonar (AI web search with citations)
2. Find related images, videos, news articles, and PDF documents
3. Optionally search social media (Twitter/X, Facebook, Reddit, Telegram)
4. Produce a structured report with all media embedded inline

Everything streams in real-time. You see each tool call as it happens.

## Tools

| Tool | What it does | Source | Cost |
|------|-------------|--------|------|
| Web Research | AI-powered search with citations | Perplexity Sonar via OpenRouter | ~$0.01/query |
| Images | Photos, diagrams, visual evidence | SearXNG | Free |
| Videos | Video results with thumbnails | SearXNG + YouTube | Free |
| News | Recent news coverage | SearXNG | Free |
| Documents | PDFs, reports, papers | SearXNG | Free |
| Read Page | Full page content as markdown | Jina Reader | Free |
| Video Transcript | YouTube video transcripts | youtube-transcript-api | Free |
| Social Media | Twitter/X, Facebook, Reddit, Telegram | Grok (Twitter) / SearXNG (rest) | ~$0.005 for Twitter, free for rest |

## Configuration

`.env` file:

```
OPENROUTER_API_KEY=sk-or-v1-...    # Required. Get one at openrouter.ai
MODEL=deepseek/deepseek-chat        # Orchestrator model (decides which tools to call)
SEARXNG_URL=http://your-server:8888 # SearXNG instance for search
```

### Orchestrator model

The `MODEL` controls which LLM decides what tools to call and writes the final report. It does NOT do the web research (that's Sonar). Any OpenRouter model works:

```
MODEL=deepseek/deepseek-chat        # Default. Cheap, good enough.
MODEL=google/gemini-2.5-flash       # Better tool calling, parallel calls.
MODEL=anthropic/claude-haiku        # Most instruction-obedient.
```

## Architecture

```
Browser (index.html)
  |
  WebSocket
  |
main.py (Starlette server)
  |
agent.py (tool-calling loop)
  |
  +-- Perplexity Sonar (web research)
  +-- SearXNG (images, videos, news, docs, social)
  +-- Jina Reader (page scraping)
  +-- youtube-transcript-api (transcripts)
  +-- Grok (Twitter/X search)
```

All LLM calls go through OpenRouter with a single API key. SearXNG is self-hosted (free, unlimited). Media results are collected during the agent loop and injected server-side, so they always render regardless of what the LLM writes.

## SearXNG Setup

Deploy SearXNG via Docker on any server:

```bash
# See https://docs.searxng.org/admin/installation-docker.html
# Expose on port 8888, enable JSON format, disable rate limiter for API use
```

Set `SEARXNG_URL` in `.env` to point to your instance.

## Files

```
agent.py      # Agent logic, tools, LLM loop
main.py       # WebSocket server
index.html    # UI (single file, no framework)
.env          # API keys and config
pyproject.toml
```
