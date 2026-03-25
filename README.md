# Bayanat Research Agent

AI-powered research tool for human rights investigations. Single HTML interface, WebSocket streaming, multi-source search.

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

Everything streams in real-time over WebSocket. You see each tool call as it happens.

## Tools

| Tool | What it does | Source | Cost |
|------|-------------|--------|------|
| Web Research | AI-powered search with citations | Perplexity Sonar via OpenRouter | ~$0.01/query |
| Images | Photos, satellite imagery, visual evidence | SearXNG | Free |
| Videos | Video evidence, testimonies, documentaries | SearXNG + YouTube thumbnails | Free |
| News | Recent news coverage | SearXNG | Free |
| Documents | PDFs, reports, legal documents | SearXNG | Free |
| Read Page | Full page content as markdown | Jina Reader | Free |
| Video Transcript | YouTube video transcripts | youtube-transcript-api | Free |
| Social Media | Twitter/X, Facebook, Reddit, Telegram | Grok (Twitter) / SearXNG (rest) | ~$0.005 for Twitter, free for rest |

## Configuration

`.env` file:

```
OPENROUTER_API_KEY=sk-or-v1-...    # Required. Get one at openrouter.ai
MODEL=deepseek/deepseek-chat        # Orchestrator model (decides which tools to call)
SEARXNG_URL=http://searxng.example.com:8888  # SearXNG instance for search
```

### Changing the orchestrator model

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

All LLM calls go through OpenRouter with a single API key. SearXNG is self-hosted (free, unlimited). Media results (images, videos, news, documents, social posts) are collected during the agent loop and injected into the report server-side, so they always render regardless of what the LLM writes.

## Tech Stack

- **Backend**: Python, Starlette, uvicorn
- **Frontend**: Single HTML file, vanilla JS, WebSocket, no build step
- **Search**: SearXNG (self-hosted on Hetzner)
- **LLMs**: OpenRouter (DeepSeek, Sonar, Grok)
- **Scraping**: Jina Reader (free, no API key)

## Files

```
agent.py      # Agent logic, tools, LLM loop
main.py       # WebSocket server
index.html    # UI (single file, no framework)
.env          # API keys and config
pyproject.toml
```

## Running in Production

```bash
uv run python main.py
# Runs on 0.0.0.0:8000 with auto-reload
```

For production, put it behind nginx/caddy with HTTPS and basic auth.
