# Research Agent

Multi-source AI research agent. Type a question, get a report with images, videos, news, documents, and social media posts. Single HTML file, no framework, streams results over WebSocket.

## Quick Start

```bash
cp .env.example .env
# Edit .env with your OpenRouter and Serper keys
uv sync
uv run python main.py
# Open http://localhost:8000
```

## What It Does

Type a research query. The agent will:

1. Research the topic using Perplexity Sonar (AI web search with citations)
2. Find related images, videos, news articles, and PDF documents
3. Run a deterministic YouTube-biased video lane before the report is written
4. Build evidence-oriented dossiers for YouTube results with timestamped transcripts, metadata, thumbnails, and verification pivots
5. Optionally search social media (Twitter/X, Facebook, Reddit, Telegram)
6. Produce a structured report with all media embedded inline

Everything streams in real-time. You see each tool call as it happens.

## Tools

| Tool | What it does | Source | Cost |
|------|-------------|--------|------|
| Web Research | AI-powered search with citations | Perplexity Sonar Pro via OpenRouter | Token cost plus Perplexity search request fee |
| Images | Photos, diagrams, visual evidence | Serper | Serper query credits |
| Videos | Video results with thumbnails | Serper + YouTube | Serper query credits |
| Video Dossiers | YouTube metadata, timestamped transcript, thumbnails, verification pivots | YouTube oEmbed + youtube-transcript-api | Free |
| News | Recent news coverage | Serper | Serper query credits |
| Documents | PDFs, reports, papers | Serper | Serper query credits |
| Read Page | Full page content as markdown | Jina Reader | Free |
| Video Transcript | YouTube video transcripts | youtube-transcript-api | Free |
| X/Twitter | Public X posts and discussion search | Grok 4.20 via OpenRouter xAI native web/X search | xAI model tokens plus native search charge |
| Other Social | Facebook, Reddit, Telegram, Instagram search | Serper with site filters | Serper query credits |

## Configuration

`.env` file:

```
OPENROUTER_API_KEY=sk-or-v1-...     # Required. Get one at openrouter.ai
SERPER_API_KEY=...                  # Required for images/videos/news/docs/social. Get one at serper.dev
MODEL=google/gemini-2.5-flash       # Orchestrator model. UI selection overrides this per run.
```

### Orchestrator model

The `MODEL` controls which LLM decides what tools to call and writes the final report. It does NOT do the web research (that's Sonar). Any OpenRouter model works:

```
MODEL=google/gemini-2.5-flash       # Default. Good tool calling, long context, balanced cost.
MODEL=deepseek/deepseek-v4-flash    # Cheapest current DeepSeek V4 option for routine runs.
MODEL=deepseek/deepseek-v4-pro      # Stronger DeepSeek V4 option for hard synthesis.
MODEL=anthropic/claude-haiku-4.5    # Fast Claude option, higher token cost than V4 Flash.
```

The browser model picker currently exposes the same set. Keep `.env.example`, `agent.py`, and `index.html` in sync when changing model IDs.

### Keys and cost model

You need two paid API keys for the full app:

- `OPENROUTER_API_KEY` pays for the orchestrator model, Perplexity Sonar Pro web research, and Grok-powered X/Twitter search.
- `SERPER_API_KEY` pays for Google Search API lanes: images, videos, news, documents, and non-X social search.

Free/no-key lanes:

- Jina Reader is used through `https://r.jina.ai/` for page extraction.
- YouTube oEmbed and `youtube-transcript-api` are used for video dossiers and transcripts when public captions are available.

Approximate public pricing checked May 11, 2026:

- Gemini 2.5 Flash on OpenRouter: $0.30/M input tokens, $2.50/M output tokens.
- DeepSeek V4 Flash on OpenRouter: $0.14/M input tokens, $0.28/M output tokens.
- DeepSeek V4 Pro on OpenRouter: $0.435/M input tokens, $0.87/M output tokens.
- Claude Haiku 4.5 on OpenRouter: $1/M input tokens, $5/M output tokens.
- Perplexity Sonar Pro: $3/M input tokens, $15/M output tokens, plus request fees by search context.
- Grok 4.20 on OpenRouter: $2/M input tokens, $6/M output tokens. Native web/X search may add provider search charges.
- Serper: prepaid credits, currently from about $1.00 per 1,000 queries at the starter tier down to about $0.30 per 1,000 at volume.

Costs vary with query length, number of tool calls, model choice, and provider pricing changes. Check OpenRouter, Perplexity, xAI, and Serper before promising a fixed per-report price.

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
  +-- Serper (images, videos, news, docs, social)
  +-- Jina Reader (page scraping)
  +-- YouTube oEmbed + youtube-transcript-api (video dossiers)
  +-- Grok 4.20 via OpenRouter native web/X search (Twitter/X search)
```

LLM calls go through OpenRouter. Media and non-X social search use Serper. X/Twitter search uses an xAI Grok model with OpenRouter's native web/X search plugin enabled. Media results are collected during the agent loop and injected server-side, so they always render regardless of what the LLM writes.

## Video Evidence Dossiers

Video search runs before the LLM report pass, using the original query plus YouTube/evidence-focused variants. It automatically enriches likely YouTube results with:

- video ID, URL, title, channel/source, publish date, duration
- timestamped transcript when captions are available
- thumbnail URLs and a reverse-image-search pivot
- a downloadable evidence JSON bundle in the UI

## Files

```
agent.py      # Agent logic, tools, LLM loop
main.py       # WebSocket server
index.html    # UI (single file, no framework)
.env          # API keys and config
pyproject.toml
```
