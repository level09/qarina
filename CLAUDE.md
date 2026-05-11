# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                              # install deps
uv run python main.py                # run server on :8000 (reload enabled)
uv run python -m unittest discover tests   # run all tests
uv run python -m unittest tests.test_video_dossiers.VideoDossierTests.test_extract_video_id_supports_common_youtube_urls   # run one test
uv run ruff check .                  # lint (ruff_cache exists; project uses ruff)
```

Required env vars in `.env`: `OPENROUTER_API_KEY`, `SERPER_API_KEY`, optional `MODEL` (orchestrator), `KNOWLEDGE_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `KNOWLEDGE_DIR`, `HISTORY_DB`.

## Architecture

Single-process Starlette app. Browser opens `index.html`, opens WebSocket to `/ws`, sends `{query, config}`, then receives a stream of typed events that render progressively. **There is no framework on the frontend** — `index.html` is one file with vanilla JS + marked.js for markdown.

Request flow:

```
index.html  --WS-->  main.py (ws_research)
                       └─ spawns thread running agent.run(query, config)
                          which is a generator yielding event dicts.
                          loop.call_soon_threadsafe pushes events back to WS.
```

`agent.run()` is the heart of the system. It is a **generator** (not async) that yields event dicts. Phases:

1. **Parallel prep** (ThreadPoolExecutor): `_generate_followups`, `knowledge.get_prior_knowledge`, `_generate_plan`.
2. **Deterministic video prefetch** — `_prefetch_video_evidence` runs Serper video search with YouTube-biased query variants *before* the LLM loop, enriches top 3 YouTube results with `_analyze_video_url` (oEmbed metadata + `youtube-transcript-api` transcript + thumbnail pivots), and emits `media`/`video_dossiers` events. After this, `search_videos` is removed from the LLM's tool list so it doesn't redundantly call it.
3. **LLM tool-calling loop** (`_run_agent_loop`) — OpenAI SDK pointed at OpenRouter. Each tool call/result is emitted as an event so the UI can render it live.
4. **Media appendix** — `_build_media_appendix` injects collected media into the report server-side, so media always renders regardless of what the LLM wrote.
5. **Persistence** — saves to `history.db` (SQLite via `history.py`) and indexes report into LightRAG (`knowledge.py`) in a background thread.

### Tool routing

Tools defined in `TOOLS` list, dispatched in `execute_tool()`:

- `web_research` → Perplexity Sonar Pro via OpenRouter (`_sonar_research`)
- `search_images` / `search_videos` / `search_news` / `search_documents` → Serper Google Search API
- `search_social` with `platform=twitter` → **Grok 4.20 via OpenRouter with native X/web search plugin** (`_grok_x_search`). Other platforms → Serper site-filtered search.
- `read_page` → Jina Reader (`https://r.jina.ai/<url>`, no key)
- `get_video_transcript` / `analyze_video_url` → `youtube-transcript-api` + YouTube oEmbed

`TOOL_SOURCE_MAP` maps UI source toggles to tool names; disabled sources are stripped from `active_tools` before the LLM loop.

### Knowledge store (LightRAG)

`knowledge.py` runs LightRAG on a private background asyncio event loop (because LightRAG is async-only but the agent is sync). All access goes through `_run(coro)` which schedules via `run_coroutine_threadsafe`. Init is lazy + lock-guarded + fail-once (`_init_failed` flag). LLM/embeddings both route through OpenRouter. If init fails, knowledge silently no-ops — never raises into the agent path.

`get_prior_knowledge` filters out LightRAG's "no relevant data" filler responses (checks for "sorry", "i don't have", etc.) before returning.

### Event types emitted by `agent.run()`

`start`, `phase`, `followups`, `prior_knowledge`, `plan`, `tool_call`, `tool_result`, `tool_error`, `media` (images/videos/news/docs/social), `video_dossiers`, `markdown` (streamed report chunks), `done`, `error`. The frontend dispatches on `type`.

### Frontend routing

`main.py` routes both `/` and `/research/{id}` to the same `index.html`. The HTML reads its own URL to either start a new query or load `/api/history/{id}`. Tab persistence and browser history are handled client-side.

## Conventions specific to this repo

- **Keep `.env.example`, the `MODEL` default in `agent.py`, and the model picker in `index.html` in sync** when changing model IDs. Mentioned in README; easy to forget.
- Media injection is server-side and authoritative — do not move it into the LLM prompt. The LLM only writes prose; media is appended deterministically.
- `_prefetch_video_evidence` runs before the LLM loop intentionally. Don't make video search LLM-only — the deterministic lane is the product's core evidence guarantee.
- New tools: add to `TOOLS`, add a branch in `execute_tool`, add to `TOOL_SOURCE_MAP` if it should be UI-toggleable, and emit `media` events if it collects renderable items.
- The agent yields plain dicts; serialization happens in `main.py`. Don't `json.dumps` inside the generator.
- `history.py` uses `threading.local` for SQLite connections — every thread that touches history gets its own connection. Don't share connections across threads.
