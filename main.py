import asyncio
import logging
import threading
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from agent import run

log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO)


async def index(request):
    html = Path(__file__).parent.joinpath("index.html").read_text()
    return HTMLResponse(html)


async def ws_research(ws: WebSocket):
    await ws.accept()
    try:
        data = await ws.receive_json()
    except WebSocketDisconnect:
        return

    query = data.get("query", "").strip()
    if not query:
        await ws.send_json({"type": "error", "message": "Empty query"})
        await ws.close()
        return

    config = data.get("config", {})
    log.info(f"Research query: {query} | config: {config}")
    q = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def worker():
        try:
            for ev in run(query, config=config):
                loop.call_soon_threadsafe(q.put_nowait, ev)
        except Exception as e:
            log.error(f"Agent error: {e}", exc_info=True)
            loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "message": str(e)})
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    try:
        while True:
            ev = await q.get()
            if ev is None:
                break
            await ws.send_json(ev)
    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as e:
        log.error(f"WS send error: {e}", exc_info=True)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


app = Starlette(
    routes=[
        Route("/", index),
        WebSocketRoute("/ws", ws_research),
    ],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
