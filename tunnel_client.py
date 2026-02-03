import asyncio
import json
import aiohttp
from typing import Callable, Awaitable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TunnelClient:
    def __init__(
        self,
        tunnel_url: str,
        token: str,
        request_handler: Callable[[dict], Awaitable[dict]],
    ):
        self.tunnel_url = tunnel_url
        self.token = token
        self.request_handler = request_handler
        self.ws = None
        self.running = False
        self.reconnect_delay = 1
        self.max_reconnect_delay = 30

    async def connect(self):
        self.running = True
        while self.running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                if not self.running:
                    break
                logger.error(f"Connection error: {e}")
                logger.info(f"Reconnecting in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

    async def _connect_and_listen(self):
        url = f"{self.tunnel_url}?token={self.token}"
        logger.info(f"Connecting to tunnel: {self.tunnel_url}")

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url, ssl=False) as ws:
                self.ws = ws
                self.reconnect_delay = 1
                logger.info("Connected to tunnel")

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._handle_message(msg.data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {ws.exception()}")
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.info("WebSocket closed")
                        break

    async def _handle_message(self, data: str):
        try:
            message = json.loads(data)

            if message.get("type") == "connected":
                logger.info(f"Tunnel authenticated for user {message.get('userId')}")
                return

            if "id" in message and "method" in message:
                response = await self.request_handler(message)
                response["id"] = message["id"]
                await self.ws.send_str(json.dumps(response))
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            if "id" in message:
                error_response = {
                    "id": message["id"],
                    "status": 500,
                    "headers": {"content-type": "application/json"},
                    "body": json.dumps({"error": str(e)}),
                }
                await self.ws.send_str(json.dumps(error_response))

    def stop(self):
        self.running = False
        if self.ws:
            asyncio.create_task(self.ws.close())


async def create_local_request_handler(app):
    from starlette.testclient import TestClient
    from httpx import ASGITransport, AsyncClient

    async def handler(request: dict) -> dict:
        method = request.get("method", "GET")
        path = request.get("path", "/")
        headers = request.get("headers", {})
        body = request.get("body")

        if headers.get("x-body-encoding") == "base64":
            import base64
            body = base64.b64decode(body)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost"
        ) as client:
            response = await client.request(
                method=method,
                url=path,
                headers=headers,
                content=body if body else None,
            )

            return {
                "status": response.status_code,
                "headers": dict(response.headers),
                "body": response.text,
            }

    return handler
