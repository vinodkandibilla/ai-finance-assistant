"""Worker-scoped MCP connection ownership and TTL-cached tool discovery."""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from time import monotonic
from typing import Any, cast

from mcp import ClientSession
from mcp.types import ListToolsResult

from finassist.core.config import Settings

logger = logging.getLogger(__name__)


class _SharedSession:
    def __init__(self, session: ClientSession, stop: asyncio.Event, ttl: float):
        self.session, self.stop, self.ttl = session, stop, ttl
        self.catalog: ListToolsResult | None = None
        self.expires = 0.0
        self.catalog_lock = asyncio.Lock()

    async def list_tools(self) -> ListToolsResult:
        async with self.catalog_lock:
            if self.stop.is_set():
                raise ConnectionError("Market session disconnected")
            if self.catalog is None or monotonic() >= self.expires:
                try:
                    self.catalog = await self.session.list_tools()
                except Exception:
                    self.stop.set()
                    raise
                self.expires = monotonic() + self.ttl
            return self.catalog.model_copy(deep=True)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.stop.is_set():
            raise ConnectionError("Market session disconnected")
        try:
            return await self.session.call_tool(name, arguments)
        except TimeoutError:
            raise  # A request deadline must not tear down everyone else's connection.
        except Exception:
            self.stop.set()
            raise


class MarketSessionManager:
    """One lazy connection per app worker, entered/exited by its own owner task.

    Failed calls are not replayed. A subsequent request reconnects. Request
    cancellation never owns or closes the transport. Notification refresh is deferred.
    """

    def __init__(
        self,
        settings: Settings,
        connection: Callable[[Settings], AbstractAsyncContextManager[ClientSession]],
    ) -> None:
        self.settings, self.connection = settings, connection
        self.lock = asyncio.Lock()
        self.owner: asyncio.Task[None] | None = None
        self.ready: asyncio.Future[_SharedSession] | None = None
        self.stop: asyncio.Event | None = None
        self.closed = False

    async def _own(self, ready: asyncio.Future[_SharedSession], stop: asyncio.Event) -> None:
        try:
            async with self.connection(self.settings) as session:
                ready.set_result(
                    _SharedSession(session, stop, self.settings.market_tool_catalog_ttl_seconds)
                )
                await stop.wait()
        except Exception as error:
            logger.warning("Market MCP connection failed: %s", type(error).__name__)
            if not ready.done():
                ready.set_exception(ConnectionError("Market MCP connection unavailable"))
                ready.exception()  # Consume failure even if the waiting request was cancelled.
        finally:
            stop.set()
            if not ready.done():
                ready.cancel()

    async def get_session(self) -> ClientSession:
        async with self.lock:
            if self.closed:
                raise ConnectionError("Market session manager is closed")
            if self.owner is not None and self.stop is not None and self.stop.is_set():
                await asyncio.shield(self.owner)
            if self.owner is None or self.owner.done():
                self.ready = asyncio.get_running_loop().create_future()
                self.stop = asyncio.Event()
                self.owner = asyncio.create_task(self._own(self.ready, self.stop))
            assert self.ready is not None
            ready = self.ready
        return cast(ClientSession, await asyncio.shield(ready))

    async def aclose(self) -> None:
        async with self.lock:
            self.closed = True
            if self.stop is not None:
                self.stop.set()
            owner = self.owner
        if owner is not None:
            try:
                await asyncio.wait_for(asyncio.shield(owner), timeout=5)
            except TimeoutError:
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)

    @asynccontextmanager
    async def borrow(self) -> AsyncIterator[ClientSession]:
        yield await self.get_session()
