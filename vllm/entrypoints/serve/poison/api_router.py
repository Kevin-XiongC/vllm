# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
import signal
from typing import TYPE_CHECKING, Literal

import aiohttp
import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient

logger = init_logger(__name__)

router = APIRouter(prefix="/dpexit")

# ---------------------------------------------------------------------------
# Tunable timeouts
# ---------------------------------------------------------------------------
BROADCAST_TIMEOUT_SECS = 5.0
DRAIN_TIMEOUT_SECS = 300.0  # match K8s terminationGracePeriodSeconds


# ---------------------------------------------------------------------------
# DrainCoordinator — encapsulates all per-process drain state
# ---------------------------------------------------------------------------


class DrainCoordinator:
    """Manages the graceful-drain state machine for a single API-server process.

    One module-level singleton is created; all endpoint handlers and public
    convenience functions delegate to it.  This replaces the previous design
    that scattered state across ~11 module-level global variables.
    """

    def __init__(self) -> None:
        # Drain state machine
        self.status: Literal["idle", "wait_all", "terminate"] = "idle"

        # DP peer HTTP URLs — populated on rank 0 after registration
        self.dp_peer_urls: list[str] = []

        # Drain confirmations (rank 0 only): which ranks reported done
        self.drain_confirmations: set[int] = set()

        # Whether rank 0 has already broadcast /dpexit/poison at least once
        self.poison_broadcast_done: bool = False

        # DP peer HTTP registration (rank 0 collects, all ranks register)
        self.dp_peer_registry: dict[int, str] = {}  # rank -> url

        # Context injected at API-server startup
        self.engine_client: EngineClient | None = None
        self.uvicorn_server: uvicorn.Server | None = None
        self.dp_rank: int = 0
        self.dp_size: int = 1
        self.rank0_url: str | None = None

    # ----- startup helpers -----

    def set_context(
        self,
        engine_client: "EngineClient",
        uvicorn_server: "uvicorn.Server",
        dp_rank: int,
        dp_size: int,
        self_url: str | None = None,
        rank0_url: str | None = None,
    ) -> None:
        """Inject runtime context needed by the drain sequence."""
        self.engine_client = engine_client
        self.uvicorn_server = uvicorn_server
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        if rank0_url:
            self.rank0_url = rank0_url
        elif self.dp_peer_urls:
            self.rank0_url = self.dp_peer_urls[0]
        elif self_url:
            self.rank0_url = self_url
        logger.info(
            "Drain context set: dp_rank=%d, dp_size=%d, rank0_url=%s, "
            "peer_urls=%d entries",
            self.dp_rank,
            self.dp_size,
            self.rank0_url,
            len(self.dp_peer_urls),
        )

    def get_status(self) -> Literal["idle", "wait_all", "terminate"]:
        return self.status

    # ----- DP peer HTTP registration -----

    def _finalize_peer_urls(self) -> None:
        """Convert registry dict to sorted list and signal ready."""
        self.dp_peer_urls = [
            self.dp_peer_registry[r] for r in sorted(self.dp_peer_registry)
        ]
        logger.info(
            "All %d DP peers registered: %s",
            len(self.dp_peer_urls),
            self.dp_peer_urls,
        )

    def register_self(self, rank: int, url: str) -> None:
        """Rank 0 registers itself (direct call, no HTTP)."""
        self.dp_peer_registry[rank] = url
        logger.info("Self-registered: rank %d -> %s", rank, url)
        if len(self.dp_peer_registry) >= self.dp_size:
            self._finalize_peer_urls()

    async def register_with_rank0(
        self, rank0_url: str, rank: int, self_url: str
    ) -> None:
        """POST self URL to rank 0's /dpexit/register_dp_peer (with retry)."""
        for attempt in range(60):
            try:
                async with (
                    aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=5.0)
                    ) as session,
                    session.post(
                        f"{rank0_url}/dpexit/register_dp_peer",
                        json={"rank": rank, "url": self_url},
                    ) as resp,
                ):
                    if resp.status == 200:
                        logger.info(
                            "Registered with rank 0 (%s): rank %d -> %s",
                            rank0_url,
                            rank,
                            self_url,
                        )
                        return
                    logger.warning(
                        "Registration returned HTTP %d (attempt %d)",
                        resp.status,
                        attempt,
                    )
            except Exception as exc:
                logger.debug("Registration attempt %d failed: %s", attempt, exc)
            await asyncio.sleep(2.0)
        logger.error(
            "Failed to register with rank 0 (%s) after 60 attempts. "
            "Exiting — this rank cannot participate in coordinated shutdown.",
            rank0_url,
        )
        os._exit(1)

    # ----- core drain logic -----

    async def initiate_drain(self, rank0_url: str | None = None) -> None:
        """Entry point for the drain sequence on any rank.

        Safe to call multiple times -- subsequent calls are no-ops.

        Steps:
          1. Set status = wait_all  (DrainMiddleware starts rejecting new reqs)
          2. Drain in-flight requests via engine.pause_generation(mode="wait")
          3. Set status = terminate
          4. Coordinate exit:
               - dp_size == 1  -> local exit
               - dp_rank == 0  -> count self, call _maybe_broadcast_exit()
               - dp_rank  > 0  -> POST /dpexit/drain-done to rank 0
        """
        if self.status != "idle":
            logger.debug(
                "Already draining (status=%s), ignoring duplicate call.",
                self.status,
            )
            return

        self.status = "wait_all"
        if rank0_url:
            self.rank0_url = rank0_url

        logger.info(
            "Graceful drain started on DP rank %d/%d  (status->wait_all)",
            self.dp_rank,
            self.dp_size,
        )

        # Step 2: drain in-flight requests
        if self.engine_client is not None:
            try:
                await asyncio.wait_for(
                    self.engine_client.pause_generation(mode="wait"),
                    timeout=DRAIN_TIMEOUT_SECS,
                )
                logger.info("Engine drained on DP rank %d.", self.dp_rank)
            except asyncio.TimeoutError:
                logger.warning(
                    "Drain timed out after %.0fs on rank %d -- forcing exit.",
                    DRAIN_TIMEOUT_SECS,
                    self.dp_rank,
                )
            except Exception:
                logger.exception(
                    "Unexpected error while draining on rank %d.", self.dp_rank
                )
        else:
            logger.warning(
                "Rank %d: no engine client available; skipping drain.",
                self.dp_rank,
            )

        self.status = "terminate"
        logger.info("DP rank %d finished draining  (status->terminate).", self.dp_rank)

        # Step 4: coordinate exit
        if self.dp_size <= 1:
            self._do_local_exit()
        elif self.dp_rank == 0:
            self.drain_confirmations.add(0)
            await self._maybe_broadcast_exit()
        else:
            await self._report_done_to_rank0()

    async def _report_done_to_rank0(self) -> None:
        if not self.rank0_url:
            logger.error(
                "Rank %d has no rank-0 URL to report drain-done; forcing local exit.",
                self.dp_rank,
            )
            self._do_local_exit()
            return

        try:
            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30.0)
                ) as session,
                session.post(
                    f"{self.rank0_url}/dpexit/drain-done",
                    json={"rank": self.dp_rank},
                ) as resp,
            ):
                logger.info(
                    "Reported drain-done to rank 0 (%s): HTTP %d",
                    self.rank0_url,
                    resp.status,
                )
        except Exception:
            logger.exception(
                "Failed to report drain-done to rank 0 (%s).",
                self.rank0_url,
            )
        # Exit locally after reporting (or failing to report).
        # Don't wait for /dpexit/exit from rank 0 -- avoids hanging if rank 0 dies.
        self._do_local_exit()

    async def _maybe_broadcast_exit(self) -> None:
        """Rank 0: broadcast /dpexit/exit when every DP rank has confirmed."""
        if len(self.drain_confirmations) < self.dp_size:
            logger.debug(
                "Waiting for remaining ranks: %d/%d confirmed.",
                len(self.drain_confirmations),
                self.dp_size,
            )
            return

        self_url = (
            self.dp_peer_urls[self.dp_rank]
            if self.dp_rank < len(self.dp_peer_urls)
            else None
        )
        other_peers = [u for u in self.dp_peer_urls if u != self_url]
        logger.info(
            "All %d DP ranks drained -- broadcasting /dpexit/exit to %d peers.",
            self.dp_size,
            len(other_peers),
        )
        await self._broadcast_exit(other_peers)
        self._do_local_exit()

    async def _broadcast_exit(self, targets: list[str]) -> None:
        """Send POST /dpexit/exit to every target concurrently."""
        timeout = aiohttp.ClientTimeout(total=BROADCAST_TIMEOUT_SECS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            results = await asyncio.gather(
                *[session.post(f"{t}/dpexit/exit") for t in targets],
                return_exceptions=True,
            )
        for target, result in zip(targets, results):
            if isinstance(result, Exception):
                logger.warning("Failed to send /dpexit/exit to %s: %s", target, result)

    def _do_local_exit(self) -> None:
        """Signal the uvicorn server to finish open connections and exit."""
        if self.uvicorn_server is not None:
            logger.info("Setting uvicorn should_exit=True on rank %d.", self.dp_rank)
            self.uvicorn_server.should_exit = True
        else:
            logger.warning(
                "No uvicorn server reference on rank %d; sending SIGTERM.",
                self.dp_rank,
            )
            loop = asyncio.get_event_loop()
            loop.call_later(0.5, os.kill, os.getpid(), signal.SIGTERM)

    # ----- broadcast helpers -----

    async def broadcast_poison(
        self,
        targets: list[str],
        rank0_url: str | None = None,
    ) -> dict[str, str]:
        """POST /dpexit/poison to all targets, optionally including rank0_url."""
        results: dict[str, str] = {}
        body: dict = {}
        if rank0_url:
            body["rank0_url"] = rank0_url

        timeout = aiohttp.ClientTimeout(total=BROADCAST_TIMEOUT_SECS)
        async with aiohttp.ClientSession(timeout=timeout) as session:

            async def _notify(target: str) -> None:
                try:
                    kw: dict = {"json": body} if body else {}
                    async with session.post(f"{target}/dpexit/poison", **kw) as resp:
                        results[target] = f"ok ({resp.status})"
                except Exception as exc:
                    results[target] = f"error: {exc}"
                    logger.warning(
                        "Failed to broadcast /dpexit/poison to %s: %s", target, exc
                    )

            await asyncio.gather(*[_notify(t) for t in targets])
        return results

    # ----- signal handler support -----

    def handle_shutdown_signal(self, loop: asyncio.AbstractEventLoop) -> None:
        """Handle first SIGTERM/SIGINT: broadcast poison to peers + drain.

        Called from the signal handler in launcher.py.  Encapsulates the
        broadcast + drain initiation so the caller needs no internal state.
        """
        if self.dp_rank == 0 and self.dp_peer_urls:
            self_url = (
                self.dp_peer_urls[self.dp_rank]
                if self.dp_rank < len(self.dp_peer_urls)
                else None
            )
            other_peers = [u for u in self.dp_peer_urls if u != self_url]
            if other_peers:
                loop.create_task(self.broadcast_poison(other_peers, rank0_url=self_url))
        loop.create_task(self.initiate_drain())


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_coordinator = DrainCoordinator()


# ---------------------------------------------------------------------------
# Public convenience functions (delegate to _coordinator)
# ---------------------------------------------------------------------------


def set_drain_context(
    engine_client: "EngineClient",
    uvicorn_server: "uvicorn.Server",
    dp_rank: int,
    dp_size: int,
    self_url: str | None = None,
    rank0_url: str | None = None,
) -> None:
    """Inject runtime context needed by the drain sequence."""
    _coordinator.set_context(
        engine_client=engine_client,
        uvicorn_server=uvicorn_server,
        dp_rank=dp_rank,
        dp_size=dp_size,
        self_url=self_url,
        rank0_url=rank0_url,
    )


def get_drain_status() -> Literal["idle", "wait_all", "terminate"]:
    return _coordinator.get_status()


def register_self(rank: int, url: str) -> None:
    """Rank 0 registers itself (direct call, no HTTP)."""
    _coordinator.register_self(rank, url)


async def register_with_rank0(rank0_url: str, rank: int, self_url: str) -> None:
    """POST self URL to rank 0's /dpexit/register_dp_peer (with retry)."""
    await _coordinator.register_with_rank0(rank0_url, rank, self_url)


async def initiate_drain(rank0_url: str | None = None) -> None:
    """Start the drain sequence on this rank."""
    await _coordinator.initiate_drain(rank0_url=rank0_url)


def handle_shutdown_signal(loop: asyncio.AbstractEventLoop) -> None:
    """Handle first shutdown signal: broadcast poison + start drain."""
    _coordinator.handle_shutdown_signal(loop)


# ---------------------------------------------------------------------------
# HTTP Endpoints (access state via _coordinator)
# ---------------------------------------------------------------------------


@router.post("/register_dp_peer")
async def register_dp_peer_endpoint(raw_request: Request):
    """Accept DP peer URL registration (rank 0 only)."""
    if _coordinator.dp_rank != 0:
        return JSONResponse(
            content={"error": "Only rank 0 accepts peer registration"},
            status_code=403,
        )

    body = await raw_request.json()
    rank: int | None = body.get("rank")
    url: str | None = body.get("url")
    if rank is None or url is None:
        return JSONResponse(
            content={"error": "Missing 'rank' or 'url' in request body"},
            status_code=400,
        )

    _coordinator.dp_peer_registry[rank] = url
    logger.info(
        "DP peer registered: rank %d -> %s  (%d/%d)",
        rank,
        url,
        len(_coordinator.dp_peer_registry),
        _coordinator.dp_size,
    )

    if len(_coordinator.dp_peer_registry) >= _coordinator.dp_size:
        _coordinator._finalize_peer_urls()

    return JSONResponse(
        content={
            "registered": len(_coordinator.dp_peer_registry),
            "total": _coordinator.dp_size,
        }
    )


@router.post("/poison")
async def poison(raw_request: Request):
    """Trigger graceful drain shutdown (coordinator-based poison pill).

    Behaviour by rank:

    * **Rank 0** (has peer URLs): broadcasts ``/dpexit/poison`` to all peers
      -- passing ``rank0_url`` so they know where to report ``/dpexit/drain-done``
      -- then starts its own drain as a background task.

    * **Any rank** receiving ``/dpexit/poison``: records ``rank0_url`` (if present),
      starts local drain asynchronously, and returns immediately.

    * Explicit body ``{"targets": [...]}`` overrides the auto-broadcast list.
    """
    c = _coordinator

    # Parse optional body
    rank0_url_override: str | None = None
    explicit_targets: list[str] = []
    ct = raw_request.headers.get("content-type", "")
    if ct.startswith("application/json"):
        body = await raw_request.json()
        explicit_targets = body.get("targets", [])
        rank0_url_override = body.get("rank0_url")

    if rank0_url_override:
        c.rank0_url = rank0_url_override

    # If already draining, skip broadcast and return immediately.
    if c.status != "idle":
        logger.debug(
            "Already draining (status=%s), ignoring duplicate /dpexit/poison.",
            c.status,
        )
        return JSONResponse(
            content={
                "message": "Already draining",
                "status": c.status,
                "broadcast": {},
            }
        )

    # Determine who to broadcast to (excluding self to avoid amplification)
    self_url = c.dp_peer_urls[c.dp_rank] if c.dp_rank < len(c.dp_peer_urls) else None
    if explicit_targets:
        broadcast_targets = explicit_targets
    elif c.dp_rank == 0 and c.dp_peer_urls:
        broadcast_targets = [u for u in c.dp_peer_urls if u != self_url]
    else:
        broadcast_targets = []
    broadcast_results: dict[str, str] = {}

    if broadcast_targets:
        c.poison_broadcast_done = True
        self_rank0_url = c.dp_peer_urls[0] if c.dp_peer_urls else None
        logger.info(
            "Broadcasting /dpexit/poison to %d peers (rank0_url=%s)",
            len(broadcast_targets),
            self_rank0_url,
        )
        broadcast_results = await c.broadcast_poison(
            broadcast_targets, rank0_url=self_rank0_url
        )
        logger.info("Broadcast complete: %s", broadcast_results)

    # Drain runs in background; response is returned immediately.
    asyncio.create_task(c.initiate_drain())

    return JSONResponse(
        content={
            "message": "Drain initiated",
            "status": "wait_all",
            "broadcast": broadcast_results,
        }
    )


@router.post("/drain-done")
async def drain_done(raw_request: Request):
    """Receive drain-complete notification from a non-rank-0 DP peer.

    Only meaningful on rank 0.  Records the confirmation and triggers
    coordinated exit once all DP ranks have checked in.
    """
    c = _coordinator

    if c.dp_rank != 0:
        return JSONResponse(
            content={"error": "Only rank 0 accepts /dpexit/drain-done"},
            status_code=403,
        )

    body = await raw_request.json()
    rank: int | None = body.get("rank")
    if rank is None:
        return JSONResponse(
            content={"error": "Missing 'rank' in request body"},
            status_code=400,
        )
    if not isinstance(rank, int) or not (0 <= rank < c.dp_size):
        return JSONResponse(
            content={"error": f"Invalid rank {rank!r} (dp_size={c.dp_size})"},
            status_code=400,
        )

    c.drain_confirmations.add(rank)
    logger.info(
        "drain-done from rank %d  (%d/%d confirmed)",
        rank,
        len(c.drain_confirmations),
        c.dp_size,
    )

    # Ensure all peers are draining. This handles the case where a
    # non-rank-0 peer received SIGTERM and drained without a prior
    # /dpexit/poison broadcast. Only broadcast once to avoid amplification.
    # Exclude rank 0's own URL -- rank 0 starts its own drain directly.
    if c.dp_peer_urls and not c.poison_broadcast_done:
        c.poison_broadcast_done = True
        self_url = c.dp_peer_urls[0] if c.dp_peer_urls else None
        other_peers = [u for u in c.dp_peer_urls if u != self_url]
        if other_peers:
            asyncio.create_task(c.broadcast_poison(other_peers, rank0_url=self_url))
        # Start rank 0's own drain directly (not via /dpexit/poison to self).
        asyncio.create_task(c.initiate_drain())

    asyncio.create_task(c._maybe_broadcast_exit())

    return JSONResponse(
        content={
            "message": f"Acknowledged drain-done from rank {rank}",
            "confirmed": len(c.drain_confirmations),
            "total": c.dp_size,
        }
    )


@router.post("/exit")
async def exit_handler(raw_request: Request):
    """Clean-exit signal from rank 0 after all peers have drained."""
    logger.info(
        "Received /dpexit/exit on rank %d -- shutting down.", _coordinator.dp_rank
    )
    _coordinator._do_local_exit()
    return JSONResponse(content={"message": "Shutdown initiated"})


def attach_router(app: FastAPI):
    app.include_router(router)
