# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Awaitable

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from vllm.entrypoints.serve.poison.api_router import get_drain_status

# Paths that must remain reachable even while draining.
_DRAIN_EXEMPT_PATHS: frozenset[str] = frozenset(
    {
        "/dpexit/poison",
        "/dpexit/drain-done",
        "/dpexit/exit",
        "/health",
        "/metrics",
    }
)


class DrainMiddleware:
    """Reject new inference requests with 503 while the server is draining.

    Exempted paths (``/dpexit/*``, ``/health``, ``/metrics``) remain reachable
    so that the drain-coordination protocol and health-checks continue to
    function during shutdown.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] != "http":
            return self.app(scope, receive, send)

        if get_drain_status() != "idle":
            path: str = scope.get("path", "")
            if path not in _DRAIN_EXEMPT_PATHS:
                response = JSONResponse(
                    content={
                        "error": (
                            "Server is draining — not accepting new requests. "
                            "Retry on another instance."
                        )
                    },
                    status_code=503,
                )
                return response(scope, receive, send)

        return self.app(scope, receive, send)
