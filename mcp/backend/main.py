"""
CalmCrypto MCP Server - FastMCP implementation.

Provides MCP tools for cryptocurrency signal evaluation and price prediction.
"""

import contextlib
import contextvars
import logging
import os
import re
import pathlib
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount
from mcp.server.fastmcp import FastMCP
import uvicorn

# Tool registrations
from mcp_service import register_py_eval, register_tool_notes, register_request_log
from calmcrypto_tools import (
    register_list_assets,
    register_benchmark_all_assets,
    register_signal_eval,
    register_predict_price,
)

load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MCP_TOKEN_CTX = contextvars.ContextVar("mcp_token", default=None)

# Initialize FastMCP using MCP_NAME (env) for tool name and base path
MCP_NAME = os.getenv("MCP_NAME", "calmcrypto")
_safe_name = re.sub(r"[^a-z0-9_-]", "-", MCP_NAME.lower()).strip("-") or "service"
BASE_PATH = f"/{_safe_name}"
STREAM_PATH = f"{BASE_PATH}/"
logger.info(f"Safe service name: {_safe_name}")
logger.info(f"Stream path: {STREAM_PATH}")

# Initialize FastMCP
mcp = FastMCP(_safe_name, streamable_http_path=STREAM_PATH, json_response=True)

# CSV storage directory
CSV_DIR = pathlib.Path("data/mcp-calmcrypto")
CSV_DIR.mkdir(parents=True, exist_ok=True)

# Request logging directory
REQUESTS_DIR = CSV_DIR / "requests"
REQUESTS_DIR.mkdir(parents=True, exist_ok=True)

# Register CalmCrypto tools
register_list_assets(mcp, CSV_DIR, REQUESTS_DIR)
register_benchmark_all_assets(mcp, CSV_DIR, REQUESTS_DIR)
register_signal_eval(mcp, CSV_DIR, REQUESTS_DIR)
register_predict_price(mcp, CSV_DIR, REQUESTS_DIR)

# Register utility tools
register_py_eval(mcp, CSV_DIR, REQUESTS_DIR)
register_tool_notes(mcp, CSV_DIR, REQUESTS_DIR)
register_request_log(mcp, CSV_DIR, REQUESTS_DIR)

# Add custom error handling for stream disconnections
original_logger = logging.getLogger("mcp.server.streamable_http")


class StreamErrorFilter(logging.Filter):
    def filter(self, record):
        if "ClosedResourceError" in str(record.getMessage()):
            return False
        return True


original_logger.addFilter(StreamErrorFilter())

# Build the main ASGI app with Streamable HTTP mounted
mcp_asgi = mcp.streamable_http_app()


@contextlib.asynccontextmanager
async def lifespan(_: Starlette):
    async with mcp.session_manager.run():
        yield


async def health_check(request):
    return JSONResponse({"status": "healthy", "service": "calmcrypto-mcp"})


app = Starlette(
    routes=[
        Mount("/", app=mcp_asgi),
    ],
    lifespan=lifespan,
)

# Add health endpoint before auth middleware
app.routes.insert(0, Route("/health", health_check, methods=["GET"]))


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Simple token gate for all service requests under BASE_PATH.

    Accepts tokens via Authorization header: "Bearer <token>" (default and recommended).
    If env MCP_ALLOW_URL_TOKENS=true, also accepts:
      - Query parameter: ?token=<token>
      - URL path form: {BASE_PATH}/<token>/... (token is stripped before forwarding)

    Configure allowed tokens via env var MCP_TOKENS (comma-separated). If unset or empty,
    authentication is disabled (allows all) but logs a warning.
    """

    def __init__(self, app):
        super().__init__(app)
        raw = os.getenv("MCP_TOKENS", "")
        self.allowed_tokens = {t.strip() for t in raw.split(",") if t.strip()}
        self.allow_url_tokens = (
            os.getenv("MCP_ALLOW_URL_TOKENS", "").lower()
            in ("1", "true", "yes")
        )
        self.require_auth = (
            os.getenv("MCP_REQUIRE_AUTH", "").lower()
            in ("1", "true", "yes")
        )
        if not self.allowed_tokens:
            if self.require_auth:
                logger.warning(
                    "MCP_TOKENS is not set; MCP_REQUIRE_AUTH=true -> all %s requests will be rejected (401)",
                    BASE_PATH,
                )
            else:
                logger.warning(
                    "MCP_TOKENS is not set; token auth is DISABLED for %s endpoints",
                    BASE_PATH,
                )

    async def dispatch(self, request, call_next):
        path = request.url.path or "/"
        if not path.startswith(BASE_PATH):
            return await call_next(request)

        def accept(token_value: str, source: str):
            request.state.mcp_token = token_value
            logger.info(
                "Authenticated %s %s via %s token %s",
                request.method,
                path,
                source,
                token_value,
            )
            return MCP_TOKEN_CTX.set(token_value)

        async def proceed(token_value: str, source: str):
            token_scope = accept(token_value, source)
            try:
                return await call_next(request)
            finally:
                MCP_TOKEN_CTX.reset(token_scope)

        # If auth is not required, strip any token-like path segment and allow
        if not self.require_auth:
            segs = [s for s in path.split("/") if s != ""]
            if len(segs) >= 2 and segs[0] == _safe_name:
                remainder = "/".join([_safe_name] + segs[2:])
                new_path = "/" + (remainder + "/" if path.endswith("/") or not segs[2:] else remainder)
                if new_path == BASE_PATH:
                    new_path = STREAM_PATH
                request.scope["path"] = new_path
                if "raw_path" in request.scope:
                    request.scope["raw_path"] = new_path.encode("utf-8")
                logger.info("Auth disabled, rewriting path %s -> %s", path, new_path)
            else:
                logger.info("Auth disabled, allowing request to %s", path)
            return await call_next(request)

        # If no tokens configured but auth is required
        if not self.allowed_tokens:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"})

        # Authorization: Bearer <token>
        token = None
        auth = request.headers.get("authorization") or request.headers.get("Authorization")
        if auth and auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()

        # Header token valid -> allow
        if token and token in self.allowed_tokens:
            return await proceed(token, "header")

        # If URL tokens are allowed, check query and path variants
        if self.allow_url_tokens:
            url_token = request.query_params.get("token")
            if url_token and url_token in self.allowed_tokens:
                return await proceed(url_token, "query")

            segs = [s for s in path.split("/") if s != ""]
            if len(segs) >= 2 and segs[0] == _safe_name:
                candidate = segs[1]
                if candidate in self.allowed_tokens:
                    remainder = "/".join([_safe_name] + segs[2:])
                    new_path = "/" + (remainder + "/" if path.endswith("/") and not remainder.endswith("/") else remainder)
                    if new_path == BASE_PATH:
                        new_path = STREAM_PATH
                    request.scope["path"] = new_path
                    if "raw_path" in request.scope:
                        request.scope["raw_path"] = new_path.encode("utf-8")
                    return await proceed(candidate, "path")

        # Reject unauthorized
        if self.allow_url_tokens:
            detail = "Unauthorized"
        else:
            detail = "Use Authorization: Bearer <token>; URL/query tokens are not allowed"
        return JSONResponse({"detail": detail}, status_code=401, headers={"WWW-Authenticate": "Bearer"})


# Install auth middleware last to wrap the full app
app.add_middleware(TokenAuthMiddleware)


def main():
    """Run the uvicorn server."""
    PORT = int(os.getenv("PORT", "8007"))

    logger.info(f"Starting {MCP_NAME} MCP server (HTTP) on port {PORT} at {STREAM_PATH}")

    uvicorn.run(
        app=app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=PORT,
        log_level=os.getenv("LOG_LEVEL", "info"),
        access_log=True,
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_keep_alive=120,
    )


if __name__ == "__main__":
    main()
