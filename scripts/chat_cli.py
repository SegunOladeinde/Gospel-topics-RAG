"""Interactive command-line chat client for the LDS Doctrinal RAG backend.

Usage
-----
From the project root (virtual environment managed by uv):

    uv run python scripts/chat_cli.py

The backend server must already be running on http://localhost:8000.
Start it (in a separate terminal) with:

    uv run uvicorn src.backend.main:app --reload --port 8000

Type your question at the  User:  prompt.
Type  exit ,  quit , or  q  to end the session.
"""

import json
import os
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional dependency: httpx is much friendlier than urllib for JSON APIs.
# It ships with the project already (FastAPI's TestClient uses it).
# Fall back to urllib if somehow not available.
# ---------------------------------------------------------------------------
try:
    import httpx
    _BACKEND = "httpx"
except ImportError:
    import urllib.request
    import urllib.error
    _BACKEND = "urllib"

# ---------------------------------------------------------------------------
# Load .env manually (no dotenv library needed at runtime for a CLI script)
# ---------------------------------------------------------------------------
def _load_dotenv(path: Path) -> None:
    """Parse a .env file and inject values into os.environ (skip if missing)."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE    = os.getenv("RAG_API_BASE", "http://localhost:8000")
API_KEY     = os.getenv("APP_API_KEY", "")
QUERY_URL   = f"{API_BASE}/api/v1/query"
HEADERS     = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
}

# Terminal width used for wrapping the AI response
WRAP_WIDTH  = 88

# ANSI colour codes (disabled automatically on Windows without ANSI support)
_ANSI = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _ANSI else text

CYAN    = lambda t: _c("96", t)      # noqa: E731  — user label
GREEN   = lambda t: _c("92", t)      # noqa: E731  — bot label
YELLOW  = lambda t: _c("93", t)      # noqa: E731  — sources / notes
DIM     = lambda t: _c("2",  t)      # noqa: E731  — dim metadata
RED     = lambda t: _c("91", t)      # noqa: E731  — errors
BOLD    = lambda t: _c("1",  t)      # noqa: E731  — headings

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post_httpx(payload: dict) -> dict:
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(QUERY_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
        return resp.json()


def _post_urllib(payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(QUERY_URL, data=data, headers=HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def post_query(messages: list[dict]) -> dict:
    """POST the conversation to /api/v1/query and return the parsed JSON."""
    payload = {"messages": messages}
    if _BACKEND == "httpx":
        return _post_httpx(payload)
    return _post_urllib(payload)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _wrap(text: str, indent: str = "     ") -> str:
    """Word-wrap text to WRAP_WIDTH, keeping a hanging indent on continuation lines."""
    paragraphs = text.split("\n")
    wrapped = []
    for para in paragraphs:
        if not para.strip():
            wrapped.append("")
            continue
        lines = textwrap.wrap(para, width=WRAP_WIDTH, subsequent_indent=indent)
        wrapped.extend(lines if lines else [""])
    return "\n".join(wrapped)


def print_banner() -> None:
    print()
    print(BOLD("╔══════════════════════════════════════════════════════════╗"))
    print(BOLD("║      LDS Doctrinal RAG — Interactive Chat CLI            ║"))
    print(BOLD("╚══════════════════════════════════════════════════════════╝"))
    print(DIM(f"  Backend : {API_BASE}"))
    print(DIM(f"  API key : {'set ✓' if API_KEY else 'NOT SET ✗  (set APP_API_KEY in .env)'}"))
    print(DIM("  Type  exit / quit / q  to end the session."))
    print()


def print_response(data: dict) -> None:
    answer  = data.get("answer", "").strip()
    sources = data.get("sources", [])

    print()
    print(GREEN("Bot: ") + _wrap(answer, indent="      "))

    if sources:
        print()
        print(YELLOW("  Sources:"))
        for src in sources:
            print(YELLOW(f"    • {src}"))

    print()


def print_error(message: str) -> None:
    print()
    print(RED(f"  ✗  {message}"))
    print()


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

def main() -> None:
    print_banner()

    if not API_KEY:
        print(RED("  Warning: APP_API_KEY is not set. Every request will return 401."))
        print(DIM("  Add  APP_API_KEY=<your-key>  to your .env file and restart.\n"))

    messages: list[dict] = []
    turn = 0

    while True:
        # Prompt
        try:
            user_input = input(CYAN("You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + DIM("  Session ended."))
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print(DIM("  Goodbye! 👋"))
            break

        # Build and ship the request
        messages.append({"role": "user", "content": user_input})
        turn += 1

        print(DIM(f"  [turn {turn}] Thinking…"))

        try:
            data = post_query(messages)
        except Exception as exc:  # noqa: BLE001
            # Keep the user message in history so they can retry
            err = str(exc)
            if "Connection refused" in err or "ConnectionError" in err or "Failed to establish" in err:
                print_error(
                    "Cannot reach the backend server at "
                    f"{API_BASE}.\n"
                    "  Make sure uvicorn is running:\n"
                    "    uv run uvicorn src.backend.main:app --reload --port 8000"
                )
            elif "401" in err:
                print_error("Unauthorised (401). Check APP_API_KEY in your .env.")
            elif "422" in err:
                print_error("Validation error (422). The request payload was rejected by the server.")
            elif "502" in err:
                print_error("The backend could not reach OpenAI (502). Check OPENAI_API_KEY.")
            else:
                print_error(f"Unexpected error: {exc}")
            # Pop the failed message so the broken turn isn't stuck in history
            messages.pop()
            continue

        # Append assistant reply to history for the next turn
        answer = data.get("answer", "")
        messages.append({"role": "assistant", "content": answer})

        print_response(data)


if __name__ == "__main__":
    main()
