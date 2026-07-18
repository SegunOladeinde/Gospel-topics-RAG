import { NextRequest, NextResponse } from "next/server";

// Server-only: this file runs on the Next.js server, never in the browser,
// so BACKEND_URL and BACKEND_API_KEY are safe to read here. Neither must be
// prefixed with NEXT_PUBLIC_, which would bundle it into client-side JS.
// No hardcoded fallback for BACKEND_URL on purpose -- an unconfigured
// deployment should fail loudly instead of silently talking to localhost.
const BACKEND_URL = process.env.BACKEND_URL;
const BACKEND_API_KEY = process.env.BACKEND_API_KEY;

// Proxies the browser's chat request to the FastAPI backend, attaching the
// shared API key server-side so it's never exposed to client-side code.
export async function POST(request: NextRequest) {
  if (!BACKEND_URL || !BACKEND_API_KEY) {
    return NextResponse.json(
      {
        detail:
          "Server misconfiguration: BACKEND_URL and/or BACKEND_API_KEY is not set.",
      },
      { status: 500 }
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body." }, { status: 400 });
  }

  let backendResponse: Response;
  try {
    backendResponse = await fetch(BACKEND_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": BACKEND_API_KEY,
      },
      body: JSON.stringify(body),
    });
  } catch {
    return NextResponse.json(
      { detail: "Failed to reach the backend service." },
      { status: 502 }
    );
  }

  const data = await backendResponse.json().catch(() => null);
  return NextResponse.json(data, { status: backendResponse.status });
}
