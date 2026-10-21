import { NextRequest, NextResponse } from "next/server";

import { auth } from "@clerk/nextjs/server";

import { buildBackendHeaders } from "@/lib/admin";
import { checkIsAdmin } from "@/lib/currentUserIsAdmin";

// Server-only: this file runs on the Next.js server, never in the browser,
// so BACKEND_URL and BACKEND_API_KEY are safe to read here. Neither must be
// prefixed with NEXT_PUBLIC_, which would bundle it into client-side JS.
// No hardcoded fallback for BACKEND_URL on purpose -- an unconfigured
// deployment should fail loudly instead of silently talking to localhost.
// BACKEND_URL is the full endpoint, e.g.
// https://gospel-topics-rag.onrender.com/api/v1/query
const BACKEND_URL = process.env.BACKEND_URL;
const BACKEND_API_KEY = process.env.BACKEND_API_KEY;

// A sleeping Render instance takes up to about a minute to boot and load the
// indexes, so give the backend 55s. That stays under the 60s maxDuration
// below, so we still get to send the friendly message before Vercel kills
// the function.
const BACKEND_TIMEOUT_MS = 55_000;
export const maxDuration = 60;

const WAKING_UP_MESSAGE =
  "Waking up the server, please try again in a moment.";

// Proxies the browser's chat request to the FastAPI backend, attaching the
// shared API key server-side so it's never exposed to client-side code.
export async function POST(request: NextRequest) {
  const missing = [
    !BACKEND_URL && "BACKEND_URL",
    !BACKEND_API_KEY && "BACKEND_API_KEY",
  ].filter(Boolean);
  if (!BACKEND_URL || !BACKEND_API_KEY) {
    console.error(`Missing environment variable(s): ${missing.join(", ")}`);
    return NextResponse.json(
      {
        detail: `Server misconfiguration: ${missing.join(" and ")} ${
          missing.length > 1 ? "are" : "is"
        } not set.`,
      },
      { status: 500 }
    );
  }

  const { userId } = await auth();
  const isAdmin = await checkIsAdmin(userId);

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
      headers: buildBackendHeaders(request.headers, {
        apiKey: BACKEND_API_KEY,
        userId,
        isAdmin,
      }),
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(BACKEND_TIMEOUT_MS),
    });
  } catch (err) {
    // Timeout: almost always a cold start that's still booting.
    // Other network errors: the instance is down, restarting or deploying.
    const timedOut = err instanceof Error && err.name === "TimeoutError";
    console.error(
      timedOut
        ? `Backend did not respond within ${BACKEND_TIMEOUT_MS / 1000}s`
        : "Failed to reach the backend service",
      err
    );
    return NextResponse.json(
      { detail: WAKING_UP_MESSAGE },
      { status: timedOut ? 504 : 503 }
    );
  }

  const data = await backendResponse.json().catch(() => null);
  if (data === null) {
    // Render answers 502/503/504 with an HTML page while an instance is
    // waking up or redeploying, so there's no JSON detail to pass through.
    const waking = [502, 503, 504].includes(backendResponse.status);
    return NextResponse.json(
      {
        detail: waking
          ? WAKING_UP_MESSAGE
          : `Unexpected response from the backend (${backendResponse.status}).`,
      },
      { status: backendResponse.ok ? 502 : backendResponse.status }
    );
  }
  return NextResponse.json(data, { status: backendResponse.status });
}
