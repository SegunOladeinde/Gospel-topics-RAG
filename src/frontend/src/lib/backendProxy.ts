import { NextResponse } from "next/server";

import { buildBackendHeaders } from "./admin";
import { backendEndpoint } from "./backendUrl";

// Server-only: imported by the API routes, which run on the Next.js server,
// so BACKEND_URL and BACKEND_API_KEY are safe to read here. Neither must be
// prefixed with NEXT_PUBLIC_, which would bundle it into client-side JS.
// No hardcoded fallback for BACKEND_URL on purpose -- an unconfigured
// deployment should fail loudly instead of silently talking to localhost.
// BACKEND_URL is the full query endpoint, e.g.
// https://gospel-topics-rag.onrender.com/api/v1/query
const BACKEND_URL = process.env.BACKEND_URL;
const BACKEND_API_KEY = process.env.BACKEND_API_KEY;

// A sleeping Render instance takes up to about a minute to boot and load the
// indexes, so give the backend 55s. That stays under the routes' 60s
// maxDuration, so we still get to send the friendly message before Vercel
// kills the function.
const BACKEND_TIMEOUT_MS = 55_000;

export const WAKING_UP_MESSAGE =
  "Waking up the server, please try again in a moment.";

interface ProxyOptions {
  // Relative to BACKEND_URL; null means BACKEND_URL itself (the query endpoint).
  path: string | string[] | null;
  method: "GET" | "POST" | "DELETE";
  body?: unknown;
  userId: string | null;
  isAdmin: boolean;
}

// Forwards one request to the FastAPI backend with the shared API key and
// the caller's identity attached server-side (see buildBackendHeaders: the
// browser can't set them), and turns cold starts and Render's HTML error
// pages into a JSON { detail } the UI can show.
export async function proxyToBackend(
  incoming: Headers,
  { path, method, body, userId, isAdmin }: ProxyOptions
): Promise<NextResponse> {
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

  let backendResponse: Response;
  try {
    backendResponse = await fetch(
      path === null ? BACKEND_URL : backendEndpoint(BACKEND_URL, path),
      {
        method,
        headers: buildBackendHeaders(incoming, {
          apiKey: BACKEND_API_KEY,
          userId,
          isAdmin,
        }),
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: AbortSignal.timeout(BACKEND_TIMEOUT_MS),
        cache: "no-store",
      }
    );
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

  if (backendResponse.status === 204) {
    return new NextResponse(null, { status: 204 });
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
  return NextResponse.json(data, {
    status: backendResponse.status,
    headers: { "Cache-Control": "private, no-store" },
  });
}
