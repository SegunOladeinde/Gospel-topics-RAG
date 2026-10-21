import { NextRequest, NextResponse } from "next/server";

import { auth } from "@clerk/nextjs/server";

import { proxyToBackend, QUERY_BACKEND_TIMEOUT_MS } from "@/lib/backendProxy";
import { checkIsAdmin } from "@/lib/currentUserIsAdmin";

// The most the Hobby plan allows with Fluid compute (on by default for new
// projects): a Render cold start plus the whole answer streaming through.
// Once awake, the backend sends a keep-alive comment straight away and every
// few seconds while it searches, so the stream never sits silent.
export const maxDuration = 300;

// Streaming version of /api/query: the answer arrives as server-sent events
// (token ... done, or error). Refusals come back as ordinary JSON errors.
// Stopping in the browser aborts request.signal, which cancels the backend
// request so the model stops generating.
export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body." }, { status: 400 });
  }

  const { userId } = await auth();
  return proxyToBackend(request.headers, {
    path: ["query", "stream"],
    method: "POST",
    body,
    userId,
    isAdmin: await checkIsAdmin(userId),
    stream: true,
    signal: request.signal,
    timeoutMs: QUERY_BACKEND_TIMEOUT_MS,
  });
}
