import { NextRequest, NextResponse } from "next/server";

import { auth } from "@clerk/nextjs/server";

import { proxyToBackend } from "@/lib/backendProxy";
import { checkIsAdmin } from "@/lib/currentUserIsAdmin";

// Long enough for a Render cold start (see BACKEND_TIMEOUT_MS in backendProxy)
// plus the answer itself streaming through.
export const maxDuration = 60;

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
  });
}
