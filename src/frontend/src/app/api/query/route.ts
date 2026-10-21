import { NextRequest, NextResponse } from "next/server";

import { auth } from "@clerk/nextjs/server";

import { proxyToBackend, QUERY_BACKEND_TIMEOUT_MS } from "@/lib/backendProxy";
import { checkIsAdmin } from "@/lib/currentUserIsAdmin";

// The most the Hobby plan allows with Fluid compute, as for the streaming
// route: long enough to wait out a Render cold start.
export const maxDuration = 300;

// Proxies the browser's chat request to the FastAPI backend, attaching the
// shared API key and the caller's identity server-side so neither can be
// set or seen by client-side code.
export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body." }, { status: 400 });
  }

  const { userId } = await auth();
  return proxyToBackend(request.headers, {
    path: null,
    method: "POST",
    body,
    userId,
    isAdmin: await checkIsAdmin(userId),
    timeoutMs: QUERY_BACKEND_TIMEOUT_MS,
  });
}
