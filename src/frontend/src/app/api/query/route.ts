import { NextRequest, NextResponse } from "next/server";

import { auth } from "@clerk/nextjs/server";

import { proxyToBackend } from "@/lib/backendProxy";
import { checkIsAdmin } from "@/lib/currentUserIsAdmin";

// Long enough for a Render cold start (see BACKEND_TIMEOUT_MS in backendProxy).
export const maxDuration = 60;

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
  });
}
