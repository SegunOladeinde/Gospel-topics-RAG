import { NextRequest } from "next/server";

import { auth } from "@clerk/nextjs/server";

import { proxyToBackend } from "@/lib/backendProxy";
import { checkIsAdmin } from "@/lib/currentUserIsAdmin";

export const maxDuration = 60;

// How many questions the caller has left, for the remaining-questions
// indicator. Guests are counted by IP, so the client IP is forwarded too.
export async function GET(request: NextRequest) {
  const { userId } = await auth();
  return proxyToBackend(request.headers, {
    path: "usage",
    method: "GET",
    userId,
    isAdmin: await checkIsAdmin(userId),
  });
}
