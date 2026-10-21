import { NextRequest, NextResponse } from "next/server";

import { auth } from "@clerk/nextjs/server";

import { proxyToBackend } from "@/lib/backendProxy";

export const maxDuration = 60;

// The signed-in user's saved chats for the sidebar. Guests have none.
export async function GET(request: NextRequest) {
  const { userId } = await auth();
  if (!userId) {
    return NextResponse.json({ conversations: [] });
  }
  return proxyToBackend(request.headers, {
    path: "conversations",
    method: "GET",
    userId,
    isAdmin: false,
  });
}
