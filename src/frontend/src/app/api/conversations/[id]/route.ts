import { NextRequest, NextResponse } from "next/server";

import { auth } from "@clerk/nextjs/server";

import { proxyToBackend } from "@/lib/backendProxy";

export const maxDuration = 60;

// One saved chat with its messages. The backend only returns it if it
// belongs to the signed-in user.
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { userId } = await auth();
  if (!userId) {
    return NextResponse.json({ detail: "Conversation not found." }, { status: 404 });
  }
  const { id } = await params;
  return proxyToBackend(request.headers, {
    path: ["conversations", id],
    method: "GET",
    userId,
    isAdmin: false,
  });
}
