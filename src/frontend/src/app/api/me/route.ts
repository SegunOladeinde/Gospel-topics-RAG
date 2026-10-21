import { NextResponse } from "next/server";

import { auth } from "@clerk/nextjs/server";

import { checkIsAdmin } from "@/lib/currentUserIsAdmin";

// Tells the UI whether to show the "Admin · unlimited" badge. Display only:
// the real check happens again in /api/query on every question.
export async function GET() {
  const { userId } = await auth();
  return NextResponse.json(
    { isAdmin: await checkIsAdmin(userId) },
    { headers: { "Cache-Control": "private, no-store" } }
  );
}
