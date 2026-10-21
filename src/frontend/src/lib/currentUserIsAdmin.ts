import { currentUser } from "@clerk/nextjs/server";

import { isAdminUser } from "./admin";

// Server-side only. currentUser() is a Clerk API call, so it's only made for
// signed-in users. If Clerk is unreachable the user is treated as a normal
// (limited) user rather than failing the request.
export async function checkIsAdmin(userId: string | null): Promise<boolean> {
  if (!userId) return false;
  try {
    return isAdminUser(await currentUser(), process.env.ADMIN_EMAILS);
  } catch (err) {
    console.error("Could not load the Clerk user for the admin check", err);
    return false;
  }
}
