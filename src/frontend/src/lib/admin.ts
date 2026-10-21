// Admin detection for the API routes. Server-side only: ADMIN_EMAILS has no
// NEXT_PUBLIC_ prefix, so it's never bundled into browser code, and nothing
// the browser sends can make someone an admin.
//
// Kept free of Next.js and Clerk imports so tests/admin.test.mjs can run it
// with plain `node --test`.

// The parts of Clerk's User we rely on (currentUser() returns a superset).
export interface AdminCheckUser {
  emailAddresses: {
    emailAddress: string;
    verification: { status: string | null } | null;
  }[];
}

// "a@x.com, B@Y.com ,," -> {"a@x.com", "b@y.com"}. Unset or blank -> empty.
export function parseAdminEmails(raw: string | undefined): Set<string> {
  return new Set(
    (raw ?? "")
      .split(",")
      .map((email) => email.trim().toLowerCase())
      .filter(Boolean)
  );
}

// True only if one of the user's VERIFIED email addresses is in ADMIN_EMAILS.
// An unverified address could belong to anyone who typed it in, so it never
// counts.
export function isAdminUser(
  user: AdminCheckUser | null | undefined,
  adminEmailsEnv: string | undefined
): boolean {
  const adminEmails = parseAdminEmails(adminEmailsEnv);
  if (!user || adminEmails.size === 0) return false;
  return user.emailAddresses.some(
    (address) =>
      address.verification?.status === "verified" &&
      adminEmails.has(address.emailAddress.trim().toLowerCase())
  );
}

// Headers for the FastAPI backend, built from scratch. Only the client IP is
// taken from the incoming request; any X-User-Role (or X-User-Id, X-API-Key)
// the browser sent is dropped, and X-User-Role is set only when the server
// itself decided the user is an admin.
export function buildBackendHeaders(
  incoming: Headers,
  options: { apiKey: string; userId: string | null; isAdmin: boolean }
): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "X-API-Key": options.apiKey,
    "X-User-Id": options.userId || "",
    "X-Forwarded-For":
      incoming.get("x-forwarded-for") || incoming.get("x-real-ip") || "unknown-ip",
  };
  if (options.isAdmin && options.userId) {
    headers["X-User-Role"] = "admin";
  }
  return headers;
}
