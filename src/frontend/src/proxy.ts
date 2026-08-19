/**
 * Clerk authentication middleware for Next.js 16.
 *
 * Why this file is named `proxy.ts` and not `middleware.ts`:
 * Next.js 16 changed the middleware entry-point name to `proxy.ts`.
 * For Next.js ≤ 15 it would be `middleware.ts`; the code is identical.
 *
 * clerkMiddleware() makes the current user's auth state available to
 * every route and API handler via the `auth()` helper from
 * `@clerk/nextjs/server`. By default ALL routes are PUBLIC — you must
 * explicitly call `auth.protect()` on server code you want to gate.
 *
 * The matcher excludes:
 *   - Next.js internal assets (_next/*)
 *   - Static files (images, fonts, favicons, etc.)
 * and always runs for:
 *   - All page and API routes
 *   - Clerk's own internal proxy routes (/__clerk/*)
 */
import { clerkMiddleware } from "@clerk/nextjs/server";

export default clerkMiddleware();

export const config = {
    matcher: [
        // Skip Next.js internals and all static files, unless found in search params
        "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
        // Always run for API routes
        "/(api|trpc)(.*)",
        // Always run for Clerk-specific frontend proxy routes
        "/__clerk/(.*)",
    ],
};
