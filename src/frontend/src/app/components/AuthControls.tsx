/**
 * AuthControls — persistent floating Clerk auth UI
 *
 * Placement: direct child of <body> via layout.tsx, rendered as a sibling
 * above <Providers> so it escapes every nested stacking context / overflow
 * boundary created by the chat layout.
 *
 * z-[100] sits above:
 *   - sidebar overlay   z-40
 *   - backdrop scrim    z-30
 *   - sticky header     z-10
 *
 * "use client" is required because Clerk hooks are client-only.
 * CLERK_SECRET_KEY is never referenced here.
 */
"use client";
import { useEffect, useState } from "react";
import { Show, SignInButton, SignUpButton, UserButton, useAuth } from "@clerk/nextjs";

// Asks the server whether the signed-in user is an admin (ADMIN_EMAILS is
// server-only). Display only: /api/query re-checks on every question.
function AdminBadge() {
    const { userId } = useAuth();
    // Remember which user the answer was for, so a stale answer never shows
    // after switching accounts.
    const [adminUserId, setAdminUserId] = useState<string | null>(null);

    useEffect(() => {
        if (!userId) return;
        let cancelled = false;
        fetch("/api/me")
            .then((res) => (res.ok ? res.json() : null))
            .then((data: { isAdmin?: boolean } | null) => {
                if (!cancelled) setAdminUserId(data?.isAdmin ? userId : null);
            })
            .catch(() => {});
        return () => {
            cancelled = true;
        };
    }, [userId]);

    if (!userId || adminUserId !== userId) return null;
    return (
        <span
            title="Admin · unlimited questions"
            className="whitespace-nowrap rounded-lg border border-[#7c6cfc]/30 bg-[#7c6cfc]/10 backdrop-blur-md px-2.5 py-1 text-xs font-medium text-[#7c6cfc]
              dark:text-[#a599ff] dark:border-[#7c6cfc]/40 dark:bg-[#7c6cfc]/20"
        >
            Admin<span className="hidden sm:inline"> · unlimited</span>
        </span>
    );
}

export function AuthControls() {
    return (
        <div className="fixed top-3.5 right-3 sm:top-4 sm:right-4 md:right-8 z-[100] flex items-center gap-1.5 sm:gap-2 md:gap-3">
            {/* Unauthenticated state */}
            <Show when="signed-out">
                <SignInButton mode="modal">
                    <button
                        type="button"
                        className="rounded-lg px-2.5 py-1 text-xs sm:px-3 sm:py-1.5 sm:text-sm font-medium transition-colors
                          whitespace-nowrap text-slate-700 hover:text-slate-900
                          bg-white/70 hover:bg-white/90 backdrop-blur-md
                          border border-white/80 shadow-sm
                          dark:text-neutral-200 dark:hover:text-white
                          dark:bg-white/[0.06] dark:hover:bg-white/[0.10]
                          dark:border-white/[0.10]"
                    >
                        Sign in
                    </button>
                </SignInButton>
                <SignUpButton mode="modal">
                    <button
                        type="button"
                        className="rounded-lg bg-[#7c6cfc] px-2.5 py-1 text-xs sm:px-3 sm:py-1.5 sm:text-sm font-medium text-white whitespace-nowrap
                          transition-all hover:bg-[#9b8dff] hover:scale-[1.03] active:scale-95 shadow-sm"
                    >
                        Sign up
                    </button>
                </SignUpButton>
            </Show>

            {/* Authenticated state — avatar opens Clerk's account portal */}
            <Show when="signed-in">
                <AdminBadge />
                <UserButton
                    appearance={{
                        elements: {
                            avatarBox: "h-8 w-8 rounded-full ring-2 ring-[#7c6cfc]/40",
                        },
                    }}
                />
            </Show>
        </div>
    );
}
