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
import { Show, SignInButton, SignUpButton, UserButton } from "@clerk/nextjs";

export function AuthControls() {
    return (
        <div className="fixed top-4 right-4 md:right-8 z-[100] flex items-center gap-2 md:gap-3">
            {/* Unauthenticated state */}
            <Show when="signed-out">
                <SignInButton mode="modal">
                    <button
                        type="button"
                        className="rounded-lg px-3 py-1.5 text-sm font-medium transition-colors
                          text-slate-700 hover:text-slate-900
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
                        className="rounded-lg bg-[#7c6cfc] px-3 py-1.5 text-sm font-medium text-white
                          transition-all hover:bg-[#9b8dff] hover:scale-[1.03] active:scale-95 shadow-sm"
                    >
                        Sign up
                    </button>
                </SignUpButton>
            </Show>

            {/* Authenticated state — avatar opens Clerk's account portal */}
            <Show when="signed-in">
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
