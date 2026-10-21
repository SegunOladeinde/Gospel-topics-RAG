"use client";
import { useEffect, useRef } from "react";
import { MessageBubble } from "./MessageBubble";
import { SignInButton } from "@clerk/nextjs";
import type { Message } from "../types";
import { limitMessage, type ChatError } from "@/lib/usage";
import { scrollActionFor, type FeedSnapshot } from "@/lib/chatScroll";

interface Props {
    messages: Message[];
    isLoading: boolean;
    error: ChatError | null;
}

function ThinkingBubble() {
    return (
        <div className="flex justify-start">
            <div className="flex items-center gap-1.5 rounded-2xl rounded-bl-sm border px-5 py-4 relative
        border-white/80 bg-white/40 shadow-[0_8px_30px_rgb(0,0,0,0.06)] backdrop-blur-xl
        dark:bg-white/[0.02] dark:backdrop-blur-2xl dark:border-white/[0.05]">
                {[0, 1, 2].map((i) => (
                    <span
                        key={i}
                        className="h-1.5 w-1.5 rounded-full animate-pulse bg-slate-400 dark:bg-neutral-400 relative z-10"
                        style={{ animationDelay: `${i * 0.15}s`, animationDuration: "0.8s" }}
                    />
                ))}
            </div>
        </div>
    );
}

export function ChatFeed({ messages, isLoading, error }: Props) {
    const feedRef = useRef<HTMLDivElement>(null);
    const anchorRef = useRef<HTMLDivElement>(null);
    const prevRef = useRef<FeedSnapshot>({ count: 0, lastRole: null, hasError: false });
    // Set when you scroll up yourself while waiting; cleared when you ask again.
    const userScrolledUpRef = useRef(false);

    // Notice manual scrolling (wheel, touch, keys) on the page's scroll area.
    useEffect(() => {
        const scroller = feedRef.current?.closest<HTMLElement>("[data-chat-scroll]");
        if (!scroller) return;
        const nearBottom = () => scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80;
        const onWheel = (e: WheelEvent) => {
            if (e.deltaY < 0) userScrolledUpRef.current = true;
            else if (nearBottom()) userScrolledUpRef.current = false;
        };
        const onTouchMove = () => {
            userScrolledUpRef.current = !nearBottom();
        };
        const onKeyDown = (e: KeyboardEvent) => {
            if (["PageUp", "ArrowUp", "Home"].includes(e.key)) userScrolledUpRef.current = true;
        };
        scroller.addEventListener("wheel", onWheel, { passive: true });
        scroller.addEventListener("touchmove", onTouchMove, { passive: true });
        scroller.addEventListener("keydown", onKeyDown);
        return () => {
            scroller.removeEventListener("wheel", onWheel);
            scroller.removeEventListener("touchmove", onTouchMove);
            scroller.removeEventListener("keydown", onKeyDown);
        };
    }, []);

    useEffect(() => {
        const next: FeedSnapshot = {
            count: messages.length,
            lastRole: messages.at(-1)?.role ?? null,
            hasError: Boolean(error),
        };
        if (next.lastRole === "user" && next.count === prevRef.current.count + 1) {
            userScrolledUpRef.current = false;
        }
        const action = scrollActionFor(prevRef.current, next, userScrolledUpRef.current);
        prevRef.current = next;

        if (action === "bottom") {
            anchorRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
        } else if (action === "bottom-instant") {
            anchorRef.current?.scrollIntoView({ behavior: "instant", block: "end" });
        } else if (action === "message-start") {
            feedRef.current
                ?.querySelector(`[data-message-index="${next.count - 1}"]`)
                ?.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    }, [messages, error]);

    // The thinking dots appear just after your question; keep them in view.
    useEffect(() => {
        if (isLoading && !userScrolledUpRef.current) {
            anchorRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
        }
    }, [isLoading]);

    return (
        <div ref={feedRef} className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-8 pb-40 md:px-6">
            {messages.map((msg, i) => (
                // scroll-mt keeps the start of an answer clear of the sticky header.
                <div key={i} data-message-index={i} className="scroll-mt-20">
                    <MessageBubble message={msg} />
                </div>
            ))}

            {/* Dots only until the answer starts streaming in. */}
            {isLoading && messages.at(-1)?.role !== "assistant" && <ThinkingBubble />}

            {error && (
                <div className="rounded-2xl border px-5 py-4 text-sm relative overflow-hidden backdrop-blur-md
          border-red-200/80 bg-red-50/80 text-red-700
          dark:bg-red-500/10 dark:border-red-500/20 dark:backdrop-blur-xl dark:text-red-400">
                    <span className="relative z-10 font-medium">
                        {error.code === "GUEST_LIMIT_REACHED" ? (
                            <div className="flex flex-col gap-3">
                                <div>{limitMessage(error)}</div>
                                <div>
                                    <SignInButton mode="modal">
                                        <button className="rounded-lg bg-[#7c6cfc] px-4 py-2 text-sm font-medium text-white transition-all hover:bg-[#9b8dff] hover:scale-[1.03] active:scale-95 shadow-sm">
                                            Sign In to Continue
                                        </button>
                                    </SignInButton>
                                </div>
                            </div>
                        ) : error.code === "USER_LIMIT_REACHED" ? (
                            limitMessage(error)
                        ) : (
                            `⚠ ${error.message}`
                        )}
                    </span>
                </div>
            )}

            <div ref={anchorRef} />
        </div>
    );
}
