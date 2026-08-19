"use client";
import { useEffect, useRef } from "react";
import { MessageBubble } from "./MessageBubble";
import type { Message } from "../types";

interface Props {
    messages: Message[];
    isLoading: boolean;
    error: string | null;
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
    const anchorRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        anchorRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, isLoading]);

    return (
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-8 pb-40 md:px-6">
            {messages.map((msg, i) => (
                <MessageBubble key={i} message={msg} />
            ))}

            {isLoading && <ThinkingBubble />}

            {error && (
                <div className="rounded-2xl border px-5 py-4 text-sm relative overflow-hidden backdrop-blur-md
          border-red-200/80 bg-red-50/80 text-red-700
          dark:bg-red-500/10 dark:border-red-500/20 dark:backdrop-blur-xl dark:text-red-400">
                    <span className="relative z-10 font-medium">⚠ {error}</span>
                </div>
            )}

            <div ref={anchorRef} />
        </div>
    );
}
