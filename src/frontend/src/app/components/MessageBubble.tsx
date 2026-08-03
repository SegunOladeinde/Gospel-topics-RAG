"use client";
import ReactMarkdown from "react-markdown";
import type { Message } from "../types";

interface Props {
    message: Message;
}

export function MessageBubble({ message }: Props) {
    const isUser = message.role === "user";

    return (
        <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
            <div
                className={`relative max-w-[82%] rounded-2xl px-5 py-4 ${isUser
                        ? // User bubble
                        "rounded-br-sm bg-gradient-to-br from-[#8d7cff] to-[#6b58ff] text-white shadow-md shadow-[#7c6cfc]/20 ring-1 ring-inset ring-white/20"
                        : // Assistant card
                        "rounded-bl-sm border" +
                        /* Light: Heavy glass */
                        " border-white/80 bg-white/40 backdrop-blur-xl shadow-[0_8px_30px_rgb(0,0,0,0.06)] text-slate-800" +
                        /* Dark: Smoked glass */
                        " dark:border-white/[0.05] dark:bg-white/[0.02] dark:backdrop-blur-2xl dark:text-neutral-100"
                    }`}
            >
                {isUser ? (
                    <p className="whitespace-pre-wrap text-[15px] leading-relaxed relative z-10">
                        {message.content}
                    </p>
                ) : (
                    <div
                        className="
              relative z-10 max-w-none text-[15px] leading-relaxed break-words
              [&_p]:my-2 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0
              [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5
              [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5
              [&_li]:my-0.5
              [&_h1]:my-2 [&_h1]:text-base [&_h1]:font-semibold text-slate-900 dark:text-white
              [&_h2]:my-2 [&_h2]:text-base [&_h2]:font-semibold text-slate-900 dark:text-white
              [&_h3]:my-1.5 [&_h3]:text-sm [&_h3]:font-semibold text-slate-900 dark:text-white
              [&_strong]:font-semibold [&_strong]:text-slate-900 dark:[&_strong]:text-white
              [&_a]:text-[#7c6cfc] [&_a]:underline dark:[&_a]:text-[#9b8dff]
              [&_code]:rounded-md [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:text-[13px]
              [&_code]:bg-white/50 [&_code]:text-slate-800
              dark:[&_code]:bg-white/[0.05] dark:[&_code]:text-neutral-300 dark:[&_code]:border dark:[&_code]:border-white/10
              [&_blockquote]:my-3 [&_blockquote]:border-l-[3px] [&_blockquote]:pl-4
              [&_blockquote]:border-slate-400/50 [&_blockquote]:text-slate-600
              dark:[&_blockquote]:border-[#7c6cfc]/50 dark:[&_blockquote]:text-neutral-300 dark:[&_blockquote]:bg-white/[0.02] dark:[&_blockquote]:py-1 dark:[&_blockquote]:pr-2 dark:[&_blockquote]:rounded-r-md
            "
                    >
                        <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>
                )}

                {/* Source pills — assistant only */}
                {!isUser && message.sources && message.sources.length > 0 && (
                    <div className="relative z-10 mt-4 flex flex-wrap gap-2 border-t pt-4 border-white/60 dark:border-white/[0.05]">
                        {message.sources.map((src, i) => (
                            <span
                                key={i}
                                className="rounded-lg border border-[#7c6cfc]/30 bg-[#7c6cfc]/10 backdrop-blur-md px-3 py-1 text-xs font-medium text-[#7c6cfc] dark:text-[#a599ff] dark:border-[#7c6cfc]/40 dark:bg-[#7c6cfc]/20"
                            >
                                {src}
                            </span>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
