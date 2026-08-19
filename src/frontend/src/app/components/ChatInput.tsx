"use client";
import { Send } from "lucide-react";
import type { FormEvent } from "react";

interface Props {
    value: string;
    onChange: (val: string) => void;
    onSubmit: (question: string) => void;
    isLoading: boolean;
}

export function ChatInput({ value, onChange, onSubmit, isLoading }: Props) {
    const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        const trimmed = value.trim();
        if (trimmed) onSubmit(trimmed);
    };

    return (
        <div className="sticky bottom-0 mt-auto pt-4 pb-4 px-4 md:pt-6 md:pb-6 md:px-6">
            <form
                onSubmit={handleSubmit}
                className="mx-auto flex w-full max-w-full md:max-w-2xl lg:max-w-3xl items-center gap-3 rounded-2xl px-4 py-3
          bg-white/50 backdrop-blur-2xl border border-white/60 shadow-sm
          dark:bg-neutral-900/50 dark:backdrop-blur-2xl dark:border-white/10
          focus-within:border-[#7c6cfc]/60 dark:focus-within:border-[#7c6cfc]/50 transition-all duration-300"
            >
                <input
                    type="text"
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    placeholder="Ask a doctrinal question..."
                    disabled={isLoading}
                    className="flex-1 min-h-[24px] bg-transparent text-[15px] outline-none disabled:opacity-50 relative z-10
            text-slate-900 placeholder:text-slate-500
            dark:text-neutral-100 dark:placeholder:text-neutral-500"
                />
                <button
                    type="submit"
                    disabled={isLoading || !value.trim()}
                    aria-label="Send message"
                    className="relative z-10 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[#7c6cfc] text-white transition-all duration-200 hover:bg-[#9b8dff] hover:scale-105 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:scale-100"
                >
                    <Send size={15} className="-ml-0.5" />
                </button>
            </form>
        </div>
    );
}
