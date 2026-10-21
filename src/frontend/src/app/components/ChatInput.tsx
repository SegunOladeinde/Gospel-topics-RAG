"use client";
import { Send, Square } from "lucide-react";
import { useLayoutEffect, useRef, type FormEvent, type KeyboardEvent } from "react";
import { MAX_QUESTION_CHARS, counterText, isSendKey } from "@/lib/chatInput";

// The box grows with the text up to this many lines, then scrolls.
const MAX_LINES = 6;
const LINE_HEIGHT_PX = 24;
const VERTICAL_PADDING_PX = 12; // py-1.5 on the textarea; scrollHeight includes it

interface Props {
    value: string;
    onChange: (val: string) => void;
    onSubmit: (question: string) => void;
    isLoading: boolean;
    // Remaining-questions text under the box; null shows nothing.
    usageText?: string | null;
    // While an answer is on its way, Send becomes Stop.
    onStop?: () => void;
    canStop?: boolean;
}

export function ChatInput({ value, onChange, onSubmit, isLoading, usageText, onStop, canStop = false }: Props) {
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const counter = counterText(value.length);

    // Auto-grow: measure the content, then cap at MAX_LINES (scroll beyond).
    useLayoutEffect(() => {
        const textarea = textareaRef.current;
        if (!textarea) return;
        textarea.style.height = "auto";
        const maxHeight = MAX_LINES * LINE_HEIGHT_PX + VERTICAL_PADDING_PX;
        textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`;
        textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
    }, [value]);

    const submit = () => {
        const trimmed = value.trim();
        if (trimmed && !isLoading) onSubmit(trimmed);
    };

    const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        submit();
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (
            isSendKey({
                key: e.key,
                shiftKey: e.shiftKey,
                isComposing: e.nativeEvent.isComposing,
                keyCode: e.keyCode,
            })
        ) {
            e.preventDefault();
            submit();
        }
    };

    return (
        <div className="sticky bottom-0 mt-auto pt-4 pb-4 px-4 md:pt-6 md:pb-6 md:px-6">
            <form
                onSubmit={handleSubmit}
                className="mx-auto flex w-full max-w-full md:max-w-2xl lg:max-w-3xl items-end gap-3 rounded-2xl px-4 py-3
          bg-white/50 backdrop-blur-2xl border border-white/60 shadow-sm
          dark:bg-neutral-900/50 dark:backdrop-blur-2xl dark:border-white/10
          focus-within:border-[#7c6cfc]/60 dark:focus-within:border-[#7c6cfc]/50 transition-all duration-300"
            >
                {/* 16px text: anything smaller makes iOS Safari zoom in on focus. */}
                <textarea
                    ref={textareaRef}
                    rows={1}
                    value={value}
                    // maxLength stops typing/pasting; the slice also covers autofill and scripts.
                    onChange={(e) => onChange(e.target.value.slice(0, MAX_QUESTION_CHARS))}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask a doctrinal question..."
                    disabled={isLoading}
                    maxLength={MAX_QUESTION_CHARS}
                    aria-label="Your question"
                    enterKeyHint="send"
                    style={{ lineHeight: `${LINE_HEIGHT_PX}px` }}
                    className="flex-1 min-w-0 resize-none bg-transparent py-1.5 text-base outline-none disabled:opacity-50 relative z-10
            text-slate-900 placeholder:text-slate-500
            dark:text-neutral-100 dark:placeholder:text-neutral-500"
                />
                {canStop && onStop ? (
                    <button
                        type="button"
                        onClick={onStop}
                        aria-label="Stop generating"
                        title="Stop"
                        className="relative z-10 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl transition-all duration-200
                          bg-slate-800 text-white hover:bg-slate-700 active:scale-95
                          dark:bg-neutral-200 dark:text-neutral-900 dark:hover:bg-white"
                    >
                        <Square size={12} fill="currentColor" />
                    </button>
                ) : (
                    <button
                        type="submit"
                        disabled={isLoading || !value.trim()}
                        aria-label="Send message"
                        className="relative z-10 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[#7c6cfc] text-white transition-all duration-200 hover:bg-[#9b8dff] hover:scale-105 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:scale-100"
                    >
                        <Send size={15} className="-ml-0.5" />
                    </button>
                )}
            </form>
            {(usageText || counter) && (
                <div className="mx-auto mt-2 flex w-full max-w-full md:max-w-2xl lg:max-w-3xl items-center justify-between gap-3 px-1 text-xs text-slate-500 dark:text-neutral-500">
                    <p aria-live="polite" className="min-w-0 flex-1 text-center">
                        {usageText}
                    </p>
                    {counter && (
                        <p
                            aria-live="polite"
                            className={`shrink-0 tabular-nums ${value.length >= MAX_QUESTION_CHARS ? "text-amber-600 dark:text-amber-400" : ""}`}
                        >
                            {counter}
                        </p>
                    )}
                </div>
            )}
        </div>
    );
}
