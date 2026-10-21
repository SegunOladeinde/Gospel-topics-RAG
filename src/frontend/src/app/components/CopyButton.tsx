"use client";
import { useEffect, useState } from "react";
import { Check, Copy } from "lucide-react";

const COPIED_FOR_MS = 2000;

// Copies text to the clipboard and says "Copied" for a moment.
export function CopyButton({ text, disabled = false }: { text: string; disabled?: boolean }) {
    const [state, setState] = useState<"idle" | "copied" | "failed">("idle");

    useEffect(() => {
        if (state === "idle") return;
        const timer = setTimeout(() => setState("idle"), COPIED_FOR_MS);
        return () => clearTimeout(timer);
    }, [state]);

    const copy = async () => {
        try {
            await navigator.clipboard.writeText(text);
            setState("copied");
        } catch {
            // No clipboard access (e.g. an insecure context or a denied permission).
            setState("failed");
        }
    };

    return (
        <button
            type="button"
            onClick={copy}
            disabled={disabled}
            aria-label={state === "copied" ? "Answer copied" : "Copy answer"}
            className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium transition-colors
              text-slate-500 hover:bg-black/5 hover:text-slate-800
              dark:text-neutral-400 dark:hover:bg-white/[0.06] dark:hover:text-neutral-200
              disabled:cursor-not-allowed disabled:opacity-40"
        >
            {state === "copied" ? <Check size={13} /> : <Copy size={13} />}
            <span aria-live="polite">
                {state === "copied" ? "Copied" : state === "failed" ? "Couldn't copy" : "Copy"}
            </span>
        </button>
    );
}
