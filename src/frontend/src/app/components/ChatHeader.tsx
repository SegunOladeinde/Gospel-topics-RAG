"use client";
import { PanelLeft, BookOpen } from "lucide-react";

interface Props {
    onToggleSidebar: () => void;
    collapsedView?: boolean;
}

export function ChatHeader({ onToggleSidebar, collapsedView }: Props) {
    return (
        <header className="sticky top-0 z-10 flex min-h-[56px] items-center gap-3 px-4 py-3
      border-b border-white/60 bg-white/30 backdrop-blur-xl shadow-[0_4px_24px_rgb(0,0,0,0.02)]
      dark:bg-white/[0.02] dark:backdrop-blur-2xl dark:border-white/[0.05]">

            {collapsedView && (
                <button
                    type="button"
                    onClick={onToggleSidebar}
                    className="rounded-lg p-1.5 transition-colors text-slate-500 hover:bg-black/5 hover:text-slate-800 dark:text-neutral-400 dark:hover:bg-white/[0.05] dark:hover:text-neutral-200"
                    aria-label="Expand sidebar"
                >
                    <PanelLeft size={18} />
                </button>
            )}

            <div className={`flex items-center gap-2.5 transition-opacity duration-300 ${collapsedView || collapsedView === undefined ? "opacity-100" : "opacity-0 hidden"}`}>
                <div className="flex h-6 w-6 items-center justify-center rounded-md bg-[#7c6cfc]/15">
                    <BookOpen size={13} className="text-[#7c6cfc]" />
                </div>
                <div className="flex flex-col leading-tight">
                    <span className="text-sm font-semibold text-slate-800 dark:text-neutral-100">
                        Gospel RAG
                    </span>
                </div>
            </div>
        </header>
    );
}
