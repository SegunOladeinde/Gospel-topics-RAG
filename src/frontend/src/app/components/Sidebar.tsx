"use client";
import { BookOpen, MessageSquarePlus, X, ChevronLeft } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";

interface Props {
    isOpen: boolean;
    isCollapsed: boolean;
    onToggleCollapse: () => void;
    onClose: () => void;
    onNewChat: () => void;
    firstMessage?: string;
    isMobile: boolean;
}

export function Sidebar({
    isOpen, isCollapsed, onToggleCollapse, onClose, onNewChat, firstMessage, isMobile
}: Props) {

    const widthClass = (!isMobile && isCollapsed) ? "w-[68px]" : "w-64";

    return (
        <aside
            className={`fixed inset-y-0 left-0 z-40 flex shrink-0 flex-col transition-all duration-300 ease-in-out
        ${widthClass} ${isOpen ? "translate-x-0" : "-translate-x-full"}
        md:static md:z-auto
        bg-white/40 backdrop-blur-2xl border-r border-white/60 shadow-sm
        dark:bg-neutral-950/40 dark:backdrop-blur-2xl dark:border-white/10`}
        >
            <div className={`flex items-center py-5 ${isCollapsed ? "px-4 justify-center" : "px-5 justify-between"}`}>
                <div className="flex items-center gap-2.5 overflow-hidden">
                    <div className="flex shrink-0 h-7 w-7 items-center justify-center rounded-lg bg-[#7c6cfc]/20">
                        <BookOpen size={14} className="text-[#7c6cfc]" />
                    </div>
                    {!isCollapsed && (
                        <span className="text-sm font-semibold whitespace-nowrap text-slate-800 dark:text-neutral-100">
                            Gospel RAG
                        </span>
                    )}
                </div>

                {isMobile ? (
                    <button onClick={onClose} className="rounded-md p-1 text-slate-400 hover:text-slate-700 dark:text-neutral-500 dark:hover:text-neutral-300">
                        <X size={16} />
                    </button>
                ) : (
                    !isCollapsed && (
                        <button onClick={onToggleCollapse} className="rounded-md p-1 opacity-0 transition-opacity group-hover:opacity-100 hover:bg-black/5 dark:hover:bg-white/10 text-slate-500 dark:text-neutral-400 lg:opacity-100">
                            <ChevronLeft size={16} />
                        </button>
                    )
                )}
            </div>

            <div className={`px-3 pb-3`}>
                <button
                    type="button"
                    onClick={onNewChat}
                    className={`flex items-center justify-center rounded-xl text-sm font-medium transition-all duration-200
            ${isCollapsed ? "w-10 h-10 mx-auto p-0" : "w-full gap-2.5 px-3.5 py-2.5"}
            /* Light */ border border-white/60 bg-white/40 text-slate-700 shadow-sm hover:bg-white/60
            /* Dark */ dark:border dark:border-white/[0.05] dark:bg-white/[0.03] dark:text-neutral-200 dark:hover:border-white/[0.1] dark:hover:bg-white/[0.05]`}
                    title={isCollapsed ? "New Chat" : undefined}
                >
                    <MessageSquarePlus size={15} className="shrink-0 text-slate-500 dark:text-neutral-300" />
                    {!isCollapsed && <span>New Chat</span>}
                </button>
            </div>

            <nav className={`flex-1 overflow-y-auto px-3 py-2 transition-opacity duration-200 ${isCollapsed ? "opacity-0 invisible" : "opacity-100 visible"}`}>
                <p className="mb-2 px-2.5 text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-neutral-500">Recent</p>
                {firstMessage ? (
                    <div className="rounded-lg px-2.5 py-2 text-sm text-slate-700 transition-colors hover:bg-white/50 dark:text-neutral-300 dark:hover:bg-white/[0.05]">
                        <p className="truncate">{firstMessage}</p>
                    </div>
                ) : (
                    <p className="px-2.5 text-sm text-slate-500 dark:text-neutral-600">No conversations yet</p>
                )}
            </nav>

            <div className={`border-t border-white/40 dark:border-white/[0.05] ${isCollapsed ? "flex justify-center py-3" : "px-4 py-3"}`}>
                {isCollapsed ? (
                    <div className="flex h-10 items-center justify-center" title="Toggle Theme">
                        <ThemeToggle iconOnly />
                    </div>
                ) : (
                    <ThemeToggle />
                )}
            </div>

            {!isCollapsed && (
                <div className="space-y-2 border-t border-white/40 px-5 py-4 dark:border-white/[0.05]">
                    <p className="text-[11px] leading-relaxed text-slate-500 dark:text-neutral-500">Grounded in scraped LDS source material.</p>
                    <p className="text-[11px] leading-relaxed text-slate-500 dark:text-neutral-500">Not affiliated with The Church of Jesus Christ of Latter-day Saints.</p>
                </div>
            )}
        </aside>
    );
}
