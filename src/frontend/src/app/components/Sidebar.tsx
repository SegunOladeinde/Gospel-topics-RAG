"use client";
import { useState } from "react";
import { BookOpen, MessageSquarePlus, X, ChevronLeft, Trash2 } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";
import type { ConversationSummary } from "../types";

interface Props {
    isOpen: boolean;
    isCollapsed: boolean;
    onToggleCollapse: () => void;
    onClose: () => void;
    onNewChat: () => void;
    conversations: ConversationSummary[];
    activeConversationId: string | null;
    onSelectConversation: (id: string) => void;
    onDeleteConversation: (id: string) => Promise<boolean>;
    // While an answer is on its way, chats can't be deleted.
    isBusy: boolean;
    isSignedIn: boolean;
    isMobile: boolean;
}

export function Sidebar({
    isOpen, isCollapsed, onToggleCollapse, onClose, onNewChat,
    conversations, activeConversationId, onSelectConversation, onDeleteConversation,
    isBusy, isSignedIn, isMobile
}: Props) {
    // The chat showing the "Delete this chat?" confirm step, if any.
    const [confirmingId, setConfirmingId] = useState<string | null>(null);
    const [failedId, setFailedId] = useState<string | null>(null);
    const [deletingId, setDeletingId] = useState<string | null>(null);

    const confirmDelete = async (id: string) => {
        setDeletingId(id);
        const ok = await onDeleteConversation(id);
        setDeletingId(null);
        setConfirmingId(null);
        setFailedId(ok ? null : id);
    };

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
                {conversations.length > 0 ? (
                    <ul className="space-y-0.5">
                        {conversations.map((conversation) => {
                            const isActive = conversation.id === activeConversationId;
                            if (confirmingId === conversation.id) {
                                const isDeleting = deletingId === conversation.id;
                                return (
                                    <li
                                        key={conversation.id}
                                        className="rounded-lg px-2.5 py-2 text-sm bg-red-50/80 dark:bg-red-500/10"
                                    >
                                        <p className="truncate text-slate-700 dark:text-neutral-300">Delete this chat?</p>
                                        <div className="mt-1.5 flex gap-2">
                                            <button
                                                type="button"
                                                onClick={() => confirmDelete(conversation.id)}
                                                disabled={isDeleting}
                                                className="rounded-md bg-red-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-red-500 disabled:opacity-50"
                                            >
                                                {isDeleting ? "Deleting…" : "Delete"}
                                            </button>
                                            <button
                                                type="button"
                                                onClick={() => setConfirmingId(null)}
                                                disabled={isDeleting}
                                                className="rounded-md px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-black/5 dark:text-neutral-300 dark:hover:bg-white/10"
                                            >
                                                Cancel
                                            </button>
                                        </div>
                                    </li>
                                );
                            }
                            return (
                                <li key={conversation.id} className="group relative">
                                    <button
                                        type="button"
                                        onClick={() => onSelectConversation(conversation.id)}
                                        aria-current={isActive ? "page" : undefined}
                                        title={conversation.title}
                                        className={`w-full rounded-lg py-2 pl-2.5 pr-8 text-left text-sm transition-colors
                                            ${isActive
                                                ? "bg-white/60 text-slate-900 dark:bg-white/[0.08] dark:text-white"
                                                : "text-slate-700 hover:bg-white/50 dark:text-neutral-300 dark:hover:bg-white/[0.05]"}`}
                                    >
                                        <p className="truncate">{conversation.title}</p>
                                        {failedId === conversation.id && (
                                            <p className="text-[11px] text-red-600 dark:text-red-400">Couldn&apos;t delete. Try again.</p>
                                        )}
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => {
                                            setFailedId(null);
                                            setConfirmingId(conversation.id);
                                        }}
                                        disabled={isBusy}
                                        aria-label={`Delete chat: ${conversation.title}`}
                                        title="Delete chat"
                                        className={`absolute right-1.5 top-2 rounded-md p-1 text-slate-400 transition-opacity
                                            hover:bg-black/5 hover:text-red-600 dark:text-neutral-500 dark:hover:bg-white/10 dark:hover:text-red-400
                                            focus-visible:opacity-100 disabled:cursor-not-allowed disabled:opacity-0
                                            ${isMobile ? "opacity-100" : "opacity-0 group-hover:opacity-100"}`}
                                    >
                                        <Trash2 size={14} />
                                    </button>
                                </li>
                            );
                        })}
                    </ul>
                ) : (
                    <p className="px-2.5 text-sm text-slate-500 dark:text-neutral-600">No conversations yet</p>
                )}
                {!isSignedIn && (
                    <p className="mt-3 px-2.5 text-[11px] leading-relaxed text-slate-500 dark:text-neutral-500">
                        Sign in to save your chats.
                    </p>
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
                    <p className="text-[11px] leading-relaxed text-slate-500 dark:text-neutral-500">Grounded in the Bible Dictionary, Topical Guide, Gospel Topics, the Institute manuals (Book of Mormon, Foundations of the Restoration) and Church History Topics.</p>
                    <p className="text-[11px] leading-relaxed text-slate-500 dark:text-neutral-500">Not affiliated with The Church of Jesus Christ of Latter-day Saints.</p>
                </div>
            )}
        </aside>
    );
}
