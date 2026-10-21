"use client";
import { useEffect, useRef, useState } from "react";
import { useAuth, useClerk } from "@clerk/nextjs";
import type { ConversationSummary, Message, SourceLink } from "../types";
import {
    shouldPromptSignIn,
    toChatError,
    trimForRequest,
    type ChatError,
    type Usage,
} from "@/lib/usage";
import { createSseParser } from "@/lib/sse";

const STREAM_URL = "/api/query/stream";
const TITLE_MAX_CHARS = 80; // same as the backend's make_title()

// The final "done" event of a streamed answer (QueryResponse on the backend).
interface QueryDone {
    answer: string;
    sources?: string[];
    source_links?: SourceLink[];
    conversation_id?: string | null;
    usage?: Usage | null;
}

interface StoredMessage {
    role: "user" | "assistant";
    content: string;
    sources: string[];
    source_links: SourceLink[];
}

function makeTitle(question: string): string {
    const title = question.split(/\s+/).filter(Boolean).join(" ");
    return title.length <= TITLE_MAX_CHARS ? title : `${title.slice(0, TITLE_MAX_CHARS - 1).trimEnd()}…`;
}

// The caller's allowance, or undefined if it couldn't be fetched.
async function fetchUsage(): Promise<Usage | null | undefined> {
    try {
        const res = await fetch("/api/usage");
        if (!res.ok) return undefined;
        const data: { usage?: Usage | null } = await res.json();
        return data.usage ?? null;
    } catch {
        return undefined;
    }
}

// Chat state lives in React state (see CLAUDE.md). Signed-in users' chats are
// also saved by the backend as they're answered, and loaded back from it, so
// they survive reloads and "New chat" never loses the previous one. Guests'
// chats are kept for the life of the tab only.
export function useChat() {
    const { isLoaded, isSignedIn, userId } = useAuth();
    const { openSignIn } = useClerk();

    const [messages, setMessages] = useState<Message[]>([]);
    const [conversationId, setConversationId] = useState<string | null>(null);
    const [conversations, setConversations] = useState<ConversationSummary[]>([]);
    // Messages of every chat opened in this tab, so switching back is instant.
    const [chatCache, setChatCache] = useState<Record<string, Message[]>>({});
    const [usage, setUsage] = useState<Usage | null>(null);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [isLoadingConversation, setIsLoadingConversation] = useState(false);
    const [error, setError] = useState<ChatError | null>(null);
    // The chat on screen, readable after an await (state would be stale).
    const activeIdRef = useRef<string | null>(null);
    // The request for the answer streaming in, so Stop can cancel it.
    const abortRef = useRef<AbortController | null>(null);

    const showConversation = (id: string | null, chatMessages: Message[]) => {
        activeIdRef.current = id;
        setConversationId(id);
        setMessages(chatMessages);
    };

    // Signing out must not leave the previous user's chats on screen for the
    // next person at this computer. (Signing in keeps the current chat.)
    const [prevUserId, setPrevUserId] = useState(userId);
    if (userId !== prevUserId) {
        setPrevUserId(userId);
        if (!userId) {
            setMessages([]);
            setConversationId(null);
            setConversations([]);
            setChatCache({});
            setError(null);
        }
    }

    // Keeps the ref in step when conversationId changes outside
    // showConversation (signing out).
    useEffect(() => {
        activeIdRef.current = conversationId;
    }, [conversationId]);

    // Load the allowance and the saved chats whenever who's asking changes.
    useEffect(() => {
        if (!isLoaded) return;
        let cancelled = false;

        fetchUsage().then((next) => {
            if (!cancelled && next !== undefined) setUsage(next);
        });

        if (userId) {
            fetch("/api/conversations")
                .then((res) => (res.ok ? res.json() : null))
                .then((data: { conversations?: { id: string; title: string }[] } | null) => {
                    if (cancelled || !data?.conversations) return;
                    const saved = data.conversations.map((c) => ({ id: c.id, title: c.title, saved: true }));
                    // Keep any unsaved chats from before signing in, after the saved ones.
                    setConversations((prev) => [...saved, ...prev.filter((c) => !c.saved)]);
                })
                .catch(() => {});
        }

        return () => {
            cancelled = true;
        };
    }, [isLoaded, userId]);

    const handleNewChat = () => {
        if (isLoading) return;
        showConversation(null, []);
        setError(null);
        setInput("");
    };

    const selectConversation = async (id: string) => {
        if (isLoading || id === activeIdRef.current) return;
        setError(null);
        const cached = chatCache[id];
        if (cached) {
            showConversation(id, cached);
            return;
        }
        showConversation(id, []);
        setIsLoadingConversation(true);
        try {
            const res = await fetch(`/api/conversations/${encodeURIComponent(id)}`);
            const data = await res.json().catch(() => null);
            if (!res.ok) throw toChatError(res.status, data);
            const loaded: Message[] = (data.messages as StoredMessage[]).map((m) => ({
                role: m.role,
                content: m.content,
                sources: m.sources,
                sourceLinks: m.source_links,
            }));
            setChatCache((prev) => ({ ...prev, [id]: loaded }));
            // Only show it if the user hasn't moved on to another chat meanwhile.
            if (activeIdRef.current === id) setMessages(loaded);
        } catch (err) {
            if (activeIdRef.current !== id) return;
            setError(
                err && typeof err === "object" && "message" in err
                    ? (err as ChatError)
                    : { message: "Couldn't load that chat. Please try again." }
            );
        } finally {
            setIsLoadingConversation(false);
        }
    };

    // Saved chats are deleted on the server first; unsaved (tab-only) chats
    // just disappear. Deleting the open chat leaves an empty new chat.
    // Returns false if the server refused, so the sidebar can say so.
    const deleteConversation = async (id: string): Promise<boolean> => {
        const target = conversations.find((c) => c.id === id);
        if (!target) return true;
        if (target.saved) {
            try {
                const res = await fetch(`/api/conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
                // 404: already gone (another tab), which is what we wanted anyway.
                if (!res.ok && res.status !== 404) return false;
            } catch {
                return false;
            }
        }
        setConversations((prev) => prev.filter((c) => c.id !== id));
        setChatCache((prev) => {
            const next = { ...prev };
            delete next[id];
            return next;
        });
        if (activeIdRef.current === id && !isLoading) {
            showConversation(null, []);
            setError(null);
        }
        return true;
    };

    // Remember a finished (or stopped) turn in the sidebar and the cache.
    const keepChat = (id: string, saved: boolean, chatMessages: Message[]) => {
        setChatCache((prev) => ({ ...prev, [id]: chatMessages }));
        setConversations((prev) => {
            const existing = prev.find((c) => c.id === id);
            const entry: ConversationSummary = {
                id,
                title: existing?.title ?? makeTitle(chatMessages[0].content),
                saved: saved || Boolean(existing?.saved),
            };
            // Most recently active first, like the backend's list.
            return [entry, ...prev.filter((c) => c.id !== id)];
        });
    };

    // Stop the answer that's streaming in; what arrived so far stays.
    const stopGeneration = () => {
        abortRef.current?.abort();
    };

    const sendQuestion = async (question: string) => {
        if (!question.trim() || isLoading || isLoadingConversation) return;

        // A guest who has used their free question gets the sign-in modal
        // straight away instead of a request that's bound to be refused.
        if (shouldPromptSignIn(usage, !!isSignedIn)) {
            openSignIn();
            return;
        }

        const askedIn = activeIdRef.current;
        const savedChat = conversations.some((c) => c.id === askedIn && c.saved);
        const updatedMessages: Message[] = [
            ...messages,
            { role: "user", content: question },
        ];
        setMessages(updatedMessages);
        setInput("");
        setIsLoading(true);
        setError(null);

        const controller = new AbortController();
        abortRef.current = controller;
        let answer = "";
        const stillHere = () => activeIdRef.current === askedIn;
        const withAnswer = (extra: Partial<Message>): Message[] => [
            ...updatedMessages,
            { role: "assistant", content: answer, ...extra },
        ];

        try {
            const res = await fetch(STREAM_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    // Only role + content: sources aren't needed to answer.
                    messages: trimForRequest(updatedMessages).map(({ role, content }) => ({ role, content })),
                    ...(savedChat && askedIn ? { conversation_id: askedIn } : {}),
                }),
                signal: controller.signal,
            });

            if (!res.ok) {
                const data = await res.json().catch(() => null);
                const chatError = toChatError(res.status, data);
                if (data?.usage) setUsage(data.usage);
                if (chatError.code === "GUEST_LIMIT_REACHED") openSignIn();
                throw chatError;
            }

            const result: { done: QueryDone | null; error: string | null } = { done: null, error: null };
            if ((res.headers.get("content-type") ?? "").includes("application/json")) {
                // A backend without streaming: the whole answer at once.
                result.done = await res.json();
                answer = result.done?.answer ?? "";
            } else if (res.body) {
                const parser = createSseParser(({ event, data }) => {
                    if (event === "token") {
                        answer += (data as { text?: string }).text ?? "";
                        if (stillHere()) setMessages(withAnswer({ streaming: true }));
                    } else if (event === "done") {
                        result.done = data as QueryDone;
                    } else if (event === "error") {
                        result.error = (data as { detail?: string }).detail ?? null;
                    }
                });
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                for (;;) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    parser.push(decoder.decode(value, { stream: true }));
                }
                parser.push(decoder.decode());
                parser.end();
            }

            if (result.error || !result.done) {
                // Keep any partial answer on screen above the error.
                if (answer && stillHere()) setMessages(withAnswer({ stopped: true }));
                throw { message: result.error ?? "The answer was interrupted. Please try again." } as ChatError;
            }

            const data = result.done;
            const finalMessages = withAnswer({
                content: data.answer || answer || "No answer was returned.",
                sources: data.sources ?? [],
                sourceLinks: data.source_links,
            });
            if (data.usage !== undefined) setUsage(data.usage);

            // The backend returns the saved conversation's id for signed-in
            // users; otherwise keep the chat in this tab under a local id.
            const id: string = data.conversation_id ?? askedIn ?? `local-${Date.now()}`;
            keepChat(id, Boolean(data.conversation_id), finalMessages);
            // Don't yank the user back if they switched chats while waiting.
            if (stillHere()) showConversation(id, finalMessages);
        } catch (err) {
            if (err instanceof DOMException && err.name === "AbortError") {
                // Stopped by the user. The backend counts the question but
                // doesn't save a half answer, so keep it in this tab only.
                const stoppedMessages = withAnswer({ stopped: true });
                const id = askedIn ?? `local-${Date.now()}`;
                keepChat(id, false, stoppedMessages);
                if (stillHere()) showConversation(id, stoppedMessages);
                // The stopped question still counted; no "done" event brought
                // the new allowance, so ask for it.
                fetchUsage().then((next) => {
                    if (next !== undefined) setUsage(next);
                });
                return;
            }
            if (!stillHere()) return;
            setError(
                err && typeof err === "object" && "message" in err && typeof (err as ChatError).message === "string"
                    ? (err as ChatError)
                    : { message: "Something went wrong while contacting the server." }
            );
        } finally {
            if (abortRef.current === controller) abortRef.current = null;
            setIsLoading(false);
        }
    };

    return {
        messages,
        input,
        setInput,
        isLoading,
        isLoadingConversation,
        error,
        usage,
        conversations,
        conversationId,
        isSignedIn: !!isSignedIn,
        sendQuestion,
        handleNewChat,
        selectConversation,
        deleteConversation,
        stopGeneration,
    };
}
