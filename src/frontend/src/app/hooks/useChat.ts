"use client";
import { useState } from "react";
import type { Message } from "../types";

const API_URL = "/api/query";

export function useChat() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleNewChat = () => {
        setMessages([]);
        setError(null);
        setInput("");
    };

    const sendQuestion = async (question: string) => {
        if (!question.trim() || isLoading) return;

        const updatedMessages: Message[] = [
            ...messages,
            { role: "user", content: question },
        ];
        setMessages(updatedMessages);
        setInput("");
        setIsLoading(true);
        setError(null);

        try {
            const res = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ messages: updatedMessages }),
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(
                    (errData as { detail?: string })?.detail ??
                    `Request failed (${res.status})`
                );
            }

            const data = await res.json();
            setMessages((prev) => [
                ...prev,
                {
                    role: "assistant",
                    content: data.answer ?? "No answer was returned.",
                    sources: data.sources ?? [],
                },
            ]);
        } catch (err) {
            setError(
                err instanceof Error
                    ? err.message
                    : "Something went wrong while contacting the server."
            );
        } finally {
            setIsLoading(false);
        }
    };

    return { messages, input, setInput, isLoading, error, sendQuestion, handleNewChat };
}
