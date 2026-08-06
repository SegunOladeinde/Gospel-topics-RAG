"use client";

import { useState, FormEvent } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
}

const API_URL = "http://localhost:8000/api/v1/query";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const question = input.trim();
    if (!question || isLoading) return;

    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        throw new Error(`Request failed with status ${res.status}`);
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

  return (
    <div className="flex min-h-screen flex-col bg-zinc-50 dark:bg-zinc-950">
      <header className="border-b border-zinc-200 bg-white px-6 py-4 dark:border-zinc-800 dark:bg-zinc-900">
        <h1 className="text-xl font-semibold text-zinc-900 dark:text-zinc-50">
          LDS Doctrinal RAG
        </h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Ask questions grounded in scraped LDS source material.
        </p>
      </header>

      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-4 overflow-y-auto px-4 py-6">
        {messages.length === 0 && (
          <div className="mt-16 text-center text-zinc-400 dark:text-zinc-600">
            Ask a question to get started.
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${
              message.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                message.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-white text-zinc-900 shadow-sm dark:bg-zinc-800 dark:text-zinc-50"
              }`}
            >
              <p className="whitespace-pre-wrap text-sm leading-relaxed">
                {message.content}
              </p>

              {message.role === "assistant" &&
                message.sources &&
                message.sources.length > 0 && (
                  <div className="mt-3 border-t border-zinc-200 pt-2 dark:border-zinc-700">
                    <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                      Sources
                    </p>
                    <ul className="space-y-1">
                      {message.sources.map((source, sourceIndex) => (
                        <li
                          key={sourceIndex}
                          className="text-xs text-zinc-500 dark:text-zinc-400"
                        >
                          {source}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-2xl bg-white px-4 py-3 text-sm text-zinc-400 shadow-sm dark:bg-zinc-800 dark:text-zinc-500">
              Thinking…
            </div>
          </div>
        )}

        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-400">
            {error}
          </div>
        )}
      </main>

      <form
        onSubmit={handleSubmit}
        className="border-t border-zinc-200 bg-white px-4 py-4 dark:border-zinc-800 dark:bg-zinc-900"
      >
        <div className="mx-auto flex max-w-3xl gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a doctrinal question..."
            disabled={isLoading}
            className="flex-1 rounded-full border border-zinc-300 bg-white px-4 py-2 text-sm text-zinc-900 outline-none focus:border-blue-500 disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-50"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="rounded-full bg-blue-600 px-5 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
