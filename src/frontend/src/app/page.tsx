"use client";

import { useEffect, useRef, useState, useSyncExternalStore, SubmitEvent } from "react";
import { Send, MessageSquarePlus, BookOpen, Loader2, PanelLeft } from "lucide-react";
import ReactMarkdown from "react-markdown";

const MOBILE_QUERY = "(max-width: 767px)";

function subscribeToMobileQuery(onChange: () => void) {
  const mql = window.matchMedia(MOBILE_QUERY);
  mql.addEventListener("change", onChange);
  return () => mql.removeEventListener("change", onChange);
}

function useIsMobile() {
  return useSyncExternalStore(
    subscribeToMobileQuery,
    () => window.matchMedia(MOBILE_QUERY).matches,
    () => false
  );
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
}

const API_URL = "http://localhost:8000/api/v1/query";

const SUGGESTED_QUESTIONS = [
  "What is the significance of the Urim and Thummim?",
  "Explain the Word of Wisdom",
  "What does the Topical Guide say about faith?",
];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isMobile = useIsMobile();
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [isDesktopSidebarClosed, setIsDesktopSidebarClosed] = useState(false);
  const isSidebarOpen = isMobile ? isMobileSidebarOpen : !isDesktopSidebarClosed;

  const toggleSidebar = () => {
    if (isMobile) {
      setIsMobileSidebarOpen((prev) => !prev);
    } else {
      setIsDesktopSidebarClosed((prev) => !prev);
    }
  };

  const scrollAnchorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollAnchorRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleNewChat = () => {
    setMessages([]);
    setError(null);
    setInput("");
  };

  const sendQuestion = async (question: string) => {
    if (!question || isLoading) return;

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

  const handleSubmit = async (e: SubmitEvent<HTMLFormElement>) => {
    e.preventDefault();
    await sendQuestion(input.trim());
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-gray-50">
      {isSidebarOpen && (
        <div
          onClick={() => setIsMobileSidebarOpen(false)}
          className="fixed inset-0 z-30 bg-black/40 md:hidden"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-64 shrink-0 flex-col overflow-hidden bg-slate-900 text-slate-200 transition-all duration-300 ease-in-out md:static md:z-auto md:translate-x-0 ${
          isSidebarOpen ? "translate-x-0" : "-translate-x-full md:w-0"
        }`}
      >
        <div className="flex flex-col gap-1 px-5 py-5">
          <div className="flex items-center gap-2">
            <BookOpen size={22} className="shrink-0 text-blue-400" />
            <span className="text-sm font-semibold text-white">
              LDS Doctrinal RAG
            </span>
          </div>
          <span className="pl-[30px] text-xs text-slate-500">
            Guidance provided via Chat is AI Generated; please confirm
            through Prayer and Scripture study.
          </span>
          <span className="pl-[30px] text-xs text-slate-500">
            Ask gospel-related questions
          </span>
        </div>

        <div className="px-3">
          <button
            type="button"
            onClick={handleNewChat}
            className="flex w-full items-center gap-2 rounded-lg border border-slate-700 bg-slate-800/60 px-3 py-2.5 text-sm font-medium text-slate-100 transition-colors hover:bg-slate-700/70"
          >
            <MessageSquarePlus size={16} />
            New Chat
          </button>
        </div>

        <nav className="mt-6 flex-1 space-y-1 px-3">
          <p className="px-2 pb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
            Recent
          </p>
          {messages.length === 0 ? (
            <p className="px-2 text-sm text-slate-500">No conversations yet</p>
          ) : (
            <p className="truncate px-2 text-sm text-slate-400">
              {messages[0]?.content}
            </p>
          )}
        </nav>

        <div className="border-t border-slate-800 px-5 py-4 text-xs text-slate-500">
          Grounded in scraped LDS source material.
        </div>

        <div className="border-t border-slate-800 px-5 py-3 text-xs text-zinc-500">
          This project is not affiliated with The Church of Jesus Christ of
          Latter-day Saints.
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col bg-gray-50">
        <header className="flex items-center gap-3 border-b border-gray-200 bg-white px-4 py-3">
          <button
            type="button"
            onClick={toggleSidebar}
            className="rounded-lg p-2 text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-700"
            aria-label="Toggle sidebar"
          >
            <PanelLeft size={20} />
          </button>
          <div className="flex flex-col">
            <span className="text-sm font-medium text-gray-700">
              ASK Gospel Topics RAG
            </span>
            <span className="text-xs text-gray-400">
              Guidance provided via Chat is AI Generated; please confirm
              through Prayer and Scripture study.
            </span>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto scroll-smooth">
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-5 px-4 py-8">
            {messages.length === 0 && (
              <div className="mt-16 flex flex-col items-center gap-6 text-center">
                <div>
                  <h2 className="text-lg font-semibold text-gray-700">
                    Ask a doctrinal question to get started
                  </h2>
                  <p className="mt-1 text-sm text-gray-400">
                    Try one of these, or ask your own.
                  </p>
                  <p className="mt-2 text-xs text-gray-400">
                    Guidance Provide through Chat could be wrong, please confirm all
                    answers through Divinely Appointed source (Bishop, Institute or Seminary Teacher...) Prayer and Sripures Study.
                  </p>
                </div>

                <div className="grid w-full gap-3 sm:grid-cols-3">
                  {SUGGESTED_QUESTIONS.map((question) => (
                    <button
                      key={question}
                      type="button"
                      onClick={() => sendQuestion(question)}
                      disabled={isLoading}
                      className="rounded-xl border border-gray-200 bg-white p-4 text-left text-sm text-gray-700 shadow-sm transition-colors hover:border-blue-300 hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {question}
                    </button>
                  ))}
                </div>
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
                      : "bg-white text-gray-900 shadow-sm"
                  }`}
                >
                  {message.role === "user" ? (
                    <p className="whitespace-pre-wrap text-sm leading-relaxed">
                      {message.content}
                    </p>
                  ) : (
                    <div
                      className="max-w-none text-sm leading-relaxed break-words
                        [&_p]:my-2 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0
                        [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5
                        [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5
                        [&_li]:my-0.5
                        [&_h1]:my-2 [&_h1]:text-base [&_h1]:font-semibold
                        [&_h2]:my-2 [&_h2]:text-base [&_h2]:font-semibold
                        [&_h3]:my-2 [&_h3]:text-sm [&_h3]:font-semibold
                        [&_strong]:font-semibold
                        [&_a]:text-blue-600 [&_a]:underline
                        [&_code]:rounded [&_code]:bg-gray-100 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-xs
                        [&_blockquote]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-gray-300 [&_blockquote]:pl-3 [&_blockquote]:text-gray-600"
                    >
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>
                  )}

                  {message.role === "assistant" &&
                    message.sources &&
                    message.sources.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-1.5 border-t border-gray-100 pt-3">
                        {message.sources.map((source, sourceIndex) => (
                          <span
                            key={sourceIndex}
                            className="cursor-pointer rounded-full bg-slate-200 px-2.5 py-1 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-300"
                          >
                            {source}
                          </span>
                        ))}
                      </div>
                    )}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="flex max-w-[80%] items-center gap-2 rounded-2xl bg-white px-4 py-3 text-sm text-gray-400 shadow-sm">
                  <Loader2 size={14} className="animate-spin" />
                  Thinking...
                </div>
              </div>
            )}

            {error && (
              <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </div>
            )}

            <div ref={scrollAnchorRef} />
          </div>
        </div>

        <div className="sticky bottom-0 bg-gray-50 px-4 pb-6 pt-2">
          <form
            onSubmit={handleSubmit}
            className="mx-auto flex w-full max-w-3xl items-center gap-2 rounded-full bg-white px-2 py-2 shadow-lg ring-1 ring-black/5"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a doctrinal question..."
              disabled={isLoading}
              className="flex-1 rounded-full bg-transparent px-4 py-2 text-sm text-gray-900 outline-none disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-blue-600 text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
              aria-label="Send message"
            >
              <Send size={16} />
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}
