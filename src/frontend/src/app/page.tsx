"use client";

import { useState, useSyncExternalStore } from "react";
import { useChat } from "./hooks/useChat";
import { Sidebar } from "./components/Sidebar";
import { ChatHeader } from "./components/ChatHeader";
import { BentoWelcomeGrid } from "./components/BentoWelcomeGrid";
import { ChatFeed } from "./components/ChatFeed";
import { ChatInput } from "./components/ChatInput";

const MOBILE_BREAKPOINT = "(max-width: 767px)";

function subscribeMobile(cb: () => void) {
  const mql = window.matchMedia(MOBILE_BREAKPOINT);
  mql.addEventListener("change", cb);
  return () => mql.removeEventListener("change", cb);
}

function useIsMobile() {
  return useSyncExternalStore(
    subscribeMobile,
    () => window.matchMedia(MOBILE_BREAKPOINT).matches,
    () => false
  );
}

export default function Home() {
  const { messages, input, setInput, isLoading, error, sendQuestion, handleNewChat } =
    useChat();

  const isMobile = useIsMobile();
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  // Desktop sidebar now tracks collapsed vs expanded gracefully via layout push
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  const isSidebarOpen = isMobile ? isMobileSidebarOpen : true;

  const toggleSidebar = () => {
    if (isMobile) {
      setIsMobileSidebarOpen((p) => !p);
    } else {
      setIsSidebarCollapsed((p) => !p);
    }
  };

  const closeMobileSidebar = () => setIsMobileSidebarOpen(false);
  const firstUserMessage = messages.find((m) => m.role === "user")?.content;

  return (
    <div className="flex h-dvh w-full overflow-hidden">
      {/* Mobile scrim */}
      {isMobile && isMobileSidebarOpen && (
        <div
          aria-hidden
          onClick={closeMobileSidebar}
          className="fixed inset-0 z-30 bg-black/50 backdrop-blur-sm"
        />
      )}

      {/* Sidebar now coordinates closely with its own internal expansion state */}
      <Sidebar
        isOpen={isSidebarOpen}
        onClose={closeMobileSidebar}
        isCollapsed={!isMobile && isSidebarCollapsed}
        onToggleCollapse={toggleSidebar}
        onNewChat={handleNewChat}
        firstMessage={firstUserMessage}
        isMobile={isMobile}
      />

      <main className="flex min-w-0 flex-1 flex-col transition-all duration-300">
        {!isMobile && isSidebarCollapsed && (
          <ChatHeader onToggleSidebar={toggleSidebar} collapsedView />
        )}
        {isMobile && <ChatHeader onToggleSidebar={toggleSidebar} />}

        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <BentoWelcomeGrid onSelect={sendQuestion} disabled={isLoading} />
          ) : (
            <ChatFeed messages={messages} isLoading={isLoading} error={error} />
          )}
        </div>

        <ChatInput
          value={input}
          onChange={setInput}
          onSubmit={sendQuestion}
          isLoading={isLoading}
        />
      </main>
    </div>
  );
}
