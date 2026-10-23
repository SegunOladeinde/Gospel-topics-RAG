// Built by the backend (src/backend/source_links.py). url is null when the
// source can't be mapped to a page, and the chip renders as plain text.
export interface SourceLink {
  id: string;
  label: string;
  url: string | null;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  sourceLinks?: SourceLink[];
  // Still arriving token by token.
  streaming?: boolean;
  // Stopped (or cut off) before the answer finished.
  stopped?: boolean;
  // Every claim was checked against the sources and supported (backend claims.py).
  verified?: boolean;
}

// One entry in the sidebar. saved=false for chats that only live in this tab
// (guests, or a deployment without a database).
export interface ConversationSummary {
  id: string;
  title: string;
  saved: boolean;
}
