// Where the chat should scroll after the message list changes. Pure, so
// tests/chatScroll.test.mjs can cover it with plain `node --test`.

export interface FeedSnapshot {
  count: number;
  lastRole: "user" | "assistant" | null;
  hasError: boolean;
}

export type ScrollAction =
  | "none"
  | "bottom" // smooth, to the end (your question and the thinking dots)
  | "bottom-instant" // a whole chat was loaded or switched to
  | "message-start"; // smooth, to the top of the newest answer

export function scrollActionFor(
  prev: FeedSnapshot,
  next: FeedSnapshot,
  userScrolledUp: boolean
): ScrollAction {
  if (next.count === 0) return "none";

  if (next.count === prev.count + 1) {
    // You just asked: follow your question down, always.
    if (next.lastRole === "user") return "bottom";
    // The answer arrived: show where it starts, so it reads from the top,
    // unless you've scrolled up to read something else meanwhile.
    if (next.lastRole === "assistant") return userScrolledUp ? "none" : "message-start";
  }

  if (next.count === prev.count) {
    // Same messages (e.g. an answer streaming in): leave the view alone,
    // except to show an error that just appeared.
    return next.hasError && !prev.hasError && !userScrolledUp ? "bottom" : "none";
  }

  // Anything else is a different chat: start at its end.
  return "bottom-instant";
}
