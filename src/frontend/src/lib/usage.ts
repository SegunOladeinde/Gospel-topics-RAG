// Question allowance helpers shared by the chat hook and UI. Pure functions
// with no Next.js/React imports so tests/usage.test.mjs can run them with
// plain `node --test`.

// Mirrors Usage in src/backend/usage.py. limit/remaining are null for admins.
export interface Usage {
  is_guest: boolean;
  is_admin: boolean;
  limit: number | null;
  used: number;
  remaining: number | null;
  reset_at: string | null;
}

export type LimitCode = "GUEST_LIMIT_REACHED" | "USER_LIMIT_REACHED";

export interface ChatError {
  message: string;
  code?: LimitCode;
  resetAt?: string | null;
}

// The backend accepts at most this many messages per request.
export const MAX_REQUEST_MESSAGES = 20;

// Keep the newest messages; the backend trims history further for the model.
export function trimForRequest<T>(messages: T[]): T[] {
  return messages.slice(-MAX_REQUEST_MESSAGES);
}

// Turn a failed /api/query response body into something the UI can show.
export function toChatError(status: number, body: unknown): ChatError {
  const data = (body ?? {}) as { detail?: unknown; reset_at?: unknown };
  const detail = typeof data.detail === "string" ? data.detail : undefined;
  if (status === 429 && (detail === "GUEST_LIMIT_REACHED" || detail === "USER_LIMIT_REACHED")) {
    return {
      message: detail,
      code: detail,
      resetAt: typeof data.reset_at === "string" ? data.reset_at : null,
    };
  }
  return { message: detail ?? `Request failed (${status})` };
}

// "3:42 PM" if it's today where the user is, otherwise "Thu 3:42 PM".
// locale/timeZone are only passed by tests; the browser's own are used otherwise.
export function formatResetTime(
  resetAt: string,
  now: Date = new Date(),
  locale?: string,
  timeZone?: string
): string {
  const reset = new Date(resetAt);
  const day = (date: Date) => date.toLocaleDateString(locale ?? "en-CA", { timeZone });
  const sameDay = day(reset) === day(now);
  return reset.toLocaleString(locale, {
    timeZone,
    hour: "numeric",
    minute: "2-digit",
    ...(sameDay ? {} : { weekday: "short" }),
  });
}

export function limitMessage(error: ChatError, now?: Date, locale?: string, timeZone?: string): string {
  if (error.code === "GUEST_LIMIT_REACHED") {
    return "You've used your free question. Sign in to keep asking: signed-in users get 7 questions a day.";
  }
  if (error.code === "USER_LIMIT_REACHED") {
    return error.resetAt
      ? `You've reached your limit of 7 questions in 24 hours. You can ask again at ${formatResetTime(
          error.resetAt,
          now,
          locale,
          timeZone
        )}.`
      : "You've reached your limit of 7 questions in 24 hours. Please try again later.";
  }
  return error.message;
}

// Text for the remaining-questions indicator, or null to show nothing
// (admins get the "Admin · unlimited" badge instead; no usage = no limits).
export function usageLabel(usage: Usage | null, now?: Date, locale?: string, timeZone?: string): string | null {
  if (!usage || usage.is_admin || usage.limit === null || usage.remaining === null) return null;
  if (usage.is_guest) {
    return usage.remaining > 0
      ? "1 free question · sign in for 7 a day"
      : "Free question used · sign in for 7 a day";
  }
  if (usage.remaining > 0) {
    return `${usage.remaining} of ${usage.limit} questions left`;
  }
  return usage.reset_at
    ? `No questions left · more at ${formatResetTime(usage.reset_at, now, locale, timeZone)}`
    : "No questions left";
}

// A signed-out visitor who has used their free question should get the
// sign-in modal instead of a request that's bound to be refused.
export function shouldPromptSignIn(usage: Usage | null, isSignedIn: boolean): boolean {
  return !isSignedIn && !!usage?.is_guest && usage.remaining === 0;
}
