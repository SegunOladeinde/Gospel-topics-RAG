// Rules for the chat input box. Pure (no React) so tests/chatInput.test.mjs
// can run them with plain `node --test`.

// Same limit as MAX_QUESTION_CHARS in src/backend/main.py.
export const MAX_QUESTION_CHARS = 2000;
// The character counter only appears once a question gets long.
export const COUNTER_FROM_CHARS = 1500;

interface KeyInfo {
  key: string;
  shiftKey: boolean;
  // True while an IME (e.g. Japanese, Chinese, Korean input) is composing a
  // character; Enter then confirms the character rather than sending.
  isComposing: boolean;
  // Some browsers report composition only via keyCode 229.
  keyCode: number;
}

// Enter sends; Shift+Enter (and Enter while composing) makes a new line.
export function isSendKey({ key, shiftKey, isComposing, keyCode }: KeyInfo): boolean {
  return key === "Enter" && !shiftKey && !isComposing && keyCode !== 229;
}

// "1,612 / 2,000" once the question is long enough to need it, else null.
export function counterText(length: number): string | null {
  if (length < COUNTER_FROM_CHARS) return null;
  return `${length.toLocaleString("en-US")} / ${MAX_QUESTION_CHARS.toLocaleString("en-US")}`;
}
