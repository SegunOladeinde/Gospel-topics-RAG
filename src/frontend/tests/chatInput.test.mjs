// Tests for src/lib/chatInput.ts. Run with `npm test`.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { COUNTER_FROM_CHARS, MAX_QUESTION_CHARS, counterText, isSendKey } from "../src/lib/chatInput.ts";

const key = (over) => ({ key: "Enter", shiftKey: false, isComposing: false, keyCode: 13, ...over });

describe("isSendKey", () => {
  it("sends on Enter", () => {
    assert.equal(isSendKey(key()), true);
  });

  it("makes a new line on Shift+Enter", () => {
    assert.equal(isSendKey(key({ shiftKey: true })), false);
  });

  it("never sends while an IME is composing", () => {
    assert.equal(isSendKey(key({ isComposing: true })), false);
    assert.equal(isSendKey(key({ keyCode: 229 })), false); // composition reported via keyCode only
  });

  it("ignores other keys", () => {
    assert.equal(isSendKey(key({ key: "a", keyCode: 65 })), false);
  });
});

describe("counterText", () => {
  it("stays hidden for normal-length questions", () => {
    assert.equal(counterText(0), null);
    assert.equal(counterText(COUNTER_FROM_CHARS - 1), null);
  });

  it("appears from 1,500 characters", () => {
    assert.equal(counterText(1500), "1,500 / 2,000");
    assert.equal(counterText(MAX_QUESTION_CHARS), "2,000 / 2,000");
  });

  it("matches the backend's limit", () => {
    assert.equal(MAX_QUESTION_CHARS, 2000);
  });
});
