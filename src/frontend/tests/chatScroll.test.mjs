// Tests for src/lib/chatScroll.ts. Run with `npm test`.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { scrollActionFor } from "../src/lib/chatScroll.ts";

const snap = (count, lastRole, hasError = false) => ({ count, lastRole, hasError });

describe("scrollActionFor", () => {
  it("follows your question to the bottom", () => {
    assert.equal(scrollActionFor(snap(2, "assistant"), snap(3, "user"), false), "bottom");
    // Even if you'd scrolled up earlier: asking again means you're back.
    assert.equal(scrollActionFor(snap(2, "assistant"), snap(3, "user"), true), "bottom");
  });

  it("shows the start of a new answer, not its end", () => {
    assert.equal(scrollActionFor(snap(3, "user"), snap(4, "assistant"), false), "message-start");
  });

  it("leaves you alone if you scrolled up while waiting", () => {
    assert.equal(scrollActionFor(snap(3, "user"), snap(4, "assistant"), true), "none");
  });

  it("doesn't move while an answer streams in (same message count)", () => {
    assert.equal(scrollActionFor(snap(4, "assistant"), snap(4, "assistant"), false), "none");
  });

  it("shows an error that just appeared", () => {
    assert.equal(scrollActionFor(snap(3, "user"), snap(3, "user", true), false), "bottom");
    assert.equal(scrollActionFor(snap(3, "user", true), snap(3, "user", true), false), "none");
  });

  it("jumps to the end of a chat you open or switch to", () => {
    assert.equal(scrollActionFor(snap(0, null), snap(6, "assistant"), false), "bottom-instant");
    assert.equal(scrollActionFor(snap(6, "assistant"), snap(2, "assistant"), false), "bottom-instant");
  });

  it("does nothing for an empty chat", () => {
    assert.equal(scrollActionFor(snap(4, "assistant"), snap(0, null), false), "none");
  });
});
