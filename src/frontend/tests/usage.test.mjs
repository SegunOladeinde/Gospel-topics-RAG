// Tests for src/lib/usage.ts and src/lib/backendUrl.ts. Run with `npm test`.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { backendEndpoint } from "../src/lib/backendUrl.ts";
import {
  MAX_REQUEST_MESSAGES,
  formatResetTime,
  limitMessage,
  shouldPromptSignIn,
  toChatError,
  trimForRequest,
  usageLabel,
} from "../src/lib/usage.ts";

const NOW = new Date("2026-10-21T12:00:00Z");
const user = (over) => ({ is_guest: false, is_admin: false, limit: 7, used: 0, remaining: 7, reset_at: null, ...over });
const guest = (over) => ({ is_guest: true, is_admin: false, limit: 1, used: 0, remaining: 1, reset_at: null, ...over });
const admin = { is_guest: false, is_admin: true, limit: null, used: 0, remaining: null, reset_at: null };

describe("formatResetTime", () => {
  it("shows just the time when the reset is today", () => {
    assert.equal(formatResetTime("2026-10-21T15:42:00Z", NOW, "en-US", "UTC"), "3:42 PM");
  });

  it("adds the weekday when the reset is another day", () => {
    assert.equal(formatResetTime("2026-10-22T09:05:00Z", NOW, "en-US", "UTC"), "Thu 9:05 AM");
  });

  it("uses the viewer's time zone to decide what 'today' is", () => {
    // 01:00 UTC on the 22nd is still the 21st in New York.
    assert.equal(formatResetTime("2026-10-22T01:00:00Z", NOW, "en-US", "America/New_York"), "9:00 PM");
  });
});

describe("toChatError", () => {
  it("keeps the limit code and reset time from a 429", () => {
    assert.deepEqual(toChatError(429, { detail: "USER_LIMIT_REACHED", reset_at: "2026-10-21T15:42:00Z" }), {
      message: "USER_LIMIT_REACHED",
      code: "USER_LIMIT_REACHED",
      resetAt: "2026-10-21T15:42:00Z",
    });
    assert.equal(toChatError(429, { detail: "GUEST_LIMIT_REACHED", reset_at: null }).code, "GUEST_LIMIT_REACHED");
  });

  it("passes other errors through as plain messages", () => {
    assert.deepEqual(toChatError(503, { detail: "Waking up the server" }), { message: "Waking up the server" });
    assert.deepEqual(toChatError(500, null), { message: "Request failed (500)" });
    // The slowapi burst limit is also a 429 but not one of our codes.
    assert.equal(toChatError(429, { error: "Rate limit exceeded" }).code, undefined);
  });
});

describe("limitMessage", () => {
  it("tells signed-in users when they can ask again", () => {
    const message = limitMessage(
      { message: "USER_LIMIT_REACHED", code: "USER_LIMIT_REACHED", resetAt: "2026-10-21T15:42:00Z" },
      NOW,
      "en-US",
      "UTC"
    );
    assert.match(message, /ask again at 3:42 PM/);
  });

  it("falls back gracefully without a reset time", () => {
    assert.match(limitMessage({ message: "x", code: "USER_LIMIT_REACHED", resetAt: null }), /try again later/);
  });

  it("asks guests to sign in", () => {
    assert.match(limitMessage({ message: "x", code: "GUEST_LIMIT_REACHED" }), /Sign in/);
  });
});

describe("usageLabel", () => {
  it("counts down for signed-in users", () => {
    assert.equal(usageLabel(user({ used: 2, remaining: 5 })), "5 of 7 questions left");
  });

  it("shows the reset time when out of questions", () => {
    const label = usageLabel(user({ used: 7, remaining: 0, reset_at: "2026-10-21T15:42:00Z" }), NOW, "en-US", "UTC");
    assert.equal(label, "No questions left · more at 3:42 PM");
  });

  it("nudges guests to sign in", () => {
    assert.match(usageLabel(guest()), /1 free question/);
    assert.match(usageLabel(guest({ used: 1, remaining: 0 })), /Free question used/);
  });

  it("shows nothing for admins or when limits are off", () => {
    assert.equal(usageLabel(admin), null);
    assert.equal(usageLabel(null), null);
  });
});

describe("shouldPromptSignIn", () => {
  it("is true only for a signed-out guest with no questions left", () => {
    assert.equal(shouldPromptSignIn(guest({ remaining: 0 }), false), true);
    assert.equal(shouldPromptSignIn(guest({ remaining: 1 }), false), false);
    assert.equal(shouldPromptSignIn(user({ remaining: 0 }), true), false);
    assert.equal(shouldPromptSignIn(null, false), false);
  });

  it("doesn't prompt someone who just signed in while guest usage is still showing", () => {
    assert.equal(shouldPromptSignIn(guest({ remaining: 0 }), true), false);
  });
});

describe("trimForRequest", () => {
  it("keeps the newest messages within the backend's limit", () => {
    const messages = Array.from({ length: 25 }, (_, i) => i);
    const trimmed = trimForRequest(messages);
    assert.equal(trimmed.length, MAX_REQUEST_MESSAGES);
    assert.equal(trimmed.at(-1), 24);
  });
});

describe("backendEndpoint", () => {
  const url = "https://gospel-topics-rag.onrender.com/api/v1/query";

  it("resolves sibling endpoints of the query URL", () => {
    assert.equal(backendEndpoint(url, "usage"), "https://gospel-topics-rag.onrender.com/api/v1/usage");
    assert.equal(
      backendEndpoint(url, ["conversations", "abc123"]),
      "https://gospel-topics-rag.onrender.com/api/v1/conversations/abc123"
    );
  });

  it("encodes ids so they can't escape /api/v1/", () => {
    assert.equal(
      backendEndpoint(url, ["conversations", "../../health"]),
      "https://gospel-topics-rag.onrender.com/api/v1/conversations/..%2F..%2Fhealth"
    );
  });
});
