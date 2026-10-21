// Tests for src/lib/sse.ts. Run with `npm test`.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { createSseParser } from "../src/lib/sse.ts";

const collect = () => {
  const events = [];
  return { events, parser: createSseParser((e) => events.push(e)) };
};

describe("createSseParser", () => {
  it("parses token and done events", () => {
    const { events, parser } = collect();
    parser.push('event: token\ndata: {"text":"Hel"}\n\nevent: token\ndata: {"text":"lo"}\n\n');
    parser.push('event: done\ndata: {"answer":"Hello","sources":[]}\n\n');
    assert.deepEqual(events, [
      { event: "token", data: { text: "Hel" } },
      { event: "token", data: { text: "lo" } },
      { event: "done", data: { answer: "Hello", sources: [] } },
    ]);
  });

  it("handles events split across chunks at any point", () => {
    const full = 'event: token\ndata: {"text":"Faith"}\n\nevent: done\ndata: {"answer":"Faith"}\n\n';
    for (let cut = 1; cut < full.length; cut++) {
      const { events, parser } = collect();
      parser.push(full.slice(0, cut));
      parser.push(full.slice(cut));
      assert.equal(events.length, 2, `cut at ${cut}`);
      assert.deepEqual(events[0], { event: "token", data: { text: "Faith" } });
    }
  });

  it("accepts CRLF line endings", () => {
    const { events, parser } = collect();
    parser.push('event: error\r\ndata: {"detail":"boom"}\r\n\r\n');
    assert.deepEqual(events, [{ event: "error", data: { detail: "boom" } }]);
  });

  it("keeps unicode intact", () => {
    const { events, parser } = collect();
    parser.push('event: token\ndata: {"text":"Alma’s “faith” — ✓"}\n\n');
    assert.equal(events[0].data.text, "Alma’s “faith” — ✓");
  });

  it("flushes a final event without a trailing blank line on end()", () => {
    const { events, parser } = collect();
    parser.push('event: done\ndata: {"answer":"x"}');
    assert.equal(events.length, 0);
    parser.end();
    assert.deepEqual(events, [{ event: "done", data: { answer: "x" } }]);
  });

  it("passes non-JSON data through as text", () => {
    const { events, parser } = collect();
    parser.push("data: hello\n\n");
    assert.deepEqual(events, [{ event: "message", data: "hello" }]);
  });
});

it("keep-alive comments produce no events", () => {
  const events = [];
  const parser = createSseParser((e) => events.push(e));
  parser.push(": keep-alive\n\n: keep-alive\n\nevent: token\ndata: {\"text\":\"Hi\"}\n\n");
  parser.end();
  assert.deepEqual(events, [{ event: "token", data: { text: "Hi" } }]);
});
