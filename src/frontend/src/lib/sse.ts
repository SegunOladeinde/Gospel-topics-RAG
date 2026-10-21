// Parser for the backend's server-sent events (see /api/v1/query/stream).
// Network chunks can split an event anywhere, even mid-character, so text
// is buffered until a blank line ends each event. Pure (no React) so
// tests/sse.test.mjs can run it with plain `node --test`.

export interface SseEvent {
  event: string;
  data: unknown;
}

export function createSseParser(onEvent: (event: SseEvent) => void) {
  let buffer = "";

  const flush = (block: string) => {
    let event = "message";
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (dataLines.length === 0) return;
    const raw = dataLines.join("\n");
    let data: unknown = raw;
    try {
      data = JSON.parse(raw);
    } catch {
      // Not JSON: pass the text through as-is.
    }
    onEvent({ event, data });
  };

  return {
    // Feed decoded text as it arrives.
    push(text: string) {
      buffer += text.replace(/\r\n/g, "\n");
      let end = buffer.indexOf("\n\n");
      while (end !== -1) {
        flush(buffer.slice(0, end));
        buffer = buffer.slice(end + 2);
        end = buffer.indexOf("\n\n");
      }
    },
    // Call once the stream ends, for a final event without a trailing blank line.
    end() {
      if (buffer.trim()) flush(buffer);
      buffer = "";
    },
  };
}
