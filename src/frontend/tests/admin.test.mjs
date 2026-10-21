// Tests for src/lib/admin.ts. Run with `npm test` (node --test; Node strips
// the TypeScript types itself, so no test framework is needed).
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { buildBackendHeaders, isAdminUser, parseAdminEmails } from "../src/lib/admin.ts";

const verified = (emailAddress) => ({ emailAddress, verification: { status: "verified" } });
const unverified = (emailAddress) => ({ emailAddress, verification: { status: "unverified" } });
const user = (...emailAddresses) => ({ emailAddresses });

describe("parseAdminEmails", () => {
  it("trims, lowercases and drops blanks", () => {
    assert.deepEqual(
      [...parseAdminEmails(" Admin@Example.com, ,other@example.com,, ")],
      ["admin@example.com", "other@example.com"]
    );
  });

  it("is empty when unset or blank", () => {
    assert.equal(parseAdminEmails(undefined).size, 0);
    assert.equal(parseAdminEmails("").size, 0);
    assert.equal(parseAdminEmails(" , ").size, 0);
  });
});

describe("isAdminUser", () => {
  const env = "admin@example.com, second@example.com";

  it("is true for a verified email in ADMIN_EMAILS, case-insensitively", () => {
    assert.equal(isAdminUser(user(verified("Admin@Example.COM")), env), true);
  });

  it("checks every email address on the account", () => {
    assert.equal(isAdminUser(user(verified("me@else.com"), verified("second@example.com")), env), true);
  });

  it("is false for an unverified matching email", () => {
    assert.equal(isAdminUser(user(unverified("admin@example.com")), env), false);
    assert.equal(
      isAdminUser(user({ emailAddress: "admin@example.com", verification: null }), env),
      false
    );
  });

  it("is false when only an unverified address matches, even next to a verified one", () => {
    assert.equal(isAdminUser(user(verified("me@else.com"), unverified("admin@example.com")), env), false);
  });

  it("is false for a verified email that isn't listed", () => {
    assert.equal(isAdminUser(user(verified("someone@example.com")), env), false);
  });

  it("makes nobody admin when ADMIN_EMAILS is empty or unset", () => {
    for (const empty of [undefined, "", "  ", ","]) {
      assert.equal(isAdminUser(user(verified("admin@example.com")), empty), false);
    }
  });

  it("is false with no signed-in user", () => {
    assert.equal(isAdminUser(null, env), false);
    assert.equal(isAdminUser(undefined, env), false);
  });
});

describe("buildBackendHeaders", () => {
  const spoofed = () =>
    new Headers({
      "x-user-role": "admin",
      "x-user-id": "someone_else",
      "x-api-key": "browser-supplied",
      "x-forwarded-for": "203.0.113.7",
    });

  it("ignores a spoofed X-User-Role from the browser", () => {
    const headers = buildBackendHeaders(spoofed(), { apiKey: "server-key", userId: "user_1", isAdmin: false });
    assert.equal(
      Object.keys(headers).some((name) => name.toLowerCase() === "x-user-role"),
      false
    );
  });

  it("never forwards the browser's user id or API key", () => {
    const headers = buildBackendHeaders(spoofed(), { apiKey: "server-key", userId: null, isAdmin: false });
    assert.equal(headers["X-API-Key"], "server-key");
    assert.equal(headers["X-User-Id"], "");
    assert.equal(headers["X-Forwarded-For"], "203.0.113.7");
  });

  it("sets X-User-Role: admin only when the server decided so", () => {
    const headers = buildBackendHeaders(new Headers(), { apiKey: "k", userId: "user_1", isAdmin: true });
    assert.equal(headers["X-User-Role"], "admin");
  });

  it("never marks a signed-out request as admin", () => {
    const headers = buildBackendHeaders(new Headers(), { apiKey: "k", userId: null, isAdmin: true });
    assert.equal(headers["X-User-Role"], undefined);
  });

  it("falls back to x-real-ip, then unknown-ip", () => {
    assert.equal(
      buildBackendHeaders(new Headers({ "x-real-ip": "198.51.100.1" }), { apiKey: "k", userId: null, isAdmin: false })[
        "X-Forwarded-For"
      ],
      "198.51.100.1"
    );
    assert.equal(
      buildBackendHeaders(new Headers(), { apiKey: "k", userId: null, isAdmin: false })["X-Forwarded-For"],
      "unknown-ip"
    );
  });
});
