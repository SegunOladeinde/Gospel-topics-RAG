// BACKEND_URL is the full query endpoint, e.g.
// https://gospel-topics-rag.onrender.com/api/v1/query. The other backend
// endpoints sit next to it, so resolve them relative to it:
//   backendEndpoint(url, "usage")             -> .../api/v1/usage
//   backendEndpoint(url, ["conversations", id]) -> .../api/v1/conversations/<id>
// Path segments are URL-encoded, so an id can't climb out of /api/v1/.
// Pure (no Next.js imports) so tests/backendUrl.test.mjs can run it.
export function backendEndpoint(backendUrl: string, path: string | string[]): string {
  const segments = (Array.isArray(path) ? path : [path]).map(encodeURIComponent);
  return new URL(segments.join("/"), backendUrl).toString();
}
