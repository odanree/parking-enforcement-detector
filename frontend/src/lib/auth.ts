/**
 * Phase 0 frontend auth — API key stored in localStorage, injected into
 * every fetch() and WebSocket URL via global interceptors.
 *
 * Why interceptors instead of an API client: there are 18 existing fetch
 * call sites across hooks/components. Monkey-patching window.fetch and
 * window.WebSocket at boot lets all of them work unchanged.
 *
 * See docs/adr/030-aws-migration-strangler-fig.md (Phase 0) and
 * docs/adr/031-phase-0-api-key-auth.md for backend rationale.
 */

const STORAGE_KEY = 'ped.apiKey';

/** Paths that are served without auth (must match backend _PUBLIC_*). */
const PUBLIC_PATH_PREFIXES = ['/assets/'];
const PUBLIC_PATH_EXACT = new Set(['/', '/favicon.svg', '/health']);

/** Paths that need the token as a query param instead of Authorization header. */
const QUERY_PARAM_PATH_PREFIXES = ['/snapshots/', '/dataset/'];

function isPublicPath(pathname: string): boolean {
  if (PUBLIC_PATH_EXACT.has(pathname)) return true;
  return PUBLIC_PATH_PREFIXES.some((p) => pathname.startsWith(p));
}

function needsQueryParam(pathname: string): boolean {
  return QUERY_PARAM_PATH_PREFIXES.some((p) => pathname.startsWith(p));
}

export function getApiKey(): string | null {
  return localStorage.getItem(STORAGE_KEY);
}

/**
 * Append the API key as ?token= to a same-origin URL that a browser
 * resource loader will fetch (e.g. `<img src>`, `<a href>` downloads).
 * fetch() and WebSocket calls do NOT need this — those are handled by
 * the global interceptors installed at boot.
 *
 * Returns the path unchanged when no key is stored (image will 401,
 * which is what we want during the login-prompt phase).
 */
export function authedUrl(path: string): string {
  const key = getApiKey();
  if (!key) return path;
  const sep = path.includes('?') ? '&' : '?';
  return `${path}${sep}token=${encodeURIComponent(key)}`;
}

export function setApiKey(key: string): void {
  localStorage.setItem(STORAGE_KEY, key.trim());
}

export function clearApiKey(): void {
  localStorage.removeItem(STORAGE_KEY);
}

/** Prompt the user for a key if none is stored. Returns the key or null on cancel. */
export function promptForApiKey(): string | null {
  const existing = getApiKey();
  if (existing) return existing;
  const entered = window.prompt(
    'Enter PED API key (set as PED_API_KEY on the server):',
    '',
  );
  if (!entered) return null;
  setApiKey(entered);
  return entered;
}

function toUrlObject(input: RequestInfo | URL): URL {
  if (input instanceof URL) return input;
  const raw = typeof input === 'string' ? input : input.url;
  // Relative URLs need a base to construct a URL object.
  return new URL(raw, window.location.origin);
}

/**
 * Install fetch + WebSocket interceptors. Call once at boot BEFORE any
 * React component mounts (so no hook sneaks a call in first).
 */
export function installAuthInterceptors(): void {
  const nativeFetch = window.fetch.bind(window);
  const NativeWebSocket = window.WebSocket;

  window.fetch = async (input: RequestInfo | URL, init: RequestInit = {}) => {
    let url: URL;
    try {
      url = toUrlObject(input);
    } catch {
      return nativeFetch(input as RequestInfo, init);
    }

    // Only touch requests to this app's own origin — no leaking the key
    // to third-party CDNs, telemetry, etc.
    if (url.origin !== window.location.origin) {
      return nativeFetch(input as RequestInfo, init);
    }

    if (isPublicPath(url.pathname)) {
      return nativeFetch(input as RequestInfo, init);
    }

    const key = getApiKey();
    if (!key) {
      // 401 without ever hitting the network — lets the caller show the login UI.
      return new Response('no api key set', { status: 401 });
    }

    if (needsQueryParam(url.pathname)) {
      // <img> tags etc. — token has to ride on the URL because you can't
      // set headers on a fetch that a browser resource loader kicked off.
      url.searchParams.set('token', key);
      const nextInit: RequestInit = { ...init };
      return nativeFetch(url.toString(), nextInit);
    }

    const headers = new Headers(init.headers ?? {});
    headers.set('Authorization', `Bearer ${key}`);
    return nativeFetch(input as RequestInfo, { ...init, headers });
  };

  // WebSocket: rewrite the URL to include ?token= before the handshake fires.
  // Wrap the native constructor rather than subclassing — subclassing WebSocket
  // is inconsistent across browsers.
  const patchedWs = function (
    url: string | URL,
    protocols?: string | string[],
  ) {
    const target = url instanceof URL ? url : new URL(url, window.location.href);
    const key = getApiKey();
    // Compare host (host:port) not origin — origin includes the scheme, and
    // ws://localhost:8000 has a different origin from http://localhost:8000
    // even though it's the same server.
    if (key && target.host === window.location.host) {
      target.searchParams.set('token', key);
    }
    return new NativeWebSocket(target.toString(), protocols);
  };

  // Mirror the constructor-level readonly constants (CONNECTING/OPEN/CLOSING/CLOSED)
  // via defineProperties so we don't trip TS's readonly check on the WebSocket type.
  Object.defineProperties(patchedWs, {
    prototype:  { value: NativeWebSocket.prototype },
    CONNECTING: { value: NativeWebSocket.CONNECTING, enumerable: true },
    OPEN:       { value: NativeWebSocket.OPEN,       enumerable: true },
    CLOSING:    { value: NativeWebSocket.CLOSING,    enumerable: true },
    CLOSED:     { value: NativeWebSocket.CLOSED,     enumerable: true },
  });

  window.WebSocket = patchedWs as unknown as typeof WebSocket;
}
