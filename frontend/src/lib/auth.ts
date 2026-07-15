/**
 * Phase 0.5 frontend auth — HttpOnly cookie sessions + one-shot WS tickets.
 *
 * No secrets touch JavaScript. Login POSTs the API key to /api/auth/login,
 * the browser stores the resulting HttpOnly cookie, and every subsequent
 * same-origin request (fetch, <img src>, <script src>) carries it
 * automatically. WebSocket handshakes can't set headers, so before opening
 * a WS we mint a short-lived single-use ticket via authed REST.
 *
 * See docs/adr/032-phase-0-5-session-cookie-auth.md for rationale, and
 * docs/adr/031 for the Phase 0 mechanism this supersedes.
 */

/** Same-origin credentials mode used for every API call. */
const CREDS: RequestCredentials = 'same-origin';

// ── Fetch interceptor: force credentials: 'same-origin' on our origin ────────

export function installAuthInterceptors(): void {
  const nativeFetch = window.fetch.bind(window);
  window.fetch = (input: RequestInfo | URL, init: RequestInit = {}) => {
    let url: string;
    if (typeof input === 'string') {
      url = input;
    } else if (input instanceof URL) {
      url = input.toString();
    } else {
      url = input.url;
    }
    try {
      const u = new URL(url, window.location.origin);
      if (u.origin === window.location.origin) {
        return nativeFetch(input as RequestInfo, { ...init, credentials: CREDS });
      }
    } catch {
      // Fall through — non-URL input, let native fetch handle it.
    }
    return nativeFetch(input as RequestInfo, init);
  };
}

// ── Auth state ────────────────────────────────────────────────────────────────

export async function isAuthenticated(): Promise<boolean> {
  try {
    const res = await fetch('/api/auth/status', { credentials: CREDS });
    if (!res.ok) return false;
    const data = await res.json();
    return !!data.authenticated;
  } catch {
    return false;
  }
}

export async function login(apiKey: string): Promise<boolean> {
  const res = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ api_key: apiKey.trim() }),
    credentials: CREDS,
  });
  return res.ok;
}

export async function logout(): Promise<void> {
  await fetch('/api/auth/logout', { method: 'POST', credentials: CREDS });
}

/**
 * Prompt the user to log in if they aren't already. Loops until authenticated
 * or the user cancels the prompt.
 */
export async function ensureLoggedIn(): Promise<boolean> {
  if (await isAuthenticated()) return true;
  while (true) {
    const key = window.prompt('Enter PED API key (set as PED_API_KEY on the server):', '');
    if (!key) return false;
    if (await login(key)) return true;
    window.alert('Invalid API key — try again.');
  }
}

// ── WebSocket helper — must be awaited because the ticket fetch is async ────

/**
 * Open a WebSocket to a same-origin URL with authentication. Mints a one-shot
 * ticket via /api/auth/ws-ticket and passes it on the handshake URL.
 *
 * The ticket is opaque, single-use, and expires in 30s — safe to appear on
 * the URL (unlike a bearer key would be).
 */
export async function openAuthedWebSocket(url: string | URL): Promise<WebSocket> {
  const target = url instanceof URL ? url : new URL(url, window.location.href);
  const res = await fetch('/api/auth/ws-ticket', {
    method: 'POST',
    credentials: CREDS,
  });
  if (!res.ok) {
    throw new Error(`Failed to mint WS ticket (${res.status})`);
  }
  const { ticket } = (await res.json()) as { ticket: string };
  target.searchParams.set('ticket', ticket);
  return new WebSocket(target.toString());
}
