/**
 * Browser-fallback bridge (apps/desktop, non-Electron mode).
 *
 * apps/desktop is normally launched only inside Electron, which injects
 * `window.hermesDesktop` via electron/preload.cjs. This module lets the SAME
 * renderer bundle run in a plain browser tab against an already-running
 * `hermes serve` instance, by installing a minimal stand-in for that global
 * before anything else in the app reads it (see the first import in
 * main.tsx).
 *
 * Scope, deliberately narrow: this only implements the handful of
 * `window.hermesDesktop` methods that `app/gateway/hooks/use-gateway-boot.ts`
 * calls *unconditionally* during boot (`getConnection`, `api`,
 * `getBootProgress`, `onBootProgress`, `onBackendExit`). Every other call
 * site elsewhere in the app is already optional-chained
 * (`window.hermesDesktop?.foo?.()`) and gracefully no-ops when a method is
 * missing, so leaving everything else unimplemented is intentional, not an
 * oversight — see the file-by-file audit in the task that introduced this
 * module. Anything backed by real native OS access (file browser, git,
 * terminal PTY) has no browser equivalent; desktop-controller.tsx hides
 * those panes when `isBrowserFallbackActive()` is true rather than letting
 * them fail at the call site.
 *
 * The connection this bridge builds always reports `mode: 'remote'`, so it
 * rides the exact same "remote gateway" degradation paths the app already
 * has for a genuine Electron → remote-hermes-serve connection (see
 * `src/lib/desktop-fs.ts`'s `isDesktopFsRemoteMode()` branches) instead of
 * needing new ones.
 *
 * Activation: a `?gateway=<url>` query param, or a build-time
 * `VITE_GATEWAY_URL` env var (query param wins, so one static build can
 * point at different backends without a rebuild). An optional `?token=` /
 * `VITE_GATEWAY_TOKEN` supplies a static session token. There is no config
 * storage — nowhere to persist it without native disk access — so this is
 * re-resolved on every page load.
 *
 * Auth/WS-URL construction mirrors electron/connection-config.cjs
 * (`buildGatewayWsUrl` / `buildGatewayWsUrlWithTicket`,
 * `/api/ws?token=` vs `/api/ws?ticket=`) and the browser dashboard's own
 * pattern in web/src/lib/api.ts (`X-Hermes-Session-Token` header,
 * `credentials: 'include'`, ticket-minting via `POST /api/auth/ws-ticket`
 * for gated/OAuth deployments).
 */

import type { DesktopBootProgress, HermesApiRequest, HermesConnection } from '@/global'

const SESSION_HEADER = 'X-Hermes-Session-Token'

function readParam(name: string): string {
  if (typeof window === 'undefined') {
    return ''
  }

  try {
    return new URLSearchParams(window.location.search).get(name)?.trim() || ''
  } catch {
    return ''
  }
}

function resolveGatewayUrl(): string {
  return readParam('gateway') || import.meta.env.VITE_GATEWAY_URL?.trim() || ''
}

function resolveStaticToken(): string {
  return readParam('token') || import.meta.env.VITE_GATEWAY_TOKEN?.trim() || ''
}

function normalizeBaseUrl(rawUrl: string): string {
  const parsed = new URL(rawUrl)

  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`Gateway URL must be http:// or https://, got ${parsed.protocol}`)
  }

  parsed.hash = ''
  parsed.search = ''
  parsed.pathname = parsed.pathname.replace(/\/+$/, '')

  return parsed.toString().replace(/\/+$/, '')
}

function buildWsUrl(baseUrl: string, param: 'ticket' | 'token', value: string): string {
  const parsed = new URL(baseUrl)
  const scheme = parsed.protocol === 'https:' ? 'wss' : 'ws'
  const prefix = parsed.pathname.replace(/\/+$/, '')

  return `${scheme}://${parsed.host}${prefix}/api/ws?${param}=${encodeURIComponent(value)}`
}

// A `hermes serve` backend shares its asyncio event loop with the agent
// itself — a heavy turn (delegation running several subagents, a long tool
// call) can starve it for several seconds at a time, but it's still alive
// and will answer once it catches its breath. A bare `fetch()` has no
// timeout at all, so without this every one of the bridge's calls below
// would hang exactly as long as the backend is stalled — including the one
// inside `resolveConnection()` that both the initial boot *and* every
// reconnect-after-drop attempt awaits. An unbounded hang there doesn't just
// delay boot, it wedges the reconnect loop for good (its `reconnecting`
// guard never clears because the awaited promise never settles), so a
// transient stall permanently reads as "Gateway offline" with no recovery.
// 20s is generous relative to the backend's own stall tolerance (it treats
// up to 10s as "still alive" per WSTransport's write timeout) while still
// guaranteeing every call here eventually settles one way or another, so
// the existing catch/retry-with-backoff paths actually get a turn to run.
const BRIDGE_FETCH_TIMEOUT_MS = 20_000

async function fetchWithTimeout(url: string, init: RequestInit = {}): Promise<Response> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), BRIDGE_FETCH_TIMEOUT_MS)

  try {
    return await fetch(url, { ...init, signal: controller.signal })
  } finally {
    clearTimeout(timer)
  }
}

async function fetchJson<T>(baseUrl: string, token: string, request: HermesApiRequest): Promise<T> {
  if (!baseUrl) {
    throw new Error('Browser-fallback bridge: no gateway connection resolved yet.')
  }

  const headers = new Headers({ 'Content-Type': 'application/json' })

  if (token) {
    headers.set(SESSION_HEADER, token)
  }

  const res = await fetchWithTimeout(`${baseUrl}${request.path}`, {
    body: request.body === undefined ? undefined : JSON.stringify(request.body),
    credentials: 'include',
    headers,
    method: request.method ?? 'GET'
  }).catch(err => {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error(`${request.path}: timed out waiting on gateway (backend busy?)`)
    }

    throw err
  })

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)

    throw new Error(`${res.status}: ${text}`)
  }

  if (res.status === 204) {
    return undefined as T
  }

  return (await res.json()) as T
}

async function mintWsTicket(baseUrl: string): Promise<string> {
  const res = await fetchWithTimeout(`${baseUrl}/api/auth/ws-ticket`, { credentials: 'include', method: 'POST' }).catch(
    err => {
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw new Error('/api/auth/ws-ticket: timed out waiting on gateway (backend busy?)')
      }

      throw err
    }
  )

  if (!res.ok) {
    throw new Error(`/api/auth/ws-ticket: HTTP ${res.status}`)
  }

  const body = (await res.json()) as { ticket: string }

  return body.ticket
}

// Classify the gateway's auth model from its public /api/status the same way
// electron/connection-config.cjs's comment describes: `auth_required: true`
// means gated/OAuth (cookie + single-use WS ticket); everything else is the
// legacy static-token model (header + `?token=` on the WS upgrade). A 401 on
// an unauthenticated probe is read the same way a gated deployment would
// answer it. Any other failure (unreachable, CORS) falls through to 'token'
// so the real error surfaces from the connect attempt instead of here.
async function probeAuthMode(baseUrl: string, token: string): Promise<'oauth' | 'token'> {
  try {
    const res = await fetchWithTimeout(`${baseUrl}/api/status`, {
      credentials: 'include',
      headers: token ? { [SESSION_HEADER]: token } : undefined
    })

    if (res.status === 401) {
      return 'oauth'
    }

    if (res.ok) {
      const body = (await res.json().catch(() => null)) as { auth_required?: boolean } | null

      if (body?.auth_required) {
        return 'oauth'
      }
    }
  } catch {
    // Unreachable / CORS-blocked / timed out (backend momentarily stalled) —
    // let the real connect attempt surface it rather than wedging boot/
    // reconnect here indefinitely.
  }

  return 'token'
}

interface DedicatedBackend {
  backendId: string
  baseUrl: string
  token: string
}

const DEDICATED_BACKEND_STORAGE_PREFIX = 'hermes-dedicated-backend:'

function dedicatedBackendStorageKey(seedBaseUrl: string): string {
  return `${DEDICATED_BACKEND_STORAGE_PREFIX}${seedBaseUrl}`
}

function loadStoredDedicatedBackend(seedBaseUrl: string): DedicatedBackend | null {
  try {
    const raw = window.localStorage.getItem(dedicatedBackendStorageKey(seedBaseUrl))

    if (!raw) {
      return null
    }

    const parsed = JSON.parse(raw) as Partial<DedicatedBackend>

    if (typeof parsed.backendId === 'string' && typeof parsed.baseUrl === 'string' && typeof parsed.token === 'string') {
      return parsed as DedicatedBackend
    }
  } catch {
    // Corrupt JSON / storage unavailable (private browsing can throw on
    // read too, not just write) — treat identically to "nothing stored".
  }

  return null
}

function storeDedicatedBackend(seedBaseUrl: string, backend: DedicatedBackend): void {
  try {
    window.localStorage.setItem(dedicatedBackendStorageKey(seedBaseUrl), JSON.stringify(backend))
  } catch {
    // Storage full/unavailable — not fatal, just means every reload
    // re-spawns instead of reusing. The dedicated backend itself still
    // works fine for the rest of this page's lifetime either way.
  }
}

async function probeDedicatedBackendAlive(backend: DedicatedBackend): Promise<boolean> {
  try {
    const res = await fetchWithTimeout(`${backend.baseUrl}/api/status`, {
      headers: { [SESSION_HEADER]: backend.token }
    })

    return res.ok
  } catch {
    return false
  }
}

async function spawnOrReuseDedicatedBackend(
  seedBaseUrl: string,
  seedToken: string,
  staleId?: string
): Promise<DedicatedBackend> {
  const res = await fetchWithTimeout(`${seedBaseUrl}/api/desktop/spawn-backend`, {
    body: JSON.stringify(staleId ? { backend_id: staleId } : {}),
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...(seedToken ? { [SESSION_HEADER]: seedToken } : {})
    },
    method: 'POST'
  })

  if (!res.ok) {
    throw new Error(`/api/desktop/spawn-backend: HTTP ${res.status}`)
  }

  const body = (await res.json()) as { backend_id: string; base_url: string; token: string }

  return { backendId: body.backend_id, baseUrl: body.base_url, token: body.token }
}

// Resolves the backend apps/desktop should actually talk to: a dedicated,
// single-tenant `hermes dashboard` child spawned (or reused) by the "seed"
// gateway from ?gateway=, rather than the seed itself.
//
// PoC rationale: the seed may be a shared, multi-tenant dashboard the
// operator also uses for other sessions/cron/messaging platforms — its
// asyncio event loop can stall for seconds at a time under concurrent
// agent work happening elsewhere entirely (see tools/cpu_offload.py and
// WSTransport's "loop stalled" warning in tui_gateway/ws.py), which is
// exactly the failure mode that made this bridge need generous fetch
// timeouts and patient reconnect logic in the first place. apps/desktop's
// Electron mode never hits this because it spawns a dedicated backend per
// window (see electron/main.cjs); this gives a browser tab the same
// escape hatch via hermes_cli/web_server.py's /api/desktop/spawn-backend
// — see that endpoint's docstring for the server-side half of this. No
// wire-protocol change: the dedicated child speaks the exact same
// tui_gateway WS JSON-RPC dialect, just isolated onto its own process.
//
// Always falls back to `null` (meaning: connect straight to the seed,
// exactly like before this existed) on any failure — an older seed
// without this endpoint, or one too stalled to even answer a cheap spawn
// request, must never block boot.
//
// PoC opt-out: ?dedicated=0 skips this outright, for comparing behavior
// against the seed directly while this is being validated.
async function resolveDedicatedBackend(seedBaseUrl: string, seedToken: string): Promise<DedicatedBackend | null> {
  if (readParam('dedicated') === '0') {
    return null
  }

  const stored = loadStoredDedicatedBackend(seedBaseUrl)

  if (stored && (await probeDedicatedBackendAlive(stored))) {
    return stored
  }

  try {
    const backend = await spawnOrReuseDedicatedBackend(seedBaseUrl, seedToken, stored?.backendId)

    storeDedicatedBackend(seedBaseUrl, backend)

    return backend
  } catch {
    return null
  }
}

let cachedConnection: HermesConnection | null = null

async function resolveConnection(): Promise<HermesConnection> {
  const rawUrl = resolveGatewayUrl()

  if (!rawUrl) {
    throw new Error(
      'No gateway URL configured for browser mode. Add ?gateway=http://host:port to the page URL (and, if the ' +
        'gateway requires one, &token=...).'
    )
  }

  const seedBaseUrl = normalizeBaseUrl(rawUrl)
  const seedToken = resolveStaticToken()

  // A dedicated backend is always loopback-bound + static-token (it's
  // spawned locally, purely for this one browser session — see
  // spawn-backend server-side), so its auth mode is already known and the
  // probeAuthMode() round-trip below is skipped entirely for that path.
  const dedicated = await resolveDedicatedBackend(seedBaseUrl, seedToken)
  const baseUrl = dedicated?.baseUrl ?? seedBaseUrl
  const token = dedicated?.token ?? seedToken
  const authMode = dedicated ? 'token' : await probeAuthMode(baseUrl, token)

  const wsUrl =
    authMode === 'oauth'
      ? buildWsUrl(baseUrl, 'ticket', await mintWsTicket(baseUrl))
      : buildWsUrl(baseUrl, 'token', token)

  const connection: HermesConnection = {
    authMode,
    baseUrl,
    isFullscreen: false,
    logs: [],
    mode: 'remote',
    nativeOverlayWidth: 0,
    source: 'env',
    token,
    windowButtonPosition: null,
    wsUrl
  }

  cachedConnection = connection

  return connection
}

let installed = false

/** True once this module has installed the browser-fallback bridge — i.e. the
 *  app is running outside Electron with a resolved gateway URL. UI that only
 *  makes sense with native OS access (file browser, git review, terminal)
 *  checks this to hide/disable itself instead of failing at the call site. */
export function isBrowserFallbackActive(): boolean {
  return installed
}

if (typeof window !== 'undefined' && !window.hermesDesktop && resolveGatewayUrl()) {
  installed = true

  const bridge: Partial<Window['hermesDesktop']> = {
    api: <T>(request: HermesApiRequest) =>
      fetchJson<T>(cachedConnection?.baseUrl ?? '', cachedConnection?.token ?? '', request),
    getBootProgress: () =>
      Promise.resolve<DesktopBootProgress>({
        error: null,
        fakeMode: false,
        message: '',
        phase: 'renderer.ready',
        progress: 100,
        running: false,
        timestamp: Date.now()
      }),
    getConnection: () => resolveConnection(),
    getGatewayWsUrl: async () => {
      const conn = cachedConnection ?? (await resolveConnection())

      return conn.authMode === 'oauth'
        ? buildWsUrl(conn.baseUrl, 'ticket', await mintWsTicket(conn.baseUrl))
        : conn.wsUrl
    },
    onBackendExit: () => () => {},
    onBootProgress: () => () => {}
  }

  window.hermesDesktop = bridge as Window['hermesDesktop']
}
