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

async function fetchJson<T>(baseUrl: string, token: string, request: HermesApiRequest): Promise<T> {
  if (!baseUrl) {
    throw new Error('Browser-fallback bridge: no gateway connection resolved yet.')
  }

  const headers = new Headers({ 'Content-Type': 'application/json' })

  if (token) {
    headers.set(SESSION_HEADER, token)
  }

  const res = await fetch(`${baseUrl}${request.path}`, {
    body: request.body === undefined ? undefined : JSON.stringify(request.body),
    credentials: 'include',
    headers,
    method: request.method ?? 'GET'
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
  const res = await fetch(`${baseUrl}/api/auth/ws-ticket`, { credentials: 'include', method: 'POST' })

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
    const res = await fetch(`${baseUrl}/api/status`, {
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
    // Unreachable / CORS-blocked — let the real connect attempt surface it.
  }

  return 'token'
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

  const baseUrl = normalizeBaseUrl(rawUrl)
  const token = resolveStaticToken()
  const authMode = await probeAuthMode(baseUrl, token)

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
