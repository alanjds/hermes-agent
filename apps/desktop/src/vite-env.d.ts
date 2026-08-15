/// <reference types="vite/client" />

// Browser-fallback mode (see src/lib/browser-bridge.ts): build-time
// alternative to the `?gateway=`/`?token=` query params, for a static build
// pinned at a single hermes serve instance.
interface ImportMetaEnv {
  readonly VITE_GATEWAY_TOKEN?: string
  readonly VITE_GATEWAY_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
