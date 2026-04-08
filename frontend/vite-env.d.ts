/// <reference types="vite/client" />

import type { ElectronAPI } from '../shared/electron-api-schema'

interface ImportMetaEnv {
  readonly VITE_LTX_BACKEND_URL?: string
  readonly VITE_LTX_BACKEND_TOKEN?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
}
