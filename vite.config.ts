import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import electron from 'vite-plugin-electron'
import renderer from 'vite-plugin-electron-renderer'
import path from 'path'

const rendererHost = process.env.LTX_RENDERER_HOST || '0.0.0.0'
const rendererPort = Number(process.env.LTX_RENDERER_PORT || '5173')
const backendPort = process.env.LTX_PORT || process.env.VITE_LTX_BACKEND_PORT || '18000'
const backendTarget = process.env.VITE_LTX_BACKEND_URL || `http://127.0.0.1:${backendPort}`
const isLightningStudio = process.env.LIGHTNING_CLOUDSPACE_HOST != null || process.env.LIGHTNING_INTERACTIVE === 'true'
const isBrowserOnly = process.env.LTX_BROWSER_ONLY === '1' || isLightningStudio

const browserProxy = isBrowserOnly ? {
  '/health': {
    target: backendTarget,
    changeOrigin: true,
    ws: true,
  },
  '/api': {
    target: backendTarget,
    changeOrigin: true,
    ws: true,
    rewrite: (requestPath: string) => requestPath,
  },
  '/ws': {
    target: backendTarget,
    changeOrigin: true,
    ws: true,
  },
} : undefined

export default defineConfig({
  plugins: isBrowserOnly
    ? [react()]
    : [
        react(),
        electron([
          {
            entry: 'electron/main.ts',
            onstart(options) {
              if (process.env.ELECTRON_DEBUG) {
                // --inspect and --remote-debugging-port must come before '.' (the app path)
                options.startup(['--inspect=9229', '--remote-debugging-port=9222', '.', '--no-sandbox'])
              } else {
                options.startup()
              }
            },
            vite: {
              build: {
                outDir: 'dist-electron',
                sourcemap: true,
                rollupOptions: {
                  external: ['electron']
                }
              }
            }
          },
          {
            entry: 'electron/preload.ts',
            onstart(options) {
              options.reload()
            },
            vite: {
              build: {
                outDir: 'dist-electron',
                sourcemap: true,
                rollupOptions: {
                  output: {
                    format: 'cjs'  // Preload must be CommonJS
                  }
                }
              }
            }
          }
        ]),
        renderer()
      ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './frontend')
    }
  },
  server: {
    host: rendererHost,
    port: rendererPort,
    strictPort: true,
    proxy: browserProxy,
  },
  preview: {
    host: rendererHost,
    port: rendererPort,
    strictPort: true,
  },
  base: './',  // Use relative paths for Electron file:// protocol
  build: {
    outDir: 'dist'
  }
})
