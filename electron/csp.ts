import { session } from 'electron'
import { isDev } from './config'

function getExtraConnectSrc(): string[] {
  const envUrls = [
    process.env.LTX_RENDERER_URL,
    process.env.VITE_LTX_BACKEND_URL,
    process.env.LTX_BACKEND_PUBLIC_URL,
  ]
    .filter((value): value is string => Boolean(value))
    .map((value) => value.trim())

  const hosts = new Set<string>([
    "'self'",
    'http://localhost:*',
    'http://127.0.0.1:*',
    'ws://localhost:*',
    'ws://127.0.0.1:*',
  ])

  for (const value of envUrls) {
    hosts.add(value)
    if (value.startsWith('http://')) {
      hosts.add(value.replace('http://', 'ws://'))
    }
    if (value.startsWith('https://')) {
      hosts.add(value.replace('https://', 'wss://'))
    }
  }

  return Array.from(hosts)
}

// Enforce Content Security Policy via response headers (tamper-proof from renderer)
export function setupCSP(): void {
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    const connectSrc = getExtraConnectSrc().join(' ')
    const csp = isDev
      ? [
          "default-src 'self'",
          "script-src 'self' 'unsafe-inline'",
          "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
          "font-src 'self' https://fonts.gstatic.com",
          `connect-src ${connectSrc}`,
          "img-src 'self' data: blob: file:",
          "media-src 'self' blob: file:",
          "object-src 'none'",
          "base-uri 'self'",
          "form-action 'self'",
          "frame-ancestors 'none'",
        ].join('; ')
      : [
          "default-src 'self'",
          "script-src 'self'",
          "style-src 'self' https://fonts.googleapis.com",
          "font-src 'self' https://fonts.gstatic.com",
          `connect-src ${connectSrc}`,
          "img-src 'self' data: blob: file:",
          "media-src 'self' blob: file:",
          "object-src 'none'",
          "base-uri 'self'",
          "form-action 'self'",
          "frame-ancestors 'none'",
        ].join('; ')

    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [csp],
      },
    })
  })
}
