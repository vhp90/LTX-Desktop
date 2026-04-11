let cached: { url: string; token: string } | null = null

export async function getBackendCredentials(): Promise<{ url: string; token: string }> {
  if (!cached) cached = await window.electronAPI.getBackend()
  return cached
}

export function resetBackendCredentials(): void {
  cached = null
}

function joinBackendUrl(baseUrl: string, path: string, ws = false): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`

  try {
    const url = new URL(baseUrl)
    if (ws) {
      url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
    }

    if (url.searchParams.has('port')) {
      const basePath = url.pathname.endsWith('/') ? url.pathname.slice(0, -1) : url.pathname
      url.pathname = `${basePath}${normalizedPath}` || normalizedPath
      return url.toString()
    }

    return new URL(normalizedPath, url).toString()
  } catch {
    const trimmedBase = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl
    return `${trimmedBase}${normalizedPath}`
  }
}

export async function backendFetch(path: string, init?: RequestInit): Promise<Response> {
  const { url, token } = await getBackendCredentials()
  const headers = new Headers(init?.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return fetch(joinBackendUrl(url, path), { ...init, headers })
}

export async function backendWsUrl(path: string): Promise<string> {
  const { url, token } = await getBackendCredentials()
  const ws = joinBackendUrl(url, path, true)
  const sep = ws.includes('?') ? '&' : '?'
  return `${ws}${sep}token=${token}`
}
