import type { ElectronAPI } from '../../shared/electron-api-schema'

function getBrowserBackendUrl(): string {
  // Check for explicit backend URL first (for custom deployments)
  const url = import.meta.env.VITE_LTX_BACKEND_URL?.trim()
  if (url) return url

  const current = new URL(window.location.href)
  const isRemoteHost = current.hostname !== '127.0.0.1' && current.hostname !== 'localhost'
  
  // In Lightning Studio (or any remote host), use the dev proxy (same origin)
  if (isRemoteHost) {
    return `${current.protocol}//${current.host}`
  }

  // For localhost, use the inferred backend port if specified
  const inferredBackendPort = import.meta.env.VITE_LTX_BACKEND_PORT?.trim() || '18000'
  if (current.searchParams.has('port')) {
    return `${current.protocol}//${current.hostname}:${inferredBackendPort}`
  }

  return `http://127.0.0.1:${inferredBackendPort}`
}

function getBrowserBackendToken(): string {
  return import.meta.env.VITE_LTX_BACKEND_TOKEN?.trim() || ''
}

function openExternal(url: string): boolean {
  window.open(url, '_blank', 'noopener,noreferrer')
  return true
}

function unsupportedResult<T>(error: string): Promise<T> {
  return Promise.reject(new Error(error))
}

export function createBrowserElectronApi(): ElectronAPI {
  const backendUrl = getBrowserBackendUrl()
  const backendToken = getBrowserBackendToken()

  const api = {
    getBackend: async () => ({ url: backendUrl, token: backendToken }),
    getModelsPath: async () => '',
    readLocalFile: async () => unsupportedResult('Reading local files is unavailable in browser mode.'),
    checkGpu: async () => ({ available: false }),
    getAppInfo: async () => ({
      version: 'browser-preview',
      isPackaged: false,
      modelsPath: '',
      userDataPath: '',
    }),
    checkFirstRun: async () => ({ needsSetup: false, needsLicense: false }),
    acceptLicense: async () => true,
    completeSetup: async () => true,
    fetchLicenseText: async () => {
      const response = await fetch('https://huggingface.co/Lightricks/LTX-2.3/raw/main/LICENSE')
      if (!response.ok) {
        throw new Error(`Failed to fetch license (HTTP ${response.status})`)
      }
      return await response.text()
    },
    getNoticesText: async () => 'Third-party notices are available in the packaged desktop app.',
    openLtxApiKeyPage: async () => openExternal('https://console.ltx.video/'),
    openFalApiKeyPage: async () => openExternal('https://fal.ai/dashboard/keys'),
    openParentFolderOfFile: async () => undefined,
    showItemInFolder: async () => undefined,
    getLogs: async () => ({ logPath: '', lines: ['Browser preview mode: desktop logs are unavailable.'] }),
    getLogPath: async () => ({ logPath: '', logDir: '' }),
    openLogFolder: async () => false,
    getResourcePath: async () => null,
    getDownloadsPath: async () => '',
    addVisualAssetToProject: async () => unsupportedResult('Importing local assets is unavailable in browser mode.'),
    addGenericAssetToProject: async () => unsupportedResult('Importing local assets is unavailable in browser mode.'),
    makeThumbnailsForProjectAsset: async () => unsupportedResult('Thumbnail generation is unavailable in browser mode.'),
    makeDimensionsForProjectAsset: async () => unsupportedResult('Dimension probing is unavailable in browser mode.'),
    getProjectAssetsPath: async () => '',
    openProjectAssetsPathChangeDialog: async () => ({ success: false as const, error: 'Unavailable in browser mode.' }),
    showSaveDialog: async () => null,
    saveFile: async () => ({ success: false as const, error: 'Saving files is unavailable in browser mode.' }),
    saveBinaryFile: async () => ({ success: false as const, error: 'Saving files is unavailable in browser mode.' }),
    showOpenDirectoryDialog: async () => null,
    searchDirectoryForFiles: async () => ({}),
    checkFilesExist: async () => ({}),
    showOpenFileDialog: async () => null,
    exportNative: async () => ({ success: false as const, error: 'Native export is unavailable in browser mode.' }),
    exportCancel: async () => ({ success: false as const, error: 'Native export is unavailable in browser mode.' }),
    checkPythonReady: async () => ({ ready: true }),
    startPythonSetup: async () => undefined,
    startPythonBackend: async () => undefined,
    getBackendHealthStatus: async () => ({ status: 'alive' as const }),
    extractVideoFrame: async () => unsupportedResult('Frame extraction is unavailable in browser mode.'),
    writeLog: async ({ level, message }) => {
      const prefix = `[browser:${level}]`
      if (level.toLowerCase() === 'error') {
        console.error(prefix, message)
      } else {
        console.log(prefix, message)
      }
    },
    openModelsDirChangeDialog: async () => ({ success: false as const, error: 'Unavailable in browser mode.' }),
    getAnalyticsState: async () => ({ analyticsEnabled: false, installationId: 'browser-preview' }),
    setAnalyticsEnabled: async () => undefined,
    sendAnalyticsEvent: async () => undefined,
    onPythonSetupProgress: () => undefined,
    removePythonSetupProgress: () => undefined,
    onBackendHealthStatus: () => () => undefined,
    getPathForFile: () => '',
    platform: navigator.platform.toLowerCase().includes('mac')
      ? 'darwin'
      : navigator.platform.toLowerCase().includes('win')
        ? 'win32'
        : 'linux',
  } satisfies ElectronAPI

  return api
}
