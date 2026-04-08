import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { createBrowserElectronApi } from './lib/browser-electron-api'
import './index.css'

if (!window.electronAPI) {
  window.electronAPI = createBrowserElectronApi()
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
