import { Plus, SlidersHorizontal, X } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'

import { backendFetch } from '../lib/backend'

export interface LoraStackItem {
  path: string
  strength: number
  sdOpsPreset: 'ltx_comfy'
}

interface LoraStackPanelProps {
  loras: LoraStackItem[]
  onChange: (loras: LoraStackItem[]) => void
  disabled?: boolean
}

interface LocalLoraListResponse {
  files: Array<{ name: string; path: string }>
}

export function LoraStackPanel({ loras, onChange, disabled = false }: LoraStackPanelProps) {
  const [defaultPath, setDefaultPath] = useState<string>()
  const [downloadedLoras, setDownloadedLoras] = useState<Array<{ name: string; path: string }>>([])

  useEffect(() => {
    let cancelled = false

    const loadDefaults = async () => {
      try {
        const modelsPath = await window.electronAPI.getModelsPath()
        if (!cancelled && modelsPath) {
          const normalized = modelsPath.replace(/[\\/]+models$/, '')
          setDefaultPath(`${normalized}/loras/external`)
        }
      } catch {
        // Browser mode and preview mode can safely ignore this.
      }

      try {
        const response = await backendFetch('/api/models/local-loras')
        if (!response.ok) {
          throw new Error('Failed to load local LoRAs')
        }
        const payload = await response.json() as LocalLoraListResponse
        if (!cancelled) {
          setDownloadedLoras(payload.files)
        }
      } catch {
        if (!cancelled) {
          setDownloadedLoras([])
        }
      }
    }

    void loadDefaults()
    return () => {
      cancelled = true
    }
  }, [])

  const handleAdd = useCallback(async () => {
    const selected = await window.electronAPI.showOpenFileDialog({
      title: 'Select LTX 2.3 LoRA files',
      defaultPath,
      filters: [{ name: 'Safetensors', extensions: ['safetensors'] }],
      properties: ['openFile', 'multiSelections'],
    })
    if (!selected?.length) return

    const next = [...loras]
    const existing = new Set(loras.map((item) => item.path))
    for (const path of selected) {
      if (existing.has(path)) continue
      next.push({ path, strength: 1.0, sdOpsPreset: 'ltx_comfy' })
    }
    onChange(next)
  }, [defaultPath, loras, onChange])

  const handleAddDownloaded = useCallback((path: string) => {
    if (loras.some((item) => item.path === path)) return
    onChange([...loras, { path, strength: 1.0, sdOpsPreset: 'ltx_comfy' }])
  }, [loras, onChange])

  const handleRemove = useCallback((index: number) => {
    onChange(loras.filter((_, currentIndex) => currentIndex !== index))
  }, [loras, onChange])

  const handleStrengthChange = useCallback((index: number, strength: number) => {
    const safeStrength = Number.isFinite(strength) ? Math.max(0, Math.min(2, strength)) : 1
    onChange(
      loras.map((item, currentIndex) => (
        currentIndex === index ? { ...item, strength: safeStrength } : item
      )),
    )
  }, [loras, onChange])

  return (
    <div className="mb-3 rounded-2xl border border-zinc-800 bg-zinc-950/95 px-4 py-3 shadow-2xl">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="rounded-lg bg-zinc-900 p-2 text-zinc-300">
            <SlidersHorizontal className="h-4 w-4" />
          </div>
          <div>
            <p className="text-sm font-medium text-white">LTX 2.3 LoRA Stack</p>
            <p className="text-xs text-zinc-500">Local LTX-targeted LoRAs with separate strengths.</p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => void handleAdd()}
          disabled={disabled}
          className="inline-flex items-center gap-1.5 rounded-lg bg-white/10 px-3 py-2 text-xs font-medium text-white transition hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Plus className="h-3.5 w-3.5" />
          Add LoRA
        </button>
      </div>

      {loras.length > 0 ? (
        <div className="mt-3 space-y-2">
          {loras.map((lora, index) => (
            <div key={`${lora.path}-${index}`} className="rounded-xl border border-zinc-800 bg-zinc-900/80 px-3 py-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm text-zinc-100">{lora.path.split('/').pop() || lora.path}</p>
                  <p className="mt-1 truncate text-xs text-zinc-500">{lora.path}</p>
                </div>
                <button
                  type="button"
                  onClick={() => handleRemove(index)}
                  disabled={disabled}
                  className="rounded-md p-1.5 text-zinc-500 transition hover:bg-zinc-800 hover:text-zinc-200 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              <div className="mt-3 flex items-center gap-3">
                <label className="text-xs font-medium uppercase tracking-wide text-zinc-500">Strength</label>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.05}
                  value={lora.strength}
                  disabled={disabled}
                  onChange={(e) => handleStrengthChange(index, Number(e.target.value))}
                  className="flex-1 accent-orange-400"
                />
                <input
                  type="number"
                  min={0}
                  max={2}
                  step={0.05}
                  value={lora.strength}
                  disabled={disabled}
                  onChange={(e) => handleStrengthChange(index, Number(e.target.value))}
                  className="w-20 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 outline-none"
                />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-3 text-xs text-zinc-500">
          No LoRAs loaded. Add `.safetensors` files trained for LTX / LTX 2.3.
        </p>
      )}

      {downloadedLoras.length > 0 ? (
        <div className="mt-3 rounded-xl border border-zinc-800 bg-zinc-900/50 px-3 py-3">
          <p className="text-xs font-medium uppercase tracking-wide text-zinc-500">Downloaded LoRAs</p>
          <div className="mt-2 space-y-2">
            {downloadedLoras.map((item) => (
              <div key={item.path} className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate text-sm text-zinc-100">{item.name}</p>
                  <p className="truncate text-xs text-zinc-500">{item.path}</p>
                </div>
                <button
                  type="button"
                  onClick={() => handleAddDownloaded(item.path)}
                  disabled={disabled || loras.some((lora) => lora.path === item.path)}
                  className="rounded-lg bg-white/10 px-2.5 py-1.5 text-xs font-medium text-white transition hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Add
                </button>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  )
}
