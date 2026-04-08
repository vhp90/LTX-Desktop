#!/usr/bin/env node

import { spawn } from 'child_process'

const isLightningStudio = !!process.env.LIGHTNING_CLOUDSPACE_HOST || process.env.LIGHTNING_INTERACTIVE === 'true'
const scriptName = isLightningStudio ? 'dev:studio' : 'dev:renderer'

const child = spawn('corepack', ['pnpm', 'run', scriptName], {
  stdio: 'inherit',
  shell: false,
  env: process.env,
})

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal)
    return
  }
  process.exit(code ?? 0)
})
