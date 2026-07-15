import { spawn } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const python = process.env.QUANTLAB_PYTHON || (process.platform === 'win32' ? 'python' : 'python3')
const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const vite = path.join(webRoot, 'node_modules', 'vite', 'bin', 'vite.js')
const children = []

function start(command, args, name) {
  const child = spawn(command, args, { stdio: 'inherit', shell: false, env: process.env })
  child.on('exit', (code) => {
    if (code && code !== 0) {
      console.error(`${name} exited with code ${code}`)
      shutdown(code)
    }
  })
  children.push(child)
}

function shutdown(code = 0) {
  for (const child of children) {
    if (!child.killed) child.kill()
  }
  process.exit(code)
}

process.on('SIGINT', () => shutdown())
process.on('SIGTERM', () => shutdown())

start(python, ['-m', 'uvicorn', 'quantlab.api:app', '--host', '127.0.0.1', '--port', '8000'], 'API')
start(process.execPath, [vite, '--host', '127.0.0.1'], 'web')

console.log('\nQuantLab is starting at http://127.0.0.1:5173\n')
