import { spawn } from 'node:child_process'

const python = process.env.QUANTLAB_PYTHON || (process.platform === 'win32' ? 'python' : 'python3')
const shell = process.platform === 'win32'
const children = []

function start(command, args, name) {
  const child = spawn(command, args, { stdio: 'inherit', shell, env: process.env })
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
start('npm', ['run', 'frontend'], 'web')

console.log('\nQuantLab is starting at http://127.0.0.1:5173\n')
