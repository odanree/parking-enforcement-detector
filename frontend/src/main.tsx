import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { useAppStore } from './store.ts'
import { installAuthInterceptors, ensureLoggedIn } from './lib/auth.ts'

// Phase 0.5 trust boundary — install fetch interceptor (forces same-origin
// credentials so the session cookie rides on every request) BEFORE any React
// component mounts, then gate the render on a successful login.
installAuthInterceptors()

async function boot() {
  const ok = await ensureLoggedIn()
  if (!ok) {
    document.body.innerHTML =
      '<pre style="padding:2em;font-family:system-ui">Login required. Reload the page to try again.</pre>'
    return
  }
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
}

boot()

// Playwright E2E helper — opens the event modal without needing a real event in the log
// eslint-disable-next-line @typescript-eslint/no-explicit-any
;(window as any).__openModal = (ev: any) => useAppStore.getState().openModal(ev)
