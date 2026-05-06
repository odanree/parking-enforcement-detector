import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { useAppStore } from './store.ts'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// Playwright E2E helper — opens the event modal without needing a real event in the log
// eslint-disable-next-line @typescript-eslint/no-explicit-any
;(window as any).__openModal = (ev: any) => useAppStore.getState().openModal(ev)
