import { Outlet } from '@tanstack/solid-router'
import { Suspense } from 'solid-js'
import { DataState } from '../components/v6'

export function AuthShell() {
  return (
    <div style={{ 'min-height': '100vh', background: 'var(--bg)' }}>
      <header class="chrome">
        <div class="chrome-inner">
          <div class="brand">
            <div class="brand-mark">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round">
                <polygon points="12,3 21,8 21,16 12,21 3,16 3,8" stroke-linejoin="round" />
                <circle cx="12" cy="12" r="2" fill="currentColor" stroke="none" />
              </svg>
            </div>
            <span class="brand-name">UniFall</span>
            <span class="brand-sub">Monitor</span>
          </div>
          <div class="chrome-spacer" />
          <span class="mono" style={{ color: 'var(--text-3)' }}>research admin · sign-in required</span>
        </div>
      </header>
      <Suspense
        fallback={
          <DataState title="Loading sign-in" description="Preparing the admin login route." />
        }
      >
        <Outlet />
      </Suspense>
    </div>
  )
}
