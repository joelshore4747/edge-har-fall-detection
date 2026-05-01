import { Link } from '@tanstack/solid-router'
import { Show } from 'solid-js'
import { useTheme } from '../../app/theme'

export type WorldKey = 'editorial' | 'analytics'

function avatarInitials(name: string | null | undefined): string {
  if (!name) return 'U'
  const parts = name
    .replace(/[^a-zA-Z0-9 ]/g, ' ')
    .split(' ')
    .filter(Boolean)
  if (parts.length === 0) return name.slice(0, 2).toUpperCase()
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase()
  return (parts[0][0] + parts[1][0]).toUpperCase()
}

export function Chrome(props: {
  world: WorldKey
  username: string | null | undefined
  role?: string
  onSignOut?: () => void
  signingOut?: boolean
  signoutLabel?: string
}) {
  const theme = useTheme()
  return (
    <header class="chrome">
      <div class="chrome-inner">
        <Link to="/" class="brand">
          <div class="brand-mark">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round">
              <polygon points="12,3 21,8 21,16 12,21 3,16 3,8" stroke-linejoin="round" />
              <circle cx="12" cy="12" r="2" fill="currentColor" stroke="none" />
            </svg>
          </div>
          <span class="brand-name">UniFall</span>
          <span class="brand-sub">Monitor</span>
        </Link>

        <div class="worlds">
          <Link
            to="/"
            class={`world-btn is-editorial${props.world === 'editorial' ? ' active' : ''}`}
          >
            <span class="dot" />
            Editorial
          </Link>
          <Link
            to="/dashboard"
            class={`world-btn is-analytics${props.world === 'analytics' ? ' active' : ''}`}
          >
            <span class="dot" />
            Analytics
          </Link>
        </div>

        <div class="chrome-spacer" />

        <button class="icon-btn" aria-label="Toggle theme" onClick={theme.toggle}>
          <Show
            when={theme.theme() === 'dark'}
            fallback={
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
              </svg>
            }
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
            </svg>
          </Show>
        </button>

        <div class="user-pill">
          <span class="avatar">{avatarInitials(props.username)}</span>
          <span class="who">
            {props.username ?? 'Guest'}
            <small>{props.role ?? 'Admin'}</small>
          </span>
        </div>

        <Show when={props.onSignOut}>
          <button
            class="signout-btn"
            disabled={props.signingOut}
            onClick={() => props.onSignOut?.()}
          >
            {props.signingOut ? 'Signing out…' : props.signoutLabel ?? 'Sign out'}
          </button>
        </Show>
      </div>
    </header>
  )
}
