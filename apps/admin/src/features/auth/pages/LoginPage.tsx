import { createMutation, useQueryClient } from '@tanstack/solid-query'
import { Link, useNavigate } from '@tanstack/solid-router'
import { Show, createSignal, onMount } from 'solid-js'
import { DEMO_MODE_ENABLED, getErrorMessage, loginAdmin } from '../../../lib/api'

export function LoginPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [username, setUsername] = createSignal('')
  const [password, setPassword] = createSignal('')

  const loginMutation = createMutation(() => ({
    mutationFn: loginAdmin,
    onSuccess: async (admin) => {
      queryClient.setQueryData(['admin', 'me'], admin)
      await navigate({ to: '/dashboard', replace: true })
    },
  }))

  // Pre-warm the chunk the user will need immediately after sign-in so the
  // post-login navigation doesn't pause on a cold module fetch.
  onMount(() => {
    void import('../../dashboard/pages/DashboardPage').catch(() => {})
    if (DEMO_MODE_ENABLED) {
      void import('../../../lib/demo').catch(() => {})
    }
  })

  const handleSubmit = (event: SubmitEvent) => {
    event.preventDefault()
    loginMutation.mutate({ username: username().trim(), password: password() })
  }

  const handleDemoEnter = () => {
    loginMutation.mutate({ username: 'demo-admin', password: 'demo-mode' })
  }

  return (
    <div class="form-shell">
      <div class="form-card">
        <p
          class="eyebrow"
          style={{ color: 'var(--text-3)', 'margin-bottom': '12px' }}
        >
          {DEMO_MODE_ENABLED ? 'Preview access' : 'Secure access'}
        </p>
        <h2
          style={{
            'font-family': 'var(--serif)',
            'font-size': '40px',
            'letter-spacing': '-.025em',
            'line-height': 1,
            'margin-bottom': '12px',
          }}
        >
          {DEMO_MODE_ENABLED ? (
            <>
              Demo <em style={{ 'font-style': 'italic', color: 'var(--terracotta)' }}>workspace.</em>
            </>
          ) : (
            <>
              Admin <em style={{ 'font-style': 'italic', color: 'var(--terracotta)' }}>sign in.</em>
            </>
          )}
        </h2>
        <p style={{ color: 'var(--text-3)', 'font-size': '13.5px', 'line-height': 1.65, 'margin-bottom': '24px' }}>
          {DEMO_MODE_ENABLED
            ? 'The app is currently using seeded mock data so you can review the v6 editorial and analytics shells before real backend roles are connected.'
            : 'Only administrator accounts can open this dashboard. Use the FastAPI admin credentials configured in your backend.'}
        </p>

        <Show
          when={DEMO_MODE_ENABLED}
          fallback={
            <form onSubmit={handleSubmit}>
              <div class="form-field">
                <label for="username">Username</label>
                <input
                  id="username"
                  type="text"
                  autocomplete="username"
                  value={username()}
                  onInput={(e) => setUsername(e.currentTarget.value)}
                  placeholder="admin"
                />
              </div>
              <div class="form-field">
                <label for="password">Password</label>
                <input
                  id="password"
                  type="password"
                  autocomplete="current-password"
                  value={password()}
                  onInput={(e) => setPassword(e.currentTarget.value)}
                  placeholder="enter your admin password"
                />
              </div>
              <Show when={loginMutation.error}>
                <div
                  style={{
                    background: 'var(--terracotta-dim)',
                    border: '1px solid var(--terracotta-line)',
                    color: 'var(--terracotta)',
                    padding: '10px 14px',
                    'border-radius': '8px',
                    'font-size': '12.5px',
                    'margin-bottom': '14px',
                  }}
                >
                  {getErrorMessage(loginMutation.error, 'Unable to sign in with those credentials.')}
                </div>
              </Show>
              <button class="btn-primary" disabled={loginMutation.isPending} type="submit">
                {loginMutation.isPending ? 'Signing in…' : 'Open admin workspace'}
              </button>
            </form>
          }
        >
          <Show when={loginMutation.error}>
            <div
              style={{
                background: 'var(--terracotta-dim)',
                border: '1px solid var(--terracotta-line)',
                color: 'var(--terracotta)',
                padding: '10px 14px',
                'border-radius': '8px',
                'font-size': '12.5px',
                'margin-bottom': '14px',
              }}
            >
              {getErrorMessage(loginMutation.error, 'Unable to open the demo dashboard right now.')}
            </div>
          </Show>
          <button class="btn-primary" disabled={loginMutation.isPending} onClick={handleDemoEnter} type="button">
            {loginMutation.isPending ? 'Opening demo…' : 'Open demo workspace'}
          </button>
        </Show>

        <div style={{ 'margin-top': '24px', 'text-align': 'center' }}>
          <Link
            to="/"
            class="mono"
            style={{ color: 'var(--text-3)', 'font-size': '11px', 'letter-spacing': '.1em', 'text-transform': 'uppercase' }}
          >
            ← back to editorial
          </Link>
        </div>
      </div>
    </div>
  )
}
