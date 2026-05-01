import { createMutation, useQueryClient } from '@tanstack/solid-query'
import { Outlet, useNavigate, useRouterState } from '@tanstack/solid-router'
import { Suspense } from 'solid-js'
import { useAdminSession } from '../app/admin-session'
import { Chrome } from '../components/layout/Chrome'
import { Sidebar, analyticsSidebar } from '../components/layout/Sidebar'
import { DataState } from '../components/v6'
import { DEMO_MODE_ENABLED, logoutAdmin } from '../lib/api'

const ROUTE_LABEL: Record<string, string> = {
  '/dashboard': 'Operations dashboard',
  '/sessions': 'Sessions list',
  '/library': 'Asset library',
  '/explore': 'Cohort explorer',
}

export function AnalyticsShell() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const admin = useAdminSession()
  const routerState = useRouterState()

  const logoutMutation = createMutation(() => ({
    mutationFn: logoutAdmin,
    onSuccess: async () => {
      queryClient.clear()
      await navigate({ to: '/login', replace: true })
    },
  }))

  const currentLabel = () => {
    const path = routerState().location.pathname
    if (path.startsWith('/sessions/')) return 'Session detail'
    return ROUTE_LABEL[path] ?? 'Analytics'
  }

  return (
    <>
      <Chrome
        world="analytics"
        username={admin.username}
        role={admin.role}
        onSignOut={() => logoutMutation.mutate()}
        signingOut={logoutMutation.isPending}
        signoutLabel={DEMO_MODE_ENABLED ? 'Leave demo' : 'Sign out'}
      />
      <main class="app">
        <Sidebar sections={analyticsSidebar} />
        <section class="view">
          <div class="shell">
            <div class="analytics-topbar">
              <div class="crumbs">
                <span>Analytics</span>
                <span style={{ color: 'var(--text-4)' }}>/</span>
                <strong>{currentLabel()}</strong>
              </div>
              <div style={{ 'margin-left': 'auto', display: 'flex', gap: '8px', 'align-items': 'center' }}>
                <span class="tag live">● live · 5s polling</span>
                <span class="mono">api.unifallmonitor.com</span>
              </div>
            </div>
            <div class="analytics-canvas">
              <Suspense
                fallback={
                  <DataState
                    title="Loading analytics view"
                    description="Fetching the next admin workspace chunk."
                  />
                }
              >
                <Outlet />
              </Suspense>
            </div>
          </div>
        </section>
      </main>
    </>
  )
}
