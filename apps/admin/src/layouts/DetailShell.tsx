import { createMutation, useQueryClient } from '@tanstack/solid-query'
import { Link, Outlet, useNavigate } from '@tanstack/solid-router'
import { Suspense } from 'solid-js'
import { useAdminSession } from '../app/admin-session'
import { Chrome } from '../components/layout/Chrome'
import { Sidebar, analyticsSidebar } from '../components/layout/Sidebar'
import { DataState } from '../components/v6'
import { DEMO_MODE_ENABLED, logoutAdmin } from '../lib/api'

export function DetailShell() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const admin = useAdminSession()

  const logoutMutation = createMutation(() => ({
    mutationFn: logoutAdmin,
    onSuccess: async () => {
      queryClient.clear()
      await navigate({ to: '/login', replace: true })
    },
  }))

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
                <Link to="/sessions">Sessions</Link>
                <span style={{ color: 'var(--text-4)' }}>/</span>
                <strong>Session detail</strong>
              </div>
              <div style={{ 'margin-left': 'auto' }}>
                <Link class="btn-ghost" to="/sessions">
                  ← Back to list
                </Link>
              </div>
            </div>
            <div class="analytics-canvas">
              <Suspense
                fallback={
                  <DataState
                    title="Loading evidence view"
                    description="Fetching the session detail and preparing the evidence workspace."
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
