import { createMutation, useQueryClient } from '@tanstack/solid-query'
import { Outlet, useNavigate } from '@tanstack/solid-router'
import { Suspense } from 'solid-js'
import { useAdminSession } from '../app/admin-session'
import { Chrome } from '../components/layout/Chrome'
import { Sidebar, editorialSidebar } from '../components/layout/Sidebar'
import { DataState } from '../components/v6'
import { DEMO_MODE_ENABLED, logoutAdmin } from '../lib/api'

export function EditorialShell() {
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
        world="editorial"
        username={admin.username}
        role={admin.role}
        onSignOut={() => logoutMutation.mutate()}
        signingOut={logoutMutation.isPending}
        signoutLabel={DEMO_MODE_ENABLED ? 'Leave demo' : 'Sign out'}
      />
      <main class="app">
        <Sidebar sections={editorialSidebar} />
        <section class="view">
          <div class="shell">
            <div class="editorial-canvas">
              <Suspense
                fallback={
                  <DataState
                    title="Loading editorial page"
                    description="Preparing the next briefing, results, or docs route."
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
