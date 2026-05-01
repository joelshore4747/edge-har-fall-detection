import { useQueryClient, type QueryClient } from '@tanstack/solid-query'
import {
  Outlet,
  createRootRouteWithContext,
  createRoute,
  createRouter,
  redirect,
} from '@tanstack/solid-router'
import { lazy } from 'solid-js'
import { AdminSessionProvider } from './app/admin-session'
import { FullPageLoader } from './components/v6'
import {
  DEMO_MODE_ENABLED,
  type AdminAuthSession,
  getAdminMe,
  getDemoAdminSessionSnapshot,
} from './lib/api'
// AuthShell is the fallback shell rendered while other chunks load, so it
// stays in the main bundle. The other four layouts each ship in their own
// chunk so visiting /login does not pull editorial/analytics/detail layouts.
// The plain-function wrappers below mirror the pattern used for the page
// views — TanStack Solid Router rejects the raw Solid lazy() return type
// because its `preload` resolves to the module rather than `Promise<void>`.
import { AuthShell } from './layouts/AuthShell'

export interface RouterContext {
  queryClient: QueryClient
}

const ADMIN_ME_QUERY_KEY = ['admin', 'me'] as const

// Warm the dashboard chunk as soon as the protected gate authenticates.
// Vite resolves the dynamic import to the hashed chunk and the browser
// preloads it under standard <link rel="modulepreload"> rules, so the cold
// navigation into /dashboard skips the chunk-fetch pause.
let dashboardChunkPromise: Promise<unknown> | null = null
function preloadDashboardChunk() {
  if (!dashboardChunkPromise) {
    dashboardChunkPromise = import('./features/dashboard/pages/DashboardPage').catch(() => {
      dashboardChunkPromise = null
    })
  }
}

async function ensureAdminMe({ context }: { context: RouterContext }) {
  if (DEMO_MODE_ENABLED) {
    const admin = getDemoAdminSessionSnapshot()
    if (!admin) {
      throw redirect({ to: '/login', replace: true })
    }
    context.queryClient.setQueryData(ADMIN_ME_QUERY_KEY, admin)
    return
  }

  try {
    await context.queryClient.ensureQueryData({
      queryKey: ADMIN_ME_QUERY_KEY,
      queryFn: getAdminMe,
      staleTime: 30_000,
    })
  } catch {
    throw redirect({ to: '/login', replace: true })
  }
}

async function ensureAdminMeForProtected(args: { context: RouterContext }) {
  await ensureAdminMe(args)
  preloadDashboardChunk()
}

const AnalyticsShellView = lazy(async () => {
  const module = await import('./layouts/AnalyticsShell')
  return { default: module.AnalyticsShell }
})

const DetailShellView = lazy(async () => {
  const module = await import('./layouts/DetailShell')
  return { default: module.DetailShell }
})

const EditorialShellView = lazy(async () => {
  const module = await import('./layouts/EditorialShell')
  return { default: module.EditorialShell }
})

function AnalyticsShell() {
  return <AnalyticsShellView />
}

function DetailShell() {
  return <DetailShellView />
}

function EditorialShell() {
  return <EditorialShellView />
}

const LoginPageView = lazy(async () => {
  const module = await import('./features/auth/pages/LoginPage')
  return { default: module.LoginPage }
})

const BriefingPageView = lazy(async () => {
  const module = await import('./features/briefing/pages/BriefingPage')
  return { default: module.BriefingPage }
})

const DashboardPageView = lazy(async () => {
  const module = await import('./features/dashboard/pages/DashboardPage')
  return { default: module.DashboardPage }
})

const DocsPageView = lazy(async () => {
  const module = await import('./features/docs/pages/DocsPage')
  return { default: module.DocsPage }
})

const ArchitecturePageView = lazy(async () => {
  const module = await import('./features/architecture/pages/ArchitecturePage')
  return { default: module.ArchitecturePage }
})

const ExplorePageView = lazy(async () => {
  const module = await import('./features/explore/pages/ExplorePage')
  return { default: module.ExplorePage }
})

const LibraryPageView = lazy(async () => {
  const module = await import('./features/library/pages/LibraryPage')
  return { default: module.LibraryPage }
})

const ResultsPageView = lazy(async () => {
  const module = await import('./features/results/pages/ResultsPage')
  return { default: module.ResultsPage }
})

const SessionDetailPageView = lazy(async () => {
  const module = await import('./features/sessions/pages/SessionDetailPage')
  return { default: module.SessionDetailPage }
})

const SessionsPageView = lazy(async () => {
  const module = await import('./features/sessions/pages/SessionsPage')
  return { default: module.SessionsPage }
})

function LoginPage() {
  return <LoginPageView />
}

function BriefingPage() {
  return <BriefingPageView />
}

function DashboardPage() {
  return <DashboardPageView />
}

function DocsPage() {
  return <DocsPageView />
}

function ArchitecturePage() {
  return <ArchitecturePageView />
}

function ExplorePage() {
  return <ExplorePageView />
}

function LibraryPage() {
  return <LibraryPageView />
}

function ResultsPage() {
  return <ResultsPageView />
}

function SessionsPage() {
  return <SessionsPageView />
}

function RootLayout() {
  return <Outlet />
}

function AdminGate() {
  const queryClient = useQueryClient()
  const admin = queryClient.getQueryData<AdminAuthSession>(ADMIN_ME_QUERY_KEY)
  if (!admin) {
    // beforeLoad guarantees a value is present before this component mounts.
    // The fallback is only reachable if a downstream caller invalidates the
    // cache without re-running the gate; render a neutral splash instead of
    // throwing so the app can recover on the next navigation.
    return (
      <FullPageLoader
        title="Restoring admin session"
        subtitle="Reloading the workspace credentials."
      />
    )
  }
  return (
    <AdminSessionProvider admin={admin}>
      <Outlet />
    </AdminSessionProvider>
  )
}

function AdminGateLayout() {
  return <AdminGate />
}

function PublicEditorialGate() {
  return <AdminGate />
}

const rootRoute = createRootRouteWithContext<RouterContext>()({
  component: RootLayout,
  notFoundComponent: () => (
    <FullPageLoader
      title="Page Not Found"
      subtitle="The requested research or admin view does not exist in the current UniFallMonitor app."
    />
  ),
})

const authShellRoute = createRoute({
  getParentRoute: () => rootRoute,
  id: 'auth-shell',
  component: AuthShell,
})

const loginRoute = createRoute({
  getParentRoute: () => authShellRoute,
  path: '/login',
  component: LoginPage,
})

const protectedRoute = createRoute({
  getParentRoute: () => rootRoute,
  id: 'admin-gate',
  beforeLoad: ensureAdminMeForProtected,
  component: AdminGateLayout,
})

const publicEditorialGateRoute = createRoute({
  getParentRoute: () => rootRoute,
  id: 'public-editorial-gate',
  beforeLoad: ensureAdminMe,
  component: PublicEditorialGate,
})

const editorialRoute = createRoute({
  getParentRoute: () => publicEditorialGateRoute,
  id: 'editorial-shell',
  component: EditorialShell,
})

const analyticsRoute = createRoute({
  getParentRoute: () => protectedRoute,
  id: 'analytics-shell',
  component: AnalyticsShell,
})

const detailRoute = createRoute({
  getParentRoute: () => protectedRoute,
  id: 'detail-shell',
  component: DetailShell,
})

const briefingRoute = createRoute({
  getParentRoute: () => editorialRoute,
  path: '/',
  component: BriefingPage,
})

const resultsRoute = createRoute({
  getParentRoute: () => editorialRoute,
  path: '/results',
  component: ResultsPage,
})

const docsRoute = createRoute({
  getParentRoute: () => editorialRoute,
  path: '/docs',
  component: DocsPage,
})

const architectureRoute = createRoute({
  getParentRoute: () => editorialRoute,
  path: '/architecture',
  component: ArchitecturePage,
})

const dashboardRoute = createRoute({
  getParentRoute: () => analyticsRoute,
  path: '/dashboard',
  component: DashboardPage,
})

const sessionsRoute = createRoute({
  getParentRoute: () => analyticsRoute,
  path: '/sessions',
  component: SessionsPage,
})

const libraryRoute = createRoute({
  getParentRoute: () => analyticsRoute,
  path: '/library',
  component: LibraryPage,
})

const exploreRoute = createRoute({
  getParentRoute: () => analyticsRoute,
  path: '/explore',
  component: ExplorePage,
})

function SessionDetailRouteComponent() {
  const rawParams = sessionDetailRoute.useParams() as unknown
  const sessionId =
    typeof rawParams === 'function'
      ? (rawParams as () => { sessionId: string })().sessionId
      : (rawParams as { sessionId: string }).sessionId

  return <SessionDetailPageView sessionId={sessionId} />
}

const sessionDetailRoute = createRoute({
  getParentRoute: () => detailRoute,
  path: '/sessions/$sessionId',
  component: SessionDetailRouteComponent,
})

const routeTree = rootRoute.addChildren([
  authShellRoute.addChildren([loginRoute]),
  publicEditorialGateRoute.addChildren([
    editorialRoute.addChildren([briefingRoute, resultsRoute, architectureRoute, docsRoute]),
  ]),
  protectedRoute.addChildren([
    analyticsRoute.addChildren([dashboardRoute, sessionsRoute, libraryRoute, exploreRoute]),
    detailRoute.addChildren([sessionDetailRoute]),
  ]),
])

export const router = createRouter({
  routeTree,
  defaultPreload: 'intent',
  context: { queryClient: undefined as unknown as QueryClient },
})

declare module '@tanstack/solid-router' {
  interface Register {
    router: typeof router
  }
}
