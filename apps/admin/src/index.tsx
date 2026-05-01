/* @refresh reload */
import { QueryClient } from '@tanstack/solid-query'
import { render } from 'solid-js/web'
import App from './App'
import {
  DEMO_MODE_ENABLED,
  getAdminMe,
  getDemoAdminSessionSnapshot,
} from './lib/api'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      refetchOnWindowFocus: false,
      staleTime: 30_000,
    },
    mutations: {
      retry: false,
    },
  },
})

// Race the auth probe against chunk loading: by the time the lazy layout
// chunks resolve, the cache typically already has a result, so the protected
// gate's beforeLoad becomes a synchronous cache hit instead of a fresh
// network round-trip stacked on top of the chunk waterfall.
if (DEMO_MODE_ENABLED) {
  const admin = getDemoAdminSessionSnapshot()
  if (admin) {
    queryClient.setQueryData(['admin', 'me'], admin)
  }
} else {
  void queryClient.prefetchQuery({
    queryKey: ['admin', 'me'],
    queryFn: getAdminMe,
  })
}

const root = document.getElementById('root')

render(() => <App queryClient={queryClient} />, root!)

queueMicrotask(() => {
  const splash = document.getElementById('boot-splash')
  splash?.parentNode?.removeChild(splash)
})
