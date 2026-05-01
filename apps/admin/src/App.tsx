import { QueryClientProvider, type QueryClient } from '@tanstack/solid-query'
import { RouterProvider } from '@tanstack/solid-router'
import { router } from './router'

export default function App(props: { queryClient: QueryClient }) {
  return (
    <QueryClientProvider client={props.queryClient}>
      <RouterProvider router={router} context={{ queryClient: props.queryClient }} />
    </QueryClientProvider>
  )
}
