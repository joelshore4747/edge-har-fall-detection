import { createContext, useContext, type ParentProps } from 'solid-js'
import type { AdminAuthSession } from '../lib/api'

const AdminSessionContext = createContext<AdminAuthSession>()

export function AdminSessionProvider(props: ParentProps<{ admin: AdminAuthSession }>) {
  return (
    <AdminSessionContext.Provider value={props.admin}>
      {props.children}
    </AdminSessionContext.Provider>
  )
}

export function useAdminSession() {
  const session = useContext(AdminSessionContext)
  if (!session) {
    throw new Error('useAdminSession must be used within an AdminSessionProvider.')
  }
  return session
}
