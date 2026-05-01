import { Link, useRouterState } from '@tanstack/solid-router'
import { For, type JSXElement } from 'solid-js'

type SidebarItem = {
  to: string
  label: string
  count?: string | number
  countAlert?: boolean
  icon?: JSXElement
  exact?: boolean
}

type SidebarSection = {
  heading: string
  items: SidebarItem[]
}

const editorialIcon = {
  briefing: (
    <svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
    </svg>
  ),
  results: (
    <svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  ),
  methods: (
    <svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    </svg>
  ),
  architecture: (
    <svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <rect x="3" y="3" width="7" height="7" />
      <rect x="14" y="3" width="7" height="7" />
      <rect x="3" y="14" width="7" height="7" />
      <rect x="14" y="14" width="7" height="7" />
    </svg>
  ),
}

const analyticsIcon = {
  dashboard: (
    <svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <rect x="3" y="3" width="7" height="9" />
      <rect x="14" y="3" width="7" height="5" />
      <rect x="14" y="12" width="7" height="9" />
      <rect x="3" y="16" width="7" height="5" />
    </svg>
  ),
  sessions: (
    <svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M3 6h18M3 12h18M3 18h18" />
    </svg>
  ),
  library: (
    <svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M2 4h6a2 2 0 0 1 2 2v14a2 2 0 0 0-2-2H2zM22 4h-6a2 2 0 0 0-2 2v14a2 2 0 0 1 2-2h6z" />
    </svg>
  ),
  explore: (
    <svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.5-3.5" />
    </svg>
  ),
}

export const editorialSidebar: SidebarSection[] = [
  {
    heading: 'Narrative',
    items: [
      { to: '/', label: 'Briefing', icon: editorialIcon.briefing, exact: true },
      { to: '/results', label: 'Results', icon: editorialIcon.results },
      { to: '/architecture', label: 'Architecture', icon: editorialIcon.architecture },
      { to: '/docs', label: 'Methods & docs', icon: editorialIcon.methods },
    ],
  },
]

export const analyticsSidebar: SidebarSection[] = [
  {
    heading: 'Operations',
    items: [
      { to: '/dashboard', label: 'Dashboard', icon: analyticsIcon.dashboard },
      { to: '/sessions', label: 'Sessions', icon: analyticsIcon.sessions },
      { to: '/library', label: 'Library', icon: analyticsIcon.library },
      { to: '/explore', label: 'Explore', icon: analyticsIcon.explore },
    ],
  },
]

export function Sidebar(props: { sections: SidebarSection[] }) {
  const routerState = useRouterState()
  const currentPath = () => routerState().location.pathname

  return (
    <aside class="sidebar">
      <For each={props.sections}>
        {(section) => (
          <div class="side-section">
            <div class="side-h">{section.heading}</div>
            <For each={section.items}>
              {(item) => {
                const active = () =>
                  item.exact ? currentPath() === item.to : currentPath().startsWith(item.to)
                return (
                  <Link
                    to={item.to}
                    class={`nav-item${active() ? ' active' : ''}`}
                  >
                    {item.icon}
                    {item.label}
                    {item.count !== undefined && (
                      <span class={`count${item.countAlert ? ' alert' : ''}`}>{item.count}</span>
                    )}
                  </Link>
                )
              }}
            </For>
          </div>
        )}
      </For>
    </aside>
  )
}
