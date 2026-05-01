import { createSignal, onMount } from 'solid-js'

const STORAGE_KEY = 'unifall-theme'

const [theme, setThemeSignal] = createSignal<'light' | 'dark'>('light')

function applyTheme(next: 'light' | 'dark') {
  if (typeof document === 'undefined') return
  if (next === 'dark') document.documentElement.setAttribute('data-theme', 'dark')
  else document.documentElement.removeAttribute('data-theme')
}

export function useTheme() {
  onMount(() => {
    const saved = localStorage.getItem(STORAGE_KEY) as 'light' | 'dark' | null
    if (saved === 'dark' || saved === 'light') {
      setThemeSignal(saved)
      applyTheme(saved)
    }
  })

  return {
    theme,
    toggle() {
      const next = theme() === 'dark' ? 'light' : 'dark'
      setThemeSignal(next)
      applyTheme(next)
      try {
        localStorage.setItem(STORAGE_KEY, next)
      } catch {
        // localStorage may be unavailable (private mode); ignore.
      }
    },
  }
}
