import { defineConfig, type PluginOption } from 'vite'
import tailwindcss from '@tailwindcss/vite'
import solid from 'vite-plugin-solid'

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000'

const ANALYZE_BUNDLE =
  process.env.ANALYZE === 'true' || process.env.ANALYZE === '1'

const optionalPlugins: PluginOption[] = []
if (ANALYZE_BUNDLE) {
  // Lazy require so the plugin is only resolved when ANALYZE=true is set.
  // Keeps the default install footprint smaller for users who never analyse.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { visualizer } = require('rollup-plugin-visualizer') as typeof import('rollup-plugin-visualizer')
  optionalPlugins.push(
    visualizer({
      filename: 'dist/stats.html',
      template: 'treemap',
      gzipSize: true,
      brotliSize: false,
    }) as PluginOption,
  )
}

// Group every dependency under node_modules into a small set of stable vendor
// chunks so a one-line app change doesn't invalidate the whole 180 KB bundle on
// users' browser caches. The categories are coarse on purpose — finer
// splitting hurts HTTP/2 head-of-line and inflates total transfer.
//
// `app-shared` collects design-system primitives and the API client that
// every feature page reaches for, so they live in one cache-stable chunk
// instead of getting hoisted into the main entry on every app code change.
function vendorChunkFor(modulePath: string): string | undefined {
  if (!modulePath.includes('node_modules')) {
    if (/[\\/]src[\\/]components[\\/]v6\.tsx?$/.test(modulePath)) {
      return 'app-shared'
    }
    if (/[\\/]src[\\/]lib[\\/]api\.tsx?$/.test(modulePath)) {
      return 'app-shared'
    }
    return undefined
  }
  if (/[\\/]node_modules[\\/](solid-js|@solidjs[\\/]|solid-)/.test(modulePath)) {
    return 'vendor-solid'
  }
  if (/[\\/]node_modules[\\/]@tanstack[\\/]/.test(modulePath)) {
    return 'vendor-tanstack'
  }
  if (/[\\/]node_modules[\\/]motion[\\/]/.test(modulePath)) {
    return 'vendor-motion'
  }
  return 'vendor-misc'
}

export default defineConfig({
  plugins: [solid(), tailwindcss(), ...optionalPlugins],
  server: {
    proxy: {
      '/v1': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      '/health': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
    },
  },
  build: {
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks(id) {
          return vendorChunkFor(id)
        },
      },
    },
  },
})
