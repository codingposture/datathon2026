import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = new URL(env.DJANGO_ORIGIN || 'http://127.0.0.1:8000');
  if (!['http:', 'https:'].includes(target.protocol) || !['127.0.0.1', 'localhost', '[::1]'].includes(target.hostname) || target.username || target.password || target.pathname !== '/' || target.search || target.hash) {
    throw new Error('DJANGO_ORIGIN must be a loopback HTTP(S) origin, such as http://127.0.0.1:8000.');
  }
  if (env.VITE_API_BASE && !/^\/(?!\/)[A-Za-z0-9_-]+(?:\/[A-Za-z0-9_-]+)*$/.test(env.VITE_API_BASE)) throw new Error('VITE_API_BASE must be a same-origin path without a trailing slash.');
  const csp = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self' ws://127.0.0.1:* ws://localhost:*; object-src 'none'; base-uri 'none'; form-action 'none'";
  return {
    plugins: [react(), {
      name: 'local-api-configuration',
      // React Fast Refresh uses an inline preamble in development only.
      transformIndexHtml(html, context) {
        return context.server ? html.replace("script-src 'self';", "script-src 'self' 'unsafe-inline';") : html;
      },
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (!req.url?.startsWith('/api/')) return next();
          const path = new URL(req.url, 'http://127.0.0.1').pathname;
          const allowed = (req.method === 'POST' && path === '/api/v1/analyze') || (req.method === 'GET' && /^\/api\/v1\/jobs\/[^/]+$/.test(path));
          if (!allowed) { res.writeHead(404); res.end('Unknown analysis API route.'); return; }
          next();
        });
      }
    }],
    server: {
      headers: { 'Content-Security-Policy': csp },
      proxy: {
        '/api/': {
          target: target.origin,
          changeOrigin: true,
          // Forward the user's X-API-Key unchanged; Django validates it.
          timeout: 30000,
          proxyTimeout: 30000,
          configure(proxy) {
            proxy.on('error', (_error, _req, res) => {
              if (res && !res.headersSent && typeof res.writeHead === 'function') {
                res.writeHead(502, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ detail: 'Cannot reach the local Django service. Check that it is running.' }));
              }
            });
          }
        }
      }
    },
    // Preview serves the production bundle only; it does not proxy API requests.
    preview: { headers: { 'Content-Security-Policy': csp.replace("script-src 'self' 'unsafe-inline'", "script-src 'self'") } },
    test: { environment: 'jsdom', setupFiles: './tests/setup.js', clearMocks: true }
  };
});
