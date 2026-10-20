// Convenient Windows/local entry point: node start.mjs
// Dependencies must already be installed. No videos are read or sent on startup.
import { createServer } from 'vite';
import { fileURLToPath } from 'node:url';

try {
  const server = await createServer({
    root: fileURLToPath(new URL('.', import.meta.url)),
    server: { host: '127.0.0.1', port: 4173, strictPort: true }
  });
  await server.listen();
  server.printUrls();
} catch (error) {
  console.error(`Unable to start the frontend: ${error.message}`);
  process.exitCode = 1;
}
