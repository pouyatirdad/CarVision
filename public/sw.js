/* CarVision service worker: lets the app (shell + WASM runtime + labels) open offline after one visit.
 *
 * - Navigations (index.html): network-first, fall back to the cached copy -> new deployments show up
 *   immediately when online, and the app still opens when offline.
 * - Same-origin static assets (hashed JS/CSS/WASM, labels): cache-first, filled lazily.
 * - The ONNX models are NOT handled here: the app stores them itself in the Cache API
 *   ("carvision-models-v1") with download progress and integrity checks.
 * Nothing about user images ever passes through here (they never leave the page / worker).
 */
const CACHE = 'carvision-shell-v1';

self.addEventListener('install', () => self.skipWaiting());

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      for (const key of await caches.keys()) {
        if (key.startsWith('carvision-shell-') && key !== CACHE) await caches.delete(key);
      }
      await self.clients.claim();
    })(),
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.includes('/models/')) return; // handled by the app's own model cache

  if (req.mode === 'navigate') {
    event.respondWith(
      (async () => {
        const cache = await caches.open(CACHE);
        try {
          const fresh = await fetch(req);
          if (fresh.ok) cache.put(req, fresh.clone());
          return fresh;
        } catch {
          return (await cache.match(req, { ignoreSearch: true })) || (await cache.match(self.registration.scope)) || Response.error();
        }
      })(),
    );
    return;
  }

  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE);
      const hit = await cache.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) cache.put(req, res.clone());
      return res;
    })(),
  );
});
