// Service Worker for Marine Diesel RPM Meter PWA
// Strategy: Cache-First for all static assets, offline fallback

const CACHE_NAME = 'rpm-meter-v12';
const ASSETS = [
    './',
    './index.html',
    './css/style.css',
    './js/app.js',
    './js/audio-engine.js',
    './js/dsp.js',
    './js/gauge.js',
    './js/spectrum.js',
    './manifest.json',
    './icons/icon-192.png',
    './icons/icon-512.png',
    './offline.html'
];

// Install: pre-cache all static assets
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(ASSETS))
            .then(() => self.skipWaiting())
    );
});

// Activate: clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) =>
            Promise.all(
                keys
                    .filter((key) => key !== CACHE_NAME)
                    .map((key) => caches.delete(key))
            )
        ).then(() => self.clients.claim())
    );
});

// Fetch: Cache-First strategy with offline fallback
self.addEventListener('fetch', (event) => {
    // Only handle same-origin GET requests
    if (event.request.method !== 'GET') return;

    event.respondWith(
        caches.match(event.request)
            .then((cached) => {
                if (cached) return cached;
                return fetch(event.request)
                    .then((response) => {
                        // Cache new successful responses
                        if (response.ok) {
                            const clone = response.clone();
                            caches.open(CACHE_NAME)
                                .then((cache) => cache.put(event.request, clone));
                        }
                        return response;
                    })
                    .catch(() => {
                        // Offline fallback for navigation requests
                        if (event.request.mode === 'navigate') {
                            return caches.match('./offline.html');
                        }
                    });
            })
    );
});
