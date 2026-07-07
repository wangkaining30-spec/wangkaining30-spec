const CACHE = "kechen-v1";
const ASSETS = ["/wangkaining30-spec/", "/wangkaining30-spec/style.css", "/wangkaining30-spec/script.js"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)));
});

self.addEventListener("fetch", e => {
  e.respondWith(
    caches.match(e.request).then(r => r || fetch(e.request).then(resp => {
      if (resp.ok) { const clone = resp.clone(); caches.open(CACHE).then(c => c.put(e.request, clone)); }
      return resp;
    }))
  );
});
