import { defineConfig } from 'vite';

// GitHub Pages serves project sites from https://<user>.github.io/<repo>/, so the base path must be
// the repository name. The deploy workflow sets BASE_PATH automatically; locally it defaults to "/".
// Use BASE_PATH=./ for a fully relocatable build (works from any sub-folder / static host).
export default defineConfig({
  base: process.env.BASE_PATH || '/',
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 1500, // onnxruntime-web is large by nature
  },
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  server: { port: 5173 },
});
