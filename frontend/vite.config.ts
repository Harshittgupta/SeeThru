/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev proxy is why the app uses a RELATIVE `/v1` baseURL everywhere (T65).
// Vite inlines VITE_* env vars at BUILD time, so an env var could not repoint a
// built image between environments anyway -- and a relative baseURL behind an
// nginx proxy_pass means same-origin in prod: no CORS, no env var, one portable
// image. In dev, this proxy forwards /v1 to the local backend.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/v1": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
  },
});
