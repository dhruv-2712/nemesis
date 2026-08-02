import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Local dev: proxy /api to the backend so the client can use same-origin
    // paths (matching the Vercel rewrite proxy used in production).
    proxy: { "/api": "http://localhost:8000" },
  },
});
