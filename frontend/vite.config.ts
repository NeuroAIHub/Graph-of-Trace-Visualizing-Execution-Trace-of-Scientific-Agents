import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Standalone Graph of Trace viewer.
// During dev, set GOT_API to proxy /api to a backend that serves got.json;
// otherwise the app reads a static ./got.json from the public/ dir.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 4500,
  },
});
