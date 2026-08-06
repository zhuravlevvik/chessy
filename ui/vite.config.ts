import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: resolve(__dirname, "../src/chessy/api/static"),
    emptyOutDir: true,
    sourcemap: false,
  },
  test: {
    environment: "jsdom",
    setupFiles: "./tests/setup.ts",
  },
});
