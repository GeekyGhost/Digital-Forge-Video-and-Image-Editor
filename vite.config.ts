import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: { ignored: ["**/src-tauri/**", "**/models/**", "**/cache/**"] },
  },
  build: {
    // Paths are relative to the project root (npm runs vite from there).
    rollupOptions: {
      input: {
        main: "index.html",
        // VFX Phase 0 spike page. Built into dist/ but only opened when the app is launched with
        // AIV_SPIKES=1 (Run-VFX-Spikes.bat). Delete this line with the spikes/ folder.
        spikes: "spikes/index.html",
      },
    },
  },
});
