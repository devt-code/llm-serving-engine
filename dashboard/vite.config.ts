import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" so the built site works from any path (GitHub Pages, a subfolder, a file server).
export default defineConfig({ plugins: [react()], base: "./" });
