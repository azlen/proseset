import { serve } from "bun";
import index from "./index.html";

const dictText = await Bun.file("public/dictionary.txt").text();
const puzzlesText = await Bun.file("public/puzzles.json").text();
const manifestText = await Bun.file("public/manifest.json").text();
const swText = await Bun.file("public/sw.js").text();

const server = serve({
  port: 5173,
  routes: {
    "/dictionary.txt": () => new Response(dictText, {
      headers: { "Content-Type": "text/plain" },
    }),
    "/puzzles.json": () => new Response(puzzlesText, {
      headers: { "Content-Type": "application/json" },
    }),
    "/manifest.json": () => new Response(manifestText, {
      headers: { "Content-Type": "application/json" },
    }),
    "/sw.js": () => new Response(swText, {
      headers: { "Content-Type": "application/javascript" },
    }),
    "/icon-192.png": async () => new Response(await Bun.file("public/icon-192.png").arrayBuffer(), {
      headers: { "Content-Type": "image/png" },
    }),
    "/icon-512.png": async () => new Response(await Bun.file("public/icon-512.png").arrayBuffer(), {
      headers: { "Content-Type": "image/png" },
    }),
    "/*": index,
  },
  development: process.env.NODE_ENV !== "production" && {
    hmr: true,
    console: true,
  },
});

console.log(`Server running at ${server.url}`);
