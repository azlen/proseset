import { serve } from "bun";
import index from "./index.html";

const dictText = await Bun.file("public/dictionary.txt").text();
const puzzlesText = await Bun.file("public/puzzles/megapuzzle2-1000-20260430-154557.json").text();
const manifestText = await Bun.file("public/manifest.json").text();
const swText = await Bun.file("public/sw.js").text();

const port = Number(process.env.PORT ?? 5173);

const server = serve({
  port,
  hostname: "127.0.0.1",
  routes: {
    "/dictionary.txt": () => new Response(dictText, {
      headers: { "Content-Type": "text/plain" },
    }),
    "/newpuzzle.json": () => new Response(puzzlesText, {
      headers: { "Content-Type": "application/json" },
    }),
    "/puzzles/megapuzzle2-1000-20260430-154557.json": () => new Response(puzzlesText, {
      headers: { "Content-Type": "application/json" },
    }),
    "/manifest.json": () => new Response(manifestText, {
      headers: { "Content-Type": "application/json" },
    }),
    "/sw.js": () => new Response(swText, {
      headers: {
        "Content-Type": "application/javascript",
        "Cache-Control": "no-cache, no-store, must-revalidate",
      },
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
