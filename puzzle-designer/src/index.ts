import { serve } from "bun";
import index from "./index.html";
import { PuzzleEngine, type GameData } from "./server/engine";

// Load game data
console.log("Loading game data...");
const dataFile = Bun.file("public/data.json");
const gameData: GameData = await dataFile.json();
console.log(
  `Loaded: ${gameData.deckWords.length} deck words, ${gameData.segWords.length} seg words, ${Object.keys(gameData.decompositions).length} decomposable targets`
);

const engine = new PuzzleEngine(gameData);
console.log("Engine ready.");

const server = serve({
  routes: {
    "/*": index,

    "/api/search": {
      async GET(req) {
        const url = new URL(req.url);
        const q = url.searchParams.get("q") || "";
        const results = engine.searchWords(q, 30);
        return Response.json(results);
      },
    },

    "/api/analyze": {
      async POST(req) {
        const { deck } = (await req.json()) as { deck: string[] };
        if (!Array.isArray(deck)) {
          return Response.json({ error: "deck must be an array" }, { status: 400 });
        }
        const result = engine.analyze(deck);
        return Response.json(result);
      },
    },

    "/api/suggest": {
      async POST(req) {
        const { deck } = (await req.json()) as { deck: string[] };
        if (!Array.isArray(deck)) {
          return Response.json({ error: "deck must be an array" }, { status: 400 });
        }
        const suggestions = engine.rankCandidates(deck, 30);
        return Response.json(suggestions);
      },
    },

    "/api/browse": {
      async POST(req) {
        const body = (await req.json()) as {
          deck: string[];
          minLength?: number;
          maxMissing?: number;
          limit?: number;
          search?: string;
        };
        if (!Array.isArray(body.deck)) {
          return Response.json({ error: "deck must be an array" }, { status: 400 });
        }
        const results = engine.browse(body.deck, {
          minLength: body.minLength,
          maxMissing: body.maxMissing,
          limit: body.limit,
          search: body.search,
        });
        return Response.json(results);
      },
    },
  },

  development: process.env.NODE_ENV !== "production" && {
    hmr: true,
    console: true,
  },
});

console.log(`Server running at ${server.url}`);
