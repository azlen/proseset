import { useEffect, useMemo, useState } from "react";
import { GameApp } from "./GameApp";
import { loadPuzzles, puzzleFromRaw, type RawPuzzle } from "../lib/puzzle";

const SCHEDULE_KEY = "doublespeak-planner-schedule";
const SLOT_COUNT = 21;

type DragPayload =
  | { type: "puzzle"; id: number }
  | { type: "slot"; index: number };

function formatDate(offset: number): string {
  const date = new Date();
  date.setDate(date.getDate() + offset);
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    weekday: "short",
  });
}

function puzzleSearchText(puzzle: RawPuzzle): string {
  return [
    puzzle.id,
    puzzle.anchor_word,
    puzzle.cards.join(" "),
    puzzle.made_words.join(" "),
    puzzle.longest_made_words?.join(" "),
    puzzle.enabled_target_words?.join(" "),
  ].join(" ").toLowerCase();
}

function getPuzzleTitle(puzzle: RawPuzzle): string {
  return puzzle.anchor_word ?? puzzle.longest_made_words?.[0] ?? `Puzzle ${puzzle.id}`;
}

export function AdminPlanner() {
  const [puzzles, setPuzzles] = useState<RawPuzzle[]>([]);
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [schedule, setSchedule] = useState<Array<number | null>>(() => {
    try {
      const saved = localStorage.getItem(SCHEDULE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved) as Array<number | null>;
        return Array.from({ length: SLOT_COUNT }, (_, i) => parsed[i] ?? null);
      }
    } catch {
      // Ignore invalid local planner state.
    }
    return Array.from({ length: SLOT_COUNT }, () => null);
  });

  useEffect(() => {
    loadPuzzles().then((loaded) => {
      setPuzzles(loaded);
      setSelectedId((current) => current ?? loaded[0]?.id ?? null);
    });
  }, []);

  useEffect(() => {
    localStorage.setItem(SCHEDULE_KEY, JSON.stringify(schedule));
  }, [schedule]);

  const puzzleById = useMemo(() => {
    return new Map(puzzles.map((puzzle) => [puzzle.id, puzzle]));
  }, [puzzles]);

  const selectedPuzzle = selectedId === null ? null : puzzleById.get(selectedId) ?? null;

  const filteredPuzzles = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return puzzles;
    const terms = normalized.split(/\s+/);
    return puzzles.filter((puzzle) => {
      const text = puzzleSearchText(puzzle);
      return terms.every((term) => text.includes(term));
    });
  }, [puzzles, query]);

  const setSlot = (index: number, puzzleId: number | null) => {
    setSchedule((current) => {
      const next = [...current];
      next[index] = puzzleId;
      return next;
    });
  };

  const addToFirstOpenSlot = (puzzleId: number) => {
    setSchedule((current) => {
      const next = [...current];
      const openIndex = next.findIndex((id) => id === null);
      next[openIndex === -1 ? next.length - 1 : openIndex] = puzzleId;
      return next;
    });
  };

  const handleDrop = (targetIndex: number, rawPayload: string) => {
    if (!rawPayload) return;
    const payload = JSON.parse(rawPayload) as DragPayload;
    if (payload.type === "puzzle") {
      setSlot(targetIndex, payload.id);
      setSelectedId(payload.id);
      return;
    }

    setSchedule((current) => {
      const next = [...current];
      const moved = next[payload.index] ?? null;
      next[payload.index] = next[targetIndex] ?? null;
      next[targetIndex] = moved;
      if (moved !== null) setSelectedId(moved);
      return next;
    });
  };

  return (
    <div className="h-[100dvh] w-full overflow-hidden bg-background text-foreground">
      <div className="grid h-full grid-cols-[minmax(300px,380px)_minmax(260px,340px)_minmax(390px,1fr)]">
        <section className="flex min-h-0 flex-col border-r border-border">
          <div className="border-b border-border px-4 py-3">
            <div className="flex items-baseline justify-between gap-3">
              <h1 className="text-xl font-black tracking-tight">Puzzle planner</h1>
              <a className="text-sm text-muted-foreground hover:text-foreground" href="/">
                Game
              </a>
            </div>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search cards or made words"
              className="mt-3 w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-foreground"
            />
            <div className="mt-2 text-xs text-muted-foreground">
              {filteredPuzzles.length} / {puzzles.length} puzzles
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto">
            {filteredPuzzles.map((puzzle) => {
              const selected = puzzle.id === selectedId;
              return (
                <button
                  key={puzzle.id}
                  draggable
                  onDragStart={(event) => {
                    event.dataTransfer.setData(
                      "application/json",
                      JSON.stringify({ type: "puzzle", id: puzzle.id } satisfies DragPayload),
                    );
                  }}
                  onClick={() => setSelectedId(puzzle.id)}
                  className={`block w-full border-b border-border px-4 py-3 text-left transition-colors ${
                    selected ? "bg-muted" : "hover:bg-muted/60"
                  }`}
                >
                  <div className="flex items-baseline justify-between gap-3">
                    <div className="min-w-0 truncate text-sm font-black uppercase">
                      {getPuzzleTitle(puzzle)}
                    </div>
                    <div className="shrink-0 text-xs text-muted-foreground">#{puzzle.id}</div>
                  </div>
                  <div className="mt-1 flex gap-2 text-xs text-muted-foreground">
                    <span>{puzzle.num_valid_combos ?? 0} combos</span>
                    <span>{puzzle.num_made_words_4plus} words</span>
                  </div>
                  <div className="mt-2 line-clamp-2 text-xs uppercase text-muted-foreground">
                    {puzzle.cards.join(" ")}
                  </div>
                </button>
              );
            })}
          </div>
        </section>

        <section className="flex min-h-0 flex-col border-r border-border">
          <div className="border-b border-border px-4 py-3">
            <h2 className="text-sm font-black uppercase tracking-wide">Calendar order</h2>
            <div className="mt-1 text-xs text-muted-foreground">
              Drag puzzles here, or drag scheduled days to reorder.
            </div>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto p-3">
            <div className="space-y-2">
              {schedule.map((puzzleId, index) => {
                const puzzle = puzzleId === null ? null : puzzleById.get(puzzleId) ?? null;
                return (
                  <div
                    key={index}
                    draggable={puzzle !== null}
                    onDragStart={(event) => {
                      event.dataTransfer.setData(
                        "application/json",
                        JSON.stringify({ type: "slot", index } satisfies DragPayload),
                      );
                    }}
                    onDragOver={(event) => event.preventDefault()}
                    onDrop={(event) => {
                      event.preventDefault();
                      handleDrop(index, event.dataTransfer.getData("application/json"));
                    }}
                    className="min-h-20 rounded-lg border border-dashed border-border bg-background p-3"
                  >
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <div className="text-xs font-bold uppercase text-muted-foreground">
                        {formatDate(index)}
                      </div>
                      {puzzle && (
                        <button
                          onClick={() => setSlot(index, null)}
                          className="text-xs text-muted-foreground hover:text-foreground"
                        >
                          Clear
                        </button>
                      )}
                    </div>
                    {puzzle ? (
                      <button
                        onClick={() => setSelectedId(puzzle.id)}
                        className="block w-full text-left"
                      >
                        <div className="truncate text-sm font-black uppercase">
                          {getPuzzleTitle(puzzle)}
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">
                          #{puzzle.id} · {puzzle.num_valid_combos ?? 0} combos · {puzzle.num_made_words_4plus} words
                        </div>
                      </button>
                    ) : (
                      <div className="text-sm text-muted-foreground">Drop puzzle</div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        <section className="grid min-h-0 grid-cols-[minmax(260px,320px)_minmax(390px,1fr)]">
          <aside className="min-h-0 overflow-y-auto border-r border-border p-4">
            {selectedPuzzle ? (
              <>
                <div className="text-xs font-bold uppercase tracking-wide text-muted-foreground">
                  Selected
                </div>
                <h2 className="mt-1 text-2xl font-black uppercase tracking-tight">
                  {getPuzzleTitle(selectedPuzzle)}
                </h2>
                <div className="mt-2 text-sm text-muted-foreground">
                  Puzzle #{selectedPuzzle.id}
                </div>
                <button
                  onClick={() => addToFirstOpenSlot(selectedPuzzle.id)}
                  className="mt-4 w-full rounded-full border-2 border-foreground bg-foreground px-4 py-2 text-sm font-bold text-background"
                >
                  Add to next open day
                </button>

                <div className="mt-5 grid grid-cols-2 gap-2 text-center">
                  <div className="rounded-lg bg-muted px-3 py-3">
                    <div className="text-xl font-black">{selectedPuzzle.num_valid_combos ?? 0}</div>
                    <div className="text-xs uppercase text-muted-foreground">Combos</div>
                  </div>
                  <div className="rounded-lg bg-muted px-3 py-3">
                    <div className="text-xl font-black">{selectedPuzzle.num_made_words_4plus}</div>
                    <div className="text-xs uppercase text-muted-foreground">Words</div>
                  </div>
                </div>

                <div className="mt-5">
                  <div className="mb-2 text-xs font-bold uppercase text-muted-foreground">Cards</div>
                  <div className="flex flex-wrap gap-1.5">
                    {selectedPuzzle.cards.map((card) => (
                      <span key={card} className="rounded-md bg-muted px-2 py-1 text-xs font-bold uppercase">
                        {card}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="mt-5">
                  <div className="mb-2 text-xs font-bold uppercase text-muted-foreground">Longest words</div>
                  <div className="flex flex-wrap gap-1.5">
                    {(selectedPuzzle.longest_made_words ?? selectedPuzzle.made_words.slice(0, 10)).map((word) => (
                      <span key={word} className="rounded-md bg-muted px-2 py-1 text-xs font-bold uppercase">
                        {word}
                      </span>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div className="text-sm text-muted-foreground">Loading puzzles...</div>
            )}
          </aside>

          <main className="min-h-0 overflow-hidden">
            {selectedPuzzle ? (
              <GameApp
                key={selectedPuzzle.id}
                initialPuzzle={puzzleFromRaw(selectedPuzzle)}
                persistProgress={false}
                showRandom={false}
                compact
              />
            ) : (
              <div className="flex h-full items-center justify-center text-muted-foreground">
                Select a puzzle
              </div>
            )}
          </main>
        </section>
      </div>
    </div>
  );
}
