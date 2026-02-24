import { useState, useEffect, useCallback, useRef } from "react";
import "./index.css";

// ============================================================================
//                              TYPES
// ============================================================================

interface BrowseDecomp {
  leftFrag: string;
  interior: string[];
  rightFrag: string;
  score: number;
  resolvedCards: string[];
  inDeck: boolean[];
  missing: number;
}

interface BrowseEntry {
  word: string;
  length: number;
  frequency: number;
  decomps: BrowseDecomp[];
  bestMissing: number;
}

interface ComboResult {
  cards: string[];
  concat: string;
  segmentations: string[][];
}

interface MadeWord {
  word: string;
  length: number;
  frequency: number;
  combos: { cards: string[]; segmentation: string[] }[];
}

interface AnalysisResult {
  enabledTargets: unknown[];
  nearTargets: unknown[];
  combos: ComboResult[];
  madeWords: MadeWord[];
}

// ============================================================================
//                              API
// ============================================================================

async function apiSearch(query: string): Promise<string[]> {
  const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
  return res.json();
}

async function apiBrowse(
  deck: string[],
  opts: { minLength?: number; maxMissing?: number; limit?: number; search?: string } = {}
): Promise<BrowseEntry[]> {
  const res = await fetch("/api/browse", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ deck, ...opts }),
  });
  return res.json();
}

async function apiAnalyze(deck: string[]): Promise<AnalysisResult> {
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ deck }),
  });
  return res.json();
}

// ============================================================================
//                              COMPONENTS
// ============================================================================

/** A single resolved card in a decomposition row - clickable to add */
function CardChip({
  card,
  inDeck,
  onAdd,
  leftFrag,
  rightFrag,
}: {
  card: string;
  inDeck: boolean;
  onAdd: (card: string) => void;
  leftFrag?: string;
  rightFrag?: string;
}) {
  // Show the card with fragment highlighting
  let display: React.ReactNode = card;
  if (leftFrag) {
    // This card provides a left fragment - the fragment part is at the end
    const leftover = card.slice(0, card.length - leftFrag.length);
    display = (
      <>
        <span className="opacity-30">{leftover}</span>
        <span className="underline">{leftFrag}</span>
      </>
    );
  } else if (rightFrag) {
    // This card provides a right fragment - the fragment part is at the start
    const leftover = card.slice(rightFrag.length);
    display = (
      <>
        <span className="underline">{rightFrag}</span>
        <span className="opacity-30">{leftover}</span>
      </>
    );
  }

  if (inDeck) {
    return (
      <span className="inline-block px-1 border border-black bg-black text-white">
        {display}
      </span>
    );
  }

  return (
    <span
      className="inline-block px-1 border border-black border-dashed cursor-pointer hover:bg-black hover:text-white transition-colors"
      onClick={() => onAdd(card)}
      title={`Add "${card}" to deck`}
    >
      {display}
    </span>
  );
}

/** A decomposition row showing all resolved cards */
function DecompRow({
  decomp,
  onAdd,
}: {
  decomp: BrowseDecomp;
  onAdd: (card: string) => void;
}) {
  let cardIdx = 0;

  const chips: React.ReactNode[] = [];

  if (decomp.leftFrag) {
    const card = decomp.resolvedCards[cardIdx];
    const has = decomp.inDeck[cardIdx];
    chips.push(
      <CardChip
        key={cardIdx}
        card={card}
        inDeck={has}
        onAdd={onAdd}
        leftFrag={decomp.leftFrag}
      />
    );
    cardIdx++;
  }

  for (const interior of decomp.interior) {
    const has = decomp.inDeck[cardIdx];
    chips.push(
      <CardChip key={cardIdx} card={interior} inDeck={has} onAdd={onAdd} />
    );
    cardIdx++;
  }

  if (decomp.rightFrag) {
    const card = decomp.resolvedCards[cardIdx];
    const has = decomp.inDeck[cardIdx];
    chips.push(
      <CardChip
        key={cardIdx}
        card={card}
        inDeck={has}
        onAdd={onAdd}
        rightFrag={decomp.rightFrag}
      />
    );
    cardIdx++;
  }

  return (
    <div className="flex items-center gap-1 flex-wrap">
      {chips.map((chip, i) => (
        <span key={i} className="flex items-center gap-1">
          {i > 0 && <span className="opacity-20">+</span>}
          {chip}
        </span>
      ))}
    </div>
  );
}

/** Search input for adding cards manually */
function SearchInput({
  onAdd,
  deck,
}: {
  onAdd: (word: string) => void;
  deck: string[];
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<string[]>([]);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const deckSet = new Set(deck);

  useEffect(() => {
    if (query.length === 0) {
      setResults([]);
      return;
    }
    const timeout = setTimeout(async () => {
      const r = await apiSearch(query);
      setResults(r.filter((w) => !deckSet.has(w)));
      setSelectedIdx(0);
    }, 100);
    return () => clearTimeout(timeout);
  }, [query]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIdx((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && results.length > 0) {
      e.preventDefault();
      onAdd(results[selectedIdx]);
      setQuery("");
      setResults([]);
    } else if (e.key === "Escape") {
      setQuery("");
      setResults([]);
    }
  };

  return (
    <div className="relative">
      <div className="flex items-center border-b border-black">
        <span className="px-2 opacity-30">&gt;</span>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value.toLowerCase())}
          onKeyDown={handleKeyDown}
          placeholder="add card..."
          className="w-full py-1 px-1 bg-transparent outline-none placeholder:opacity-20 text-xs"
        />
      </div>
      {results.length > 0 && (
        <div className="absolute z-50 left-0 right-0 bg-white border border-t-0 border-black max-h-40 overflow-y-auto text-xs">
          {results.map((word, i) => (
            <div
              key={word}
              className={`px-3 py-0.5 cursor-pointer ${i === selectedIdx ? "bg-black text-white" : "hover:bg-gray-100"}`}
              onClick={() => {
                onAdd(word);
                setQuery("");
                setResults([]);
                inputRef.current?.focus();
              }}
            >
              {word}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================================
//                              MAIN APP
// ============================================================================

export function App() {
  const [deck, setDeck] = useState<string[]>([]);
  const [browseResults, setBrowseResults] = useState<BrowseEntry[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [targetSearch, setTargetSearch] = useState("");
  const [minLength, setMinLength] = useState(8);
  const [activeTab, setActiveTab] = useState<"browse" | "combos" | "made">("browse");
  const deckSet = new Set(deck);

  // Fetch browse results when deck or filters change
  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    apiBrowse(deck, {
      minLength,
      maxMissing: deck.length === 0 ? 4 : undefined,
      limit: 150,
      search: targetSearch || undefined,
    }).then((r) => {
      if (!cancelled) {
        setBrowseResults(r);
        setLoading(false);
      }
    });

    return () => { cancelled = true; };
  }, [deck, minLength, targetSearch]);

  // Fetch brute-force analysis when deck has 3+ cards
  useEffect(() => {
    if (deck.length < 3) {
      setAnalysis(null);
      return;
    }
    let cancelled = false;
    apiAnalyze(deck).then((a) => {
      if (!cancelled) setAnalysis(a);
    });
    return () => { cancelled = true; };
  }, [deck]);

  const addCard = useCallback(
    (word: string) => {
      if (!deckSet.has(word)) {
        setDeck((d) => [...d, word]);
      }
    },
    [deckSet]
  );

  const removeCard = useCallback((word: string) => {
    setDeck((d) => d.filter((w) => w !== word));
  }, []);

  const enabledCount = browseResults.filter((e) => e.bestMissing === 0).length;
  const nearCount = browseResults.filter((e) => e.bestMissing === 1).length;

  return (
    <div className="min-h-screen flex flex-col">
      {/* HEADER */}
      <header className="border-b border-black px-4 py-1.5 flex items-center justify-between">
        <h1 className="text-xs font-bold tracking-widest uppercase">
          Proseset Designer
        </h1>
        <div className="flex items-center gap-4 text-xs">
          {enabledCount > 0 && (
            <span className="font-bold">{enabledCount} enabled</span>
          )}
          {nearCount > 0 && (
            <span className="opacity-50">{nearCount} near</span>
          )}
          <span className="opacity-30">{deck.length}/12 cards</span>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* LEFT - Deck */}
        <div className="w-48 border-r border-black flex flex-col flex-shrink-0">
          <div className="border-b border-black px-2 py-1 text-[10px] font-bold uppercase tracking-wider opacity-40">
            Deck
          </div>
          <SearchInput onAdd={addCard} deck={deck} />

          <div className="flex-1 overflow-y-auto">
            {deck.length === 0 && (
              <div className="px-2 py-4 text-[10px] opacity-20 text-center">
                Click cards in the target list to add them, or search above
              </div>
            )}
            {deck.map((word) => (
              <div
                key={word}
                className="flex items-center justify-between px-2 py-px hover:bg-gray-50 group text-xs"
              >
                <span>{word}</span>
                <button
                  onClick={() => removeCard(word)}
                  className="opacity-0 group-hover:opacity-30 hover:!opacity-100 cursor-pointer"
                >
                  x
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* MAIN - Browse / Analysis */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* TAB BAR + FILTERS */}
          <div className="flex items-center border-b border-black text-xs">
            {(
              [
                ["browse", "Targets"],
                ["combos", "Combos"],
                ["made", "Made Words"],
              ] as const
            ).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={`px-3 py-1 border-r border-black cursor-pointer ${
                  activeTab === key
                    ? "bg-black text-white font-bold"
                    : "hover:bg-gray-50"
                }`}
              >
                {label}
                {key === "combos" && analysis && (
                  <span className="ml-1 opacity-60">({analysis.combos.length})</span>
                )}
                {key === "made" && analysis && (
                  <span className="ml-1 opacity-60">({analysis.madeWords.length})</span>
                )}
              </button>
            ))}

            {activeTab === "browse" && (
              <div className="flex items-center gap-2 ml-2 flex-1">
                <span className="opacity-30">min</span>
                <input
                  type="number"
                  value={minLength}
                  onChange={(e) => setMinLength(parseInt(e.target.value) || 4)}
                  className="w-10 px-1 py-0.5 border border-black bg-transparent text-center"
                  min={4}
                  max={20}
                />
                <span className="opacity-30 ml-2">search</span>
                <input
                  type="text"
                  value={targetSearch}
                  onChange={(e) => setTargetSearch(e.target.value.toLowerCase())}
                  placeholder="filter targets..."
                  className="flex-1 px-2 py-0.5 bg-transparent outline-none placeholder:opacity-20"
                />
              </div>
            )}

            {loading && (
              <span className="px-2 py-1 opacity-30 ml-auto">...</span>
            )}
          </div>

          {/* CONTENT */}
          <div className="flex-1 overflow-y-auto">
            {/* BROWSE TAB */}
            {activeTab === "browse" && (
              <div>
                {browseResults.length === 0 && !loading && (
                  <div className="p-4 opacity-30 text-xs text-center">
                    No targets match the current filters.
                  </div>
                )}
                {browseResults.map((entry) => (
                  <div
                    key={entry.word}
                    className={`border-b border-gray-100 px-3 py-1.5 ${
                      entry.bestMissing === 0
                        ? "bg-gray-50"
                        : ""
                    }`}
                  >
                    <div className="flex items-baseline gap-2 mb-0.5">
                      <span
                        className={`font-bold ${
                          entry.bestMissing === 0 ? "" : "opacity-70"
                        }`}
                      >
                        {entry.word}
                      </span>
                      <span className="text-[10px] opacity-30">
                        {entry.length}
                      </span>
                      <span className="text-[10px] opacity-30">
                        f={entry.frequency.toFixed(1)}
                      </span>
                      {entry.bestMissing === 0 && (
                        <span className="text-[10px] font-bold">ENABLED</span>
                      )}
                      {entry.bestMissing > 0 && (
                        <span className="text-[10px] opacity-30">
                          {entry.bestMissing} missing
                        </span>
                      )}
                    </div>
                    {entry.decomps.slice(0, 2).map((d, i) => (
                      <div key={i} className={`ml-4 ${i > 0 ? "mt-0.5 opacity-40" : ""}`}>
                        <DecompRow decomp={d} onAdd={addCard} />
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            )}

            {/* COMBOS TAB */}
            {activeTab === "combos" && (
              <div className="p-3">
                {(!analysis || analysis.combos.length === 0) && (
                  <div className="opacity-30 text-xs">
                    {deck.length < 3
                      ? "Need at least 3 cards."
                      : "No valid combos found."}
                  </div>
                )}
                {analysis?.combos.slice(0, 100).map((c, i) => (
                  <div key={i} className="mb-2 text-xs">
                    <div>
                      <span className="font-bold">
                        {c.cards.join(" + ")}
                      </span>
                      <span className="opacity-30"> = </span>
                      <span className="opacity-50">{c.concat}</span>
                    </div>
                    {c.segmentations.slice(0, 2).map((seg, j) => (
                      <div key={j} className="pl-4 opacity-60">
                        {"\u2192"} {seg.join(" | ")}
                      </div>
                    ))}
                    {c.segmentations.length > 2 && (
                      <div className="pl-4 opacity-30">
                        +{c.segmentations.length - 2} more
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* MADE WORDS TAB */}
            {activeTab === "made" && (
              <div className="p-3">
                {(!analysis || analysis.madeWords.length === 0) && (
                  <div className="opacity-30 text-xs">
                    {deck.length < 3
                      ? "Need at least 3 cards."
                      : "No made words yet."}
                  </div>
                )}
                {analysis?.madeWords.slice(0, 100).map((mw) => (
                  <div key={mw.word} className="mb-1 text-xs">
                    <div className="flex items-baseline gap-2">
                      <span className="font-bold">{mw.word}</span>
                      <span className="opacity-30">
                        {mw.length} f={mw.frequency.toFixed(1)}
                      </span>
                      <span className="opacity-20">
                        {mw.combos.length} combo{mw.combos.length !== 1 ? "s" : ""}
                      </span>
                    </div>
                    {mw.combos.slice(0, 1).map((c, i) => (
                      <div key={i} className="pl-4 opacity-50">
                        {c.cards.join(" + ")} {"\u2192"} {c.segmentation.join(" | ")}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
