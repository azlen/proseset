import { useEffect, useRef, useState } from "react";
import { Check, Type, X } from "lucide-react";

export interface CardFont {
  id: string;
  name: string;
  family: string;
  weight: number;
  character: string;
}

export const CARD_FONTS: CardFont[] = [
  {
    id: "nunito",
    name: "Nunito",
    family: '"Nunito", ui-sans-serif, system-ui, sans-serif',
    weight: 600,
    character: "Friendly",
  },
  {
    id: "dm-sans",
    name: "DM Sans",
    family: '"DM Sans", ui-sans-serif, system-ui, sans-serif',
    weight: 600,
    character: "Neutral",
  },
  {
    id: "outfit",
    name: "Outfit",
    family: '"Outfit", ui-sans-serif, system-ui, sans-serif',
    weight: 600,
    character: "Geometric",
  },
  {
    id: "space-grotesk",
    name: "Space Grotesk",
    family: '"Space Grotesk", ui-sans-serif, system-ui, sans-serif',
    weight: 600,
    character: "Modern",
  },
  {
    id: "lexend",
    name: "Lexend",
    family: '"Lexend", ui-sans-serif, system-ui, sans-serif',
    weight: 600,
    character: "Readable",
  },
  {
    id: "barlow-condensed",
    name: "Barlow Condensed",
    family: '"Barlow Condensed", ui-sans-serif, system-ui, sans-serif',
    weight: 600,
    character: "Condensed",
  },
  {
    id: "archivo-black",
    name: "Archivo Black",
    family: '"Archivo Black", ui-sans-serif, system-ui, sans-serif',
    weight: 400,
    character: "Bold",
  },
  {
    id: "bebas-neue",
    name: "Bebas Neue",
    family: '"Bebas Neue", ui-sans-serif, system-ui, sans-serif',
    weight: 400,
    character: "Display",
  },
  {
    id: "bitter",
    name: "Bitter",
    family: '"Bitter", ui-serif, Georgia, serif',
    weight: 600,
    character: "Slab serif",
  },
  {
    id: "fraunces",
    name: "Fraunces",
    family: '"Fraunces", ui-serif, Georgia, serif',
    weight: 600,
    character: "Expressive",
  },
  {
    id: "ibm-plex-mono",
    name: "IBM Plex Mono",
    family: '"IBM Plex Mono", ui-monospace, monospace',
    weight: 600,
    character: "Monospace",
  },
];

interface FontPickerProps {
  value: CardFont;
  sampleWord: string;
  onChange: (font: CardFont) => void;
}

export function FontPicker({ value, sampleWord, onChange }: FontPickerProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;

    const handlePointerDown = (event: PointerEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  return (
    <div className="relative" ref={containerRef}>
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        className="flex max-w-36 items-center gap-1.5 rounded-full border border-border bg-background px-2.5 py-1.5 text-xs font-bold shadow-sm transition-colors hover:bg-muted"
        aria-expanded={open}
        aria-controls="card-font-picker"
      >
        <Type className="size-3.5 shrink-0" aria-hidden="true" />
        <span className="truncate">{value.name}</span>
      </button>

      {open && (
        <section
          id="card-font-picker"
          role="dialog"
          aria-label="Choose a card font"
          className="font-picker-panel absolute right-0 top-[calc(100%+0.5rem)] z-50 w-[min(23rem,calc(100vw-2rem))] overflow-hidden rounded-2xl border border-border bg-background shadow-xl"
        >
          <div className="flex items-start justify-between border-b border-border px-4 py-3">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-bold">Card font lab</h2>
                <span className="rounded-full bg-muted px-1.5 py-0.5 text-[0.625rem] font-bold uppercase tracking-wide text-muted-foreground">
                  Temporary
                </span>
              </div>
              <p className="mt-0.5 text-xs text-muted-foreground">Pick a sample. Changes apply instantly.</p>
            </div>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="-mr-1 rounded-full p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              aria-label="Close font picker"
            >
              <X className="size-4" aria-hidden="true" />
            </button>
          </div>

          <div className="font-picker-options grid grid-cols-2 gap-2 overflow-y-auto p-3">
            {CARD_FONTS.map((font) => {
              const selected = font.id === value.id;
              return (
                <button
                  key={font.id}
                  type="button"
                  onClick={() => onChange(font)}
                  aria-pressed={selected}
                  className={`relative min-w-0 rounded-xl border px-3 py-2.5 text-left transition-colors ${
                    selected
                      ? "border-foreground bg-foreground text-background"
                      : "border-border bg-card hover:border-foreground/40 hover:bg-muted/50"
                  }`}
                >
                  <span className={`block truncate pr-4 text-[0.6875rem] font-bold ${selected ? "text-background/70" : "text-muted-foreground"}`}>
                    {font.name}
                  </span>
                  {selected && (
                    <Check className="absolute right-2.5 top-2.5 size-3.5" aria-hidden="true" />
                  )}
                  <span
                    className="mt-2 block truncate text-lg uppercase leading-none tracking-wide"
                    style={{ fontFamily: font.family, fontWeight: font.weight }}
                  >
                    {sampleWord}
                  </span>
                  <span className={`mt-2 block text-[0.625rem] ${selected ? "text-background/60" : "text-muted-foreground/80"}`}>
                    {font.character}
                  </span>
                </button>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}
