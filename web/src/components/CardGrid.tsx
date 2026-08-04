import { WordCard } from "./WordCard";

interface CardGridProps {
  cards: string[];
  selectedCards: string[];
  cardUseCounts: ReadonlyMap<string, number>;
  onSelectCard: (card: string) => void;
  onDeselectCard: (card: string) => void;
}

export function CardGrid({
  cards,
  selectedCards,
  cardUseCounts,
  onSelectCard,
  onDeselectCard,
}: CardGridProps) {
  return (
    <div className="grid grid-cols-3 gap-2 w-full">
      {cards.map((card) => {
        const selIdx = selectedCards.indexOf(card);
        return (
          <WordCard
            key={card}
            word={card}
            selected={selIdx !== -1}
            useCount={cardUseCounts.get(card) ?? 0}
            selectionIndex={selIdx !== -1 ? selIdx : null}
            onSelect={() => onSelectCard(card)}
            onDeselect={() => onDeselectCard(card)}
          />
        );
      })}
    </div>
  );
}
