export function scoreWord(word: string): number {
  if (word.length < 4) return 0;
  return word.length - 3;
}

export function scoreWords(words: string[]): number {
  const uniqueWords = new Set(words);
  let score = 0;
  for (const word of uniqueWords) {
    score += scoreWord(word);
  }
  return score;
}
