#!/usr/bin/env python3
"""
Generate puzzle JSON for the Proseset web game.

Usage:
  python3 generate_puzzle_json.py --deck word1,word2,...,word12 --date 2026-02-22
  python3 generate_puzzle_json.py --deck word1,word2,...,word12 --date 2026-02-22 --output web/public/puzzles/
"""

from __future__ import annotations
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Generate puzzle JSON for Proseset web game")
    parser.add_argument("--deck", required=True, help="Comma-separated 12 words")
    parser.add_argument("--date", required=True, help="Puzzle date (YYYY-MM-DD)")
    parser.add_argument("--output", default="web/public/puzzles/", help="Output directory")
    args = parser.parse_args()

    deck = [w.strip().lower() for w in args.deck.split(",")]
    if len(deck) != 12:
        print(f"Error: expected 12 cards, got {len(deck)}")
        sys.exit(1)

    puzzle = {
        "date": args.date,
        "cards": deck,
    }

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, f"{args.date}.json")
    with open(output_path, "w") as f:
        json.dump(puzzle, f, indent=2)

    print(f"Written to {output_path}")
    print(f"  Date: {args.date}")
    print(f"  Cards: {', '.join(deck)}")


if __name__ == "__main__":
    main()
