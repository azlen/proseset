# Proseset

Read README.md for game rules and concept.

## Directory Structure

- `puzzle*.py` — Numbered puzzle generation engines (use the latest: `puzzle6.py`)
- `run*.py` — Numbered run scripts for testing/iterating (use the latest: `run15.py`)
- `megapuzzle1.py` — Batch generates many diverse puzzles
- `generate_puzzle_json.py` — Exports puzzle data to JSON for the web game
- `third_party/` — External word lists (TWL Scrabble dictionary)
- `web/` — Web game frontend (Bun/React/Tailwind). Main entrypoint for all frontend work.
- `puzzle-designer/` — Separate puzzle design/editing tool
