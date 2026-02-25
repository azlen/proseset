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

## Frontend Verification

When making frontend/UI changes in `web/`, always take a screenshot with Playwright to visually verify the result before reporting success. Use `bunx playwright screenshot http://localhost:5173 screenshot.png` (or similar) to capture the page and review it.
