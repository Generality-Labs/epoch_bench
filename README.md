# epoch-bench

Epoch AI's **Chess Puzzles** Inspect task, reconstructed so it can be audited with
`inspect_audit`.

- `bench/task/chess_puzzles/__init__.py` and `puzzle_generator.py` are Epoch's published
  source: https://gist.github.com/greg-burnham/fc1c6f9ed32989073f58321b656b7b05 (fetched
  2026-09-07). One deviation: the CSV path resolves next to the module rather than
  against the working directory, so the task loads as an installed package.
- `bench/model.py` stands in for Epoch's private module of that name. The extractor
  model is bound through the `grader` model role (default `google/gemini-2.0-flash-001`,
  which 25 of the 38 public logs used; 12 used `openai/gpt-5-mini-2025-08-07`).
- `puzzles.csv` is rebuilt from the public logs (id, FEN, target). All 38 logs carry the
  identical 100 samples.

Logs: https://epoch-benchmarks-production-public.s3.us-east-2.amazonaws.com/inspect_ai_logs/<id>.eval
(manifest in `audit/logs.csv`).

## Auditing it

`audit/hawk-chess.eval-set.yaml` runs the audit on Hawk with the general
`inspect_audit` auditor image; the chess engine is installed at sample start by the
`setup` script rather than baked into a bespoke image. Locally:

```sh
inspect eval inspect_audit/audit -T 'task=bench/Chess Puzzles' -T logs=<dir of the 38 logs> \
  -T 'setup=apt-get update -qq && apt-get install -y -qq stockfish && pip install -q python-chess' \
  --model openrouter/openai/gpt-5.6-luna --model-role grader=openrouter/google/gemini-2.0-flash-001 --limit 10
```
