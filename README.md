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


## BFCL (`bench/bfcl`)

`bench/task/bfcl/__init__.py` builds the Berkeley Function-Calling Leaderboard dataset with
inspect_evals' own loaders, converters, solver and scorer (`inspect_evals/bfcl`, pinned in
`pyproject.toml` under the `bfcl` extra), minus two gates that only matter for running a
model: the memory categories' snapshot check and the web-search categories' SerpAPI key
check. The items of every scored category can therefore be loaded and audited; V1-V3
attempts generated with it are identical to inspect_evals'. Install with `pip install
"epoch-bench[bfcl] @ git+<this repo>@<sha>"`. Samples carry `metadata.stratum`, the
leaderboard's weighting stratum (non_live_ast, live_ast, irrelevance, multi_turn, memory,
web_search).


## HLE (`bench/hle`)

`bench/task/hle/__init__.py` is `inspect_evals/hle` (pinned in `pyproject.toml` under the
`hle` extra, inspect_evals@4e0615e, task version 8-E) with one default flipped:
`only_hle_verified_gold=False`, so the population is the original 2,500-question static
set (cais/hle @ 5a81a4c) rather than HLE-Verified's 668-item gold subset. Dataset,
images, sample ids (`hle_` + first 8 hex digits of md5(uid)), prompt, solver and the
default two-judge regex scorer (model roles `grader`, `grader_2`) are inspect_evals'.
cais/hle is gated: loading needs `HF_TOKEN`. Install with `pip install
"epoch-bench[hle] @ git+<this repo>@<sha>"`.


## Chess Puzzles 2.1.0 (`bench/chess_puzzles_v2`)

`bench/task/chess_puzzles_v2/` is Epoch's current Chess Puzzles task (version 2.1.0)
vendored from https://github.com/epoch-research/chess-puzzles-mirror @ 62f4f8d3: the
`qa_agent` scaffold with a `submit_answer` tool and output-token budget, and the
deterministic scorer (exact UCI match, also accepting SAN and UCI variants read against
the position with python-chess). The mirror's package is also named `bench`, so its files
are copied here with only their imports changed; the task's two required arguments
default to Epoch's published settings (`max_output_tokens=1000000`,
`compaction_threshold_tokens=945000`). Sample ids are 1-100; the puzzles are the same 100
as the 1.x reconstruction above. python-chess is a base dependency. Install with `pip install "epoch-bench @
git+<this repo>@<sha>"`.
