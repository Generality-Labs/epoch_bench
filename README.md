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


## SimpleQA Verified (branch `simpleqa-verified`)

`bench/simpleqa_verified_anti_abstention`: SimpleQA Verified as Epoch runs it since 2026-08-27.

- `bench/task/simpleqa_verified/grader_templates.py` and `simpleqa.py` down to `return score` are Epoch's published source,
  unchanged: https://gist.github.com/tadamcz/27b929c0923a4c539b0655536cd220ea (one revision, 2025-12-09, benchmark version 1.0.1;
  fetched 2026-11-24).
- The gist predates the anti-abstention instruction. Everything after `return score` is reconstructed: the instruction text is quoted
  on Epoch's SimpleQA Verified page and is identical on all 1,000 samples of their public log `ViC6pxQZUVS3E9M9iEubF7.eval`
  (Claude Opus 4.8, task `bench/simpleqa_verified_anti_abstention`, benchmark version 1.2.0).
- The grader is the model bound to the `grader` role, resolved when the scorer calls it. Epoch's logs of this task show
  `gpt-5.6-terra` at reasoning effort medium.
- `audit/simpleqa_verified/compare_with_epoch_log.py` (no model calls) checks the reconstruction against that log: 1,000 of 1,000 ids,
  user messages and targets identical; grader template identical.
- `audit/simpleqa_verified/hawk-fable52-*.eval-set.yaml`: a seeded random 200 of the 1,000 on Claude Fable 5.2 at reasoning effort max.
