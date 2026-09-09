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

## Chess with binary feedback

`bench/Chess Puzzles Feedback` is a separate experimental task using the same
100 positions and targets. It gives a model up to five sequential attempts in
one conversation, appending only `correct` or `incorrect` after each answer and
stopping immediately on success. Earlier responses stay in the conversation.
The original `bench/Chess Puzzles` task is unchanged.

The model must end its response with a plain `MOVE: b1c3` line. Parsing uses the
last non-empty line, normalises letter case, and accepts a promotion suffix
(`q`, `r`, `b`, or `n`). No LLM judge or chess engine is used: the parsed move is
compared with the stored target. Missing/malformed submissions consume a try
and receive `incorrect`, just like wrong moves. The prompt states this contract.
If generation hits its response-token cap without a parseable move, the episode
ends with `incomplete.reason=response_token_limit`; no incorrect feedback is sent
and no submitted guess is counted. These episodes must be reported separately
when interpreting the cumulative-success scores (which still count them unsolved).
This deliberately differs from the original task's permissive LLM extraction.

For example, after installing this package:

```sh
inspect eval 'bench/Chess Puzzles Feedback' --model mockllm/model --limit 1 -T max_attempts=5
```

Replace the mock model with a provider/model for an actual experiment. Set
`--max-tokens` to bound each generation and `--token-limit` to bound each full
puzzle conversation; the latter includes input/history token usage. Inspect
limits and provider errors remain visible in the logs. API retries are distinct
from the five chess guesses.

Scores `solved_by_1` through `solved_by_5` report cumulative success. Per-attempt
metadata records the exact completion, parsed move, format validity, correctness,
repeated guesses, and provider-reported token usage. `first_success` and
`attempts_used` are also saved. If a sample hits a limit, later curve points do
not represent five fully executed attempts; inspect coverage/limit counts too.
Epochs repeat the entire feedback episode independently and average its scores.

On Hawk, use task name `Chess Puzzles Feedback` under package name `bench`, with
`args: {max_attempts: 5}`, and list the desired models in the normal model grid.
No sandbox or grader model role is required. Publish and pin the package revision
containing this task before submitting a remote job.

Local tests (Inspect's mock provider; no API key required):

```sh
python -m unittest discover -s tests -v
```
