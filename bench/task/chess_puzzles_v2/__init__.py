"""Epoch AI's Chess Puzzles, version 2.1.0, from epoch-research/chess-puzzles-mirror.

Vendored from https://github.com/epoch-research/chess-puzzles-mirror at
62f4f8d3 (2026-11-05 refresh, task version 2.1.0). The mirror's own package is also
named `bench`, so it cannot be installed next to this one; its files are copied here
unchanged apart from imports:

- `scorer.py`, `puzzles.csv`, `CHANGELOG.md`: from `bench/task/chess_puzzles/`.
- `qa_agent.py`, `submission.py`, `forced_answer.py`: from `bench/solver/`, with
  `bench.solver.*` imports pointing here.

The task below is the mirror's `Chess Puzzles` task, registered under the mirror's own name
(`bench/Chess Puzzles`, the task name Epoch's 2.x logs record, so an audit can match
those logs to it), with one change: its two required
arguments get defaults, Epoch's published settings (1,000,000 output tokens, compaction
at 945,000 input tokens, i.e. 90% of a 1.05M-token window), so the task loads by name
with no arguments. Sample ids are 1-100 in CSV order (`auto_id=True`), as in Epoch's
2.x logs. The puzzles are the same 100 FEN/move pairs as `bench/task/chess_puzzles`
(the 1.x reconstruction). The Decision variant is not included.
"""

from pathlib import Path

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import csv_dataset
from inspect_ai.solver import prompt_template

from bench.task.chess_puzzles_v2.qa_agent import qa_agent
from bench.task.chess_puzzles_v2.scorer import chess_move

PUZZLES_FILENAME = str(Path(__file__).parent / "puzzles.csv")
VERSION = "2.1.0"
MIRROR_REVISION = "62f4f8d3"

QUESTION_TEMPLATE = (
    "Below is a FEN string representing the state of a chess game. Analyze the position and "
    "determine the best next move for the player whose turn it is.\n"
    "\n"
    "{prompt}"
)


SUBMIT_DESCRIPTION = (
    "Submit the best next move as a four-character string stating the starting and ending squares "
    "of the piece to be moved, e.g. b1c3. The conversation will end immediately."
)


FINAL_ANSWER_PROMPT = (
    "You have run out of output tokens. Do not reason any further: call `submit_answer` now with "
    "your best move."
)


@task(name="Chess Puzzles")
def chess_puzzles_v2(
    *,
    epochs: int = 1,
    max_output_tokens: int = 1_000_000,
    compaction_threshold_tokens: int = 945_000,
) -> Task:
    """Single-move chess puzzles, answered through qa_agent (Epoch's Chess Puzzles 2.1.0)."""
    dataset = csv_dataset(PUZZLES_FILENAME, auto_id=True)

    plan = [
        prompt_template(QUESTION_TEMPLATE),
        qa_agent(
            max_output_tokens=max_output_tokens,
            compaction_threshold_tokens=compaction_threshold_tokens,
            submit_description=SUBMIT_DESCRIPTION,
            final_answer_prompt=FINAL_ANSWER_PROMPT,
        ),
    ]

    return Task(
        dataset=dataset,
        plan=plan,
        scorer=chess_move(),
        epochs=Epochs(epochs, ["mean"]),
        metadata={
            "epochai-benchmark-version": VERSION,
            "wrapper": "bench/task/chess_puzzles_v2",
            "mirror_revision": MIRROR_REVISION,
        },
    )
