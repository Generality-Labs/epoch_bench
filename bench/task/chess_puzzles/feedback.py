"""Chess with binary feedback, separate from Epoch's published single-shot task."""

import re
from pathlib import Path

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import csv_dataset
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, TaskState, solver

# Exactly one move on the last non-empty line; no searching the reasoning for gold.
MOVE_LINE = re.compile(r"MOVE:\s*([a-h][1-8][a-h][1-8][qrbn]?)", re.IGNORECASE)


def extract_move(completion: str) -> str | None:
    lines = completion.strip().splitlines()
    match = MOVE_LINE.fullmatch(lines[-1].strip()) if lines else None
    return match.group(1).lower() if match else None


@solver
def binary_feedback(max_attempts: int):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        attempts: list[dict] = []
        state.metadata["chess_feedback_attempts"] = attempts
        for number in range(1, max_attempts + 1):
            state = await generate(state)
            completion = state.output.completion
            move = extract_move(completion)
            stop_reason = (
                state.output.choices[0].stop_reason if state.output.choices else None
            )
            if stop_reason == "max_tokens" and move is None:
                state.metadata["chess_feedback_incomplete"] = {
                    "reason": "response_token_limit",
                    "attempt": number,
                    "completion": completion,
                }
                break
            correct = move == state.target.text.strip().lower()
            attempts.append(
                {
                    "attempt": number,
                    "completion": completion,
                    "move": move,
                    "valid_format": move is not None,
                    "stop_reason": stop_reason,
                    "correct": correct,
                    "repeated_move": move is not None
                    and any(a["move"] == move for a in attempts),
                    "usage": state.output.usage.model_dump(exclude_none=True)
                    if state.output.usage
                    else None,
                }
            )
            # A sample-wide Inspect limit must not be cleared to get another try.
            state.messages.append(
                ChatMessageUser(content="correct" if correct else "incorrect")
            )
            if correct or state.completed or number == max_attempts:
                break
        state.completed = True
        return state

    return solve


@scorer(metrics={"*": [accuracy(), stderr()]})
def feedback_accuracy(max_attempts: int):
    async def score(state: TaskState, target: Target) -> Score:
        attempts = state.metadata.get("chess_feedback_attempts", [])
        first_success = next((a["attempt"] for a in attempts if a["correct"]), None)
        return Score(
            value={
                f"solved_by_{k}": CORRECT
                if first_success is not None and first_success <= k
                else INCORRECT
                for k in range(1, max_attempts + 1)
            },
            answer=attempts[-1]["move"] if attempts else None,
            metadata={
                "attempts": attempts,
                "attempts_used": len(attempts),
                "first_success": first_success,
                "incomplete": state.metadata.get("chess_feedback_incomplete"),
            },
        )

    return score


@task(name="Chess Puzzles Feedback")
def chess_puzzles_feedback(max_attempts: int = 5, epochs: int = 1) -> Task:
    """Up to max_attempts sequential guesses with only binary correctness feedback."""
    if max_attempts < 1 or epochs < 1:
        raise ValueError("max_attempts and epochs must be positive")
    dataset = csv_dataset(str(Path(__file__).parent / "puzzles.csv"))
    for sample in dataset:
        sample.input = (
            "Analyze this chess position and find the best next move for the side to move. "
            f"You have up to {max_attempts} attempts. After each submission you will receive "
            "only 'correct' or 'incorrect'; stop when correct. A malformed submission "
            "counts as an incorrect attempt. End your response with a plain final line "
            "MOVE: b1c3, using starting and ending squares (add q/r/b/n for promotion). "
            "Letter case is ignored. Submit exactly one move on that final line.\n\n"
            f"{sample.input}"
        )
    return Task(
        dataset=dataset,
        solver=binary_feedback(max_attempts),
        scorer=feedback_accuracy(max_attempts),
        epochs=Epochs(epochs, "mean"),
        metadata={
            "feedback_protocol_version": "1.1.0",
            "max_attempts": max_attempts,
            "answer_parser": "final_move_line",
        },
    )
