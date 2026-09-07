"""Stand-in for Epoch's private `bench.model`.

The published task source (see README) imports `default_grader_model` from here.
Epoch's logs show the extractor was `google/gemini-2.0-flash-001` in 25 of 38
logs and `openai/gpt-5-mini-2025-08-07` in 12, so the grader drifted across the
leaderboard. Bind it through the `grader` model role so a run can name either.
"""

from inspect_ai.model import Model, get_model


def default_grader_model() -> Model:
    return get_model(role="grader", default="google/gemini-2.0-flash-001")
