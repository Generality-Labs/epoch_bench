"""HLE as inspect_evals runs it, on the original 2,500-question set.

`inspect_evals/hle` defaults to `only_hle_verified_gold=True`, which keeps only the
668 items HLE-Verified left unmodified. An audit of the original benchmark (Epoch's
review sampled the full static set) needs every item, so this task is
`inspect_evals/hle` with that one default flipped. Everything else is delegated to
inspect_evals at the pinned revision in pyproject.toml: the dataset (cais/hle, gated:
loading needs HF_TOKEN), its revision, the sample ids (`hle_` + the first 8 hex digits
of md5(uid)), images, the system prompt, the solver, and the default two-judge regex
scorer (model roles `grader` and `grader_2`), so attempts generated with it are
identical to `inspect_evals/hle(only_hle_verified_gold=False)`.
"""

from __future__ import annotations

from typing import Any, Literal

from inspect_ai import Task, task


@task
def hle(
    answer_type: Literal["exactMatch", "multipleChoice"] | None = None,
    category: str | list[str] | None = None,
    subject: str | list[str] | None = None,
    include_multi_modal: bool = True,
    only_hle_verified_gold: bool = False,
    rolling: bool = False,
    system_prompt: str = "system_mc",
    scorer: dict[str, Any] | None = None,
) -> Task:
    """inspect_evals/hle over the full static set by default.

    Args:
        answer_type: "exactMatch" or "multipleChoice"; None keeps both.
        category: Category or categories to keep; None keeps all.
        subject: Raw subject or subjects to keep; None keeps all.
        include_multi_modal: Keep the items that carry an image (about 14%).
        only_hle_verified_gold: Restrict to HLE-Verified's gold subset. False here
            (inspect_evals defaults to True), so the population is the original
            2,500 questions.
        rolling: Use cais/hle-rolling instead of the static set.
        system_prompt: Title of the system prompt in inspect_evals' prompts.py.
        scorer: Judge spec as inspect_evals/hle takes it; None keeps its default
            (regex judge, graders `grader` and `grader_2`).
    """
    from inspect_evals.hle import hle as inspect_evals_hle

    kwargs: dict[str, Any] = {
        "answer_type": answer_type,
        "category": category,
        "subject": subject,
        "include_multi_modal": include_multi_modal,
        "only_hle_verified_gold": only_hle_verified_gold,
        "rolling": rolling,
        "system_prompt": system_prompt,
    }
    if scorer is not None:
        kwargs["scorer"] = scorer
    inner = inspect_evals_hle(**kwargs)
    inner.metadata = {**(inner.metadata or {}), "wrapper": "bench/hle"}
    return inner
