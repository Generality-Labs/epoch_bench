from typing import Any, Literal, TypeGuard

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, csv_dataset, hf_dataset
from inspect_ai.model import Model
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    scorer,
    CORRECT,
    INCORRECT,
    NOANSWER,
    accuracy,
)
from inspect_ai.scorer import mean, stderr
from inspect_ai.solver import TaskState, generate
from inspect_ai.tool import ToolDef, ToolFunction

from .grader_templates import (
    SIMPLEQA_GRADER_TEMPLATE,
    SIMPLEQA_VERIFIED_GRADER_TEMPLATE,
)
from ...model import default_grader_model


@task
def simpleqa_verified() -> Task:
    return Task(
        dataset=hf_dataset(
            path="codelion/SimpleQA-Verified",
            split="train",
            sample_fields=FieldSpec(
                input="problem",
                target="answer",
                metadata=[
                    "original_index",
                    "topic",
                    "answer_type",
                    "mutli_step",
                    "requires_reasoning",
                    "urls",
                ],
            ),
        ),
        solver=[generate()],
        scorer=simpleqa_scorer(
            grader_model=default_grader_model(),
            grader_template=SIMPLEQA_VERIFIED_GRADER_TEMPLATE,
        ),
        metadata={"epochai-benchmark-version": "1.0.1", "inspect-log-public": True},
    )


def is_float_dict_list(v: list[Any]) -> TypeGuard[list[dict[str, float]]]:
    return all(isinstance(d, dict) and all(isinstance(v, float) for v in d.values()) for d in v)


@scorer(metrics=[accuracy(), stderr()])
def simpleqa_scorer(
    grader_model: Model,
    grader_template: str = SIMPLEQA_GRADER_TEMPLATE,
) -> Scorer:
    async def score(
        state: TaskState,
        target: Target,
    ) -> Score:
        grader_prompt = grader_template.format(
            question=state.input_text,
            target=target.text,
            predicted_answer=state.output.completion,
        )

        def submit(grade: Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]) -> None:
            """Submit the grade for the predicted answer.

            Args:
                grade: The grade - CORRECT, INCORRECT, or NOT_ATTEMPTED
            """
            pass

        grade_tool = ToolDef(
            tool=submit,
            name="submit",
        )

        result = await grader_model.generate(
            grader_prompt,
            tools=[grade_tool],
            tool_choice=ToolFunction(name="submit"),
        )

        if not result.message.tool_calls or "grade" not in result.message.tool_calls[0].arguments:
            raise RuntimeError("Grader model did not correctly submit a grade")

        grade = result.message.tool_calls[0].arguments["grade"]

        match grade:
            case "CORRECT":
                value = CORRECT
            case "INCORRECT":
                value = INCORRECT
            case "NOT_ATTEMPTED":
                # default value_to_float gives 0 for NOANSWER, which is what we want
                # (just score based on percentage correct)
                value = NOANSWER
            case _:
                raise RuntimeError(f"Unexpected grade submitted by grader model: {grade}")

        return Score(
            value=value,
            answer=state.output.completion,
            explanation=f"The grader model responded with: {grade}",
        )

    return score



# ---- Not in Epoch's gist: everything below is reconstructed (see README).
# Epoch began appending this instruction on 2026-08-27 (their SimpleQA Verified page, Changelog) and renamed the task
# bench/simpleqa_verified_anti_abstention. The text is quoted on that page and is identical on all 1,000 samples of their
# public log ViC6pxQZUVS3E9M9iEubF7.eval, where the user message is the question, a blank line, then this.
ANTI_ABSTENTION_SUFFIX = (
    "This is a guessing game, not a test. There's no penalty for being wrong, and an uncertain guess is far more "
    'useful to me than "I don\'t know". Give your single best guess.'
)


class _GraderRole:
    """The model bound to the `grader` role, looked up when the scorer calls it. Epoch's default_grader_model() is private; the stand-in in
    bench/model.py resolves at task-construction time, before roles are bound, and falls back to a Gemini model. Epoch's logs of this task
    show the grader as gpt-5.6-terra at reasoning effort medium; name it as the `grader` role in the run config."""

    async def generate(self, *args, **kwargs):
        from inspect_ai.model import get_model

        return await get_model(role="grader").generate(*args, **kwargs)


def anti_abstention_dataset():
    dataset = hf_dataset(
        path="codelion/SimpleQA-Verified",
        split="train",
        sample_fields=FieldSpec(
            input="problem",
            target="answer",
            metadata=["original_index", "topic", "answer_type", "mutli_step", "requires_reasoning", "urls"],
        ),
    )
    for sample in dataset:
        sample.input = f"{sample.input}\n\n{ANTI_ABSTENTION_SUFFIX}"
    return dataset


@task
def simpleqa_verified_anti_abstention() -> Task:
    return Task(
        dataset=anti_abstention_dataset(),
        solver=[generate()],
        scorer=simpleqa_scorer(
            grader_model=_GraderRole(),
            grader_template=SIMPLEQA_VERIFIED_GRADER_TEMPLATE,
        ),
        metadata={"reconstructed-from": "gist 27b929c0 (v1.0.1) + suffix from Epoch's page and log ViC6pxQZUVS3E9M9iEubF7 (v1.2.0)"},
    )
