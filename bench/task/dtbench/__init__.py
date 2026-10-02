"""DTBench capability task, retaining the published prompt and strict grader."""
import hashlib
import json
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Metric, SampleScore, Score, ScoreReducer, Target, metric, scorer, score_reducer
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver

from .source_grading import evaluate_multi_choice_answer
from .source_util import ALPHABET, MULTIPLE_CHOICE_INSTRUCTION

NO_COT_INSTRUCTION = 'Please select exactly one of the following answers. Please respond with the string "FINAL ANSWER: " followed by the letter corresponding to your answer and then a period, for example, "FINAL ANSWER: B." Your response should not have any other content. In particular, do not think step by step!'


@lru_cache
def source_data():
    return json.loads(Path(__file__).with_name("questions.json").read_text())


def capability_questions():
    return [q for q in source_data()["questions"] if not q["attitude_q"] and q["double_checked"]]


def question_prompt(text, options, allow_cot=True):
    instruction = MULTIPLE_CHOICE_INSTRUCTION if allow_cot else NO_COT_INSTRUCTION
    return text + "\n\n" + instruction + "\n\n" + "".join(f"{ALPHABET[i]}) {o}\n" for i, o in enumerate(options))


def normalize_no_cot(response):
    """Same postprocessing as Subject.get_answer_subject_inner, before grading."""
    compact = response.replace("\n", "").replace(" ", "")
    if compact[:2] in [letter + ")" for letter in ALPHABET]:
        return "FINAL ANSWER: " + compact[0] + ".\n\n" + response + '\n\nNote that "FINAL ANSWER" was added via code. Otherwise, the answer would have been invalid.'
    if not compact.startswith("FINALANSWER:"):
        return "no CoT response that didn't start with answer and is therefore invalid:\n\n" + response.replace("FINAL ANSWER: ", "F_I_N_A_L A_N_S_W_E_R:")
    return response


@solver
def no_cot_postprocess() -> Solver:
    async def solve(state: TaskState, generate: Generate):
        raw = state.output.completion
        state.metadata["raw_completion_before_source_postprocessing"] = raw
        result = normalize_no_cot(raw)
        state.output.message.content = result
        if "completion" in type(state.output).model_fields:
            state.output.completion = result
        # Keep the log's final assistant message consistent with the scored output.
        for msg in reversed(state.messages):
            if msg.role == "assistant":
                msg.content = result
                break
        return state
    return solve


@dataclass
class GradingQuestion:
    """The subset of the source Question API used by its unchanged grader."""
    permissible_answers: list[str]
    correct_letters: list[str]
    attitude_q: bool = False

    def get_answer_letters(self, theory=None):
        return self.correct_letters

    def get_correct_answer_strings(self, theory=None):
        return [letter + ") " + self.permissible_answers[ALPHABET.index(letter)] for letter in self.correct_letters]


def grade_completion(completion, options, correct_letters):
    result = evaluate_multi_choice_answer(completion, GradingQuestion(options, correct_letters), print_outp=False, print_inp=False)
    return None if result == "Invalid answer" else bool(result)


@metric
def valid_accuracy() -> Metric:
    def metric(scores: list[SampleScore]):
        values = [float(s.score.value) for s in scores if isinstance(s.score.value, (int, float)) and math.isfinite(s.score.value)]
        return sum(values) / len(values) if values else float("nan")
    return metric


@metric
def valid_response_rate() -> Metric:
    def metric(scores: list[SampleScore]):
        valid = sum((s.score.metadata or {}).get("valid_attempts", int(isinstance(s.score.value, (int, float)) and math.isfinite(s.score.value))) for s in scores)
        total = sum((s.score.metadata or {}).get("attempts", 1) for s in scores)
        return valid / total if total else float("nan")
    return metric


@score_reducer
def valid_response_mean() -> ScoreReducer:
    """Source aggregation: mean valid trials within each item, then equal items."""
    def reduce(scores: list[Score]):
        values = [float(s.value) for s in scores if isinstance(s.value, (int, float)) and math.isfinite(s.value)]
        return Score(value=sum(values) / len(values) if values else float("nan"),
                     metadata={"valid_attempts": len(values), "attempts": len(scores)})
    return reduce


@scorer(metrics=[valid_accuracy(), valid_response_rate()])
def dtbench_score():
    async def score(state: TaskState, target: Target):
        options = [c.value for c in state.choices] if state.choices else state.metadata["permissible_answers"]
        result = grade_completion(state.output.completion, options, list(target.target))
        return Score(value=float(result) if result is not None else float("nan"),
                     reason="invalid_response_format" if result is None else None,
                     explanation="Original DTBench strict FINAL ANSWER grader; invalid responses are excluded from valid-response accuracy.",
                     metadata={"valid_attempts": int(result is not None), "attempts": 1})
    return score


@task(name="DTBench")
def dtbench(seed: int = 42, allow_cot: bool = True, epochs: int = 1) -> Task:
    """407 validated capability questions; attitude items are not accuracy tasks.

    The source shuffles choices once per run and retains that order across trials.
    This port makes that shuffle reproducible by run seed and stable question ID.
    Change seed for another permutation. No Inspect multiple_choice prompt is added.
    """
    samples = []
    data = source_data()
    for q in capability_questions():
        options = list(q["permissible_answers"])
        correct_text = options[q["correct_answer"]]
        rng_seed = int.from_bytes(hashlib.sha256(f"{seed}:{q['qid']}".encode()).digest(), "big")
        random.Random(rng_seed).shuffle(options)
        letter = ALPHABET[options.index(correct_text)]
        samples.append(Sample(
            id=q["qid"], input=question_prompt(q["question_text"], options, allow_cot),
            choices=options, target=letter,
            metadata={"question_text": q["question_text"], "permissible_answers": options,
                      "correct_answer_text": correct_text, "tags": q["tags"],
                      "source_file": q["source_file"], "source_commit": data["source_commit"],
                      "source_sha256": data["source_sha256"], "allow_cot": allow_cot},
        ))
    solvers = [generate()]
    if not allow_cot:
        solvers.append(no_cot_postprocess())
    return Task(dataset=MemoryDataset(samples, name="DTBench-capabilities"),
                solver=solvers, scorer=dtbench_score(),
                epochs=Epochs(epochs, [valid_response_mean()]),
                metadata={"source_url": "https://github.com/casparoe/newcomblike_questions_dataset",
                          "source_commit": data["source_commit"], "source_sha256": data["source_sha256"],
                          "population": "407 double-checked capability questions",
                          "invalid_response_policy": "exclude from valid-response accuracy; report validity separately"})
