import json
import math
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import json5
from inspect_ai import eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import SampleScore, Score

from bench.task.dtbench import capability_questions, dtbench, grade_completion, normalize_no_cot, question_prompt, source_data, valid_accuracy, valid_response_mean, valid_response_rate
from bench.task.dtbench import source_util

UPSTREAM = Path(__file__).resolve().parents[1] / "audit/dtbench/reference_source"


class DTBenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        util = types.ModuleType("util")
        util.ALPHABET = source_util.ALPHABET
        util.MULTIPLE_CHOICE_INSTRUCTION = source_util.MULTIPLE_CHOICE_INSTRUCTION
        util.flatten = lambda xs: [x for ys in xs for x in ys]
        fake_subject = types.ModuleType("subject")
        fake_subject.Subject = type("Subject", (), {})
        grading = types.ModuleType("grading")
        question = types.ModuleType("question")
        with patch.dict(sys.modules, {"util": util, "subject": fake_subject, "grading": grading}):
            exec((UPSTREAM / "grading.py").read_text(), grading.__dict__)
            exec((UPSTREAM / "question.py").read_text(), question.__dict__)
        cls.original_grading = grading
        cls.original_question = question
        cls.originals = {}
        archive = Path(__file__).resolve().parents[1] / "bench/task/dtbench/source_data.zip"
        with zipfile.ZipFile(archive) as z:
            z.setpassword(b"onebox")
            for name in z.namelist():
                if name.startswith("setting"):
                    for q in question.get_questions_from_nested_dict("", json5.loads(z.read(name)), []):
                        cls.originals[q.qid] = q

    def test_all_source_questions_and_prompts(self):
        self.assertEqual(len(capability_questions()), 407)
        for record in source_data()["questions"]:
            q = self.originals[record["qid"]]
            self.assertEqual(record["question_text"], q.question_text)
            self.assertEqual(record["tags"], q.tags)
            self.assertEqual(record["permissible_answers"], q.permissible_answers)
            if not q.attitude_q:
                self.assertEqual(record["correct_answer"], q.correct_answers[0])
                self.assertEqual(question_prompt(q.question_text, q.permissible_answers), q.generate_question_prompt())

    def test_shuffle_and_original_grader_parity(self):
        for seed in [0, 42, 13579]:
            for sample in dtbench(seed=seed).dataset:
                q = self.originals[sample.id]
                self.assertEqual(sample.choices[source_util.ALPHABET.index(sample.target)], q.get_correct_answer_strings(include_letters=False)[0])
                mapped = self.original_question.Question(q.question_text, [], sample.choices.index(q.get_correct_answer_strings(include_letters=False)[0]), sample.choices)
                for answer in ["", "FINAL ANSWER: Z.", "FINAL ANSWER: A. FINAL ANSWER: B.", "FINAL ANSWER: a.",
                               "FINAL ANSWER: " + sample.target + ".", "FINAL ANSWER: A or B", "FINAL ANSWER: A)", "answer B"]:
                    expected = self.original_grading.evaluate_multi_choice_answer(answer, mapped, print_outp=False, print_inp=False)
                    expected = None if expected == "Invalid answer" else expected
                    self.assertEqual(grade_completion(answer, sample.choices, [sample.target]), expected)

    def test_mock_run_and_invalid_log_roundtrip(self):
        task = dtbench(epochs=2)
        target = task.dataset[0].target
        outputs = [ModelOutput.from_content("mockllm/model", "FINAL ANSWER: " + target + "."),
                   ModelOutput.from_content("mockllm/model", "not parseable")]
        model = get_model("mockllm/model", custom_outputs=outputs, memoize=False)
        with tempfile.TemporaryDirectory() as directory:
            log = eval(task, model=model, limit=1, log_dir=directory, display="none", max_samples=1)[0]
            self.assertEqual(log.status, "success")
            saved = read_eval_log(log.location)
            values = [s.scores["dtbench_score"].value for s in saved.samples]
            self.assertEqual(values[0], 1.0)
            self.assertTrue(math.isnan(values[1]))
            metrics = saved.results.scores[0].metrics
            self.assertAlmostEqual(metrics["valid_accuracy"].value, 1.0)
            self.assertAlmostEqual(metrics["valid_response_rate"].value, 0.5)
            self.assertEqual(saved.samples[0].messages[0].content, task.dataset[0].input)

    def test_no_cot_source_postprocessing(self):
        util = types.ModuleType("util")
        util.ALPHABET = source_util.ALPHABET
        util.MULTIPLE_CHOICE_INSTRUCTION = source_util.MULTIPLE_CHOICE_INSTRUCTION
        provider = types.ModuleType("get_llm_response")
        provider.LIST_OF_MODELS = ["test"]
        provider.CONTEXT_LENGTHS_IN_TOKENS = {"test": 10000}
        subject = types.ModuleType("source_subject")
        for response in ["A) because", "\n FINAL ANSWER: B.", "analysis\nFINAL ANSWER: B."]:
            provider.get_llm_response = lambda prompt: None
            provider.get_llm_response = lambda prompt, model: response
            with patch.dict(sys.modules, {"util": util, "get_llm_response": provider}):
                exec((UPSTREAM / "subject.py").read_text(), subject.__dict__)
                expected = subject.Subject("test", allow_cot=False).get_answer_subject_inner("question")
            self.assertEqual(normalize_no_cot(response), expected)
        task = dtbench(allow_cot=False)
        model = get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", task.dataset[0].target + ")")], memoize=False)
        with tempfile.TemporaryDirectory() as directory:
            log = eval(task, model=model, limit=1, log_dir=directory, display="none")[0]
            self.assertEqual(log.status, "success")
            self.assertEqual(log.samples[0].scores["dtbench_score"].value, 1.0)
            self.assertIn("raw_completion_before_source_postprocessing", log.samples[0].metadata)

    def test_all_no_cot_prompts_match_source_subject(self):
        util = types.ModuleType("util")
        util.ALPHABET = source_util.ALPHABET
        util.MULTIPLE_CHOICE_INSTRUCTION = source_util.MULTIPLE_CHOICE_INSTRUCTION
        provider = types.ModuleType("get_llm_response")
        provider.LIST_OF_MODELS = ["test"]
        provider.CONTEXT_LENGTHS_IN_TOKENS = {"test": 10000}
        captured = []
        provider.get_llm_response = lambda prompt, model: (captured.append(prompt) or "FINAL ANSWER: A.")
        subject = types.ModuleType("source_subject")
        with patch.dict(sys.modules, {"util": util, "get_llm_response": provider}):
            exec((UPSTREAM / "subject.py").read_text(), subject.__dict__)
        original = subject.Subject("test", allow_cot=False)
        for sample in dtbench(allow_cot=False).dataset:
            original.get_answer_subject_inner(question_prompt(sample.metadata["question_text"], sample.choices))
            self.assertEqual(captured[-1], sample.input)

    def test_source_aggregation_weights_items_equally(self):
        reducer = valid_response_mean()
        first = reducer([Score(value=1), Score(value=0), Score(value=float("nan"))])
        second = reducer([Score(value=1)] * 10)
        scores = [SampleScore(sample_id="a", score=first), SampleScore(sample_id="b", score=second)]
        self.assertEqual(valid_accuracy()(scores), 0.75)
        self.assertAlmostEqual(valid_response_rate()(scores), 12/13)


if __name__ == "__main__":
    unittest.main()
