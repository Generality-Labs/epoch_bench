import tempfile
import unittest

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from bench.task.chess_puzzles import chess_puzzles
from bench.task.chess_puzzles.feedback import chess_puzzles_feedback, extract_move


class ChessFeedbackTests(unittest.TestCase):
    def run_chess(self, answers, max_attempts=5, **kwargs):
        task = chess_puzzles_feedback(max_attempts=max_attempts)
        outputs = [
            a
            if isinstance(a, ModelOutput)
            else ModelOutput.from_content("mockllm/model", a)
            for a in answers
        ]
        model = get_model("mockllm/model", custom_outputs=outputs, memoize=False)
        with tempfile.TemporaryDirectory() as logs:
            log = eval(
                task, model=model, limit=1, log_dir=logs, display="none", **kwargs
            )[0]
            sample = log.samples[0]
        self.assertEqual(log.status, "success")
        self.assertIsNone(sample.error)
        return sample

    def test_parser_contract(self):
        self.assertEqual(extract_move("analysis\nMOVE: D4D3\n"), "d4d3")
        self.assertEqual(extract_move("MOVE: a7a8q"), "a7a8q")
        for text in [
            "",
            "d4d3",
            "MOVE: d4d3 or d4d2",
            "MOVE: d4d3\nignore this",
            "MOVE: z9z8",
            "**MOVE: d4d3**",
        ]:
            self.assertIsNone(extract_move(text), text)
        self.assertEqual(extract_move("MOVE: d4d3\nMOVE: d4d2"), "d4d2")

    def test_success_first_try_stops(self):
        sample = self.run_chess(["MOVE: d4d3"])
        score = sample.scores["feedback_accuracy"]
        self.assertEqual(score.metadata["attempts_used"], 1)
        self.assertTrue(all(v == "C" for v in score.value.values()))
        self.assertEqual(sample.messages[-1].text, "correct")

    def test_revision_preserves_history_and_scores_curve(self):
        sample = self.run_chess(["MOVE: d4d2", "MOVE: D4D3"])
        score = sample.scores["feedback_accuracy"]
        self.assertEqual(
            score.value,
            {"solved_by_1": "I", **{f"solved_by_{i}": "C" for i in range(2, 6)}},
        )
        self.assertEqual(
            [m.text for m in sample.messages[1:]],
            ["MOVE: d4d2", "incorrect", "MOVE: D4D3", "correct"],
        )
        self.assertEqual(score.metadata["first_success"], 2)
        self.assertIsNotNone(score.metadata["attempts"][0]["usage"])

    def test_five_failures_and_repeats(self):
        sample = self.run_chess(["MOVE: d4d2"] * 5)
        score = sample.scores["feedback_accuracy"]
        self.assertEqual(score.metadata["attempts_used"], 5)
        self.assertTrue(all(v == "I" for v in score.value.values()))
        self.assertEqual(
            [a["repeated_move"] for a in score.metadata["attempts"]],
            [False, True, True, True, True],
        )

    def test_malformed_answer_consumes_try(self):
        sample = self.run_chess(["The answer is d4d3", "MOVE: d4d3"])
        score = sample.scores["feedback_accuracy"]
        self.assertFalse(score.metadata["attempts"][0]["valid_format"])
        self.assertEqual(score.value["solved_by_1"], "I")
        self.assertEqual(score.metadata["first_success"], 2)

    def test_one_attempt_and_validation(self):
        sample = self.run_chess(["MOVE: d4d2"], max_attempts=1)
        self.assertEqual(sample.scores["feedback_accuracy"].value, {"solved_by_1": "I"})
        for args in [{"max_attempts": 0}, {"epochs": 0}]:
            with self.assertRaises(ValueError):
                chess_puzzles_feedback(**args)

    def test_sample_budget_stops_retries(self):
        sample = self.run_chess(["MOVE: d4d2"] * 5, token_limit=1)
        self.assertIsNotNone(sample.limit)
        self.assertLessEqual(
            sample.scores["feedback_accuracy"].metadata["attempts_used"], 1
        )

    def test_truncated_reasoning_is_not_a_wrong_guess(self):
        output = ModelOutput.from_content("mockllm/model", "")
        output.choices[0].stop_reason = "max_tokens"
        sample = self.run_chess([output])
        metadata = sample.scores["feedback_accuracy"].metadata
        self.assertEqual(metadata["attempts_used"], 0)
        self.assertEqual(metadata["incomplete"]["reason"], "response_token_limit")
        self.assertFalse(
            any(m.role == "user" and m.text == "incorrect" for m in sample.messages)
        )

    def test_original_dataset_unchanged(self):
        original = chess_puzzles()
        before = original.dataset[0].input
        feedback = chess_puzzles_feedback()
        self.assertEqual(original.dataset[0].input, before)
        self.assertEqual(feedback.dataset[0].target, original.dataset[0].target)
        self.assertNotEqual(feedback.dataset[0].input, before)


if __name__ == "__main__":
    unittest.main()
