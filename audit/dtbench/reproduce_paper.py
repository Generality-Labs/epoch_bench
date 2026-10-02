"""Reproduce the paper's capability mean from converted logs, without pickles."""
import argparse
import ast
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

from bench.task.dtbench import capability_questions


def literal_assignment(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing literal {name} in {path}")


def reproduce(logs, source, output):
    code = source / "benchmark/code"
    subjects = literal_assignment(code / "subject_lists_for_analysis.py", "raw_subjects_for_comparison")
    labels = literal_assignment(code / "model_info.py", "relabelling")
    manifest = json.loads((logs / "manifest.json").read_text())
    files = {row["subject"]: logs / row["file"] for row in manifest["logs"]}
    missing = sorted(set(subjects) - files.keys())
    if missing:
        raise ValueError(f"Missing cohort logs: {missing}")
    questions = {q["qid"]: q for q in capability_questions() if "trivia" not in q["tags"]}
    means = {}
    for subject in subjects:
        attempts = defaultdict(list)
        for sample in read_eval_log(files[subject]).samples or []:
            value = sample.scores["dtbench_score"].value
            if sample.id in questions and isinstance(value, (int, float)) and math.isfinite(value):
                attempts[sample.id].append(float(value))
        means[subject] = {qid: statistics.mean(values) for qid, values in attempts.items()}
    shared = set(questions).intersection(*(set(values) for values in means.values()))
    if not shared:
        raise ValueError("Cohort has no common valid items")
    published = {r["subject"]: float(r["cap_score"]) for r in csv.DictReader(
        (code / "analysis_csvs/cap_versus_att.csv").open()
    )}
    rows = []
    for subject, values in means.items():
        label = labels.get(subject, subject)
        actual = statistics.mean(values[qid] for qid in shared)
        target = published.get(label)
        rows.append({"subject": subject, "label": label, "score": actual,
                     "published_paper_score": target,
                     "difference": actual - target if target is not None else None})
    result = {"non_trivia_capability_items": len(questions), "cohort_models": len(subjects),
              "shared_valid_items": len(shared), "shared_qids": sorted(shared),
              "excluded_qids": sorted(set(questions) - shared), "models": rows,
              "random_score": statistics.mean(1 / len(questions[qid]["permissible_answers"]) for qid in shared),
              "scope": "Original paper ordinary-model cohort; not the current CRI leaderboard population."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    compared = [abs(r["difference"]) for r in rows if r["difference"] is not None]
    print({"cohort_models": len(subjects), "shared_valid_items": len(shared),
           "matched_published_rows": len(compared), "maximum_difference": max(compared) if compared else None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reproduce(args.logs, args.source, args.output)
