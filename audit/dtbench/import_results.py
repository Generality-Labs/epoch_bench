"""Convert the published JSON result archive to provenance-marked Inspect logs.

Default: all current capability records for ordinary and no-CoT subjects.
--include-interventions also imports background/persona/best-of-N variants, whose
full elicitation is not recorded. Nothing is claimed to be a native Inspect run.
"""
import argparse
import collections
import hashlib
import json
import math
import subprocess
import tempfile
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalMetric, EvalResults, EvalSample, EvalScore, EvalSpec, write_eval_log
from inspect_ai.log._log import EvalMetricDefinition, EvalScorer
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelOutput
from inspect_ai.scorer import Score

from bench.task.dtbench import capability_questions, dtbench_score, grade_completion, question_prompt, source_data, valid_response_mean
from bench.task.dtbench.source_util import ALPHABET


def archive_records(path):
    """Use native unzip decryption, streaming adjacent JSON objects without extraction."""
    with zipfile.ZipFile(path) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
    process = subprocess.Popen(["unzip", "-P", "onebox", "-p", str(path)], stdout=subprocess.PIPE, text=True, encoding="utf-8")
    decoder = json.JSONDecoder()
    buffer = ""
    count = 0
    try:
        while True:
            chunk = process.stdout.read(1024 * 1024)
            buffer += chunk
            pos = 0
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos == len(buffer):
                    break
                try:
                    record, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    break
                yield names[count], record
                count += 1
                pos = end
            buffer = buffer[pos:]
            if not chunk:
                if buffer.strip():
                    raise ValueError("Incomplete JSON at end of result archive")
                break
        if process.wait() != 0 or count != len(names):
            raise RuntimeError(f"Result archive incomplete: {count}/{len(names)} records")
    finally:
        if process.poll() is None:
            process.kill()


def record_matches(question, record):
    return (record.get("question_text") == question["question_text"]
            and set(record.get("permissible_answers_list", [])) == set(question["permissible_answers"])
            and set(record.get("correct_answer_strings", [])) == {question["permissible_answers"][question["correct_answer"]]})


def convert(archive, output, include_interventions=False):
    output.mkdir(parents=True, exist_ok=True)
    questions = {q["qid"]: q for q in capability_questions()}
    data = source_data()
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    totals = collections.Counter()
    subject_counts = collections.Counter()
    mismatches = []
    groups = {}
    with tempfile.TemporaryDirectory(prefix="dtbench-import-") as temp:
        temp = Path(temp)
        for index, (filename, record) in enumerate(archive_records(archive), 1):
            totals["archive_records"] += 1
            subject = record.get("subject", "")
            qid = record.get("qid")
            if qid not in questions:
                totals["outside_current_capability_population"] += 1
                continue
            if not record_matches(questions[qid], record):
                totals["outdated_or_mismatching_item"] += 1
                continue
            subject_counts[subject] += 1
            if not include_interventions and ("+" in subject or subject.endswith("_de")):
                totals["intervention_not_selected"] += 1
                continue
            options = record["permissible_answers_list"]
            letters = [ALPHABET[options.index(a)] for a in record["correct_answer_strings"]]
            result = grade_completion(record["answer_text"], options, letters)
            recorded = record.get("answer_correct?") if record.get("valid_answer") else None
            if result != recorded or bool(record.get("valid_answer")) != (result is not None):
                totals["grader_disagreement"] += 1
                mismatches.append({"file": filename, "subject": subject, "qid": qid,
                                   "recorded_valid": record.get("valid_answer"), "recorded_correct": recorded,
                                   "regraded_correct": result})
            # Retain original grades; the audit's concordance check must discover
            # disagreements, rather than receiving quietly corrected grades.
            slug = hashlib.sha256(subject.encode()).hexdigest()[:16]
            if subject not in groups:
                groups[subject] = temp / f"{slug}.jsonl"
            with groups[subject].open("a") as f:
                f.write(json.dumps({"file": filename, "record": record}) + "\n")
            totals["selected_records"] += 1
            if index % 20000 == 0:
                print(f"Read {index:,} records; selected {totals['selected_records']:,}", flush=True)
        manifest = {"source_commit": data["source_commit"], "dataset_sha256": data["source_sha256"],
                    "archive_sha256": archive_hash, "counts": dict(totals),
                    "subject_counts": dict(subject_counts), "grader_disagreements": mismatches, "logs": []}
        for subject, spool in sorted(groups.items()):
            rows = [json.loads(line) for line in spool.read_text().splitlines()]
            rows.sort(key=lambda r: (r["record"].get("date_and_time", ""), r["file"]))
            epochs = collections.Counter()
            samples = []
            partial = "+" in subject or subject.endswith("_de")
            allow_cot = "_nocot" not in subject
            for row in rows:
                r = row["record"]
                qid = r["qid"]
                epochs[qid] += 1
                options = r["permissible_answers_list"]
                targets = [ALPHABET[options.index(a)] for a in r["correct_answer_strings"]]
                # Provider transcripts were not saved by the source. The question
                # and choice order are real; this user prompt is source-derived.
                prompt = question_prompt(r["question_text"], options, allow_cot)
                score = Score(value=float(r["answer_correct?"]) if r.get("valid_answer") else float("nan"),
                              reason=None if r.get("valid_answer") else "invalid_response_format",
                              metadata={"valid_attempts": int(bool(r.get("valid_answer"))), "attempts": 1,
                                        "source_file": row["file"], "source_grade": True})
                metadata = {"archive_file": row["file"], "archive_sha256": archive_hash,
                            "source_commit": data["source_commit"], "source_subject": subject,
                            "source_timestamp": r.get("date_and_time"), "tags": questions[qid]["tags"],
                            "question_text": r["question_text"], "permissible_answers": options,
                            "correct_answer_text": r["correct_answer_strings"][0],
                            "source_valid_answer": r.get("valid_answer"),
                            "prompt_provenance": "source reconstruction; provider transcript not recorded",
                            "prompt_incomplete": partial,
                            "completion_provenance": "verbatim published answer_text; may include source no-CoT postprocessing",
                            "epochs_provenance": "ordinal among archived attempts for this subject and qid, not an original epoch label"}
                samples.append(EvalSample(id=qid, epoch=epochs[qid], input=prompt, choices=options, target=targets,
                    messages=[ChatMessageUser(content=prompt), ChatMessageAssistant(content=r["answer_text"], model=subject)],
                    output=ModelOutput.from_content(subject, r["answer_text"], stop_reason="unknown"),
                    scores={"dtbench_score": score}, metadata=metadata,
                    uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, archive_hash + ":" + row["file"]))))
            max_epoch = max(epochs.values())
            item_means = []
            valid_count = 0
            for qid in epochs:
                values = [s.scores["dtbench_score"].value for s in samples if s.id == qid and math.isfinite(s.scores["dtbench_score"].value)]
                valid_count += len(values)
                if values:
                    item_means.append(sum(values)/len(values))
            accuracy = sum(item_means)/len(item_means) if item_means else float("nan")
            rate = valid_count / len(samples)
            log = EvalLog(status="success", eval=EvalSpec(
                created=datetime.now(UTC).isoformat(), eval_id=str(uuid.uuid4()), run_id=str(uuid.uuid4()),
                task="DTBench", task_registry_name="bench/DTBench", task_version=1,
                task_args={} if allow_cot else {"allow_cot": False}, model="archive/" + subject,
                dataset=EvalDataset(name="DTBench-capabilities-published-archive", samples=len(epochs), sample_ids=sorted(epochs)),
                config=EvalConfig(epochs=max_epoch, epochs_reducer=["bench/valid_response_mean"]),
                scorers=[EvalScorer(name="bench/dtbench_score", metrics=[EvalMetricDefinition(name="bench/valid_accuracy"), EvalMetricDefinition(name="bench/valid_response_rate")])],
                metadata={"imported_archive": True, "source_subject": subject,
                          "source_commit": data["source_commit"], "archive_sha256": archive_hash,
                          "no_original_provider_logs": True, "unequal_trial_counts": len(set(epochs.values())) > 1}),
                samples=samples,
                results=EvalResults(total_samples=len(samples), completed_samples=len(samples),
                    scores=[EvalScore(name="dtbench_score", scorer="bench/dtbench_score", params={},
                        metrics={"valid_accuracy": EvalMetric(name="bench/valid_accuracy", value=accuracy),
                                 "valid_response_rate": EvalMetric(name="bench/valid_response_rate", value=rate)})]))
            filename = hashlib.sha256(subject.encode()).hexdigest()[:16] + ".eval"
            write_eval_log(log, output / filename)
            manifest["logs"].append({"file": filename, "subject": subject, "items": len(epochs),
                "attempts": len(samples), "valid_attempts": sum(math.isfinite(s.scores["dtbench_score"].value) for s in samples),
                "valid_accuracy": accuracy if item_means else None,
                "max_attempts_per_item": max_epoch, "prompt_incomplete": partial})
            print(f"Wrote {subject}: {len(epochs)} items / {len(samples)} attempts", flush=True)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--include-interventions", action="store_true")
    args = parser.parse_args()
    convert(args.archive, args.output, args.include_interventions)
