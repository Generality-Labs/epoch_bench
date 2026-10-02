# DTBench Inspect port and archive import

`bench/DTBench` loads the 407 double-checked capability questions from
`casparoe/newcomblike_questions_dataset` at
`284fda2da5ee5059b9b7b59ef419553e1bd5a067`. The 130 attitude questions remain
in the bundled source data for reference; they do not have objective accuracy
targets and are outside this task's population.

The pinned source ZIP and its SHA256 are bundled. `build_dataset.py` flattens
the nested setup prefixes and inherited tags exactly as the source does.
The published `grading.py` is vendored with only its util import changed to a
relative import. Its strict, case-sensitive `FINAL ANSWER: ` parser is retained.
The original multiple-choice instruction is used verbatim. Inspect's generic
multiple-choice solver is not used. The shuffle is reproducible by seed and
question ID, with the resulting option order and target stored in each sample.
Source no-CoT output postprocessing is retained and its raw completion saved.
The unchanged source Question, Subject and grading modules are also retained in
`reference_source/` as the independent test oracle, under the bundled MIT license.
The tests need no separate source checkout or provider access.

Invalid responses are unscored (NaN) for capability accuracy, matching the
source's exclusion. The `valid_response_rate` metric reports their frequency.
Repeated trials are averaged over valid responses within each item, then items
are weighted equally. Items with no valid responses are absent from that
accuracy mean. The source paper also sometimes restricts comparisons to the
question IDs shared by the chosen model cohort; the imported per-subject score
does not impose a new cohort intersection.

## Published results

`import_results.py` reads the original password-protected result archive. It
imports all matching capability records for ordinary and no-CoT subjects by
default, retaining invalid responses and every repeat. Background/persona and
best-of-N variants can be included with `--include-interventions`. Outdated
questions and out-of-population records are counted in the manifest. The original
recorded grades are preserved, and any disagreement with the port is reported.

Each imported `.eval` contains the exact archived completion and shuffled
choices, a target derived from the archived correct-answer text, original
subject and timestamp, ZIP filename and hash, and a unique `(qid, epoch)` pair.
The epoch is an assigned archival trial ordinal, not an original run label.
The Inspect model name uses an `archive/` namespace so retired source names
remain replayable without pretending to be active provider routes. The original
subject name is preserved separately; re-scoring uses an explicit model override.
The question prompt is reconstructed from the source, explicitly marked as
such. Original provider request/response events, token counts, finish reasons,
sampling settings and intermediate best-of-N calls were not saved and are not
invented. Imported logs are provenance-marked conversions, not new model runs.
Intervention prompts can be incomplete because many background texts were not
published. Those logs should not support claims about the full elicitation.
The archive contains the original study's older models; it does not establish
the latest CRI leaderboard's per-item outcomes.

```sh
uv run --with json5 python audit/dtbench/build_dataset.py
inspect eval bench/DTBench --model mockllm/model --limit 2
python audit/dtbench/import_results.py /path/to/results_db_new.zip /path/to/logs
```

The task can be passed directly to `inspect_audit/audit`, with the converted
logs as its `logs` argument. Auditors can use the actual grader and check its
concordance with each recorded grade. The task, importer and audit all use the
original stable `qid` for joining; positional dataset indexes are never used.

## Live leaderboard check

`validate_live.py` runs GPT-4o mini, GPT-4.1 nano and GPT-4.1 mini over all
407 capability questions with three trials per item, default provider sampling
and the models' full output limits. It uses the existing operator key and reserves
maximum in-flight token costs against a shared validation allowance. Usage is
recorded conservatively without cache discounts. The allowance can stop a run;
incomplete runs must not be treated as leaderboard reproduction.

```sh
python audit/dtbench/validate_live.py --output-dir /path/to/validation \
  --key-file /path/to/audit.env --budget 6
```

Compare raw accuracy with `dtbench_accuracy` in the published CRI CSV.
The CSV's `dtbench` column is chance-adjusted: `(accuracy - 0.4) / 0.6 * 100`.
Its `dtbench_ci` confidence half-width must also be multiplied by `0.6 / 100`
to obtain raw-accuracy uncertainty. Uncertainty over items differs from repeat
sampling noise on a fixed dataset. The current leaderboard does not publish its
item IDs or provider transcripts, so population/protocol equivalence cannot be
inferred from aggregate agreement alone. The original archive provides the
separate deterministic grader and aggregation reproduction check.
