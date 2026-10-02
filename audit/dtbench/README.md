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

The source parity and mock execution tests also pass against Hawk's pinned
METR Inspect revision `2203ee7a8f06d08eac37bcc1a5050cc2fc38b415`
(`0.3.264.dev13`), as well as PyPI Inspect `0.3.266`. Cached completion
postprocessing supports both field and property versions of Inspect's output API.

Invalid responses are unscored (NaN) for capability accuracy, matching the
source's exclusion. The `valid_response_rate` metric reports their frequency.
Repeated trials are averaged over valid responses within each item, then items
are weighted equally. Items with no valid responses are absent from that
accuracy mean. The original generation script includes trivia. The paper's
capability analysis excludes trivia and restricts comparisons to question IDs
with a valid response from every model in the chosen cohort. Imported per-subject
scores retain the full generation population and impose neither filter.

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

Converted archives have unequal recorded trial counts. When streaming them or
an item slice, use `read_eval_log_samples(path, all_samples_required=False)`.
The conversion's `success` status means all selected archive records were
converted; it does not promise a rectangular item-by-epoch matrix. The default
strict iterator otherwise asks for unrecorded epochs and raises `IndexError`.
Read the actual stored trials and keep the original IDs and trial ordinals.

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

## Exact paper aggregation check

`reproduce_paper.py` reads the pinned source's literal ordinary-model cohort and
the converted logs, without loading source pickles. It excludes the 32 capability
trivia items, intersects valid item support across all 59 models, and computes
each model's equal-item mean. Against the pinned published
`analysis_csvs/cap_versus_att.csv`, all 59 scores agree within `3.4e-16`.
The released cohort has 370 shared valid items from 375 non-trivia capability
items. These are the source CSV's scores, not current CRI leaderboard scores.

```sh
python audit/dtbench/reproduce_paper.py /path/to/converted-logs \
  /path/to/pinned-source /path/to/paper-reproduction.json
```

## Validation results (2026-10-02)

Every model was run on all 407 capability items with three trials per item
(1,221 responses each). The table uses the source's equal-item mean over valid
answers, and raw current CRI accuracy rather than its chance-adjusted index.

| Model | Inspect accuracy | Published raw accuracy | Difference | Invalid responses |
| --- | ---: | ---: | ---: | ---: |
| GPT-4o mini | 55.6102% | 54.40% | +1.2102 pp | 0 |
| GPT-4.1 nano | 52.5578% | 52.53% | +0.0278 pp | 149 |
| GPT-4.1 mini | 67.5676% | 68.80% | -1.2324 pp | 4 |

All differences are within combined item-sampling uncertainty. This supports
consistency with the current leaderboard; it does not establish identical
provider settings or populations without the current per-item traces. Nano
has three items with no valid answer and therefore an accuracy denominator of
404. One failed Mini provider request was retried individually; its final log
records that repair and has no request errors or truncations. Conservative
usage-based validation cost was $2.6643, without cache discounts.

The strict parser's invalid-answer exclusions are preserved. A separate Nano
sensitivity check, removing Markdown asterisks and accepting case-insensitive
unambiguous answer markers, recovers 143 of the 149 invalid responses. Its
407-item accuracy is 52.2113%, a -0.3465 percentage-point change; this is not
used as the native score. Exact duplicate items 79.17/79.18 are also preserved
for source fidelity, rather than silently deduplicated.

Deterministic evidence is separate from these live comparisons: all 188,820
converted archived responses agree with their original recorded grades, and
all 59 ordinary-model paper scores reproduce to a maximum absolute error of
3.33e-16 on the original 370 shared valid items.
