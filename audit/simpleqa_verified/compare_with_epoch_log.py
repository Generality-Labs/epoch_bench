"""No model calls. Builds the dataset of bench/simpleqa_verified_anti_abstention and checks it against one of Epoch's public logs of that task:
same ids, same user message and same target for every question, same grader template, same solver.
python audit/simpleqa_verified/compare_with_epoch_log.py PATH/ViC6pxQZUVS3E9M9iEubF7.eval"""
import sys

from inspect_ai.log import read_eval_log

from bench.task.simpleqa_verified.grader_templates import SIMPLEQA_VERIFIED_GRADER_TEMPLATE
from bench.task.simpleqa_verified.simpleqa import anti_abstention_dataset

log = read_eval_log(sys.argv[1]); theirs = {s.id: s for s in log.samples}; mine = {i + 1: s for i, s in enumerate(anti_abstention_dataset())}
print("task in log:", log.eval.task, "| questions: mine", len(mine), "theirs", len(theirs), "| same ids:", set(mine) == set(theirs))
print("same user message:", sum(mine[i].input == theirs[i].messages[0].content for i in theirs), "of", len(theirs))
print("same target:", sum(str(mine[i].target) == str(theirs[i].target) for i in theirs), "of", len(theirs))
print("grader template identical:", SIMPLEQA_VERIFIED_GRADER_TEMPLATE == log.eval.scorers[0].options["grader_template"])
print("their solver:", [s.solver for s in log.plan.steps], "| their grader:", log.eval.scorers[0].options["grader_model"]["model"], log.eval.scorers[0].options["grader_model"]["config"], "| their model config:", log.eval.model_generate_config.model_dump(exclude_none=True))
