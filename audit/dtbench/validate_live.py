"""Run inexpensive leaderboard checks; load the existing audit key without logging it."""
import asyncio
import argparse
import json
import os
from pathlib import Path

from dotenv import dotenv_values
from inspect_ai import eval_async
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import solver

from bench.task.dtbench import dtbench

ROOT = None
BUDGET = 6.0
MODELS = ['openai/gpt-4o-mini', 'openai/gpt-4.1-nano', 'openai/gpt-4.1-mini']
RATES = {'openai/gpt-4o-mini': (0.15, 0.6), 'openai/gpt-4.1-nano': (0.1, 0.4), 'openai/gpt-4.1-mini': (0.4, 1.6)}
SPENT = 0.0
RESERVED = 0.0


@solver
def budgeted_generate(model_id, max_tokens):
    async def solve(state, generate):
        global SPENT, RESERVED
        input_rate, output_rate = RATES[model_id]
        # Conservative input bound; reserve in-flight maximum output before calls.
        reserve = (len(str(state.messages).encode()) * input_rate + max_tokens * output_rate) / 1e6
        if SPENT + RESERVED + reserve > BUDGET:
            raise RuntimeError('Shared validation allowance exhausted')
        RESERVED += reserve
        try:
            state = await generate(state)
            usage = state.output.usage
            if usage is None:
                raise RuntimeError('Provider did not report usage; stop rather than lose budget accounting')
            SPENT += (usage.input_tokens * input_rate + usage.output_tokens * output_rate) / 1e6
            (ROOT / 'budget.json').write_text(json.dumps({'spent_upper_bound_usd': SPENT, 'reserved_usd': RESERVED - reserve}))
            return state
        finally:
            RESERVED -= reserve
    return solve


async def run(key_file):
    if key_file:
        os.environ['OPENROUTER_API_KEY'] = dotenv_values(key_file)['OPENROUTER_API_KEY']
    if not os.environ.get('OPENROUTER_API_KEY'):
        raise RuntimeError('Set OPENROUTER_API_KEY or supply --key-file')
    # Direct calls use the operator's audit key, including on hosts with Hawk hooks.
    os.environ['INSPECT_ACTION_RUNNER_REFRESH_URL'] = ''
    ROOT.mkdir(parents=True, exist_ok=True)
    for model_id in MODELS:
        print('Starting', model_id, flush=True)
        config = GenerateConfig(max_tokens=16384 if '4o' in model_id else 32768,
                                max_connections=24, max_retries=3, timeout=180)
        task = dtbench(epochs=3)
        task.solver = budgeted_generate(model_id, config.max_tokens)
        logs = await eval_async(task, model=get_model('openrouter/' + model_id, config=config),
                                log_dir=str(ROOT / 'logs'),
                                max_samples=24, fail_on_error=0.02)
        log = logs[0]
        summary = {'model': model_id, 'status': log.status, 'log': log.location,
                   'metrics': {k: v.value for k, v in log.results.scores[0].metrics.items()} if log.results and log.results.scores else {},
                   'stats': log.stats.model_dump(mode='json') if log.stats else {},
                   'error': log.error.message if log.error else None}
        (ROOT / (model_id.split('/')[-1] + '.json')).write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        if log.status != 'success':
            raise RuntimeError('Parity run did not complete: ' + model_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--budget', type=float, default=6)
    args = parser.parse_args()
    ROOT = args.output_dir
    BUDGET = args.budget
    asyncio.run(run(args.key_file))
