"""Run the frozen native Heart solver to a real terminal or a predeclared capture budget."""
import argparse
import json
import math
from pathlib import Path
import time

from m29_native_first_step import PrefixComplete, Recorder, append, json_array, run

MAX_EVALUATIONS = 60
SOFT_SECONDS = 690.


def terminal_fields(solved):
    if solved is None:
        return None
    keys = ('status','function_evaluations','restart_count','restart_function_evaluations',
            'normalized_equation_residual','physical_relative_residual','raw_equation_residual',
            'multiplier_mu','multipliers','quadric_coefficients','equation_profile','event_hash','trace_hash')
    def numeric(v):
        if isinstance(v,list):
            return [numeric(i) for i in v]
        return None if isinstance(v,float) and not math.isfinite(v) else v
    fields = json.loads(json.dumps({k:solved[k] for k in keys},default=json_array))
    return {k:numeric(v) for k,v in fields.items()}


class Trajectory(Recorder):
    def __init__(self, *args):
        super().__init__(*args)
        self.started = time.monotonic()
        self.indices = []

    def may_evaluate(self):
        return self.calls < MAX_EVALUATIONS and time.monotonic()-self.started < SOFT_SECONDS

    def observe(self, event):
        if self.fatal:
            raise RuntimeError('M30_CAPTURE_INTEGRITY_FAILED')
        append(self.events, event)
        trial = (event['type']=='evaluation' and event['outcome']=='accepted' and event['iteration']>=1 and
                 event['reason'] in ('residual_decreased','topology_transition_relinearized'))
        if trial:
            self.indices.append(event['evaluation_index'])
            if self.accepted is None:
                self.accepted = event['evaluation_index']
        if self.may_evaluate():
            return
        # A final accepted evaluation is allowed to commit and report convergence.
        # An exhausted native attempt can also complete normally at this boundary.
        if event['type']=='attempt_terminal' or event['type']=='evaluation' and event['outcome']=='accepted':
            return
        self.status = 'BUDGET_EXHAUSTED' if self.calls >= MAX_EVALUATIONS else 'TIME_BUDGET_EXHAUSTED'
        raise PrefixComplete

    def receipt_fields(self, solved):
        return dict(schema='w2-m30-trajectory-v1', status=self.status or 'NATIVE_TERMINAL',
                    maximum_capture_evaluations=MAX_EVALUATIONS, soft_seconds=SOFT_SECONDS,
                    elapsed_capture_seconds=time.monotonic()-self.started,
                    accepted_evaluation_indices=self.indices,
                    native_terminal_summary=terminal_fields(solved))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-request',type=Path,required=True)
    p.add_argument('--module-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(run(args.prepared_request,args.module_manifest,args.output,recorder_type=Trajectory),indent=2))
