"""Durable full-data Stage1 -> Stage2 run; fail fast and preserve checkpoints."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/official_budget_v1'
RUN.mkdir(parents=True, exist_ok=True)


def status(**fields):
    fields.update(pid=os.getpid(), updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    tmp = RUN / 'status.tmp'
    tmp.write_text(json.dumps(fields, indent=2) + '\n')
    tmp.replace(RUN / 'status.json')


def main():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    for stage, epochs, effective, candidates in [(1, 2500, 80, [80, 40, 20, 10, 5, 1]),
                                                   (2, 1000, 30, [30, 15, 10, 5, 1])]:
        folder = RUN / ('stage%d' % stage)
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / 'last.pt'
        selected = folder / 'physical_batch.json'
        if selected.exists():
            previous = json.loads(selected.read_text())['physical_batch']
            candidates = [n for n in candidates if n <= previous]
        for physical in candidates:
            cmd = [sys.executable, '-u', str(ROOT / 'engineering/train.py'),
                   '--stage', str(stage), '--epochs', str(epochs), '--batch-size', str(effective),
                   '--physical-batch', str(physical), '--workers', '4', '--log-every', '10',
                   '--run-dir', str(folder)]
            if checkpoint.exists():
                cmd += ['--resume', str(checkpoint)]
            elif stage == 2:
                cmd += ['--pretrained', str(RUN / 'stage1/last.pt')]
            print('START', ' '.join(cmd), flush=True)
            selected.write_text(json.dumps({'physical_batch': physical, 'effective_batch': effective})+'\n')
            with (RUN / ('stage%d.log' % stage)).open('a') as log:
                child = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
                status(state='training', stage=stage, child_pid=child.pid, epochs=epochs,
                       physical_batch=physical, effective_batch=effective, command=cmd)
                result = child.wait()
            if result == 86 and physical > 1:
                print('CUDA OOM: reduce physical batch only; retain epochs/effective batch.', flush=True)
                continue
            if result:
                status(state='failed', stage=stage, exit_code=result, log=str(log.name))
                return result
            break
        print('FINISHED stage', stage, flush=True)
    status(state='training_complete', note='Generation quality and paper metrics still need assessment.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
