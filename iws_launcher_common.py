"""Shared launcher for the three separate I/W/S four-round sets."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LADDER = 'data/checkpoints/pickplace/ladder/ladder.json'
BASE = 'data/checkpoints/pickplace/ladder/bc_rnn_PickPlaceCan_p063.pth'

# Displayed success rates are Michelle's verified rates, NOT local results.
# Cross-check: all ten checkpoints were locally evaluated for 50 rollouts each.
# Some individual estimates differed; the overall increasing competence trend
# was comparable, so Michelle's original values are the source of truth here.
SCHEDULES = {
    'I': [('p003', .08), ('p007', .18), ('p036', .32), ('p043', .40)],
    'W': [('p043', .40), ('p036', .32), ('p007', .18), ('p003', .08)],
    'S': [('p036', .32), ('p039', .36), ('p043', .40), ('p039', .36)],
}

def launch(condition):
    p = argparse.ArgumentParser(description=f'Launch {condition} four-round policy study')
    p.add_argument('--participant', required=True)
    args = p.parse_args()
    manifest_path = ROOT / LADDER
    if not manifest_path.is_file():
        p.error(f'Missing {manifest_path}')
    rungs = json.loads(manifest_path.read_text(encoding='utf-8'))['rungs']
    for ckpt, rate in SCHEDULES[condition]:
        if not any(r['name'] == ckpt and abs(r['verified_success'] - rate) < 1e-8
                   for r in rungs):
            p.error(f'Checkpoint / verified rate mismatch: {ckpt} {rate}')
    rates = [str(rate) for _, rate in SCHEDULES[condition]]
    print(f'{condition}: ' + ' -> '.join(f'{float(r):.0%}' for r in rates), flush=True)
    cmd = [sys.executable, 'scripts/run_session.py',
           '--config', 'configs/study_pickplace.yaml',
           '--mode', 'dagger', '--algo', 'bc_rnn',
           '--checkpoint', BASE, '--condition', 'yoked',
           '--ladder', LADDER, '--yoked-schedule', *rates,
           '--rounds', '4', '--rollouts-per-round', '1',
           '--participant', args.participant, '--no-survey']
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))
