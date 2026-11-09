"""Process-level synthetic checks for one calibration and three independent tests."""
import csv
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from uuid import uuid4

import joblib
import numpy as np

from online_core import (Config, METHODS, INTENSITY, LABEL, STAGE, SEGMENT,
                         TRAIN_VALID, VALID, TRAIN, metadata_columns, time_accuracy)
from online_experiment import write_json


def check_case(root, output, m, mode, stop_mode):
    output.mkdir(parents=True)
    cfg = Config(task_m=m, training_blocks=1, max_extra_blocks=0,
                 train_cue_seconds=(1., 1.), skip_seconds=.2, min_samples=5,
                 online_mode=mode, online_stop_mode=stop_mode,
                 online_seconds=1.2, online_cue_count=3,
                 online_cue_seconds=(.3, .4), test_warmup_seconds=.2,
                 refresh_rate=600, timing_diagnostics=True)
    config_path = output/'config.json'
    destination = output/'run'
    write_json(config_path, cfg.to_dict())
    run = subprocess.run([sys.executable, str(root/'online_experiment.py'),
                          '--config', str(config_path), '--simulate', '--headless',
                          '--output', str(destination)], cwd=root,
                         capture_output=True, text=True, timeout=180)
    (output/'run.log').write_text(run.stdout+run.stderr, encoding='utf-8')
    if run.returncode:
        raise RuntimeError(run.stdout+'\n'+run.stderr)
    summary = json.loads((destination/'summary.json').read_text(encoding='utf-8'))
    training = json.loads((destination/'training.json').read_text(encoding='utf-8'))
    predictions = json.loads((destination/'predictions.json').read_text(encoding='utf-8'))
    events = json.loads((destination/'events.json').read_text(encoding='utf-8'))
    with (destination/'frames.csv').open(encoding='utf-8-sig', newline='') as f:
        frames = list(csv.DictReader(f))
    with np.load(destination/'recording.npz') as recording:
        meta = recording['metadata']
        assert list(recording['metadata_columns']) == metadata_columns(m)
        assert len(meta) == len(recording['eeg']) == len(recording['states'])
    frozen = joblib.load(destination/'frozen_model.joblib')
    assert set(frozen['models']) == set(METHODS)
    assert summary['completed'] and summary['model_frozen'] and not summary['error']
    assert len(summary['tests']) == 3 and set(summary['test_method_order']) == set(METHODS)
    assert len({t['cue_seed'] for t in summary['tests']}) == 3
    assert np.all(np.asarray(training['counts']) >= cfg.min_samples)
    assert set(training['methods']) == set(METHODS)
    assert len(frames) == json.loads((destination/'timing_diagnostics_summary.json').read_text())['frame_rows']
    assert [int(f['frame']) for f in frames] == list(range(len(frames)))
    for edge, label, segment, edge_sample, start, stop in training['windows']:
        rows = meta[start:stop]
        assert stop-start == 200 and edge_sample-start == 100
        assert np.all(rows[:, LABEL] == label+1)
        assert np.all(rows[:, SEGMENT] == segment)
        assert np.all(rows[:, STAGE] == TRAIN)
        assert np.all(rows[:, TRAIN_VALID] == 1)
        assert np.all(rows[:, VALID] == 1)
    for test in summary['tests']:
        i = test['test_id']
        assert test['completed']
        assert test['stop_reason'] == ('time_limit' if stop_mode == 'time' else 'completed_cue_count')
        assert test['ready_time'] < test['space_accepted_at'] < test['stimulus_start'] < test['online_start'] < test['online_end']
        assert test['warmup_actual_seconds'] >= cfg.test_warmup_seconds
        assert test['method'] == summary['test_method_order'][i-1]
        test_frames = [f for f in frames if f['test_id'] == str(i)]
        stimulus_ids = {int(f['stimulus_id']) for f in test_frames}
        assert len(stimulus_ids) == 1 and min(stimulus_ids) > 0
        assert [int(f['stimulus_frame']) for f in test_frames] == list(range(len(test_frames)))
        assert all(f['method'] == test['method'] for f in test_frames)
        cues = [(e['time'], e['cue']) for e in events if e['test_id'] == i and e['stage'] == 'online']
        published = [(p['publication_time'], p['prediction']) for p in predictions
                     if p['test_id'] == i and p['publication_time'] is not None]
        assert published and len(published) == test['prediction_count']
        assert all(test['online_start'] <= t <= test['online_end'] for t, _ in published)
        assert all(p['method'] == test['method'] for p in predictions if p['test_id'] == i)
        assert all(np.all(meta[p['start']:p['stop'], INTENSITY+m+2] == i)
                   for p in predictions if p['test_id'] == i and p['publication_time'] is not None)
        result = time_accuracy(cues, published, test['online_start'], test['online_end'], mode == 'cued')
        assert result == {key: test[key] for key in result}
        if mode == 'cued':
            labels = [label for _, label in cues]
            assert all(a != b for a, b in zip(labels, labels[1:]))
            assert 0 <= test['time_accuracy'] <= 1
            if stop_mode == 'segments':
                assert test['cue_count_completed'] == test['cue_count_presented'] == cfg.online_cue_count
            else:
                assert cfg.online_seconds <= test['duration_seconds'] < cfg.online_seconds+.1
        else:
            assert test['time_accuracy'] is None and test['correct_seconds'] is None
    black = np.isin(meta[:, STAGE], (0, 2, 3, 4, 6, 7))
    assert np.all(meta[black, INTENSITY:INTENSITY+m] == 0)
    assert np.all(meta[black, INTENSITY+m] == 0)
    print(f"PASS M{m} {mode} {stop_mode}: {summary['prediction_count']} predictions", flush=True)
    return dict(m=m, mode=mode, stop_mode=stop_mode, passed=True,
                method_order=summary['test_method_order'])


def main():
    root = Path(__file__).resolve().parent
    output = root/'online_results'/'software_checks'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
    output.mkdir(parents=True)
    cases = [(2, 'cued', 'time'), (4, 'cued', 'time'),
             (2, 'cued', 'segments'), (4, 'cued', 'segments'),
             (4, 'uncued', 'time')]
    reports = [check_case(root, output/f'case_{i}', *case)
               for i, case in enumerate(cases, 1)]
    write_json(output/'verification.json', reports)
    print(f"All {len(reports)} process-level cases passed: {output}")


if __name__ == '__main__':
    main()
