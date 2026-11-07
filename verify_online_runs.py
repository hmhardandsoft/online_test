"""Run all six synthetic process-level combinations plus uncued mode, no hardware."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from uuid import uuid4
import numpy as np
import csv
from online_core import (Config, METHODS, INTENSITY, LABEL, STAGE, SEGMENT, TRAIN_VALID, VALID,
                         TRAIN, ONLINE, WARMUP, metadata_columns)
from online_experiment import write_json


def main():
    root = Path(__file__).resolve().parent
    output = root/"online_results"/"software_checks"/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid4().hex[:6])
    output.mkdir(parents=True)
    cfg = Config(training_blocks=1, train_cue_seconds=(1., 1.), skip_seconds=.2,
                 min_samples=5, online_seconds=2., online_cue_seconds=(.35, .55))
    config_path = output/"smoke_config.json"
    write_json(config_path, cfg.to_dict())
    reports = []
    cases = [(m, method, "cued") for m in (2, 4) for method in METHODS]
    cases.append((4, "Aug_FB_eTRCA", "uncued"))
    for m, method, mode in cases:
        destination = output/f"M{m}_{method}_{mode}"
        command = [sys.executable, str(root/"online_experiment.py"), "--config", str(config_path),
                   "--simulate", "--headless", "--task-m", str(m), "--method", method,
                   "--online-mode", mode, "--output", str(destination)]
        run = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=90)
        (output/f"{destination.name}.log").write_text(run.stdout+run.stderr, encoding="utf-8")
        if run.returncode:
            raise RuntimeError(f"{destination.name} failed: {run.stdout}\n{run.stderr}")
        summary = json.loads((destination/"summary.json").read_text(encoding="utf-8"))
        training = json.loads((destination/"training.json").read_text(encoding="utf-8"))
        predictions = json.loads((destination/"predictions.json").read_text(encoding="utf-8"))
        events = json.loads((destination/"events.json").read_text(encoding="utf-8"))
        assert summary["completed"] and summary["model_frozen"] and not summary["error"]
        assert summary["prediction_count"] > 0
        assert 2 <= summary["duration_seconds"] < 2.2
        assert np.all(np.asarray(training["counts"]) >= cfg.min_samples)
        assert (destination/"frozen_model.joblib").is_file()
        stages = list(dict.fromkeys(event["stage"] for event in events))
        assert stages == ["training", "draining", "fitting", "catchup", "waiting_test", "test_warmup", "online"], stages
        assert summary["test_ready_time"] < summary["test_space_accepted_at"] < summary["test_stimulus_start"] < summary["online_start"]
        assert summary["test_warmup_actual_seconds"] >= cfg.test_warmup_seconds
        assert summary["test_start_input"] == "automatic_synthetic_test"
        published = [p for p in predictions if p["publication_time"] is not None]
        assert all(summary["online_start"] <= p["publication_time"] <= summary["online_end"] for p in published)
        assert {p["edge"] for p in published} == set(range(1, m+1))
        with np.load(destination/"recording.npz") as recording:
            meta, eeg, states = recording["metadata"], recording["eeg"], recording["states"]
            assert len(eeg) == len(meta) == len(states)
            assert list(recording["metadata_columns"]) == metadata_columns(m)
            black = np.isin(meta[:, STAGE], (0, 2, 3, 4, 6, 7))
            assert np.all(meta[black, INTENSITY:INTENSITY+m] == 0)
            assert np.all(meta[black, INTENSITY+m] == 0)
            assert np.all(meta[black, INTENSITY+m+1] == -1)
            for edge, label, segment, edge_sample, start, stop in training["windows"]:
                epoch_meta = meta[start:stop]
                assert stop-start == 200 and edge_sample-start == 100
                assert np.all(epoch_meta[:, LABEL] == label+1)
                assert np.all(epoch_meta[:, SEGMENT] == segment)
                assert np.all(epoch_meta[:, STAGE] == TRAIN)
                assert np.all(epoch_meta[:, TRAIN_VALID] == 1)
                assert np.all(epoch_meta[:, VALID] == 1)
            for prediction in published:
                a, b = prediction["start"], prediction["stop"]
                window_meta = meta[a:b]
                assert np.all(np.isin(window_meta[:, STAGE], (WARMUP, ONLINE)))
                assert np.all(window_meta[:, INTENSITY+m] == 2)
        with (destination/"frames.csv").open(encoding="utf-8-sig", newline="") as handle:
            frames = list(csv.DictReader(handle))
        assert [int(f["frame"]) for f in frames] == list(range(len(frames)))
        for stimulus_id in (1, 2):
            stimulus_frames = [f for f in frames if int(f["stimulus_id"]) == stimulus_id]
            assert [int(f["stimulus_frame"]) for f in stimulus_frames] == list(range(len(stimulus_frames)))
            assert json.loads(stimulus_frames[0]["intensity"]) == [float(2*i < m) for i in range(m)]
        warmup_event = next(e for e in events if e['stage'] == 'test_warmup')
        first_online_event = next(e for e in events if e['stage'] == 'online')
        assert warmup_event['cue'] == first_online_event['cue']
        assert warmup_event['segment'] == first_online_event['segment']
        cues = [e["cue"] for e in events if e["stage"] == "online"]
        if mode == "cued":
            assert len(cues) >= 3 and all(a != b for a, b in zip(cues, cues[1:]))
            assert 0 <= summary["time_accuracy"] <= 1
        else:
            assert cues == [0] and summary["time_accuracy"] is None and summary["correct_seconds"] is None
        reports.append(dict(case=destination.name, passed=True, duration=summary["duration_seconds"],
                            predictions=summary["prediction_count"], counts=training["counts"],
                            fit_seconds=training["fit_seconds"]))
        print(f"PASS {destination.name}: {summary['prediction_count']} predictions", flush=True)
    write_json(output/"verification.json", reports)
    print(f"All {len(reports)} process-level cases passed: {output}")


if __name__ == "__main__":
    main()
